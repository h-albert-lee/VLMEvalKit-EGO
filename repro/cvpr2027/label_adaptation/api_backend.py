"""Offline API request encoding and receipt classification; no network transport."""
import base64
from decimal import Decimal, InvalidOperation
import json
from .protocol import canonical, checked_image, require, seal, sha

PNG_MAGIC = b'\x89PNG\r\n\x1a\n'

def make_body(root, record):
    """Verify frozen bytes before dispatch and preserve baseline serialization."""
    require(record.get('backend') == 'openrouter', 'Wrong backend')
    require(seal(record)['request_key'] == record['request_key'], 'Request changed')
    require(record['parameters'].get('model') == record['model'], 'Model parameter mismatch')
    require(record['parameters'].get('stream') is False, 'Only non-streaming requests supported')
    messages = []
    for message in record['messages']:
        require(message['role'] in ('user', 'assistant'), 'Unexpected message role')
        content = []
        for part in message['content']:
            if part['type'] == 'text':
                require(isinstance(part['text'], str), 'Invalid text')
                content.append({'type': 'text', 'text': part['text']})
            else:
                require(part['type'] == 'image', 'Unexpected content type')
                _, data = checked_image(root, part)
                require(data.startswith(PNG_MAGIC), 'Expected frozen PNG')
                require(part.get('mime', 'image/png') == 'image/png', 'Unexpected image MIME')
                require(part.get('detail', 'high') == 'high', 'Unexpected image detail')
                content.append({'type': 'image_url', 'image_url': {
                    'url': 'data:image/png;base64,' + base64.b64encode(data).decode(),
                    'detail': 'high'}})
        if message['role'] == 'assistant':
            require(len(content) == 1 and content[0]['type'] == 'text', 'Invalid demonstration answer')
            content = content[0]['text']
        messages.append({'role': message['role'], 'content': content})
    return canonical({**record['parameters'], 'messages': messages})

def classify_receipt(receipt, record):
    """Pure local parsing. Invalid JSON and failed outputs remain explicit."""
    result = {'raw_text': '', 'raw_response': None, 'usage': {}, 'cost_usd': None,
              'billing_uncertain': True, 'stop_required': True, 'error': None,
              'finish_reason': None, 'response_model': None, 'response_provider': None,
              'request_id': None, 'http_status': receipt.get('http_status'),
              'truncated': False, 'operational_issue': None,
              'body_sha256': receipt['body_sha256'], 'attempt_id': receipt['attempt_id'],
              'request_key': record['request_key']}
    if receipt.get('transport_error'):
        result['error'] = 'transport_' + receipt['transport_error']['type']
        result['error_detail'] = receipt['transport_error'].get('detail', '')
        return result
    raw_bytes = base64.b64decode(receipt['raw_body_base64'], validate=True)
    require(sha(raw_bytes) == receipt['raw_body_sha256'], 'Transport receipt body changed')
    text = raw_bytes.decode('utf-8', errors='replace')
    if receipt['http_status'] != 200:
        result['error'] = 'http_' + str(receipt['http_status'])
        result['raw_error'] = text
        explicit_free_moderation = (receipt['http_status'] == 403 and
            'no credits were charged' in text.lower() and 'your input was flagged' in text.lower())
        if explicit_free_moderation:
            result.update(cost_usd=0, billing_uncertain=False, stop_required=False)
        return result
    try:
        raw = json.loads(text)
    except (ValueError, TypeError):
        result['error'] = 'invalid_response_json'
        return result
    if not isinstance(raw, dict):
        result['error'] = 'invalid_response_shape'
        return result
    result['raw_response'] = raw
    usage = raw.get('usage')
    if isinstance(usage, dict):
        result['usage'] = usage
        cost = usage.get('cost')
        try:
            require(not isinstance(cost, bool), 'Boolean cost')
            amount = Decimal(str(cost))
            require(amount.is_finite() and amount >= 0, 'Invalid cost')
            result.update(cost_usd=str(amount), billing_uncertain=False)
        except (InvalidOperation, TypeError, ValueError):
            pass
    result.update(request_id=raw.get('id'), response_model=raw.get('model'),
                  response_provider=raw.get('provider'))
    choices = raw.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        result['error'] = 'invalid_response_choices'
        return result
    choice = choices[0]
    result['finish_reason'] = choice.get('finish_reason')
    result['truncated'] = choice.get('finish_reason') == 'length'
    if result['truncated']:
        result['operational_issue'] = 'truncated_response'
    message = choice.get('message')
    if not isinstance(message, dict):
        result['error'] = 'invalid_response_message'
        return result
    content = message.get('content')
    if isinstance(content, list):
        if not all(isinstance(p, dict) for p in content):
            result['error'] = 'invalid_content_parts'
            return result
        texts = [p.get('text') for p in content if p.get('type') == 'text']
        if not all(isinstance(t, str) for t in texts):
            result['error'] = 'invalid_text_part'
            return result
        content = ''.join(texts)
    if content is None:
        content = ''
    if not isinstance(content, str):
        result['error'] = 'invalid_response_content'
        return result
    result['raw_text'] = content
    if raw.get('model') not in (record['model'], record['revision']) or raw.get('provider') != record['provider']:
        result['error'] = 'provider_or_model_mismatch'
    elif raw.get('error'):
        result['error'] = 'api_response_error'
    elif message.get('refusal'):
        result['error'] = 'model_refusal'
    elif not content.strip():
        result['error'] = 'empty_response'
    elif choice.get('finish_reason') not in ('stop', 'length'):
        result['error'] = 'unexpected_finish_reason'
    result['stop_required'] = bool(result['error'] or result['billing_uncertain'] or result['truncated'])
    return result
