"""Pinned Qwen2.5-VL-72B encoding core for answer-only B0/B1/B2 adaptation.

Imports no GPU package until construction. Does not launch jobs, alter installed
packages, start servers, read credentials or choose examples. The caller owns
the frozen messages, output ledger, checkpointing and per-suite HF cache.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

MODEL_ID = 'Qwen/Qwen2.5-VL-72B-Instruct'
MODEL_KEY = 'Qwen--Qwen2.5-VL-72B-Instruct'
REVISION = '89c86200743eec961a297729e7990e8f2ddbc4c5'
BASELINE_PARAMETERS = {'max_new_tokens': 256, 'do_sample': False, 'dtype': 'bfloat16', 'seed': 0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def validate_parameters(parameters):
    require(isinstance(parameters, dict), 'Frozen parameters must be an object')
    for name, expected in BASELINE_PARAMETERS.items():
        require(parameters.get(name) == expected and type(parameters.get(name)) is type(expected),
                'Parameter differs from answer-only baseline: ' + name)
    if 'enable_thinking' in parameters:
        require(parameters['enable_thinking'] is False, 'Native thinking setting must remain disabled')
    require(set(parameters) <= set(BASELINE_PARAMETERS) | {'enable_thinking'},
            'Unknown generation parameter; explicitly audit before changing baseline')


def checked_path(root, image):
    require(isinstance(image.get('path'), str) and bool(image['path']), 'Image path missing')
    root = Path(root).resolve()
    path = Path(image['path'])
    path = (path if path.is_absolute() else root / path).resolve()
    require(path.is_relative_to(root), 'Image path escapes suite root')
    require(path.is_file(), 'Frozen input image missing')
    expected = image.get('sha256')
    require(isinstance(expected, str) and len(expected) == 64, 'Image SHA-256 missing')
    require(sha(path.read_bytes()) == expected, 'Frozen PNG bytes changed')
    return path


def encode_request(processor, request, root):
    """Encode exactly the prepared multi-turn dialogue; close every image handle.

    The processor receives real images in the same role/content order as the
    frozen request. No GT, sidecar data, inferred text or new instructions enter
    the dialogue here. The image bytes are checked before any template call.
    """
    from PIL import Image
    messages = request.get('messages')
    require(isinstance(messages, list) and bool(messages), 'Prepared messages missing')
    condition, variant = request.get('condition'), request.get('variant')
    require(condition in ('single', 'sparse'), 'Unknown frame condition')
    if variant in ('B0', 'B0_minimal', 'B1', 'B1_rules'):
        expected_roles = ['user']
        expected_images = 1 if condition == 'single' else 3
    elif variant in ('B2', 'B2_rules_4shot'):
        expected_roles = ['user', 'assistant'] * 4 + ['user']
        expected_images = 5 if condition == 'single' else 7
    else:
        raise ValueError('Only frozen B0, B1 name-correction, and B2 conditions are supported')
    require([message.get('role') for message in messages] == expected_roles,
            'Prepared conversation role sequence changed')
    checked = []
    for message in messages:
        content = message.get('content')
        require(isinstance(content, list) and bool(content), 'Message content must be a nonempty array')
        for part in content:
            require(isinstance(part, dict) and part.get('type') in ('image', 'text'), 'Unsupported content item')
            if part['type'] == 'text':
                require(isinstance(part.get('text'), str), 'Text content must be a string')
            else:
                require(message['role'] == 'user', 'Only user messages may contain prepared images')
                checked.append((part, checked_path(root, part)))
    require(len(checked) == expected_images, 'B0/B2 image count changed')
    # Open no image until every referenced path and hash passed verification.
    by_part = {id(part): path for part, path in checked}
    with ExitStack() as stack:
        wire = []
        for message in messages:
            content = []
            for part in message['content']:
                if part['type'] == 'text':
                    content.append({'type': 'text', 'text': part['text']})
                else:
                    with Image.open(by_part[id(part)]) as opened:
                        require(opened.format == 'PNG', 'Expected original rendered PNG input')
                        image = opened.convert('RGB')
                    stack.callback(image.close)
                    content.append({'type': 'image', 'image': image})
            wire.append({'role': message['role'], 'content': content})
        # Identical compatibility knob to the frozen baseline encoder: only
        # templates that declare this argument receive enable_thinking=False.
        kwargs = {'enable_thinking': False} if 'enable_thinking' in str(processor.chat_template) else {}
        batch = processor.apply_chat_template(wire, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors='pt', **kwargs)
    return batch, {'image_sha256_in_message_order': [part['sha256'] for part, _ in checked],
                   'messages_sha256': sha(canonical(messages)),
                   'chat_template_sha256': sha(str(processor.chat_template).encode()),
                   'template_kwargs': kwargs}
