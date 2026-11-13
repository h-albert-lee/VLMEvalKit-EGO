"""Portable frontier evaluation core. No live transport or launcher is supplied."""
import base64
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
from egoown_cvpr.core import digest, parse_answer


def request(*args, **kwargs):
    raise RuntimeError("Live transport disabled; supply an explicitly authorized transport")


def atomic(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)

def append(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(obj, ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())

def read_log(path):
    path = Path(path)
    if not path.exists(): return []
    # A torn write must be inspected, never silently discarded/re-billed.
    return [json.loads(s) for s in path.read_text().splitlines() if s]

def make_payload(root, model, item):
    content = [{'type':'text','text':item['prompt']}]
    for path, sha in zip(item['images'], item['image_sha256']):
        file = (root/path).resolve()
        if not file.is_relative_to(root.resolve()): raise ValueError('Image escapes task root')
        raw = file.read_bytes()
        if digest(raw) != sha: raise ValueError('Frozen image changed')
        content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(raw).decode(),'detail':'high'}})
    computed = digest(json.dumps({'prompt':item['prompt'],'image_sha256':item['image_sha256'],'version':item['version']},sort_keys=True).encode())
    if computed != item['input_sha256']: raise ValueError('Frozen input changed')
    return {**model['params'], 'messages':[{'role':'user','content':content}]}

def infer(root, key, model, item, attempt_id, *, transport=None):
    request_transport = transport or request
    began = time.time()
    base = dict(attempt_id=attempt_id, model=model['id'], model_key=model['key'],
                canonical_slug=model['canonical_slug'], id=item['id'], condition=item['condition'],
                input_sha256=item['input_sha256'], options=item['options'], gt_label=item['gt_label'],
                started_at=began, parameters=model['params'])
    try:
        payload = make_payload(root, model, item)
        response = request_transport(key, 'chat/completions', payload, timeout=model.get('request_timeout_seconds',600))
        usage = response.get('usage') or {}
        cost = usage.get('cost')
        if cost is None and response.get('id'):
            for delay in [1,3,8]:
                try:
                    time.sleep(delay)
                    info = request_transport(key, 'generation?id='+urllib.parse.quote(response['id']), timeout=30)['data']
                    cost = info.get('total_cost'); base['generation_metadata'] = info
                    if cost is not None: break
                except Exception: pass
        choice = (response.get('choices') or [{}])[0]
        message = choice.get('message') or {}
        raw = message.get('content')
        if isinstance(raw,list): raw=''.join(x.get('text','') for x in raw if x.get('type')=='text')
        raw = raw if isinstance(raw,str) else ''
        error = 'api_response_error' if response.get('error') else None
        return {**base, 'finished_at':time.time(), 'seconds':time.time()-began,
                'raw_response':response, 'raw':raw, 'pred_label':None if error else parse_answer(raw,item['options']),
                'provider':response.get('provider'), 'response_model':response.get('model'),
                'service_tier':response.get('service_tier'),
                'finish_reason':choice.get('finish_reason'), 'usage':usage,
                'cost_usd':None if cost is None else float(cost), 'error':error,
                'billing_uncertain':cost is None, 'retryable':False}
    except urllib.error.HTTPError as e:
        # Preserve provider error details, with the task key redacted even if echoed.
        raw_error = e.read().decode(errors='replace').replace(key,'[REDACTED]')[:16000]
        # Rate limits/validation rejections do not produce a prediction. Retry only 429.
        return {**base, 'finished_at':time.time(), 'error':'http_'+str(e.code), 'http_status':e.code,
                'raw_error':raw_error, 'cost_usd':0.0 if e.code in [400,401,402,403,404,422,429] else None,
                'billing_uncertain':e.code not in [400,401,402,403,404,422,429], 'retryable':e.code==429,
                'raw':'', 'pred_label':None}
    except Exception as e:
        # Unknown completion status is not automatically re-sent (possible double charge).
        return {**base, 'finished_at':time.time(), 'error':type(e).__name__,
                'error_detail':str(e).replace(key,'[REDACTED]')[:1000],
                'cost_usd':None, 'billing_uncertain':True, 'retryable':False, 'raw':'','pred_label':None}

def is_recorded_moderation_rejection(p):
    """Keep an explicitly non-billed provider refusal as a final failed record."""
    if (p.get('error') != 'http_403' or p.get('http_status') != 403
            or p.get('cost_usd') != 0 or p.get('billing_uncertain') is not False):
        return False
    try:
        error = json.loads(p['raw_error'])['error']
        message = error.get('message', '').lower()
        reasons = error.get('metadata', {}).get('reasons')
        return (error.get('code') == 403 and isinstance(reasons, list) and bool(reasons)
                and 'requires moderation' in message
                and 'your input was flagged' in message
                and 'no credits were charged' in message)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False

def pilot_report(config, inputs, predictions, pilot_ids):
    report = {'jobs':[], 'cost_usd':0.0, 'passed':True, 'expected':len(pilot_ids)*len(config['models'])*len(config['conditions'])}
    for m in config['models']:
        for cond in config['conditions']:
            pp=[p for p in predictions if p['model_key']==m['key'] and p['condition']==cond and p['id'] in pilot_ids]
            issues = dict(errors=sum(bool(p.get('error')) for p in pp),
                          empty=sum(not p.get('raw') for p in pp),
                          truncated=sum(p.get('finish_reason')=='length' for p in pp),
                          provider_mismatch=sum(p.get('provider')!=m['expected_provider'] for p in pp),
                          model_mismatch=sum(p.get('response_model') not in [m['id'],m['canonical_slug']] for p in pp),
                          tier_mismatch=sum(p.get('service_tier') not in [None,'default'] for p in pp),
                          billing_uncertain=sum(p.get('cost_usd') is None for p in pp))
            passed=len(pp)==len(pilot_ids) and not any(issues.values())
            cost=sum(p.get('cost_usd') or 0 for p in pp)
            report['jobs'].append(dict(model=m['id'],condition=cond,n=len(pp),passed=passed,
                                       unparsed=sum(p.get('pred_label') is None for p in pp),cost_usd=cost,**issues))
            report['cost_usd']+=cost; report['passed'] &= passed
    report['projected_full_usd']=report['cost_usd']*config['n']/len(pilot_ids)
    report['projection_is_estimate']=True
    # Accuracy/unparsed rate are not gates: never tune prompt/parser to boost scores.
    return report
