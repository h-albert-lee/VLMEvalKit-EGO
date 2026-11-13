"""Portable extraction of the historical repeat/text-only/name-hidden interventions.

This wrapper is new; the intervention text and input-hash serialization match
the historical preparation script. It does not launch an evaluation.
"""
import copy
import hashlib
import json

SPARSE_VIEW = 'The three images are in chronological order. Red boxes designate the same target object. Judge at the final frame, using the earlier frames as context.'
REPEAT_VIEW = 'The final frame is repeated three times; these are not different time points. Red boxes designate the target. Judge at this moment.'
TEXT_VIEW = 'No images or narration are provided. Only the target name and task rules are available.'


def derive_control(base, condition, target_name=None):
    if base.get('condition') != 'sparse' or len(base['images']) != 3 or len(base['image_sha256']) != 3:
        raise ValueError('Expected a frozen three-frame sparse query')
    if base['prompt'].count(SPARSE_VIEW) != 1:
        raise ValueError('Sparse view instruction must occur exactly once')
    out = copy.deepcopy(base)
    if condition == 'repeat':
        out['prompt'] = base['prompt'].replace(SPARSE_VIEW, REPEAT_VIEW, 1)
        out['images'] = base['images'][-1:] * 3
        out['image_sha256'] = base['image_sha256'][-1:] * 3
    elif condition == 'text_only':
        out['prompt'] = base['prompt'].replace(SPARSE_VIEW, TEXT_VIEW, 1)
        out['images'], out['image_sha256'] = [], []
    elif condition == 'name_hidden':
        if not isinstance(target_name, str) or not target_name:
            raise ValueError('A target name is required to check the replaced prefix')
        prefix = 'Judge the attribution of the designated target object: ' + target_name + '.'
        if not base['prompt'].startswith(prefix):
            raise ValueError('Target-name prefix differs')
        out['prompt'] = 'Judge the attribution of the designated target object: (the object marked by the red boxes).' + base['prompt'][len(prefix):]
    else:
        raise ValueError('Unknown control condition')
    out['condition'] = condition
    out['prompt_sha256'] = hashlib.sha256(out['prompt'].encode()).hexdigest()
    out['input_sha256'] = hashlib.sha256(json.dumps(dict(prompt=out['prompt'], image_sha256=out['image_sha256'], version=out['version']), sort_keys=True).encode()).hexdigest()
    return out
