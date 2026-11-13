"""Portable B0/B1/B2 message construction from caller-supplied frozen inputs."""
import copy
from . import protocol as p

VARIANTS = ('B0_minimal', 'B1_rules', 'B2_rules_4shot')

def ordered_content(parts, wire_order):
    p.require(wire_order in {'text_then_images', 'images_then_text'}, 'Unknown model wire order')
    text, images = [x for x in parts if x['type'] == 'text'], [x for x in parts if x['type'] == 'image']
    p.require(len(text)+len(images) == len(parts), 'Unsupported content item')
    return copy.deepcopy(text+images if wire_order == 'text_then_images' else images+text)

def make_request(suite, model, query, variant, block, demos, pilot_ids):
    prompt = query['baseline_prompt']
    p.require(prompt.count(block) == 1, 'Ablation block is not present exactly once')
    if variant == 'B0_minimal':
        prompt = prompt.replace(block, '', 1)
    p.require(variant in VARIANTS, 'Unknown B variant')
    query_parts = [{'type': 'text', 'text': prompt}] + [
        {'type': 'image', 'path': path, 'sha256': h, 'mime': 'image/png', 'detail': 'high'}
        for path, h in zip(query['images'], query['image_sha256'])]
    messages = []
    if variant == 'B2_rules_4shot':
        for demo in demos:
            messages.extend([{'role': 'user', 'content': ordered_content(demo['message']['content'], model['wire_order'])},
                             {'role': 'assistant', 'content': [{'type': 'text', 'text': demo['assistant_letter']}]}])
    messages.append({'role': 'user', 'content': ordered_content(query_parts, model['wire_order'])})
    return p.seal({'suite_id': suite, 'model_key': model['key'], 'model': model['id'],
                   'revision': model['revision'], 'provider': model['provider'], 'backend': model['backend'],
                   'parameters': model['parameters'], 'messages': messages, 'id': query['id'],
                   'condition': query['condition'], 'variant': variant, 'options': query['options'],
                   'pilot': query['id'] in pilot_ids, 'baseline_input_sha256': query['input_sha256'],
                   'original_baseline_input_sha256': query['original_baseline_input_sha256'],
                   'target_name': query['target_name'], 'wire_order': model['wire_order'],
                   'demo_ids': [d['id'] for d in demos] if variant == 'B2_rules_4shot' else []})
