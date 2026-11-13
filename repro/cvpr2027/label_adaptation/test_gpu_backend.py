"""Offline tests. No torch import, GPU use, network, or model downloads."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from PIL import Image

from .gpu_backend import BASELINE_PARAMETERS, encode_request, validate_parameters


class FakeProcessor:
    def __init__(self, template='baseline-template'):
        self.chat_template = template
        self.observed = None
        self.kwargs = None
        self.image_handles = []
        self.calls = 0

    def apply_chat_template(self, messages, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        self.observed = []
        for message in messages:
            parts = []
            for part in message['content']:
                if part['type'] == 'image':
                    image = part['image']
                    self.image_handles.append(image)
                    parts.append(('image', image.getpixel((0, 0))))
                else:
                    parts.append(('text', part['text']))
            self.observed.append((message['role'], parts))
        return {'encoded': True}


class GPUEncodingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = []
        for i in range(7):
            path = self.root / f'{i}.png'
            Image.new('RGB', (2, 2), color=(i * 30, i * 10, i)).save(path)
            self.images.append({'type': 'image', 'path': path.name,
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

    def tearDown(self):
        self.temp.cleanup()

    def request(self, condition='sparse', variant='B2_rules_4shot'):
        count = 1 if condition == 'single' else 3
        messages = []
        offset = 0
        if variant.startswith('B2'):
            for i in range(4):
                messages += [{'role': 'user', 'content': [copy.deepcopy(self.images[i]),
                    {'type': 'text', 'text': f'Frozen example {i}; exact options'}]},
                    {'role': 'assistant', 'content': [{'type': 'text', 'text': chr(65 + i)}]}]
            offset = 4
        messages.append({'role': 'user', 'content': [copy.deepcopy(x) for x in self.images[offset:offset + count]] +
                         [{'type': 'text', 'text': 'Frozen query: target cup; answer one letter.'}]})
        return {'condition': condition, 'variant': variant, 'messages': messages,
                'gt_label': 'THIS MUST NEVER ENTER THE PROMPT', 'parameters': dict(BASELINE_PARAMETERS)}

    def test_multiturn_roles_images_text_exact_and_no_metadata_injection(self):
        request = self.request()
        before = copy.deepcopy(request)
        processor = FakeProcessor()
        batch, metadata = encode_request(processor, request, self.root)
        self.assertEqual(request, before)
        self.assertEqual([r[0] for r in processor.observed], ['user', 'assistant'] * 4 + ['user'])
        colors = [p[1] for _, parts in processor.observed for p in parts if p[0] == 'image']
        self.assertEqual(colors, [(i * 30, i * 10, i) for i in range(7)])
        self.assertEqual([parts[0][1] for role, parts in processor.observed if role == 'assistant'], ['A', 'B', 'C', 'D'])
        self.assertNotIn('THIS MUST NEVER', str(processor.observed))
        self.assertEqual(processor.observed[-1][1][-1][1], request['messages'][-1]['content'][-1]['text'])
        self.assertEqual(metadata['image_sha256_in_message_order'], [x['sha256'] for x in self.images])
        self.assertEqual(batch, {'encoded': True})
        for image in processor.image_handles:
            with self.assertRaises(ValueError):
                image.getpixel((0, 0))

    def test_all_B0_B2_frame_counts(self):
        for condition, variant, expected in [('single', 'B0_minimal', 1), ('sparse', 'B0_minimal', 3),
                                             ('single', 'B1_rules', 1), ('sparse', 'B1_rules', 3),
                                             ('single', 'B2_rules_4shot', 5), ('sparse', 'B2_rules_4shot', 7)]:
            processor = FakeProcessor()
            _, info = encode_request(processor, self.request(condition, variant), self.root)
            self.assertEqual(len(info['image_sha256_in_message_order']), expected)

    def test_baseline_template_knob_only_when_supported(self):
        regular, thinking = FakeProcessor(), FakeProcessor('supports enable_thinking')
        for processor in [regular, thinking]:
            encode_request(processor, self.request('single', 'B0_minimal'), self.root)
            self.assertEqual(processor.kwargs['add_generation_prompt'], True)
            self.assertEqual(processor.kwargs['return_tensors'], 'pt')
        self.assertNotIn('enable_thinking', regular.kwargs)
        self.assertIs(thinking.kwargs['enable_thinking'], False)

    def test_hash_mismatch_fails_before_template(self):
        request = self.request()
        request['messages'][-1]['content'][0]['sha256'] = '0' * 64
        processor = FakeProcessor()
        with self.assertRaisesRegex(ValueError, 'PNG bytes changed'):
            encode_request(processor, request, self.root)
        self.assertEqual(processor.calls, 0)

    def test_invalid_role_count_and_root_escape_rejected(self):
        request = self.request()
        request['messages'][1]['role'] = 'user'
        with self.assertRaisesRegex(ValueError, 'role sequence'):
            encode_request(FakeProcessor(), request, self.root)
        request = self.request('single', 'B0_minimal')
        request['messages'][0]['content'][0]['path'] = '../outside.png'
        with self.assertRaisesRegex(ValueError, 'escapes'):
            encode_request(FakeProcessor(), request, self.root)

    def test_parameters_cannot_silently_drift(self):
        validate_parameters(dict(BASELINE_PARAMETERS))
        for changes in [{'max_new_tokens': 1024}, {'do_sample': True}, {'dtype': 'float16'}, {'seed': 1}, {'temperature': 0}]:
            with self.assertRaises(ValueError):
                validate_parameters({**BASELINE_PARAMETERS, **changes})

    def test_images_closed_when_processor_fails(self):
        class Broken(FakeProcessor):
            def apply_chat_template(self, messages, **kwargs):
                super().apply_chat_template(messages, **kwargs)
                raise RuntimeError('processor failure')
        processor = Broken()
        with self.assertRaises(RuntimeError):
            encode_request(processor, self.request(), self.root)
        for image in processor.image_handles:
            with self.assertRaises(ValueError):
                image.getpixel((0, 0))


if __name__ == '__main__':
    unittest.main()
