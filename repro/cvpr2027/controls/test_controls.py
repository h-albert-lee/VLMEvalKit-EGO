"""Offline tests using synthetic records; no dataset or model download."""
import copy
import unittest
from analysis import paired_summary
from input_variants import derive_control, SPARSE_VIEW, TEXT_VIEW
from selection import select_cohort
from egoown_cvpr.modeling import encode_input

class DummyProcessor:
    chat_template = 'enable_thinking'
    def apply_chat_template(self, messages, **kwargs):
        return {'messages': messages, 'kwargs': kwargs}

class ControlsTest(unittest.TestCase):
    def test_text_only_encodes_no_image(self):
        out = encode_input(DummyProcessor(), {'images': [], 'prompt': 'test'})
        self.assertEqual(out['messages'], [{'role': 'user', 'content': [{'type': 'text', 'text': 'test'}]}])
        self.assertFalse(out['kwargs']['enable_thinking'])

    def test_refusal_unparsed_are_wrong(self):
        rows = [{'id': 'a', 'label': 'MINE'}, {'id': 'b', 'label': 'AMBIGUOUS'}]
        base = {'a': {'pred_label': None, 'error': 'refusal'}, 'b': {'pred_label': 'AMBIGUOUS'}}
        other = {'a': {'pred_label': 'MINE'}, 'b': {'pred_label': None}}
        x = paired_summary(rows, base, other)
        self.assertEqual((x['wrong_to_correct'], x['correct_to_wrong'], x['accuracy_delta_other_minus_sparse']), (1, 1, 0))

    def test_bootstrap_keeps_pairs(self):
        rows = [{'id': str(i), 'label': 'MINE'} for i in range(10)]
        base = {r['id']: {'pred_label': 'OTHERS'} for r in rows}
        other = {r['id']: {'pred_label': 'MINE'} for r in rows}
        self.assertEqual(paired_summary(rows, base, other, True)['paired_bootstrap_95_percentile'], [1, 1])

    def query(self):
        return dict(condition='sparse', prompt='Judge the attribution of the designated target object: cup.\n' + SPARSE_VIEW, images=['first.png', 'second.png', 'last.png'], image_sha256=['a', 'b', 'c'], version='synthetic', options=['MINE', 'OTHERS', 'SHARED', 'AMBIGUOUS'])

    def test_input_interventions_preserve_query_identity(self):
        base = self.query(); before = copy.deepcopy(base)
        repeat = derive_control(base, 'repeat')
        self.assertEqual(repeat['images'], ['last.png'] * 3)
        self.assertEqual(repeat['image_sha256'], ['c'] * 3)
        text = derive_control(base, 'text_only')
        self.assertEqual(text['images'], [])
        self.assertIn(TEXT_VIEW, text['prompt'])
        hidden = derive_control(base, 'name_hidden', 'cup')
        self.assertEqual(hidden['images'], base['images'])
        self.assertNotIn(': cup.', hidden['prompt'])
        self.assertEqual(hidden['options'], base['options'])
        self.assertEqual(base, before)
        self.assertEqual(len({x['input_sha256'] for x in [repeat, text, hidden]}), 3)
        with self.assertRaises(ValueError): derive_control(base, 'name_hidden', 'bowl')

    def test_selection_excludes_groups_and_changed_names(self):
        rows = [dict(id=f'{tier}-{i}', group_id=f'{tier}-{i}', taxonomy=tier, source='source-a' if i % 2 else 'source-b', label='MINE' if tier == 'T3' else 'AMBIGUOUS', target_name='cup') for tier in ['T3', 'T4'] for i in range(130)]
        old = {r['id']: dict(r) for r in rows}; old['T3-1']['target_name'] = 'bowl'
        selected, audit, excluded = select_cohort(rows, old, [rows[0]], [])
        self.assertEqual(len(selected), 200)
        self.assertEqual(len({r['group_id'] for r in selected}), 200)
        self.assertNotIn('T3-0', {r['id'] for r in selected})
        self.assertNotIn('T3-1', {r['id'] for r in selected})
        self.assertEqual(selected, select_cohort(list(reversed(rows)), old, [rows[0]], [])[0])

if __name__ == '__main__': unittest.main()
