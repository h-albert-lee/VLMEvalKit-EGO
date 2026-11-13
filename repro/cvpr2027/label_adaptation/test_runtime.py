import tempfile
import unittest
from pathlib import Path
from .run import Ledger, combine
from .protocol import write


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def request(self):
        return {'request_key': 'key', 'input_fingerprint': 'frozen'}

    def test_durable_response_recovers_without_second_dispatch(self):
        ledger = Ledger(self.root)
        attempt = ledger.start(self.request())
        write(self.root / 'responses/key.json', {'request_key': 'key', 'attempt_id': attempt, 'cost_usd': 0.5})
        resumed = Ledger(self.root)
        self.assertEqual(resumed.spent(), 0.5)
        self.assertFalse(resumed.unresolved())
        with self.assertRaisesRegex(ValueError, 'already started'):
            resumed.start(self.request())

    def test_missing_response_blocks_dispatch(self):
        ledger = Ledger(self.root)
        ledger.start(self.request())
        resumed = Ledger(self.root)
        with self.assertRaisesRegex(ValueError, 'Unresolved'):
            resumed.start({'request_key': 'next', 'input_fingerprint': 'other'})

    def test_modified_response_is_rejected(self):
        ledger = Ledger(self.root)
        attempt = ledger.start(self.request())
        response = {'request_key': 'key', 'attempt_id': attempt, 'cost_usd': 0.5}
        ledger.finish(response)
        write(self.root / 'responses/key.json', {**response, 'cost_usd': 0.1})
        with self.assertRaisesRegex(ValueError, 'Receipt changed'):
            Ledger(self.root)

    def test_error_does_not_become_answer(self):
        record = {k: 'x' for k in ('suite_id', 'request_key', 'input_fingerprint', 'id', 'model_key', 'model', 'revision', 'provider', 'condition', 'variant')}
        record.update(backend='transformers', options=['MINE', 'OTHERS', 'SHARED', 'AMBIGUOUS'])
        result = combine(record, 'attempt', {'raw_text': 'A', 'error': 'RuntimeError'})
        self.assertIsNone(result['pred_label'])
        self.assertEqual(result['cost_usd'], 0)


if __name__ == '__main__':
    unittest.main()
