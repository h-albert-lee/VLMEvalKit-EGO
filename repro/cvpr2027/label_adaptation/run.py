"""Local first-attempt ledger and result combination; no execution launcher."""
from decimal import Decimal
import time
import uuid
from .protocol import *

class Ledger:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'ledger.jsonl'
        self.starts, self.finishes, self.results = {}, {}, {}
        for row in readl(self.path):
            target = self.starts if row['event'] == 'start' else self.finishes
            require(row['event'] in ('start', 'finish'), 'Unknown event')
            require(row['request_key'] not in target, 'Duplicate event')
            target[row['request_key']] = row
        require(set(self.finishes) <= set(self.starts), 'Finish without start')
        for key, start in self.starts.items():
            receipt = self.directory / 'responses' / (key + '.json')
            if receipt.exists():
                value = load(receipt)
                require(value['request_key'] == key and value['attempt_id'] == start['attempt_id'], 'Receipt identity mismatch')
                if key not in self.finishes:
                    self.finish(value)
                require(self.finishes[key]['sha256'] == sha(receipt.read_bytes()), 'Receipt changed')
                self.results[key] = value
            else:
                require(key not in self.finishes, 'Finished receipt missing')

    def unresolved(self):
        return set(self.starts) - set(self.finishes)

    def spent(self):
        return sum((Decimal(str(r['cost_usd'])) for r in self.results.values() if r.get('cost_usd') is not None), Decimal(0))

    def start(self, record, reservation='0'):
        require(not self.unresolved(), 'Unresolved prior attempt; do not resend')
        require(record['request_key'] not in self.starts, 'Request already started')
        row = dict(event='start', request_key=record['request_key'], attempt_id=str(uuid.uuid4()),
                   at=time.time(), reservation_usd=str(reservation), input_fingerprint=record['input_fingerprint'])
        append(self.path, row)
        self.starts[record['request_key']] = row
        return row['attempt_id']

    def finish(self, response):
        key = response['request_key']
        require(key in self.starts and response['attempt_id'] == self.starts[key]['attempt_id'], 'Unknown response identity')
        path = self.directory / 'responses' / (key + '.json')
        if not path.exists():
            write(path, response)
        if key not in self.finishes:
            row = dict(event='finish', request_key=key, attempt_id=response['attempt_id'],
                       at=time.time(), sha256=sha(path.read_bytes()), cost_usd=response.get('cost_usd'))
            append(self.path, row)
            self.finishes[key] = row
        self.results[key] = response

def checkpoint(root, lane, ledger, requests):
    jobs = {}
    for row in ledger.results.values():
        job = row['model_key'] + '--' + row['condition'] + '--' + row['variant']
        item = jobs.setdefault(job, dict(completed=0, errors=0, unparsed=0))
        item['completed'] += 1
        item['errors'] += bool(row.get('error'))
        item['unparsed'] += row.get('pred_label') is None and not row.get('error')
    write(root / 'runtime' / lane / 'state.json', dict(
        lane=lane, expected=len(requests), completed=len(ledger.results),
        complete=len(ledger.results) == len(requests) and not ledger.unresolved(),
        actual_cost_usd=str(ledger.spent()), unresolved=list(ledger.unresolved()), jobs=jobs, updated_at=time.time()))

def combine(record, attempt, response):
    result = {key: record[key] for key in ('suite_id', 'request_key', 'input_fingerprint', 'id', 'model_key',
            'model', 'revision', 'provider', 'backend', 'condition', 'variant', 'options')} | response | dict(
        attempt_id=attempt, completed_at=time.time(),
        pred_label=None if response.get('error') else parse_answer(response['raw_text'], record['options']),
        cost_usd=response.get('cost_usd', 0 if record['backend'] == 'transformers' else None))
    if record['backend'] == 'transformers' and response.get('error'):
        result['stop_required'] = True
    return result
