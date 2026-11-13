"""Summarize all first attempts against the suite's frozen final labels."""
from collections import defaultdict
from .protocol import *
from .run import Ledger


def summarize(root):
    root = Path(root)
    verify_freeze(root)
    rows = readl(root / 'scoring-manifest.jsonl')
    by_id = {r['id']: r for r in rows}
    final_ids = {r['id'] for r in readl(root / 'final-scoring-manifest.jsonl')}
    requests = readl(root / 'requests.jsonl')
    lookup = {r['request_key']: r for r in requests}
    results = readl(root / 'baseline-imported.jsonl')
    for lane in ('gpu', 'api'):
        runtime = root / 'runtime' / lane
        if runtime.exists():
            ledger = Ledger(runtime)
            require(not ledger.unresolved(), 'Unresolved attempts must be reported before scoring')
            results.extend(ledger.results.values())
    require(len({r['request_key'] for r in results}) == len(results), 'Duplicate result')
    jobs = defaultdict(list)
    expected = defaultdict(set)
    for r in requests:
        expected[(r['model_key'], r['condition'], r['variant'])].add(r['id'])
    for r in results:
        q = lookup[r['request_key']]
        for key in ('id', 'input_fingerprint', 'model_key', 'condition', 'variant'):
            require(r[key] == q[key], 'Result identity mismatch: ' + key)
        if not r.get('reused'):
            for key in ('model', 'revision', 'provider', 'backend', 'options'):
                require(r[key] == q[key], 'Result settings mismatch: ' + key)
        raw = r.get('raw_response', '') if r.get('reused') else r.get('raw_text', '')
        parsed = None if r.get('error') else parse_answer(raw, q['options'])
        require(parsed == r.get('pred_label'), 'Saved prediction differs from raw response')
        jobs[(q['model_key'], q['condition'], q['variant'])].append(r)
    summary = {}
    for job, ids in expected.items():
        predictions = jobs[job]
        complete = {r['id'] for r in predictions} == ids
        primary_ids = ids & final_ids
        primary_predictions = [r for r in predictions if r['id'] in primary_ids]
        primary_complete = {r['id'] for r in primary_predictions} == primary_ids
        summary['--'.join(job)] = dict(expected=len(ids), completed=len(predictions), complete=complete,
            final_primary_n=len(primary_ids), final_primary_complete=primary_complete,
            metrics=metrics([by_id[i] for i in sorted(primary_ids)], primary_predictions) if primary_complete else None,
            exploratory_with_legacy_metrics=metrics([by_id[i] for i in sorted(ids)], predictions) if complete and ids != primary_ids else None,
            errors=sum(bool(r.get('error')) for r in predictions),
            unparsed=sum(r.get('pred_label') is None and not r.get('error') for r in predictions))
    report = dict(expected=len(requests), completed=len(results), complete=len(results) == len(requests),
                  primary_evaluation_n=len(final_ids), execution_cohort_n=len(by_id),
                  final_manifest_sha256=sha((root / 'scoring-manifest.jsonl').read_bytes()), jobs=summary)
    write(root / 'analysis' / 'summary.json', report)
    return report
