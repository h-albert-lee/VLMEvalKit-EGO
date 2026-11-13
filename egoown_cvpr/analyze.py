"""Source/cohort breakdowns and group-bootstrap uncertainty for frozen runs."""
import argparse,json
from pathlib import Path
import numpy as np
from .core import LABELS,read_jsonl,metrics,write_json

def summarize(matrix):
    support=matrix.sum(axis=1);tp=np.diag(matrix[:,:4]);pred=matrix[:,:4].sum(axis=0)
    recall=np.divide(tp,support,out=np.zeros(4),where=support>0)
    f1=np.divide(2*tp,support+pred,out=np.zeros(4),where=(support+pred)>0)
    return np.array([tp.sum()/max(1,matrix.sum()),f1.mean(),recall.mean()])

def group_matrices(rows,preds,groups):
    p={r['id']:r for r in preds}; gi={g:i for i,g in enumerate(groups)}
    arr=np.zeros((len(groups),4,7),dtype=np.int64)
    for row in rows:
        v=p.get(row['id']);label='MISSING' if v is None else 'ERROR' if v.get('error') else v.get('pred_label') or 'UNPARSED'
        arr[gi[row['group_id']],LABELS.index(row['label']),(LABELS+['UNPARSED','ERROR','MISSING']).index(label)]+=1
    return arr

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--run',required=True);ap.add_argument('--compare-run');ap.add_argument('--subset',choices=['full','common200'],default='full');ap.add_argument('--out',required=True);a=ap.parse_args()
    root=Path(a.root);meta=json.loads((root/'dataset.json').read_text());rows=read_jsonl(root/'manifest.jsonl')
    if a.subset=='common200':
        ids=set(json.loads((root/'subsets/common200.json').read_text()));rows=[r for r in rows if r['id'] in ids]
    ids={r['id'] for r in rows}
    def read_run(path):
        d=json.loads((Path(path)/'run.json').read_text())
        if d['config']['dataset_sha256']!=meta['manifest_sha256'] or d['config']['dry_run']:raise ValueError('Not a real run on this dataset')
        p=read_jsonl(Path(path)/'predictions.jsonl')
        if len({x['id'] for x in p})!=len(p):raise ValueError('Duplicate prediction IDs')
        if not ids<={x['id'] for x in p}:raise ValueError('Incomplete run for requested analysis subset')
        return [r for r in p if r['id'] in ids]
    pred=read_run(a.run);report={'overall':metrics(rows,pred),'by_source':{},'by_review_stage':{}}
    for key,outkey in [('source','by_source'),('review_stage','by_review_stage')]:
        for val in sorted({r[key] for r in rows}):
            selected=[r for r in rows if r[key]==val];keep={r['id'] for r in selected}
            report[outkey][val]=metrics(selected,[p for p in pred if p['id'] in keep])
    groups=sorted({r['group_id'] for r in rows});mat=group_matrices(rows,pred,groups)
    other=group_matrices(rows,read_run(a.compare_run),groups) if a.compare_run else None
    rng=np.random.default_rng(20261105);boot=[];deltas=[]
    for _ in range(1000):
        sample=rng.integers(0,len(groups),len(groups));v=summarize(mat[sample].sum(axis=0));boot.append(v)
        if other is not None:deltas.append(v-summarize(other[sample].sum(axis=0)))
    keys=['accuracy','macro_f1','balanced_accuracy']
    report['group_bootstrap_95ci']={k:np.quantile(np.array(boot)[:,i],[.025,.975]).tolist() for i,k in enumerate(keys)}
    if deltas:report['paired_difference_95ci']={k:np.quantile(np.array(deltas)[:,i],[.025,.975]).tolist() for i,k in enumerate(keys)}
    report['uncertainty_note']='1000 group bootstrap draws; available video/session groups, not guaranteed independent identities across sources. Comparison is this run minus compare-run on identical IDs.'
    write_json(a.out,report)
if __name__=='__main__':main()
