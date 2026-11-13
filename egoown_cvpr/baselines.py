"""CPU-only group-held-out controls; no VLM calls or image model training."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from .core import load_bundle,metrics,write_json,VERSION

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    meta,rows=load_bundle(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    y=np.array([r['label'] for r in rows]);groups=np.array([r['group_id'] for r in rows])
    geom=np.array([[f['bbox'][k] for f in r['frames'] for k in ['x_min','y_min','x_max','y_max']] for r in rows])
    names=np.array([[r['target_name']] for r in rows]);sources=np.array([[r['source']] for r in rows])
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20261105).split(geom,y,groups))
    for train,test in folds:assert not set(groups[train])&set(groups[test])
    specs={'majority':(np.ones((len(rows),1)),lambda:DummyClassifier(strategy='most_frequent')),
           'geometry':(geom,lambda:make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))),
           'object_name':(names,lambda:make_pipeline(OneHotEncoder(handle_unknown='ignore'),LogisticRegression(max_iter=2000))),
           'source':(sources,lambda:make_pipeline(OneHotEncoder(handle_unknown='ignore'),LogisticRegression(max_iter=2000)))}
    for name,(x,factory) in specs.items():
        predictions=[]
        for fold,(tr,te) in enumerate(folds):
            model=factory();model.fit(x[tr],y[tr]);pred=model.predict(x[te])
            predictions.extend({'id':rows[i]['id'],'pred_label':str(v),'fold':fold,'error':None} for i,v in zip(te,pred))
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in predictions))
        write_json(out/(name+'-metrics.json'),metrics(rows,predictions))
    write_json(out/'run.json',{'version':VERSION,'dataset_sha256':meta['manifest_sha256'],'method':'5-fold StratifiedGroupKFold; no tuning; fixed C=1; preprocessing fit inside each training fold','status':'complete'})

if __name__=='__main__':main()
