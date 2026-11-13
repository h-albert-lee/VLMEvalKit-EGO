"""Create a self-contained frozen bundle. Never promote proposed labels to GT."""
import argparse, collections, copy, hashlib, json, os, re, shutil, tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from .core import LABELS, check_box, digest, read_jsonl, write_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pool',required=True);ap.add_argument('--review',required=True);ap.add_argument('--new-frames',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args(); dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'dataset.json').exists(): raise RuntimeError('Refusing to overwrite frozen dataset')
    poolpath=Path(args.pool);pool=read_jsonl(poolpath)
    for r in pool:
        for f in r['frames']: f['path']=str((poolpath.parent/f['path']).resolve())
    snap=json.loads(Path(args.review).read_text()); h=snap['reviewers']['hanwool'];j=snap['reviewers']['jangha']
    exclusions=[];added=[]
    for a,b in zip(h,j):
        assert a['id']==b['id']
        reason=None
        if a['status']!='verified' or b['status']!='verified': reason='rejected_or_on_hold'
        elif a['label']!=b['label']: reason='reviewer_disagreement'
        elif a['label'] not in LABELS: reason='missing_label'
        if reason:
            exclusions.append({'id':a['id'],'reason':reason,'hanwool':{'label':a['label'],'status':a['status']},'jangha':{'label':b['label'],'status':b['status']}});continue
        # Verified metadata is immutable between the two UI stores.
        assert a['frames']==b['frames']
        r={'id':a['id'],'source':'egolife','batch':'egolife221-20261103','video_id':a['id'].split(':')[0],
           'group_id':'egolife:'+a['id'].split(':')[0], 'label':a['label'],'taxonomy':None,
           'review_stage':'two_human_labels_agree','review_provenance':{w:{k:v[k] for k in ['status','label','reason']} for w,v in [('hanwool',a),('jangha',b)]},'frames':[],'quality_flags':a.get('quality_notes',[])}
        try:
            for f in a['frames']:
                if len(f['objects'])!=1: raise ValueError('missing_or_nonunique_target_box')
                obj=f['objects'][0];check_box(obj['bbox']);p=Path(args.new_frames)/f['path']
                r['frames'].append({'tag':f['tag'],'timestamp_sec':f['timestamp_sec'],'path':str(p.resolve()),'bbox':obj['bbox'],'target_name':obj['label'],'sha256':digest(p.read_bytes())})
            if any(any(t in v['reason'] for t in ['보류','중복','제외']) for v in [a,b]):
                raise ValueError('verified_but_review_note_requires_hold')
            added.append(r)
        except (ValueError,FileNotFoundError) as e:
            exclusions.append({'id':a['id'],'reason':str(e)})
    raw=pool+added;seen=set();triples=set();accepted=[];files={}
    for original in raw:
        r=copy.deepcopy(original)
        if r['id'] in seen: raise ValueError('Duplicate ID: '+r['id'])
        seen.add(r['id'])
        assert r['label'] in LABELS and r.get('review_stage')
        r['frames'].sort(key=lambda f:['t-2','t-1','t'].index(f['tag']))
        assert len(r['frames'])==3
        key=tuple((f['sha256'],tuple(check_box(f['bbox']))) for f in r['frames'])
        if key in triples:
            exclusions.append({'id':r['id'],'reason':'exact_frames_and_target_duplicate'});continue
        triples.add(key)
        r['target_name']=r['frames'][-1]['target_name']
        # Avoid guessed seconds or cross-video splits for the legacy EgoLife timestamps.
        if r['source']=='egolife':
            m=re.search(r'(A\d+_[A-Z]+)_?(DAY\d+)',r['id'])
            if m:r['group_id']='egolife:'+m.group(1)+':'+m.group(2)
        for f in r['frames']:
            src=Path(f['path']);rel='frames/'+f['sha256']+src.suffix.lower()
            files.setdefault(rel,(src,f['sha256']));f['path']=rel
            f.pop('source_path',None)
        r['ready_for_evaluation']=True
        r['taxonomy_independently_validated']=False
        accepted.append(r)
    (dest/'frames').mkdir(exist_ok=True)
    def copy_frame(item):
        rel,(src,sha)=item;data=src.read_bytes()
        if digest(data)!=sha:raise ValueError('Changed source frame: '+str(src))
        target=dest/rel
        if not target.exists():
            # Copy rather than hardlink: future source edits must not alter frozen bytes.
            target.write_bytes(data)
        elif digest(target.read_bytes())!=sha: raise ValueError('Corrupt existing copy')
        return len(data)
    with ThreadPoolExecutor(max_workers=8) as ex: total=sum(ex.map(copy_frame,files.items()))
    accepted.sort(key=lambda r:r['id'])
    manifest=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in accepted).encode()
    (dest/'manifest.jsonl').write_bytes(manifest)
    # A small common panel is fixed once; all models must reuse these exact IDs.
    subset=[]
    for label in LABELS:
        groups=collections.defaultdict(list)
        for r in accepted:
            if r['label']==label:groups[r['group_id']].append(r)
        for g in groups:groups[g].sort(key=lambda r:digest(r['id'].encode()))
        picked=[]
        while len(picked)<50 and any(groups.values()):
            for g in sorted(groups,key=lambda s:digest(s.encode())):
                if groups[g] and len(picked)<50:picked.append(groups[g].pop(0)['id'])
        subset.extend(picked)
    write_json(dest/'subsets/common200.json',subset)
    write_json(dest/'exclusions.json',exclusions)
    meta={'version':'egoown-private-20261105-v1','created_at':datetime.now(timezone.utc).isoformat(),'n':len(accepted),
          'manifest_sha256':digest(manifest),'labels':dict(collections.Counter(r['label'] for r in accepted)),
          'sources':dict(collections.Counter(r['source'] for r in accepted)),'review_stages':dict(collections.Counter(r['review_stage'] for r in accepted)),
          'new_egolife_admitted':sum(r['batch']=='egolife221-20261103' for r in accepted),
          'unique_frames':len(files),'frame_bytes':total,'excluded_this_freeze':dict(collections.Counter(r['reason'] for r in exclusions)),
          'review_snapshot_sha256':digest(Path(args.review).read_bytes()),'selection':'All eligible collected rows; no class downsampling or oversampling.',
          'limitations':['Legacy EgoLife absolute timestamp mapping unverified; frame order uses stored tags, no fixed second-spacing assertion.',
                          'Legacy saved human labels and paired agreement are distinct review cohorts, not uniform independent double annotation.',
                          'Taxonomy not independently revalidated. Group IDs are conservative available metadata, not guaranteed cross-source identity resolution.']}
    write_json(dest/'dataset.json',meta)
    # Modest shard size for resumable download; only image assets are tarred.
    shards=[];(dest/'shards').mkdir(exist_ok=True);tf=None;size=0
    for rel in sorted(files):
        n=(dest/rel).stat().st_size
        if tf is None or size+n>256*1024*1024:
            if tf:tf.close()
            sp=dest/'shards'/f'frames-{len(shards):04d}.tar';shards.append(sp);tf=tarfile.open(sp,'w');size=0
        info=tf.gettarinfo(str(dest/rel),arcname=rel);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
        with (dest/rel).open('rb') as f:tf.addfile(info,f)
        size+=n
    if tf:tf.close()
    write_json(dest/'shards.json',[{'path':str(p.relative_to(dest)),'sha256':digest(p.read_bytes()),'bytes':p.stat().st_size} for p in shards])
    (dest/'README.md').write_text('---\npretty_name: EgoOwn private evaluation cache\n---\n# Private personal evaluation cache\n\nSource material retains its original terms. This repository is not a relicensing or public release.\n\nFrozen metadata: `dataset.json`. Raw frames are in checksummed tar shards. No API keys, review links, or credentials are included.\n')
    print(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
