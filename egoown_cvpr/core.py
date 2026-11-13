from pathlib import Path
import hashlib
import json
import math
import random
import re
from PIL import Image, ImageDraw

LABELS = ['MINE', 'OTHERS', 'SHARED', 'AMBIGUOUS']
VERSION = 'cvpr-20261105-target-overlay-v2'
MAX_IMAGE_EDGE = 1024
CONDITIONS = ['sparse', 'single', 'repeat', 'no_box', 'crop', 'geometry_only', 'text_only']

def digest(data):
    return hashlib.sha256(data).hexdigest()

def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def check_box(b):
    if not isinstance(b, dict):
        raise ValueError('Missing target bbox')
    v = [b[k] for k in ('x_min', 'y_min', 'x_max', 'y_max')]
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v):
        raise ValueError('Non-finite bbox')
    if not (0 <= v[0] < v[2] <= 1 and 0 <= v[1] < v[3] <= 1):
        raise ValueError('BBox must be normalized xyxy')
    return v

def safe_path(root, relative):
    root = Path(root).resolve(); path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path escapes bundle')
    return path

def load_bundle(root):
    root = Path(root); meta = json.loads((root/'dataset.json').read_text())
    raw = (root/'manifest.jsonl').read_bytes()
    if digest(raw) != meta['manifest_sha256']:
        raise ValueError('Frozen manifest checksum mismatch')
    rows = read_jsonl(root/'manifest.jsonl')
    if len(rows) != meta['n'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Manifest cardinality mismatch')
    for r in rows:
        if r['label'] not in LABELS or len(r['frames']) != 3:
            raise ValueError('Invalid GT or frame count')
        if [f['tag'] for f in r['frames']] != ['t-2', 't-1', 't']:
            raise ValueError('Frame order mismatch')
        for f in r['frames']:
            check_box(f['bbox'])
            p = safe_path(root, f['path'])
            if not p.is_file() or digest(p.read_bytes()) != f['sha256']:
                raise ValueError('Missing/corrupt frozen frame: '+r['id'])
    return meta, rows

def render(root, frame, mode='overlay'):
    path = safe_path(root, frame['path'])
    raw = path.read_bytes()
    if digest(raw) != frame['sha256']:
        raise ValueError('Input frame checksum mismatch')
    box = check_box(frame['bbox'])
    key = digest(json.dumps([frame['sha256'], box, mode, VERSION]).encode())
    dest = Path(root)/'rendered'/f'{key}.png'
    if dest.exists():
        return str(dest)
    with Image.open(path) as src:
        im = src.convert('RGB')
    im.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    w,h=im.size
    xy = (max(0,int(box[0]*w)),max(0,int(box[1]*h)),min(w-1,int(math.ceil(box[2]*w))),min(h-1,int(math.ceil(box[3]*h))))
    if mode == 'crop':
        im = im.crop((xy[0],xy[1],xy[2]+1,xy[3]+1))
    elif mode == 'overlay':
        # One fixed color for every GT/source. No labels or annotation evidence.
        ImageDraw.Draw(im).rectangle(xy, outline=(255,0,0), width=max(2,round(min(w,h)/280)))
    elif mode != 'raw':
        raise ValueError(mode)
    dest.parent.mkdir(exist_ok=True)
    import os, tempfile
    with tempfile.NamedTemporaryFile(dir=dest.parent,suffix='.png',delete=False) as f:
        tmp=Path(f.name)
    try:
        im.save(tmp,format='PNG'); os.replace(tmp,dest)
    finally:
        tmp.unlink(missing_ok=True)
    return str(dest)

def build_input(root, row, condition='sparse', seed=0):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    order=LABELS.copy(); random.Random(int(digest(f'{seed}:{row["id"]}'.encode())[:16],16)).shuffle(order)
    frames=row['frames']
    if condition=='single': frames=frames[-1:]
    if condition=='repeat': frames=[frames[-1]]*3
    paths=[]
    if condition not in ['geometry_only','text_only']:
        mode={'no_box':'raw','crop':'crop'}.get(condition,'overlay')
        paths=[render(root,f,mode) for f in frames]
    target=row['target_name'] if condition != 'geometry_only' else '(target identity withheld; geometry only)'
    view={
        'sparse':'The three images are in chronological order. Red boxes designate the same target object. Judge at the final frame, using the earlier frames as context.',
        'single':'Only the final frame is shown. The red box designates the target. Judge at this moment.',
        'repeat':'The final frame is repeated three times; these are not different time points. Red boxes designate the target. Judge at this moment.',
        'no_box':'The three images are chronological. Locate the named target in the scene and judge at the final frame.',
        'crop':'The three images are chronological crops of the target. Judge at the final frame.',
        'geometry_only':'No images are shown. Only normalized target box coordinates, in chronological order, are provided: '+json.dumps([f['bbox'] for f in frames]),
        'text_only':'No images or narration are provided. Only the target name and task rules are available.'
    }[condition]
    # Operational rules retained from the reviewed task, not a new GT policy.
    prompt=f'''Judge the attribution of the designated target object: {target}.
Ownership is not the same as temporary possession. Use the supplied scene evidence and these benchmark conventions, not unobserved legal ownership.
MINE: attributed to the camera wearer.
OTHERS: attributed to another person.
SHARED: communal by function or established shared use.
AMBIGUOUS: insufficient evidence for attribution.
Boundary rules:
1. Tableware assigned to a person's place setting is attributed to that person even before contact.
2. A communal serving item remains SHARED during temporary individual use.
3. Communal-function items default to SHARED; personal-function items without attribution cues are AMBIGUOUS.
4. Holding or proximity alone does not establish attribution.
5. Moving one's item to shared space does not by itself transfer attribution.
{view}
Options:
'''+ '\n'.join(f'{chr(65+i)}. {label}' for i,label in enumerate(order))+'\nAnswer with exactly one capital letter: A, B, C, or D.'
    inputs={'condition':condition,'seed':seed,'version':VERSION,'id':row['id'],'prompt':prompt,'images':paths,'options':order}
    inputs['input_sha256']=digest(json.dumps({'prompt':prompt,'image_sha256':[digest(Path(p).read_bytes()) for p in paths],'version':VERSION},sort_keys=True).encode())
    return inputs

def parse_answer(text, options):
    # No picking a stray article A or a label merely mentioned in reasoning.
    text=re.sub(r'<think>.*?</think>','',str(text),flags=re.S).strip()
    m=re.fullmatch(r'[\[(]?([A-D])[\]).]?',text)
    if not m:
        m=re.search(r'(?:^|\n)\s*(?:Final answer|Answer)\s*:\s*([A-D])[\]).]?\s*$',text,re.I)
    if m: return options[ord(m.group(1).upper())-65]
    # Some checkpoints echo the complete option (e.g. "C. OTHERS").
    # Accept it only when the letter and label agree; never select a label
    # from additional prose or from contradictory answers.
    m=re.fullmatch(r'([A-D])[.)]\s*(MINE|OTHERS|SHARED|AMBIGUOUS)',text)
    if m:
        label=options[ord(m.group(1))-65]
        return label if label==m.group(2) else None
    # A separate final answer paragraph is also explicit (Gemma's format).
    # Reject multiple answer paragraphs and simulated follow-up turns.
    m=re.search(r'\n\s*\n([A-D])[.)]?\s*\Z',text)
    if m and not re.search(r'(?:^|\n)(?:Human|user|assistant)\s*:',text,re.I):
        answers=re.findall(r'(?:\A|\n\s*\n)([A-D])[.)]?(?=\s*(?:\n\s*\n|\Z))',text)
        if len(answers)==1:return options[ord(m.group(1))-65]
    return text if text in LABELS else None

def metrics(rows, predictions):
    by_id={p['id']:p for p in predictions}
    if len(by_id)!=len(predictions) or not set(by_id)<=set(r['id'] for r in rows):
        raise ValueError('Duplicate or unknown predictions')
    conf={a:{b:0 for b in LABELS+['UNPARSED','ERROR','MISSING']} for a in LABELS}
    for r in rows:
        p=by_id.get(r['id']); label='MISSING' if p is None else 'ERROR' if p.get('error') else p.get('pred_label') or 'UNPARSED'
        conf[r['label']][label]+=1
    per={}
    for l in LABELS:
        n=sum(conf[l].values()); tp=conf[l][l]; pred=sum(conf[a][l] for a in LABELS)
        pr=tp/pred if pred else 0; rec=tp/n if n else None
        per[l]={'n':n,'precision':pr,'recall':rec,'f1':2*pr*rec/(pr+rec) if n and pr+rec else 0 if n else None}
    n=len(rows); non=sum(per[l]['n'] for l in LABELS[:-1])
    return {'n':n,'returned':len(predictions),'accuracy':sum(conf[l][l] for l in LABELS)/n if n else None,
        'macro_f1':sum(v['f1'] for v in per.values() if v['n'])/sum(v['n']>0 for v in per.values()) if n else None,
        'balanced_accuracy':sum(v['recall'] for v in per.values() if v['n'])/sum(v['n']>0 for v in per.values()) if n else None,
        'over_abstention':sum(conf[l]['AMBIGUOUS'] for l in LABELS[:-1])/non if non else None,'per_class':per,'confusion':conf}
