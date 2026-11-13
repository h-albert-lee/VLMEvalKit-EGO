"""VLMEvalKit compatibility for the pinned bundle; no legacy row dropping."""
import pandas as pd
from .core import load_bundle, build_input

def prepare(root,condition,seed):
    meta,rows=load_bundle(root)
    data=[]
    for r in rows:
        inp=build_input(root,r,condition,seed)
        options=['PERSON_k' if x=='OTHERS' else x for x in inp['options']]
        label='PERSON_k' if r['label']=='OTHERS' else r['label']
        data.append({'index':r['id'],'clip_id':r['id'],'source_dataset':r['source'],'video_id':r['video_id'],
            'taxonomy':r.get('taxonomy'),'answer_label':label,'human_label':label,'option_order':'|'.join(options),
            **{chr(65+i):v for i,v in enumerate(options)},'answer':chr(65+options.index(label)),
            'image_path':inp['images'],'cvpr_prompt':inp['prompt'],'input_sha256':inp['input_sha256'],
            'question':'Judge the designated target object.', 'manifest_sha256':meta['manifest_sha256']})
    return pd.DataFrame(data)
