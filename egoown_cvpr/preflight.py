"""Validate pinned processors without downloading model weights."""
import argparse,json
from pathlib import Path
from .core import build_input,read_jsonl,write_json,VERSION
from .modeling import make_processor,encode_input

def main():
    import transformers
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--models',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    row=read_jsonl(Path(a.root)/'manifest.jsonl')[0];results=[]
    for m in json.loads(Path(a.models).read_text()):
        r={'model':m['id'],'revision':m['revision'],'transformers':transformers.__version__,'protocol':VERSION}
        try:
            arch='Qwen2_5OmniThinkerForConditionalGeneration' if m['model_type']=='qwen2_5_omni' else m['architectures'][0]
            getattr(transformers,arch)
            processor=make_processor(m);checks={}
            for condition in ['sparse','single','text_only']:
                batch=encode_input(processor,build_input(a.root,row,condition))
                checks[condition]={k:list(v.shape) for k,v in batch.items() if hasattr(v,'shape')}
            json.dumps(processor.to_dict())
            r.update(status='passed',processor=type(processor).__name__,tensors=checks)
        except Exception as e:r.update(status='failed',error=type(e).__name__+': '+str(e)[:1000])
        results.append(r);write_json(a.out,results);print(m['id'],r['status'],flush=True)
    if any(r['status']!='passed' for r in results):raise SystemExit(1)
if __name__=='__main__':main()
