"""One immutable model/condition/subset run, resumable on another GPU host."""
import argparse, importlib.metadata, json, os, platform, time, uuid
from pathlib import Path
from .core import VERSION,CONDITIONS,load_bundle,build_input,parse_answer,metrics,read_jsonl,write_json,digest
from .hub import upload_files, private_repo
from .modeling import make_processor,encode_input

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--models',required=True);p.add_argument('--model',required=True)
    p.add_argument('--condition',choices=CONDITIONS,default='sparse');p.add_argument('--subset',choices=['full','common200','smoke16'],default='full');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--out',required=True);p.add_argument('--results-repo');p.add_argument('--max-new-tokens',type=int,default=256);p.add_argument('--dry-run',action='store_true');p.add_argument('--checkpoint-every',type=int,default=25)
    p.add_argument('--max-consecutive-errors',type=int,default=3)
    a=p.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    meta,rows=load_bundle(root)
    if a.subset in ['common200','smoke16']:
        ordered_ids=json.loads((root/'subsets/common200.json').read_text())
        if a.subset=='smoke16':
            labels={r['id']:r['label'] for r in rows};counts={};chosen=[]
            for rid in ordered_ids:
                label=labels[rid]
                if counts.get(label,0)<4:chosen.append(rid);counts[label]=counts.get(label,0)+1
            if len(chosen)!=16:raise ValueError('Smoke panel needs four rows per class')
            ordered_ids=chosen
        ids=set(ordered_ids);rows=[r for r in rows if r['id'] in ids]
        if len(rows)!=len(ids):raise ValueError('Subset mismatch')
    modelinfo=next(m for m in json.loads(Path(a.models).read_text()) if m['id']==a.model)
    if not modelinfo.get('revision') or not modelinfo.get('config_access'):raise ValueError('Model preflight not passed')
    config={'model':modelinfo,'dataset_sha256':meta['manifest_sha256'],'version':VERSION,'condition':a.condition,'subset':a.subset,'ids_sha256':digest('\n'.join(r['id'] for r in rows).encode()),'seed':a.seed,'max_new_tokens':a.max_new_tokens,'thinking':False,'do_sample':False,'n':len(rows),'dry_run':a.dry_run}
    config['code_sha256']=digest(b''.join(p.name.encode()+p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py'))))
    signature=digest(json.dumps(config,sort_keys=True).encode())
    runfile=out/'run.json'
    if runfile.exists():
        run=json.loads(runfile.read_text())
        if run['signature']!=signature:raise ValueError('Output directory belongs to a different run')
    else:
        run={'run_id':str(uuid.uuid4()),'signature':signature,'config':config,'status':'prepared'};write_json(runfile,run)
    prefix=('preflight/' if a.dry_run else 'runs/')+run['run_id']
    if a.results_repo:private_repo(a.results_repo)
    previous=read_jsonl(out/'predictions.jsonl') if (out/'predictions.jsonl').exists() else []
    if len({r['id'] for r in previous})!=len(previous) or not {r['id'] for r in previous}<=set(r['id'] for r in rows):raise ValueError('Invalid resume IDs')
    done={r['id'] for r in previous}; pending=[];chunk=len(list(out.glob('chunk-*.jsonl')))
    def sync(final=False):
        nonlocal pending,chunk
        files=[]
        if pending:
            f=out/f'chunk-{chunk:06d}.jsonl';f.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pending));pending=[];chunk+=1;files.append((prefix+'/'+f.name,f))
        write_json(out/'metrics.json',metrics(rows,previous));write_json(runfile,run)
        files.extend([(prefix+'/metrics.json',out/'metrics.json'),(prefix+'/run.json',runfile)])
        if (out/'predictions.jsonl').exists():
            files.append((prefix+'/predictions.jsonl',out/'predictions.jsonl'))
        if a.results_repo:upload_files(a.results_repo,files,'Checkpoint '+run['run_id'])
    if not a.dry_run:
        run['status']='loading_model';sync()
        try:
            import torch,transformers
            if not torch.cuda.is_available():raise RuntimeError('Real model inference requires a CUDA GPU; use --dry-run for local input validation')
            arch=modelinfo['architectures'][0]
            if modelinfo['model_type']=='qwen2_5_omni':
                arch='Qwen2_5OmniThinkerForConditionalGeneration'
            cls=getattr(transformers,arch,None)
            if cls is None:raise RuntimeError('Installed transformers lacks '+arch)
            processor=make_processor(modelinfo)
            model=cls.from_pretrained(a.model,revision=modelinfo['revision'],torch_dtype=torch.bfloat16,device_map='auto').eval()
            generation_kwargs={'max_new_tokens':a.max_new_tokens,'do_sample':False}
            if modelinfo['model_type']=='qwen2_5_omni':
                # The pinned Omni generation_config.json omits stop tokens.
                # Restore the Thinker's own tokens instead of generating past
                # <|im_end|> into a fictitious next conversation turn.
                for name in ['eos_token_id','pad_token_id']:
                    value=getattr(model.config,name,None)
                    if value is None:raise ValueError('Omni Thinker config lacks '+name)
                    generation_kwargs[name]=value
            run['generation_kwargs']=generation_kwargs
            run['loaded_generation_config']=model.generation_config.to_dict()
            if any(str(v) in ['cpu','disk'] for v in getattr(model,'hf_device_map',{}).values()):
                raise RuntimeError('Insufficient GPU memory: refusing implicit CPU/disk offload')
            run['environment']={'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'packages':{p:importlib.metadata.version(p) for p in ['huggingface_hub','Pillow','numpy']}}
            run['processor']=processor.to_dict() if hasattr(processor,'to_dict') else {'class':type(processor).__name__}
        except BaseException as e:
            run.update(status='model_load_failed',error=type(e).__name__+': '+str(e)[:1000]);sync();raise
    run['status']='running';sync();consecutive_errors=0
    try:
        for row in rows:
            if row['id'] in done:continue
            inp=build_input(root,row,a.condition,a.seed)
            record={'id':row['id'],'input_sha256':inp['input_sha256'],'options':inp['options'],'pred_label':None,'raw_response':None,'error':None}
            begin=time.time()
            if a.dry_run:
                record.update({'error':'DRY_RUN_NO_INFERENCE','image_count':len(inp['images']),'prompt':inp['prompt'],'image_paths':inp['images']})
            else:
                try:
                    batch=encode_input(processor,inp)
                    record['input_tokens']=int(batch['input_ids'].shape[-1])
                    batch=batch.to(model.device)
                    for device in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(device)
                    with torch.inference_mode():output=model.generate(**batch,**generation_kwargs)
                    generated=output[0][batch['input_ids'].shape[-1]:]
                    record['generated_token_ids']=generated.tolist()
                    text=processor.decode(generated,skip_special_tokens=True)
                    record['raw_response']=text;record['pred_label']=parse_answer(text,inp['options'])
                    record['peak_gpu_bytes']=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
                except Exception as e:
                    record['error']=type(e).__name__+': '+str(e)[:1000]
                    if torch.cuda.is_available():torch.cuda.empty_cache()
            record['seconds']=round(time.time()-begin,4)
            with (out/'predictions.jsonl').open('a') as f:
                f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            previous.append(record);pending.append(record);done.add(row['id'])
            if len(pending)>=a.checkpoint_every:sync()
            if not a.dry_run:
                consecutive_errors=consecutive_errors+1 if record['error'] else 0
                if consecutive_errors>=a.max_consecutive_errors:
                    raise RuntimeError('Stopping after consecutive inference errors; inspect saved predictions before resuming')
        run['status']='dry_run_complete' if a.dry_run else 'complete_with_errors' if any(r['error'] for r in previous) else 'complete'
    except BaseException:
        run['status']='interrupted';sync();raise
    sync(True)
    print(json.dumps({'run_id':run['run_id'],'status':run['status'],'processed':len(done),'n':len(rows)}))

if __name__=='__main__':main()
