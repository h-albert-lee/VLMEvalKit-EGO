"""Private Hub I/O; uploads only explicitly selected artifact files."""
import json, time, tarfile
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
from .core import digest, safe_path, write_json

def private_repo(repo, create=False):
    api=HfApi()
    if repo.split('/')[0]!=api.whoami()['name']:
        raise ValueError('Expected a personal, not organization, repository')
    if create:api.create_repo(repo,repo_type='dataset',private=True,exist_ok=True)
    if not api.repo_info(repo,repo_type='dataset').private:
        raise ValueError('Refusing to use a public repository')
    return api

def upload_files(repo, files, message):
    api=private_repo(repo)
    ops=[CommitOperationAdd(path_in_repo=rel,path_or_fileobj=str(path)) for rel,path in files]
    for attempt in range(5):
        try:return api.create_commit(repo,repo_type='dataset',operations=ops,commit_message=message)
        except Exception as e:
            if getattr(getattr(e,'response',None),'status_code',None) not in [409,412,429,500,502,503,504] or attempt==4:raise
            time.sleep(2**attempt)

def publish_bundle(root,repo):
    root=Path(root);api=private_repo(repo,create=True)
    if 'dataset.json' in api.list_repo_files(repo,repo_type='dataset'):
        raise ValueError('Dataset already published; use a new versioned repository')
    shards=json.loads((root/'shards.json').read_text())
    # Each closed shard is independently resumable. Dataset marker is last.
    for s in shards:
        upload_files(repo,[(s['path'],root/s['path'])],'Upload evaluation frame shard')
    names=['manifest.jsonl','subsets/common200.json','exclusions.json','shards.json','README.md','dataset.json']
    result=upload_files(repo,[(n,root/n) for n in names],'Freeze private evaluation dataset')
    if not api.repo_info(repo,repo_type='dataset').private:raise ValueError('Privacy changed during upload')
    return result.oid

def download_bundle(repo, revision, root):
    if not revision:raise ValueError('Pin a dataset commit revision')
    private_repo(repo);root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for name in ['dataset.json','manifest.jsonl','subsets/common200.json','shards.json']:
        src=Path(hf_hub_download(repo,name,repo_type='dataset',revision=revision));dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes())
    for shard in json.loads((root/'shards.json').read_text()):
        path=Path(hf_hub_download(repo,shard['path'],repo_type='dataset',revision=revision))
        if digest(path.read_bytes())!=shard['sha256']:raise ValueError('Shard checksum mismatch')
        with tarfile.open(path) as tf:
            for m in tf.getmembers():
                safe_path(root,m.name)
                if not m.isfile() or not m.name.startswith('frames/'):raise ValueError('Unsafe archive member')
            tf.extractall(root,filter='data')
    return root

def resume_run(repo, run_id, root, namespace='runs'):
    import uuid
    uuid.UUID(run_id)
    if namespace not in ['runs','preflight']:raise ValueError('Invalid run namespace')
    api=private_repo(repo);revision=api.repo_info(repo,repo_type='dataset').sha
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise ValueError('Resume destination must be empty')
    prefix=namespace+'/'+run_id+'/'
    for name in api.list_repo_files(repo,repo_type='dataset',revision=revision):
        if name.startswith(prefix) and (name.endswith('.json') or name.endswith('.jsonl')):
            p=Path(hf_hub_download(repo,name,repo_type='dataset',revision=revision))
            destination=safe_path(root,name[len(prefix):])
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(p.read_bytes())
    if not (root/'run.json').exists():raise ValueError('Run not found')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['publish','download','resume']);p.add_argument('--repo',required=True);p.add_argument('--root',required=True);p.add_argument('--revision');p.add_argument('--run-id');p.add_argument('--namespace',choices=['runs','preflight'],default='runs');a=p.parse_args()
    if a.action=='publish':print(publish_bundle(a.root,a.repo))
    elif a.action=='download':download_bundle(a.repo,a.revision,a.root)
    else:resume_run(a.repo,a.run_id,a.root,a.namespace)
