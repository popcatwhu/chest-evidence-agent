"""Resume independent HTTP ranges and verify each weight's official LFS SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import requests
from huggingface_hub import HfApi,hf_hub_download
sys.path.insert(0,str(Path(__file__).resolve().parent))
from download_medical_models import MODELS,ROOT


def sha256(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(8*1024*1024):result.update(chunk)
    return result.hexdigest()


def download(size,workers=16):
    repo,revision=MODELS[size]
    destination=ROOT.parent/'models'/repo.split('/')[-1]
    destination.mkdir(exist_ok=True,parents=True)
    info=HfApi().model_info(repo,revision=revision,files_metadata=True)
    if info.gated:raise ValueError('Access agreement required')
    for item in info.siblings:
        if item.rfilename.endswith(('.json','.txt','.model','.jinja')) or item.rfilename in {'README.md','LICENSE'}:
            hf_hub_download(repo,item.rfilename,revision=revision,local_dir=str(destination))
    chunk_size=16*1024*1024
    states={};pending=[];local=threading.local()
    for item in info.siblings:
        if not item.rfilename.endswith('.safetensors'):continue
        path=destination/item.rfilename
        expected=item.lfs.sha256;length=item.size
        if path.exists() and path.stat().st_size==length and sha256(path)==expected:
            print('Verified existing',path.name,flush=True);continue
        part=path.with_suffix('.range.part');progress=path.with_suffix('.range.json')
        header={'repository':repo,'revision':revision,'sha256':expected,'size':length,'chunk_size':chunk_size}
        done=set()
        if progress.exists():
            saved=json.loads(progress.read_text())
            if saved['header']!=header:raise ValueError('Partial download metadata differs')
            done=set(saved['done'])
            if not part.exists():raise ValueError('Checkpoint exists but partial bytes are missing')
        fd=os.open(part,os.O_CREAT|os.O_RDWR,0o644);os.ftruncate(fd,length)
        states[item.rfilename]={'path':path,'part':part,'progress':progress,'header':header,'done':done,'fd':fd}
        for start in range(0,length,chunk_size):
            if start not in done:pending.append((item.rfilename,start,min(start+chunk_size,length)-1))
    def fetch(job):
        name,start,end=job
        if not hasattr(local,'session'):local.session=requests.Session()
        for attempt in range(6):
            try:
                # Distinct URLs avoid intermediaries serving a different cached range.
                url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true&range_chunk={start}&attempt={attempt}'
                with local.session.get(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','Cache-Control':'no-cache'},stream=True,timeout=(20,60)) as response:
                    expected_range=f'bytes {start}-{end}/{states[name]["header"]["size"]}'
                    if response.status_code!=206 or response.headers.get('Content-Range')!=expected_range:
                        raise RuntimeError(f'Unexpected HTTP range response: status {response.status_code}')
                    offset=start
                    for block in response.iter_content(1024*1024):
                        if offset+len(block)>end+1:raise RuntimeError('Range exceeds requested length')
                        view=memoryview(block)
                        while view:
                            written=os.pwrite(states[name]['fd'],view,offset)
                            if written<=0:raise RuntimeError('Partial file write failed')
                            offset+=written;view=view[written:]
                    if offset!=end+1:raise RuntimeError('Incomplete range')
                return job
            except (requests.RequestException,RuntimeError):
                if attempt==5:raise RuntimeError(f'Range download failed: {name}, offset {start}') from None
                time.sleep(min(2**attempt,8))
    started=time.monotonic();transferred=0;last_print=started
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(fetch,job):job for job in pending}
            for future in as_completed(futures):
                name,start,end=future.result();state=states[name]
                state['done'].add(start);transferred+=end-start+1
                temp=state['progress'].with_suffix('.tmp')
                temp.write_text(json.dumps({'header':state['header'],'done':sorted(state['done'])}));temp.replace(state['progress'])
                now=time.monotonic()
                if now-last_print>=20:
                    print(size,'range MiB',round(transferred/2**20),'MiB/s',round(transferred/2**20/(now-started),2),flush=True)
                    last_print=now
        for state in states.values():
            os.fsync(state['fd']);os.close(state['fd']);state['fd']=None
            if sha256(state['part'])!=state['header']['sha256']:
                reused=state['path'].with_suffix('.reused.json')
                if reused.exists():
                    offsets=set(json.loads(reused.read_text())['offsets'])
                    state['done'].difference_update(offsets)
                    state['progress'].write_text(json.dumps({'header':state['header'],'done':sorted(state['done'])}))
                    reused.unlink()
                raise RuntimeError('Official SHA256 mismatch; resumable checkpoint reset for reused bytes: '+state['path'].name)
            state['part'].replace(state['path']);state['progress'].unlink(missing_ok=True)
            print('SHA256 verified',state['path'].name,flush=True)
        (destination/'source_revision.json').write_text(json.dumps({'repository':repo,'revision':revision},indent=2))
        print('Completed',destination,flush=True)
    finally:
        for state in states.values():
            if state['fd'] is not None:os.close(state['fd'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('size',choices=MODELS)
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args();download(args.size,args.workers)
