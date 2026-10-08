"""Wait for local weights, finish existing jobs, then measure the larger model on fixed cases."""
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import requests

model=Path('../models/Qwen3-VL-32B-Instruct-NF4').resolve()
required=['config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json']
print('Waiting for 32B NF4 weights',flush=True)
for attempt in range(360):
    try:
        index=json.loads((model/'model.safetensors.index.json').read_text())
        complete=all((model/name).exists() for name in [*required,*set(index['weight_map'].values())])
    except (FileNotFoundError,KeyError,json.JSONDecodeError):complete=False
    if complete:break
    time.sleep(10)
else:raise RuntimeError('Download did not finish within 60 minutes')

for attempt in range(180):
    with sqlite3.connect('data/cases.sqlite3') as db:
        active=db.execute("SELECT count(*) FROM runs WHERE status IN ('running','queued')").fetchone()[0]
    if not active:break
    time.sleep(2)
else:raise RuntimeError('Active jobs remain; current server has been left unchanged')

old_pid=int(sys.argv[1])
try:os.kill(old_pid,signal.SIGINT)
except ProcessLookupError:pass
for attempt in range(100):
    try:requests.get('http://127.0.0.1:7860/api/health',timeout=1)
    except requests.RequestException:break
    time.sleep(.2)

task_env=dict(os.environ)
task_env['CHEST_MODEL_PATH']=str(model)
task_env['CHEST_CXR_EXPERT']='1'
task_env['CHEST_MAX_INPUT_TOKENS']='8192'
log=open('data/large-model-service.log','w')
process=subprocess.Popen(['.venv/bin/python','-m','uvicorn','chest_agent.app:app','--host','127.0.0.1','--port','7860'],
    env=task_env,stdout=log,stderr=log,start_new_session=True)
Path('data/large-model-service.pid').write_text(str(process.pid))
for attempt in range(100):
    try:
        health=requests.get('http://127.0.0.1:7860/api/health',timeout=1).json()
        if health.get('model_name')==model.name:break
    except requests.RequestException:pass
    time.sleep(.2)
else:raise RuntimeError('Large model service did not start; inspect data/large-model-service.log')
print('32B service ready for inference:',process.pid,health,flush=True)
subprocess.run(['.venv/bin/python','-u','scripts/quality_compare.py','qwen32_nf4'],check=True)
subprocess.run(['.venv/bin/python','-u','scripts/quality_compare.py','heldout_qwen32_nf4',
    '--cases','data/quality/heldout.json'],check=True)
print('Large-model comparisons finished',flush=True)
