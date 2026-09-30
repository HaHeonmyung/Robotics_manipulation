#!/usr/bin/env python3
"""Run the predeclared paired-seed experiment sequentially on one GPU. Resume completed jobs."""
import json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
protocol=json.loads((ROOT/'protocol.json').read_text())

def run(command, log, completed):
    if completed.exists():
        print('SKIP completed',completed,flush=True); return
    log.parent.mkdir(parents=True,exist_ok=True)
    print('RUN', ' '.join(map(str,command)),flush=True)
    (ROOT/'status.json').write_text(json.dumps({'command':list(map(str,command)),'log':str(log),'started_unix':time.time()},indent=2))
    with log.open('w') as f: subprocess.run(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT,check=True)
    if not completed.exists(): raise RuntimeError(f'Process exited without expected result: {completed}')

for seed in protocol['training_seeds']:
    for method in protocol['methods']:
        dest=ROOT/'runs'/f'{method}_seed{seed}'
        run([sys.executable,ROOT/'run.py','train','--condition',method,'--num_envs',protocol['training_num_envs'],
             '--iterations',protocol['training_iterations'],'--seed',seed,'--headless','--output',dest],
            dest/'console.log',dest/'completed.json')
# No held-out scores are read until all fixed-budget training jobs finish.
for seed in protocol['training_seeds']:
    for method in protocol['methods']:
        ckpt=ROOT/'runs'/f'{method}_seed{seed}'/'final.pt'
        for condition in ['nominal',*protocol['held_out']]:
            dest=ROOT/'evaluation'/f'{method}_seed{seed}'/condition
            run([sys.executable,ROOT/'run.py','evaluate','--condition',condition,
                 '--num_envs',protocol['evaluation_num_envs'],'--seed',protocol['evaluation_seed'],
                 '--checkpoint',ckpt,'--headless','--output',dest],dest/'console.log',dest/'metrics.json')
(ROOT/'status.json').write_text(json.dumps({'state':'training_and_evaluation_complete','finished_unix':time.time()},indent=2))
print('BATCH_COMPLETE',flush=True)
