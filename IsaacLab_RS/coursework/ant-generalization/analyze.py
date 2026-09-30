#!/usr/bin/env python3
"""Aggregate all predeclared runs; keep episode variability distinct from training-seed variability."""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
A=ROOT/'analysis'; A.mkdir(exist_ok=True)
conditions=['nominal',*P['held_out']]
rows=[]
for seed in P['training_seeds']:
 for method in P['methods']:
  for condition in conditions:
   f=ROOT/'evaluation'/f'{method}_seed{seed}'/condition/'metrics.json'
   r=json.loads(f.read_text());r.update(method=method,training_seed=seed)
   rows.append(r)
with (A/'per_seed_results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for condition in conditions:
 for method in P['methods']:
  rr=[r for r in rows if r['condition']==condition and r['method']==method]
  means=np.array([r['return_mean'] for r in rr]); surv=[r['survival_rate'] for r in rr]
  summary.append(dict(condition=condition,method=method,mean_return=float(means.mean()),
   sd_across_training_seeds=float(means.std(ddof=1)),mean_episode_sd=float(np.mean([r['return_std_population'] for r in rr])),
   mean_survival_rate=float(np.mean(surv)),mean_forward_m=float(np.mean([r['forward_m_mean'] for r in rr])),
   mean_episode_steps=float(np.mean([r['episode_steps_mean'] for r in rr]))))
(A/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (A/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
x=np.arange(len(conditions))
for i,method in enumerate(P['methods']):
 s=[r for r in summary if r['method']==method]
 ax.bar(x+(i-.5)*.36,[r['mean_return'] for r in s],.36,yerr=[r['sd_across_training_seeds'] for r in s],
        label=method,capsize=4,color=['#51677A','#009D93'][i])
ax.set_xticks(x,conditions);ax.set_ylabel('First-episode return');ax.legend(frameon=False)
ax.set_title('100 first episodes per condition and training seed; error bars = SD across 3 training seeds')
ax.spines[['top','right']].set_visible(False)
fig.savefig(A/'held_out_returns.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
curves={}
for i,method in enumerate(P['methods']):
 curves[method]=[]
 for seed in P['training_seeds']:
  ea=EventAccumulator(str(ROOT/'runs'/f'{method}_seed{seed}'));ea.Reload()
  tags=ea.Tags()['scalars']
  tag='Train/mean_reward'
  if tag not in tags: raise RuntimeError(f'Missing training return tag; available {tags}')
  ev=ea.Scalars(tag)
  points=[{'iteration':e.step,'transitions':(e.step+1)*P['training_num_envs']*P['rollout_steps_per_env'],'return':e.value} for e in ev]
  curves[method].append({'seed':seed,'points':points})
  ax.plot([p['transitions']/1e6 for p in points],[p['return'] for p in points],alpha=.65,
          color=['#51677A','#009D93'][i],label=method if seed==P['training_seeds'][0] else None)
ax.set_xlabel('Environment transitions (millions)');ax.set_ylabel('Training mean episode return')
ax.set_title('Each line is one training seed; training distributions differ')
ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
fig.savefig(A/'learning_curves.png',dpi=180);plt.close(fig)
(A/'learning_curves.json').write_text(json.dumps(curves)+'\n')
heldout=[]
for method in P['methods']:
 s=[r for r in summary if r['method']==method and r['condition']!='nominal']
 heldout.append({'method':method,'unweighted_held_out_mean_return':float(np.mean([r['mean_return'] for r in s])),
                'worst_condition_mean_return':float(min(r['mean_return'] for r in s))})
(A/'held_out_aggregate.json').write_text(json.dumps(heldout,indent=2)+'\n')
print(json.dumps(summary,indent=2))
