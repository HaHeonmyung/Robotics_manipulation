#!/usr/bin/env python3
"""Fixed-budget Ant PPO training and exactly one episode per evaluation environment."""
import argparse
import json
import time
import sys
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('mode',choices=['train','evaluate'])
p.add_argument('--condition',default='baseline',choices=['baseline','randomized','nominal','slippery','heavy','strong_kicks','combined'])
p.add_argument('--num_envs',type=int,default=512)
p.add_argument('--iterations',type=int,default=500)
p.add_argument('--seed',type=int,default=11)
p.add_argument('--checkpoint',type=Path)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--video',action='store_true')
p.add_argument('--video_frames',type=int,default=600)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
if args.mode=='evaluate' and args.checkpoint is None: p.error('--checkpoint required')
if args.video: args.enable_cameras=True
args.output=args.output.resolve()
args.output.mkdir(parents=True,exist_ok=True)
if args.mode=='train' and (args.output/'final.pt').exists(): p.error('Refusing to overwrite completed training')
app=AppLauncher(args).app

import gymnasium as gym
import torch
import numpy as np
import isaaclab_tasks
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg
from environment import make_config, parameters

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True

def write_json(path, data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')

def main():
    start=time.monotonic()
    cfg=make_config(args.condition,args.num_envs,args.seed,args.device or 'cuda:0')
    cfg.log_dir=str(args.output)
    agent=AntPPORunnerCfg()
    agent.seed=args.seed
    agent.device=cfg.sim.device
    agent.max_iterations=args.iterations
    agent.save_interval=100
    # Only the environment dynamics differ between baseline and randomized.
    dump_yaml(str(args.output/'env.yaml'),cfg)
    dump_yaml(str(args.output/'agent.yaml'),agent)
    write_json(args.output/'arguments.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    env=gym.make('Isaac-Ant-v0',cfg=cfg,render_mode='rgb_array' if args.video else None)
    base=env.unwrapped
    robot=base.scene['robot']
    masses=robot.root_physx_view.get_masses().cpu().numpy()
    materials=robot.root_physx_view.get_material_properties().cpu().numpy()
    np.savez_compressed(args.output/'physics.npz',masses=masses,materials=materials)
    write_json(args.output/'physics-summary.json',{
        'condition':parameters(args.condition),'mass_min':float(masses.min()),'mass_max':float(masses.max()),
        'static_friction_min':float(materials[...,0].min()),'static_friction_max':float(materials[...,0].max()),
        'dynamic_friction_min':float(materials[...,1].min()),'dynamic_friction_max':float(materials[...,1].max()),
    })
    wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
    runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=str(args.output) if args.mode=='train' else None,device=agent.device)
    if args.mode=='train':
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=True)
        runner.save(str(args.output/'final.pt'))
        if runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        write_json(args.output/'completed.json',{
            'condition':args.condition,'seed':args.seed,'iterations':args.iterations,'num_envs':args.num_envs,
            'steps_per_env':agent.num_steps_per_env,
            'transitions':args.iterations*args.num_envs*agent.num_steps_per_env,
            'wall_seconds':time.monotonic()-start,'checkpoint':'final.pt'})
        print('TRAINING_COMPLETE',flush=True)
    else:
        runner.load(str(args.checkpoint.resolve()),load_optimizer=False)
        policy=runner.get_inference_policy(device=base.device)
        # Runner/wrapper constructors consume reset RNG. Start all compared episodes from an explicit common seed.
        base.seed(args.seed)
        obs,_=wrapped.reset()
        active=torch.ones(args.num_envs,dtype=torch.bool,device=base.device)
        totals=torch.zeros(args.num_envs,dtype=torch.float64,device=base.device)
        lengths=torch.zeros(args.num_envs,dtype=torch.int64,device=base.device)
        survived=torch.zeros_like(active)
        forward=torch.zeros(args.num_envs,device=base.device)
        video=None
        if args.video:
            import imageio.v2 as imageio
            video=imageio.get_writer(str(args.output/'rollout.mp4'),fps=round(1/base.step_dt),macro_block_size=16)
        try:
            for step in range(base.max_episode_length+1):
                # Save a representative env-0 first episode only; do not splice later auto-reset episodes.
                if video is not None and step<args.video_frames and active[0].item():
                    video.append_data(base.render())
                with torch.inference_mode():
                    # Integrate pre-step forward velocity to avoid terminal auto-reset position contamination.
                    forward[active] += robot.data.root_lin_vel_w[active,0]*base.step_dt
                    obs,rewards,dones,extras=wrapped.step(policy(obs))
                    totals[active]+=rewards[active]
                    lengths[active]+=1
                    ended=active & dones.bool()
                    survived[ended]=base.reset_time_outs[ended] & ~base.reset_terminated[ended]
                    active &= ~dones.bool()
                if not active.any().item(): break
            if active.any().item(): raise RuntimeError('Some first episodes did not complete; partial statistics rejected')
        finally:
            if video is not None: video.close()
        values=totals.cpu().numpy(); steps=lengths.cpu().numpy(); alive=survived.cpu().numpy()
        import csv
        with (args.output/'episodes.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['env_id','return','steps','survived_full_episode','integrated_forward_m'])
            for i in range(args.num_envs): w.writerow([i,float(values[i]),int(steps[i]),bool(alive[i]),float(forward[i])])
        result={'condition':args.condition,'num_envs':args.num_envs,'evaluation_seed':args.seed,
            'checkpoint':str(args.checkpoint.resolve()),'return_mean':float(values.mean()),
            'return_std_population':float(values.std(ddof=0)),
            'episode_steps_mean':float(steps.mean()),'survival_rate':float(alive.mean()),
            'forward_m_mean':float(forward.mean()),'wall_seconds':time.monotonic()-start}
        write_json(args.output/'metrics.json',result)
        print('EVALUATION_COMPLETE',json.dumps(result),flush=True)
    # Close env before the documented fast app shutdown. All outputs are already flushed.
    wrapped.close()

try:
    main()
finally:
    app.close(skip_cleanup=True)
