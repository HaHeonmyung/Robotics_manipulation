#!/bin/bash
# guided demos: (1) Mimic 1000 -> BC (absolute obs) -> eval, (2) same Mimic data + relative obs -> BC -> eval
cd ~/Robotics_manipulation/IsaacLab_RS
R=coursework/toy2box-bc/results
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-arena
export PYTHONUNBUFFERED=1
SKIP=${1:-}
coursework/toy2box-bc/run_baseline.sh datasets/ext/toy2box_agibot_guided_v0.hdf5 guided20_1000 1000 > $R/guided20_1000.log 2>&1
# relobs on the same generated data
python - <<PY
import h5py, numpy as np, os
src="datasets/guided20_1000_mimic_1000.hdf5"; dst="datasets/guided20_relobs_mimic_1000.hdf5"
if not os.path.exists(dst):
    with h5py.File(src,"r") as fi, h5py.File(dst,"w") as fo:
        go=fo.create_group("data")
        for k,v in fi["data"].attrs.items(): go.attrs[k]=v
        for name in fi["data"]:
            fi.copy(fi["data"][name], go, name=name); g=go[name]["obs"]; ee=g["eef_pos"][()]
            g.create_dataset("toy_truck_rel", data=(g["toy_truck_positions"][()]-ee).astype(np.float32))
            g.create_dataset("box_rel", data=(g["box_positions"][()]-ee).astype(np.float32))
    print("relobs data written")
PY
TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-RelObs-v0
mkdir -p $R/guided20_relobs
for S in 101 111; do
  ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task $TASK --algo bc --dataset datasets/guided20_relobs_mimic_1000.hdf5 --seed $S --name guided20_relobs_seed$S > $R/guided20_relobs/03_train_seed$S.log 2>&1
done
for S in 101 111; do
  CK=$(ls -d logs/robomimic/$TASK/guided20_relobs_seed$S/*/models | sort | tail -1)/model_epoch_2000.pth
  echo "seed $S checkpoint: $CK" >> $R/guided20_relobs/summary.txt
  ./isaaclab.sh -p coursework/toy2box-bc/play_diagnose.py --task $TASK --checkpoint $CK --num_rollouts 100 --horizon 800 --seed 42 --output $R/guided20_relobs/eval_seed$S.json --save_traj $R/guided20_relobs/eval_seed${S}_traj.npz --headless > $R/guided20_relobs/04_eval_seed$S.log 2>&1 &
  sleep 40
done
wait
python3 coursework/toy2box-bc/aggregate.py $R/guided20_relobs/eval_seed101.json $R/guided20_relobs/eval_seed111.json | tee -a $R/guided20_relobs/summary.txt
echo GUIDED_DONE > $R/guided_done
