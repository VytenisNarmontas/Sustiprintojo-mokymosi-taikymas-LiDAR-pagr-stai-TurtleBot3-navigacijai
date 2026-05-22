#!/usr/bin/env bash
set -euo pipefail

cd ~/rl_gate_train/rl_gate_train_lidar_feature_v2
source .venv/bin/activate

N_ENVS="${N_ENVS:-8}"
DEVICE="${DEVICE:-cpu}"

echo "============================================================"
echo "LiDAR RESIDUAL PPO CURRICULUM"
echo "PWD=$(pwd)"
echo "N_ENVS=$N_ENVS DEVICE=$DEVICE"
echo "============================================================"

python check_lidar_residual_obs.py

run_stage () {
  NAME="$1"
  TIMESTEPS="$2"
  SAVE_DIR="$3"
  LOAD_DIR="$4"
  EXTRA_ARGS="$5"
  LR="$6"

  echo
  echo "============================================================"
  echo "TRAIN $NAME"
  echo "TIMESTEPS=$TIMESTEPS"
  echo "SAVE_DIR=$SAVE_DIR"
  echo "LOAD_DIR=$LOAD_DIR"
  echo "LR=$LR"
  echo "EXTRA_ARGS=$EXTRA_ARGS"
  echo "============================================================"

  rm -rf "$SAVE_DIR"

  if [ -z "$LOAD_DIR" ]; then
    python train_lidar_residual.py \
      --total-timesteps "$TIMESTEPS" \
      --n-envs "$N_ENVS" \
      --save-dir "$SAVE_DIR" \
      --learning-rate "$LR" \
      --device "$DEVICE" \
      $EXTRA_ARGS
  else
    python train_lidar_residual.py \
      --total-timesteps "$TIMESTEPS" \
      --n-envs "$N_ENVS" \
      --load-dir "$LOAD_DIR" \
      --save-dir "$SAVE_DIR" \
      --learning-rate "$LR" \
      --device "$DEVICE" \
      $EXTRA_ARGS
  fi
}

run_stage "level 1 residual"       80000  models_lidar_residual_l1     ""                         "--level 1 --no-lidar-noise"                       3e-4
run_stage "level 2 residual"       100000 models_lidar_residual_l2     models_lidar_residual_l1     "--level 2 --no-lidar-noise"                       2e-4
run_stage "level 3 residual"       140000 models_lidar_residual_l3     models_lidar_residual_l2     "--level 3 --no-lidar-noise"                       1.5e-4
run_stage "level 4 residual"       180000 models_lidar_residual_l4     models_lidar_residual_l3     "--level 4 --no-lidar-noise"                       1e-4
run_stage "mixed residual"         200000 models_lidar_residual_mixed  models_lidar_residual_l4     "--mixed-levels 3,4 --no-lidar-noise"              8e-5
run_stage "close-start residual"   180000 models_lidar_residual_close  models_lidar_residual_mixed  "--close-start --no-lidar-noise"                   6e-5

echo
echo "============================================================"
echo "NOISY FINE-TUNE"
echo "============================================================"

rm -rf models_lidar_residual_final

python train_lidar_residual.py \
  --close-start \
  --total-timesteps 120000 \
  --n-envs "$N_ENVS" \
  --load-dir models_lidar_residual_close \
  --save-dir models_lidar_residual_final \
  --learning-rate 3e-5 \
  --device "$DEVICE"

echo
echo "============================================================"
echo "FINAL TESTS"
echo "============================================================"

python test_lidar_residual.py \
  --model-dir models_lidar_residual_final \
  --level 4 \
  --episodes 100 \
  --no-lidar-noise \
  --device "$DEVICE"

python test_lidar_residual.py \
  --model-dir models_lidar_residual_final \
  --close-start \
  --episodes 100 \
  --no-lidar-noise \
  --device "$DEVICE"

echo
echo "DONE. Final residual PPO model:"
echo "models_lidar_residual_final"
