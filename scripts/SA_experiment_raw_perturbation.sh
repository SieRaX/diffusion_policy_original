#!/usr/bin/env bash
# Preliminary perturbation-sensitivity experiment on already-trained flow-matching
# checkpoints. Recipe file: run from the repo root and copy the block you need.
# Task / obs variant / dims are DERIVED FROM THE CHECKPOINT (no task= override).
#
# HYDRA + '=' in the checkpoint name: checkpoints are named epoch=NNNN.ckpt, so the
# value contains '=' which Hydra's override parser rejects (mismatched input '=').
# We wrap the checkpoint value in SINGLE quotes -> checkpoint="'$CKPT'" so Hydra
# treats it as a literal string. Only `checkpoint` needs this; other overrides don't.

export HYDRA_FULL_ERROR=1
export MUJOCO_GL=osmesa
PY=/home/cspark/anaconda3/envs/adp/bin/python

# 0) unit tests (verify implementation)
$PY -m pytest diffusion_policy/experiments/spatial_attention_prelim_perturb/tests/ -q

# 1) low_dim example  (stride 1)
demo_index=0  # which demo to perturb (0-9 for the 10 demos in the reproduction dataset)
LOWDIM_CKPT=$(ls -td outputs_HDD4/tool_hang_lowdim_abs_ph_reproduction/train_by_seed_flow_matching/seed42_2026.07.16-12.50.49_train_flow_matching_unet_lowdim_tool_hang_lowdim_cnn_16/checkpoints/epoch=*.ckpt | head -1)
# runner writes to <ckpt-without-ext>/prelim_perturb/episode_<demo> by default
LOWDIM_OUT="${LOWDIM_CKPT%.ckpt}/prelim_perturb/episode_$demo_index"
$PY -m diffusion_policy.experiments.spatial_attention_prelim_perturb.run_perturb \
  checkpoint="'$LOWDIM_CKPT'" output_dir="'$LOWDIM_OUT'" demo_index=$demo_index device=cuda:0
# single-quote BOTH values: the output path contains epoch=NNNN, so '=' is present
$PY -m diffusion_policy.experiments.spatial_attention_prelim_perturb.analysis.run_analysis \
  npz="'$LOWDIM_OUT/perturb.npz'" output_dir="'$LOWDIM_OUT'"

# 2) image example  (stride 5 — episodes are long and rendering is costly)
demo_index=0  # which demo to perturb (0-9 for the 10 demos in the reproduction dataset)
IMAGE_CKPT=$(ls -td outputs_HDD4/lift_image_abs_ph_reproduction/train_by_seed_flow_matching/seed42_2026.07.16-02.40.58_train_flow_matching_unet_hybrid_lift_image_cnn_16/checkpoints/epoch=*.ckpt | head -1)
IMAGE_OUT="${IMAGE_CKPT%.ckpt}/prelim_perturb/episode_0"
$PY -m diffusion_policy.experiments.spatial_attention_prelim_perturb.run_perturb \
  checkpoint="'$IMAGE_CKPT'" output_dir="'$IMAGE_OUT'" demo_index=$demo_index stride=5 device=cuda:0
$PY -m diffusion_policy.experiments.spatial_attention_prelim_perturb.analysis.run_analysis \
  npz="'$IMAGE_OUT/perturb.npz'" output_dir="'$IMAGE_OUT'"

# Other cells: swap the reproduction dir in the glob (e.g. can_lowdim_rel_ph_reproduction),
# the *_OUT name, and stride. Tune measurement params by appending e.g.
#   K=8 N=16 stride=1 sigma_pos.object=0.005   to a run_perturb line.
