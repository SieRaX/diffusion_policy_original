#!/usr/bin/env bash
# Experiment 1 — per-state MSE convergence dynamics of a flow-matching policy.
# Recipe file: run from the repo root and copy the block you need (this is NOT
# meant to be executed top-to-bottom — the 8 trainings alone would run for days).
#
# Stage order / deps:
#   1) train      -> writes checkpoints to <run_dir>/checkpoints
#   2) dense_eval -> reads <run_dir>, writes <out_dir>/dense_eval.npz
#   3) analysis   -> reads <out_dir>/dense_eval.npz, writes figures + summary.md
#
# NOTE on zsh: do NOT stuff multi-word flags into a variable and pass it unquoted
# (e.g. COMMON="--config-dir x"; ... $COMMON ...). zsh does not word-split unquoted
# variables, so it arrives as ONE mangled argument and Hydra rejects the overrides.
# Flags are written out literally below; only PY (a bare path, single word) is a
# variable, which is safe in both zsh and bash.

export HYDRA_FULL_ERROR=1
export MUJOCO_GL=osmesa
PY=/home/cspark/anaconda3/envs/adp/bin/python

# training.num_epochs=1000 is set on each train line (overrides the base 5000/3050).
# checkpoint_every=10 (exp1 config default) -> ~100 kept checkpoints per run
# (epochs 0,10,...,1000); change num_epochs on a line to trade off run length vs. cost.

# =============================================================================
# 1) TRAINING  (8 runs; differ only by --config-name, task=, hydra.run.dir=)
# =============================================================================

# --- low_dim (config-name train_flow_matching_lowdim_exp1) ---
$PY train.py \
  --config-dir diffusion_policy/experiments/spatial_attention_exp1/config \
  --config-name train_flow_matching_lowdim_exp1 \
  task=lift_lowdim_abs training.device=cuda:0 training.num_epochs=1000 \
  hydra.run.dir='outputs_HDD4/${task.name}_abs_${task.dataset_type}_reproduction/train_by_seed_flow_matching/seed${training.seed}_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_${horizon}'

$PY train.py \
  --config-dir diffusion_policy/experiments/spatial_attention_exp1/config \
  --config-name train_flow_matching_lowdim_exp1 \
  task=lift_lowdim training.device=cuda:1 training.num_epochs=1000 \
  hydra.run.dir='outputs_HDD4/${task.name}_rel_${task.dataset_type}_reproduction/train_by_seed_flow_matching/seed${training.seed}_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_${horizon}'

$PY train.py \
  --config-dir diffusion_policy/experiments/spatial_attention_exp1/config \
  --config-name train_flow_matching_lowdim_exp1 \
  task=can_lowdim_abs training.device=cuda:0 training.num_epochs=1000 \
  hydra.run.dir='outputs_HDD4/${task.name}_abs_${task.dataset_type}_reproduction/train_by_seed_flow_matching/seed${training.seed}_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_${horizon}'

$PY train.py \
  --config-dir diffusion_policy/experiments/spatial_attention_exp1/config \
  --config-name train_flow_matching_lowdim_exp1 \
  task=can_lowdim training.device=cuda:0 training.num_epochs=1000 \
  hydra.run.dir='outputs_HDD4/${task.name}_rel_${task.dataset_type}_reproduction/train_by_seed_flow_matching/seed${training.seed}_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_${horizon}'

# --- image (config-name train_flow_matching_hybrid_exp1) ---
$PY train.py \
  --config-dir diffusion_policy/experiments/spatial_attention_exp1/config \
  --config-name train_flow_matching_hybrid_exp1 \
  task=lift_image_abs training.device=cuda:0 training.num_epochs=1000 \
  hydra.run.dir='outputs_HDD4/${task.name}_abs_${task.dataset_type}_reproduction/train_by_seed_flow_matching/seed${training.seed}_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_${horizon}'

$PY train.py \
  --config-dir diffusion_policy/experiments/spatial_attention_exp1/config \
  --config-name train_flow_matching_hybrid_exp1 \
  task=lift_image training.device=cuda:0 training.num_epochs=1000 \
  hydra.run.dir='outputs_HDD4/${task.name}_rel_${task.dataset_type}_reproduction/train_by_seed_flow_matching/seed${training.seed}_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_${horizon}'

$PY train.py \
  --config-dir diffusion_policy/experiments/spatial_attention_exp1/config \
  --config-name train_flow_matching_hybrid_exp1 \
  task=can_image_abs training.device=cuda:0 training.num_epochs=1000 \
  hydra.run.dir='outputs_HDD4/${task.name}_abs_${task.dataset_type}_reproduction/train_by_seed_flow_matching/seed${training.seed}_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_${horizon}'

$PY train.py \
  --config-dir diffusion_policy/experiments/spatial_attention_exp1/config \
  --config-name train_flow_matching_hybrid_exp1 \
  task=can_image training.device=cuda:0 training.num_epochs=1000 \
  hydra.run.dir='outputs_HDD4/${task.name}_rel_${task.dataset_type}_reproduction/train_by_seed_flow_matching/seed${training.seed}_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_${horizon}'

# =============================================================================
# 2) DENSE EVAL  (per finished run; rebuilds dataset/CRN from the checkpoint cfg)
# =============================================================================

# Each cell resolves the latest timestamped train dir via a glob (ls -td ... | head -1)
# and writes dense_eval.npz + figures under <run>/episode$EP. RUN is a single-word
# path (safe in zsh). Run the RUN= line together with the command below it.
#
# EP selects the episode. EP=0 is the training/designated episode; set EP=1, 2, ...
# to dense-eval a DIFFERENT episode from the SAME checkpoints (no re-training).
# Per-episode subdirs keep episodes from overwriting each other.
EP=0

RUN=$(ls -td outputs_HDD4/lift_lowdim_abs_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
$PY -m diffusion_policy.experiments.spatial_attention_exp1.dense_eval.run_dense_eval \
  run_dir="$RUN" output_dir="$RUN/episode$EP" episode_index=$EP device=cuda:0

RUN=$(ls -td outputs_HDD4/lift_lowdim_rel_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
$PY -m diffusion_policy.experiments.spatial_attention_exp1.dense_eval.run_dense_eval \
  run_dir="$RUN" output_dir="$RUN/episode$EP" episode_index=$EP device=cuda:0

RUN=$(ls -td outputs_HDD4/can_lowdim_abs_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
$PY -m diffusion_policy.experiments.spatial_attention_exp1.dense_eval.run_dense_eval \
  run_dir="$RUN" output_dir="$RUN/episode$EP" episode_index=$EP device=cuda:0

RUN=$(ls -td outputs_HDD4/can_lowdim_rel_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
$PY -m diffusion_policy.experiments.spatial_attention_exp1.dense_eval.run_dense_eval \
  run_dir="$RUN" output_dir="$RUN/episode$EP" episode_index=$EP device=cuda:0

RUN=$(ls -td outputs_HDD4/lift_image_abs_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
$PY -m diffusion_policy.experiments.spatial_attention_exp1.dense_eval.run_dense_eval \
  run_dir="$RUN" output_dir="$RUN/episode$EP" episode_index=$EP device=cuda:1

RUN=$(ls -td outputs_HDD4/lift_image_rel_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
$PY -m diffusion_policy.experiments.spatial_attention_exp1.dense_eval.run_dense_eval \
  run_dir="$RUN" output_dir="$RUN/episode$EP" episode_index=$EP device=cuda:0

RUN=$(ls -td outputs_HDD4/can_image_abs_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
$PY -m diffusion_policy.experiments.spatial_attention_exp1.dense_eval.run_dense_eval \
  run_dir="$RUN" output_dir="$RUN/episode$EP" episode_index=$EP device=cuda:0

RUN=$(ls -td outputs_HDD4/can_image_rel_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
$PY -m diffusion_policy.experiments.spatial_attention_exp1.dense_eval.run_dense_eval \
  run_dir="$RUN" output_dir="$RUN/episode$EP" episode_index=$EP device=cuda:0

# =============================================================================
# 3) ANALYSIS
#    low_dim runs: per-run figures + summary.
#    image runs: pass npz_compare=<matching low_dim npz> to ALSO emit the
#    cross-variant (low_dim vs image) episode-timeline overlay for that task+abs.
#    (Never compare across abs_action — different action space.)
# =============================================================================

# Reads each run's episode$EP npz. Set EP to match the dense-eval above (self-
# contained here too so this block can be copied on its own).
EP=0

# low_dim (per-run) — resolve each run dir and read its episode$EP dense_eval.npz
RUN=$(ls -td outputs_HDD4/lift_lowdim_abs_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
$PY -m diffusion_policy.experiments.spatial_attention_exp1.analysis.run_analysis \
  npz="$RUN/episode$EP/dense_eval.npz" output_dir="$RUN/episode$EP"

RUN=$(ls -td outputs_HDD4/lift_lowdim_rel_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
$PY -m diffusion_policy.experiments.spatial_attention_exp1.analysis.run_analysis \
  npz="$RUN/episode$EP/dense_eval.npz" output_dir="$RUN/episode$EP"

RUN=$(ls -td outputs_HDD4/can_lowdim_abs_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
$PY -m diffusion_policy.experiments.spatial_attention_exp1.analysis.run_analysis \
  npz="$RUN/episode$EP/dense_eval.npz" output_dir="$RUN/episode$EP"

RUN=$(ls -td outputs_HDD4/can_lowdim_rel_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
$PY -m diffusion_policy.experiments.spatial_attention_exp1.analysis.run_analysis \
  npz="$RUN/episode$EP/dense_eval.npz" output_dir="$RUN/episode$EP"

# image (per-run + cross-variant overlay vs the matching low_dim run).
# IMG = image run dir, LOW = matching low_dim run dir (same task+abs).
IMG=$(ls -td outputs_HDD4/lift_image_abs_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
LOW=$(ls -td outputs_HDD4/lift_lowdim_abs_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
$PY -m diffusion_policy.experiments.spatial_attention_exp1.analysis.run_analysis \
  npz="$IMG/episode$EP/dense_eval.npz" output_dir="$IMG/episode$EP" npz_compare="$LOW/episode$EP/dense_eval.npz"

IMG=$(ls -td outputs_HDD4/lift_image_rel_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
LOW=$(ls -td outputs_HDD4/lift_lowdim_rel_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
$PY -m diffusion_policy.experiments.spatial_attention_exp1.analysis.run_analysis \
  npz="$IMG/episode$EP/dense_eval.npz" output_dir="$IMG/episode$EP" npz_compare="$LOW/episode$EP/dense_eval.npz"

IMG=$(ls -td outputs_HDD4/can_image_abs_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
LOW=$(ls -td outputs_HDD4/can_lowdim_abs_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
$PY -m diffusion_policy.experiments.spatial_attention_exp1.analysis.run_analysis \
  npz="$IMG/episode$EP/dense_eval.npz" output_dir="$IMG/episode$EP" npz_compare="$LOW/episode$EP/dense_eval.npz"

IMG=$(ls -td outputs_HDD4/can_image_rel_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
LOW=$(ls -td outputs_HDD4/can_lowdim_rel_ph_reproduction/train_by_seed_flow_matching/seed* | head -1)
$PY -m diffusion_policy.experiments.spatial_attention_exp1.analysis.run_analysis \
  npz="$IMG/episode$EP/dense_eval.npz" output_dir="$IMG/episode$EP" npz_compare="$LOW/episode$EP/dense_eval.npz"
