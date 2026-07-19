"""Hydra entry point for the dense-eval stage (ONE entry point).

Usage (from repo root):
    MUJOCO_GL=osmesa python -m \
      diffusion_policy.experiments.spatial_attention_exp1.dense_eval.run_dense_eval \
      run_dir=outputs/exp1_lift_lowdim_abs/train \
      output_dir=outputs/exp1_lift_lowdim_abs device=cuda:0
"""
import pathlib

import hydra
from omegaconf import OmegaConf

# robomimic/mujoco envs are constructed nowhere here, but the dataset import path
# pulls robosuite; keep spawn for parity with the other entry points.
import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

OmegaConf.register_new_resolver("eval", eval, replace=True)

from diffusion_policy.experiments.spatial_attention_exp1.dense_eval import dense_eval


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('config')),
    config_name='dense_eval')
def main(cfg):
    OmegaConf.resolve(cfg)
    dense_eval.run(
        run_dir=cfg.run_dir,
        output_dir=cfg.output_dir,
        device=cfg.device,
        subsample_size=cfg.get('subsample_size', None),
        subsample_seed=cfg.get('subsample_seed', None),
        max_eval_batch=cfg.get('max_eval_batch', None),
        episode_index=cfg.get('episode_index', None),
        fractions=cfg.get('fractions', None),
        output_name=cfg.get('output_name', 'dense_eval.npz'),
    )


if __name__ == '__main__':
    main()
