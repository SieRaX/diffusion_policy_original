"""Hydra entry point (ONE entry point) for the perturbation-sensitivity experiment.

Usage (from repo root):
    MUJOCO_GL=osmesa python -m \
      diffusion_policy.experiments.spatial_attention_prelim_perturb.run_perturb \
      checkpoint=<ckpt.ckpt> demo_index=0 device=cuda:0
"""
import pathlib

import hydra
from omegaconf import OmegaConf

# robosuite/mujoco env is constructed in the runner; keep spawn like the repo's
# other env-constructing entry points.
import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

OmegaConf.register_new_resolver("eval", eval, replace=True)

from diffusion_policy.experiments.spatial_attention_prelim_perturb import runner


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('config')),
    config_name='perturb')
def main(cfg):
    OmegaConf.resolve(cfg)
    runner.run(cfg)


if __name__ == '__main__':
    main()
