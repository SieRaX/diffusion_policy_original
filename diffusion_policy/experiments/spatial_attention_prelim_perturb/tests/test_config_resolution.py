"""Checkpoint-derived config resolution: fails loudly on a missing field, detects
the obs variant, and rejects non-flow-matching policies."""
import pytest
from omegaconf import OmegaConf

from diffusion_policy.experiments.spatial_attention_prelim_perturb.runner import _resolve_from_checkpoint

FM_LOWDIM = ('diffusion_policy.policy.flow_matching_unet_lowdim_policy.'
             'FlowMatchingUnetLowdimPolicy')
FM_IMAGE = ('diffusion_policy.policy.flow_matching_unet_hybrid_image_policy.'
            'FlowMatchingUnetHybridImagePolicy')


def _lowdim_cfg():
    return OmegaConf.create({
        'task': {'name': 'lift_lowdim', 'task_name': 'lift', 'obs_keys': ['object'],
                 'abs_action': True, 'dataset': {'dataset_path': '/data/x.hdf5'}},
        'n_obs_steps': 2, 'horizon': 16,
        'policy': {'_target_': FM_LOWDIM},
    })


def test_resolves_lowdim():
    r = _resolve_from_checkpoint(_lowdim_cfg(), None)
    assert r['variant'] == 'lowdim'
    assert r['task_name'] == 'lift_lowdim'
    assert r['abs_action'] is True
    assert r['dataset_path'] == '/data/x.hdf5'
    assert r['n_obs_steps'] == 2 and r['horizon'] == 16


def test_image_variant_detected():
    cfg = OmegaConf.create({
        'task': {'name': 'lift_image',
                 'shape_meta': {'obs': {'agentview_image': {'shape': [3, 84, 84], 'type': 'rgb'}},
                                'action': {'shape': [10]}},
                 'dataset': {'dataset_path': '/data/x'}},
        'n_obs_steps': 2, 'horizon': 16, 'policy': {'_target_': FM_IMAGE},
    })
    r = _resolve_from_checkpoint(cfg, None)
    assert r['variant'] == 'image'
    assert r['shape_meta']['action']['shape'] == [10]


def test_missing_field_fails_loudly():
    cfg = _lowdim_cfg()
    del cfg['n_obs_steps']
    with pytest.raises(KeyError):
        _resolve_from_checkpoint(cfg, None)


def test_non_flow_matching_rejected():
    cfg = _lowdim_cfg()
    cfg.policy._target_ = 'diffusion_policy.policy.diffusion_unet_lowdim_policy.DiffusionUnetLowdimPolicy'
    with pytest.raises(ValueError):
        _resolve_from_checkpoint(cfg, None)


def test_dataset_override_wins():
    r = _resolve_from_checkpoint(_lowdim_cfg(), '/override/path.hdf5')
    assert r['dataset_path'] == '/override/path.hdf5'
