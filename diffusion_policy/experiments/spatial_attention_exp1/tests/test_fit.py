import numpy as np

from diffusion_policy.experiments.spatial_attention_exp1.analysis import fit as fitmod


def test_fit_recovers_exponential():
    C = 20
    x = np.arange(C)
    c, b, lam = 0.05, 1.0, 0.3
    mse = c + b * np.exp(-lam * x)
    fc, fb, flam, ok = fitmod.fit_state(mse)
    assert ok
    assert abs(flam - lam) < 0.05
    assert abs(fc - c) < 0.02


def test_fallback_used_when_curvefit_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("forced failure")
    monkeypatch.setattr(fitmod, 'curve_fit', boom)

    C = 10
    x = np.arange(C)
    mse = 0.1 + 2.0 * np.exp(-0.5 * x)
    fc, fb, flam, ok = fitmod.fit_state(mse)
    assert ok is False
    # fallback floor = mean of the last 3 checkpoints
    assert abs(fc - float(np.mean(mse[-3:]))) < 1e-9
    # fallback lambda finite and non-negative
    assert np.isfinite(flam) and flam >= 0.0


def test_fallback_degenerate_few_points(monkeypatch):
    monkeypatch.setattr(fitmod, 'curve_fit',
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    # constant series -> resid <= 0 everywhere -> lambda 0
    mse = np.array([0.3, 0.3, 0.3, 0.3])
    fc, fb, flam, ok = fitmod.fit_state(mse)
    assert ok is False
    assert flam == 0.0
    assert np.isfinite(fc)


def test_fit_all_shapes_finite():
    C, S = 15, 7
    x = np.arange(C)
    mse = np.stack([0.02 + (s + 1) * np.exp(-0.1 * (s + 1) * x) for s in range(S)], axis=1)
    c, b, lam, ok = fitmod.fit_all(mse)
    assert c.shape == (S,) and lam.shape == (S,) and ok.shape == (S,)
    assert np.all(np.isfinite(c)) and np.all(np.isfinite(lam))
