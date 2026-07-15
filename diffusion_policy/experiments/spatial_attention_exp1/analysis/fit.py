"""Per-state convergence fitting: MSE_i(o) ≈ c(o) + b(o)·exp(−λ(o)·i).

``i`` is the checkpoint index (0..C-1). Primary fit uses scipy ``curve_fit`` with
non-negativity bounds. Fallback (when curve_fit fails / errors): ĉ = mean of the
last 3 checkpoints, then λ (and b) by linear regression of ``log(MSE − ĉ)`` over the
checkpoints where ``MSE > ĉ``. Fit-failure states are flagged so the report can
record the failure fraction.
"""
import numpy as np
from scipy.optimize import curve_fit


def _model(i, c, b, lam):
    return c + b * np.exp(-lam * i)


def fit_state(mse, x=None):
    """Return (c, b, lam, success) for a single state's MSE-over-checkpoints."""
    mse = np.asarray(mse, dtype=float)
    C = len(mse)
    if x is None:
        x = np.arange(C, dtype=float)

    c0 = float(np.min(mse))
    b0 = float(max(mse[0] - c0, 1e-6))
    p0 = [c0, b0, 0.1]
    try:
        popt, _ = curve_fit(
            _model, x, mse, p0=p0,
            bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
            maxfev=10000)
        c, b, lam = (float(v) for v in popt)
        if not np.all(np.isfinite([c, b, lam])):
            raise RuntimeError("non-finite fit")
        return c, b, lam, True
    except Exception:
        # fallback: floor = mean of last 3, lambda by log-linear regression
        c_hat = float(np.mean(mse[-3:])) if C >= 3 else float(np.min(mse))
        resid = mse - c_hat
        m = resid > 0
        if int(m.sum()) >= 2:
            slope, intercept = np.polyfit(x[m], np.log(resid[m]), 1)
            lam = float(max(-slope, 0.0))
            b = float(np.exp(intercept))
        else:
            lam = 0.0
            b = float(max(mse[0] - c_hat, 0.0))
        return c_hat, b, lam, False


def fit_all(mse_CS, x=None):
    """Fit every state. ``mse_CS`` is (C, S). Returns arrays (c, b, lam, success)."""
    C, S = mse_CS.shape
    c = np.empty(S)
    b = np.empty(S)
    lam = np.empty(S)
    success = np.empty(S, dtype=bool)
    for s in range(S):
        c[s], b[s], lam[s], success[s] = fit_state(mse_CS[:, s], x)
    return c, b, lam, success
