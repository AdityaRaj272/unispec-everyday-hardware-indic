"""
Choosing the draft size g for a device.

Speed of speculative decoding (tokens per second) = tau(g) / s(g):
    tau(g) - tokens gained per decode pass   (the same on every device)
    s(g)   - seconds per decode pass         (depends on the device)
So the speedup over plain decoding is  tau(g) * s_base / s(g),  where s_base is
the time of a one-token pass.

Three ways to pick g, all compared in the results:

1. grid       - just measure the speedup at every g and take the best one
                (what the mid-report gamma sweep did).
2. unispec    - UniSpec's device-aware calibration: fit a regression curve to
                tau(g) and to s(g) from real decoding runs on this device, then
                maximise tau/s over a continuous range of g with Differential
                Evolution.
3. ppd_split  - the split used by PPD's hardware-aware tree pruning: tau(g)
                is measured once (on any device) and only the pass time is
                profiled on the target device, using a cheap synthetic timing
                of "one pass with n tokens" -- no decoding runs needed there.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _tau_model(g, a, b):
    """Saturating curve: tau(0) = 1, grows with g, levels off at 1 + a."""
    return 1.0 + a * (1.0 - np.exp(-np.asarray(g, dtype=float) / b))


def fit_tau(gs: Sequence[float], taus: Sequence[float]):
    from scipy.optimize import curve_fit
    gs, taus = np.asarray(gs, float), np.asarray(taus, float)
    try:
        (a, b), _ = curve_fit(_tau_model, gs, taus, p0=[max(taus.max() - 1, 0.1), 3.0],
                              bounds=([0.0, 0.05], [100.0, 1000.0]), maxfev=20000)
        return lambda g: _tau_model(g, a, b), {"kind": "1+a(1-exp(-g/b))", "a": a, "b": b}
    except Exception:
        return (lambda g: np.interp(g, gs, taus)), {"kind": "interp"}


def fit_poly(xs: Sequence[float], ys: Sequence[float], deg: int = 2):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    deg = min(deg, len(xs) - 1)
    coef = np.polyfit(xs, ys, deg)
    return np.poly1d(coef), {"kind": f"poly{deg}", "coef": coef.tolist()}


def _best_integer(f, lo: int, hi: int) -> int:
    gs = np.arange(lo, hi + 1)
    return int(gs[int(np.argmax([f(g) for g in gs]))])


def grid_optimum(gs: Sequence[int], speedups: Sequence[float]) -> Dict:
    i = int(np.argmax(speedups))
    return {"method": "grid", "g_star": int(gs[i]), "pred_speedup": float(speedups[i])}


def unispec_optimum(gs, taus, pass_secs, base_pass_sec: float) -> Dict:
    """UniSpec Eq. (3)-(4): fit f_accept and f_time, maximise their ratio with
    Differential Evolution over the continuous interval [g_min, g_max]."""
    from scipy.optimize import differential_evolution
    f_tau, tau_info = fit_tau(gs, taus)
    f_time, time_info = fit_poly(gs, pass_secs, deg=2)
    lo, hi = int(min(gs)), int(max(gs))

    def speed(g):
        t = float(f_time(g))
        return float(f_tau(g)) * base_pass_sec / t if t > 0 else 0.0

    res = differential_evolution(lambda x: -speed(x[0]), [(lo, hi)], seed=0, tol=1e-8)
    g_cont = float(res.x[0])
    g_int = max((int(np.floor(g_cont)), int(np.ceil(g_cont))), key=speed)
    g_int = min(max(g_int, lo), hi)
    return {"method": "unispec", "g_star": g_int, "g_continuous": g_cont,
            "pred_speedup": speed(g_int), "tau_fit": tau_info, "time_fit": time_info,
            "curve": {int(g): speed(g) for g in range(lo, hi + 1)}}


def ppd_split_optimum(gs, taus, pass_lens, profile: Dict[int, float]) -> Dict:
    """tau(g) and the mean tokens-per-pass from any device's decoding runs,
    pass time from this device's synthetic profile (n tokens -> seconds)."""
    # The measured profile is read by interpolation rather than a smooth fit: on
    # CPUs the step from a 1-token pass to a 2-token pass is a sharp jump that a
    # polynomial would blur.
    f_tau, _ = fit_tau(gs, taus)
    ns = sorted(profile)
    ts = [profile[n] for n in ns]
    lo, hi = int(min(gs)), int(max(gs))
    s1 = float(profile[1]) if 1 in profile else float(ts[0])

    def speed(g):
        n_in = float(np.interp(g, gs, pass_lens))
        return float(f_tau(g)) * s1 / float(np.interp(n_in, ns, ts))

    g_star = _best_integer(speed, lo, hi)
    return {"method": "ppd_split", "g_star": g_star, "pred_speedup": speed(g_star),
            "time_fit": {"kind": "interp"}, "curve": {int(g): speed(g) for g in range(lo, hi + 1)}}
