"""
Every number the report quotes, computed from results/.

A "run" is one folder in results/: a device (e.g. cpu-intel-core-i7-8550u) or
the same device under another setting (e.g. cuda-tesla-t4__wide-grid). Adding
a device's folder and rebuilding the report updates every table that lists
runs; nothing here is typed in by hand.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from specdec.experiments import aggregate  # noqa: E402
from specdec.plots import LANG_ORDER, _load, wrong_g_table  # noqa: E402

TAG_NAMES = {"wide-grid": "extended sweep", "on-battery": "on battery", "power-saver": "power saver"}


@dataclass
class Run:
    folder: str
    exp: Dict[str, Dict]
    info: Dict
    kind: str                      # cpu | cuda | mps
    tag: str = ""
    label: str = ""                # e.g. "Tesla T4 GPU (extended sweep)"
    short: str = ""                # e.g. "Tesla T4"
    sweep: Dict[int, Dict] = field(default_factory=dict)

    @property
    def is_main(self) -> bool:
        return not self.tag

    def has(self, name: str) -> bool:
        return name in self.exp


def _hardware_name(info: Dict) -> str:
    if info["device"] == "cpu":
        n = info["cpu"].replace("(R)", "").replace("(TM)", "").replace(" CPU", "")
        return " ".join(n.split("@")[0].split())
    if info["device"] == "mps":
        return "Apple Silicon"
    return info.get("gpu", "GPU")


def load_runs(results_root: str) -> List[Run]:
    runs = []
    for folder, exp in _load(results_root).items():
        info = next(iter(exp.values()))["device"]
        kind = info["device"].split(":")[0]
        tag = folder.split("__", 1)[1] if "__" in folder else ""
        hw = _hardware_name(info)
        what = {"cpu": "CPU", "cuda": "GPU", "mps": "GPU"}[kind]
        label = f"{hw} {what}" + (f" ({TAG_NAMES.get(tag, tag)})" if tag else "")
        r = Run(folder, exp, info, kind, tag, label, hw)
        if "sweep" in exp:
            r.sweep = aggregate(exp["sweep"]["rows"])
        runs.append(r)
    # CPUs first, then GPUs; main runs before their variants
    runs.sort(key=lambda r: ({"cpu": 0, "mps": 1, "cuda": 2}[r.kind], r.short, r.tag))
    return runs


def main_runs(runs: List[Run]) -> List[Run]:
    return [r for r in runs if r.is_main]


# ---------------------------------------------------------------------------
# Per-experiment summaries
# ---------------------------------------------------------------------------

def profile_at(run: Run, ctx: str = "1024") -> Dict[int, float]:
    curves = run.exp["profile"]["curves"]
    c = curves.get(ctx) or curves[sorted(curves, key=int)[len(curves) // 2]]
    return {int(n): t * 1000 for n, t in c.items()}


def tau_on_common(runs: List[Run]):
    rs = [r for r in runs if r.sweep]
    common = set.intersection(*({x["prompt_id"] for x in r.exp["sweep"]["rows"]} for r in rs))
    aggs = {r.folder: aggregate([x for x in r.exp["sweep"]["rows"] if x["prompt_id"] in common]) for r in rs}
    gs = sorted(set.intersection(*(set(a) for a in aggs.values())) - {0})
    gap = max(abs(aggs[a.folder][g]["tau"] - aggs[b.folder][g]["tau"])
              for g in gs for a in rs for b in rs) if len(rs) > 1 else 0.0
    return rs, common, aggs, gs, gap


def best_g(run: Run):
    gs = [g for g in run.sweep if g > 0]
    g = max(gs, key=lambda x: run.sweep[x]["speedup"])
    return g, run.sweep[g]["speedup"]


def slower_than_baseline(run: Run, g: int) -> List[float]:
    rows = [x for x in run.exp["sweep"]["rows"] if x["g"] == g]
    return sorted(x["speedup"] for x in rows if x["speedup"] < 1.0)


def sweep_exact_range(run: Run):
    ex = [v["exact_match"] for g, v in run.sweep.items() if g > 0]
    return min(ex), max(ex)


def copy_correlation(run: Run, g: int = 8):
    rows = [x for x in run.exp.get("sweep", {}).get("rows", []) + run.exp.get("language", {}).get("rows", [])
            if x["g"] == g and "copy_from_prompt" in x]
    seen, xs, ys = set(), [], []
    for x in rows:
        if x["prompt_id"] not in seen:
            seen.add(x["prompt_id"])
            xs.append(x["copy_from_prompt"])
            ys.append(x["decode_speedup"])
    if len(xs) < 3:
        return None, 0
    return float(np.corrcoef(xs, ys)[0, 1]), len(xs)


def split_copy(run: Run):
    base = [x for x in run.exp["sweep"]["rows"] if x["g"] == 0]
    out = {}
    for s in ("calib", "eval"):
        v = [x["copy_from_prompt"] for x in base if x["split"] == s]
        out[s] = float(np.mean(v)) if v else float("nan")
    return out


def language_rows(run: Run):
    lg = run.exp["language"]
    g = run.exp.get("calibration", {}).get("g_star") or max(lg["grid"])
    rows = []
    for lang in LANG_ORDER:
        s = lg["summary"][lang][str(g)]
        b = lg["summary"][lang]["0"]
        rows.append(dict(lang=lang, g=g, tau=s["tau"], acc=s["acceptance_rate"], e2e=s["speedup"],
                         gen=s["decode_speedup"], copy=s.get("copy_from_prompt", float("nan")),
                         rep=s.get("self_repeat", float("nan")), prefill=b["prefill_sec"],
                         latency=b["latency_sec"], prompt_tokens=lg["summary"][lang]["prompt_tokens"],
                         n=int(s["n"])))
    return rows


def fertility(runs: List[Run]):
    for r in runs:
        if r.has("language"):
            return r.exp["language"]["fertility"]
    return {}


def thermal_cycles(run: Run):
    th = run.exp["thermal"]
    r = th["rounds"]
    n = len({x["prompt_id"] for x in r})
    cyc = [r[i:i + n] for i in range(0, len(r) - n + 1, n)]
    return th["g_star"], [dict(t0=c[0]["t_min"], t1=c[-1]["t_min"],
                               p1=float(np.median([x["pass_1_sec"] for x in c])) * 1000,
                               pg=float(np.median([x["pass_g_sec"] for x in c])) * 1000,
                               tps=float(np.mean([x["base_tok_per_sec"] for x in c])),
                               spd=float(np.mean([x["speedup"] for x in c]))) for c in cyc]


def wrong_g(runs: List[Run]):
    data = {r.folder: r.exp for r in runs}
    labels = {r.folder: r.label for r in runs}
    return wrong_g_table(data, labels)


METHOD_NAMES = {"grid": "Grid search", "unispec": "UniSpec fit", "ppd_split": "PPD split",
                "old_g_without_recalibration": "Previous g*, not re-calibrated"}


def method_name(key: str, runs: List[Run]) -> str:
    if key.startswith("ppd_split_tau_from:"):
        src = key.split(":", 1)[1]
        lab = next((r.label for r in runs if r.folder == src), src)
        return f"PPD split, τ from {lab}"
    return METHOD_NAMES.get(key, key)
