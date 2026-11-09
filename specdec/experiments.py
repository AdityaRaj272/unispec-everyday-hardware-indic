"""
The experiments. Each one writes a JSON file into results/<device>/ as soon as
it finishes, so a long laptop run can be interrupted and resumed.

  correctness  speculative output must equal plain greedy output, token for token
  profile      hardware fingerprint: seconds for one pass with n new tokens, s(n)
  sweep        real decoding at every draft size g: tau(g), pass time, speedup
  calibrate    choose g* three ways (grid / UniSpec / PPD split), check on held-out prompts
  language     tokens-per-word, acceptance and speedup for En/Hi/Mr/Bn/Ta
  thermal      sustained load: does the device (and its best g) drift over time?
"""
from __future__ import annotations

import glob
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import calibrate as cal
from .data import encode, load_prompts
from .drafters import NGramDrafter
from .engine import GenResult, TorchVerifier, greedy_generate, speculative_generate


# --------------------------------------------------------------------------
# Presets: how much work each experiment does
# --------------------------------------------------------------------------

PRESETS = {
    # a few minutes; checks that everything runs end to end
    "quick": dict(max_new=48, n_calib=2, n_eval=2, repeats=1, grid=[1, 2, 4, 8],
                  profile_ctx=[512], profile_n=[1, 2, 3, 5, 9], profile_reps=3,
                  n_correct=2, lang_n=1, lang_grid=[4], thermal_min=1.0),
    # laptop CPU
    "cpu": dict(max_new=128, n_calib=6, n_eval=6, repeats=1,
                grid=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
                profile_ctx=[512, 1024, 2048], profile_n=[1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33],
                profile_reps=7, n_correct=4, lang_n=5, lang_grid=[4, 8], thermal_min=20.0),
    # free T4 / any CUDA or Apple GPU
    "gpu": dict(max_new=128, n_calib=15, n_eval=15, repeats=3,
                grid=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
                profile_ctx=[512, 1024, 2048], profile_n=[1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33],
                profile_reps=15, n_correct=8, lang_n=20, lang_grid=[2, 4, 8], thermal_min=20.0),
}


@dataclass
class Ctx:
    model: object
    tok: object
    verifier: TorchVerifier
    eos: List[int]
    sync: object
    info: Dict
    out_dir: str
    cfg: Dict
    results_root: str = "results"
    log: List[str] = field(default_factory=list)

    def path(self, name: str) -> str:
        return os.path.join(self.out_dir, f"{name}.json")

    def save(self, name: str, obj) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        obj = {"device": self.info, "config": self.cfg, **obj}
        with open(self.path(name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, ensure_ascii=False)
        print(f"  -> saved {self.path(name)}")

    def load(self, name: str) -> Optional[Dict]:
        p = self.path(name)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _gen(ctx: Ctx, ids: List[int], g: int) -> GenResult:
    """g = 0 means plain greedy decoding (the baseline)."""
    if g == 0:
        return greedy_generate(ctx.verifier, ids, ctx.cfg["max_new"], ctx.eos, ctx.sync)
    return speculative_generate(ctx.verifier, ids, NGramDrafter(draft_len=g),
                                ctx.cfg["max_new"], ctx.eos, ctx.sync)


def _median_run(ctx: Ctx, ids: List[int], g: int, repeats: int) -> Dict:
    runs = [_gen(ctx, ids, g) for _ in range(repeats)]
    best = sorted(runs, key=lambda r: r.latency_sec)[len(runs) // 2]
    d = best.to_dict()
    d["latency_sec"] = statistics.median(r.latency_sec for r in runs)
    d["decode_sec"] = statistics.median(r.decode_sec for r in runs)
    d["token_ids"] = best.token_ids
    return d


def _warmup(ctx: Ctx, ids: List[int]) -> None:
    for g in (0, 4):
        speculative_generate(ctx.verifier, ids, None if g == 0 else NGramDrafter(g), 16, ctx.eos, ctx.sync)


def copy_stats(prompt_ids: List[int], out_ids: List[int], n: int = 4) -> Dict[str, float]:
    """Where do the output's 4-token sequences come from? An n-gram drafter can
    only guess what already exists: either text copied from the article, or the
    model repeating its own earlier output (common for very small models)."""
    grams = [tuple(out_ids[i:i + n]) for i in range(len(out_ids) - n + 1)]
    if not grams:
        return {"copy_from_prompt": 0.0, "self_repeat": 0.0}
    src = {tuple(prompt_ids[i:i + n]) for i in range(len(prompt_ids) - n + 1)}
    seen, from_src, rep = set(), 0, 0
    for gm in grams:
        from_src += gm in src
        rep += gm in seen
        seen.add(gm)
    return {"copy_from_prompt": from_src / len(grams), "self_repeat": rep / len(grams)}


def run_paired(ctx: Ctx, prompts: List[Dict], gs: List[int], repeats: int, tag: str) -> List[Dict]:
    """For every prompt: baseline then every g, back to back (paired), so slow
    drift of the machine (heat, background apps) affects all methods equally.
    The order of g is shuffled per prompt for the same reason."""
    rows = []
    for k, p in enumerate(prompts):
        ids = encode(ctx.tok, p)
        base = _median_run(ctx, ids, 0, repeats)
        cs = copy_stats(ids, base["token_ids"])
        base.update(cs)
        order = list(gs)
        random.Random(k).shuffle(order)
        line = []
        for g in order:
            d = _median_run(ctx, ids, g, repeats)
            d.update(prompt_id=p["id"], lang=p["lang"], split=p["split"], g=g,
                     n_prompt_tokens=len(ids),
                     speedup=base["latency_sec"] / d["latency_sec"],
                     decode_speedup=base["decode_sec"] / d["decode_sec"],
                     exact_match=int(d["token_ids"] == base["token_ids"]),
                     base_latency_sec=base["latency_sec"], base_decode_sec=base["decode_sec"],
                     base_mean_pass_sec=base["mean_pass_sec"], **cs)
            d.pop("token_ids")
            rows.append(d)
            line.append(f"g{g}:{d['speedup']:.2f}x")
        base.update(prompt_id=p["id"], lang=p["lang"], split=p["split"], g=0,
                    n_prompt_tokens=len(ids), speedup=1.0, decode_speedup=1.0, exact_match=1)
        base.pop("token_ids")
        rows.append(base)
        print(f"  [{tag}] {k + 1}/{len(prompts)} {p['id']} ({len(ids)} tok, base "
              f"{base['latency_sec']:.1f}s)  " + " ".join(sorted(line, key=lambda s: int(s[1:s.index(':')]))))
    return rows


def aggregate(rows: List[Dict], split: Optional[str] = None) -> Dict[int, Dict]:
    """Per-g averages. Speedups are averaged per prompt (paired ratios)."""
    out: Dict[int, Dict] = {}
    for g in sorted({r["g"] for r in rows}):
        rs = [r for r in rows if r["g"] == g and (split is None or r["split"] == split)]
        if not rs:
            continue
        keys = ("tau", "acceptance_rate", "mean_pass_sec", "mean_pass_len", "speedup",
                "decode_speedup", "exact_match", "latency_sec", "prefill_sec", "draft_sec",
                "n_tokens", "copy_from_prompt", "self_repeat")
        out[g] = {k: float(np.mean([r[k] for r in rs])) for k in keys if all(k in r for r in rs)}
        out[g]["n"] = len(rs)
    return out


# --------------------------------------------------------------------------
# 1. Correctness gate
# --------------------------------------------------------------------------

def correctness(ctx: Ctx) -> Dict:
    prompts = load_prompts("english", n=ctx.cfg["n_correct"])
    _warmup(ctx, encode(ctx.tok, prompts[0]))
    res = []
    for p in prompts:
        ids = encode(ctx.tok, p)
        base = _gen(ctx, ids, 0).token_ids
        for g in (2, 8, 32):
            got = _gen(ctx, ids, g).token_ids
            same = sum(1 for a, b in zip(got, base) if a == b) if got != base else len(base)
            first_diff = next((i for i, (a, b) in enumerate(zip(got, base)) if a != b), None)
            res.append({"prompt_id": p["id"], "g": g, "exact_match": int(got == base),
                        "first_diff_at": first_diff, "n_tokens": len(base), "matching_prefix": same})
    rate = float(np.mean([r["exact_match"] for r in res]))
    print(f"  exact match vs greedy: {rate * 100:.0f}% of {len(res)} runs "
          f"({ctx.info['dtype']}; fp32 should give 100%, fp16 may miss on rare near-ties)")
    out = {"exact_match_rate": rate, "runs": res}
    ctx.save("correctness", out)
    return out


# --------------------------------------------------------------------------
# 2. Hardware profile s(n)
# --------------------------------------------------------------------------

def profile(ctx: Ctx, name: str = "profile") -> Dict:
    """Time one forward pass that adds n tokens on top of a cache of L tokens.
    No drafting involved: this is purely what the hardware charges per pass."""
    rng = random.Random(0)
    vocab = ctx.tok.vocab_size
    v = ctx.verifier
    curves = {}
    for L in ctx.cfg["profile_ctx"]:
        v.reset()
        v.forward([rng.randrange(100, vocab) for _ in range(L)], n_keep=1)
        ctx.sync()
        times = {}
        for n in ctx.cfg["profile_n"]:
            toks = [rng.randrange(100, vocab) for _ in range(n)]
            samples = []
            for rep in range(ctx.cfg["profile_reps"] + 2):     # first 2 are warm-up
                v.crop(L)
                ctx.sync()
                t = time.perf_counter()
                v.forward(toks, n_keep=n)
                ctx.sync()
                if rep >= 2:
                    samples.append(time.perf_counter() - t)
            times[n] = statistics.median(samples)
        curves[L] = times
        print(f"  ctx {L:5d}: " + "  ".join(f"n={n}:{t * 1000:.0f}ms" for n, t in times.items()))
    out = {"curves": {str(L): {str(n): t for n, t in c.items()} for L, c in curves.items()}}
    ctx.save(name, out)
    return out


# --------------------------------------------------------------------------
# 3. Draft-size sweep
# --------------------------------------------------------------------------

def sweep(ctx: Ctx) -> Dict:
    c = ctx.cfg
    prompts = load_prompts("english", "calib", c["n_calib"]) + load_prompts("english", "eval", c["n_eval"])
    _warmup(ctx, encode(ctx.tok, prompts[0]))
    rows = run_paired(ctx, prompts, c["grid"], c["repeats"], "sweep")
    agg = aggregate(rows)
    print("   g   tau  accept  pass(ms)  tok/pass  speedup")
    for g, a in agg.items():
        print(f"  {g:2d}  {a['tau']:4.2f}  {a['acceptance_rate']:5.2f}  {a['mean_pass_sec'] * 1000:7.1f}"
              f"  {a['mean_pass_len']:7.1f}  {a['speedup']:6.2f}x")
    out = {"rows": rows}
    ctx.save("sweep", out)
    return out


# --------------------------------------------------------------------------
# 4. Calibration + held-out check
# --------------------------------------------------------------------------

def _profile_at(prof: Dict, ctx_len: float) -> Dict[int, float]:
    """Pick the profiled context length closest to the typical one."""
    Ls = sorted(int(L) for L in prof["curves"])
    L = min(Ls, key=lambda x: abs(x - ctx_len))
    return {int(n): t for n, t in prof["curves"][str(L)].items()}


def _other_device_sweeps(ctx: Ctx) -> Dict[str, List[Dict]]:
    out = {}
    for p in glob.glob(os.path.join(ctx.results_root, "*", "sweep.json")):
        d = os.path.basename(os.path.dirname(p))
        if os.path.abspath(os.path.dirname(p)) != os.path.abspath(ctx.out_dir):
            with open(p, encoding="utf-8") as f:
                out[d] = json.load(f)["rows"]
    return out


def _base_condition_sweep(ctx: Ctx) -> Optional[Dict]:
    """For a tagged condition (e.g. power-saver) without its own sweep: the
    same device's untagged sweep. tau does not depend on the condition."""
    name = os.path.basename(os.path.normpath(ctx.out_dir))
    if "__" not in name:
        return None
    p = os.path.join(ctx.results_root, name.split("__")[0], "sweep.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def calibrate(ctx: Ctx) -> Dict:
    sw, prof = ctx.load("sweep"), ctx.load("profile")
    if prof is None:
        raise SystemExit("calibrate needs profile.json -- run `profile` first")
    if sw is None:
        base = _base_condition_sweep(ctx)
        if base is None:
            raise SystemExit("calibrate needs sweep.json -- run `sweep` first")
        return recalibrate_from_profile(ctx, base, prof)
    rows = sw["rows"]
    calib = aggregate([r for r in rows if r["split"] == "calib"])
    gs = [g for g in calib if g > 0]
    taus = [calib[g]["tau"] for g in gs]
    lens = [calib[g]["mean_pass_len"] for g in gs]
    base_pass = calib[0]["mean_pass_sec"]
    typical_ctx = float(np.mean([r["n_prompt_tokens"] for r in rows])) + ctx.cfg["max_new"] / 2
    prof_curve = _profile_at(prof, typical_ctx)

    choices = {
        "grid": cal.grid_optimum(gs, [calib[g]["speedup"] for g in gs]),
        "unispec": cal.unispec_optimum(gs, taus, [calib[g]["mean_pass_sec"] for g in gs], base_pass),
        "ppd_split": cal.ppd_split_optimum(gs, taus, lens, prof_curve),
    }
    # PPD's "measure acceptance once": reuse tau from another device's sweep, if any
    for dev, orows in _other_device_sweeps(ctx).items():
        oc = aggregate([r for r in orows if r["split"] == "calib" and r["g"] > 0])
        ogs = sorted(oc)
        choices[f"ppd_split_tau_from:{dev}"] = cal.ppd_split_optimum(
            ogs, [oc[g]["tau"] for g in ogs], [oc[g]["mean_pass_len"] for g in ogs], prof_curve)

    # Held-out check: measured speedup at each chosen g on the eval prompts.
    # If any chosen g is not in the sweep grid, ALL chosen g are re-measured
    # together in one session, so no choice is compared against numbers taken
    # under different machine conditions (e.g. another day, or on battery).
    sweep_ev = aggregate([r for r in rows if r["split"] == "eval"])
    chosen = sorted({ch["g_star"] for ch in choices.values()})
    extra_rows = []
    if set(chosen) - set(sweep_ev):
        print(f"  measuring g={chosen} together on held-out prompts (some are not in the sweep grid)")
        prompts = load_prompts("english", "eval", ctx.cfg["n_eval"])
        extra_rows = run_paired(ctx, prompts, chosen, ctx.cfg["repeats"], "held-out")
        ev = aggregate(extra_rows)
    else:
        ev = sweep_ev
    for name, ch in choices.items():
        e = ev[ch["g_star"]]
        ch.update(heldout_speedup=e["speedup"], heldout_decode_speedup=e["decode_speedup"])
        print(f"  {name:40s} g*={ch['g_star']:3d}  predicted {ch['pred_speedup']:.2f}x  "
              f"held-out {e['speedup']:.2f}x (decode {e['decode_speedup']:.2f}x)")
    best_eval = max((g for g in sweep_ev if g > 0), key=lambda g: sweep_ev[g]["speedup"])
    out = {"choices": choices, "typical_ctx": typical_ctx, "calib_agg": calib,
           "eval_agg": ev, "sweep_eval_agg": sweep_ev, "eval_best_g": best_eval,
           "extra_rows": extra_rows, "g_star": choices["unispec"]["g_star"]}
    print(f"  best grid g on held-out prompts in the sweep (hindsight): {best_eval} -> "
          f"{sweep_ev[best_eval]['speedup']:.2f}x")
    ctx.save("calibration", out)
    return out


def recalibrate_from_profile(ctx: Ctx, base_sweep: Dict, prof: Dict) -> Dict:
    """A new condition on a known device (battery saver, hot, ...): only the
    2-minute profile is re-measured; tau comes from the earlier sweep (PPD's
    split). Then the new g* and the old g* are both measured on held-out
    prompts under the new condition -- the cost of not re-calibrating."""
    rows = base_sweep["rows"]
    calib = aggregate([r for r in rows if r["split"] == "calib" and r["g"] > 0])
    gs = sorted(calib)
    typical_ctx = float(np.mean([r["n_prompt_tokens"] for r in rows])) + ctx.cfg["max_new"] / 2
    new = cal.ppd_split_optimum(gs, [calib[g]["tau"] for g in gs], [calib[g]["mean_pass_len"] for g in gs],
                                _profile_at(prof, typical_ctx))
    base_name = os.path.basename(os.path.normpath(ctx.out_dir)).split("__")[0]
    p = os.path.join(ctx.results_root, base_name, "calibration.json")
    old_g = json.load(open(p, encoding="utf-8"))["g_star"] if os.path.exists(p) else None
    test = sorted({new["g_star"]} | ({old_g} if old_g else set()))
    print(f"  tau from {base_name}; new g*={new['g_star']} (old g*={old_g}); measuring both on held-out prompts")
    prompts = load_prompts("english", "eval", ctx.cfg["n_eval"])
    _warmup(ctx, encode(ctx.tok, prompts[0]))
    erows = run_paired(ctx, prompts, test, ctx.cfg["repeats"], "recal")
    ev = aggregate(erows)
    new.update(heldout_speedup=ev[new["g_star"]]["speedup"],
               heldout_decode_speedup=ev[new["g_star"]]["decode_speedup"])
    choices = {"ppd_split_tau_from:" + base_name: new}
    if old_g:
        choices["old_g_without_recalibration"] = {
            "method": "old", "g_star": old_g, "pred_speedup": float("nan"),
            "heldout_speedup": ev[old_g]["speedup"], "heldout_decode_speedup": ev[old_g]["decode_speedup"]}
    for n, c in choices.items():
        print(f"  {n:40s} g*={c['g_star']:3d}  held-out {c['heldout_speedup']:.2f}x")
    out = {"choices": choices, "typical_ctx": typical_ctx, "eval_agg": ev, "extra_rows": erows,
           "eval_best_g": max((g for g in ev if g > 0), key=lambda g: ev[g]["speedup"]),
           "g_star": new["g_star"], "tau_borrowed_from": base_name}
    ctx.save("calibration", out)
    return out


# --------------------------------------------------------------------------
# 5. Language axis
# --------------------------------------------------------------------------

LANGS = ["english", "hindi", "marathi", "bengali", "tamil"]


def fertility(tok, langs=LANGS) -> Dict[str, Dict]:
    """Tokens per word on the article text (no model run needed)."""
    out = {}
    for lang in langs:
        rows = load_prompts(lang)
        words = sum(len(r["text"].split()) for r in rows)
        toks = sum(len(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows)
        chars = sum(len(r["text"]) for r in rows)
        out[lang] = {"tokens_per_word": toks / words, "chars_per_token": chars / toks}
    return out


def language(ctx: Ctx, extra_tokenizers: List[str] = ()) -> Dict:
    from transformers import AutoTokenizer
    fert = {ctx.info.get("model", "model"): fertility(ctx.tok)}
    for name in extra_tokenizers:
        try:
            fert[name] = fertility(AutoTokenizer.from_pretrained(name))
        except Exception as e:                                # gated / unavailable
            print(f"  skipping tokenizer {name}: {e.__class__.__name__}")
    for name, f in fert.items():
        print(f"  tokens/word [{name}]: " + "  ".join(f"{l[:2]}={v['tokens_per_word']:.2f}" for l, v in f.items()))

    calib = ctx.load("calibration")
    gs = sorted(set(ctx.cfg["lang_grid"]) | ({calib["g_star"]} if calib else set()))
    rows = []
    for lang in LANGS:
        prompts = load_prompts(lang, "eval", ctx.cfg["lang_n"])
        if lang == LANGS[0]:
            _warmup(ctx, encode(ctx.tok, prompts[0]))
        rows += run_paired(ctx, prompts, gs, ctx.cfg["repeats"], lang)
    summary = {}
    for lang in LANGS:
        a = aggregate([r for r in rows if r["lang"] == lang])
        summary[lang] = {str(g): v for g, v in a.items()}
        summary[lang]["prompt_tokens"] = float(np.mean([r["n_prompt_tokens"] for r in rows if r["lang"] == lang]))
        print(f"  {lang:8s} prompt {summary[lang]['prompt_tokens']:.0f} tok  " +
              "  ".join(f"g{g}: tau {v['tau']:.2f} acc {v['acceptance_rate']:.2f} {v['speedup']:.2f}x"
                        for g, v in a.items() if g > 0))
    out = {"fertility": fert, "grid": gs, "summary": summary, "rows": rows}
    ctx.save("language", out)
    return out


# --------------------------------------------------------------------------
# 6. Thermal / sustained-load stability
# --------------------------------------------------------------------------

def thermal(ctx: Ctx, minutes: Optional[float] = None) -> Dict:
    """Run back-to-back rounds for `minutes`. Each round: time a 1-token pass and
    a (1+g*)-token pass, and generate one prompt at baseline and at g*. If the
    device throttles, pass times rise and the speedup at the old g* may change."""
    minutes = minutes or ctx.cfg["thermal_min"]
    calib = ctx.load("calibration")
    g_star = calib["g_star"] if calib else 8
    prompts = load_prompts("english", "eval", ctx.cfg["n_eval"])
    rng = random.Random(0)
    vocab = ctx.tok.vocab_size
    L = 1024
    try:
        import psutil
    except Exception:
        psutil = None
    rounds, t_start, k = [], time.time(), 0
    while time.time() - t_start < minutes * 60:
        v = ctx.verifier
        v.reset()
        v.forward([rng.randrange(100, vocab) for _ in range(L)], n_keep=1)
        pass_t = {}
        for n in (1, 1 + g_star):
            toks = [rng.randrange(100, vocab) for _ in range(n)]
            ss = []
            for _ in range(5):
                v.crop(L)
                ctx.sync()
                t = time.perf_counter()
                v.forward(toks, n_keep=n)
                ctx.sync()
                ss.append(time.perf_counter() - t)
            pass_t[n] = statistics.median(ss)
        p = prompts[k % len(prompts)]
        ids = encode(ctx.tok, p)
        b, s = _gen(ctx, ids, 0), _gen(ctx, ids, g_star)
        r = {"t_min": (time.time() - t_start) / 60, "prompt_id": p["id"],
             "pass_1_sec": pass_t[1], "pass_g_sec": pass_t[1 + g_star],
             "speedup": b.latency_sec / s.latency_sec, "base_tok_per_sec": b.n_tokens / b.latency_sec,
             "cpu_mhz": psutil.cpu_freq().current if psutil and psutil.cpu_freq() else None}
        rounds.append(r)
        print(f"  t={r['t_min']:5.1f}min  1-tok pass {r['pass_1_sec'] * 1000:6.1f}ms  "
              f"{1 + g_star}-tok pass {r['pass_g_sec'] * 1000:6.1f}ms  base {r['base_tok_per_sec']:5.1f} tok/s  "
              f"speedup@g{g_star} {r['speedup']:.2f}x")
        k += 1
    out = {"g_star": g_star, "minutes": minutes, "rounds": rounds}
    ctx.save("thermal", out)
    return out
