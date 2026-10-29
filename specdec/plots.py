"""
Figures and summary tables from every device folder in results/.

Re-run `python run.py plots` whenever a new device's results are added; every
figure and results/SUMMARY.md are regenerated with that device included.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

import numpy as np

# Fixed categorical order (colour-blind-checked); a device keeps its colour in every figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LANG_ORDER = ["english", "hindi", "marathi", "bengali", "tamil"]


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "lines.markersize": 6,
        "font.family": "DejaVu Sans", "axes.axisbelow": True,
    })


def _load(results_root: str) -> Dict[str, Dict[str, Dict]]:
    """{device_folder: {experiment: json}}; quick smoke-test folders are ignored
    unless they are all there is."""
    out = {}
    for d in sorted(glob.glob(os.path.join(results_root, "*"))):
        if not os.path.isdir(d):
            continue
        exp = {}
        for p in glob.glob(os.path.join(d, "*.json")):
            with open(p, encoding="utf-8") as f:
                exp[os.path.splitext(os.path.basename(p))[0]] = json.load(f)
        if exp:
            out[os.path.basename(d)] = exp
    real = {k: v for k, v in out.items() if not k.endswith("__quick")}
    return real or out


def _label(dev: str, exp: Dict) -> str:
    info = next(iter(exp.values()))["device"]
    hw = info.get("gpu") or info["cpu"]
    hw = hw.replace("(R)", "").replace("(TM)", "").replace(" CPU", "").split("@")[0].strip()
    tag = dev.split("__")[1] if "__" in dev else ""
    return f"{hw} ({info['dtype']})" + (f" [{tag}]" if tag else "")


def _agg(rows: List[Dict], split=None) -> Dict[int, Dict]:
    from .experiments import aggregate
    return aggregate(rows, split)


def _legend(ax, **kw):
    h, l = ax.get_legend_handles_labels()
    if len(h) >= 2:
        ax.legend(**kw)


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def fig_fingerprint(data, colors, labels, out_dir):
    """Pass time vs tokens per pass -- the hardware fingerprint."""
    import matplotlib.pyplot as plt
    devs = [d for d in data if "profile" in data[d]]
    if not devs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for d in devs:
        curves = data[d]["profile"]["curves"]
        L = "1024" if "1024" in curves else sorted(curves, key=int)[len(curves) // 2]
        ns = sorted(int(n) for n in curves[L])
        ts = np.array([curves[L][str(n)] for n in ns])
        axes[0].plot(ns, ts * 1000, marker="o", color=colors[d], label=labels[d])
        axes[1].plot(ns, ts / ts[0], marker="o", color=colors[d], label=labels[d])
    axes[0].set(title="Time of one model pass", xlabel="tokens processed in the pass (1 + draft size g)",
                ylabel="milliseconds")
    axes[0].set_ylim(0, None)
    axes[1].axhline(1, color=INK2, lw=1, ls="--")
    axes[1].set(title="Same, relative to a 1-token pass", xlabel="tokens processed in the pass (1 + draft size g)",
                ylabel="× cost of a 1-token pass")
    _legend(axes[1], loc="upper left")
    fig.suptitle("Hardware fingerprint: what checking extra guessed tokens costs (context 1024 tokens)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig1_hardware_fingerprint.png")


def fig_sweep(data, colors, labels, out_dir):
    """tau(g) (device-independent) and measured speedup(g) per device."""
    import matplotlib.pyplot as plt
    devs = [d for d in data if "sweep" in data[d]]
    if not devs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for d in devs:
        a = _agg(data[d]["sweep"]["rows"])
        gs = [g for g in a if g > 0]
        axes[0].plot(gs, [a[g]["tau"] for g in gs], marker="o", color=colors[d], label=labels[d])
        axes[1].plot(gs, [a[g]["speedup"] for g in gs], marker="o", color=colors[d], label=labels[d])
        cal = data[d].get("calibration")
        if cal:
            g = cal["g_star"]          # may sit between grid points: place it on the line
            y = float(np.interp(np.log2(g), np.log2(gs), [a[x]["speedup"] for x in gs]))
            axes[1].plot([g], [y], marker="*", ms=16, color=colors[d],
                         markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
            axes[1].annotate(f"g*={g}", (g, y), xytext=(0, 10), textcoords="offset points",
                             ha="center", fontsize=8, color=INK2)
    axes[0].set(title="τ: tokens gained per model pass", xlabel="draft size g (tokens guessed)", ylabel="tokens per pass")
    axes[0].set_xscale("log", base=2)
    axes[1].axhline(1, color=INK2, lw=1, ls="--")
    axes[1].text(1, 1.0, " no guessing = 1.0×", color=INK2, fontsize=8, va="bottom")
    axes[1].set(title="Measured speedup over normal decoding", xlabel="draft size g (tokens guessed)",
                ylabel="speedup (×)")
    axes[1].set_xscale("log", base=2)
    for ax in axes:
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    _legend(axes[1], loc="lower left")
    fig.suptitle("Same guesses accepted everywhere (left), but the payoff depends on the device (right; ★ = calibrated g*)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig2_draft_size_sweep.png")


def wrong_g_table(data, labels):
    """Speedup on device A when using the g* calibrated on device B."""
    devs = [d for d in data if "calibration" in data[d] and "sweep" in data[d]]
    if not devs:
        return []
    gstars = {d: data[d]["calibration"]["g_star"] for d in devs}
    table = []
    for d in devs:
        rows = data[d]["sweep"]["rows"] + data[d]["calibration"].get("extra_rows", [])
        a = _agg(rows, "eval")
        gs = sorted(g for g in a if g > 0)

        def at(g):
            gg = min(gs, key=lambda x: abs(x - g))
            return gg, a[gg]["speedup"]

        entry = {"device": labels[d], "own_g": gstars[d], "own": at(gstars[d])[1],
                 "largest_g": gs[-1], "largest": a[gs[-1]]["speedup"], "others": {}}
        for o in devs:
            if o != d:
                gg, s = at(gstars[o])
                entry["others"][labels[o]] = {"g": gstars[o], "g_used": gg, "speedup": s}
        table.append(entry)
    return table


def fig_wrong_g(data, colors, labels, out_dir):
    import matplotlib.pyplot as plt
    table = wrong_g_table(data, labels)
    if not table:
        return None, table
    cats = ["own calibrated g*"] + [f"g* of {o}" for o in labels if any(o in t["others"] for t in table)] \
        + ["largest g tried"]
    fig, ax = plt.subplots(figsize=(max(7, 2.4 * len(table) + 3), 4.4))
    width = min(0.8 / len(cats), 0.22)
    shades = ["#2a78d6", "#86b6ef", "#b7d3f6", "#cde2fb", "#9ec5f4"]
    for j, cat in enumerate(cats):
        xs, ys, gl = [], [], []
        for i, t in enumerate(table):
            if cat.startswith("own"):
                v, g = t["own"], t["own_g"]
            elif cat.startswith("largest"):
                v, g = t["largest"], t["largest_g"]
            else:
                o = cat[len("g* of "):]
                if o not in t["others"]:
                    continue
                v, g = t["others"][o]["speedup"], t["others"][o]["g"]
            xs.append(i + (j - (len(cats) - 1) / 2) * width)
            ys.append(v)
            gl.append(g)
        color = "#8c8a83" if cat.startswith("largest") else shades[min(j, len(shades) - 1)]
        bars = ax.bar(xs, ys, width=width * 0.92, color=color, label=cat, edgecolor=SURFACE, linewidth=1)
        for x, y, g in zip(xs, ys, gl):
            ax.text(x, y + 0.01, f"{y:.2f}×\ng={g}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.axhline(1, color=INK, lw=1, ls="--")
    ax.set_xlim(-0.6, len(table) - 0.4)
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels([t["device"] for t in table])
    ax.set(ylabel="speedup on held-out prompts (×)", title="Cost of using the wrong draft size")
    ax.set_ylim(0, None)
    ax.grid(axis="x", visible=False)
    _legend(ax, loc="upper right", fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir, "fig3_wrong_g_cost.png"), table


def fig_calibration(data, colors, labels, out_dir):
    """Predicted vs measured speedup for the three ways of choosing g."""
    import matplotlib.pyplot as plt
    devs = [d for d in data if "calibration" in data[d]]
    if not devs:
        return None
    fig, axes = plt.subplots(1, len(devs), figsize=(4.6 * len(devs), 4), squeeze=False)
    for ax, d in zip(axes[0], devs):
        ch = data[d]["calibration"]["choices"]
        names = list(ch)
        short = [n.replace("ppd_split_tau_from:", "PPD split\n(τ from ").replace("unispec", "UniSpec fit")
                 .replace("ppd_split", "PPD split").replace("grid", "grid search") for n in names]
        short = [s + ")" if "(τ from" in s else s for s in short]
        y = np.arange(len(names))
        pred = [ch[n]["pred_speedup"] for n in names]
        meas = [ch[n]["heldout_speedup"] for n in names]
        ax.barh(y + 0.2, pred, 0.38, color="#86b6ef", label="predicted")
        ax.barh(y - 0.2, meas, 0.38, color="#2a78d6", label="measured (held-out)")
        for yi, p, m, n in zip(y, pred, meas, names):
            if p == p:                                         # not NaN
                ax.text(p + 0.02, yi + 0.2, f"{p:.2f}×", va="center", fontsize=8, color=INK2)
            ax.text(m + 0.02, yi - 0.2, f"{m:.2f}×  (g*={ch[n]['g_star']})", va="center", fontsize=8, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(short, fontsize=8)
        ax.axvline(1, color=INK, lw=1, ls="--")
        ax.set_xlim(0, np.nanmax(pred + meas) * 1.35)
        ax.set(title=labels[d], xlabel="speedup (×)")
        ax.grid(axis="y", visible=False)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Three ways to choose g: what each predicted, and what it delivered on unseen prompts",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig4_calibration_methods.png")


def fig_language(data, colors, labels, out_dir):
    import matplotlib.pyplot as plt
    devs = [d for d in data if "language" in data[d]]
    if not devs:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(len(LANG_ORDER))
    # (a) tokens per word, per tokenizer (device-independent: take the first device)
    fert = data[devs[0]]["language"]["fertility"]
    w = 0.8 / max(len(fert), 1)
    for k, (tk, f) in enumerate(fert.items()):
        ax = axes[0]
        vals = [f[l]["tokens_per_word"] for l in LANG_ORDER]
        ax.bar(x + (k - (len(fert) - 1) / 2) * w, vals, w * 0.92, color=PALETTE[(k + 5) % len(PALETTE)],
               label=tk.split("/")[-1], edgecolor=SURFACE)
    axes[0].set(title="Tokenizer pieces per word", ylabel="tokens per word")
    _legend(axes[0], loc="upper left", fontsize=8)
    for d in devs:
        lg = data[d]["language"]
        g = data[d].get("calibration", {}).get("g_star") or max(lg["grid"])
        acc = [lg["summary"][l][str(g)]["tau"] for l in LANG_ORDER]
        spd = [lg["summary"][l][str(g)]["speedup"] for l in LANG_ORDER]
        dec = [lg["summary"][l][str(g)]["decode_speedup"] for l in LANG_ORDER]
        axes[1].plot(x, acc, marker="o", color=colors[d], label=f"{labels[d]}, g={g}")
        axes[2].plot(x, dec, marker="o", color=colors[d], ls="--", label=f"{labels[d]}: generation only")
        axes[2].plot(x, spd, marker="o", color=colors[d], label=f"{labels[d]}: end to end")
    axes[1].set(title="τ: tokens gained per pass", ylabel="tokens per pass")
    axes[2].axhline(1, color=INK2, lw=1, ls=":")
    axes[2].set(title="Speedup at the device's g*", ylabel="speedup (×)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([l.capitalize() for l in LANG_ORDER])
    axes[0].grid(axis="x", visible=False)
    fig.suptitle("Language axis: do more pieces per word mean fewer accepted guesses and less speedup?\n"
                 "(end to end includes reading the prompt, which guessing cannot speed up)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower right", ncol=2, fontsize=8, bbox_to_anchor=(0.99, -0.08))
    return _save(fig, out_dir, "fig5_language.png")


def fig_thermal(data, colors, labels, out_dir):
    import matplotlib.pyplot as plt
    devs = [d for d in data if "thermal" in data[d]]
    if not devs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for d in devs:
        th = data[d]["thermal"]
        r, g = th["rounds"], th["g_star"]
        t = [x["t_min"] for x in r]
        axes[0].plot(t, [x["pass_1_sec"] * 1000 for x in r], marker="o", ms=4, color=colors[d],
                     label=f"{labels[d]}: 1-token pass")
        axes[0].plot(t, [x["pass_g_sec"] * 1000 for x in r], marker="s", ms=4, ls="--", color=colors[d],
                     label=f"{labels[d]}: {1 + g}-token pass")
        # every round uses the next prompt; average over full cycles so prompt
        # content does not masquerade as drift
        n = len({x["prompt_id"] for x in r})
        cyc = [r[i:i + n] for i in range(0, len(r) - n + 1, n)]
        axes[1].plot([np.mean([x["t_min"] for x in c]) for c in cyc], [np.mean([x["speedup"] for x in c]) for c in cyc],
                     marker="o", color=colors[d], label=f"{labels[d]}, g={g}")
    axes[0].set(title="Time of one model pass under continuous load", xlabel="minutes of continuous load",
                ylabel="milliseconds")
    axes[0].set_ylim(0, None)
    axes[1].axhline(1, color=INK2, lw=1, ls="--")
    axes[1].set(title="Speedup at the calibrated g*, averaged per cycle of prompts",
                xlabel="minutes of continuous load", ylabel="speedup (×)")
    axes[1].set_ylim(0.8, None)
    _legend(axes[0], loc="lower right", fontsize=8)
    _legend(axes[1], loc="lower left", fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir, "fig6_thermal.png")


def fig_copy(data, colors, labels, out_dir, g=8):
    """Per prompt: how much of the output is copied from the article vs the
    generation-only speedup at a fixed g. One panel per device."""
    import matplotlib.pyplot as plt
    devs = [d for d in data if "sweep" in data[d] or "language" in data[d]]
    if not devs:
        return None
    fig, axes = plt.subplots(1, len(devs), figsize=(5.6 * len(devs), 4.4), squeeze=False)
    for ax, d in zip(axes[0], devs):
        rows = data[d].get("sweep", {}).get("rows", []) + data[d].get("language", {}).get("rows", [])
        seen, pts = set(), []
        for r in rows:
            key = (r["prompt_id"], r["g"])
            if r["g"] == g and key not in seen and "copy_from_prompt" in r:
                seen.add(key)
                pts.append(r)
        # two colours (English vs Indian languages) + marker shape per language
        shapes = {"english": "o", "hindi": "s", "marathi": "^", "bengali": "D", "tamil": "v"}
        for lang in LANG_ORDER:
            rs = [r for r in pts if r["lang"] == lang]
            if rs:
                ax.scatter([r["copy_from_prompt"] * 100 for r in rs], [r["decode_speedup"] for r in rs], s=60,
                           marker=shapes[lang], color=PALETTE[0] if lang == "english" else PALETTE[1],
                           edgecolor=SURFACE, linewidth=1.2, label=lang.capitalize(), zorder=3, alpha=0.9)
        ax.axhline(1, color=INK2, lw=1, ls="--")
        ax.set(title=labels[d], xlabel="% of the output's 4-token sequences copied from the article",
               ylabel=f"generation-only speedup at g={g} (×)")
        _legend(ax, loc="upper left", fontsize=8)
    fig.suptitle("Guessing pays off exactly as much as the output copies its input (one dot per article)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig7_copying_vs_speedup.png")


def summary_md(data, labels, table, out_path):
    L = ["# Results summary", "", "Generated by `python run.py plots` from every folder in `results/`.", ""]
    L += ["## Devices", "", "| device folder | hardware | precision | threads | RAM (GB) | torch |", "|---|---|---|---|---|---|"]
    for d, exp in data.items():
        i = next(iter(exp.values()))["device"]
        L.append(f"| `{d}` | {i.get('gpu') or i['cpu']} | {i['dtype']} | {i.get('torch_threads')} | "
                 f"{i.get('ram_gb', '')} | {i['torch']} |")
    L += ["", "## Correctness (speculative output identical to normal decoding)", "",
          "| device | exact match |", "|---|---|"]
    for d, exp in data.items():
        if "correctness" in exp:
            L.append(f"| {labels[d]} | {exp['correctness']['exact_match_rate'] * 100:.0f}% |")
    L += ["", "## Draft-size sweep (all prompts)", ""]
    for d, exp in data.items():
        if "sweep" not in exp:
            continue
        a = _agg(exp["sweep"]["rows"])
        L += [f"**{labels[d]}**", "", "| g | τ (tokens/pass) | acceptance | pass time (ms) | tokens/pass | speedup | decode-only speedup |",
              "|---|---|---|---|---|---|---|"]
        for g, v in a.items():
            L.append(f"| {g} | {v['tau']:.2f} | {v['acceptance_rate']:.2f} | {v['mean_pass_sec'] * 1000:.1f} | "
                     f"{v['mean_pass_len']:.1f} | {v['speedup']:.2f}× | {v['decode_speedup']:.2f}× |")
        L.append("")
    L += ["## Calibration (held-out prompts)", "", "| device | method | g* | predicted | measured |", "|---|---|---|---|---|"]
    for d, exp in data.items():
        if "calibration" in exp:
            for n, c in exp["calibration"]["choices"].items():
                L.append(f"| {labels[d]} | {n} | {c['g_star']} | {c['pred_speedup']:.2f}× | {c['heldout_speedup']:.2f}× |")
    if table:
        L += ["", "## Cost of the wrong draft size", "", "| device | own g* | other device's g* | largest g |", "|---|---|---|---|"]
        for t in table:
            others = "; ".join(f"{o}: g={v['g']} → {v['speedup']:.2f}×" for o, v in t["others"].items()) or "—"
            L.append(f"| {t['device']} | g={t['own_g']} → {t['own']:.2f}× | {others} | g={t['largest_g']} → {t['largest']:.2f}× |")
    for d, exp in data.items():
        if "language" not in exp:
            continue
        lg = exp["language"]
        g = exp.get("calibration", {}).get("g_star") or max(lg["grid"])
        L += ["", f"## Languages — {labels[d]} (g = {g})", "",
              "| language | prompt tokens | " + " | ".join(f"tokens/word ({k.split('/')[-1]})" for k in lg["fertility"]) +
              " | τ | acceptance | 4-grams copied from article | 4-grams repeating own output | speedup | generation-only speedup |",
              "|---|---|" + "---|" * len(lg["fertility"]) + "---|---|---|---|---|---|"]
        for l in LANG_ORDER:
            s = lg["summary"][l][str(g)]
            fert = " | ".join(f"{f[l]['tokens_per_word']:.2f}" for f in lg["fertility"].values())
            L.append(f"| {l.capitalize()} | {lg['summary'][l]['prompt_tokens']:.0f} | {fert} | {s['tau']:.2f} | "
                     f"{s['acceptance_rate']:.2f} | {s.get('copy_from_prompt', float('nan')):.0%} | "
                     f"{s.get('self_repeat', float('nan')):.0%} | {s['speedup']:.2f}× | {s['decode_speedup']:.2f}× |")
    for d, exp in data.items():
        if "thermal" in exp:
            r, g = exp["thermal"]["rounds"], exp["thermal"]["g_star"]
            n = len({x["prompt_id"] for x in r})
            cyc = [r[i:i + n] for i in range(0, len(r) - n + 1, n)]
            L += ["", f"## Sustained load — {labels[d]} (g = {g})", "",
                  "| cycle | minutes | 1-token pass (ms) | " + f"{1 + g}-token pass (ms) | normal tok/s | speedup |",
                  "|---|---|---|---|---|---|"]
            for i, c in enumerate(cyc):
                L.append(f"| {i + 1} | {c[0]['t_min']:.1f}–{c[-1]['t_min']:.1f} | "
                         f"{np.median([x['pass_1_sec'] for x in c]) * 1000:.0f} | "
                         f"{np.median([x['pass_g_sec'] for x in c]) * 1000:.0f} | "
                         f"{np.mean([x['base_tok_per_sec'] for x in c]):.1f} | {np.mean([x['speedup'] for x in c]):.2f}× |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"  wrote {out_path}")


def make_all(results_root: str, out_dir: str) -> None:
    _style()
    data = _load(results_root)
    if not data:
        print("no results yet")
        return
    labels = {d: _label(d, e) for d, e in data.items()}
    colors = {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(data)}
    fig_fingerprint(data, colors, labels, out_dir)
    fig_sweep(data, colors, labels, out_dir)
    _, table = fig_wrong_g(data, colors, labels, out_dir)
    fig_calibration(data, colors, labels, out_dir)
    fig_language(data, colors, labels, out_dir)
    fig_thermal(data, colors, labels, out_dir)
    fig_copy(data, colors, labels, out_dir)
    summary_md(data, labels, table, os.path.join(results_root, "SUMMARY.md"))
