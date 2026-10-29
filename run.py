"""
One entry point for every experiment, on any device.

  python run.py all                      # correctness, profile, sweep, calibrate, language
  python run.py thermal --minutes 20     # sustained-load stability run (laptops)
  python run.py profile calibrate --tag power-saver   # same device, another condition:
                                                      # re-profile only, tau re-used (~10 min)
  python run.py all --preset quick       # few-minute smoke test
  python run.py plots                    # figures + summary tables from everything in results/

The device (CUDA GPU, Apple GPU or CPU), its number precision and the amount of
work (preset) are picked automatically; override with --device/--dtype/--preset.
Results go to results/<device-name>[__<tag>]/ and finished steps are skipped
unless --force is given, so an interrupted run can simply be restarted.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

STEPS = ["correctness", "profile", "sweep", "calibrate", "language", "thermal", "plots"]
ALL = ["correctness", "profile", "sweep", "calibrate", "language"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="+", choices=STEPS + ["all"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:0 | mps")
    ap.add_argument("--dtype", default="auto", help="auto | fp32 | fp16 | bf16")
    ap.add_argument("--preset", default="auto", help="auto | quick | cpu | gpu")
    ap.add_argument("--tag", default="", help="label for a condition, e.g. power-saver, on-battery")
    ap.add_argument("--minutes", type=float, default=None, help="duration of the thermal run")
    ap.add_argument("--extra-tokenizers", nargs="*", default=["sarvamai/sarvam-1"],
                    help="other tokenizers to compare tokens-per-word with (language step)")
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--force", action="store_true", help="re-run steps that already have results")
    args = ap.parse_args()

    steps = [s for s in (ALL if "all" in args.steps else []) + args.steps if s != "all"]
    steps = list(dict.fromkeys(steps))                       # de-duplicate, keep order
    model_steps = [s for s in steps if s != "plots"]

    if model_steps:
        run_model_steps(args, model_steps)
    if "plots" in steps:
        from specdec import plots
        plots.make_all(args.results, os.path.join(HERE, "figures"))


def run_model_steps(args, steps):
    import torch
    from specdec import experiments as ex
    from specdec.device import describe, device_id, eos_ids_for, load_model, pick_device, pick_dtype, sync_fn

    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    preset = args.preset if args.preset != "auto" else ("cpu" if device == "cpu" else "gpu")
    info = describe(device, dtype)
    info["model"] = args.model
    info["preset"] = preset
    name = device_id(info) + (f"__{args.tag}" if args.tag else "")
    if preset == "quick":
        name += "__quick"
    out_dir = os.path.join(args.results, name)
    print(f"device: {name}  ({info.get('gpu') or info['cpu']}, {info['dtype']}, preset={preset})")
    print(f"results -> {out_dir}")

    t = time.time()
    model, tok = load_model(args.model, device, dtype)
    print(f"model loaded in {time.time() - t:.0f}s")
    ctx = ex.Ctx(model=model, tok=tok, verifier=ex.TorchVerifier(model), eos=eos_ids_for(tok, model),
                 sync=sync_fn(device), info=info, out_dir=out_dir, cfg=dict(ex.PRESETS[preset]),
                 results_root=args.results)

    for step in steps:
        if step != "thermal" and ctx.load(step if step != "calibrate" else "calibration") and not args.force:
            print(f"\n== {step}: already done, skipping (use --force to redo)")
            continue
        print(f"\n== {step}")
        t = time.time()
        if step == "correctness":
            ex.correctness(ctx)
        elif step == "profile":
            ex.profile(ctx)
        elif step == "sweep":
            ex.sweep(ctx)
        elif step == "calibrate":
            ex.calibrate(ctx)
        elif step == "language":
            ex.language(ctx, args.extra_tokenizers)
        elif step == "thermal":
            ex.thermal(ctx, args.minutes)
        print(f"   ({(time.time() - t) / 60:.1f} min)")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
