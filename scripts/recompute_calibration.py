"""
Recompute calibration.json for every results folder from its stored sweep and
profile. No model is run, so it works for any device's results on any machine.

    python scripts/recompute_calibration.py
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from specdec import experiments as ex  # noqa: E402

for d in sorted(glob.glob(os.path.join(ROOT, "results", "*", ""))):
    if not all(os.path.exists(os.path.join(d, f)) for f in ("sweep.json", "profile.json")):
        continue
    with open(os.path.join(d, "sweep.json"), encoding="utf-8") as f:
        sw = json.load(f)
    print(os.path.basename(os.path.normpath(d)))
    ex.calibrate(ex.Ctx(model=None, tok=None, verifier=None, eos=[], sync=None, info=sw["device"],
                        out_dir=d, cfg=sw["config"], results_root=os.path.join(ROOT, "results")))
