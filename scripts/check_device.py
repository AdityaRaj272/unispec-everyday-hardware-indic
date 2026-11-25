"""Does the model write the same text on this device as on the CPU?

The benchmark's correctness step compares a device only with itself, so it
cannot notice a device that computes garbage. This compares it with the CPU.

    python scripts/check_device.py mps fp16
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specdec.data import encode, load_prompts
from specdec.device import eos_ids_for, load_model, pick_dtype
from specdec.engine import TorchVerifier, greedy_generate

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
device, dtype = sys.argv[1], sys.argv[2]


def generate(dev, dt):
    model, tok = load_model(MODEL, dev, pick_dtype(dev, dt))
    ids = encode(tok, load_prompts("english", n=1)[0])
    out = greedy_generate(TorchVerifier(model), ids, 40, eos_ids_for(tok, model)).token_ids
    return out, tok.decode(out)


want, want_text = generate("cpu", "fp32")
got, got_text = generate(device, dtype)
same = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
print(f"CPU fp32 : {want_text!r}")
print(f"{device} {dtype}: {got_text!r}")
print(f"first {same} of {len(want)} tokens identical")
# half precision may flip a near-tie later on; garbage differs at once
sys.exit(0 if same >= 10 else 1)
