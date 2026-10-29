"""
Device detection, number precision, hardware description and model loading.

Precision choice per device:
  * CPU                      -> fp32 (CPUs have no fast fp16 path)
  * NVIDIA Ampere or newer   -> bf16
  * NVIDIA Volta/Turing (T4) -> fp16 (no bf16 hardware)
  * NVIDIA Pascal P100 (6.0) -> fp16;  other Pascal (e.g. MX150, 6.1) -> fp32,
                                because consumer Pascal runs fp16 at 1/64 rate
  * Apple Silicon (mps)      -> fp16
"""
from __future__ import annotations

import os
import platform
import re
from typing import List, Optional


def pick_device(requested: Optional[str] = None) -> str:
    import torch
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, requested: Optional[str] = None):
    import torch
    table = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if requested and requested != "auto":
        return table[requested]
    if device.startswith("cuda"):
        major, minor = torch.cuda.get_device_capability(0)
        if major >= 8:
            return torch.bfloat16
        if major >= 7 or (major, minor) == (6, 0):
            return torch.float16
        return torch.float32
    if device == "mps":
        return torch.float16
    return torch.float32


def sync_fn(device: str):
    import torch
    if device.startswith("cuda"):
        return torch.cuda.synchronize
    if device == "mps":
        return torch.mps.synchronize
    return lambda: None


def _cpu_name() -> str:
    try:
        import cpuinfo
        name = cpuinfo.get_cpu_info().get("brand_raw")
        if name:
            return name
    except Exception:
        pass
    return platform.processor() or platform.machine()


def describe(device: str, dtype) -> dict:
    """Everything needed to tell two result files apart. No personal data."""
    import torch
    import transformers
    info = {
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": _cpu_name(),
        "cpu_cores": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
        info["physical_cores"] = psutil.cpu_count(logical=False)
        bat = psutil.sensors_battery()
        info["on_battery"] = None if bat is None else (not bat.power_plugged)
    except Exception:
        pass
    if device.startswith("cuda"):
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_capability=f"{p.major}.{p.minor}",
                    gpu_vram_gb=round(p.total_memory / 1024**3, 1))
    elif device == "mps":
        info["gpu"] = "Apple Silicon (MPS)"
    return info


def device_id(info: dict) -> str:
    """Short folder-friendly name, e.g. 'cpu-intel-core-i7-8550u' or 'cuda-tesla-t4'."""
    if info["device"] == "cpu":
        name = info["cpu"]
        name = re.sub(r"\(R\)|\(TM\)|CPU|@.*$|\d+(th|st|nd|rd) Gen", " ", name)
    else:
        name = info.get("gpu") or info["device"]
    kind = info["device"].split(":")[0]
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{kind}-{slug}"


def load_model(name: str, device: str, dtype):
    """Single-device load. Never device_map='auto' (it shards a small model
    across GPUs and adds transfer cost to every pass)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    kw = dict(attn_implementation="sdpa", low_cpu_mem_usage=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kw)
    except TypeError:                                   # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kw)
    model = model.to(device).eval()
    model.config.use_cache = True
    return model, tok


def eos_ids_for(tok, model) -> List[int]:
    ids = set()
    for v in (tok.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple)):
            ids.update(int(x) for x in v)
    for s in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(s)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(i)
    return sorted(ids)
