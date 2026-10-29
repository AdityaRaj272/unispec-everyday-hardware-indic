"""
Lossless greedy speculative decoding engine.

Adapted from the RetroSpec v3 engine. The key state invariant:

    cache holds seq[: len(seq) - len(pending)]
    pending = tokens that are in `seq` but NOT yet in the KV cache

After every step `pending == [seq[-1]]`, so the KV cache never contains a
rejected draft token. That is what makes speculative output identical to
plain greedy decoding.

Changes for this project:
  * the verifier only computes logits for the positions that are actually
    needed (the last draft_len + 1), which matters a lot on CPUs where the
    151k-wide output projection over a whole prompt is expensive;
  * every verifier forward pass is timed individually, so the pass-time curve
    s(g) can be read straight out of real decoding runs (device calibration);
  * prefill time is recorded separately from decode time.

The driver loop is backend-agnostic (see `Verifier`) so it is unit-tested
against a mock model without any model download. See tests/test_engine.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence


# --------------------------------------------------------------------------
# Verifier backends
# --------------------------------------------------------------------------

class Verifier:
    """Protocol.

    forward(tokens, n_keep) -> preds
        `tokens` are appended to the cache. Returns the greedy next-token
        prediction for each of the LAST `n_keep` positions of `tokens`.
    crop(n)      -> truncate the cache to exactly n positions
    cache_len()  -> current number of cached positions
    reset()      -> drop the cache
    """

    def forward(self, tokens: Sequence[int], n_keep: int) -> List[int]: ...
    def crop(self, n: int) -> None: ...
    def cache_len(self) -> int: ...
    def reset(self) -> None: ...


class TorchVerifier(Verifier):
    """HuggingFace causal-LM backend."""

    def __init__(self, model, device=None):
        import torch

        self.torch = torch
        self.model = model
        self.device = device or next(model.parameters()).device
        self.cache = None
        self._keep_kw = "logits_to_keep"      # transformers >= 4.50
        self._supports_cache_position = True

    def reset(self) -> None:
        self.cache = None

    def cache_len(self) -> int:
        return _cache_len(self.cache)

    def crop(self, n: int) -> None:
        self.cache = _crop_cache(self.cache, n)

    def forward(self, tokens: Sequence[int], n_keep: int) -> List[int]:
        torch = self.torch
        base = self.cache_len()
        ids = torch.tensor([list(tokens)], dtype=torch.long, device=self.device)
        total = base + ids.shape[1]

        kwargs = dict(
            input_ids=ids,
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
            attention_mask=torch.ones((1, total), dtype=torch.long, device=self.device),
        )
        if self._supports_cache_position:
            kwargs["cache_position"] = torch.arange(base, total, device=self.device)
        if self._keep_kw:
            kwargs[self._keep_kw] = n_keep

        with torch.inference_mode():
            out = self._call(kwargs, n_keep)

        self.cache = out.past_key_values
        logits = out.logits[0, -n_keep:]     # also correct if all logits came back
        return logits.argmax(dim=-1).tolist()

    def _call(self, kwargs, n_keep):
        # Fall back gracefully on older transformers versions.
        while True:
            try:
                return self.model(**kwargs)
            except TypeError as e:
                msg = str(e)
                if self._keep_kw and self._keep_kw in msg:
                    kwargs.pop(self._keep_kw)
                    self._keep_kw = "num_logits_to_keep" if self._keep_kw == "logits_to_keep" else None
                    if self._keep_kw:
                        kwargs[self._keep_kw] = n_keep
                    continue
                if self._supports_cache_position and "cache_position" in msg:
                    self._supports_cache_position = False
                    kwargs.pop("cache_position", None)
                    continue
                raise


# --------------------------------------------------------------------------
# Cache helpers (robust across transformers versions)
# --------------------------------------------------------------------------

def _cache_len(cache) -> int:
    if cache is None:
        return 0
    get_len = getattr(cache, "get_seq_length", None)
    if callable(get_len):
        try:
            return int(get_len())
        except Exception:
            pass
    layers = getattr(cache, "layers", None)
    if layers:
        keys = getattr(layers[0], "keys", None)
        if keys is not None:
            return int(keys.shape[-2])
    kc = getattr(cache, "key_cache", None)
    if kc:
        return int(kc[0].shape[-2])
    if isinstance(cache, (tuple, list)) and cache:
        return int(cache[0][0].shape[-2])
    return 0


def _crop_cache(cache, n: int):
    if cache is None:
        return None
    if _cache_len(cache) <= n:
        return cache

    crop = getattr(cache, "crop", None)
    if callable(crop):
        try:
            crop(n)
            return cache
        except Exception:
            pass

    layers = getattr(cache, "layers", None)
    if layers:
        for layer in layers:
            if getattr(layer, "keys", None) is not None:
                layer.keys = layer.keys[:, :, :n, :]
                layer.values = layer.values[:, :, :n, :]
        return cache

    kc = getattr(cache, "key_cache", None)
    if kc is not None:
        for i in range(len(kc)):
            cache.key_cache[i] = cache.key_cache[i][:, :, :n, :]
            cache.value_cache[i] = cache.value_cache[i][:, :, :n, :]
        return cache

    if isinstance(cache, (tuple, list)):
        return tuple((k[:, :, :n, :], v[:, :, :n, :]) for k, v in cache)
    return cache


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass
class GenResult:
    token_ids: List[int]                 # newly generated tokens only
    latency_sec: float = 0.0             # prefill + decode
    prefill_sec: float = 0.0
    forward_passes: int = 0              # decode passes (prefill excluded)
    drafted: int = 0
    accepted: int = 0                    # drafted tokens that survived verification
    draft_sec: float = 0.0
    pass_sec: List[float] = field(default_factory=list)    # per decode pass
    pass_len: List[int] = field(default_factory=list)      # tokens fed per decode pass

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def decode_sec(self) -> float:
        return self.latency_sec - self.prefill_sec

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tau(self) -> float:
        """Average tokens gained per decode pass (UniSpec's tau). Baseline == 1."""
        decoded = self.n_tokens - 1          # the first token comes from prefill
        return decoded / self.forward_passes if self.forward_passes else 1.0

    def to_dict(self) -> dict:
        n = len(self.pass_sec)
        return {
            "n_tokens": self.n_tokens,
            "latency_sec": self.latency_sec,
            "prefill_sec": self.prefill_sec,
            "decode_sec": self.decode_sec,
            "forward_passes": self.forward_passes,
            "tau": self.tau,
            "drafted": self.drafted,
            "accepted": self.accepted,
            "acceptance_rate": self.acceptance_rate,
            "draft_sec": self.draft_sec,
            "mean_pass_sec": sum(self.pass_sec) / n if n else 0.0,
            "mean_pass_len": sum(self.pass_len) / n if n else 0.0,
        }


# --------------------------------------------------------------------------
# The generators
# --------------------------------------------------------------------------

def greedy_generate(verifier: Verifier, prompt_ids: Sequence[int], max_new_tokens: int = 128,
                    eos_ids: Iterable[int] = (), sync: Optional[Callable[[], None]] = None) -> GenResult:
    """Plain greedy decoding through the *same* loop as the speculative
    method, so speedups compare like with like."""
    return speculative_generate(verifier, prompt_ids, draft_fn=None,
                                max_new_tokens=max_new_tokens, eos_ids=eos_ids, sync=sync)


def speculative_generate(
    verifier: Verifier,
    prompt_ids: Sequence[int],
    draft_fn: Optional[Callable[[List[int]], Sequence[int]]] = None,
    max_new_tokens: int = 128,
    eos_ids: Iterable[int] = (),
    sync: Optional[Callable[[], None]] = None,
) -> GenResult:
    """Lossless greedy speculative decoding. draft_fn=None gives plain greedy."""
    eos = set(eos_ids or ())
    verifier.reset()

    seq: List[int] = list(prompt_ids)
    n_prompt = len(seq)
    pending: List[int] = list(seq)

    r = GenResult(token_ids=[])
    if sync:
        sync()
    t0 = time.perf_counter()
    first = True

    while len(seq) - n_prompt < max_new_tokens:
        remaining = max_new_tokens - (len(seq) - n_prompt)

        # ---- 1. draft (never during prefill: there is no output yet) ---------
        draft: List[int] = []
        if draft_fn is not None and remaining > 1 and not first:
            td = time.perf_counter()
            draft = list(draft_fn(seq))[: remaining - 1]
            r.draft_sec += time.perf_counter() - td

        # ---- 2. one verifier forward over pending + draft --------------------
        tokens_in = pending + draft
        P, K = len(pending), len(draft)
        base = verifier.cache_len()
        tp = time.perf_counter()
        preds = verifier.forward(tokens_in, n_keep=K + 1)    # preds[i] follows tokens_in[P-1+i]
        dt = time.perf_counter() - tp                         # .tolist() inside forward syncs
        if first:
            r.prefill_sec = time.perf_counter() - t0
            first = False
        else:
            r.forward_passes += 1
            r.pass_sec.append(dt)
            r.pass_len.append(len(tokens_in))
            r.drafted += K

        # ---- 3. longest matching prefix ----------------------------------------
        n = 0
        while n < K and draft[n] == preds[n]:
            n += 1

        accepted = draft[:n] + [preds[n]]     # preds[n]: bonus token, or the correction
        r.accepted += n
        if n < K:
            verifier.crop(base + P + n)       # rejected tokens leave the cache

        seq.extend(accepted)
        pending = [seq[-1]]                   # invariant restored

        hit_eos = False
        for i, t in enumerate(accepted):
            if t in eos:
                del seq[len(seq) - len(accepted) + i + 1:]
                hit_eos = True
                break
        if hit_eos:
            break

    if sync:
        sync()
    r.latency_sec = time.perf_counter() - t0
    r.token_ids = seq[n_prompt:]
    return r
