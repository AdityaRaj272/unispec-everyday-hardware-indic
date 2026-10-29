"""
Losslessness tests for the decoding engine. No model, no GPU, runs in < 1 s.

The mock "model" is a deterministic function of the tokens actually sitting in
the KV cache. If the cache ever holds a rejected token, its predictions drift
and the output stops matching greedy decoding -- these tests fail loudly then.
(Ported from the RetroSpec v3 test suite.)
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specdec.drafters import NGramDrafter
from specdec.engine import Verifier, greedy_generate, speculative_generate

VOCAB = 50
EOS = 0


def _next_token(ctx):
    """Deterministic pseudo-LM: depends on the last 3 tokens."""
    h = 0
    for t in ctx[-3:]:
        h = (h * 1315423911 + t * 2654435761 + 12345) % 1000003
    return 1 + (h % (VOCAB - 1))


class MockVerifier(Verifier):
    def __init__(self):
        self.cache = []

    def reset(self):
        self.cache = []

    def cache_len(self):
        return len(self.cache)

    def crop(self, n):
        assert n <= len(self.cache), "crop must not extend the cache"
        del self.cache[n:]

    def forward(self, tokens, n_keep):
        base = len(self.cache)
        self.cache.extend(tokens)
        preds = [_next_token(self.cache[: base + j + 1]) for j in range(len(tokens))]
        return preds[-n_keep:]


def reference_greedy(prompt, n):
    seq, out = list(prompt), []
    for _ in range(n):
        t = _next_token(seq)
        seq.append(t)
        out.append(t)
        if t == EOS:
            break
    return out


def make_drafter(rng, hit_rate, gamma):
    """Right `hit_rate` of the time, so accept and reject paths both run."""
    def draft(seq):
        k = rng.randint(0, gamma)
        out, ctx = [], list(seq)
        for _ in range(k):
            t = _next_token(ctx) if rng.random() < hit_rate else rng.randint(1, VOCAB - 1)
            out.append(t)
            ctx.append(t)
        return out
    return draft


def test_baseline_matches_reference():
    for seed in range(20):
        rng = random.Random(seed)
        prompt = [rng.randint(1, VOCAB - 1) for _ in range(rng.randint(1, 30))]
        got = greedy_generate(MockVerifier(), prompt, max_new_tokens=64).token_ids
        assert got == reference_greedy(prompt, 64), f"baseline drift at seed {seed}"


def test_speculative_is_bit_exact():
    for seed in range(60):
        rng = random.Random(seed)
        prompt = [rng.randint(1, VOCAB - 1) for _ in range(rng.randint(1, 40))]
        gamma = rng.randint(1, 16)
        hit = rng.choice([0.0, 0.3, 0.6, 0.9, 1.0])
        res = speculative_generate(MockVerifier(), prompt, draft_fn=make_drafter(rng, hit, gamma),
                                   max_new_tokens=64)
        assert res.token_ids == reference_greedy(prompt, 64), f"mismatch seed={seed} g={gamma} hit={hit}"


def test_ngram_drafter_is_bit_exact():
    for seed in range(30):
        rng = random.Random(seed)
        # repetitive prompt so the n-gram drafter actually fires
        chunk = [rng.randint(1, VOCAB - 1) for _ in range(8)]
        prompt = chunk * 4 + [rng.randint(1, VOCAB - 1) for _ in range(5)]
        for g in (1, 3, 8, 32):
            res = speculative_generate(MockVerifier(), prompt, draft_fn=NGramDrafter(draft_len=g),
                                       max_new_tokens=48)
            assert res.token_ids == reference_greedy(prompt, 48), f"seed={seed} g={g}"


def test_token_budget_never_exceeded():
    for budget in range(1, 20):
        rng = random.Random(budget)
        prompt = [rng.randint(1, VOCAB - 1) for _ in range(10)]
        res = speculative_generate(MockVerifier(), prompt, draft_fn=make_drafter(rng, 1.0, 8),
                                   max_new_tokens=budget)
        assert res.n_tokens <= budget, f"budget {budget} -> {res.n_tokens}"
        assert res.token_ids == reference_greedy(prompt, budget)


def test_perfect_drafter_saves_forward_passes():
    rng = random.Random(0)
    prompt = [rng.randint(1, VOCAB - 1) for _ in range(10)]

    def perfect(seq):
        ctx, out = list(seq), []
        for _ in range(6):
            t = _next_token(ctx)
            out.append(t)
            ctx.append(t)
        return out

    res = speculative_generate(MockVerifier(), prompt, draft_fn=perfect, max_new_tokens=60)
    assert res.token_ids == reference_greedy(prompt, 60)
    assert res.tau > 3.0, res.tau
    assert res.acceptance_rate > 0.95


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
