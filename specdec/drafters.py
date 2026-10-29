"""
Training-free n-gram drafter (prompt-lookup style), adapted from RetroSpec v3.

Given the text so far, find the most recent earlier place where the last few
tokens (an n-gram, n = max_n down to min_n) already appeared, and guess that
whatever followed back then will follow again. Costs no model, no training and
no extra memory. Works when the output re-uses spans of the input (summaries,
answering from a passage); does little on open-ended questions.

`draft_len` is the draft size g -- the knob that device calibration tunes.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


class NGramDrafter:
    """Exact suffix match, longest n first, most recent occurrence wins.
    The index is updated incrementally: O(1) amortised per token."""

    def __init__(self, draft_len: int = 6, max_n: int = 4, min_n: int = 2):
        self.draft_len, self.max_n, self.min_n = draft_len, max_n, min_n
        self.reset()

    def reset(self) -> None:
        self.idx: Dict[int, Dict[Tuple[int, ...], int]] = {
            n: {} for n in range(self.min_n, self.max_n + 1)
        }
        self.pos = 0

    def _sync(self, seq: Sequence[int]) -> None:
        if len(seq) < self.pos:              # reused on a new, shorter sequence
            self.reset()
        # index up to len(seq)-2 so the live suffix can never match itself
        limit = len(seq) - 1
        while self.pos < limit:
            i = self.pos
            for n in range(self.min_n, self.max_n + 1):
                s = i + 1 - n
                if s >= 0:
                    self.idx[n][tuple(seq[s: i + 1])] = i
            self.pos += 1

    def __call__(self, seq: Sequence[int]) -> List[int]:
        self._sync(seq)
        for n in range(self.max_n, self.min_n - 1, -1):
            if len(seq) < n:
                continue
            j = self.idx[n].get(tuple(seq[-n:]))
            if j is not None and j + 1 < len(seq):
                return list(seq[j + 1: j + 1 + self.draft_len])
        return []
