"""Prompt loading. Every prompt is 'summarise this article' in its own language."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "xlsum_subset.jsonl")

# The system message keeps a small model answering in the article's language
# (without it, a 0.5B model summarises Tamil articles in English).
SYSTEM = "You are a helpful assistant. Always answer in {Lang} only."
INSTRUCTION = "Summarize this {Lang} news article in {Lang} in 4-5 sentences.\n\n{text}"


def load_prompts(lang: str = "english", split: Optional[str] = None, n: Optional[int] = None,
                 path: str = DATA) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["lang"] == lang and (split is None or r["split"] == split):
                rows.append(r)
    return rows[:n] if n else rows


def encode(tok, row: Dict) -> List[int]:
    lang = row["lang"].capitalize()
    msgs = [{"role": "system", "content": SYSTEM.format(Lang=lang)},
            {"role": "user", "content": INSTRUCTION.format(Lang=lang, text=row["text"])}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if isinstance(ids, dict) or hasattr(ids, "keys"):      # BatchEncoding in newer transformers
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(t) for t in ids]
