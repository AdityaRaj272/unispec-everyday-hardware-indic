"""
Build the prompt set used by every experiment: news articles from XL-Sum
(BBC news, CC BY-NC-SA 4.0, Hasan et al., 2021) in English, Hindi, Marathi,
Bengali and Tamil. Summarisation is a "grounded" task: the summary re-uses
phrases from the article, which is what an n-gram drafter needs.

Each article is cut to its first N words so every language gets a comparable
amount of text; the token count then differs by language, which is exactly
the tokenizer effect the language study measures.

Run once:  python scripts/prepare_data.py      -> data/xlsum_subset.jsonl
"""
import argparse
import json
import os
import random

LANGS = ["english", "hindi", "marathi", "bengali", "tamil"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=40)
    ap.add_argument("--words", type=int, default=250)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "xlsum_subset.jsonl"))
    args = ap.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download

    rows = []
    for lang in LANGS:
        path = hf_hub_download("csebuetnlp/xlsum", f"{lang}/test/0000.parquet",
                               repo_type="dataset", revision="refs/convert/parquet")
        df = pd.read_parquet(path)
        df["n_words"] = df["text"].str.split().str.len()
        df = df[df["n_words"] >= args.words + 50].reset_index(drop=True)   # long enough to cut
        idx = random.Random(0).sample(range(len(df)), args.per_lang)
        for k, i in enumerate(idx):
            r = df.iloc[i]
            rows.append({
                "id": f"{lang[:2]}-{k:02d}",
                "lang": lang,
                "split": "calib" if k % 2 == 0 else "eval",
                "text": " ".join(r["text"].split()[: args.words]),
                "source_id": r["id"],
                "url": r["url"],
            })
        print(f"{lang:8s} {len(df):5d} usable articles, kept {args.per_lang}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} prompts -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
