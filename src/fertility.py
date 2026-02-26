"""Measure tokenizer fertility on each Indic language.

Fertility = subword tokens produced per whitespace word. A tokenizer whose
vocabulary barely covers Devanagari or Tamil script falls back to near
byte-level pieces, which inflates sequence length, training cost and the
number of subwords an entity is split across. This is measured before any
training because it is the study's independent variable.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import EVAL_LANGS, MODELS
import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="sentences per language")
    ap.add_argument("--out", default="results/fertility.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    toks = {k: AutoTokenizer.from_pretrained(v) for k, v in MODELS.items()}
    rows = []
    for lang in EVAL_LANGS:
        ds = data.load(lang, "validation", n=args.n)
        sents = [r for r in ds["tokens"]]
        n_words = sum(len(s) for s in sents)
        n_chars = sum(len(" ".join(s)) for s in sents)
        row = {"lang": lang, "n_sentences": len(sents), "n_words": n_words}
        for name, tok in toks.items():
            n_sub = sum(len(tok(s, is_split_into_words=True,
                                add_special_tokens=False)["input_ids"]) for s in sents)
            row[f"{name}_tokens"] = n_sub
            row[f"{name}_fertility"] = round(n_sub / n_words, 3)
            row[f"{name}_chars_per_token"] = round(n_chars / n_sub, 3)
        rows.append(row)
        print(f"  {lang}: " + "  ".join(
            f"{k}={row[f'{k}_fertility']:.2f} tok/word" for k in MODELS), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
