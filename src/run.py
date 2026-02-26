"""Train on Hindi, evaluate on seven Indic languages, for both SLMs.

The question is whether a tokenizer's fertility on a script predicts how well
and how cheaply a model fine-tunes for NER in that script. Fertility is
measured up front by `fertility.py`; this script supplies the outcome side.
"""
import argparse
import gc
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
import data
import ner
from config import EVAL_LANGS, MODELS, SEEDS, TRAIN_LANG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--out", default="results")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_grad_ckpt", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows_path = f"{args.out}/ner_raw.csv"
    stats_path = f"{args.out}/train_stats.json"
    rows, stats, done = [], {}, set()
    if args.resume and os.path.exists(rows_path):
        prev = pd.read_csv(rows_path)
        rows = prev.to_dict("records")
        done = {(r["model"], int(r["seed"])) for _, r in prev.iterrows()}
        if os.path.exists(stats_path):
            stats = json.load(open(stats_path))
        print(f"resuming: {len(done)} (model, seed) cells done", flush=True)

    raw_train = data.load(TRAIN_LANG, "train", n=args.n_train)
    raw_eval = {l: data.load(l, "test") for l in EVAL_LANGS}
    print(f"train[{TRAIN_LANG}]={len(raw_train)}  "
          + "  ".join(f"test[{l}]={len(d)}" for l, d in raw_eval.items()), flush=True)

    def dump():
        pd.DataFrame(rows).to_csv(rows_path, index=False)
        json.dump(stats, open(stats_path, "w"), indent=2)

    for key in args.models:
        model_id = MODELS[key]
        for seed in args.seeds:
            if (key, seed) in done:
                print(f"  skip {key} seed {seed}", flush=True)
                continue
            print(f"\n===== {key} | seed {seed} =====", flush=True)

            def one():
                tok = ner.load_tokenizer(model_id)
                tr = ner.encode(raw_train, tok)
                model, st = ner.train(model_id, tr, tok, seed,
                                      out_dir=f"outputs/{key}_{seed}",
                                      epochs=args.epochs,
                                      grad_ckpt=not args.no_grad_ckpt)
                stats[key] = st
                print(f"  trained in {st['train_seconds']}s  "
                      f"tokens={st['train_tokens']:,}  vram={st['peak_vram_gb']}GB",
                      flush=True)
                for lang, ds in raw_eval.items():
                    enc = ner.encode(ds, tok)
                    p, g = ner.predict(model, tok, enc)
                    m = ner.entity_f1(p, g)
                    m.update(model=key, seed=seed, lang=lang,
                             zero_shot=(lang != TRAIN_LANG), n=len(ds),
                             n_train=args.n_train, epochs=args.epochs)
                    rows.append(m)
                    print(f"    {lang}: F1={m['f1']*100:.2f} "
                          f"P={m['precision']*100:.2f} R={m['recall']*100:.2f}", flush=True)

            one()
            gc.collect()
            torch.cuda.empty_cache()
            dump()

    dump()
    print(f"\nwrote {rows_path}", flush=True)


if __name__ == "__main__":
    main()
