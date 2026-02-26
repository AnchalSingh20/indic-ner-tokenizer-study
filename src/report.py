"""Build the tables in the README from the raw result CSVs."""
import json
import os
import sys

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from config import EVAL_LANGS, TRAIN_LANG

NAMES = {"hi": "Hindi", "mr": "Marathi", "bn": "Bengali", "ta": "Tamil",
         "te": "Telugu", "pa": "Punjabi", "gu": "Gujarati"}
SCRIPTS = {"hi": "Devanagari", "mr": "Devanagari", "bn": "Bengali", "ta": "Tamil",
           "te": "Telugu", "pa": "Gurmukhi", "gu": "Gujarati"}
ORDER = ["hi", "mr", "bn", "pa", "ta", "gu", "te"]


def pm(g):
    g = g.dropna()
    if len(g) < 2:
        return f"{g.iloc[0]*100:.2f}" if len(g) else "–"
    return f"{g.mean()*100:.2f} ± {g.std(ddof=1)*100:.2f}"


def load(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def main(out="results"):
    fert = {r["lang"]: r for r in json.load(open(f"{out}/fertility.json"))}
    main_df = load(f"{out}/ner_raw.csv")
    big_df = load("results_20k/ner_raw.csv")
    md = []

    md.append("### Tokenizer fertility (subword tokens per word)\n")
    rows = []
    for l in ORDER:
        r = fert[l]
        rows.append({"Language": NAMES[l], "Script": SCRIPTS[l],
                     "Phi-4-mini": r["phi-4-mini_fertility"],
                     "Qwen2.5": r["qwen2.5-3b_fertility"],
                     "Ratio": round(r["qwen2.5-3b_fertility"] / r["phi-4-mini_fertility"], 2)})
    md.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")

    for tag, df, note in [("4,000 training sentences, 3 seeds", main_df, "mean ± std"),
                          ("20,000 training sentences, 1 seed", big_df, "single run")]:
        if df is None:
            continue
        md.append(f"\n### Entity F1 (%) — {tag} ({note})\n")
        rows = []
        for l in ORDER:
            s = df[df.lang == l]
            p = s[s.model == "phi-4-mini"]["f1"]
            q = s[s.model == "qwen2.5-3b"]["f1"]
            rows.append({
                "Language": NAMES[l] + ("" if l == TRAIN_LANG else " *(zero-shot)*"),
                "Phi-4-mini": pm(p), "Qwen2.5-3B": pm(q),
                "Gap": round((p.mean() - q.mean()) * 100, 2),
                "Qwen fertility": fert[l]["qwen2.5-3b_fertility"]})
        t = pd.DataFrame(rows)
        md.append(t.to_markdown(index=False) + "\n")
        gaps = [(df[(df.lang == l) & (df.model == "phi-4-mini")]["f1"].mean()
                 - df[(df.lang == l) & (df.model == "qwen2.5-3b")]["f1"].mean()) * 100
                for l in ORDER]
        ferts = [fert[l]["qwen2.5-3b_fertility"] for l in ORDER]
        rho, p_ = spearmanr(ferts, gaps)
        md.append(f"\nSpearman correlation, Qwen fertility vs F1 gap: "
                  f"**rho = {rho:.3f}, p = {p_:.3f}** (n = {len(ORDER)} languages)\n")

    for tag, path in [("4,000 sentences (gradient checkpointing on)", f"{out}/train_stats.json"),
                      ("20,000 sentences (gradient checkpointing off)", "results_20k/train_stats.json")]:
        if not os.path.exists(path):
            continue
        st = json.load(open(path))
        md.append(f"\n### Training cost — {tag}\n")
        rows = [{"Model": k, "Train tokens": f"{v['train_tokens']:,.0f}",
                 "Wall clock (s)": v["train_seconds"], "Tokens/s": v["tokens_per_second"],
                 "Peak VRAM (GB)": v["peak_vram_gb"],
                 "Trainable params": f"{v['trainable']:,}"} for k, v in st.items()]
        md.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")

    text = "\n".join(md)
    open(f"{out}/tables.md", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
