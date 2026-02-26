"""Load naamapadam.

The dataset still ships a loading script (`naamapadam.py`), which `datasets`
3.x+ refuses to execute. HF's automatic Parquet conversion of the same data is
served from the `refs/convert/parquet` branch, so we resolve the file list from
the datasets-server API and load those Parquet files directly. This keeps the
loader working on modern `datasets` without vendoring a copy of the data.
"""
import functools
import json
import os
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from config import DATASET, LABELS

_API = "https://datasets-server.huggingface.co/parquet?dataset={}"


@functools.lru_cache(maxsize=1)
def _parquet_index():
    url = _API.format(urllib.parse.quote(DATASET, safe=""))
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)["parquet_files"]


def parquet_urls(lang, split):
    files = [f for f in _parquet_index() if f["config"] == lang and f["split"] == split]
    if not files:
        raise ValueError(f"no parquet for {lang}/{split}")
    return [f["url"] for f in sorted(files, key=lambda f: f["filename"])]


def load(lang, split, n=None, seed=13):
    """Return a Dataset of {tokens, ner_tags}, optionally a shuffled subset."""
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files={split: parquet_urls(lang, split)},
                      split=split)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def tags_to_labels(tags):
    return [LABELS[t] for t in tags]
