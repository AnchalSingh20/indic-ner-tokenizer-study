"""Token-classification fine-tuning and entity-level evaluation.

Labels are aligned to the FIRST subword of each word; continuation subwords get
-100. That matters here: the two tokenizers split Indic words into very
different numbers of pieces, and scoring on first-subwords keeps the label set
identical for both models so the comparison stays word-level.
"""
import inspect
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import ID2LABEL, LABEL2ID, LABELS, MAX_LEN


def _dtype_kwarg(dtype=torch.bfloat16):
    from transformers import AutoModelForTokenClassification as A
    try:
        if "dtype" in inspect.signature(A.from_pretrained).parameters:
            return {"dtype": dtype}
    except (TypeError, ValueError):
        pass
    import transformers
    return ({"dtype": dtype} if int(transformers.__version__.split(".")[0]) >= 5
            else {"torch_dtype": dtype})


def encode(ds, tok, max_len=MAX_LEN):
    def fn(batch):
        enc = tok(batch["tokens"], is_split_into_words=True,
                  truncation=True, max_length=max_len)
        labels = []
        for i, tags in enumerate(batch["ner_tags"]):
            wids = enc.word_ids(batch_index=i)
            prev, lab = None, []
            for w in wids:
                if w is None:
                    lab.append(-100)
                elif w != prev:
                    lab.append(tags[w])      # first subword carries the label
                else:
                    lab.append(-100)         # continuation subwords ignored
                prev = w
            labels.append(lab)
        enc["labels"] = labels
        return enc
    return ds.map(fn, batched=True, remove_columns=ds.column_names)


def load_tokenizer(model_id):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    assert tok.is_fast, "need a fast tokenizer for word_ids()"
    return tok


def train(model_id, train_ds, tok, seed, out_dir, epochs=2, lr=1e-4,
          batch_size=8, four_bit=True, lora_r=16, lora_alpha=32,
          grad_ckpt=True):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForTokenClassification, DataCollatorForTokenClassification,
                              Trainer, TrainingArguments, set_seed)
    set_seed(seed)

    kw = dict(num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
              device_map={"": 0}, **_dtype_kwarg())
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            # The classification head must stay unquantized: PEFT's TOKEN_CLS
            # task puts it in modules_to_save, and a 4-bit Linear there fails
            # inside bitsandbytes. It is a 7-way head -- the memory is trivial.
            llm_int8_skip_modules=["score", "classifier"])
    model = AutoModelForTokenClassification.from_pretrained(model_id, **kw)
    model.config.pad_token_id = tok.pad_token_id
    # NOT prepare_model_for_kbit_training: it upcasts every non-quantized
    # parameter to fp32, and this model's ~200k-token embedding alone is 2.4 GB
    # that way -- enough to OOM on a shared GPU. Only LoRA adapters are trained,
    # and they are created in fp32 by PEFT regardless, so the upcast buys
    # nothing here. Gradient checkpointing is enabled directly instead.
    model.config.use_cache = False
    if grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    cfg = LoraConfig(task_type=TaskType.TOKEN_CLS, r=lora_r, lora_alpha=lora_alpha,
                     lora_dropout=0.05, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj",
                                     "qkv_proj", "gate_up_proj"])
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    steps = max(1, math.ceil(len(train_ds) / batch_size) * epochs)
    args = TrainingArguments(
        output_dir=out_dir, per_device_train_batch_size=batch_size,
        num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * steps)), weight_decay=0.01,
        logging_steps=100, save_strategy="no", report_to=[], bf16=True,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs=({"use_reentrant": False} if grad_ckpt else None),
        seed=seed, data_seed=seed,
        optim="paged_adamw_8bit" if four_bit else "adamw_torch")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    Trainer(model=model, args=args, train_dataset=train_ds,
            data_collator=DataCollatorForTokenClassification(tok)).train()
    wall = time.time() - t0
    model.eval()

    # total subword tokens actually processed -- the cost the tokenizer drives
    n_tokens = int(sum(len(x) for x in train_ds["input_ids"])) * epochs
    stats = {"train_seconds": round(wall, 1),
             "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
             "trainable": trainable, "total": total,
             "train_tokens": n_tokens,
             "tokens_per_second": round(n_tokens / wall, 1)}
    return model, stats


@torch.no_grad()
def predict(model, tok, ds, batch_size=16):
    """Return (pred_labels, gold_labels) as lists of per-word label strings."""
    from transformers import DataCollatorForTokenClassification
    coll = DataCollatorForTokenClassification(tok)
    preds, golds = [], []
    for i in range(0, len(ds), batch_size):
        chunk = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        batch = coll(chunk)
        labels = batch.pop("labels")
        batch = {k: v.to(model.device) for k, v in batch.items()}
        logits = model(**batch).logits.float().argmax(-1).cpu()
        for row_logits, row_labels in zip(logits, labels):
            mask = row_labels != -100
            preds.append([LABELS[p] for p in row_logits[mask].tolist()])
            golds.append([LABELS[g] for g in row_labels[mask].tolist()])
    return preds, golds


def entity_f1(preds, golds):
    from seqeval.metrics import f1_score, precision_score, recall_score
    return {"precision": precision_score(golds, preds),
            "recall": recall_score(golds, preds),
            "f1": f1_score(golds, preds)}
