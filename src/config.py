"""Shared configuration."""

DATASET = "ai4bharat/naamapadam"

# Two SLMs of comparable size from different families, so the variable under
# study is the tokenizer and pretraining mix rather than parameter count.
MODELS = {
    "phi-4-mini": "microsoft/Phi-4-mini-instruct",    # 3.84B
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",         # 3.09B
}

TRAIN_LANG = "hi"
# Hindi is the training language; the rest are evaluated zero-shot to test
# whether what transfers is the label scheme or the script.
EVAL_LANGS = ["hi", "mr", "bn", "ta", "te", "pa", "gu"]

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

MAX_LEN = 512   # at 512 neither tokenizer truncates any sentence (0.00%),
                # so sequence-length differences are a cost, not a confound
SEEDS = [42, 43, 44]
