### Tokenizer fertility (subword tokens per word)

| Language   | Script     |   Phi-4-mini |   Qwen2.5 |   Ratio |
|:-----------|:-----------|-------------:|----------:|--------:|
| Hindi      | Devanagari |        2.063 |     4.02  |    1.95 |
| Marathi    | Devanagari |        2.717 |     5.173 |    1.9  |
| Bengali    | Bengali    |        2.569 |     5.039 |    1.96 |
| Punjabi    | Gurmukhi   |        2.812 |     6.638 |    2.36 |
| Tamil      | Tamil      |        3.258 |     7.257 |    2.23 |
| Gujarati   | Gujarati   |        2.674 |     7.554 |    2.82 |
| Telugu     | Telugu     |        3.147 |     9.19  |    2.92 |


### Entity F1 (%) — 4,000 training sentences, 3 seeds (mean ± std)

| Language               | Phi-4-mini   | Qwen2.5-3B   |   Gap |   Qwen fertility |
|:-----------------------|:-------------|:-------------|------:|-----------------:|
| Hindi                  | 36.20 ± 0.23 | 33.38 ± 0.05 |  2.81 |            4.02  |
| Marathi *(zero-shot)*  | 29.76 ± 0.96 | 28.00 ± 1.02 |  1.75 |            5.173 |
| Bengali *(zero-shot)*  | 22.07 ± 0.79 | 25.16 ± 0.46 | -3.09 |            5.039 |
| Punjabi *(zero-shot)*  | 14.65 ± 0.53 | 8.97 ± 0.79  |  5.68 |            6.638 |
| Tamil *(zero-shot)*    | 16.02 ± 1.84 | 10.11 ± 0.51 |  5.91 |            7.257 |
| Gujarati *(zero-shot)* | 27.76 ± 0.97 | 18.71 ± 0.53 |  9.05 |            7.554 |
| Telugu *(zero-shot)*   | 31.33 ± 1.88 | 15.47 ± 0.65 | 15.86 |            9.19  |


Spearman correlation, Qwen fertility vs F1 gap: **rho = 0.893, p = 0.007** (n = 7 languages)


### Entity F1 (%) — 20,000 training sentences, 1 seed (single run)

| Language               |   Phi-4-mini |   Qwen2.5-3B |   Gap |   Qwen fertility |
|:-----------------------|-------------:|-------------:|------:|-----------------:|
| Hindi                  |        41.29 |        37.38 |  3.91 |            4.02  |
| Marathi *(zero-shot)*  |        30.83 |        27.66 |  3.17 |            5.173 |
| Bengali *(zero-shot)*  |        24.81 |        28.64 | -3.84 |            5.039 |
| Punjabi *(zero-shot)*  |        15.26 |        10.25 |  5.01 |            6.638 |
| Tamil *(zero-shot)*    |        17.5  |        10.56 |  6.95 |            7.257 |
| Gujarati *(zero-shot)* |        29.94 |        23.25 |  6.69 |            7.554 |
| Telugu *(zero-shot)*   |        34.31 |        16.3  | 18.01 |            9.19  |


Spearman correlation, Qwen fertility vs F1 gap: **rho = 0.857, p = 0.014** (n = 7 languages)


### Training cost — 4,000 sentences (gradient checkpointing on)

| Model      |   Train tokens |   Wall clock (s) |   Tokens/s |   Peak VRAM (GB) |   Trainable params |
|:-----------|---------------:|-----------------:|-----------:|-----------------:|-------------------:|
| phi-4-mini |        367,284 |           1710.1 |      214.8 |             6.14 |         23,090,183 |
| qwen2.5-3b |        716,016 |           1398   |      512.2 |             8.33 |         29,947,911 |


### Training cost — 20,000 sentences (gradient checkpointing off)

| Model      |   Train tokens |   Wall clock (s) |   Tokens/s |   Peak VRAM (GB) |   Trainable params |
|:-----------|---------------:|-----------------:|-----------:|-----------------:|-------------------:|
| phi-4-mini |      1,855,840 |            992.4 |     1870.1 |            13.18 |         23,090,183 |
| qwen2.5-3b |      3,617,844 |           1479.8 |     2444.9 |            27.29 |         29,947,911 |
