# Prior Tasks

This folder contains evaluation-only benchmarks used to measure retention and forgetting while the model is being trained on a main task.

Each task folder contains:
- One or more `.parquet` files — pre-stored evaluation data consumed directly by the evaluators
- `reward_function.py` — evaluation reward signal
- `data.py` — data loading utilities
- `create_parquet_exports.py` *(arc_challenge, piqa only)* — script to regenerate parquet files from the upstream HF dataset

For **hellaswag** and **mmlu_pro** no generation script is provided; their parquet files were downloaded directly from HuggingFace and are stored in the repo.

## Benchmarks

### `hellaswag`

Commonsense completion: choose the most plausible ending for a short situation.

In this repo, HellaSwag is used as a prior-task evaluation benchmark. The local parquet file stores the validation split, and the model is asked to reply with only the option number `0`, `1`, `2`, or `3`.

**Data source:** `Rowan/hellaswag` (HuggingFace), validation split. Parquet downloaded directly.

Example:

```text
Activity: Roof shingle removal
Context: A man is sitting on a roof. he
Options:
0. is using wrap to wrap a pair of skis.
1. is ripping level tiles off.
2. is holding a rubik's cube.
3. starts pulling up roofing on a roof.
Correct answer: 3
```

### `piqa`

Physical commonsense reasoning: choose which of two candidate solutions is more likely to achieve a practical goal.

In this repo, PiQA is used as a prior-task evaluation benchmark. The local parquet export stores the labeled validation split, and labels are represented as `1` or `2` to match the prompt used by the reward function.

**Data source:** `piqa` (HuggingFace), validation split. Regenerate with `create_parquet_exports.py`.

Example:

```text
Goal: How do I ready a guinea pig cage for its new occupants?
1. Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, and include a water bottle and food dish.
2. Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, and include a water bottle and food dish.
Correct answer: 1
```

### `arc_challenge`

Science multiple-choice questions from AI2's ARC challenge set.

In this repo, ARC-Challenge is used as a prior-task evaluation benchmark. The local parquet exports normalize answer labels to letters, and the current evaluation is zero-shot with the model asked to finish with `The answer is (X)`.

**Data source:** `allenai/ai2_arc`, `ARC-Challenge` config (HuggingFace). Regenerate with `create_parquet_exports.py`.

Example:

```text
Question: An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?
Options:
A. Planetary density will decrease.
B. Planetary years will become longer.
C. Planetary days will become shorter.
D. Planetary gravity will become stronger.
Correct answer: C
```

### `mmlu_pro`

Broad professional and academic multiple-choice knowledge benchmark.

In this repo, MMLU-Pro is used as a prior-task evaluation benchmark. The local parquet exports include validation and test splits, and the prompt format asks the model to end with `The answer is (X)`.

**Data source:** `TIGER-Lab/MMLU-Pro` (HuggingFace), validation and test splits. Parquet downloaded directly.

Example:

```text
Category: math
Question: Statement 1 | A ring homomorphism is one to one if and only if the kernel is {0}. Statement 2 | Q is an ideal in R.
Options:
A. True, False
B. Not Given, Not Given
C. False, False
D. Not Given, True
E. True, Not Given
F. Not Given, False
G. True, True
H. False, True
I. False, Not Given
Correct answer: H
```
