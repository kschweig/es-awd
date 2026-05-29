# Main Tasks

This folder contains the benchmarks used as the main ES/GRPO training task.

Each task folder contains:
- `train.parquet` / `test.parquet` — pre-generated data files consumed directly by the trainers
- `create_parquet_split.py` — script to regenerate the parquet files from the upstream source
- `reward_function.py` — reward signal used during training
- `data.py` — data loading utilities

## Regenerating parquet files

```bash
python tasks/<task>/create_parquet_split.py
```

Run from the repo root. Each script accepts optional arguments (e.g. train size, seed); run with `--help` for details. The default train split size is **200 samples** for all tasks.

## Benchmarks

### `countdown`

Arithmetic construction task: the model must use each provided number exactly once to build an expression with +,-, *, / that results in a target value.

In this repo, Countdown is used as a main training task. The model is trained to produce reasoning in `<think>...</think>` and the final expression in `<answer>...</answer>`. The reward checks both formatting and whether the final expression evaluates to the requested target with the exact provided numbers.

**Data source:** local `countdown.json` (included in the repo).

Example:

```text
Numbers: [44, 19, 35]
Target: 98
One valid solution: ((44 + 19) + 35)
```

### `gsm8k`

Grade-school math word problems with a single numeric final answer.

In this repo, GSM8K is used as a main training task with parquet-backed data. The model is prompted to reason step by step and place the final numeric answer after `####`.

**Data source:** `openai/gsm8k` (HuggingFace, `main` config).

Example:

```text
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: #### 72
```

### `proofwriter`

Logical reasoning benchmark built from short theories and natural-language questions. The model must decide whether a statement is entailed, contradicted, or unknown given the theory.

In this repo, ProofWriter is used as a main training task with local parquet exports so it can also be reused more easily in GRPO-style pipelines. The model is prompted to output one of `True`, `False`, or `Unknown` on the last line in the format `Answer: <label>`.

**Data source:** `tasksource/proofwriter` (HuggingFace).

Example:

```text
Theory: Anne is white. Erin is white. Fiona is blue. If something is white then it is smart. White things are smart. White, smart things are nice.
Question: Anne is not nice.
Answer: False
```
