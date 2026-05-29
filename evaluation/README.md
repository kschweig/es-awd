# Evaluation

This folder contains scripts and notebooks for evaluating ES and GRPO fine-tuning runs: measuring task performance over training steps, parameter drift, and KL divergence against the original model.

Most scripts support two run types and auto-detect which one they are looking at:

**ES run layout:**
```text
<run_dir>/
  args.json
  iteration_updates/
    iteration_*.json
  <dataset>.jsonl          # written by evaluation scripts
```

**GRPO run layout:**
```text
<run_dir>/
  model.txt                # base model HF name, e.g. Qwen/Qwen2.5-3B-Instruct
  <experiment>_global_step_<N>/
    config.json
    model.safetensors
    tokenizer.json
    ...
  <dataset>.jsonl          # written by evaluation scripts
```

## Files

### `utils.py`

Shared helpers used by all evaluation scripts.

- `ESEvalLLM`: thin `vllm.LLM` wrapper for replay/evaluation.
- `EvalWorkerExtension`: worker-side RPC helpers for snapshotting weights, replaying seeded updates, and dumping weights to disk.
- `resolve_run_paths`, `get_iteration_files`, `load_json`, `load_text`: run/file loading helpers.
- `detect_run_backend`, `resolve_grpo_run_paths`, `get_grpo_checkpoint_dirs`, `iter_hf_checkpoint_weights`: GRPO detection and checkpoint iteration.
- `generate_rollouts`, `count_generated_tokens`, `should_evaluate`, `append_jsonl`: evaluation and logging helpers.

### `evaluate_forgetting.py`

Evaluates a saved run over time on a chosen dataset.

- Measures whether the model forgets old capabilities while being trained on a new task.
- Auto-detects ES vs. GRPO run type.
- ES: replays saved updates from `iteration_updates/` into a base model loaded in vLLM.
- GRPO: reads the base model from `model.txt`, evaluates it as step 0, then loads each `*_global_step_<N>` checkpoint directly.
- Writes one JSONL record per evaluated step to `<run_dir>/<dataset>.jsonl`.

Supported datasets: `countdown`, `gsm8k`, `proofwriter`, `hellaswag`, `piqa`, `arc-challenge`, `mmlu-pro`.

### `evaluate_kl_divergence.py`

Estimates KL divergence between the original model and later checkpoints on fixed continuations generated once by the original model.

- Auto-detects ES vs. GRPO run type (same replay logic as `evaluate_forgetting.py`).
- Generates reference continuations from the base model, then measures per-step log-ratio divergence.
- Writes metrics to `<run_dir>/<dataset>_kl.jsonl`.

### `evaluate_weight_change.py`

Measures how far the model weights have drifted from the original at each training step.

- Auto-detects ES vs. GRPO run type.
- Computes global Frobenius norm, per-layer Frobenius norm (embedding, transformer layers, `lm_head`), relative Frobenius norm, and mean/max absolute difference.
- Writes metrics to `<run_dir>/weight_change.jsonl`.

### `run_forgetting.sh`

Runs `evaluate_forgetting.py` for all standard datasets on one or more run directories.

```bash
RUN_DIR=<path> DEVICE_ID=<gpu> bash evaluation/run_forgetting.sh [extra args...]
```

Environment variables:
- `RUN_DIR` — target run directory (default: a Qwen3B countdown GRPO run)
- `DEVICE_ID` — CUDA device(s) to use
- `SUBSET_SIZE` — max samples for the larger prior-task datasets (default: 2048)
- `SKIP_EXISTING=1` — skip datasets whose `.jsonl` already exists

Pass `--print-output-files` to list expected output paths without running anything.

### `run_kl_divergence.sh`

Runs `evaluate_kl_divergence.py` for all standard datasets across one run or a whole folder tree.

```bash
bash evaluation/run_kl_divergence.sh <folder> <gpu> [extra args...]
```

- If `<folder>` is already an ES or GRPO run, evaluates that run only.
- Otherwise searches recursively for eligible run subdirectories.
- Environment: `DRY_RUN=1`, `SKIP_EXISTING=1`, `SUBSET_SIZE` (default: 50).

### `run_weight_changes.sh`

Runs `evaluate_weight_change.py` across one run or a whole folder tree.

```bash
bash evaluation/run_weight_changes.sh <folder> <gpu> [extra args...]
```

- Same auto-detection logic as `run_kl_divergence.sh`.
- Environment: `DRY_RUN=1`.

### `plot_main_figure.ipynb`

Generates the main paper figure: train-task and prior-task accuracy curves for a single training run.

Configure `EXPERIMENT_DIR` and `TRAIN_TASK`. Reads `<dataset>.jsonl` files produced by `evaluate_forgetting.py`.

### `plot_forgetting_three_tasks.ipynb`

Plots train-task and prior-task accuracy curves from the JSONL files produced by `evaluate_forgetting.py`.
Lays out curves for multiple training tasks (countdown, gsm8k, proofwriter) side by side in one figure.

Configure `EXPERIMENT_DIR` and `PANEL_TASKS`.

### `plot_kl_heatmap.ipynb`

Plots a heatmap of `token_weighted_kl_divergence` across tasks and runs.

Reads `<dataset>_kl.jsonl` files produced by `evaluate_kl_divergence.py`. Configure `EXPERIMENT_DIR`.

### `plot_weight_frobenius.ipynb`

Plots the global Frobenius norm from `weight_change.jsonl` over iterations across one or more runs.

Configure a list of run specs (each with a `path` and `label`) to compare parameter drift trajectories across ES or GRPO runs.

### `plot_lambda_ablation.ipynb`

Plots forgetting/performance curves for a lambda (KL coefficient) ablation, comparing multiple runs with different lambda values on the same axes.

Configure `EXPERIMENT_DIR` and the run configs.

### `plot_influence.ipynb`

Discovers ES and GRPO run directories and builds a per-run influence summary showing how much each prior task was affected relative to the training task.

Auto-discovers runs under the repo's ES and GRPO output directories.

### `create_latex_table.ipynb`

Generates a LaTeX table of accuracy deltas (`final - original`) per task and run.

Configure `RUN_ROWS` with run paths and labels. The `Average` column excludes the training task for each row.

### `__init__.py`

Marks `evaluation/` as a Python package.

## Typical Workflow

1. After training, run `evaluate_forgetting.py` (or `run_forgetting.sh`) to write per-step task metrics.
2. Run `evaluate_kl_divergence.py` (or `run_kl_divergence.sh`) to measure distributional shift.
3. Run `evaluate_weight_change.py` (or `run_weight_changes.sh`) to log parameter drift.
4. Open the notebooks to produce figures:
   - `plot_main_figure.ipynb` or `plot_forgetting_three_tasks.ipynb` for accuracy curves
   - `plot_kl_heatmap.ipynb` for KL divergence heatmaps
   - `plot_weight_frobenius.ipynb` for global weight drift
   - `plot_lambda_ablation.ipynb` for hyperparameter ablations
   - `create_latex_table.ipynb` for the summary table

## Notes

- All scripts require vLLM and GPU execution.
- ES scripts inherit model name, precision, task, and CUDA devices from `<run_dir>/args.json`.
- For GRPO runs, the base model is read from `<run_dir>/model.txt` and precision is inferred from the exported checkpoint config unless overridden on the CLI.
