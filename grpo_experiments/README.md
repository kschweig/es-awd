# GRPO Experiments

Training runs for the catastrophic forgetting project using GRPO (Group Relative Policy Optimization) via [verl](https://github.com/volcengine/verl).

Follow the installation instructions of verl as outlined in their project docs.

## Structure

```
grpo_experiments/
├── checkpoints/          # Raw FSDP checkpoints written during training
├── merged/               # HF-format models produced by merge_fsdp_to_hf.sh
├── grpo_<model>_<task>_s<seed>.sh   # One launch script per experiment
├── merge_fsdp_to_hf.sh
└── delete_optim_checkpoints.sh
```

## Naming convention

Scripts follow `grpo_<model>_<task>_s<seed>.sh`, e.g.:

| Segment | Examples |
|---------|---------|
| model   | `qwen_1.5b`, `qwen_3b`, `qwen_7b`, `llama_3b` |
| task    | `countdown`, `proofwriter`, `gsm8k` |
| seed    | `s42`, `s43`, `s44` |

## Running an experiment

```bash
# from the repo root
bash grpo_experiments/grpo_qwen_3b_countdown_s42.sh
```

Each script starts a Ray head node, then launches `verl.trainer.main_ppo` with GRPO.
Logs are written to a `.log` file in the current directory.
Checkpoints land under `grpo_experiments/checkpoints/`.

If running experiments under mixed usage of ray, e.g. using 4 GPUs for an ES run and 4 GPUs of a node for a GRPO run, one may set the ray port differently for the GRPO runs.
Furthermore, it can happen that after runs finished the ray server still operates and leads to issues when starting new runs for GRPO.
The ```ray stop``` bash command may help in that case, tearing down any running ray instances.

## Merging FSDP checkpoints to HF format

After training, convert sharded FSDP actor checkpoints to standard HuggingFace
model directories:

```bash
bash grpo_experiments/merge_fsdp_to_hf.sh \
    grpo_experiments/checkpoints/<experiment-dir> \
    [optional-output-root]   # defaults to grpo_experiments/merged/
```

Each `global_step*` subdirectory that contains a valid `actor/` checkpoint is
merged into `<output-root>/<experiment>_global_step<N>/`.

### Required: create `model.txt` after merging

Each merged run directory **must** contain a plain-text file called `model.txt`
with the base model's Hugging Face name, e.g.:

```
Qwen/Qwen2.5-3B-Instruct
```

or

```
meta-llama/Llama-3.2-3B-Instruct
```

This file is needed by downstream evaluation scripts to identify which base model
the checkpoint was trained from. Create it once per experiment after merging:

```bash
echo "Qwen/Qwen2.5-3B-Instruct" \
    > grpo_experiments/merged/Qwen2.5-3B-Instruct_grpo_countdown_s42_global_step500/model.txt
```

## Freeing disk space

Optimizer state (`optim_*.pt`) is large and not needed after training.
Delete it for a specific run with:

```bash
bash grpo_experiments/delete_optim_checkpoints.sh checkpoints/<subfolder>
```

For runs with frequent checkpointing or large models it is recommented to run this script periodically in another tmux session to avoid running out of disk space.

## Key hyperparameters (shared across all runs)

| Parameter | Value |
|-----------|-------|
| Train batch size | 200 |
| Rollouts per prompt (`n`) | 30 |
| Max response length | 1024 |
| Learning rate | 1e-6 |
| KL loss coefficient | 0.001 |
| Total epochs | 500 |
| Checkpoint / eval frequency | every 25 steps for detailed runs (only for some selected runs for which the progress over iterations is plotted), otherwise 500 steps (just evaluate and checkpoint once at the end)
