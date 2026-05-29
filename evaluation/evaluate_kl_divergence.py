import argparse
import gc
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.utils import (
    ESEvalLLM,
    PRECISION_CHOICES,
    append_jsonl,
    detect_run_backend,
    generate_rollouts,
    get_grpo_checkpoint_dirs,
    get_iteration_files,
    infer_precision_from_config,
    iter_hf_checkpoint_weights,
    load_json,
    load_text,
    resolve_grpo_run_paths,
    resolve_run_paths,
    should_evaluate,
)
from weight_update_utils import (
    apply_seeded_noise_to_parameters,
    update_parameters_from_seeds,
)

KL_ESTIMATOR_NAME = "mc_sampled_log_ratio"
ORIGINAL_LOG_PROBS_KEY = "original_log_probs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a saved ES or GRPO run and estimate KL divergence between "
            "the original model and later checkpoints on fixed continuations "
            "that were generated once by the original vLLM model."
        )
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="hellaswag")
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument(
        "--precision",
        type=str,
        choices=PRECISION_CHOICES,
        default=None,
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default=None,
        help="CUDA_VISIBLE_DEVICES override. Defaults to the run config value if present.",
    )
    parser.add_argument("--max_iteration", type=int, default=None)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Batch size for HF fixed-sequence scoring.",
    )
    parser.add_argument(
        "--max_generation_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to decode with the original vLLM model.",
    )
    parser.add_argument("--disable_caching", action="store_true")
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_dataset_module(dataset_name: str):
    if dataset_name == "countdown":
        from tasks.countdown.data import get_data
    elif dataset_name == "gsm8k":
        from tasks.gsm8k.data import get_data
    elif dataset_name == "proofwriter":
        from tasks.proofwriter.data import get_data
    elif dataset_name == "hellaswag":
        from prior_tasks.hellaswag.data import get_data
    elif dataset_name == "piqa":
        from prior_tasks.piqa.data import get_data
    elif dataset_name in {"arc_challenge", "arc-challenge"}:
        from prior_tasks.arc_challenge.data import get_data
    elif dataset_name in {"mmlu_pro", "mmlu-pro"}:
        from prior_tasks.mmlu_pro.data import get_data
    else:
        raise ValueError(f"Unsupported forgetting dataset: {dataset_name}")
    return get_data


def maybe_select_subset(
    eval_data: list[dict[str, Any]],
    subset_size: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if subset_size is None or subset_size >= len(eval_data):
        return eval_data
    if subset_size <= 0:
        raise ValueError(f"subset_size must be positive, got {subset_size}")

    rng = random.Random(seed)
    sampled_indices = rng.sample(range(len(eval_data)), k=subset_size)
    return [eval_data[idx] for idx in sampled_indices]


def resolve_torch_dtype(precision_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[precision_name]


def resolve_eval_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "evaluate_kl_divergence.py now requires a CUDA device for HF logprob scoring, "
            "but CUDA is unavailable."
        )

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count < 1:
        raise RuntimeError(
            "evaluate_kl_divergence.py requires at least one visible CUDA device. "
            f"Found {visible_gpu_count}."
        )

    return torch.device("cuda:0")


def snapshot_weights_to_cpu(model: torch.nn.Module) -> list[torch.Tensor]:
    snapshots: list[torch.Tensor] = []
    for parameter in model.parameters():
        snapshots.append(parameter.detach().to(device="cpu", copy=True))
    return snapshots


def load_causal_lm(model_name: str, torch_dtype: torch.dtype) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )


def load_hf_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_dir: Path,
) -> tuple[list[str], list[str]]:
    checkpoint_state = dict(iter_hf_checkpoint_weights(checkpoint_dir))
    incompatible = model.load_state_dict(checkpoint_state, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise ValueError(
            "Checkpoint load mismatch for "
            f"{checkpoint_dir}: missing_keys={missing_keys}, "
            f"unexpected_keys={unexpected_keys}"
        )
    return missing_keys, unexpected_keys


def resolve_eval_batch_size(
    args: argparse.Namespace,
    run_config: dict[str, Any] | None = None,
) -> int:
    if args.eval_batch_size is not None:
        return int(args.eval_batch_size)
    if run_config is not None:
        return min(int(run_config.get("eval_batch_size", 4)), 4)
    return 4


def extract_prompt_token_ids(context: Any) -> list[int]:
    if hasattr(context, "prompt_token_ids"):
        return list(getattr(context, "prompt_token_ids"))
    if isinstance(context, dict) and "prompt_token_ids" in context:
        return list(context["prompt_token_ids"])
    if isinstance(context, (list, tuple)):
        return list(context)
    raise TypeError(
        "Unsupported prompt representation. Expected a TokensPrompt-like "
        f"object, got {type(context)!r}."
    )


def pad_sequences(
    sequences: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(sequence) for sequence in sequences)
    batch_size = len(sequences)
    input_ids = torch.full(
        (batch_size, max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
        device=device,
    )

    for row_idx, sequence in enumerate(sequences):
        seq_len = len(sequence)
        input_ids[row_idx, :seq_len] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row_idx, :seq_len] = 1

    return input_ids, attention_mask


def replay_perturb_restore_model(
    model: torch.nn.Module,
    seeds: list[int],
    sigma: float,
    mirror_sampling: bool,
    caching: bool,
) -> None:
    negates = [False, True] if mirror_sampling else [False]
    for negate in negates:
        scale = -float(sigma) if negate else float(sigma)
        for seed in seeds:
            apply_seeded_noise_to_parameters(
                model.parameters(),
                seed=seed,
                scale=scale,
                caching=caching,
            )
            apply_seeded_noise_to_parameters(
                model.parameters(),
                seed=seed,
                scale=-scale,
                caching=caching,
            )


def build_reference_sequences(
    llm: ESEvalLLM,
    eval_data: list[dict[str, Any]],
    generation_seed: int,
    max_generation_tokens: int,
    batch_size: int,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    references: list[dict[str, Any]] = []
    generated_lengths: list[int] = []
    start = time.time()

    for batch_start in tqdm(
        range(0, len(eval_data), max(1, batch_size)),
        desc="Generating reference continuations",
    ):
        batch = eval_data[batch_start : batch_start + max(1, batch_size)]
        prompts = [item["context"] for item in batch]
        outputs = generate_rollouts(
            llm,
            prompts,
            temperature=1.0,
            seed=generation_seed,
            max_tokens=max_generation_tokens,
        )

        for idx, (item, output) in enumerate(zip(batch, outputs)):
            prompt_token_ids = extract_prompt_token_ids(item["context"])
            generated_token_ids = []
            generated_text = ""
            if output.outputs:
                generated_token_ids = list(getattr(output.outputs[0], "token_ids", []) or [])
                generated_text = getattr(output.outputs[0], "text", "")

            if verbose and not references and idx == 0:
                print("Reference completion preview:")
                print(generated_text)

            generated_lengths.append(len(generated_token_ids))
            references.append(
                {
                    "prompt_token_ids": prompt_token_ids,
                    "generated_token_ids": generated_token_ids,
                    "generated_tokens": len(generated_token_ids),
                }
            )

    elapsed = time.time() - start
    length_array = np.asarray(generated_lengths, dtype=np.float64)
    reference_stats = {
        "reference_generation_elapsed": float(elapsed),
        "num_generated_tokens": int(sum(generated_lengths)),
        "avg_generated_tokens": float(length_array.mean()) if len(length_array) else 0.0,
        "std_generated_tokens": float(length_array.std()) if len(length_array) else 0.0,
        "min_generated_tokens": int(length_array.min()) if len(length_array) else 0,
        "max_generated_tokens": int(length_array.max()) if len(length_array) else 0,
    }
    return references, reference_stats


def load_eval_data(
    dataset_name: str,
    tokenizer_source: str,
    subset_size: int | None,
) -> tuple[AutoTokenizer, int, list[dict[str, Any]]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer_size = len(tokenizer)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    get_data = load_dataset_module(dataset_name)
    _, eval_data = get_data(tokenizer)
    eval_data = maybe_select_subset(eval_data, subset_size, 2357)
    return tokenizer, tokenizer_size, eval_data


def build_reference_sequences_for_model(
    model_name: str,
    precision_name: str,
    eval_data: list[dict[str, Any]],
    generation_seed: int,
    max_generation_tokens: int,
    batch_size: int,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    llm = None
    try:
        llm = ESEvalLLM(
            model=model_name,
            tensor_parallel_size=1,
            dtype=precision_name,
            worker_extension_cls="evaluation.utils.EvalWorkerExtension",
            enable_prefix_caching=False,
            enforce_eager=False,
            gpu_memory_utilization=0.9,
        )
        return build_reference_sequences(
            llm=llm,
            eval_data=eval_data,
            generation_seed=generation_seed,
            max_generation_tokens=max_generation_tokens,
            batch_size=batch_size,
            verbose=verbose,
        )
    finally:
        if llm is not None:
            del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def make_zero_metrics(
    step: int,
    num_examples: int,
    reference_stats: dict[str, float],
) -> dict[str, Any]:
    return {
        "step": int(step),
        "num_examples": int(num_examples),
        "kl_estimator": KL_ESTIMATOR_NAME,
        "prompt_averaged_kl_divergence": 0.0,
        "prompt_averaged_kl_std": 0.0,
        "prompt_averaged_kl_min": 0.0,
        "prompt_averaged_kl_max": 0.0,
        "token_weighted_kl_divergence": 0.0,
        "num_generated_tokens": int(reference_stats["num_generated_tokens"]),
        "avg_generated_tokens": float(reference_stats["avg_generated_tokens"]),
        "std_generated_tokens": float(reference_stats["std_generated_tokens"]),
        "min_generated_tokens": int(reference_stats["min_generated_tokens"]),
        "max_generated_tokens": int(reference_stats["max_generated_tokens"]),
        "elapsed": 0.0,
        "reference_generation_elapsed": float(reference_stats["reference_generation_elapsed"]),
    }


def log_eval_line(metrics: dict[str, Any]) -> None:
    print(
        f"[Eval @ step {metrics['step']}] "
        f"prompt_avg_kl={metrics['prompt_averaged_kl_divergence']:.6f} "
        f"token_weighted_kl={metrics['token_weighted_kl_divergence']:.6f} "
        f"generated_tokens={metrics['avg_generated_tokens']:.2f} "
        f"time={metrics['elapsed']:.2f}s"
    )


def gather_generated_token_log_probs(
    model: torch.nn.Module,
    input_ids_cpu: torch.Tensor,
    attention_mask_cpu: torch.Tensor,
    batch: list[dict[str, Any]],
    device: torch.device,
) -> list[torch.Tensor]:
    input_ids = input_ids_cpu.to(device=device, non_blocking=True)
    attention_mask = attention_mask_cpu.to(device=device, non_blocking=True)
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    generated_log_probs: list[torch.Tensor] = []

    for row_idx, reference in enumerate(batch):
        generated_token_ids = reference["generated_token_ids"]
        generated_tokens = len(generated_token_ids)
        if generated_tokens <= 0:
            generated_log_probs.append(torch.empty(0, dtype=torch.float32))
            continue

        prompt_len = len(reference["prompt_token_ids"])
        start_idx = prompt_len - 1
        end_idx = start_idx + generated_tokens
        target_ids = torch.tensor(
            generated_token_ids,
            dtype=torch.long,
            device=logits.device,
        )
        row_logits = logits[row_idx, start_idx:end_idx, :].float()
        row_log_probs = torch.log_softmax(row_logits, dim=-1)
        token_log_probs = row_log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)
        generated_log_probs.append(token_log_probs.detach().cpu())
        del target_ids
        del row_logits
        del row_log_probs
        del token_log_probs

    del logits
    del input_ids
    del attention_mask
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generated_log_probs


@torch.inference_mode()
def precompute_original_log_probs(
    original_model: torch.nn.Module,
    references: list[dict[str, Any]],
    eval_batch_size: int,
    pad_token_id: int,
    eval_device: torch.device,
    desc: str = "Precomputing original logprobs",
) -> None:
    batch_size = max(1, eval_batch_size)
    cpu_device = torch.device("cpu")
    original_model.to(eval_device)

    for batch_start in tqdm(range(0, len(references), batch_size), desc=desc):
        batch = references[batch_start : batch_start + batch_size]
        sequences = [
            reference["prompt_token_ids"] + reference["generated_token_ids"]
            for reference in batch
        ]
        input_ids, attention_mask = pad_sequences(sequences, pad_token_id, cpu_device)
        original_log_probs = gather_generated_token_log_probs(
            model=original_model,
            input_ids_cpu=input_ids,
            attention_mask_cpu=attention_mask,
            batch=batch,
            device=eval_device,
        )

        for reference, log_probs in zip(batch, original_log_probs):
            reference[ORIGINAL_LOG_PROBS_KEY] = log_probs

        del original_log_probs
        del input_ids
        del attention_mask


def cache_original_log_probs_for_model(
    model_name: str,
    torch_dtype: torch.dtype,
    tokenizer: AutoTokenizer,
    tokenizer_size: int,
    references: list[dict[str, Any]],
    eval_batch_size: int,
    pad_token_id: int,
    eval_device: torch.device,
) -> None:
    original_model = load_causal_lm(model_name=model_name, torch_dtype=torch_dtype)
    if len(tokenizer) != tokenizer_size:
        original_model.resize_token_embeddings(len(tokenizer))
    original_model.eval()
    original_model.requires_grad_(False)

    precompute_original_log_probs(
        original_model=original_model,
        references=references,
        eval_batch_size=eval_batch_size,
        pad_token_id=pad_token_id,
        eval_device=eval_device,
    )
    del original_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def initialize_updated_model(
    model_name: str,
    torch_dtype: torch.dtype,
    tokenizer: AutoTokenizer,
    tokenizer_size: int,
    eval_device: torch.device,
) -> torch.nn.Module:
    updated_model = load_causal_lm(model_name=model_name, torch_dtype=torch_dtype)
    if len(tokenizer) != tokenizer_size:
        updated_model.resize_token_embeddings(len(tokenizer))
    updated_model.to(eval_device)
    updated_model.eval()
    updated_model.requires_grad_(False)
    return updated_model


@torch.inference_mode()
def evaluate_kl_divergence(
    updated_model: torch.nn.Module,
    references: list[dict[str, Any]],
    step: int,
    eval_batch_size: int,
    pad_token_id: int,
    eval_device: torch.device,
    reference_stats: dict[str, float],
) -> dict[str, Any]:
    if not references:
        return make_zero_metrics(step=step, num_examples=0, reference_stats=reference_stats)

    batch_size = max(1, eval_batch_size)
    prompt_means: list[float] = []
    total_kl = 0.0
    total_positions = 0
    total_generated_tokens = 0
    start = time.time()
    cpu_device = torch.device("cpu")

    for batch_start in range(0, len(references), batch_size):
        batch = references[batch_start : batch_start + batch_size]
        sequences = [
            reference["prompt_token_ids"] + reference["generated_token_ids"]
            for reference in batch
        ]
        input_ids, attention_mask = pad_sequences(sequences, pad_token_id, cpu_device)

        updated_log_probs = gather_generated_token_log_probs(
            model=updated_model,
            input_ids_cpu=input_ids,
            attention_mask_cpu=attention_mask,
            batch=batch,
            device=eval_device,
        )

        for row_idx, reference in enumerate(batch):
            generated_tokens = int(reference["generated_tokens"])
            total_generated_tokens += generated_tokens
            if generated_tokens <= 0:
                prompt_means.append(0.0)
                continue

            original_log_probs = reference.get(ORIGINAL_LOG_PROBS_KEY)
            if original_log_probs is None:
                raise KeyError(
                    f"Missing cached {ORIGINAL_LOG_PROBS_KEY!r}; "
                    "call precompute_original_log_probs before evaluation."
                )
            kl_per_token = original_log_probs - updated_log_probs[row_idx]
            prompt_means.append(float(kl_per_token.mean().item()))
            total_kl += float(kl_per_token.sum().item())
            total_positions += int(kl_per_token.numel())

        del updated_log_probs
        del input_ids
        del attention_mask

    elapsed = time.time() - start
    prompt_array = np.asarray(prompt_means, dtype=np.float64)
    metrics = {
        "step": int(step),
        "num_examples": int(len(references)),
        "kl_estimator": KL_ESTIMATOR_NAME,
        "prompt_averaged_kl_divergence": float(prompt_array.mean()) if len(prompt_array) else 0.0,
        "prompt_averaged_kl_std": float(prompt_array.std()) if len(prompt_array) else 0.0,
        "prompt_averaged_kl_min": float(prompt_array.min()) if len(prompt_array) else 0.0,
        "prompt_averaged_kl_max": float(prompt_array.max()) if len(prompt_array) else 0.0,
        "token_weighted_kl_divergence": float(total_kl / total_positions) if total_positions else 0.0,
        "num_generated_tokens": int(total_generated_tokens),
        "avg_generated_tokens": float(reference_stats["avg_generated_tokens"]),
        "std_generated_tokens": float(reference_stats["std_generated_tokens"]),
        "min_generated_tokens": int(reference_stats["min_generated_tokens"]),
        "max_generated_tokens": int(reference_stats["max_generated_tokens"]),
        "elapsed": float(elapsed),
        "reference_generation_elapsed": float(reference_stats["reference_generation_elapsed"]),
    }
    log_eval_line(metrics)
    return metrics


def run_es_evaluation(args: argparse.Namespace, run_dir: Path) -> int:
    args_path, updates_dir = resolve_run_paths(run_dir)
    run_config = load_json(args_path)

    model_name = args.model_name or run_config["model_name"]
    dataset_name = args.dataset
    precision_name = args.precision or run_config.get("precision", "float16")
    cuda_devices = args.cuda_devices or run_config.get("cuda_devices")
    caching = False if args.disable_caching else run_config.get("caching", False)
    alpha = float(run_config["alpha"])
    sigma = float(run_config["sigma"])
    weight_decay_type = str(run_config.get("weight_decay_type", "none"))
    weight_decay_lambda = float(run_config.get("weight_decay_lambda", 0.0))
    population_size = int(run_config["population_size"])
    mirror_sampling = bool(run_config.get("mirror_sampling", False))
    eval_interval = int(run_config.get("eval_interval", 0))
    eval_batch_size = resolve_eval_batch_size(args, run_config)
    generation_seed = (
        int(run_config["global_seed"])
        if run_config.get("global_seed") is not None
        else 999
    )
    iteration_files = get_iteration_files(updates_dir, args.max_iteration)
    final_step = int(load_json(iteration_files[-1])["iteration"])

    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    eval_device = resolve_eval_device()
    torch_dtype = resolve_torch_dtype(precision_name)
    metrics_path = run_dir / f"{dataset_name}_kl.jsonl"

    print("Detected backend: es")
    print(f"Run directory: {run_dir}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Precision: {precision_name} -> {torch_dtype}")
    print(f"HF scoring device: {eval_device}")
    print("Original model logprobs are precomputed once, then the original model is deleted")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Caching enabled: {caching}")
    print(f"Mirror sampling: {mirror_sampling}")
    print(f"Weight decay type: {weight_decay_type}")
    print(f"Weight decay lambda: {weight_decay_lambda}")
    print(f"Eval interval: {eval_interval}")
    print(f"KL estimator: {KL_ESTIMATOR_NAME}")
    print(f"Eval batch size: {eval_batch_size}")
    print(f"Max generation tokens: {args.max_generation_tokens}")
    print(f"Replaying {len(iteration_files)} iteration updates")

    tokenizer, tokenizer_size, eval_data = load_eval_data(
        dataset_name=dataset_name,
        tokenizer_source=model_name,
        subset_size=args.subset_size,
    )
    print(f"Evaluating on {len(eval_data)} samples")

    if metrics_path.exists():
        metrics_path.unlink()
    print(f"Writing evaluation metrics to {metrics_path}")

    references, reference_stats = build_reference_sequences_for_model(
        model_name=model_name,
        precision_name=precision_name,
        eval_data=eval_data,
        generation_seed=generation_seed,
        max_generation_tokens=args.max_generation_tokens,
        batch_size=eval_batch_size,
        verbose=args.verbose,
    )

    cache_original_log_probs_for_model(
        model_name=model_name,
        torch_dtype=torch_dtype,
        tokenizer=tokenizer,
        tokenizer_size=tokenizer_size,
        references=references,
        eval_batch_size=eval_batch_size,
        pad_token_id=int(tokenizer.pad_token_id),
        eval_device=eval_device,
    )

    updated_model = initialize_updated_model(
        model_name=model_name,
        torch_dtype=torch_dtype,
        tokenizer=tokenizer,
        tokenizer_size=tokenizer_size,
        eval_device=eval_device,
    )

    original_parameter_snapshot = None
    if weight_decay_type != "none" and weight_decay_lambda > 0.0:
        original_parameter_snapshot = snapshot_weights_to_cpu(updated_model)

    metrics = make_zero_metrics(
        step=0,
        num_examples=len(references),
        reference_stats=reference_stats,
    )
    log_eval_line(metrics)
    append_jsonl(metrics_path, metrics)

    progress = tqdm(
        enumerate(iteration_files, start=1),
        total=len(iteration_files),
        desc="Replaying updates",
    )
    for idx, update_path in progress:
        payload = load_json(update_path)
        seeds = payload["seeds"]
        coeffs = payload["update_coefficients"]
        iteration = int(payload["iteration"])

        replay_perturb_restore_model(
            model=updated_model,
            seeds=seeds,
            sigma=sigma,
            mirror_sampling=mirror_sampling,
            caching=caching,
        )
        update_parameters_from_seeds(
            updated_model.parameters(),
            seeds=seeds,
            coeffs=coeffs,
            alpha=alpha,
            population_size=population_size,
            caching=caching,
            original_parameters=original_parameter_snapshot,
            weight_decay_type=weight_decay_type,
            weight_decay_lambda=weight_decay_lambda,
        )

        if idx % max(1, args.print_every) == 0 or idx == len(iteration_files):
            progress.set_postfix(iteration=iteration, refresh=False)

        if should_evaluate(iteration, final_step, eval_interval):
            metrics = evaluate_kl_divergence(
                updated_model=updated_model,
                references=references,
                step=iteration,
                eval_batch_size=eval_batch_size,
                pad_token_id=int(tokenizer.pad_token_id),
                eval_device=eval_device,
                reference_stats=reference_stats,
            )
            append_jsonl(metrics_path, metrics)

    return 0


def run_grpo_evaluation(args: argparse.Namespace, run_dir: Path) -> int:
    model_path, all_checkpoint_dirs = resolve_grpo_run_paths(run_dir)
    checkpoint_dirs = get_grpo_checkpoint_dirs(run_dir, args.max_iteration)

    base_model_name = args.model_name or load_text(model_path)
    tokenizer_source = str(all_checkpoint_dirs[0][1])
    dataset_name = args.dataset
    precision_name = args.precision or infer_precision_from_config(all_checkpoint_dirs[0][1])
    eval_batch_size = resolve_eval_batch_size(args)
    generation_seed = 999

    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    eval_device = resolve_eval_device()
    torch_dtype = resolve_torch_dtype(precision_name)
    metrics_path = run_dir / f"{dataset_name}_kl.jsonl"

    print("Detected backend: grpo")
    print(f"Run directory: {run_dir}")
    print(f"Base model: {base_model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Precision: {precision_name} -> {torch_dtype}")
    print(f"HF scoring device: {eval_device}")
    print("Original model logprobs are precomputed once, then the original model is deleted")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"KL estimator: {KL_ESTIMATOR_NAME}")
    print(f"Eval batch size: {eval_batch_size}")
    print(f"Max generation tokens: {args.max_generation_tokens}")
    print(f"Discovered {len(checkpoint_dirs)} checkpoint directories")

    tokenizer, tokenizer_size, eval_data = load_eval_data(
        dataset_name=dataset_name,
        tokenizer_source=tokenizer_source,
        subset_size=args.subset_size,
    )
    print(f"Evaluating on {len(eval_data)} samples")

    if metrics_path.exists():
        metrics_path.unlink()
    print(f"Writing evaluation metrics to {metrics_path}")

    references, reference_stats = build_reference_sequences_for_model(
        model_name=base_model_name,
        precision_name=precision_name,
        eval_data=eval_data,
        generation_seed=generation_seed,
        max_generation_tokens=args.max_generation_tokens,
        batch_size=eval_batch_size,
        verbose=args.verbose,
    )

    cache_original_log_probs_for_model(
        model_name=base_model_name,
        torch_dtype=torch_dtype,
        tokenizer=tokenizer,
        tokenizer_size=tokenizer_size,
        references=references,
        eval_batch_size=eval_batch_size,
        pad_token_id=int(tokenizer.pad_token_id),
        eval_device=eval_device,
    )

    updated_model = initialize_updated_model(
        model_name=base_model_name,
        torch_dtype=torch_dtype,
        tokenizer=tokenizer,
        tokenizer_size=tokenizer_size,
        eval_device=eval_device,
    )

    metrics = evaluate_kl_divergence(
        updated_model=updated_model,
        references=references,
        step=0,
        eval_batch_size=eval_batch_size,
        pad_token_id=int(tokenizer.pad_token_id),
        eval_device=eval_device,
        reference_stats=reference_stats,
    )
    append_jsonl(metrics_path, metrics)

    progress = tqdm(
        checkpoint_dirs,
        total=len(checkpoint_dirs),
        desc="Evaluating GRPO checkpoints",
    )
    for idx, (step, checkpoint_dir) in enumerate(progress, start=1):
        load_hf_checkpoint_into_model(updated_model, checkpoint_dir)

        if idx % max(1, args.print_every) == 0 or idx == len(checkpoint_dirs):
            progress.set_postfix(step=step, refresh=False)

        metrics = evaluate_kl_divergence(
            updated_model=updated_model,
            references=references,
            step=step,
            eval_batch_size=eval_batch_size,
            pad_token_id=int(tokenizer.pad_token_id),
            eval_device=eval_device,
            reference_stats=reference_stats,
        )
        append_jsonl(metrics_path, metrics)

    return 0


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    backend = detect_run_backend(run_dir)

    if backend == "es":
        return run_es_evaluation(args, run_dir)
    if backend == "grpo":
        return run_grpo_evaluation(args, run_dir)

    raise ValueError(f"Unsupported backend: {backend}")


if __name__ == "__main__":
    sys.exit(main())
