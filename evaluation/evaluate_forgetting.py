import argparse
import gc
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from evaluation.utils import (
    ESEvalLLM,
    EvalWorkerExtension,
    PRECISION_CHOICES,
    append_jsonl,
    count_generated_tokens,
    detect_run_backend,
    generate_rollouts,
    get_grpo_checkpoint_dirs,
    get_iteration_files,
    infer_precision_from_config,
    load_json,
    load_text,
    replay_perturb_restore,
    resolve_grpo_run_paths,
    resolve_run_paths,
    should_evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a saved ES run and evaluate the reconstructed model on the "
            "task eval set at the same steps used during training."
        )
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="hellaswag")
    parser.add_argument("--subset_size", type=int, default=2048)
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
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--disable_caching", action="store_true")
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

def evaluate_model(
    llm: ESEvalLLM,
    eval_data: List[Dict[str, Any]],
    evaluate_reward,
    step: int,
    eval_batch_size: int,
    eval_seed: int,
    verbose: bool = False,
) -> Dict[str, float | int]:
    if not eval_data:
        return {
            "step": step,
            "num_examples": 0,
            "avg_reward": 0.0,
            "std_reward": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
            "avg_format": 0.0,
            "avg_answer": 0.0,
            "accuracy": 0.0,
            "avg_output_tokens": 0.0,
            "std_output_tokens": 0.0,
            "elapsed": 0.0,
        }

    batch_size = max(1, eval_batch_size)
    all_rewards = []
    format_rewards = []
    answer_rewards = []
    output_token_counts = []
    start = time.time()

    for batch_start in range(0, len(eval_data), batch_size):
        batch = eval_data[batch_start : batch_start + batch_size]
        prompts = [d["context"] for d in batch]
        outputs = generate_rollouts(llm, prompts, seed=eval_seed)

        for idx, (out, data) in enumerate(zip(outputs, batch)):
            reward, _ = evaluate_reward(out.outputs[0].text, data)
            if verbose and idx == 0:
                print(out.outputs[0].text)
            all_rewards.append(reward["reward"])
            output_token_counts.append(count_generated_tokens(out))
            if "reward_info" in reward:
                format_rewards.append(reward["reward_info"].get("format_reward", 0.0))
                answer_rewards.append(reward["reward_info"].get("answer_reward", 0.0))

    elapsed = time.time() - start
    avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    std_reward = float(np.std(all_rewards)) if all_rewards else 0.0
    min_reward = float(np.min(all_rewards)) if all_rewards else 0.0
    max_reward = float(np.max(all_rewards)) if all_rewards else 0.0
    avg_format = float(np.mean(format_rewards)) if format_rewards else 0.0
    avg_answer = float(np.mean(answer_rewards)) if answer_rewards else 0.0
    accuracy = (
        sum(1 for answer in answer_rewards if answer > 0) / len(answer_rewards) * 100.0
        if answer_rewards
        else 0.0
    )
    avg_output_tokens = float(np.mean(output_token_counts)) if output_token_counts else 0.0
    std_output_tokens = float(np.std(output_token_counts)) if output_token_counts else 0.0

    print(
        f"[Eval @ step {step}] avg_reward={avg_reward:.4f} ± {std_reward:.4f} "
        f"range=[{min_reward:.4f},{max_reward:.4f}] format={avg_format:.4f} "
        f"answer={avg_answer:.4f} acc={accuracy:.1f}% "
        f"output_tokens={avg_output_tokens:.2f} ± {std_output_tokens:.2f} time={elapsed:.2f}s"
    )
    return {
        "step": step,
        "num_examples": len(eval_data),
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "avg_format": avg_format,
        "avg_answer": avg_answer,
        "accuracy": accuracy,
        "avg_output_tokens": avg_output_tokens,
        "std_output_tokens": std_output_tokens,
        "elapsed": elapsed,
    }


def load_dataset_module(dataset_name: str):
    if dataset_name == "countdown":
        from tasks.countdown.data import get_data, evaluate_reward
    elif dataset_name == "gsm8k":
        from tasks.gsm8k.data import get_data, evaluate_reward
    elif dataset_name == "proofwriter":
        from tasks.proofwriter.data import get_data, evaluate_reward
    elif dataset_name == "hellaswag":
        from prior_tasks.hellaswag.data import get_data, evaluate_reward
    elif dataset_name == "piqa":
        from prior_tasks.piqa.data import get_data, evaluate_reward
    elif dataset_name in {"arc_challenge", "arc-challenge"}:
        from prior_tasks.arc_challenge.data import get_data, evaluate_reward
    elif dataset_name in {"mmlu_pro", "mmlu-pro"}:
        from prior_tasks.mmlu_pro.data import get_data, evaluate_reward
    else:
        raise ValueError(f"Unsupported forgetting dataset: {dataset_name}")
    return get_data, evaluate_reward

def maybe_select_subset(
    eval_data: List[Dict[str, Any]],
    subset_size: int | None,
    seed: int,
) -> List[Dict[str, Any]]:
    if subset_size is None or subset_size >= len(eval_data):
        return eval_data
    if subset_size <= 0:
        raise ValueError(f"subset_size must be positive, got {subset_size}")

    rng = random.Random(seed)
    sampled_indices = rng.sample(range(len(eval_data)), k=subset_size)
    return [eval_data[idx] for idx in sampled_indices]

def build_eval_llm(model_name: str, precision_name: str) -> ESEvalLLM:
    reset_vllm_distributed_state()
    return ESEvalLLM(
        model=model_name,
        tensor_parallel_size=1,
        dtype=precision_name,
        worker_extension_cls="evaluation.utils.EvalWorkerExtension",
        enable_prefix_caching=False,
        enforce_eager=False,
        gpu_memory_utilization=0.9,
    )


def reset_vllm_distributed_state() -> None:
    try:
        cleanup_dist_env_and_memory()
    except (AssertionError, RuntimeError, ValueError):
        pass


def cleanup_llm(llm: ESEvalLLM | None) -> None:
    if llm is None:
        reset_vllm_distributed_state()
        return

    engine = getattr(llm, "llm_engine", None)
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            pass

    del llm
    reset_vllm_distributed_state()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_eval_data(
    dataset_name: str,
    tokenizer_source: str,
    subset_size: int | None,
) -> tuple[List[Dict[str, Any]], Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    get_data, evaluate_reward = load_dataset_module(dataset_name)
    _, eval_data = get_data(tokenizer)
    eval_data = maybe_select_subset(eval_data, subset_size, 2357)
    return eval_data, evaluate_reward


def prepare_metrics_path(run_dir: Path, dataset_name: str) -> Path:
    metrics_path = run_dir / f"{dataset_name}.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    print(f"Writing evaluation metrics to {metrics_path}")
    return metrics_path


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
    num_engines = int(run_config.get("num_engines", 1))
    mirror_sampling = bool(run_config.get("mirror_sampling", False))
    eval_interval = int(run_config.get("eval_interval", 0))
    eval_batch_size = int(args.eval_batch_size or run_config.get("eval_batch_size", 512))
    eval_seed = int(run_config["global_seed"]) if run_config.get("global_seed") is not None else 999
    iteration_files = get_iteration_files(updates_dir, args.max_iteration)
    final_step = int(load_json(iteration_files[-1])["iteration"])

    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    print(f"Detected backend: es")
    print(f"Run directory: {run_dir}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Precision: {precision_name}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Caching enabled: {caching}")
    print(f"Mirror sampling: {mirror_sampling}")
    print(f"Weight decay type: {weight_decay_type}")
    print(f"Weight decay lambda: {weight_decay_lambda}")
    print(f"Num engines: {num_engines}")
    print(f"Eval interval: {eval_interval}")
    print(f"Eval batch size: {eval_batch_size}")
    print(f"Replaying {len(iteration_files)} iteration updates")

    eval_data, evaluate_reward = load_eval_data(dataset_name, model_name, args.subset_size)
    print(f"Evaluating on {len(eval_data)} samples")
    metrics_path = prepare_metrics_path(run_dir, dataset_name)

    llm = None
    try:
        llm = build_eval_llm(model_name, precision_name)

        if weight_decay_type != "none" and weight_decay_lambda > 0.0:
            llm.collective_rpc("snapshot_weights_to_cpu")

        metrics = evaluate_model(
            llm=llm,
            eval_data=eval_data,
            evaluate_reward=evaluate_reward,
            step=0,
            eval_batch_size=eval_batch_size,
            eval_seed=eval_seed,
            verbose=args.verbose,
        )
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

            replay_perturb_restore(
                llm=llm,
                seeds=seeds,
                sigma=sigma,
                num_engines=num_engines,
                mirror_sampling=mirror_sampling,
                caching=caching,
            )
            llm.collective_rpc(
                "update_weights_from_seeds",
                args=(
                    seeds,
                    coeffs,
                    alpha,
                    population_size,
                    caching,
                    weight_decay_type,
                    weight_decay_lambda,
                ),
            )

            if idx % max(1, args.print_every) == 0 or idx == len(iteration_files):
                progress.set_postfix(iteration=iteration, refresh=False)

            if should_evaluate(iteration, final_step, eval_interval):
                metrics = evaluate_model(
                    llm=llm,
                    eval_data=eval_data,
                    evaluate_reward=evaluate_reward,
                    step=iteration,
                    eval_batch_size=eval_batch_size,
                    eval_seed=eval_seed,
                    verbose=args.verbose,
                )
                append_jsonl(metrics_path, metrics)

        return 0
    finally:
        cleanup_llm(llm)


def run_grpo_evaluation(args: argparse.Namespace, run_dir: Path) -> int:
    model_path, all_checkpoint_dirs = resolve_grpo_run_paths(run_dir)
    checkpoint_dirs = get_grpo_checkpoint_dirs(run_dir, args.max_iteration)

    base_model_name = args.model_name or load_text(model_path)
    tokenizer_source = base_model_name
    dataset_name = args.dataset
    precision_name = args.precision or infer_precision_from_config(all_checkpoint_dirs[0][1])
    eval_batch_size = int(args.eval_batch_size or 512)
    eval_seed = 0

    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    print(f"Detected backend: grpo")
    print(f"Run directory: {run_dir}")
    print(f"Base model: {base_model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Precision: {precision_name}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Eval batch size: {eval_batch_size}")
    print(f"Discovered {len(checkpoint_dirs)} checkpoint directories")
    print("Loading GRPO checkpoints in-place into one live vLLM model")

    eval_data, evaluate_reward = load_eval_data(dataset_name, tokenizer_source, args.subset_size)
    print(f"Evaluating on {len(eval_data)} samples")
    metrics_path = prepare_metrics_path(run_dir, dataset_name)

    llm = None
    try:
        llm = build_eval_llm(base_model_name, precision_name)
        metrics = evaluate_model(
            llm=llm,
            eval_data=eval_data,
            evaluate_reward=evaluate_reward,
            step=0,
            eval_batch_size=eval_batch_size,
            eval_seed=eval_seed,
            verbose=args.verbose,
        )
        append_jsonl(metrics_path, metrics)

        progress = tqdm(checkpoint_dirs, total=len(checkpoint_dirs), desc="Evaluating GRPO checkpoints")
        for idx, (step, checkpoint_dir) in enumerate(progress, start=1):
            load_result = llm.collective_rpc(
                "load_hf_checkpoint_weights",
                args=(str(checkpoint_dir),),
            )

            if idx % max(1, args.print_every) == 0 or idx == len(checkpoint_dirs):
                loaded_param_count = "?"
                if load_result and isinstance(load_result[0], dict):
                    loaded_param_count = load_result[0].get("loaded_param_count", "?")
                progress.set_postfix(
                    step=step,
                    loaded=loaded_param_count,
                    refresh=False,
                )

            metrics = evaluate_model(
                llm=llm,
                eval_data=eval_data,
                evaluate_reward=evaluate_reward,
                step=step,
                eval_batch_size=eval_batch_size,
                eval_seed=eval_seed,
                verbose=args.verbose,
            )
            append_jsonl(metrics_path, metrics)

        return 0
    finally:
        cleanup_llm(llm)


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
