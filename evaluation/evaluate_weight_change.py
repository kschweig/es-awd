import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from tqdm import tqdm
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from evaluation.utils import (
    ESEvalLLM,
    PRECISION_CHOICES,
    append_jsonl,
    build_layer_stat_dict,
    detect_run_backend,
    get_iteration_files,
    get_grpo_checkpoint_dirs,
    infer_weight_layer_name,
    infer_precision_from_config,
    layer_name_sort_key,
    load_json,
    load_state_dict,
    load_text,
    make_temp_path,
    replay_perturb_restore,
    resolve_grpo_run_paths,
    resolve_run_paths,
    should_evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a saved run and log the Frobenius norm of the weight "
            "change between the original model and the current model at "
            "evaluation steps, following the same cadence as "
            "evaluate_forgetting.py."
        )
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
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
        "--eval_interval",
        type=int,
        default=None,
        help="Override the run's eval_interval. Defaults to the run config value, or 1.",
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default=None,
        help="Path to the JSONL output file. Defaults to <run_dir>/weight_change.jsonl.",
    )
    parser.add_argument("--disable_caching", action="store_true")
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

def write_current_weights_to_disk(llm: ESEvalLLM, prefix: str) -> Path:
    path = make_temp_path(prefix=prefix)
    llm.collective_rpc("write_weights_to_disk", args=(str(path),))
    return path


def reset_vllm_distributed_state() -> None:
    try:
        cleanup_dist_env_and_memory()
    except (AssertionError, RuntimeError, ValueError):
        pass


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


def prepare_metrics_path(run_dir: Path, metrics_path_arg: str | None) -> Path:
    metrics_path = (
        Path(metrics_path_arg).expanduser().resolve()
        if metrics_path_arg
        else run_dir / "weight_change.jsonl"
    )
    if metrics_path.exists():
        metrics_path.unlink()
    print(f"Writing weight-change metrics to {metrics_path}")
    return metrics_path


def compute_frobenius_summary(
    original: dict[str, torch.Tensor],
    updated: dict[str, torch.Tensor],
) -> dict[str, Any]:
    original_keys = set(original.keys())
    updated_keys = set(updated.keys())
    missing_keys = sorted(original_keys - updated_keys)
    unexpected_keys = sorted(updated_keys - original_keys)
    common_keys = sorted(original_keys & updated_keys)

    shape_mismatches: list[str] = []
    dtype_mismatches: list[str] = []
    total_diff_sq = 0.0
    total_original_sq = 0.0
    total_updated_sq = 0.0
    total_abs_diff = 0.0
    total_elements = 0
    max_tensor_norm = 0.0
    max_tensor_norm_key = None
    max_abs_diff = 0.0
    max_abs_diff_key = None
    layer_diff_sq_by_name: dict[str, float] = {}

    for key in common_keys:
        original_tensor = original[key]
        updated_tensor = updated[key]

        if original_tensor.shape != updated_tensor.shape:
            shape_mismatches.append(key)
            continue

        if original_tensor.dtype != updated_tensor.dtype:
            dtype_mismatches.append(key)

        original_float = original_tensor.to(torch.float64)
        updated_float = updated_tensor.to(torch.float64)
        diff = updated_float - original_float

        tensor_diff_sq = torch.sum(diff * diff).item()
        tensor_norm = math.sqrt(tensor_diff_sq)
        tensor_max_abs = torch.max(torch.abs(diff)).item()

        total_diff_sq += tensor_diff_sq
        total_original_sq += torch.sum(original_float * original_float).item()
        total_updated_sq += torch.sum(updated_float * updated_float).item()
        total_abs_diff += torch.sum(torch.abs(diff)).item()
        total_elements += diff.numel()

        layer_name = infer_weight_layer_name(key)
        if layer_name is not None:
            layer_diff_sq_by_name[layer_name] = (
                layer_diff_sq_by_name.get(layer_name, 0.0) + tensor_diff_sq
            )

        if tensor_norm > max_tensor_norm:
            max_tensor_norm = tensor_norm
            max_tensor_norm_key = key

        if tensor_max_abs > max_abs_diff:
            max_abs_diff = tensor_max_abs
            max_abs_diff_key = key

    frobenius_norm = math.sqrt(total_diff_sq)
    original_frobenius_norm = math.sqrt(total_original_sq)
    updated_frobenius_norm = math.sqrt(total_updated_sq)
    relative_frobenius_norm = (
        frobenius_norm / original_frobenius_norm
        if original_frobenius_norm > 0.0
        else None
    )
    weight_layer_names = sorted(layer_diff_sq_by_name, key=layer_name_sort_key)
    weight_layer_frobenius_norm = build_layer_stat_dict(
        weight_layer_names,
        [math.sqrt(layer_diff_sq_by_name[layer_name]) for layer_name in weight_layer_names],
    )

    return {
        "num_original_tensors": len(original),
        "num_updated_tensors": len(updated),
        "compared_tensors": len(common_keys) - len(shape_mismatches),
        "total_elements_compared": total_elements,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "frobenius_norm": frobenius_norm,
        "original_frobenius_norm": original_frobenius_norm,
        "updated_frobenius_norm": updated_frobenius_norm,
        "relative_frobenius_norm": relative_frobenius_norm,
        "weight_layer_names": weight_layer_names,
        "weight_layer_frobenius_norm": weight_layer_frobenius_norm,
        "mean_abs_diff": total_abs_diff / total_elements if total_elements else 0.0,
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_key": max_abs_diff_key,
        "max_tensor_frobenius_norm": max_tensor_norm,
        "max_tensor_frobenius_norm_key": max_tensor_norm_key,
        "is_comparable": (
            not missing_keys and not unexpected_keys and not shape_mismatches
        ),
    }


def print_summary(summary: dict[str, Any], verbose: bool = False) -> None:
    print("\nWeight change summary")
    print(f"  num_original_tensors: {summary['num_original_tensors']}")
    print(f"  num_updated_tensors: {summary['num_updated_tensors']}")
    print(f"  compared_tensors: {summary['compared_tensors']}")
    print(f"  total_elements_compared: {summary['total_elements_compared']}")
    print(f"  missing_keys: {len(summary['missing_keys'])}")
    print(f"  unexpected_keys: {len(summary['unexpected_keys'])}")
    print(f"  shape_mismatches: {len(summary['shape_mismatches'])}")
    print(f"  dtype_mismatches: {len(summary['dtype_mismatches'])}")
    print(f"  frobenius_norm: {summary['frobenius_norm']}")
    print(f"  original_frobenius_norm: {summary['original_frobenius_norm']}")
    print(f"  updated_frobenius_norm: {summary['updated_frobenius_norm']}")
    print(f"  relative_frobenius_norm: {summary['relative_frobenius_norm']}")
    print(f"  weight_layers: {len(summary['weight_layer_names'])}")
    print(f"  mean_abs_diff: {summary['mean_abs_diff']}")
    print(f"  max_abs_diff: {summary['max_abs_diff']}")
    print(f"  max_abs_diff_key: {summary['max_abs_diff_key']}")
    print(
        "  max_tensor_frobenius_norm: "
        f"{summary['max_tensor_frobenius_norm']}"
    )
    print(
        "  max_tensor_frobenius_norm_key: "
        f"{summary['max_tensor_frobenius_norm_key']}"
    )
    print(f"  is_comparable: {summary['is_comparable']}")

    if verbose and summary["missing_keys"]:
        print("  missing_key_names:")
        for key in summary["missing_keys"]:
            print(f"    {key}")

    if verbose and summary["unexpected_keys"]:
        print("  unexpected_key_names:")
        for key in summary["unexpected_keys"]:
            print(f"    {key}")

    if verbose and summary["shape_mismatches"]:
        print("  shape_mismatch_names:")
        for key in summary["shape_mismatches"]:
            print(f"    {key}")

    if verbose and summary["dtype_mismatches"]:
        print("  dtype_mismatch_names:")
        for key in summary["dtype_mismatches"]:
            print(f"    {key}")


def build_eval_metrics(
    step: int,
    summary: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    return {
        "step": int(step),
        "elapsed": float(elapsed),
        "num_original_tensors": int(summary["num_original_tensors"]),
        "num_updated_tensors": int(summary["num_updated_tensors"]),
        "compared_tensors": int(summary["compared_tensors"]),
        "total_elements_compared": int(summary["total_elements_compared"]),
        "missing_keys": len(summary["missing_keys"]),
        "unexpected_keys": len(summary["unexpected_keys"]),
        "shape_mismatches": len(summary["shape_mismatches"]),
        "dtype_mismatches": len(summary["dtype_mismatches"]),
        "frobenius_norm": float(summary["frobenius_norm"]),
        "original_frobenius_norm": float(summary["original_frobenius_norm"]),
        "updated_frobenius_norm": float(summary["updated_frobenius_norm"]),
        "relative_frobenius_norm": (
            None
            if summary["relative_frobenius_norm"] is None
            else float(summary["relative_frobenius_norm"])
        ),
        "weight_layer_names": list(summary["weight_layer_names"]),
        "weight_layer_frobenius_norm": {
            layer_name: float(layer_norm)
            for layer_name, layer_norm in summary["weight_layer_frobenius_norm"].items()
        },
        "mean_abs_diff": float(summary["mean_abs_diff"]),
        "max_abs_diff": float(summary["max_abs_diff"]),
        "max_abs_diff_key": summary["max_abs_diff_key"],
        "max_tensor_frobenius_norm": float(summary["max_tensor_frobenius_norm"]),
        "max_tensor_frobenius_norm_key": summary["max_tensor_frobenius_norm_key"],
        "is_comparable": bool(summary["is_comparable"]),
    }


def log_eval_line(metrics: dict[str, Any]) -> None:
    print(
        f"[Eval @ step {metrics['step']}] "
        f"frobenius_norm={metrics['frobenius_norm']:.6f} "
        f"relative={metrics['relative_frobenius_norm']} "
        f"max_abs_diff={metrics['max_abs_diff']:.6f} "
        f"time={metrics['elapsed']:.2f}s"
    )


def evaluate_weight_change(
    llm: ESEvalLLM,
    original_state: dict[str, torch.Tensor],
    step: int,
    verbose: bool = False,
) -> dict[str, Any]:
    start = time.time()
    current_path: Path | None = None

    try:
        current_path = write_current_weights_to_disk(llm, "weight_change_eval_")
        current_state = load_state_dict(current_path)
        summary = compute_frobenius_summary(original_state, current_state)
    finally:
        if current_path is not None:
            current_path.unlink(missing_ok=True)

    if verbose and not summary["is_comparable"]:
        print_summary(summary, verbose=True)

    metrics = build_eval_metrics(
        step=step,
        summary=summary,
        elapsed=time.time() - start,
    )
    log_eval_line(metrics)
    return metrics


def run_es_evaluation(args: argparse.Namespace, run_dir: Path) -> int:
    args_path, updates_dir = resolve_run_paths(run_dir)
    run_config = load_json(args_path)

    model_name = args.model_name or run_config["model_name"]
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
    eval_interval = int(
        args.eval_interval
        if args.eval_interval is not None
        else run_config.get("eval_interval", 1)
    )
    iteration_files = get_iteration_files(updates_dir, args.max_iteration)
    final_step = int(load_json(iteration_files[-1])["iteration"])

    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    metrics_path = prepare_metrics_path(run_dir, args.metrics_path)

    print(f"Detected backend: es")
    print(f"Run directory: {run_dir}")
    print(f"Model: {model_name}")
    print(f"Precision: {precision_name}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Caching enabled: {caching}")
    print(f"Mirror sampling: {mirror_sampling}")
    print(f"Weight decay type: {weight_decay_type}")
    print(f"Weight decay lambda: {weight_decay_lambda}")
    print(f"Num engines: {num_engines}")
    print(f"Eval interval: {eval_interval}")
    print(f"Replaying {len(iteration_files)} iteration updates")

    llm = None
    original_path: Path | None = None

    try:
        llm = build_eval_llm(model_name, precision_name)

        if weight_decay_type != "none" and weight_decay_lambda > 0.0:
            llm.collective_rpc("snapshot_weights_to_cpu")

        original_path = write_current_weights_to_disk(llm, "original_model_weights_")
        original_state = load_state_dict(original_path)

        initial_metrics = build_eval_metrics(
            step=0,
            summary=compute_frobenius_summary(original_state, original_state),
            elapsed=0.0,
        )
        log_eval_line(initial_metrics)
        append_jsonl(metrics_path, initial_metrics)

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
                metrics = evaluate_weight_change(
                    llm=llm,
                    original_state=original_state,
                    step=iteration,
                    verbose=args.verbose,
                )
                append_jsonl(metrics_path, metrics)

        return 0
    finally:
        if original_path is not None:
            original_path.unlink(missing_ok=True)
        cleanup_llm(llm)
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def run_grpo_evaluation(args: argparse.Namespace, run_dir: Path) -> int:
    model_path, all_checkpoint_dirs = resolve_grpo_run_paths(run_dir)
    checkpoint_dirs = get_grpo_checkpoint_dirs(run_dir, args.max_iteration)

    base_model_name = args.model_name or load_text(model_path)
    precision_name = args.precision or infer_precision_from_config(all_checkpoint_dirs[0][1])

    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    metrics_path = prepare_metrics_path(run_dir, args.metrics_path)

    print(f"Detected backend: grpo")
    print(f"Run directory: {run_dir}")
    print(f"Base model: {base_model_name}")
    print(f"Precision: {precision_name}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Discovered {len(checkpoint_dirs)} checkpoint directories")
    print("Loading GRPO checkpoints in-place into one live vLLM model")

    llm = None
    original_path: Path | None = None

    try:
        llm = build_eval_llm(base_model_name, precision_name)
        original_path = write_current_weights_to_disk(llm, "original_model_weights_")
        original_state = load_state_dict(original_path)

        initial_metrics = build_eval_metrics(
            step=0,
            summary=compute_frobenius_summary(original_state, original_state),
            elapsed=0.0,
        )
        log_eval_line(initial_metrics)
        append_jsonl(metrics_path, initial_metrics)

        progress = tqdm(
            checkpoint_dirs,
            total=len(checkpoint_dirs),
            desc="Evaluating GRPO checkpoints",
        )
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

            metrics = evaluate_weight_change(
                llm=llm,
                original_state=original_state,
                step=step,
                verbose=args.verbose,
            )
            append_jsonl(metrics_path, metrics)

        return 0
    finally:
        if original_path is not None:
            original_path.unlink(missing_ok=True)
        cleanup_llm(llm)
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


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
