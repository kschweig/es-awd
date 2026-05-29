import argparse
from datetime import datetime
import json
import os
import random
import time
from typing import List, Dict, Any

import numpy as np
import ray
import torch
import wandb
from transformers import AutoTokenizer
from vllm import SamplingParams

from distributed_utils import launch_engines, cleanup

# Default Hyperparameters
SIGMA = 0.001
ALPHA = 0.0005
POPULATION_SIZE = 30
NUM_ENGINES = 4
NUM_ITERATIONS = 500
EXPERIMENT_DIR = "es-ft-experiment"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Accelerated ES Fine-tuning with static scheduling"
    )

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--task",
        type=str,
        choices=["countdown", "gsm8k", "proofwriter"],
        default="countdown",
    )
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--mirror_sampling", action="store_true")
    parser.add_argument("--num_engines", type=int, default=NUM_ENGINES)
    parser.add_argument("--num_iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--experiment_dir", type=str, default=EXPERIMENT_DIR)
    parser.add_argument("--run_name", type=str, default=None, help="Optional name for this run (overrides default naming scheme)")
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument("--caching", action="store_true", help="Enable noise caching for faster perturbations (increases memory usage)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose logs")
    parser.add_argument("--global_seed", type=int, help="Global random seed")
    parser.add_argument("--precision", type=str, choices=["float16", "bfloat16", "float32"],
                        default="float16", help="Precision for model weights")
    parser.add_argument(
        "--weight_decay_type",
        type=str,
        choices=["none", "l1", "l2"],
        default="none",
        help="Decoupled weight decay type applied to the parameter difference from the initial model.",
    )
    parser.add_argument(
        "--weight_decay_lambda",
        type=float,
        default=0.0,
        help="Regularization strength for decoupled weight decay toward the initial model.",
    )
    # Evaluation args
    parser.add_argument("--eval_interval", type=int, default=25,
                        help="Evaluate every N iterations (0 disables)")
    parser.add_argument("--eval_batch_size", type=int, default=512,
                        help="Batch size for evaluation generation")
    # WandB logging args
    parser.add_argument("--wandb_project", type=str, default="precision-tests")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default="online")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # check if population size is even when mirror sampling is enabled
    if args.mirror_sampling:
        assert args.population_size % 2 == 0, "Population size must be even when mirror sampling is enabled, but got {}".format(args.population_size)

    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)
        torch.cuda.manual_seed_all(args.global_seed)
        os.environ["PYTHONHASHSEED"] = str(args.global_seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    if args.weight_decay_lambda < 0.0:
        parser.error("--weight_decay_lambda must be non-negative")

    return args

def generate_rollouts(llm, prompts, seed: int):
    sampling_params = SamplingParams(temperature=0.0, seed=seed, max_tokens=1024)
    return llm.generate.remote(
        prompts, 
        sampling_params, 
        use_tqdm=False
    )

def count_generated_tokens(output) -> int:
    if not output.outputs:
        return 0
    first_output = output.outputs[0]
    token_ids = getattr(first_output, "token_ids", None)
    if token_ids is not None:
        return len(token_ids)
    tokens = getattr(first_output, "tokens", None)
    if tokens is not None:
        return len(tokens)
    text = getattr(first_output, "text", "")
    return len(text.split())

def postprocess_outputs(outputs, task_data):
    rewards = []
    avg_rewards = []
    format_rewards = []
    answer_rewards = []
    output_token_counts = []
    for output, data in zip(outputs, task_data):
        r, _ = evaluate_reward(output.outputs[0].text, data)
        rewards.append(r)
        avg_rewards.append(r["reward"])
        output_token_counts.append(count_generated_tokens(output))
        if "reward_info" in r:
            format_rewards.append(r["reward_info"].get("format_reward", 0.0))
            answer_rewards.append(r["reward_info"].get("answer_reward", 0.0))
    avg_format = float(np.mean(format_rewards)) if format_rewards else 0.0
    avg_answer = float(np.mean(answer_rewards)) if answer_rewards else 0.0
    accuracy = (sum(1 for a in answer_rewards if a > 0) / len(answer_rewards) * 100.0) if answer_rewards else 0.0
    avg_output_tokens = float(np.mean(output_token_counts)) if output_token_counts else 0.0
    std_output_tokens = float(np.std(output_token_counts)) if output_token_counts else 0.0
    return {
        "rewards": rewards,
        "avg_reward": float(np.mean(avg_rewards)) if avg_rewards else 0.0,
        "avg_format": avg_format,
        "avg_answer": avg_answer,
        "accuracy": accuracy,
        "avg_output_tokens": avg_output_tokens,
        "std_output_tokens": std_output_tokens,
        "output_token_counts": output_token_counts,
    }

def evaluate_model(llm, eval_data: List[Dict[str, Any]], wandb_run, step: int, args):
    if not eval_data:
        return
    batch_size = max(1, args.eval_batch_size)
    eval_seed = args.global_seed if args.global_seed is not None else 999

    all_rewards = []
    format_rewards = []
    answer_rewards = []
    output_token_counts = []
    start = time.time()

    for b in range(0, len(eval_data), batch_size):
        batch = eval_data[b:b+batch_size]
        prompts = [d["context"] for d in batch]
        outputs = ray.get(generate_rollouts(llm, prompts, seed=eval_seed))
        
        for idx, (out, data) in enumerate(zip(outputs, batch)):
            r, response = evaluate_reward(out.outputs[0].text, data)

            # for inspection
            if idx == 0:
                print(f"Eval Sample Response:\n{response}\n---")
                print(f"Target Answer: {data['target']}\n===")
                print(f"The rewards: {r}\n===")

            all_rewards.append(r["reward"])
            output_token_counts.append(count_generated_tokens(out))
            if "reward_info" in r:
                format_rewards.append(r["reward_info"].get("format_reward", 0.0))
                answer_rewards.append(r["reward_info"].get("answer_reward", 0.0))
    elapsed = time.time() - start

    avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    std_reward = float(np.std(all_rewards)) if all_rewards else 0.0
    min_reward = float(np.min(all_rewards)) if all_rewards else 0.0
    max_reward = float(np.max(all_rewards)) if all_rewards else 0.0
    avg_format = float(np.mean(format_rewards)) if format_rewards else 0.0
    avg_answer = float(np.mean(answer_rewards)) if answer_rewards else 0.0
    accuracy = (sum(1 for a in answer_rewards if a > 0) / len(answer_rewards) * 100.0) if answer_rewards else 0.0
    avg_output_tokens = float(np.mean(output_token_counts)) if output_token_counts else 0.0
    std_output_tokens = float(np.std(output_token_counts)) if output_token_counts else 0.0

    print(f"[Eval @ step {step}] avg_reward={avg_reward:.4f} ± {std_reward:.4f} "
          f"range=[{min_reward:.4f},{max_reward:.4f}] format={avg_format:.4f} "
          f"answer={avg_answer:.4f} acc={accuracy:.1f}% "
          f"output_tokens={avg_output_tokens:.2f} ± {std_output_tokens:.2f} time={elapsed:.2f}s")

    wandb_run.log({
        "eval/avg_reward": avg_reward,
        "eval/std_reward": std_reward,
        "eval/min_reward": min_reward,
        "eval/max_reward": max_reward,
        "eval/format_reward": avg_format,
        "eval/answer_reward": avg_answer,
        "eval/accuracy": accuracy,
        "eval/output_tokens_mean": avg_output_tokens,
        "eval/output_tokens_std": std_output_tokens,
        "eval/time": elapsed,
    }, step=step)

def log_train(wandb_run, pop_perfs, step):
    # Aggregate
    all_avg_rewards = [v["avg_reward"] for v in pop_perfs.values()]
    mean_reward = float(np.mean(all_avg_rewards)) if all_avg_rewards else 0.0
    std_reward = float(np.std(all_avg_rewards)) if all_avg_rewards else 0.0
    min_reward = float(np.min(all_avg_rewards)) if all_avg_rewards else 0.0
    max_reward = float(np.max(all_avg_rewards)) if all_avg_rewards else 0.0

    # Also aggregate format and answer rewards and accuracy over seeds
    all_avg_formats = [v.get("avg_format", 0.0) for v in pop_perfs.values()]
    all_avg_answers = [v.get("avg_answer", 0.0) for v in pop_perfs.values()]
    all_accuracies = [v.get("accuracy", 0.0) for v in pop_perfs.values()]
    all_output_token_counts = [
        token_count
        for v in pop_perfs.values()
        for token_count in v.get("output_token_counts", [])
    ]
    mean_format = float(np.mean(all_avg_formats)) if all_avg_formats else 0.0
    mean_answer = float(np.mean(all_avg_answers)) if all_avg_answers else 0.0
    mean_accuracy = float(np.mean(all_accuracies)) if all_accuracies else 0.0
    mean_output_tokens = float(np.mean(all_output_token_counts)) if all_output_token_counts else 0.0
    std_output_tokens = float(np.std(all_output_token_counts)) if all_output_token_counts else 0.0

    print(
        f"Mean reward: {mean_reward:.4f}, std: {std_reward:.4f}, format: {mean_format:.4f}, "
        f"answer: {mean_answer:.4f}, acc: {mean_accuracy:.1f}%, "
        f"output_tokens: {mean_output_tokens:.2f} ± {std_output_tokens:.2f}"
    )

    wandb_run.log({
        "train/avg_reward": mean_reward,
        "train/std_reward": std_reward,
        "train/min_reward": min_reward,
        "train/max_reward": max_reward,
        "train/format_reward": mean_format,
        "train/answer_reward": mean_answer,
        "train/accuracy": mean_accuracy,
        "train/output_tokens_mean": mean_output_tokens,
        "train/output_tokens_std": std_output_tokens,
    }, step=step)

    return mean_reward, std_reward

def write_iteration_update(logging_dir: str, iteration: int, seeds: List[int], coeffs: List[float]):
    updates_dir = os.path.join(logging_dir, "iteration_updates")
    os.makedirs(updates_dir, exist_ok=True)

    payload = {
        "iteration": int(iteration),
        "seeds": [int(seed) for seed in seeds],
        "update_coefficients": [float(coeff) for coeff in coeffs],
    }

    update_path = os.path.join(updates_dir, f"iteration_{iteration:06d}.json")
    with open(update_path, "w") as f:
        json.dump(payload, f, indent=2)

def main(args):
    # Set up logging
    if args.run_name:
        logging_dir = os.path.join(args.experiment_dir, args.run_name)
    else:
        dt = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logging_dir = os.path.join(args.experiment_dir, 
                                   f"{args.model_name.split('/')[-1]}_{args.task}_seed{args.global_seed}_popsize{args.population_size}_ms{args.mirror_sampling}")
        if args.weight_decay_type != "none":
            logging_dir += f"_{args.weight_decay_type}wd{args.weight_decay_lambda}"
        logging_dir += f"_{dt}"
    os.makedirs(logging_dir, exist_ok=False)
    # Save args for reproducibility
    with open(f"{logging_dir}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    # Initialize WandB
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=os.path.basename(logging_dir),
        config=vars(args),
        dir=logging_dir,
        mode=args.wandb_mode,
    )

    # Data loading and pre-processing
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_data, eval_data = get_data(tokenizer)
    prompts = [d["context"] for d in train_data]

    # start the ray engines and initialize inter-engine communication groups
    engines, pgs = launch_engines(args.num_engines, args.model_name, precision=args.precision)

    if args.weight_decay_type != "none" and args.weight_decay_lambda > 0.0:
        ray.get(engines[0].collective_rpc.remote("snapshot_weights_to_cpu"))
        if args.verbose:
            print(
                "Snapshotted the initial model weights on engine 0 for "
                f"{args.weight_decay_type} decay toward the original model."
            )

    # Initial evaluation before training
    evaluate_model(engines[0], eval_data, wandb_run, 0, args)

    # Main ES loop with static scheduling
    for i in range(1, args.num_iterations + 1):
        print(f"\n\n=== Generation {i} (static scheduling) ===")
        total_iter_start = time.time()
        perturb_weights_time_s, generate_time_s = 0.0, 0.0

        # Deterministic per-iteration seed list
        loop_seed = (args.global_seed or 42) + i    # controls the seeds used for perturbations in this iteration
        gen_seed = (args.global_seed or 42) + i     # controls the generation seed for all rollouts in this iteration
        loop_rng = np.random.default_rng(seed=loop_seed)

        if args.mirror_sampling:
            seeds = loop_rng.integers(0, 2**30, size=args.population_size // 2, dtype=np.int64).tolist()
        else:
            seeds = loop_rng.integers(0, 2**30, size=args.population_size, dtype=np.int64).tolist()

        # Population performances for logging and coefficient calculation
        pop_perfs: Dict[int | tuple[int, bool], Dict[str, Any]] = {}

        # Static batching: issue exactly one seed per engine per batch in fixed order, then wait
        for negate in [False, True] if args.mirror_sampling else [False]:
            for b in range(0, len(seeds), args.num_engines):
                batch = seeds[b:b+args.num_engines]
                # 1) Perturb
                perturb_start = time.time()
                ray.get([
                    engines[eng_idx].collective_rpc.remote("perturb_self_weights", args=(int(seed), args.sigma, args.caching, negate))
                    for eng_idx, seed in enumerate(batch)
                ])
                perturb_weights_time_s += time.time() - perturb_start

                # 2) Generate with fixed generation seed tied to iteration
                generate_start = time.time()
                handles = [
                    generate_rollouts(engines[eng_idx], prompts, seed=gen_seed)
                    for eng_idx, _ in enumerate(batch)
                ]
                # 3) Collect outputs in the same order
                outputs_per_engine = ray.get(handles)
                generate_time_s += time.time() - generate_start
                # 4) Restore weights
                perturb_start = time.time()
                ray.get([
                    engines[eng_idx].collective_rpc.remote("restore_self_weights", args=(int(seed), args.sigma, args.caching, negate))
                    for eng_idx, seed in enumerate(batch)
                ])
                perturb_weights_time_s += time.time() - perturb_start
                # 5) Score and record
                for eng_idx, seed in enumerate(batch):
                    metrics = postprocess_outputs(outputs_per_engine[eng_idx], train_data)
                    key = (int(seed), negate) if args.mirror_sampling else int(seed)
                    pop_perfs[key] = metrics

        mean_reward, std_reward = log_train(wandb_run, pop_perfs, i)

        if args.verbose:
            print(f"Perturbed and restored weights in {perturb_weights_time_s:.3f}s total")
            print(f"Generated outputs in {generate_time_s:.3f}s total")

        # Compute coefficients for each seed based on normalized rewards (z-scaling)
        coeffs = []
        for seed in seeds:
            if args.mirror_sampling:
                pos_reward = (pop_perfs[(seed, False)]["avg_reward"] - mean_reward) / (std_reward + 1e-8)
                neg_reward = (pop_perfs[(seed, True)]["avg_reward"] - mean_reward) / (std_reward + 1e-8)
                reward = pos_reward - neg_reward
            else:
                reward = (pop_perfs[seed]["avg_reward"] - mean_reward) / (std_reward + 1e-8)
            coeffs.append(reward)

        # save the seeds and coefficients for this iteration to be able to reconstruct model updates later if needed
        write_iteration_update(logging_dir, i, seeds, coeffs)

        # Update weights on engine 0
        perturb_start = time.time()
        ray.get(engines[0].collective_rpc.remote(
            "update_weights_from_seeds",
            args=(
                seeds,
                coeffs,
                args.alpha,
                args.population_size,
                args.caching,
                args.weight_decay_type,
                args.weight_decay_lambda,
            )
        ))
        if args.verbose:
            print(f"Applied updates in {(time.time() - perturb_start):.3f}s")

        # Broadcast weights to all other engines
        broadcast_start = time.time()
        ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in engines])
        torch.cuda.synchronize()
        if args.verbose:
            print(f"Broadcasted weights in {(time.time() - broadcast_start):.3f}s")

        print(f"Iteration {i} finished in {time.time() - total_iter_start:.3f}s")

        if (args.eval_interval > 0 and (i % args.eval_interval == 0)) or (i == args.num_iterations):
            evaluate_model(engines[0], eval_data, wandb_run, i, args)

    # write final model weights to disk
    final_weights_path = os.path.join(logging_dir, "final_model_weights.pt")
    ray.get(engines[0].collective_rpc.remote("write_weights_to_disk", args=(final_weights_path,)))
    print(f"Final model weights saved to {final_weights_path}")

    # Clean up Ray actors and placement groups, and finish the WandB run
    cleanup(engines, pgs, wandb_run)

if __name__ == "__main__":
    args = parse_args()

    # choose task to train and evaluate on
    if args.task == "countdown":
        from tasks.countdown.data import get_data, evaluate_reward
    elif args.task == "gsm8k":
        from tasks.gsm8k.data import get_data, evaluate_reward
    elif args.task == "proofwriter":
        from tasks.proofwriter.data import get_data, evaluate_reward
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    main(args)
