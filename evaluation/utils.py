import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from vllm import LLM, SamplingParams

from weight_update_utils import update_parameters_from_seeds


PRECISION_CHOICES = ("float16", "bfloat16", "float32")
EMBEDDING_LAYER_NAME = "embedding"
LM_HEAD_LAYER_NAME = "lm_head"
WEIGHT_EMBEDDING_PATTERN = re.compile(
    r"(?:^|\.)(?:embed_tokens|tok_embeddings|wte|wpe|embeddings|embed_in)(?:\.|$)"
)
WEIGHT_LAYER_PATTERN = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)")
WEIGHT_LM_HEAD_PATTERN = re.compile(r"(?:^|\.)(?:lm_head|embed_out)(?:\.|$)")


def format_layer_name(layer_idx: int) -> str:
    return f"layer_{int(layer_idx)}"


def build_numbered_layer_names(num_layers: int, start_idx: int = 0) -> list[str]:
    return [
        format_layer_name(layer_idx)
        for layer_idx in range(start_idx, start_idx + num_layers)
    ]


def build_hidden_layer_names(num_hidden_state_tensors: int) -> list[str]:
    if num_hidden_state_tensors <= 0:
        return []
    return [EMBEDDING_LAYER_NAME] + build_numbered_layer_names(
        num_hidden_state_tensors - 1,
        start_idx=1,
    )


def build_attention_layer_names(num_attention_layers: int) -> list[str]:
    return build_numbered_layer_names(num_attention_layers)


def build_layer_stat_dict(layer_names: list[str], values: list[float]) -> dict[str, float]:
    return {
        layer_name: float(value)
        for layer_name, value in zip(layer_names, values)
    }


def infer_weight_layer_name(parameter_name: str) -> str | None:
    if WEIGHT_EMBEDDING_PATTERN.search(parameter_name):
        return EMBEDDING_LAYER_NAME

    match = WEIGHT_LAYER_PATTERN.search(parameter_name)
    if match is not None:
        return format_layer_name(int(match.group(1)))

    if WEIGHT_LM_HEAD_PATTERN.search(parameter_name):
        return LM_HEAD_LAYER_NAME

    return None


def layer_name_sort_key(layer_name: str) -> tuple[int, int | str]:
    if layer_name == EMBEDDING_LAYER_NAME:
        return (0, 0)

    prefix = "layer_"
    if layer_name.startswith(prefix):
        suffix = layer_name[len(prefix):]
        if suffix.isdigit():
            return (1, int(suffix))

    if layer_name == LM_HEAD_LAYER_NAME:
        return (2, 0)

    return (3, layer_name)


class ESEvalLLM(LLM):
    def __init__(self, *args, **kwargs):
        model_name = kwargs.get("model")
        self._cached_repr = f"ESEvalLLM(model={model_name!r}, status='initializing')"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)
        resolved_model_name = getattr(self.model_config, "model", model_name)
        self._cached_repr = f"ESEvalLLM(model={resolved_model_name!r})"

    def __repr__(self) -> str:
        return self._cached_repr


class EvalWorkerExtension:
    @torch.inference_mode()
    def snapshot_weights_to_cpu(self):
        self.cpu_weight_snapshot = []
        for parameter in self.model_runner.model.parameters():
            cpu_copy = parameter.detach().to(device="cpu", copy=True)
            if torch.cuda.is_available():
                cpu_copy = cpu_copy.pin_memory()
            self.cpu_weight_snapshot.append(cpu_copy)
        return True

    @torch.inference_mode()
    def perturb_and_restore_self_weights(
        self,
        seed,
        noise_scale,
        caching=False,
        negate=False,
    ):
        seed = int(seed)
        scale = -float(noise_scale) if negate else float(noise_scale)
        noise_cache: dict[
            tuple[torch.device, torch.dtype, tuple[int, ...]],
            torch.Tensor,
        ] = {}

        for parameter in self.model_runner.model.parameters():
            if caching:
                key = (parameter.device, parameter.dtype, tuple(parameter.shape))
                if key not in noise_cache:
                    generator = torch.Generator(device=parameter.device)
                    generator.manual_seed(seed)
                    noise_cache[key] = torch.randn(
                        parameter.shape,
                        dtype=parameter.dtype,
                        device=parameter.device,
                        generator=generator,
                    )
                noise = noise_cache[key]
            else:
                generator = torch.Generator(device=parameter.device)
                generator.manual_seed(seed)
                noise = torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=parameter.device,
                    generator=generator,
                )

            parameter.add_(noise, alpha=scale)
            parameter.add_(noise, alpha=-scale)
        return True

    def update_weights_from_seeds(
        self,
        seeds,
        coeffs,
        alpha,
        population_size,
        caching=False,
        weight_decay_type="none",
        weight_decay_lambda=0.0,
    ):
        update_parameters_from_seeds(
            self.model_runner.model.parameters(),
            seeds=seeds,
            coeffs=coeffs,
            alpha=alpha,
            population_size=population_size,
            caching=caching,
            original_parameters=getattr(self, "cpu_weight_snapshot", None),
            weight_decay_type=weight_decay_type,
            weight_decay_lambda=weight_decay_lambda,
        )
        return True

    def write_weights_to_disk(self, path):
        torch.save(self.model_runner.model.state_dict(), path)
        return True

    @torch.inference_mode()
    def load_hf_checkpoint_weights(self, checkpoint_dir):
        checkpoint_path = Path(checkpoint_dir)
        model = self.model_runner.model

        if not hasattr(model, "load_weights"):
            raise AttributeError(
                f"Model class {type(model).__name__} does not expose load_weights()"
            )

        loaded_params = model.load_weights(iter_hf_checkpoint_weights(checkpoint_path))
        return {
            "checkpoint_dir": str(checkpoint_path),
            "loaded_param_count": len(loaded_params),
        }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def load_text(path: Path) -> str:
    with path.open("r") as handle:
        return handle.read().strip()


def parse_global_step(name: str) -> int | None:
    match = re.search(r"_global_step_(\d+)$", name)
    if match is None:
        return None
    return int(match.group(1))


def resolve_run_paths(
    run_dir: Path,
    *,
    require_final_weights: bool = False,
) -> tuple[Path, ...]:
    args_path = run_dir / "args.json"
    updates_dir = run_dir / "iteration_updates"

    if not args_path.is_file():
        raise FileNotFoundError(f"Missing run config: {args_path}")
    if not updates_dir.is_dir():
        raise FileNotFoundError(f"Missing iteration updates directory: {updates_dir}")

    if not require_final_weights:
        return args_path, updates_dir

    final_weights_path = run_dir / "final_model_weights.pt"
    if not final_weights_path.is_file():
        raise FileNotFoundError(f"Missing final weights file: {final_weights_path}")
    return args_path, updates_dir, final_weights_path


def get_grpo_checkpoint_dirs(
    run_dir: Path,
    max_iteration: int | None,
) -> list[tuple[int, Path]]:
    checkpoint_dirs: list[tuple[int, Path]] = []
    for path in run_dir.iterdir():
        if not path.is_dir():
            continue
        step = parse_global_step(path.name)
        if step is None:
            continue
        if max_iteration is not None and step > max_iteration:
            continue
        if not (path / "config.json").is_file():
            continue
        checkpoint_dirs.append((step, path))

    checkpoint_dirs.sort(key=lambda item: item[0])
    if not checkpoint_dirs:
        raise ValueError(f"No GRPO checkpoint directories found in {run_dir}")
    return checkpoint_dirs


def resolve_grpo_run_paths(run_dir: Path) -> tuple[Path, list[tuple[int, Path]]]:
    model_path = run_dir / "model.txt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing GRPO base model file: {model_path}")
    checkpoint_dirs = get_grpo_checkpoint_dirs(run_dir, max_iteration=None)
    return model_path, checkpoint_dirs


def detect_run_backend(run_dir: Path) -> str:
    has_es_layout = (run_dir / "args.json").is_file() and (run_dir / "iteration_updates").is_dir()

    has_grpo_layout = False
    if (run_dir / "model.txt").is_file():
        try:
            get_grpo_checkpoint_dirs(run_dir, max_iteration=None)
            has_grpo_layout = True
        except ValueError:
            has_grpo_layout = False

    if has_es_layout and has_grpo_layout:
        raise ValueError(
            f"Run directory {run_dir} matches both ES and GRPO layouts; cannot auto-detect."
        )
    if has_es_layout:
        return "es"
    if has_grpo_layout:
        return "grpo"

    raise FileNotFoundError(
        "Could not auto-detect run format. Expected either "
        "<run_dir>/args.json + iteration_updates/ for ES, or "
        "<run_dir>/model.txt + *_global_step_<N>/ for GRPO."
    )


def get_iteration_files(updates_dir: Path, max_iteration: int | None) -> list[Path]:
    files = sorted(updates_dir.glob("iteration_*.json"))
    if max_iteration is not None:
        files = [
            path
            for path in files
            if int(load_json(path)["iteration"]) <= max_iteration
        ]

    if not files:
        raise ValueError(f"No iteration update files found in {updates_dir}")

    return files


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    load_kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, mmap=True, weights_only=True, **load_kwargs)
    except TypeError:
        return torch.load(path, **load_kwargs)


def infer_precision_from_config(model_dir: Path, default: str = "float16") -> str:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return default

    config = load_json(config_path)
    for key in ("torch_dtype", "dtype"):
        value = config.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.replace("torch.", "").lower()
        if normalized in {"float16", "half", "fp16"}:
            return "float16"
        if normalized in {"bfloat16", "bf16"}:
            return "bfloat16"
        if normalized in {"float32", "float", "fp32"}:
            return "float32"

    return default


def iter_hf_checkpoint_weights(
    checkpoint_dir: Path,
) -> Iterable[tuple[str, torch.Tensor]]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    single_path = checkpoint_dir / "model.safetensors"

    if index_path.is_file():
        payload = load_json(index_path)
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty weight_map in {index_path}")

        shard_to_keys: dict[str, list[str]] = defaultdict(list)
        for key, shard_name in weight_map.items():
            shard_to_keys[str(shard_name)].append(str(key))

        for shard_name in sorted(shard_to_keys):
            shard_path = checkpoint_dir / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing safetensors shard: {shard_path}")
            with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
                for key in shard_to_keys[shard_name]:
                    yield key, shard.get_tensor(key)
        return

    if single_path.is_file():
        with safe_open(str(single_path), framework="pt", device="cpu") as shard:
            for key in shard.keys():
                yield key, shard.get_tensor(key)
        return

    raise FileNotFoundError(
        f"Checkpoint {checkpoint_dir} is missing model.safetensors or model.safetensors.index.json"
    )


def replay_perturb_restore(
    llm: ESEvalLLM,
    seeds: list[int],
    sigma: float,
    num_engines: int,
    mirror_sampling: bool,
    caching: bool,
) -> None:
    negates = [False, True] if mirror_sampling else [False]
    for negate in negates:
        for batch_start in range(0, len(seeds), num_engines):
            batch = seeds[batch_start : batch_start + num_engines]
            if not batch:
                continue
            llm.collective_rpc(
                "perturb_and_restore_self_weights",
                args=(int(batch[0]), sigma, caching, negate),
            )


def generate_rollouts(llm: ESEvalLLM, prompts, seed: int, max_tokens: int = 1024, temperature: float = 0.0):
    sampling_params = SamplingParams(temperature=temperature, seed=seed, max_tokens=max_tokens)
    return llm.generate(
        prompts,
        sampling_params,
        use_tqdm=False,
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


def should_evaluate(step: int, final_step: int, eval_interval: int) -> bool:
    if step == final_step:
        return True
    return eval_interval > 0 and step % eval_interval == 0


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def make_temp_path(prefix: str, suffix: str = ".pt") -> Path:
    fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return Path(temp_path)
