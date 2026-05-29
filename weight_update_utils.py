from collections.abc import Iterable
from itertools import zip_longest

import torch


def generate_noise(
    seed: int,
    shape: torch.Size | tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return torch.randn(shape, dtype=dtype, device=device, generator=gen)


@torch.inference_mode()
def apply_seeded_noise_to_parameters(
    parameters: Iterable[torch.nn.Parameter],
    seed: int,
    scale: float,
    caching: bool = False,
) -> None:
    seed = int(seed)
    scale = float(scale)
    noise_cache: dict[tuple[torch.device, torch.dtype, tuple[int, ...]], torch.Tensor] = {}

    for parameter in parameters:
        if caching:
            key = (parameter.device, parameter.dtype, tuple(parameter.shape))
            if key not in noise_cache:
                noise_cache[key] = generate_noise(
                    seed,
                    parameter.shape,
                    parameter.dtype,
                    parameter.device,
                )
            noise = noise_cache[key]
        else:
            noise = generate_noise(
                seed,
                parameter.shape,
                parameter.dtype,
                parameter.device,
            )

        parameter.add_(noise, alpha=scale)


@torch.inference_mode()
def update_parameters_from_seeds(
    parameters: Iterable[torch.nn.Parameter],
    seeds: list[int],
    coeffs: list[float],
    alpha: float,
    population_size: int,
    caching: bool = False,
    original_parameters: Iterable[torch.Tensor] | None = None,
    weight_decay_type: str = "none",
    weight_decay_lambda: float = 0.0,
) -> None:
    if len(seeds) != len(coeffs):
        raise ValueError("seeds and coeffs must have equal length")

    seeds = [int(seed) for seed in seeds]
    coeffs = [float(coeff) for coeff in coeffs]
    scale = float(alpha) / float(population_size)
    weight_decay_type, weight_decay_lambda = normalize_weight_decay_config(
        weight_decay_type=weight_decay_type,
        weight_decay_lambda=weight_decay_lambda,
    )

    accumulator_cache: dict[tuple[torch.device, torch.dtype, tuple[int, ...]], torch.Tensor] = {}

    if weight_decay_type == "none":
        parameter_pairs = ((parameter, None) for parameter in parameters)
    else:
        if original_parameters is None:
            raise ValueError(
                "original_parameters must be provided when weight decay to the initial model is enabled"
            )
        parameter_pairs = zip_longest(parameters, original_parameters, fillvalue=None)

    for parameter, original_parameter in parameter_pairs:
        if parameter is None or (weight_decay_type != "none" and original_parameter is None):
            raise ValueError(
                "parameters and original_parameters must have the same length when weight decay is enabled"
            )
        if caching:
            key = (parameter.device, parameter.dtype, tuple(parameter.shape))
            if key not in accumulator_cache:
                accumulator = torch.zeros_like(parameter)
                for seed, coeff in zip(seeds, coeffs):
                    noise = generate_noise(
                        seed,
                        parameter.shape,
                        parameter.dtype,
                        parameter.device,
                    )
                    accumulator.add_(noise, alpha=coeff)
                accumulator_cache[key] = accumulator
            accumulator = accumulator_cache[key]
        else:
            accumulator = torch.zeros_like(parameter)
            for seed, coeff in zip(seeds, coeffs):
                noise = generate_noise(
                    seed,
                    parameter.shape,
                    parameter.dtype,
                    parameter.device,
                )
                accumulator.add_(noise, alpha=coeff)

        parameter.add_(accumulator.to(dtype=parameter.dtype), alpha=scale)

        if weight_decay_type != "none":
            apply_reference_weight_decay_(
                parameter=parameter,
                reference_parameter=original_parameter,
                weight_decay_type=weight_decay_type,
                weight_decay_lambda=weight_decay_lambda,
                alpha=alpha,
            )


@torch.inference_mode()
def apply_reference_weight_decay_(
    parameter: torch.nn.Parameter,
    reference_parameter: torch.Tensor,
    weight_decay_type: str,
    weight_decay_lambda: float,
    alpha: float,
) -> None:
    reference_on_device = _copy_reference_parameter_to_device(reference_parameter, parameter)
    decay_step = float(alpha) * float(weight_decay_lambda)
    assert decay_step >= 0.0, f"decay_step must be non-negative, got decay_step={decay_step}"

    # Subtract refence parameter: w* <- w_t - w_ref
    parameter.sub_(reference_on_device)

    if weight_decay_type == "l2":
        assert decay_step < 1.0, f"decay_step must be less than 1.0 for l2 weight decay, got decay_step={decay_step}"
        parameter.mul_(1.0 - decay_step)
    elif weight_decay_type == "l1":
        parameter.copy_(
            torch.sign(parameter)
            * torch.clamp(torch.abs(parameter) - decay_step, min=0.0)
        )
    else:
        raise ValueError(
            f"Unsupported weight decay type {weight_decay_type!r}. Expected one of: l1, l2."
        )

    # Add reference parameter back: w_{t+1} <- w* + w_ref
    parameter.add_(reference_on_device)


def normalize_weight_decay_config(
    weight_decay_type: str | None,
    weight_decay_lambda: float,
) -> tuple[str, float]:
    decay_type = "none" if weight_decay_type is None else str(weight_decay_type).lower()
    if decay_type not in {"none", "l1", "l2"}:
        raise ValueError(
            f"Unsupported weight decay type {weight_decay_type!r}. Expected one of: none, l1, l2."
        )

    decay_lambda = float(weight_decay_lambda)
    if decay_lambda < 0.0:
        raise ValueError(
            f"weight_decay_lambda must be non-negative, got {weight_decay_lambda!r}"
        )

    if decay_type == "none" or decay_lambda == 0.0:
        return "none", 0.0

    return decay_type, decay_lambda


def _copy_reference_parameter_to_device(
    reference_parameter: torch.Tensor,
    parameter: torch.nn.Parameter,
) -> torch.Tensor:
    non_blocking = (
        reference_parameter.device.type == "cpu" and reference_parameter.is_pinned()
    )
    return reference_parameter.to(
        device=parameter.device,
        dtype=parameter.dtype,
        non_blocking=non_blocking,
    )
