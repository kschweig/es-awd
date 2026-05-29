import argparse
import json
from pathlib import Path
import sys

from datasets import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.countdown.template import RESPONSE_PROMPT, SYSTEM_MESSAGE, USER_TEMPLATE


DEFAULT_TRAIN_SIZE = 200
DEFAULT_DATA_SOURCE = "countdown"


def _resolve_path(script_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return script_dir / path


def _load_rows(path: Path) -> list[dict]:
    with path.open("r") as f:
        return json.load(f)


def _build_prompt(numbers: list[int], target: str) -> list[dict[str, str]]:
    user_content = USER_TEMPLATE.format(numbers=numbers, target=target)
    return [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE,
        },
        {
            "role": "user",
            "content": user_content,
        },
        {
            # Preserve the old countdown prompt construction's assistant prefill
            # so veRL sees the same prompt shape as ES training.
            "role": "assistant",
            "content": RESPONSE_PROMPT,
        },
    ]


def _convert_row(row: dict, *, data_source: str, split: str, index: int) -> dict:
    numbers = [int(number) for number in row["numbers"]]
    target = str(row["target"])
    solution = row["solution"]

    return {
        "id": row.get("id", f"{split}-{index}"),
        "data_source": data_source,
        "prompt": _build_prompt(numbers, target),
        "ability": "arithmetic",
        "reward_model": {
            "ground_truth": solution,
            "style": "rule",
        },
        "extra_info": {
            "numbers": numbers,
            "target": target,
            "solution": solution,
            "source_split": split,
            "source_index": index,
            "response_prefix": RESPONSE_PROMPT,
        },
    }


def _write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(path))
    print(f"Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create countdown train/test parquet files in the prompt/reward "
            "layout used by veRL-style training, while preserving the old "
            "chat-formatted prompt structure."
        )
    )
    parser.add_argument(
        "--input",
        default="countdown.json",
        help="Path to the countdown JSON dataset. Relative paths are resolved from this script's directory.",
    )
    parser.add_argument(
        "--train-output",
        default="train.parquet",
        help="Path for the training parquet file. Relative paths are resolved from this script's directory.",
    )
    parser.add_argument(
        "--test-output",
        default="test.parquet",
        help="Path for the test parquet file. Relative paths are resolved from this script's directory.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=DEFAULT_TRAIN_SIZE,
        help="Number of examples to place in the training split.",
    )
    parser.add_argument(
        "--data-source",
        default=DEFAULT_DATA_SOURCE,
        help="Value to store in the parquet data_source field.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_path = _resolve_path(script_dir, args.input)
    train_output_path = _resolve_path(script_dir, args.train_output)
    test_output_path = _resolve_path(script_dir, args.test_output)

    rows = _load_rows(input_path)
    train_rows = [
        _convert_row(row, data_source=args.data_source, split="train", index=index)
        for index, row in enumerate(rows[: args.train_size])
    ]
    test_rows = [
        _convert_row(
            row,
            data_source=args.data_source,
            split="test",
            index=index + args.train_size,
        )
        for index, row in enumerate(rows[args.train_size :])
    ]

    _write_parquet(train_rows, train_output_path)
    _write_parquet(test_rows, test_output_path)

    print(
        "Split summary: "
        f"train={len(train_rows)} rows, test={len(test_rows)} rows, "
        f"total={len(rows)} rows"
    )


if __name__ == "__main__":
    main()
