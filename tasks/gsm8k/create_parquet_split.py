import argparse
from pathlib import Path
import sys

from datasets import Dataset, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.gsm8k.data import _to_canonical_answer_text

DEFAULT_DATASET_NAME = "openai/gsm8k"
DEFAULT_CONFIG_NAME = "main"
DEFAULT_TRAIN_SOURCE_SPLIT = "train"
DEFAULT_TEST_SOURCE_SPLIT = "test"
DEFAULT_TRAIN_SIZE = 200
DEFAULT_MAX_TEST_SIZE = 500
DEFAULT_SEED = 0


def _resolve_path(script_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return script_dir / path


def _build_prompt(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Solve this grade-school math word problem step by step.\n\n"
                f"Problem: {question.strip()}\n\n"
                "Write short calculations and check the final result.\n"
                'On the last line, write only: #### <number>'
            ),
        }
    ]


def _convert_row(
    row: dict,
    *,
    data_source: str,
    source_split: str,
    source_index: int,
) -> dict:
    question = row["question"].strip()
    answer = row["answer"]
    target_answer = _to_canonical_answer_text(answer)

    return {
        "id": f"{source_split}-{source_index}",
        "data_source": data_source,
        "prompt": _build_prompt(question),
        "ability": "math",
        "reward_model": {
            "ground_truth": target_answer,
            "style": "rule",
        },
        "extra_info": {
            "question": question,
            "answer": answer,
            "split": source_split,
            "index": source_index,
            "config": DEFAULT_CONFIG_NAME,
        },
    }


def _write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(path))
    print(f"Wrote {len(rows)} rows to {path}")


def _select_rows(dataset, size: int, *, shuffle: bool, seed: int):
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    return dataset.select(range(min(len(dataset), size)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create local GSM8K train/test parquet files in the prompt/reward "
            "layout used by the main tasks."
        )
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset name to load.",
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Hugging Face dataset config to load.",
    )
    parser.add_argument(
        "--train-source-split",
        default=DEFAULT_TRAIN_SOURCE_SPLIT,
        help="Source split to use for the local training parquet.",
    )
    parser.add_argument(
        "--test-source-split",
        default=DEFAULT_TEST_SOURCE_SPLIT,
        help="Source split to use for the local test parquet.",
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
        "--max-test-size",
        type=int,
        default=DEFAULT_MAX_TEST_SIZE,
        help="Maximum number of examples to place in the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Shuffle seed used when --shuffle is enabled.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle each source split deterministically before taking rows.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    train_output_path = _resolve_path(script_dir, args.train_output)
    test_output_path = _resolve_path(script_dir, args.test_output)

    dataset_kwargs = {}
    if args.config_name:
        dataset_kwargs["name"] = args.config_name

    train_dataset = load_dataset(
        args.dataset_name,
        **dataset_kwargs,
        split=args.train_source_split,
    )
    test_dataset = load_dataset(
        args.dataset_name,
        **dataset_kwargs,
        split=args.test_source_split,
    )

    selected_train = _select_rows(
        train_dataset,
        args.train_size,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    selected_test = _select_rows(
        test_dataset,
        args.max_test_size,
        shuffle=args.shuffle,
        seed=args.seed + 1,
    )

    data_source = (
        f"{args.dataset_name}:{args.config_name}"
        if args.config_name
        else args.dataset_name
    )

    train_rows = [
        _convert_row(
            row,
            data_source=data_source,
            source_split=args.train_source_split,
            source_index=index,
        )
        for index, row in enumerate(selected_train)
    ]
    test_rows = [
        _convert_row(
            row,
            data_source=data_source,
            source_split=args.test_source_split,
            source_index=index,
        )
        for index, row in enumerate(selected_test)
    ]

    _write_parquet(train_rows, train_output_path)
    _write_parquet(test_rows, test_output_path)

    print(
        "Split summary: "
        f"train={len(train_rows)} rows from {args.train_source_split}, "
        f"test={len(test_rows)} rows from {args.test_source_split}, "
        f"config={args.config_name}, shuffled={args.shuffle}, seed={args.seed}"
    )


if __name__ == "__main__":
    main()
