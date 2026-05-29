import argparse
from pathlib import Path
import sys

from datasets import Dataset, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.proofwriter.reward_function import canonicalize_answer

DEFAULT_DATASET_NAME = "tasksource/proofwriter"
DEFAULT_SOURCE_SPLIT = "train"
DEFAULT_TRAIN_SIZE = 200
DEFAULT_MAX_TEST_SIZE = 1000
DEFAULT_SEED = 0


def _resolve_path(script_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return script_dir / path


def _build_prompt(theory: str, question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "You are given a theory and a question. Determine whether the question is "
                "entailed by the theory.\n\n"
                "First write your reasoning.\n"
                "On the last line, write exactly one of these formats:\n"
                "Answer: True\n"
                "Answer: False\n"
                "Answer: Unknown\n"
                "Do not add any other words, punctuation, or explanation after the final label.\n\n"
                f"Theory: {theory}\n"
                f"Question: {question}"
            ),
        },
        {
            "role": "assistant",
            "content": "Reasoning:",
        },
    ]


def _convert_row(row: dict) -> dict:
    answer = canonicalize_answer(row["answer"])
    if answer is None:
        raise ValueError(f"Unsupported ProofWriter answer: {row['answer']!r}")

    return {
        "id": row["id"],
        "data_source": DEFAULT_DATASET_NAME,
        "prompt": _build_prompt(row["theory"], row["question"]),
        "ability": "logical_reasoning",
        "reward_model": {
            "ground_truth": answer,
            "style": "rule",
        },
        "extra_info": {
            "question": row["question"],
            "theory": row["theory"],
            "answer": answer,
            "maxD": int(row["maxD"]),
            "NFact": int(row["NFact"]),
            "NRule": int(row["NRule"]),
            "QDep": None if row["QDep"] is None else int(row["QDep"]),
            "QLen": None if row["QLen"] is None else float(row["QLen"]),
            "allProofs": row["allProofs"],
            "config": row["config"],
            "source_split": DEFAULT_SOURCE_SPLIT,
        },
    }


def _write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(path))
    print(f"Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create local ProofWriter train/test parquet files in the same prompt/reward "
            "layout used by other main tasks."
        )
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset name to load.",
    )
    parser.add_argument(
        "--source-split",
        default=DEFAULT_SOURCE_SPLIT,
        help="Source split to sample from before creating local train/test parquet files.",
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
        help="Shuffle seed used before taking train/test rows.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Preserve source ordering instead of applying a deterministic shuffle first.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    train_output_path = _resolve_path(script_dir, args.train_output)
    test_output_path = _resolve_path(script_dir, args.test_output)

    dataset = load_dataset(args.dataset_name, split=args.source_split)
    if not args.no_shuffle:
        dataset = dataset.shuffle(seed=args.seed)

    total_required = args.train_size + args.max_test_size
    total_selected = min(len(dataset), total_required)
    selected_rows = [_convert_row(row) for row in dataset.select(range(total_selected))]

    train_rows = selected_rows[: args.train_size]
    test_rows = selected_rows[args.train_size :]

    _write_parquet(train_rows, train_output_path)
    _write_parquet(test_rows, test_output_path)

    print(
        "Split summary: "
        f"train={len(train_rows)} rows, test={len(test_rows)} rows, "
        f"source_split={args.source_split}, shuffled={not args.no_shuffle}, seed={args.seed}"
    )


if __name__ == "__main__":
    main()
