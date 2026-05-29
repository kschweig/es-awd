from pathlib import Path

from datasets import Dataset, load_dataset


DATASET_NAME = "piqa"
SPLIT_OUTPUTS = {
    "validation": "validation-00000-of-00001.parquet",
}


def _convert_row(row: dict) -> dict:
    if int(row["label"]) not in {0, 1}:
        raise ValueError(f"Unexpected PiQA label: {row['label']!r}")
    return {
        "goal": row["goal"],
        "sol1": row["sol1"],
        "sol2": row["sol2"],
        "label": str(int(row["label"]) + 1),
    }


def _write_split(split: str, output_path: Path) -> None:
    dataset = load_dataset(DATASET_NAME, split=split, trust_remote_code=True)
    rows = [_convert_row(row) for row in dataset]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output_path))
    print(f"Wrote {len(rows)} rows to {output_path}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    for split, filename in SPLIT_OUTPUTS.items():
        _write_split(split, script_dir / filename)


if __name__ == "__main__":
    main()
