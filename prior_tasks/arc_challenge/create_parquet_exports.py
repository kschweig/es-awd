from pathlib import Path

from datasets import Dataset, load_dataset


DATASET_NAME = "allenai/ai2_arc"
DATASET_CONFIG = "ARC-Challenge"
SPLIT_OUTPUTS = {
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
NUMERIC_TO_LETTER = {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
    "5": "E",
}


def _normalize_label(label: str) -> str:
    text = str(label).strip().upper()
    return NUMERIC_TO_LETTER.get(text, text)


def _convert_row(row: dict) -> dict:
    option_labels = [_normalize_label(label) for label in row["choices"]["label"]]
    answer = _normalize_label(row["answerKey"])
    return {
        "question_id": row["id"],
        "question": row["question"],
        "option_labels": option_labels,
        "options": row["choices"]["text"],
        "answer": answer,
        "num_choices": len(option_labels),
    }


def _write_split(split: str, output_path: Path) -> None:
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
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
