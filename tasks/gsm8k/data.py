from typing import Any, Dict, List
from vllm import TokensPrompt
from .reward_function import reward_function, _parse_numeric_value

TRAIN_DATA_PATH = "tasks/gsm8k/train.parquet"
TEST_DATA_PATH = "tasks/gsm8k/test.parquet"

def _to_canonical_answer_text(value: Any) -> str:
    numeric = _parse_numeric_value(value)
    if numeric is None:
        raise ValueError(f"Could not parse GSM8K answer from {value!r}")
    rounded = round(numeric)
    if abs(numeric - rounded) < 1e-9:
        return str(int(rounded))
    return f"{numeric:.10f}".rstrip("0").rstrip(".")

def _parse_gsm8k_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    question = row["extra_info"]["question"].strip()
    target_answer = _to_canonical_answer_text(row["reward_model"]["ground_truth"])
    prompt_messages = row["prompt"].tolist()
    return {
        "id": row.get("id", idx),
        "question": question,
        "target": target_answer,
        "solution": target_answer,
        "prompt_messages": prompt_messages,
    }

def _load_gsm8k_parquet(path: str) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Loading GSM8k parquet requires pandas + pyarrow/fastparquet."
        ) from exc

    rows = pd.read_parquet(path).to_dict(orient="records")
    return [_parse_gsm8k_row(row, idx) for idx, row in enumerate(rows)]

def get_data(tokenizer):
    train_data = _load_gsm8k_parquet(TRAIN_DATA_PATH)
    for d in train_data:
        d["context"] = process_context(d, tokenizer)

    eval_data = _load_gsm8k_parquet(TEST_DATA_PATH)
    print(f"Loaded {len(eval_data)} eval samples from {TEST_DATA_PATH}")
    for d in eval_data:
        d["context"] = process_context(d, tokenizer)

    return train_data, eval_data

def evaluate_reward(output_text, data):
    r = reward_function(output_text, data["target"])
    return r, output_text

def process_context(task_data, tokenizer):
    rendered = tokenizer.apply_chat_template(
        task_data["prompt_messages"],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompts = tokenizer(rendered)

    return TokensPrompt(prompt_token_ids=prompts['input_ids'])
