from typing import Any, Dict, List

from datasets import load_dataset
from vllm import TokensPrompt

from .reward_function import canonicalize_answer, reward_function

TRAIN_DATA_PATH = "tasks/proofwriter/train.parquet"
TEST_DATA_PATH = "tasks/proofwriter/test.parquet"


def _parse_proofwriter_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    prompt_messages = [dict(message) for message in row["prompt"]]
    target_answer = canonicalize_answer(row["reward_model"]["ground_truth"])
    if target_answer is None:
        raise ValueError(f"Could not parse ProofWriter answer from {row['reward_model']['ground_truth']!r}")

    extra_info = dict(row.get("extra_info") or {})
    return {
        "id": row.get("id", idx),
        "question": extra_info.get("question", ""),
        "theory": extra_info.get("theory", ""),
        "target": target_answer,
        "solution": target_answer,
        "prompt_messages": prompt_messages,
        "metadata": extra_info,
    }


def _load_proofwriter_parquet(path: str) -> List[Dict[str, Any]]:
    dataset = load_dataset("parquet", data_files=path, split="train")
    return [_parse_proofwriter_row(row, idx) for idx, row in enumerate(dataset)]


def get_data(tokenizer):
    train_data = _load_proofwriter_parquet(TRAIN_DATA_PATH)
    for item in train_data:
        item["context"] = process_context(item, tokenizer)

    eval_data = _load_proofwriter_parquet(TEST_DATA_PATH)
    print(f"Loaded {len(eval_data)} eval samples from {TEST_DATA_PATH}")
    for item in eval_data:
        item["context"] = process_context(item, tokenizer)

    return train_data, eval_data


def evaluate_reward(output_text, data):
    reward = reward_function(output_text, data["target"])
    return reward, output_text


def process_context(task_data, tokenizer):
    prompt_messages = task_data["prompt_messages"]
    continue_final_message = bool(
        prompt_messages and prompt_messages[-1].get("role") == "assistant"
    )
    rendered = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=not continue_final_message,
        continue_final_message=continue_final_message,
        tokenize=False,
    )
    prompt = tokenizer(rendered)
    return TokensPrompt(prompt_token_ids=prompt["input_ids"])
