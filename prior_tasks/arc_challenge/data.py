from typing import Any, Dict, List

from datasets import load_dataset
from vllm import TokensPrompt

from .reward_function import reward_function
from .template import PROMPT_HEADER_TEMPLATE, SYSTEM_MESSAGE

TEST_DATA_PATH = "prior_tasks/arc_challenge/test-00000-of-00001.parquet"


def _parse_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question_id": row["question_id"],
        "question": row["question"],
        "option_labels": row["option_labels"],
        "options": row["options"],
        "answer": str(row["answer"]),
        "num_choices": int(row["num_choices"]),
    }


def get_data(tokenizer) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    test_dataset = load_dataset(
        "parquet",
        data_files={"test": TEST_DATA_PATH},
    )["test"]

    eval_data = [_parse_row(row) for row in test_dataset]
    for item in eval_data:
        item["context"] = process_context(item, tokenizer)

    print(f"Loaded {len(eval_data)} eval samples from {TEST_DATA_PATH}")
    return [], eval_data


def evaluate_reward(output_text: str, data: Dict[str, Any]):
    reward = reward_function(output_text, data["answer"])
    return reward, output_text


def _format_options(option_labels: List[str], options: List[str]) -> str:
    return "\n".join(
        f"{label}. {option}" for label, option in zip(option_labels, options)
    )


def _format_eval_question(task_data: Dict[str, Any]) -> str:
    return (
        f"Question: {task_data['question']}\n"
        f"Options:\n{_format_options(task_data['option_labels'], task_data['options'])}\n"
        "Answer:"
    )


def process_context(task_data: Dict[str, Any], tokenizer) -> TokensPrompt:
    prompt_body = PROMPT_HEADER_TEMPLATE
    prompt_body += "\n" + _format_eval_question(task_data)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt_body},
    ]

    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt = tokenizer(rendered)
    return TokensPrompt(prompt_token_ids=prompt["input_ids"])
