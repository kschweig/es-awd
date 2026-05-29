from collections import defaultdict
from typing import Any, Dict, List

from datasets import load_dataset
from vllm import TokensPrompt

from .reward_function import reward_function
from .template import PROMPT_HEADER_TEMPLATE, SYSTEM_MESSAGE

VALIDATION_DATA_PATH = "prior_tasks/mmlu_pro/validation-00000-of-00001.parquet"
TEST_DATA_PATH = "prior_tasks/mmlu_pro/test-00000-of-00001.parquet"
OPTION_LABELS = list("ABCDEFGHIJ")
NUM_FEW_SHOT_EXAMPLES = 5


def _parse_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question_id": row["question_id"],
        "question": row["question"],
        "options": row["options"],
        "answer": str(row["answer"]),
        "answer_index": int(row["answer_index"]),
        "cot_content": row["cot_content"],
        "category": row["category"],
        "src": row["src"],
    }


def get_data(tokenizer) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    validation_dataset = load_dataset(
        "parquet",
        data_files={"validation": VALIDATION_DATA_PATH},
    )["validation"]
    test_dataset = load_dataset(
        "parquet",
        data_files={"test": TEST_DATA_PATH},
    )["test"]

    validation_examples = [_parse_row(row) for row in validation_dataset]
    few_shot_by_category: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for item in validation_examples:
        few_shot_by_category[item["category"]].append(item)

    eval_data = [_parse_row(row) for row in test_dataset]
    for item in eval_data:
        item["context"] = process_context(
            item,
            tokenizer,
            few_shot_examples=few_shot_by_category[item["category"]][:NUM_FEW_SHOT_EXAMPLES],
        )

    print(
        f"Loaded {len(validation_examples)} few-shot examples from {VALIDATION_DATA_PATH}"
    )
    print(f"Loaded {len(eval_data)} eval samples from {TEST_DATA_PATH}")
    return [], eval_data


def evaluate_reward(output_text: str, data: Dict[str, Any]):
    reward = reward_function(output_text, data["answer"])
    return reward, output_text


def _format_options(options: List[str]) -> str:
    return "\n".join(
        f"{OPTION_LABELS[idx]}. {option}" for idx, option in enumerate(options)
    )


def _format_few_shot_example(task_data: Dict[str, Any]) -> str:
    return (
        f"Question: {task_data['question']}\n"
        f"Options:\n{_format_options(task_data['options'])}\n"
        f"Answer: {task_data['cot_content']}\n"
    )


def _format_eval_question(task_data: Dict[str, Any]) -> str:
    return (
        f"Question: {task_data['question']}\n"
        f"Options:\n{_format_options(task_data['options'])}\n"
        "Answer:"
    )


def process_context(
    task_data: Dict[str, Any],
    tokenizer,
    few_shot_examples: List[Dict[str, Any]],
) -> TokensPrompt:
    prompt_body = PROMPT_HEADER_TEMPLATE.format(category=task_data["category"])
    prompt_body += "\n".join(
        _format_few_shot_example(example) for example in few_shot_examples
    )
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
