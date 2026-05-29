from typing import Any, Dict, List

from datasets import load_dataset
from vllm import TokensPrompt

from .reward_function import reward_function
from .template import SYSTEM_MESSAGE, USER_TEMPLATE

VALIDATION_DATA_PATH = "prior_tasks/piqa/validation-00000-of-00001.parquet"


def _parse_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "goal": row["goal"],
        "sol1": row["sol1"],
        "sol2": row["sol2"],
        "label": str(row["label"]),
    }


def get_data(tokenizer) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dataset = load_dataset(
        "parquet",
        data_files={"validation": VALIDATION_DATA_PATH},
    )["validation"]

    eval_data = [_parse_row(row) for row in dataset]
    for item in eval_data:
        item["context"] = process_context(item, tokenizer)

    print(f"Loaded {len(eval_data)} eval samples from {VALIDATION_DATA_PATH}")
    return [], eval_data


def evaluate_reward(output_text: str, data: Dict[str, Any]):
    reward = reward_function(output_text, data["label"])
    return reward, output_text


def process_context(task_data: Dict[str, Any], tokenizer) -> TokensPrompt:
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                goal=task_data["goal"],
                solution_1=task_data["sol1"],
                solution_2=task_data["sol2"],
            ),
        },
    ]

    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt = tokenizer(rendered)
    return TokensPrompt(prompt_token_ids=prompt["input_ids"])
