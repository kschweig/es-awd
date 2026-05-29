import json
from vllm import TokensPrompt
from .reward_function import reward_function
from .template import USER_TEMPLATE, SYSTEM_MESSAGE, RESPONSE_PROMPT

DATA_PATH = "tasks/countdown/countdown.json"

def get_data(tokenizer):
    # Train data
    with open(DATA_PATH, "r") as f:
        train_data = json.load(f)
    train_data = train_data[:200]
    for d in train_data:
        d["context"] = process_context(d, tokenizer)

    # Eval data
    eval_data = []
    with open(DATA_PATH, "r") as f:
        eval_data = json.load(f)[200:]
    print(f"Loaded {len(eval_data)} eval samples from {DATA_PATH}")

    # pre-process eval contexts
    for d in eval_data:
        d["context"] = process_context(d, tokenizer)

    return train_data, eval_data

def evaluate_reward(output_text, data):
    response = RESPONSE_PROMPT + output_text
    r = reward_function(response, data["numbers"], data["target"])
    return r, response

def process_context(task_data, tokenizer):
    numbers = task_data["numbers"]
    target = task_data["target"]
    user_content = USER_TEMPLATE.format(numbers=numbers, target=target)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content}
    ]

    messages = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    ) + RESPONSE_PROMPT

    prompts = tokenizer(
        messages,
    )

    return TokensPrompt(prompt_token_ids=prompts['input_ids'])
