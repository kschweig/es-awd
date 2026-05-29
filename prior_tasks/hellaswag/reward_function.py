import re

_STRICT_RESPONSE_RE = re.compile(r"^\s*([0-3])\s*$")
_PREDICTED_LABEL_RE = re.compile(r"[0-3]")


def reward_function(output_text: str, label: str) -> dict:
    strict_match = _STRICT_RESPONSE_RE.match(output_text)
    match = _PREDICTED_LABEL_RE.search(output_text)
    predicted_label = match.group(0) if match else None
    format_reward = 1.0 if strict_match is not None else 0.0
    answer_reward = 1.0 if predicted_label == str(label) else 0.0
    return {
        "reward": answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }
