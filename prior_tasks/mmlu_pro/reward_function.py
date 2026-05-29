import re

_STRICT_FORMAT_RE = re.compile(r"THE ANSWER IS\s*\(([A-J])\)\.?\s*$")
_PAREN_ANSWER_RE = re.compile(r"ANSWER IS\s*\(([A-J])\)")
_PLAIN_ANSWER_RE = re.compile(r"ANSWER IS\s*([A-J])")
_FALLBACK_LETTER_RE = re.compile(r"\b([A-J])\b")


def reward_function(output_text: str, answer: str) -> dict:
    upper_text = output_text.upper()
    strict_match = _STRICT_FORMAT_RE.search(upper_text)
    match = _PAREN_ANSWER_RE.search(upper_text)
    if match is None:
        match = _PLAIN_ANSWER_RE.search(upper_text)
    if match is None:
        match = _FALLBACK_LETTER_RE.search(upper_text)
    predicted_answer = match.group(1) if match else None
    format_reward = 1.0 if strict_match is not None else 0.0
    answer_reward = 1.0 if predicted_answer == str(answer) else 0.0
    return {
        "reward": answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }
