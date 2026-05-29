import re
from typing import Any, Dict, Optional

_ANSWER_RE = re.compile(r"^\s*answer\s*:\s*([A-Za-z]+)\s*$", re.IGNORECASE | re.MULTILINE)
_LENIENT_LAST_LINE_RE = re.compile(
    r"^\s*(?:final\s+answer\s*(?::|is)\s*|answer\s*(?::|is)\s*)?([A-Za-z]+)\s*[.!?,:;]*\s*$",
    re.IGNORECASE,
)
_LENIENT_LABEL_RE = re.compile(r"\b(true|false|unknown)\b", re.IGNORECASE)
_VALID_ANSWERS = {
    "true": "True",
    "false": "False",
    "unknown": "Unknown",
}


def canonicalize_answer(value: Any) -> Optional[str]:
    text = str(value).strip()
    if not text:
        return None

    text = text.rstrip(".!?,:;").strip().lower()
    return _VALID_ANSWERS.get(text)


def _last_nonempty_line(text: str) -> Optional[str]:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _extract_prediction_label(response: str) -> Optional[str]:
    matches = _ANSWER_RE.findall(response)
    if matches:
        return canonicalize_answer(matches[-1])

    last_line = _last_nonempty_line(response)
    if last_line:
        last_line_match = _LENIENT_LAST_LINE_RE.match(last_line)
        if last_line_match:
            return canonicalize_answer(last_line_match.group(1))

    lenient_matches = _LENIENT_LABEL_RE.findall(response)
    if lenient_matches:
        return canonicalize_answer(lenient_matches[-1])

    return canonicalize_answer(response)


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    last_line = _last_nonempty_line(response)
    if last_line is None:
        return 0.0
    return float(bool(_ANSWER_RE.fullmatch(last_line)))


def reward_function(
    response: str,
    target_answer: Any,
    end_token: Optional[str] = None,
) -> Dict[str, Any]:
    format_reward = format_reward_function(response, end_token)
    predicted = _extract_prediction_label(response)
    expected = canonicalize_answer(target_answer)
    answer_reward = float(predicted is not None and expected is not None and predicted == expected)
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }


def reward_function_verl(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    del data_source, extra_info

    reward_result = reward_function(
        response=solution_str,
        target_answer=ground_truth,
        end_token=kwargs.get("end_token"),
    )
    return {
        "score": reward_result["reward"],
        **reward_result["reward_info"],
    }
