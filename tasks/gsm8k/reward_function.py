import re
from typing import Any, Dict, Optional

_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")
_FRACTION_RE = re.compile(r"^\s*(-?\d+)\s*/\s*(-?\d+)\s*$")
_ANSWER_RE = re.compile(r"<answer>(.*?)<\/answer>", re.DOTALL)
_HASHES_NUM_RE = re.compile(r"####\s*([\-0-9\.,]+)")
_STRICT_HASHES_LINE_RE = re.compile(r"^####\s*([\-0-9\.,]+)$")
_BOXED_RE = re.compile(
    r"(?:\${1,2}\s*)?\\boxed\s*\{\s*([^}]*)\s*\}(?:\s*\${1,2})?",
    flags=re.IGNORECASE,
)
_SOLUTION_CLIP_CHARS = 4096

def _parse_numeric_value(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    if "####" in text:
        text = text.split("####")[-1].strip()

    text = text.replace(",", "").replace("$", "").replace("%", "").strip()

    frac_match = _FRACTION_RE.match(text)
    if frac_match:
        numerator = float(frac_match.group(1))
        denominator = float(frac_match.group(2))
        if abs(denominator) < 1e-12:
            return None
        return numerator / denominator

    match = _NUMERIC_RE.search(text)
    if not match:
        return None
    return float(match.group(0))


def _solution_tail(response: str) -> str:
    if len(response) <= _SOLUTION_CLIP_CHARS:
        return response
    return response[-_SOLUTION_CLIP_CHARS :]


def _search_region(response: str) -> str:
    tail = _solution_tail(response)
    if "</think>" in tail:
        tail = tail.rsplit("</think>", 1)[-1]
    return tail


def _extract_explicit_answer_text(response: str) -> Optional[str]:
    search_text = _search_region(response)

    hash_hits = _HASHES_NUM_RE.findall(search_text)
    if hash_hits:
        return hash_hits[-1].strip().replace(",", "").replace("$", "")

    boxed_hits = _BOXED_RE.findall(search_text)
    if boxed_hits:
        return boxed_hits[-1].strip().replace(",", "").replace("$", "")

    answer_hits = _ANSWER_RE.findall(search_text)
    if answer_hits:
        return answer_hits[-1].strip()

    return None


def _extract_prediction_number(response: str) -> Optional[float]:
    answer_text = _extract_explicit_answer_text(response)
    if answer_text is not None:
        return _parse_numeric_value(answer_text)

    # Fall back to the clipped tail, and prefer text after a closing think tag
    # when one is present so intermediate reasoning numbers are less likely to
    # be mistaken for the final answer.
    return _parse_numeric_value(_search_region(response))


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    stripped_response = response.rstrip()
    if not stripped_response:
        return 0.0

    last_line = stripped_response.splitlines()[-1].strip()
    strict_match = _STRICT_HASHES_LINE_RE.match(last_line)
    if strict_match is None:
        return 0.0

    return float(_parse_numeric_value(strict_match.group(1)) is not None)

def reward_function(
    response: str,
    target_answer: Any,
    end_token: Optional[str] = None,
) -> Dict[str, Any]:
    format_reward = format_reward_function(response, end_token)
    predicted = _extract_prediction_number(response)
    expected = _parse_numeric_value(target_answer)
    answer_reward = float(
        predicted is not None and expected is not None and abs(predicted - expected) < 1e-6
    )
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
