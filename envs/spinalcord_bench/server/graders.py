"""Deterministic task graders — scores in [0.0, 1.0]."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: str
    instruction: str
    reference_context: str


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def grade_extract_total(reply: str) -> Tuple[float, str]:
    """Easy: find invoice total 47.23 in assistant reply."""
    if not reply.strip():
        return 0.0, "empty"
    if re.search(r"47\.23", reply):
        return 1.0, "found_total"
    # partial credit if close
    if "47" in reply and "23" in reply:
        return 0.35, "partial_digits"
    return 0.0, "total_not_found"


def grade_calendar_overlap(reply: str) -> Tuple[float, str]:
    """Medium: must acknowledge overlap (YES) with minimal justification."""
    t = _norm(reply)
    if not t:
        return 0.0, "empty"
    has_yes = bool(re.search(r"\byes\b", t))
    has_overlap_word = "overlap" in t or "overlapping" in t or "conflict" in t
    time_hint = "10:30" in reply or "10:45" in reply
    if has_yes and (has_overlap_word or time_hint):
        return 1.0, "correct_overlap"
    if has_yes:
        return 0.55, "yes_but_weak_reason"
    if "no" in t:
        return 0.0, "incorrect_no"
    return 0.2, "unclear"


def grade_log_error_code(reply: str) -> Tuple[float, str]:
    """Hard: first ERROR code in synthetic log is E42."""
    if not reply.strip():
        return 0.0, "empty"
    if re.search(r"\bE42\b", reply):
        return 1.0, "found_E42"
    if "e42" in reply.lower():
        return 0.85, "e42_wrong_case"
    if re.search(r"\bE\d+\b", reply):
        return 0.25, "wrong_error_code"
    return 0.1, "no_code"


TASK_ORDER = ("extract_total", "calendar_overlap", "log_first_error")

TASKS: dict[str, TaskSpec] = {
    "extract_total": TaskSpec(
        task_id="extract_total",
        difficulty="easy",
        instruction=(
            "Extract the numeric total in USD from the context. "
            "Reply with the amount and currency (e.g. 47.23 USD)."
        ),
        reference_context=(
            "Invoice #9921. Subtotal $40.00, Tax $7.23, Total: $47.23 USD. "
            "Payment due in 14 days."
        ),
    ),
    "calendar_overlap": TaskSpec(
        task_id="calendar_overlap",
        difficulty="medium",
        instruction=(
            "Meeting A is 10:00-10:45. Meeting B is 10:30-11:00 on the same day. "
            "Do they overlap? Reply YES or NO and give one short reason."
        ),
        reference_context="Same calendar day; times are local.",
    ),
    "log_first_error": TaskSpec(
        task_id="log_first_error",
        difficulty="hard",
        instruction=(
            "From the log lines below, what is the **first ERROR code** "
            "(the token like E42 on the first ERROR line)? Reply with only that code."
        ),
        reference_context=(
            "[08:01:01] INFO worker started\n"
            "[08:01:02] WARN cache miss\n"
            "[08:01:05] ERROR code=E42 component=database connection refused\n"
            "[08:01:06] ERROR code=E99 component=api timeout\n"
        ),
    ),
}

GRADERS: dict[str, Callable[[str], Tuple[float, str]]] = {
    "extract_total": grade_extract_total,
    "calendar_overlap": grade_calendar_overlap,
    "log_first_error": grade_log_error_code,
}


def pick_task_id(seed: int | None, explicit: str | None) -> str:
    if explicit and explicit in TASKS:
        return explicit
    if seed is None:
        return TASK_ORDER[0]
    idx = seed % len(TASK_ORDER)
    return TASK_ORDER[idx]
