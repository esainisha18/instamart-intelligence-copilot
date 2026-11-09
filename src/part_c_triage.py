"""Classify support tickets with privacy, input, and output guardrails."""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mock_llm


RAW_TICKETS = ROOT / "data" / "raw" / "tickets.csv"
OUTPUT_DIR = ROOT / "outputs"
ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
BASE_DELAY = 0.05
MAX_RETRIES = 3
TOKEN_BUDGET = 6000
TOKEN_RESERVE = 30
REDACTION_PATTERNS = [
    ("emails", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("phone_numbers", re.compile(r"\b[6-9]\d{9}\b")),
    ("order_ids", re.compile(r"ORD-\d{6}")),
]
ZERO_SHOT_PROMPT = """CLASSIFY each support ticket into exactly one allowed category:
billing, delivery, technical, fraud.
Return only JSON in this format: {"category": "...", "reason": "..."}.
"""
FEW_SHOT_PROMPT = """CLASSIFY each support ticket into exactly one allowed category:
billing, delivery, technical, fraud.
Return only JSON in this format: {"category": "...", "reason": "..."}.
EXAMPLES:
Ticket: I was charged twice and need a refund.
JSON: {"category": "billing", "reason": "duplicate charge"}
Ticket: My package is late and the rider has not arrived.
JSON: {"category": "delivery", "reason": "late delivery"}
Ticket: There is an unauthorized transaction; someone hacked my account.
JSON: {"category": "fraud", "reason": "unauthorized account activity"}
"""


def est_tokens(text: str) -> int:
    return mock_llm.est_tokens(text)


def redact_text(text: str) -> tuple[str, dict[str, int]]:
    redacted = text
    counts: dict[str, int] = {}
    for label, pattern in REDACTION_PATTERNS:
        redacted, count = pattern.subn(
            {
                "emails": "[EMAIL]",
                "phone_numbers": "[PHONE]",
                "order_ids": "[ORDER_ID]",
            }[label],
            redacted,
        )
        counts[label] = count
    return redacted, counts


def validate_reply(reply: str) -> tuple[str, str | None, str | None]:
    try:
        parsed = json.loads(reply)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict) and parsed.get("category") in ALLOWED_CATEGORIES:
        reason = parsed.get("reason")
        return "classified_json", parsed["category"], str(reason or "")

    if parsed is None:
        words = re.findall(r"[a-z]+", reply.lower())
        category_words = [word for word in words if word in ALLOWED_CATEGORIES]
        if len(category_words) == 1:
            return "classified_keyword", category_words[0], reply.strip()

    return "flagged_invalid_output", None, None


def call_with_retry(messages: list[dict[str, str]]) -> tuple[Any, int]:
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages), retry_number
        except mock_llm.TransientLLMError as error:
            if retry_number >= MAX_RETRIES:
                raise RuntimeError(
                    f"LLM call failed after {MAX_RETRIES} retries"
                ) from error
            delay = (
                BASE_DELAY
                * (2**retry_number)
                * random.uniform(0.9, 1.15)
            )
            print(
                f"Retrying LLM call ({retry_number + 1}/{MAX_RETRIES}) "
                f"after {error}; sleeping {delay:.3f}s"
            )
            time.sleep(delay)
    raise AssertionError("Retry loop exited unexpectedly")


def _base_row(ticket: pd.Series) -> dict[str, Any]:
    return {
        "ticket_id": ticket["ticket_id"],
        "customer_id": ticket["customer_id"],
        "true_category": ticket["true_category"],
        "predicted_category": None,
        "reason": None,
        "status": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "retries": 0,
        "redacted_text": None,
    }


def run_triage(
    tickets: pd.DataFrame,
    system_prompt: str,
    *,
    token_budget: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mock_llm.reset()
    rows: list[dict[str, Any]] = []
    redaction_totals = {label: 0 for label, _ in REDACTION_PATTERNS}
    total_tokens = 0
    llm_calls = 0

    for _, ticket in tickets.iterrows():
        row = _base_row(ticket)
        text = ticket["text"]
        if not text.strip():
            row["status"] = "skipped_empty"
            rows.append(row)
            continue
        input_tokens = est_tokens(text)
        if input_tokens > 400:
            row["status"] = "flagged_long"
            rows.append(row)
            continue

        redacted_text, counts = redact_text(text)
        for label, count in counts.items():
            redaction_totals[label] += count
        if any(pattern.search(redacted_text) for _, pattern in REDACTION_PATTERNS):
            raise AssertionError(
                f"Redaction did not remove all sensitive patterns from {ticket['ticket_id']}"
            )
        row["redacted_text"] = redacted_text

        prompt_tokens = est_tokens(system_prompt) + est_tokens(redacted_text)
        if (
            token_budget is not None
            and total_tokens + prompt_tokens + TOKEN_RESERVE > token_budget
        ):
            row["status"] = "budget_exhausted"
            rows.append(row)
            continue

        response, retries = call_with_retry(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": redacted_text},
            ]
        )
        llm_calls += 1
        message = response.choices[0].message
        reply = message.content or ""
        status, category, reason = validate_reply(reply)
        usage = response.usage
        row.update(
            {
                "predicted_category": category,
                "reason": reason,
                "status": status,
                "prompt_tokens": int(usage.prompt_tokens),
                "completion_tokens": int(usage.completion_tokens),
                "total_tokens": int(usage.total_tokens),
                "retries": retries,
            }
        )
        total_tokens += int(usage.total_tokens)
        rows.append(row)

    results = pd.DataFrame(rows)
    classified = results["status"].isin(
        ["classified_json", "classified_keyword"]
    )
    accuracy_mask = classified & results["true_category"].isin(ALLOWED_CATEGORIES)
    accuracy = (
        float(
            results.loc[accuracy_mask, "predicted_category"]
            .eq(results.loc[accuracy_mask, "true_category"])
            .mean()
        )
        if accuracy_mask.any()
        else 0.0
    )
    statistics = {
        "classified_count": int(classified.sum()),
        "classified_json": int(results["status"].eq("classified_json").sum()),
        "classified_keyword": int(results["status"].eq("classified_keyword").sum()),
        "flagged_invalid_output": int(
            results["status"].eq("flagged_invalid_output").sum()
        ),
        "skipped_empty": int(results["status"].eq("skipped_empty").sum()),
        "flagged_long": int(results["status"].eq("flagged_long").sum()),
        "budget_exhausted": int(
            results["status"].eq("budget_exhausted").sum()
        ),
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_ticket": (
            total_tokens / len(results) if len(results) else 0.0
        ),
        "tokens_per_classified_ticket": (
            total_tokens / int(classified.sum()) if classified.any() else 0.0
        ),
        "llm_calls": llm_calls,
        "retries": int(results["retries"].sum()),
        "redactions": redaction_totals,
    }
    statistics["redactions"]["total"] = sum(redaction_totals.values())
    return results, statistics


def print_statistics(name: str, stats: dict[str, Any]) -> None:
    print(f"\n{name}:")
    for label in (
        "classified_count",
        "classified_json",
        "classified_keyword",
        "flagged_invalid_output",
        "skipped_empty",
        "flagged_long",
        "budget_exhausted",
        "accuracy",
        "total_tokens",
        "tokens_per_ticket",
        "tokens_per_classified_ticket",
        "llm_calls",
        "retries",
    ):
        value = stats[label]
        print(
            f"  {label}: {value:.4f}"
            if isinstance(value, float)
            else f"  {label}: {value}"
        )
    print(f"  redactions: {stats['redactions']}")


def confusion_table(results: pd.DataFrame) -> pd.DataFrame:
    valid = results.loc[
        results["status"].isin(["classified_json", "classified_keyword"])
        & results["true_category"].isin(ALLOWED_CATEGORIES)
    ]
    return pd.crosstab(
        valid["true_category"],
        valid["predicted_category"],
        rownames=["actual"],
        colnames=["predicted"],
    ).reindex(
        index=sorted(ALLOWED_CATEGORIES),
        columns=sorted(ALLOWED_CATEGORIES),
        fill_value=0,
    )


def route_ticket(
    status: str, category: str | None, is_premium: bool
) -> tuple[str, str]:
    if status in {"flagged_long", "flagged_invalid_output", "budget_exhausted"}:
        return "human_review", "HIGH"
    if category == "fraud":
        return "security_team", "HIGH"
    if is_premium and status in {"classified_json", "classified_keyword"}:
        return "premium_desk", "NORMAL"
    return "support_queue", "NORMAL"


def final_routing(
    results: pd.DataFrame, segments_path: Path
) -> pd.DataFrame:
    segments = pd.read_csv(segments_path)
    if not {"customer_id", "is_premium"}.issubset(segments.columns):
        raise ValueError(f"{segments_path} does not contain customer premium labels")
    premium_ids = set(
        segments.loc[
            segments["is_premium"].astype(str).str.lower().eq("true"),
            "customer_id",
        ]
    )
    routed = results.copy()
    routes = [
        route_ticket(
            row.status,
            row.predicted_category,
            row.customer_id in premium_ids,
        )
        for row in routed.itertuples(index=False)
    ]
    routed["route"] = [route for route, _ in routes]
    routed["priority"] = [priority for _, priority in routes]
    return routed


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickets = pd.read_csv(RAW_TICKETS, keep_default_na=False)
    required = {"ticket_id", "customer_id", "text", "true_category"}
    if missing := required - set(tickets.columns):
        raise ValueError(f"tickets.csv is missing columns: {sorted(missing)}")
    assert len(tickets) == 164, f"Expected 164 tickets, got {len(tickets)}"

    empty_count = int(tickets["text"].map(lambda text: not text.strip()).sum())
    long_count = int(
        tickets["text"].map(lambda text: bool(text.strip()) and est_tokens(text) > 400).sum()
    )
    assert empty_count == 4, f"Expected 4 empty tickets, got {empty_count}"
    assert long_count == 4, f"Expected 4 long tickets, got {long_count}"
    eligible = tickets.loc[
        tickets["text"].map(
            lambda text: bool(text.strip()) and est_tokens(text) <= 400
        )
    ]
    print(
        f"Input guardrails: total={len(tickets)}, eligible={len(eligible)}, "
        f"skipped_empty={empty_count}, flagged_long={long_count}"
    )

    zero_shot, zero_stats = run_triage(eligible, ZERO_SHOT_PROMPT)
    few_shot, few_stats = run_triage(eligible, FEW_SHOT_PROMPT)
    expected_distribution = {
        "classified_json": 135,
        "classified_keyword": 12,
        "flagged_invalid_output": 9,
    }
    for name, result, stats in (
        ("zero-shot", zero_shot, zero_stats),
        ("few-shot", few_shot, few_stats),
    ):
        distribution = {
            key: stats[key]
            for key in expected_distribution
        }
        print_statistics(name, stats)
        assert distribution == expected_distribution, (
            f"{name} status distribution {distribution} does not match "
            f"{expected_distribution}"
        )
        assert stats["redactions"] == {
            "emails": 55,
            "phone_numbers": 91,
            "order_ids": 132,
            "total": 278,
        }
        assert stats["accuracy"] >= (0.85 if name == "zero-shot" else 1.0) - 0.001

    zero_shot.to_csv(OUTPUT_DIR / "triage_zero_shot.csv", index=False)
    few_shot.to_csv(OUTPUT_DIR / "triage_few_shot.csv", index=False)
    zero_confusion = confusion_table(zero_shot)
    print("\nZero-shot confusion table:")
    print(zero_confusion.to_string())
    zero_confusion.to_csv(OUTPUT_DIR / "triage_zero_shot_confusion.csv")

    budget_results, budget_stats = run_triage(
        eligible,
        ZERO_SHOT_PROMPT,
        token_budget=TOKEN_BUDGET,
    )
    print_statistics(f"Zero-shot budget demo (budget={TOKEN_BUDGET})", budget_stats)
    budget_results.to_csv(
        OUTPUT_DIR / "triage_budget_demo.csv",
        index=False,
    )

    few_shot_full, final_stats = run_triage(
        tickets.loc[
            tickets["text"].map(
                lambda text: bool(text.strip()) and est_tokens(text) <= 400
            )
        ],
        FEW_SHOT_PROMPT,
        token_budget=100000,
    )
    if final_stats["budget_exhausted"]:
        raise AssertionError("Final few-shot run unexpectedly exhausted its budget")
    final_rows = final_routing(
        few_shot_full,
        OUTPUT_DIR / "customer_segments.csv",
    )
    final_rows = tickets[
        ["ticket_id", "customer_id", "text", "true_category"]
    ].merge(
        final_rows,
        on=["ticket_id", "customer_id", "true_category"],
        how="left",
        validate="one_to_one",
    )
    for index, row in final_rows.loc[final_rows["status"].isna()].iterrows():
        if not row["text"].strip():
            status, route, priority = "skipped_empty", "support_queue", "NORMAL"
        else:
            status, route, priority = "flagged_long", "human_review", "HIGH"
        final_rows.loc[index, ["status", "route", "priority"]] = [
            status,
            route,
            priority,
        ]
    final_rows = final_rows.drop(columns="text")
    assert len(final_rows) == 164
    assert not final_rows["status"].isna().any()
    final_rows.to_csv(OUTPUT_DIR / "triage_results.csv", index=False)
    print_statistics("Final few-shot routing run", final_stats)
    print("Final routing counts:")
    print(
        final_rows.groupby(["route", "priority"], dropna=False)
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
