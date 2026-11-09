"""Run resumable ReAct refund-agent scenarios using local policy and order tools."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "data" / "instamart.db"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mock_llm
from src.part_d_rag import retrieve


TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Retrieve the final order status and delivery details.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Retrieve relevant refund policy passages.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate eligible late-delivery refund amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": ["number", "null"]},
                    "status": {"type": "string"},
                },
                "required": ["order_amount", "delivery_minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Record an approved refund in the refund ledger.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"},
                },
                "required": ["order_id", "refund_inr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Record a refund requiring human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"},
                },
                "required": ["order_id", "refund_inr"],
            },
        },
    },
]


class SimulatedCrash(RuntimeError):
    """Raised by the resume demonstration after persisting a tool observation."""


def get_order(order_id: str) -> dict[str, Any]:
    with sqlite3.connect(DATABASE_PATH) as connection:
        row = connection.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]),
        "delivery_minutes": (
            float(row[3]) if row[3] is not None else None
        ),
    }


def search_policy(query: str) -> dict[str, Any]:
    return {"results": retrieve(query, k=3)}


def calculate_refund(
    order_amount: float,
    delivery_minutes: float | None,
    status: str = "completed",
) -> dict[str, float]:
    eligible = (
        status == "completed"
        and delivery_minutes is not None
        and delivery_minutes > 45
    )
    return {"refund_inr": float(order_amount) if eligible else 0.0}


def _append_ledger(
    order_id: str, refund_inr: float, action: str
) -> dict[str, float]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "order_id": order_id,
        "refund_inr": float(refund_inr),
        "action": action,
    }
    with (OUTPUT_DIR / "refund_ledger.jsonl").open(
        "a", encoding="utf-8"
    ) as ledger:
        ledger.write(json.dumps(entry) + "\n")
    return {"refund_inr": float(refund_inr)}


def issue_refund(order_id: str, refund_inr: float) -> dict[str, float]:
    return _append_ledger(order_id, refund_inr, "issue_refund")


def escalate_to_human(order_id: str, refund_inr: float) -> dict[str, float]:
    return _append_ledger(order_id, refund_inr, "escalate_to_human")


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def _checkpoint_path(run_id: str) -> Path:
    return STATE_DIR / f"{run_id}.json"


def _save_checkpoint(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _checkpoint_path(state["run_id"]).write_text(
        json.dumps(state, indent=2) + "\n",
        encoding="utf-8",
    )


def _tokens_in_history(messages: list[dict[str, Any]]) -> int:
    return sum(
        mock_llm.est_tokens(
            message.get("content")
            or json.dumps(message.get("tool_calls", ""), default=str)
        )
        for message in messages
    )


def _status_from_answer(final_answer: str) -> str:
    match = re.match(r"\s*([A-Z_]+)\s*:", final_answer)
    return match.group(1) if match else "COMPLETED"


def _fallback(status: str, reason: str) -> str:
    return f"{status}: {reason}"


def _execute(
    state: dict[str, Any],
    *,
    max_iterations: int,
    token_limit: int,
    time_limit_seconds: float,
    crash_after_tool_execution: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    initial_step = int(state["step_count"])
    while state["step_count"] - initial_step < max_iterations:
        if time.monotonic() - started >= time_limit_seconds:
            state["status"] = "FALLBACK_TIMEOUT"
            state["final_answer"] = _fallback(
                state["status"], "execution time limit reached"
            )
            break
        prompt_estimate = _tokens_in_history(state["message_history"])
        if int(state["tokens_used"]) + prompt_estimate > token_limit:
            state["status"] = "FALLBACK_TOKEN_LIMIT"
            state["final_answer"] = _fallback(
                state["status"], "token limit reached before the next model call"
            )
            break

        response = mock_llm.chat(
            state["message_history"],
            tools=TOOLS_SPEC,
        )
        state["llm_calls"] += 1
        state["tokens_used"] += int(response.usage.total_tokens)
        if int(state["tokens_used"]) > token_limit:
            state["status"] = "FALLBACK_TOKEN_LIMIT"
            state["final_answer"] = _fallback(
                state["status"], "token limit reached"
            )
            break

        message = response.choices[0].message
        if response.choices[0].finish_reason != "tool_calls" or not message.tool_calls:
            state["final_answer"] = message.content or ""
            state["status"] = _status_from_answer(state["final_answer"])
            break

        assistant_tool_calls = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments),
                },
            }
            for tool_call in message.tool_calls
        ]
        state["message_history"].append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls,
            }
        )

        for tool_call in message.tool_calls:
            if tool_call.name not in TOOL_MAP:
                raise ValueError(f"Model requested unknown tool: {tool_call.name}")
            arguments = tool_call.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            observation = TOOL_MAP[tool_call.name](**arguments)
            state["message_history"].append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "content": json.dumps(observation),
                }
            )
            state["executed_tools"].append(
                {
                    "name": tool_call.name,
                    "arguments": arguments,
                    "observation": observation,
                }
            )
            state["step_count"] += 1
            state["steps"].append(state["executed_tools"][-1])
            _save_checkpoint(state)
            if (
                crash_after_tool_execution is not None
                and len(state["executed_tools"]) == crash_after_tool_execution
            ):
                raise SimulatedCrash(
                    f"Simulated crash after {crash_after_tool_execution} tool executions"
                )
            if time.monotonic() - started >= time_limit_seconds:
                state["status"] = "FALLBACK_TIMEOUT"
                state["final_answer"] = _fallback(
                    state["status"], "execution time limit reached"
                )
                break
        if state.get("status") == "FALLBACK_TIMEOUT":
            break

    if "final_answer" not in state:
        state["status"] = "FALLBACK_MAX_ITERATIONS"
        state["final_answer"] = _fallback(
            state["status"], "maximum tool iterations reached"
        )
    state["tokens_used"] = int(state["tokens_used"])
    state["step_count"] = int(state["step_count"])
    _save_checkpoint(state)
    return state


def run_agent(
    user_message: str,
    scenario_id: str,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10,
    *,
    crash_after_tool_execution: int | None = None,
) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    state: dict[str, Any] = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "message_history": [
            {"role": "system", "content": "AGENT: Refund support assistant."},
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0,
        "step_count": 0,
        "executed_tools": [],
        "steps": [],
        "llm_calls": 0,
    }
    return _execute(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after_tool_execution=crash_after_tool_execution,
    )


def resume_agent(
    run_id: str,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10,
) -> dict[str, Any]:
    path = _checkpoint_path(run_id)
    if not path.is_file():
        raise FileNotFoundError(f"No saved run state for {run_id}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("run_id") != run_id:
        raise ValueError(f"Checkpoint run_id does not match {run_id}")
    return _execute(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
    )


def _refund_amount(state: dict[str, Any]) -> float | None:
    for step in reversed(state.get("executed_tools", [])):
        if step["name"] in {"calculate_refund", "issue_refund", "escalate_to_human"}:
            observation = step["observation"]
            if "refund_inr" in observation:
                return float(observation["refund_inr"])
    match = re.search(r"INR\s+([0-9]+(?:\.[0-9]+)?)", state["final_answer"])
    if match:
        return float(match.group(1))
    if state["status"] in {
        "DENIED",
        "NOT_FOUND",
        "FALLBACK_MAX_ITERATIONS",
        "FALLBACK_TOKEN_LIMIT",
        "FALLBACK_TIMEOUT",
    }:
        return 0.0
    return None


def _write_run(state: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "scenario_id": state["scenario_id"],
        "status": state["status"],
        "final_answer": state["final_answer"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "steps": state["steps"],
    }
    with (OUTPUT_DIR / "agent_runs.jsonl").open("a", encoding="utf-8") as runs:
        runs.write(json.dumps(record) + "\n")


def run_scenario_benchmark() -> list[dict[str, Any]]:
    scenarios = pd.read_csv(
        ROOT / "data" / "raw" / "agent_scenarios.csv",
        keep_default_na=False,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "agent_runs.jsonl").write_text("", encoding="utf-8")
    results: list[dict[str, Any]] = []
    for scenario in scenarios.itertuples(index=False):
        state = run_agent(scenario.message, scenario.scenario_id)
        results.append(state)
        _write_run(state)
        amount = _refund_amount(state)
        print(
            f"{scenario.scenario_id}: status={state['status']} "
            f"refund={amount} llm_calls={state['llm_calls']} "
            f"tokens={state['tokens_used']} tools="
            f"{[step['name'] for step in state['steps']]}"
        )

        expected_status = (
            "FALLBACK_MAX_ITERATIONS"
            if scenario.scenario_id == "S7"
            else scenario.expected_outcome
        )
        assert state["status"] == expected_status, (
            f"{scenario.scenario_id}: expected {expected_status}, "
            f"got {state['status']}"
        )
        if scenario.scenario_id in {"S1", "S2", "S3"}:
            expected_amount = float(scenario.expected_refund_inr)
            actual_amount = _refund_amount(state)
            assert actual_amount is not None and abs(
                actual_amount - expected_amount
            ) < 0.005
        if scenario.scenario_id == "S1":
            assert state["llm_calls"] == 5
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    results = run_scenario_benchmark()
    s1_message = "My order ORD-100048 reached me very late. I would like a refund."

    token_demo = run_agent(
        s1_message,
        "TOKEN_DEMO",
        token_limit=200,
    )
    assert token_demo["status"] == "FALLBACK_TOKEN_LIMIT"
    print(f"Token-limit demo: {token_demo['status']}")

    timeout_demo = run_agent(
        s1_message,
        "TIMEOUT_DEMO",
        time_limit_seconds=0,
    )
    assert timeout_demo["status"] == "FALLBACK_TIMEOUT"
    print(f"Timeout demo: {timeout_demo['status']}")

    try:
        run_agent(
            s1_message,
            "RESUME_DEMO",
            crash_after_tool_execution=2,
        )
        raise AssertionError("Expected SimulatedCrash was not raised")
    except SimulatedCrash:
        checkpoint = [
            path
            for path in STATE_DIR.glob("*.json")
            if json.loads(path.read_text(encoding="utf-8")).get("scenario_id")
            == "RESUME_DEMO"
        ]
        if not checkpoint:
            raise FileNotFoundError("Resume demo checkpoint was not saved")
        resume_path = max(checkpoint, key=lambda path: path.stat().st_mtime)
        run_id = resume_path.stem
    resumed = resume_agent(run_id)
    expected_tools = [
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund",
    ]
    actual_tools = [step["name"] for step in resumed["executed_tools"]]
    assert actual_tools == expected_tools, (
        f"Resume duplicated or skipped tools: {actual_tools}"
    )
    assert resumed["status"] == "REFUND"
    assert abs((_refund_amount(resumed) or 0) - 128.25) < 0.005
    print(
        "Crash-and-resume demo: ordered tools="
        f"{actual_tools}; total={len(actual_tools)}; final={resumed['status']} "
        f"{_refund_amount(resumed):.2f}"
    )

    summary = [
        {
            "scenario_id": state["scenario_id"],
            "status": state["status"],
            "refund_amount": _refund_amount(state),
            "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"],
        }
        for state in results
    ]
    (OUTPUT_DIR / "agent_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
