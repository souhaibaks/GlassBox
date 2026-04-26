"""Small free chat simulator for the GlassBox IronClaw tool flow.

This is not an LLM. It parses the common demo request shape and calls the same
tools an IronClaw agent would call:

    python ironclaw/chat_sim.py "Build a model to predict survived using titanic.csv"
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ironclaw.tools import autofit_tool, explain_tool


def parse_request(message: str) -> dict:
    #Parse a simple AutoML request into tool arguments
    message = message.replace('\\"', '"').replace("\\'", "'")
    target = _find_target(message)
    csv_path = _find_csv_path(message)
    task_type = _find_task_type(message)

    missing = []
    if not target:
        missing.append("target column")
    if not csv_path:
        missing.append("CSV path")
    if missing:
        raise ValueError(
            "Missing " + " and ".join(missing) + ". "
            "Try: Build a model to predict \"survived\" using titanic.csv"
        )

    return {
        "csv_path": csv_path,
        "target_col": target,
        "task_type": task_type,
        "cv": 3,
    }


def run_chat_sim(message: str) -> dict:
    #Execute the parsed request and return the agent-style response
    args = parse_request(message)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = autofit_tool(**args)
    if result.get("status") != "ok":
        return {
            "status": "error",
            "message": result.get("message", "AutoFit failed."),
            "tool_args": args,
        }

    report = result["report"]
    explanation = explain_tool(report)
    return {
        "status": "ok",
        "tool_args": args,
        "report": report,
        "explanation": explanation,
    }


def _find_target(message: str) -> str:
    patterns = [
        r"\bpredict\s+['\"]([^'\"]+)['\"]",
        r"\bpredict\s+([A-Za-z_][\w.-]*)",
        r"\btarget\s*(?:column)?\s*(?:is|=|:)\s*['\"]?([^'\"\s,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _find_csv_path(message: str) -> str:
    quoted = re.search(r"\busing\s+['\"]([^'\"]+\.csv)['\"]", message,
                       flags=re.IGNORECASE)
    if quoted:
        return quoted.group(1).strip()

    unquoted = re.search(r"\busing\s+(\S+\.csv)", message, flags=re.IGNORECASE)
    if unquoted:
        return unquoted.group(1).strip().rstrip(".,;")

    any_csv = re.search(r"(\S+\.csv)", message, flags=re.IGNORECASE)
    if any_csv:
        return any_csv.group(1).strip().rstrip(".,;")
    return ""


def _find_task_type(message: str) -> str:
    if re.search(r"\b(regression|regressor|continuous|numeric value)\b",
                 message, flags=re.IGNORECASE):
        return "regression"
    return "classification"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Simulate an IronClaw chat request for GlassBox AutoML."
    )
    parser.add_argument("message", nargs="+",
                        help="Natural-language request to parse and run.")
    args = parser.parse_args(argv)

    message = " ".join(args.message)
    response = run_chat_sim(message)
    if response["status"] != "ok":
        print("I could not run GlassBox AutoFit.")
        print(response["message"])
        return 1

    tool_args = response["tool_args"]
    report = response["report"]
    explanation = response["explanation"]

    print("Agent: parsed request")
    print(f"  csv_path  : {tool_args['csv_path']}")
    print(f"  target    : {tool_args['target_col']}")
    print(f"  task_type : {tool_args['task_type']}")
    print()
    print("Agent: calling autofit_tool...")
    print(f"  best model : {report['best_model']}")
    print(f"  CV score   : {report['best_cv_score']}")
    print()
    print("Agent: calling explain_tool...")
    print()
    print(explanation["summary"])

    factors = explanation.get("decision_factors", [])
    if factors:
        print()
        print("Decision factors:")
        for factor in factors:
            print(
                f"  {factor['rank']}. {factor['feature']:12} "
                f"{factor['importance_pct']:5.1f}%  ({factor['method']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
