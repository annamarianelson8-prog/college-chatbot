"""Automated test suite for 50+ college chatbot queries."""

from __future__ import annotations

import csv
from pathlib import Path

from nlp_utils import ensure_nltk_data
from planner import ResponsePlanner

BASE_DIR = Path(__file__).resolve().parent
TRAINING_PATH = BASE_DIR / "data" / "training_data.csv"
VIVA_SCRIPT_PATH = BASE_DIR / "VIVA_DEMO.md"


def load_queries() -> list[dict]:
    queries = []
    with open(TRAINING_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            queries.append({"text": row["text"], "expected_category": row["category"]})
    return queries


def run_tests() -> tuple[int, int, list[str]]:
    ensure_nltk_data()
    planner = ResponsePlanner()
    queries = load_queries()
    passed = 0
    failures: list[str] = []

    for item in queries:
        result = planner.plan_and_respond(item["text"])
        actual = result["category"]
        expected = item["expected_category"]
        if actual == expected:
            passed += 1
        else:
            failures.append(f"FAIL: '{item['text']}' -> expected {expected}, got {actual}")

    return passed, len(queries), failures


def write_viva_script() -> None:
    content = """# Viva Demo Script — Intelligent College Assistant Chatbot

## 1. Introduction (30 seconds)
"This chatbot is a Goal-Based Intelligent Agent built with Python and Flask. It uses symbolic AI — not paid LLM APIs — to answer student queries 24/7."

## 2. PEAS Model
Open http://127.0.0.1:5000/agent-info or enable **Show AI reasoning plan** in the UI.

| Component | Value |
|-----------|-------|
| Performance | Accurate college answers |
| Environment | Student chat on website |
| Actuators | Text replies + feedback |
| Sensors | User messages, thumbs up/down |

## 3. BFS Demo
**Query:** "What is the B.Tech first year fee?"

Expected: Category `fees`, plan shows BFS path `root -> fees -> BTech -> Y1`.

## 4. Backward Chaining Demo
**Query:** "Am I eligible for hostel?"

Expected: Backward goal `eligible_for_hostel(yes)` checked; hostel eligibility answer.

## 5. A* Demo
**Query:** "Tell me about scholarship and fee waiver eligibility"

Expected: Plan uses A* search path through scholarship/fees nodes.

## 6. Decision Tree
**Query:** "When is TechnoVision tech fest?"

Expected: Category `events`, response mentions TechnoVision dates.

## 7. Reinforcement Learning
1. Ask any question and get a reply.
2. Click **Not helpful**.
3. Open `data/feedback_log.json` — Q-value updated with negative reward.

## 8. Expert System Fallback
**Query:** "asdfgh random unknown query"

Expected: Graceful fallback with helpdesk phone and email.

## 9. Run Automated Tests
```bash
python test_chatbot.py
```
Target: 50+ queries classified correctly.

## 10. Modules Covered
- Module 1: Intelligent Agent (`agent.py`)
- Module 2: BFS & A* (`search.py`)
- Module 3: FOL + Forward/Backward Chaining (`inference.py`)
- Module 4: Classical Planning (`planner.py`)
- Module 5: Decision Tree + RL + Expert System (`learner.py`, `expert_rules.py`)
- Module 6: Real-world college helpdesk application
"""
    VIVA_SCRIPT_PATH.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    write_viva_script()
    passed, total, failures = run_tests()
    print(f"\nTest Results: {passed}/{total} passed ({100 * passed / total:.1f}%)\n")
    if failures:
        print("Failures:")
        for f in failures[:15]:
            print(f"  {f}")
        if len(failures) > 15:
            print(f"  ... and {len(failures) - 15} more")
    else:
        print("All tests passed!")
    print(f"\nViva demo script written to: {VIVA_SCRIPT_PATH}")
