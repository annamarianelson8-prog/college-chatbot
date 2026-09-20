"""Module 3: Forward and Backward chaining inference engine."""

from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"


def load_kb() -> dict:
    with open(KB_PATH, encoding="utf-8") as f:
        return json.load(f)


class InferenceEngine:
    def __init__(self, kb: dict | None = None):
        self.kb = kb or load_kb()
        self.working_memory: set[str] = set()

    def reset(self) -> None:
        self.working_memory.clear()

    def add_fact(self, literal: str) -> None:
        self.working_memory.add(literal)

    def seed_from_tokens(self, tokens: list[str], category: str) -> None:
        self.reset()
        self.add_fact(f"query_category({category})")
        mapping = {
            "btech": "course(BTech)",
            "bca": "course(BCA)",
            "mca": "course(MCA)",
            "cse": "department(CSE)",
            "ece": "department(ECE)",
            "me": "department(ME)",
            "fee": "query_fee",
            "hostel": "query_hostel",
            "exam": "query_exam",
            "scholarship": "query_scholarship",
        }
        for token in tokens:
            if token in mapping:
                self.add_fact(mapping[token])
        if category == "fees":
            self.add_fact("query_fee")
        if category == "hostel":
            self.add_fact("query_hostel")
        if category == "exams":
            self.add_fact("query_exam")
        if category == "scholarship":
            self.add_fact("query_scholarship")

    def forward_chain(self, max_iterations: int = 20) -> list[str]:
        derived: list[str] = []
        for _ in range(max_iterations):
            changed = False
            for rule in self.kb.get("rules", []):
                conditions = rule.get("if", [])
                conclusion = rule.get("then", "")
                if all(cond in self.working_memory for cond in conditions):
                    if conclusion not in self.working_memory:
                        self.working_memory.add(conclusion)
                        derived.append(conclusion)
                        changed = True
            if not changed:
                break
        return derived

    def backward_chain(self, goal: str, visited: set[str] | None = None) -> bool:
        visited = visited or set()
        if goal in self.working_memory:
            return True
        if goal in visited:
            return False
        visited.add(goal)
        for rule in self.kb.get("rules", []):
            conclusion = rule.get("then", "")
            if conclusion != goal:
                continue
            conditions = rule.get("if", [])
            if all(self.backward_chain(cond, visited) or cond in self.working_memory for cond in conditions):
                self.working_memory.add(goal)
                return True
        return goal in self.working_memory

    def infer_eligibility(self, tokens: list[str], category: str) -> dict:
        self.seed_from_tokens(tokens, category)
        results = {"forward_derived": [], "backward_goals": [], "eligible": None}

        if category == "hostel":
            if "distance" in tokens or "km" in tokens or "eligible" in tokens or "stay" in tokens:
                results["backward_goals"].append("eligible_for_hostel(yes)")
                yes = self.backward_chain("eligible_for_hostel(yes)")
                no = self.backward_chain("eligible_for_hostel(no)")
                if yes:
                    results["eligible"] = True
                elif no:
                    results["eligible"] = False
                else:
                    results["eligible"] = "maybe"
            self.forward_chain()

        elif category == "scholarship":
            results["backward_goals"].append("eligible_for_merit_scholarship(yes)")
            if "90" in tokens or "merit" in tokens or "percent" in tokens:
                self.add_fact("marks_gte(90%)")
            yes = self.backward_chain("eligible_for_merit_scholarship(yes)")
            results["eligible"] = yes
            self.forward_chain()

        elif category == "exams":
            if "attendance" in tokens or "75" in tokens:
                self.add_fact("attendance_gte(75%)")
                results["backward_goals"].append("eligible_for_exam")
                results["eligible"] = self.backward_chain("eligible_for_exam")
            self.forward_chain()

        else:
            results["forward_derived"] = self.forward_chain()

        results["working_memory"] = sorted(self.working_memory)
        return results
