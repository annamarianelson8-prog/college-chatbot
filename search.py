"""Module 2: BFS and A* search over the knowledge base graph."""

from __future__ import annotations

import heapq
import json
from collections import deque
from pathlib import Path
from typing import Callable

from nlp_utils import keyword_overlap, tokenize

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"


def load_kb() -> dict:
    with open(KB_PATH, encoding="utf-8") as f:
        return json.load(f)


class KnowledgeGraph:
    def __init__(self, kb: dict | None = None):
        self.kb = kb or load_kb()
        graph = self.kb.get("graph", {})
        self.nodes = graph.get("nodes", [])
        self.adjacency: dict[str, list[str]] = {node: [] for node in self.nodes}
        for a, b in graph.get("edges", []):
            if a in self.adjacency and b in self.adjacency:
                self.adjacency[a].append(b)
                self.adjacency[b].append(a)

    def neighbors(self, node: str) -> list[str]:
        return self.adjacency.get(node, [])

    def bfs(self, start: str, goal: str) -> list[str] | None:
        if start not in self.adjacency or goal not in self.adjacency:
            return None
        if start == goal:
            return [start]
        queue = deque([[start]])
        visited = {start}
        while queue:
            path = queue.popleft()
            node = path[-1]
            for neighbor in self.neighbors(node):
                if neighbor in visited:
                    continue
                new_path = path + [neighbor]
                if neighbor == goal:
                    return new_path
                visited.add(neighbor)
                queue.append(new_path)
        return None

    def astar(
        self,
        start: str,
        goal: str,
        query: str,
        heuristic_fn: Callable[[str, str, str], float] | None = None,
    ) -> list[str] | None:
        if start not in self.adjacency or goal not in self.adjacency:
            return None
        if start == goal:
            return [start]

        h = heuristic_fn or default_heuristic
        open_set: list[tuple[float, int, str, list[str]]] = []
        counter = 0
        heapq.heappush(open_set, (h(start, goal, query), counter, start, [start]))
        best_g: dict[str, float] = {start: 0.0}

        while open_set:
            _, _, current, path = heapq.heappop(open_set)
            if current == goal:
                return path
            for neighbor in self.neighbors(current):
                g_score = len(path)
                if neighbor in best_g and g_score >= best_g[neighbor]:
                    continue
                best_g[neighbor] = g_score
                counter += 1
                f_score = g_score + h(neighbor, goal, query)
                heapq.heappush(open_set, (f_score, counter, neighbor, path + [neighbor]))
        return None

    def search_for_category(self, category: str, query: str, multi_hop: bool = False) -> dict:
        category_map = {
            "admission": "admission",
            "fees": "fees",
            "exams": "exams",
            "hostel": "hostel",
            "faculty": "faculty",
            "library": "library",
            "events": "events",
            "contact": "contact",
            "placement": "placement",
            "scholarship": "scholarship",
            "courses": "courses",
            "general": "root",
        }
        start = "root"
        goal = category_map.get(category, "root")
        if multi_hop and category in ("scholarship", "hostel", "exams"):
            path = self.astar(start, goal, query)
            algorithm = "A*"
        else:
            path = self.bfs(start, goal)
            algorithm = "BFS"
        return {
            "algorithm": algorithm,
            "start": start,
            "goal": goal,
            "path": path or [start, goal] if goal in self.adjacency else [start],
        }


def default_heuristic(current: str, goal: str, query: str) -> float:
    query_tokens = tokenize(query)
    current_tokens = tokenize(current.replace("_", " "))
    goal_tokens = tokenize(goal.replace("_", " "))
    overlap = keyword_overlap(query_tokens, current_tokens + goal_tokens)
    return max(0.0, 2.0 - overlap * 2.0)


def get_fact(kb: dict, predicate: str, args: list | None = None) -> dict | None:
    args = args or []
    for fact in kb.get("facts", []):
        if fact.get("predicate") == predicate and fact.get("args", []) == args:
            return fact
    return None


def get_facts_by_predicate(kb: dict, predicate: str) -> list[dict]:
    return [f for f in kb.get("facts", []) if f.get("predicate") == predicate]
