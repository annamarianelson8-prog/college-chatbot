"""Module 4: Classical response planning pipeline."""

from __future__ import annotations

import uuid

from agent import GoalBasedAgent
from expert_rules import ExpertSystem
from inference import InferenceEngine
from learner import LearnerEngine
from nlp_utils import normalize_text, tokenize
from search import KnowledgeGraph


class ResponsePlanner:
    """Sequences: normalize -> classify -> search -> infer -> expert rules -> format."""

    MULTI_HOP_CATEGORIES = {"scholarship", "hostel", "exams", "fees"}

    def __init__(self):
        self.agent = GoalBasedAgent()
        self.learner = LearnerEngine()
        self.kb_graph = KnowledgeGraph()
        self.inference = InferenceEngine(self.kb_graph.kb)
        self.expert = ExpertSystem(self.kb_graph.kb)

    def plan_and_respond(self, query: str) -> dict:
        plan_steps: list[str] = []
        message_id = str(uuid.uuid4())[:8]

        plan_steps.append("Step 1: Normalize and tokenize user query")
        tokens = tokenize(query)
        normalized = normalize_text(query)

        plan_steps.append("Step 2: Classify query using Decision Tree")
        classification = self.learner.classify_query(query)
        category = classification["category"]
        confidence = classification["confidence"]

        backup = self.learner.keyword_fallback(query)
        expert_backup = self.expert.keyword_category_backup(tokens, normalized)
        if expert_backup and backup == "general":
            backup = expert_backup
        strong_phrases = (
            "technovision", "sports day", "helpdesk", "revaluation", "attendance",
            "campus wifi", "anti ragging", "cafeteria", "medical facility", "nss ncc",
            "about this college", "college overview", "what labs", "admission deadline",
            "library opening", "library timings", "department head", " hod ",
        )
        phrase_hit = any(p in normalized for p in strong_phrases)

        if backup != category and (
            category == "general"
            or confidence < 0.45
            or phrase_hit
            or (backup == "general" and category != "general" and phrase_hit)
        ):
            plan_steps.append(f"Step 2b: Refined category {category} -> {backup} (confidence={confidence})")
            category = backup

        plan_steps.append(f"Step 3: Agent perceives query; goal = {self.agent.GOAL_MAP.get(category, 'answer_general_query')}")
        self.agent.perceive(query, tokens, category)

        multi_hop = category in self.MULTI_HOP_CATEGORIES or len(tokens) > 6
        plan_steps.append(f"Step 4: Search KB using {'A*' if multi_hop else 'BFS'}")
        search_result = self.kb_graph.search_for_category(category, query, multi_hop=multi_hop)
        path = search_result.get("path", [])
        plan_steps.append(f"  Path: {' -> '.join(path)}")

        plan_steps.append("Step 5: Run inference (forward/backward chaining)")
        inference_result = self.inference.infer_eligibility(tokens, category)
        if inference_result.get("forward_derived"):
            plan_steps.append(f"  Forward derived: {', '.join(inference_result['forward_derived'])}")
        if inference_result.get("backward_goals"):
            plan_steps.append(f"  Backward goals checked: {', '.join(inference_result['backward_goals'])}")
        if inference_result.get("eligible") is not None:
            plan_steps.append(f"  Eligibility result: {inference_result['eligible']}")

        plan_steps.append("Step 6: Expert System selects response template")
        template_id, response = self.expert.match_rule(tokens, category, inference_result)

        q_value = self.learner.rl.get_q(category, template_id)
        plan_steps.append(f"Step 7: RL Q-value for ({category}, {template_id}) = {q_value}")

        self.agent.set_plan_result(
            plan_steps=plan_steps,
            response=response,
            template_id=template_id,
            search_result=search_result,
            inference_result=inference_result,
            message_id=message_id,
        )

        return {
            "reply": response,
            "category": category,
            "confidence": confidence,
            "goal": self.agent.state.goal,
            "template_id": template_id,
            "message_id": message_id,
            "plan": plan_steps,
            "search": search_result,
            "inference": {
                "working_memory": inference_result.get("working_memory", []),
                "eligible": inference_result.get("eligible"),
            },
            "agent": self.agent.describe(),
        }

    def record_feedback(self, category: str, template_id: str, rating: int) -> dict:
        return self.learner.apply_feedback(category, template_id, rating)
