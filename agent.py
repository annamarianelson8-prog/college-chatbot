"""Module 1: Goal-Based Intelligent Agent (PEAS model)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PEAS:
    performance: str = "Provide accurate, timely college information to students"
    environment: str = "College website chat interface with student queries"
    actuators: str = "Text responses, category routing, feedback prompts"
    sensors: str = "User messages, thumbs up/down feedback"


@dataclass
class AgentState:
    query: str = ""
    tokens: list[str] = field(default_factory=list)
    category: str = "general"
    goal: str = ""
    plan: list[str] = field(default_factory=list)
    response: str = ""
    template_id: str = "fallback"
    search_result: dict = field(default_factory=dict)
    inference_result: dict = field(default_factory=dict)
    message_id: str = ""


class GoalBasedAgent:
    """Intelligent agent that maps user queries to answer goals."""

    GOAL_MAP = {
        "admission": "answer_admission_query",
        "fees": "answer_fee_query",
        "exams": "answer_exam_query",
        "hostel": "verify_hostel_eligibility",
        "faculty": "answer_faculty_query",
        "library": "answer_library_query",
        "events": "answer_events_query",
        "contact": "provide_contact_info",
        "placement": "answer_placement_query",
        "scholarship": "verify_scholarship_eligibility",
        "courses": "answer_course_query",
        "general": "answer_general_query",
    }

    def __init__(self):
        self.peas = PEAS()
        self.state = AgentState()

    def perceive(self, query: str, tokens: list[str], category: str) -> AgentState:
        self.state = AgentState(
            query=query,
            tokens=tokens,
            category=category,
            goal=self.GOAL_MAP.get(category, "answer_general_query"),
        )
        return self.state

    def set_plan_result(
        self,
        plan_steps: list[str],
        response: str,
        template_id: str,
        search_result: dict,
        inference_result: dict,
        message_id: str,
    ) -> AgentState:
        self.state.plan = plan_steps
        self.state.response = response
        self.state.template_id = template_id
        self.state.search_result = search_result
        self.state.inference_result = inference_result
        self.state.message_id = message_id
        return self.state

    def act(self) -> str:
        return self.state.response

    def describe(self) -> dict:
        return {
            "peas": {
                "performance": self.peas.performance,
                "environment": self.peas.environment,
                "actuators": self.peas.actuators,
                "sensors": self.peas.sensors,
            },
            "current_goal": self.state.goal,
            "category": self.state.category,
        }
