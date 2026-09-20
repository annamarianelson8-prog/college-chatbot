"""Module 5: Expert System production rules and response selection."""

from __future__ import annotations

from search import get_fact, get_facts_by_predicate, load_kb
from nlp_utils import contains_any, tokenize


PRODUCTION_RULES = [
    {
        "id": "admission_overview",
        "category": "admission",
        "keywords": ["admission", "apply", "document", "eligibility", "deadline"],
        "template": "admission_overview",
    },
    {
        "id": "fee_btech_y1",
        "category": "fees",
        "keywords": ["fee", "btech"],
        "extra": ["y1", "first", "year", "1"],
        "template": "fee_btech_y1",
    },
    {
        "id": "fee_bca_y1",
        "category": "fees",
        "keywords": ["fee", "bca"],
        "template": "fee_bca_y1",
    },
    {
        "id": "hostel_eligible",
        "category": "hostel",
        "keywords": ["eligible", "hostel"],
        "template_eligible": "hostel_eligible_yes",
        "template_ineligible": "hostel_eligible_no",
        "template_default": "hostel_info",
    },
    {
        "id": "hostel_info",
        "category": "hostel",
        "keywords": ["hostel"],
        "template": "hostel_info",
    },
    {
        "id": "exam_schedule",
        "category": "exams",
        "keywords": ["exam", "midterm", "endsem", "attendance", "revaluation"],
        "template": "exam_schedule",
    },
    {
        "id": "faculty_cse",
        "category": "faculty",
        "keywords": ["faculty", "hod", "cse"],
        "template": "faculty_cse",
    },
    {
        "id": "faculty_ece",
        "category": "faculty",
        "keywords": ["faculty", "hod", "ece"],
        "template": "faculty_ece",
    },
    {
        "id": "library_info",
        "category": "library",
        "keywords": ["library"],
        "template": "library_info",
    },
    {
        "id": "events_info",
        "category": "events",
        "keywords": ["event", "fest", "technovision", "rhythm", "sports"],
        "template": "events_info",
    },
    {
        "id": "placement_info",
        "category": "placement",
        "keywords": ["placement", "company", "internship"],
        "template": "placement_info",
    },
    {
        "id": "scholarship_info",
        "category": "scholarship",
        "keywords": ["scholarship", "merit", "waiver"],
        "template": "scholarship_info",
    },
    {
        "id": "contact_info",
        "category": "contact",
        "keywords": ["contact", "phone", "email", "address", "helpdesk"],
        "template": "contact_info",
    },
    {
        "id": "course_cse",
        "category": "courses",
        "keywords": ["cse", "course", "seat", "duration"],
        "template": "course_cse",
    },
    {
        "id": "course_ece",
        "category": "courses",
        "keywords": ["ece", "course"],
        "template": "course_ece",
    },
    {
        "id": "course_bca",
        "category": "courses",
        "keywords": ["bca", "course"],
        "template": "course_bca",
    },
    {
        "id": "general_college",
        "category": "general",
        "keywords": ["college", "about", "overview"],
        "template": "general_college",
    },
    {
        "id": "general_facilities",
        "category": "general",
        "keywords": ["campus", "wifi", "ragging", "cafeteria", "medical", "nss", "ncc"],
        "template": "general_facilities",
    },
    {
        "id": "general_labs",
        "category": "general",
        "keywords": ["lab"],
        "template": "general_labs",
    },
]


class ExpertSystem:
    def __init__(self, kb: dict | None = None):
        self.kb = kb or load_kb()
        self.rules = PRODUCTION_RULES

    def _fact_value(self, predicate: str, args: list | None = None, unit: bool = False) -> str:
        fact = get_fact(self.kb, predicate, args)
        if not fact:
            return "N/A"
        value = str(fact.get("value", "N/A"))
        if unit and fact.get("unit"):
            value = f"{value} {fact['unit']}"
        return value

    def build_context(self) -> dict[str, str]:
        contact = self.kb.get("contact", {})
        ctx = {
            "college_name": self.kb.get("college_name", "College"),
            "last_updated": self.kb.get("last_updated", ""),
            "phone": contact.get("phone", ""),
            "email": contact.get("email", ""),
            "helpdesk_hours": contact.get("helpdesk_hours", ""),
            "address": contact.get("address", ""),
            "admission_deadline": self._fact_value("admission_deadline"),
            "admission_documents": self._fact_value("admission_documents"),
            "admission_eligibility": self._fact_value("admission_eligibility"),
            "admission_process": self._fact_value("admission_process"),
            "fee_BTech_Y1": self._fact_value("fee", ["BTech", "Y1"]),
            "fee_BCA_Y1": self._fact_value("fee", ["BCA", "Y1"]),
            "fee_exam": self._fact_value("fee", ["exam"], unit=True),
            "fee_bus": self._fact_value("fee", ["bus"], unit=True),
            "fee_hostel": self._fact_value("fee", ["hostel"], unit=True),
            "hostel_eligibility": self._fact_value("hostel_eligibility"),
            "hostel_facilities": self._fact_value("hostel_facilities"),
            "hostel_rules": self._fact_value("hostel_rules"),
            "exam_midterm_month": self._fact_value("exam_midterm_month"),
            "exam_endsem_month": self._fact_value("exam_endsem_month"),
            "exam_attendance_rule": self._fact_value("exam_attendance_rule"),
            "exam_revaluation_deadline": self._fact_value("exam_revaluation_deadline"),
            "faculty_hod_CSE": self._fact_value("faculty_hod", ["CSE"]),
            "faculty_hod_email_CSE": self._fact_value("faculty_hod_email", ["CSE"]),
            "faculty_hod_ECE": self._fact_value("faculty_hod", ["ECE"]),
            "faculty_hod_email_ECE": self._fact_value("faculty_hod_email", ["ECE"]),
            "library_hours_weekday": self._fact_value("library_hours", ["weekday"]),
            "library_hours_saturday": self._fact_value("library_hours", ["saturday"]),
            "library_hours_sunday": self._fact_value("library_hours", ["sunday"]),
            "library_books": self._fact_value("library_books"),
            "library_digital": self._fact_value("library_digital"),
            "event_name_tech_fest": self._fact_value("event_name", ["tech_fest"]),
            "event_date_tech_fest": self._fact_value("event_date", ["tech_fest"]),
            "event_name_cultural_fest": self._fact_value("event_name", ["cultural_fest"]),
            "event_date_cultural_fest": self._fact_value("event_date", ["cultural_fest"]),
            "event_name_sports_day": self._fact_value("event_name", ["sports_day"]),
            "event_date_sports_day": self._fact_value("event_date", ["sports_day"]),
            "placement_rate": self._fact_value("placement_rate"),
            "placement_companies": self._fact_value("placement_companies"),
            "internship_support": self._fact_value("internship_support"),
            "scholarship_merit": self._fact_value("scholarship_merit"),
            "scholarship_sports": self._fact_value("scholarship_sports"),
            "scholarship_eligibility": self._fact_value("scholarship_eligibility"),
            "course_duration_BTech_CSE": self._fact_value("course_duration", ["BTech", "CSE"]),
            "course_seats_BTech_CSE": self._fact_value("course_seats", ["BTech", "CSE"]),
            "course_duration_BTech_ECE": self._fact_value("course_duration", ["BTech", "ECE"]),
            "course_seats_BTech_ECE": self._fact_value("course_seats", ["BTech", "ECE"]),
            "course_duration_BCA": self._fact_value("course_duration", ["BCA"]),
            "course_seats_BCA": self._fact_value("course_seats", ["BCA"]),
            "campus_wifi": self._fact_value("campus_wifi"),
            "anti_ragging": self._fact_value("anti_ragging"),
            "cafeteria": self._fact_value("cafeteria"),
            "medical": self._fact_value("medical"),
            "nss_ncc": self._fact_value("nss_ncc"),
            "lab_facilities": self._fact_value("lab_facilities"),
        }
        return ctx

    def render_template(self, template_key: str) -> str:
        responses = self.kb.get("responses", {})
        template = responses.get(template_key, responses.get("fallback", ""))
        ctx = self.build_context()
        try:
            return template.format(**ctx)
        except KeyError:
            return responses.get("fallback", "Please contact the helpdesk.").format(**ctx)

    def match_rule(self, tokens: list[str], category: str, inference: dict | None = None) -> tuple[str, str]:
        inference = inference or {}
        category_rules = [r for r in self.rules if r.get("category") == category]
        if not category_rules:
            category_rules = self.rules

        best_rule = None
        best_score = -1
        for rule in category_rules:
            keywords = rule.get("keywords", [])
            score = sum(1 for k in keywords if k in tokens)
            extra = rule.get("extra", [])
            if extra and not contains_any(tokens, extra):
                score -= 1
            if score > best_score:
                best_score = score
                best_rule = rule

        if best_rule is None or best_score <= 0:
            fallback = self.render_template("fallback")
            return "fallback", fallback

        rule_id = best_rule["id"]
        if "template_eligible" in best_rule:
            eligible = inference.get("eligible")
            if eligible is True:
                template_key = best_rule["template_eligible"]
            elif eligible is False:
                template_key = best_rule["template_ineligible"]
            else:
                template_key = best_rule.get("template_default", "hostel_info")
        else:
            template_key = best_rule["template"]

        return rule_id, self.render_template(template_key)

    def keyword_category_backup(self, tokens: list[str], normalized: str = "") -> str | None:
        backup_map = {
            "admission": ["admission", "apply", "document"],
            "fees": ["fee", "tuition"],
            "exams": ["exam", "midterm", "attendance", "revaluation", "percent"],
            "hostel": ["hostel"],
            "faculty": ["faculty", "hod"],
            "library": ["library"],
            "events": ["fest", "event", "technovision", "rhythm", "schedule"],
            "contact": ["contact", "phone", "email", "helpdesk"],
            "placement": ["placement", "company"],
            "scholarship": ["scholarship"],
            "courses": ["course", "btech", "bca", "mca", "cse", "ece", "program", "offer"],
        }
        for category, keys in backup_map.items():
            if contains_any(tokens, keys):
                return category
            if normalized and any(k in normalized for k in keys):
                return category
        return None
