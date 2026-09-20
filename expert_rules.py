"""Module 5: Expert System production rules and response selection."""

from __future__ import annotations

import re

from search import get_fact, load_kb
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
        "extra": ["y1", "first", "year", "1"],
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
        "keywords": ["hostel", "room", "mess", "wifi", "laundry", "security"],
        "template": "hostel_info",
    },
    {
        "id": "exam_schedule",
        "category": "exams",
        "keywords": ["exam", "midterm", "endsem", "attendance", "revaluation", "supplementary"],
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
        "id": "faculty_me",
        "category": "faculty",
        "keywords": ["faculty", "hod", "me", "mechanical"],
        "template": "faculty_me",
    },
    {
        "id": "library_info",
        "category": "library",
        "keywords": ["library", "books", "journal", "digital", "ieee", "acm", "nptel"],
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
        "keywords": ["placement", "company", "internship", "recruit"],
        "template": "placement_info",
    },
    {
        "id": "scholarship_info",
        "category": "scholarship",
        "keywords": ["scholarship", "merit", "waiver", "sports", "income"],
        "template": "scholarship_info",
    },
    {
        "id": "contact_info",
        "category": "contact",
        "keywords": ["contact", "phone", "email", "address", "helpdesk", "location"],
        "template": "contact_info",
    },
    {
        "id": "courses_overview",
        "category": "courses",
        "keywords": ["course", "program", "offer", "branch", "available"],
        "template": "courses_overview",
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
        "keywords": ["ece", "course", "seat", "duration"],
        "template": "course_ece",
    },
    {
        "id": "course_me",
        "category": "courses",
        "keywords": ["me", "mechanical", "course", "seat", "duration"],
        "template": "course_me",
    },
    {
        "id": "course_bca",
        "category": "courses",
        "keywords": ["bca", "course", "seat", "duration"],
        "template": "course_bca",
    },
    {
        "id": "course_mca",
        "category": "courses",
        "keywords": ["mca", "course", "program"],
        "template": "courses_overview",
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
        "keywords": [
            "campus", "wifi", "ragging", "cafeteria", "medical",
            "nss", "ncc", "facility", "facilities"
        ],
        "template": "general_facilities",
    },
    {
        "id": "general_labs",
        "category": "general",
        "keywords": ["lab", "laboratory"],
        "template": "general_labs",
    },
]


class ExpertSystem:
    def __init__(self, kb: dict | None = None):
        self.kb = kb or load_kb()
        self.rules = PRODUCTION_RULES

    # ---------- text matching helpers ----------

    @staticmethod
    def _variants(tokens: list[str]) -> set[str]:
        """Make simple singular/plural variants so 'courses' matches 'course'."""
        result: set[str] = set()

        for token in tokens:
            word = token.lower().strip()
            if not word:
                continue

            result.add(word)

            if len(word) > 3 and word.endswith("ies"):
                result.add(word[:-3] + "y")
            elif len(word) > 3 and word.endswith("es"):
                result.add(word[:-2])
            elif len(word) > 3 and word.endswith("s"):
                result.add(word[:-1])

        return result

    def _faq_match(
        self,
        tokens: list[str],
        category: str,
        normalized: str = "",
    ) -> tuple[str, str] | None:
        """
        Search the optional FAQ section added to knowledge_base.json.

        This is deliberately placed before production-rule matching so
        natural questions can receive a direct factual answer.
        """
        faq_items = self.kb.get("faq", [])
        if not isinstance(faq_items, list):
            return None

        user_tokens = self._variants(tokens)
        best = None
        best_score = 0.0

        for item in faq_items:
            if not isinstance(item, dict):
                continue

            questions = item.get("questions", [])
            if isinstance(questions, str):
                questions = [questions]

            # If the FAQ has a category, respect it.
            item_category = item.get("category")
            if item_category and item_category != category:
                continue

            for question in questions:
                if not isinstance(question, str):
                    continue

                q_normalized = " ".join(tokenize(question)).lower()
                q_tokens = self._variants(tokenize(question))
                if not q_tokens:
                    continue

                overlap = len(user_tokens & q_tokens)
                coverage = overlap / max(1, len(q_tokens))
                user_coverage = overlap / max(1, len(user_tokens))

                # Strong exact/near-exact phrase match.
                if q_normalized and (
                    q_normalized == normalized
                    or q_normalized in normalized
                    or normalized in q_normalized
                ):
                    score = 1.0
                else:
                    # F1-like score gives good matches without requiring
                    # every question word to be identical.
                    score = (2 * coverage * user_coverage) / max(
                        0.0001, coverage + user_coverage
                    )

                # Require at least two useful matching words for a generic FAQ.
                if overlap < 2 and score < 1.0:
                    continue

                # Category is already known from the classifier, so matching
                # within that category gets a small bonus.
                if item_category == category:
                    score += 0.10

                if score > best_score:
                    best_score = score
                    best = item

        if best is None or best_score < 0.42:
            return None

        answer = best.get("answer", "")
        if not isinstance(answer, str) or not answer.strip():
            return None

        return best.get("id", "faq"), self._render_text(answer)

    # ---------- knowledge-base context ----------

    def _fact_value(
        self,
        predicate: str,
        args: list | None = None,
        unit: bool = False,
    ) -> str:
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
            "fee_BTech_Y2": self._fact_value("fee", ["BTech", "Y2"]),
            "fee_BTech_Y3": self._fact_value("fee", ["BTech", "Y3"]),
            "fee_BTech_Y4": self._fact_value("fee", ["BTech", "Y4"]),
            "fee_BCA_Y1": self._fact_value("fee", ["BCA", "Y1"]),
            "fee_BCA_Y2": self._fact_value("fee", ["BCA", "Y2"]),
            "fee_BCA_Y3": self._fact_value("fee", ["BCA", "Y3"]),
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
            "exam_supplementary": self._fact_value("exam_supplementary"),

            "faculty_hod_CSE": self._fact_value("faculty_hod", ["CSE"]),
            "faculty_hod_email_CSE": self._fact_value("faculty_hod_email", ["CSE"]),
            "faculty_hod_ECE": self._fact_value("faculty_hod", ["ECE"]),
            "faculty_hod_email_ECE": self._fact_value("faculty_hod_email", ["ECE"]),
            "faculty_hod_ME": self._fact_value("faculty_hod", ["ME"]),
            "faculty_hod_email_ME": self._fact_value("faculty_hod_email", ["ME"]),

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

            "course_duration_BTech_CSE": self._fact_value(
                "course_duration", ["BTech", "CSE"]
            ),
            "course_seats_BTech_CSE": self._fact_value(
                "course_seats", ["BTech", "CSE"]
            ),
            "course_duration_BTech_ECE": self._fact_value(
                "course_duration", ["BTech", "ECE"]
            ),
            "course_seats_BTech_ECE": self._fact_value(
                "course_seats", ["BTech", "ECE"]
            ),
            "course_duration_BTech_ME": self._fact_value(
                "course_duration", ["BTech", "ME"]
            ),
            "course_seats_BTech_ME": self._fact_value(
                "course_seats", ["BTech", "ME"]
            ),
            "course_duration_BCA": self._fact_value(
                "course_duration", ["BCA"]
            ),
            "course_seats_BCA": self._fact_value(
                "course_seats", ["BCA"]
            ),

            "campus_wifi": self._fact_value("campus_wifi"),
            "anti_ragging": self._fact_value("anti_ragging"),
            "cafeteria": self._fact_value("cafeteria"),
            "medical": self._fact_value("medical"),
            "nss_ncc": self._fact_value("nss_ncc"),
            "lab_facilities": self._fact_value("lab_facilities"),
        }

        return ctx

    # ---------- response rendering ----------

    def _render_text(self, text: str) -> str:
        """Render a direct FAQ answer or normal response using KB facts."""
        ctx = self.build_context()

        try:
            return text.format(**ctx)
        except (KeyError, ValueError):
            # If an optional placeholder is missing, keep the useful answer
            # instead of replacing the whole response with the generic fallback.
            return re.sub(r"\{[^{}]+\}", "", text).strip()

    def render_template(self, template_key: str) -> str:
        responses = self.kb.get("responses", {})
        template = responses.get(template_key)

        if not template:
            template = responses.get("fallback", "Please contact the helpdesk.")

        return self._render_text(template)

    # ---------- rule selection ----------

    def match_rule(
        self,
        tokens: list[str],
        category: str,
        inference: dict | None = None,
        normalized: str = "",
    ) -> tuple[str, str]:
        inference = inference or {}

        # 1. First try the expanded FAQ knowledge base.
        faq_result = self._faq_match(tokens, category, normalized)
        if faq_result:
            return faq_result

        # 2. Then use the production-rule expert system.
        category_rules = [
            r for r in self.rules if r.get("category") == category
        ]

        if not category_rules:
            category_rules = self.rules

        user_tokens = self._variants(tokens)

        best_rule = None
        best_score = -1

        for rule in category_rules:
            keywords = self._variants(rule.get("keywords", []))
            score = len(user_tokens & keywords)

            extra = self._variants(rule.get("extra", []))
            if extra and not (user_tokens & extra):
                score -= 1

            # Prefer more specific rules.
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
                template_key = best_rule.get(
                    "template_default",
                    "hostel_info",
                )
        else:
            template_key = best_rule["template"]

        return rule_id, self.render_template(template_key)

    def keyword_category_backup(
        self,
        tokens: list[str],
        normalized: str = "",
    ) -> str | None:
        backup_map = {
            "admission": ["admission", "apply", "document"],
            "fees": ["fee", "tuition", "cost", "price"],
            "exams": [
                "exam", "midterm", "attendance",
                "revaluation", "percent", "supplementary"
            ],
            "hostel": ["hostel", "room", "mess", "laundry"],
            "faculty": ["faculty", "hod", "professor"],
            "library": ["library", "books", "journal"],
            "events": ["fest", "event", "technovision", "rhythm", "schedule"],
            "contact": ["contact", "phone", "email", "helpdesk", "address"],
            "placement": ["placement", "company", "internship", "recruit"],
            "scholarship": ["scholarship", "merit", "waiver"],
            "courses": [
                "course", "btech", "bca", "mca", "cse",
                "ece", "mechanical", "program", "offer",
                "branch", "degree"
            ],
        }

        token_variants = self._variants(tokens)

        for category, keys in backup_map.items():
            key_variants = self._variants(keys)

            if token_variants & key_variants:
                return category

            if normalized and any(
                k in normalized for k in keys
            ):
                return category

        return None
