"""Module 5: Decision Tree classifier and tabular Reinforcement Learning."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from nlp_utils import normalize_text, tokenize

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAINING_PATH = DATA_DIR / "training_data.csv"
MODEL_PATH = DATA_DIR / "decision_tree_model.joblib"
FEEDBACK_PATH = DATA_DIR / "feedback_log.json"

CATEGORIES = [
    "admission", "fees", "exams", "hostel", "faculty",
    "library", "events", "contact", "placement", "scholarship", "courses", "general",
]


class QueryClassifier:
    def __init__(self):
        self.pipeline: Pipeline | None = None

    def load_training_data(self) -> tuple[list[str], list[str]]:
        texts: list[str] = []
        labels: list[str] = []
        with open(TRAINING_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(normalize_text(row["text"]))
                labels.append(row["category"])
        return texts, labels

    def train(self) -> Pipeline:
        texts, labels = self.load_training_data()
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", DecisionTreeClassifier(max_depth=12, random_state=42)),
        ])
        self.pipeline.fit(texts, labels)
        joblib.dump(self.pipeline, MODEL_PATH)
        return self.pipeline

    def load_or_train(self) -> Pipeline:
        if MODEL_PATH.exists():
            self.pipeline = joblib.load(MODEL_PATH)
            return self.pipeline
        return self.train()

    def classify(self, query: str) -> tuple[str, float]:
        if self.pipeline is None:
            self.load_or_train()
        text = normalize_text(query)
        category = self.pipeline.predict([text])[0]
        probas = self.pipeline.predict_proba([text])[0]
        confidence = float(max(probas))
        return category, confidence


class ReinforcementLearner:
    """Simple tabular Q-learning: state=(category, template_id), reward=+1/-1."""

    def __init__(self, alpha: float = 0.3, gamma: float = 0.9):
        self.alpha = alpha
        self.gamma = gamma
        self.q_values: dict[str, float] = {}
        self.history: list[dict] = []
        self._load()

    def _load(self) -> None:
        if FEEDBACK_PATH.exists():
            with open(FEEDBACK_PATH, encoding="utf-8") as f:
                data = json.load(f)
                self.q_values = data.get("q_values", {})
                self.history = data.get("history", [])

    def _save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            json.dump({"q_values": self.q_values, "history": self.history}, f, indent=2)

    def _key(self, category: str, template_id: str) -> str:
        return f"{category}::{template_id}"

    def get_q(self, category: str, template_id: str) -> float:
        return self.q_values.get(self._key(category, template_id), 0.0)

    def update(self, category: str, template_id: str, reward: int) -> float:
        key = self._key(category, template_id)
        old = self.q_values.get(key, 0.0)
        new = old + self.alpha * (reward - old)
        self.q_values[key] = round(new, 4)
        self.history.append({
            "category": category,
            "template_id": template_id,
            "reward": reward,
            "q_value": self.q_values[key],
        })
        self._save()
        return self.q_values[key]

    def prefer_template(self, category: str, candidates: list[str]) -> str | None:
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda t: self.get_q(category, t), reverse=True)
        return ranked[0]


class LearnerEngine:
    def __init__(self):
        self.classifier = QueryClassifier()
        self.rl = ReinforcementLearner()
        self.classifier.load_or_train()

    def classify_query(self, query: str) -> dict:
        category, confidence = self.classifier.classify(query)
        return {"category": category, "confidence": round(confidence, 3)}

    def apply_feedback(self, category: str, template_id: str, rating: int) -> dict:
        reward = 1 if rating > 0 else -1
        q = self.rl.update(category, template_id, reward)
        return {"q_value": q, "reward": reward}

    def keyword_fallback(self, query: str) -> str:
        tokens = tokenize(query)
        normalized = normalize_text(query)
        general_keywords = ["wifi", "ragging", "cafeteria", "medical", "nss", "ncc", "overview", "lab", "campus"]
        if (
            any(k in tokens for k in general_keywords)
            or "about this college" in normalized
            or "college overview" in normalized
            or normalized.startswith("tell me about")
        ):
            return "general"
        if "admission" in normalized or ("apply" in normalized and "date" in normalized):
            return "admission"
        if "library" in normalized:
            return "library"
        if "faculty" in normalized or " hod" in f" {normalized}" or "department head" in normalized:
            return "faculty"
        if (
            "technovision" in normalized
            or "sports day" in normalized
            or "rhythm" in normalized
            or "tech fest" in normalized
            or "college fest" in normalized
            or ("upcoming" in normalized and "event" in normalized)
        ):
            return "events"
        if ("last date" in normalized or "deadline" in normalized) and "apply" in normalized:
            return "admission"
        if "apply" in normalized and "admission" in normalized:
            return "admission"
        if "helpdesk" in normalized or "working hour" in normalized:
            return "contact"
        if "revaluation" in normalized or "attendance" in normalized or "percent" in normalized:
            return "exams"
        if "courses offered" in normalized or " mca" in f" {normalized}" or normalized.startswith("mca "):
            return "courses"
        keyword_map = {
            "admission": ["admission", "apply", "document", "eligibility"],
            "fees": ["fee", "tuition", "cost"],
            "exams": ["exam", "midterm", "attendance", "revaluation", "percent"],
            "hostel": ["hostel", "stay", "accommodation"],
            "faculty": ["faculty", "hod", "professor"],
            "library": ["library", "book"],
            "events": ["fest", "event", "technovision", "rhythm", "schedule", "sports"],
            "contact": ["contact", "phone", "email", "address", "helpdesk"],
            "placement": ["placement", "company", "internship"],
            "scholarship": ["scholarship", "merit", "waiver"],
            "courses": ["course", "btech", "bca", "mca", "cse", "ece", "seat", "program", "offer"],
        }
        for category, keys in keyword_map.items():
            if any(k in tokens for k in keys):
                return category
            if any(k in normalized for k in keys):
                return category
        return "general"
