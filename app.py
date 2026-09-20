"""Flask web application — Intelligent College Assistant Chatbot."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from nlp_utils import ensure_nltk_data

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
_planner = None
_planner_error = None


def load_college_meta() -> dict:
    kb_path = BASE_DIR / "knowledge_base.json"
    with open(kb_path, encoding="utf-8") as f:
        kb = json.load(f)
    contact = kb.get("contact", {})
    return {
        "college_name": kb.get("college_name", "College Assistant"),
        "phone": contact.get("phone", ""),
        "email": contact.get("email", ""),
        "address": contact.get("address", ""),
        "helpdesk_hours": contact.get("helpdesk_hours", ""),
    }


def get_planner():
    global _planner, _planner_error
    if _planner is not None:
        return _planner
    if _planner_error:
        raise RuntimeError(_planner_error)
    try:
        from planner import ResponsePlanner

        _planner = ResponsePlanner()
        return _planner
    except Exception as exc:
        _planner_error = str(exc)
        raise


@app.route("/")
def index():
    return render_template("index.html", **load_college_meta())


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message is required."}), 400
    try:
        result = get_planner().plan_and_respond(message)
        return jsonify(result)
    except Exception:
        traceback.print_exc()
        meta = load_college_meta()
        return jsonify({
            "reply": (
                f"Sorry, I could not process that just now. "
                f"Please contact {meta['college_name']} helpdesk at "
                f"{meta['phone']} or {meta['email']}."
            ),
            "category": "general",
            "goal": "fallback",
            "plan": ["Error while generating a reply. Helpdesk contact returned."],
            "message_id": "error",
            "template_id": "fallback",
        })


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}
    category = data.get("category", "general")
    template_id = data.get("template_id", "fallback")
    rating = data.get("rating", 0)
    if rating not in (1, -1):
        return jsonify({"error": "Rating must be 1 or -1."}), 400
    try:
        result = get_planner().record_feedback(category, template_id, rating)
        return jsonify({"status": "ok", **result})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/agent-info")
def agent_info():
    try:
        return jsonify(get_planner().agent.describe())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    try:
        ensure_nltk_data()
    except Exception:
        pass
    try:
        get_planner()
        print("Chatbot engine ready.")
    except Exception as exc:
        print("Warning: chatbot engine failed to load:", exc)
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
