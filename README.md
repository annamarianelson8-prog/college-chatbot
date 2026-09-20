# Intelligent College Assistant Chatbot

Zero-cost AI chatbot for college websites. Implements BFS, A*, Forward/Backward Chaining, Decision Tree, Reinforcement Learning, Expert System rules, and classical planning — aligned with S5 Artificial Intelligence syllabus.

## Quick Start

```bash
cd CHATAIPRO
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

On first run, NLTK data (`punkt`, `stopwords`) downloads automatically.

## Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Flask server (`/`, `/chat`, `/feedback`) |
| `agent.py` | Module 1 — Goal-Based Agent (PEAS) |
| `search.py` | Module 2 — BFS & A* over KB graph |
| `inference.py` | Module 3 — Forward/Backward chaining |
| `planner.py` | Module 4 — Response planning pipeline |
| `learner.py` | Module 5 — Decision Tree + RL feedback |
| `expert_rules.py` | Module 5 — Expert System production rules |
| `knowledge_base.json` | College facts, FOL rules, KB graph |
| `data/training_data.csv` | Labeled queries for Decision Tree |
| `data/college_data_template.csv` | Template to collect data from your college website |

## Customize for Your College

1. Visit your college website and fill in [data/college_data_template.csv](data/college_data_template.csv).
2. Update facts in [knowledge_base.json](knowledge_base.json) — change `college_name`, `contact`, and all `facts`.
3. Add matching question variations to [data/training_data.csv](data/training_data.csv).
4. Delete `data/decision_tree_model.joblib` and restart the app to retrain the classifier.

## Viva Demo Script

1. **PEAS**: Visit `/agent-info` or enable "Show AI reasoning plan" in the UI.
2. **BFS**: Ask *"What is the B.Tech first year fee?"* — plan shows BFS path through `fees -> BTech -> Y1`.
3. **Backward Chaining**: Ask *"Am I eligible for hostel?"* — shows eligibility inference.
4. **A***: Ask a multi-part query like *"Scholarship and fee waiver for merit students"* — plan uses A*.
5. **RL**: Click thumbs down on a reply, then check `data/feedback_log.json` for updated Q-values.

## Run Tests

```bash
python test_chatbot.py
```

Runs 50+ sample queries and prints pass/fail summary.

## Free Deployment (Optional)

**Render.com (free tier):**

1. Push project to GitHub.
2. Create a Web Service on Render, connect repo.
3. Build: `pip install -r requirements.txt`
4. Start: `gunicorn app:app`

See [render.yaml](render.yaml) for a starter config.

## Tech Stack (100% Free)

- Python 3 + Flask
- NLTK (tokenization)
- scikit-learn (Decision Tree)
- JSON knowledge base
- HTML/CSS/JS frontend

No paid LLM APIs required.
