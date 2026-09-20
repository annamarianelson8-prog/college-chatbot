# Viva Demo Script — Intelligent College Assistant Chatbot

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
