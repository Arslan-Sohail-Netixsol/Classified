# AFL Assistant Pro: Executive Summary

## 1. Product Goal
The AFL Assistant Pro is a domain-locked, interactive conversational agent designed to provide accurate AFL knowledge, historical statistics, and probabilistic predictions for match outcomes and player performance. The goal is to provide a premium, interactive "analyst in your pocket" experience for fans, while strictly guarding against hallucinations, non-AFL queries, and prompt injections.

## 2. Architecture
The system is built on a **LangGraph state machine** orchestration layer, wrapped in a **FastAPI** backend and exposed via a **Streamlit** user interface.

- **Router Node**: An LLM-powered intent classifier (with a rule-based regex fallback for high-load scenarios) that buckets queries into `prediction`, `retrieval`, `factual`, or `off_topic`.
- **Retrieval Node**: Executes deterministic database lookups for Head-to-Head records and player statistics.
- **Prediction Node**: Invokes heavily calibrated `LogisticRegression` and `HistGradientBoosting` models to predict match winners and player CPIs, injecting strict probabilistic disclaimers.
- **Validation & Clarification**: Traps hallucinated team names or invalid data ranges and routes back to the user to clarify, preventing silent failures.

## 3. Evaluation Results
The system was evaluated against a 25+ case comprehensive test suite:
- **Factual / Retrieval**: Handled seamlessly by the Day-3 semantic agent and Day-2 DB tools.
- **Scope Guardrails**: 100% success in trapping explicit prompt injections ("ignore previous instructions") and out-of-domain requests (NBA, recipes, code generation).
- **Prediction Benchmark**: The Match Winner model achieved **66.8% accuracy**, successfully outperforming the naive "Home Team Wins" baseline (approx 58%), particularly by correctly identifying historically dominant away teams.

## 4. Known Limitations
- **Data Recency**: The current models are trained on data up to the 2025 season. Unseen rookies or major team roster overhauls in 2026 are not fully captured until the next retraining cycle.
- **Rate Limiting**: The system relies on the Grok API. During bursts of traffic, the system falls back to a regex router which is less semantically accurate (e.g., misclassifying "How many players?" as a database retrieval).
- **Nuanced Queries**: Extremely complex multi-hop queries (e.g., "Which team that won the premiership also had the highest average CPI in finals?") currently exceed the standard tool interfaces.

## 5. Recommended Next Steps
1. **Upgrade Infrastructure**: Transition from the Grok free tier to a provisioned throughput tier to eliminate `429 RESOURCE_EXHAUSTED` errors and remove reliance on the brittle regex fallback.
2. **Automated Weekly Retraining Loop**: Implement the proposed Datadog-monitored weekly data pipeline to ingest weekend match results and automatically retrain models on Tuesday nights to combat concept drift.
3. **Advanced RAG integration**: Ingest AFL rulebooks and coaching manuals into a vector store to allow the `DirectAnswerNode` to answer deep tactical questions with specific rule citations.
