# AFL Assistant: Monitoring & Maintenance Plan

This document outlines the operational plan for the AFL Assistant once deployed to production (e.g. Web3Geeks or a client platform).

## 1. Production Monitoring Checklist

Structured JSON logs (emitted via `afl_assistant_core.py` and `api.py`) are ingested into Datadog/CloudWatch. The following metrics must be actively monitored:

| Metric | Target / Threshold | Action if Breached |
|---|---|---|
| **P95 Response Latency** | < 4.0 seconds | High latency usually indicates the LLM router is hitting rate limits. Scale API tier or temporarily force `use_llm_router=False` (rely on regex fallback). |
| **Tool Error Rate** | < 2% of queries | Investigate `tool_error` field in logs. Usually indicates an unexpected schema change in the scraped AFL tables. |
| **Off-Topic Leak Rate** | < 1% false positives | If genuine AFL queries are blocked (e.g. "What is the recipe for winning?"), refine the `_OFF_TOPIC_SIGNALS` or few-shot examples in the Router prompt. |
| **Prediction Brier Score** | < 0.22 (rolling 4 weeks) | A spike in Brier score means probabilities are becoming uncalibrated. Triggers immediate model retraining. |

## 2. Model Maintenance & Retraining Loop

The match-winner and top-player models are trained on historical data up to 2025. Once deployed, they require a weekly refresh loop as the real season progresses.

### The Weekly Cycle (Tuesday Nights)
1. **Data Ingestion**: Scrape the completed weekend's match results and player statistics.
2. **Feature Engineering**: Re-run the CPI (Career Performance Index) calculation to update player ratings.
3. **Model Evaluation (Shadow Mode)**:
   - Calculate the Brier score of last week's predictions against actual results.
   - Run a naive baseline check (e.g., did the model outperform "Home Team Always Wins"?).
4. **Retraining**: 
   - If performance degraded below baseline, retrain the `LogisticRegression` and `HistGradientBoostingRegressor` on the newly appended data.
   - If performance is stable, continue using the existing pickled model weights.

### Addressing Model Drift
AFL meta shifts over time (e.g., changes to the "holding the ball" rule impacting disposal counts). If the model consistently under-predicts high-disposal players, the feature weights for `prior_avg_disposals` may have drifted. In this case, the training window is shifted (e.g., dropping pre-2015 data) to prioritize modern tactical trends.
