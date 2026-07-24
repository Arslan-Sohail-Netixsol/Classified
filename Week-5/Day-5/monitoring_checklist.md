# Agent Monitoring Checklist (Production)

This one-page checklist defines how the Support Ticket Triage Agent should be monitored, alerted on, and re-evaluated once deployed to production.

## 1. What to Track (Key Metrics)

| Metric | Description | How to Measure |
|---|---|---|
| **Error / Failure Rate** | Frequency of graph crashes, API timeouts, or unhandled tool exceptions. | Track `status: Error` outputs and HTTP 5xx codes from the API. |
| **Cost Drift** | Changes in token usage per run over time (indicates prompt bloat or runaway loops). | Sum input/output tokens per session ID and plot daily averages. |
| **Latency (P95)** | End-to-end execution time for an automated response. | Time from API request to response (excluding human-in-the-loop wait time). |
| **Output Quality Drift** | Degradation in classification accuracy or hallucination rates. | Sample 5% of tickets weekly for manual grading against our 5 criteria. |
| **Escalation Rate** | The % of tickets routed to the `Billing` or `Human Review` nodes. | Count occurrences of `needs_human=True` per day. |

## 2. Alert Thresholds (PagerDuty / Slack)

Set up automated alerting for the following thresholds:
- 🚨 **Critical Error Spike:** > 2% of runs return an unhandled exception in a 5-minute window. *(Action: Rollback to previous prompt/model version)*
- 🚨 **High Latency:** P95 latency exceeds 15 seconds. *(Action: Check external KB API or LLM provider status)*
- ⚠️ **Cost Drift:** Average cost per run increases by > 15% week-over-week. *(Action: Review prompt templates and graph cycles)*
- ⚠️ **Escalation Anomaly:** Human escalation rate drops below 5% or spikes above 40%. *(Action: Investigate if classification guardrails failed or if a specific issue is trending)*

## 3. Re-Evaluation Cadence

The agent ecosystem is dynamic (LLM providers update models, business logic changes). We commit to the following cadence:
- **Weekly:** Manual review of 5% of edge-case/adversarial inputs to update guardrails.
- **Bi-Weekly:** Review the Knowledge Base (KB) hit rate. If "No relevant KB article found" occurs > 20% of the time, the KB needs updating.
- **Monthly / Post-Model-Update:** Rerun the automated Golden Dataset (including the 8 test cases from our evaluation framework) to catch regressions anytime the underlying LLM (e.g., GPT-4o) receives a backend update.
