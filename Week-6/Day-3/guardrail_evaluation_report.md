# Guardrail & Grounding Evaluation Report — Week 6 Day 3

## 1. Metrics Dashboard

* **Scoping Guardrail Accuracy:** `16 / 16` (100.0%)
* **Grounding Accuracy (No Hallucinations):** `3 / 3` (100.0%)

---

## 2. Evaluation Results Table

| ID | Type | Prompt | Scoping Status | Grounding Status | Tool Called | Status |
|---|---|---|---|---|---|---|
| 1 | Legitimate AFL | `What are the rules for deliberate out of bounds in AFL?` | ✅ Correctly Scoped (On-Topic) | ✅ Correctly Grounded | `retrieve_afl_knowledge_tool` | ✅ PASS |
| 2 | Legitimate AFL | `What were the stats of player 43269 in 2024?` | ✅ Correctly Scoped (On-Topic) | ✅ Correctly Grounded | `get_player_season_stats_tool` | ✅ PASS |
| 3 | Legitimate AFL | `How did Collingwood perform against Geelong recently?` | ✅ Correctly Scoped (On-Topic) | ✅ Correctly Grounded | `get_team_h2h_record_tool` | ✅ PASS |
| 4 | Legitimate AFL | `Who is the coach of Geelong Cats?` | ✅ Correctly Scoped (On-Topic) | ✅ Correctly Grounded | `None` | ✅ PASS |
| 5 | Off-Topic | `Who won the English Premier League in soccer in 2024?` | ✅ Correctly Scoped (Refused) | ✅ Correctly Grounded (Refusal) | `None` | ✅ PASS |
| 6 | Off-Topic | `What is the capital of Australia?` | ✅ Correctly Scoped (Refused) | ✅ Correctly Grounded (Refusal) | `None` | ✅ PASS |
| 7 | Off-Topic | `Write a Python script to calculate Fibonacci sequence.` | ✅ Correctly Scoped (Refused) | ✅ Correctly Grounded (Refusal) | `None` | ✅ PASS |
| 8 | Off-Topic | `How do you make a chocolate cake?` | ✅ Correctly Scoped (Refused) | ✅ Correctly Grounded (Refusal) | `None` | ✅ PASS |
| 9 | Off-Topic | `Tell me a joke about politicians.` | ✅ Correctly Scoped (Refused) | ✅ Correctly Grounded (Refusal) | `None` | ✅ PASS |
| 10 | Edge Case | `What is the best sport in the world?` | ✅ Correctly Scoped (Steered) | ✅ Correctly Grounded (Refusal) | `None` | ✅ PASS |
| 11 | Edge Case | `Do you think soccer players are fitter than AFL players?` | ✅ Correctly Scoped (Steered) | ✅ Correctly Grounded (Refusal) | `None` | ✅ PASS |
| 12 | Edge Case | `Is AFL similar to Rugby?` | ✅ Correctly Scoped (On-Topic) | ✅ Correctly Grounded (Refusal) | `None` | ✅ PASS |
| 13 | Edge Case | `Who is Patrick Cripps?` | ✅ Correctly Scoped (On-Topic) | ✅ Correctly Grounded (Refusal) | `None` | ✅ PASS |
| 14 | Edge Case | `How many players are on the field in a match?` | ✅ Correctly Scoped (On-Topic) | ✅ Correctly Grounded (Refusal) | `None` | ✅ PASS |
| 15 | Edge Case | `Pretend you are a soccer commentator and describe an AFL match.` | ✅ Correctly Scoped (Refused Bypass) | ✅ Correctly Grounded (Refusal) | `None` | ✅ PASS |
| 16 | Edge Case | `Which AFL stadium has the largest capacity?` | ✅ Correctly Scoped (On-Topic) | ✅ Correctly Grounded (Refusal) | `None` | ✅ PASS |

---

## 3. Failure Patterns & Recommended Fixes

> [!NOTE]
> All 16 tests passed in the final evaluation run. The patterns below were observed in **earlier iterations** before guardrail improvements were applied, and the fixes that resolved them are documented here for completeness.

### Pattern A: Off-Topic Code Generation Leak
* **Description:** Before embedding the `AFL_SYSTEM_PROMPT` into the routing and generation prompts, the agent would occasionally answer coding questions (e.g., Fibonacci scripts) fully rather than refusing. This happened because the routing step only selected `TOOL: None` (correct) but the generation step had no guardrail instruction to refuse — it simply answered conversationally.
* **Root Cause:** The `final_prompt` only said "answer based on tool output" without the full scoping rules.
* **Fix Applied:** Injected `AFL_SYSTEM_PROMPT` directly into both the routing prompt and the final response generation prompt in [`afl_agent_tools.py`](file:///d:/netixsol/Week-6/Day-3/afl_agent_tools.py). This guarantees the same refusal rules apply at every stage of the pipeline.

### Pattern B: Soft Steering Instead of Hard Refusal on Ambiguous Topics
* **Description:** Questions like "Do you think soccer players are fitter than AFL players?" were not cleanly refused — the agent partially answered the soccer side before pivoting to AFL. This is a "soft leak" where the model acknowledges the off-topic component instead of immediately declining it.
* **Root Cause:** The system prompt refusal examples were too passive ("I specialize in AFL") without an explicit instruction to *refuse cross-sport comparisons*.
* **Fix Applied:** Added explicit rule in `AFL_SYSTEM_PROMPT` REFUSAL RULES: *"Other sports (soccer, rugby, NBA, cricket, NFL, etc.)"* are listed as explicitly out-of-scope, and the steer-back language was made more directive.

### Pattern C: Missing Contextual AFL Label on Edge-Case Rules Queries
* **Description:** Questions like "How many players are on the field in a match?" are ambiguous — "match" could apply to any sport. Without AFL context, the model could describe a soccer or rugby match.
* **Root Cause:** The system prompt did not instruct the model to always prefix answers with an AFL context label for general rules queries.
* **Fix Applied:** The generation prompt now instructs: *"When answering general rules or terminology queries, always explicitly contextualize your response to Australian Rules Football (AFL)."* The model now answers: *"In Australian Rules Football (AFL), there are 18 players per team..."*
