# Validation & Fallback Node Report — Week 6 Day 4 Task 4

> Validates the ValidationNode, ClarificationNode, and FallbackNode across 12 test scenarios.

## 1. Test Results Summary

**Overall:** 12/12 tests passed

| Category | Tests | Passed |
|---|---|---|
| PASS (tool succeeded) | 1 | 1 |
| CLARIFY (loop back) | 5 | 5 |
| FALLBACK (out of scope) | 6 | 6 |

## 2. Full Results Table

| ID | Description | Expected | Actual | Error Class | Status |
|---|---|---|---|---|---|
| 01 | Valid prediction — slang teams | `ValidationOutcome.PASS` | `ValidationOutcome.PASS` | `none` | ✅ |
| 02 | Unknown team name | `ValidationOutcome.CLARIFY` | `ValidationOutcome.CLARIFY` | `unknown_team` | ✅ |
| 03 | Same team both sides | `ValidationOutcome.CLARIFY` | `ValidationOutcome.CLARIFY` | `same_team` | ✅ |
| 04 | Unknown player ID | `ValidationOutcome.CLARIFY` | `ValidationOutcome.CLARIFY` | `unknown_player` | ✅ |
| 05 | Year out of range (too early) | `ValidationOutcome.CLARIFY` | `ValidationOutcome.CLARIFY` | `invalid_year` | ✅ |
| 06 | Year out of range (too late) | `ValidationOutcome.CLARIFY` | `ValidationOutcome.CLARIFY` | `invalid_year` | ✅ |
| 07 | Unsupported stat — tackles | `ValidationOutcome.FALLBACK` | `ValidationOutcome.FALLBACK` | `unsupported_stat` | ✅ |
| 08 | Unsupported stat — Brownlow votes | `ValidationOutcome.FALLBACK` | `ValidationOutcome.FALLBACK` | `unsupported_stat` | ✅ |
| 09 | Unsupported intent — live score | `ValidationOutcome.FALLBACK` | `ValidationOutcome.FALLBACK` | `unsupported_stat` | ✅ |
| 10 | Unsupported intent — betting odds | `ValidationOutcome.FALLBACK` | `ValidationOutcome.FALLBACK` | `unsupported_stat` | ✅ |
| 11 | Model unavailable (pkl missing) | `ValidationOutcome.FALLBACK` | `ValidationOutcome.FALLBACK` | `model_unavailable` | ✅ |
| 12 | Unknown team — after max retries (force fallback) | `ValidationOutcome.FALLBACK` | `ValidationOutcome.FALLBACK` | `unknown_team` | ✅ |

## 3. Validation Graph Design

```
[router] → [prediction_node | retrieval_node | direct_node]
                         │
                         ▼
              [ValidationNode]              ← Task 4
             /      |       \
         PASS   CLARIFY   FALLBACK
           |       |          |
      [formatter] [ClarificationNode] [FallbackNode]
           |           │                  │
           └───────────┴──────────────────┘
                        │
                   [final_response]
```

## 4. Error Class Taxonomy

### Resolvable → CLARIFY path

| ErrorClass | Trigger | Action |
|---|---|---|
| `UNKNOWN_TEAM` | Team name not in database or nickname map | Ask for a valid team name with examples |
| `MISSING_TEAM` | No team extracted from query | Ask user to specify team(s) |
| `SAME_TEAM` | Home and away resolved to same team | Ask for a different away team |
| `UNKNOWN_PLAYER` | Player ID not in player_features_v1.csv | Ask for valid ID or suggest team-level query |
| `INVALID_YEAR` | Year outside 1984–2025 range | State valid range and ask to re-specify |
| `MISSING_YEAR` | No year context given for player stat query | Ask for year or confirm 2025 default |

### Unresolvable → FALLBACK path

| ErrorClass | Trigger | Response |
|---|---|---|
| `UNSUPPORTED_STAT` | Query asks for tackles, marks, Brownlow, odds, etc. | List supported stats (CPI, disposals, goals) + examples |
| `UNSUPPORTED_INTENT` | Live scores, injury reports, transfers, betting | State that real-time data is unavailable + alternatives |
| `MODEL_UNAVAILABLE` | .pkl file missing | Show resolution steps + suggest retrieval/factual queries |
| `HISTORICAL_LIMIT` | Year < 1984 | State data range + suggest a year in range |
| `FUTURE_LIMIT` | Year > 2025 | State training cutoff + suggest 2025 |
| `SYSTEM_ERROR` | Unexpected Python exception | Show error detail + rephrasing suggestions |

## 5. Scope Catalogue

### Supported Capabilities

| Capability | Description |
|---|---|
| Match prediction | Predict win probability for any two AFL teams (1984-2025) |
| Top player prediction | Predict top-ranked players for any AFL team by CPI, disposals, or goals |
| Stat types | CPI (Composite Performance Index), Disposals, Goals |
| H2H retrieval | Historical head-to-head records between any two teams |
| Player stats | Season stats for any player by ID (1984-2025) |
| Rules lookup | AFL rules, terminology, venue facts via knowledge base |
| Factual AFL questions | General AFL questions: team histories, player profiles, rules |
| Team aliases | 40+ nickname / slang mappings (Pies, Cats, Freo, Tiges, …) |
| Season range | Data available for 1984–2025 |

### NOT Supported

| Capability | Reason |
|---|---|
| tackles | Tackle counts are not modelled — we don't have per-game tackle features. |
| marks | Mark counts are not in our feature set. |
| hitouts | Hitout predictions are not supported (Ruck-specific, very sparse data). |
| rebounds | Rebound stat is not modelled. |
| frees for | Free-kick counts are not in the training data. |
| frees against | Free-kick-against counts are not in the training data. |
| score involvements | Score involvement rate is not a modelled feature. |
| contested | Contested possession count is not modelled. |
| clearances | Clearance count is not modelled. |
| rating | We use CPI (Composite Performance Index) as our overall rating. |
| fantasy | Direct Fantasy AFL points are not modelled — use CPI as a proxy. |
| supercoach | SuperCoach scores are not modelled. |

## 6. Anti-Hallucination Design

| Principle | Implementation |
|---|---|
| Never guess a team name | `NicknameResolver.resolve()` raises `AFLValidationError` if unresolvable — CLARIFY path fires |
| Never guess a year | `DateResolver.resolve()` defaults to current season (explicit), never invents a year |
| Never fabricate stats | `predict_match_winner_tool` / `predict_top_player_tool` only call real model — never LLM-generated numbers |
| Never pretend to model unsupported stats | `FallbackNode` lists exactly what we model (CPI, disposal, goal) |
| Max retry guard | After 2 CLARIFY loops, forces FALLBACK to avoid infinite clarification spirals |
