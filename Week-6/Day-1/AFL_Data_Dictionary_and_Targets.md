# AFL Data Dictionary & Target Definitions

This unified document serves as the master contract for downstream predictive modeling and domain-locked chat assistants.

---

## 1. Prediction Target Definitions

| Target Name | Type | Level | Definition | Justification |
|---|---|---|---|---|
| **T1: `win_binary`** | Classification | Match | `1` if Home team wins, `0` if Loss. | Fundamental classification task. Draws are excluded. |
| **T2: `margin_home`** | Regression | Match | `home_score - away_score` | Allows for probability calibration and spread betting. |
| **T3: `is_top_disposal`** | Classification | Player | `1` if player `avg_disposals >= 90th percentile` of the season. | Top 10% cutoff prevents era-drift (as disposals increased 30% over 40 years). |
| **T4: `is_top_goalkicker`**| Classification | Player | `1` if player `avg_goals >= 95th percentile` of the season. | Top 5% isolates elite key forwards (min 5 games played). |
| **T5: `cpi_score`** | Regression | Player | `3.0*goals + 1.5*clearances + 2.0*tackles + 1.0*disposals + 1.5*marks + 1.0*inside_50 - 0.5*clangers - 0.5*frees_against` | Comprehensive impact metric capturing both offensive and defensive acts. |
| **T6: `is_top_cpi`** | Classification | Player | `1` if player `cpi_score >= 90th percentile` of the season. | Predicts the true elite game-changers in a given season. |

---

## 2. Match-Level Feature Engineering (`match_features_v1.csv`)
**Grain:** 1 row per Match (from Home team's perspective). 
**Anti-Leakage Strategy:** All features computed via `shift(1)` chronological windows.

| Feature Name | Type | Description (Historical context before match) | Source |
|---|---|---|---|
| `feat_diff_form_5` | Float | Diff in win rate over last 5 games (Home - Away). | `result` |
| `feat_diff_margin_5` | Float | Diff in avg point margin over last 5 games. | `margin` |
| `feat_diff_rest` | Float | Diff in rest days since previous match. | `match_date` |
| `feat_diff_ladder` | Float | Diff in cumulative ladder points in current season. | `result, year` |
| `home_feat_h2h_win_5` | Float | Home team's win rate vs this opponent (last 5 H2H). | `result, opp` |
| `home_feat_venue_win_10` | Float | Home team's win rate at this specific venue (last 10). | `result, venue`|

---

## 3. Player-Level Feature Engineering (`player_features_v1.csv`)
**Grain:** 1 row per Player per Season.
**Anti-Leakage Strategy:** All predictive features use `shift(1)` pointing strictly to the prior season.

| Feature Name | Type | Description (Historical context before season) | Source |
|---|---|---|---|
| `feat_prev_cpi` | Float | Player's CPI score in the previous season. | `cpi_score` |
| `feat_prev_disposals` | Float | Player's avg disposals in the previous season. | `avg_disposals`|
| `feat_prev_goals` | Float | Player's avg goals in the previous season. | `avg_goals` |
| `feat_career_seasons` | Integer | Total seasons played prior to current year. | `year` |
| `feat_position_proxy` | Categorical| Role inference (Forward/Mid/Def/Ruck) based on prior stats. | `hit_outs`, etc |
