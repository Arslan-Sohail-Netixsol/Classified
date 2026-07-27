# Week 6 Day 1: AFL Data Foundations — EDA, Feature Engineering & Prediction Targets

## Scenario
Complete AFL dataset analysis to support two downstream systems:
- A **domain-locked chat assistant** (needs clean, reliable features)
- A **prediction system** (match winners, top players, top goal-kickers)

---

## Task 1: Data Inventory & Understanding

### Tables Documented

| Table | Rows | Grain | Primary Key | Joins To |
|---|---|---|---|---|
| `players_info` | 2,843 | 1 row per **player** (career bio) | `id` (player_id) | seasonal_stats, merged_players |
| `seasonal_stats` | 25,471 | 1 row per **player × year × is_finals** | `(player_id, year, is_finals)` | players_info via player_id |
| `team_matches` | 15,808 | 1 row per **team × match** (2 per game) | `id` (match-team level) | seasonal via team + year |
| `merged_players` | 25,471 | Same as seasonal_stats | Same | Pre-joined analytics table |

### Join Map
```
players_info.id ──────────────> seasonal_stats.player_id
                                merged_players.player_id

seasonal_stats.team ──────────> team_matches.team_name
    (after strip normalization)
```

---

## Coverage

| Dimension | Value |
|---|---|
| Year range | 1983 → 2025 (43 seasons) |
| Total unique matches | ~7,904 |
| Unique active teams | 20 |
| Players with bio | 2,843 |
| Players with seasonal stats | 3,107 |

---

## Data Quality Findings

| Issue | Severity | Fix |
|---|---|---|
| Team names have leading/trailing whitespace → 40 raw names instead of 20 | ⚠️ Medium | `.str.strip()` on team_name columns |
| 399 players in seasonal_stats have no bio info (no height/weight/age) | ⚠️ Medium | Impute with team-year median or flag |
| 398 missing crowd values (pre-1990s historical records) | ✅ Low | Not a modeling feature — ignore |
| No duplicate rows in any table | ✅ Clean | — |
| No missing values in performance columns (seasonal_stats) | ✅ Clean | — |

---

## Structural Changes (AFL History)
- **Fitzroy Lions + Brisbane Bears → Brisbane Lions (1997)**: Two historical teams become one
- **Port Adelaide Power** enters AFL in **1997**
- **Fremantle Dockers** enters AFL in **1995**
- **Gold Coast Suns** enters AFL in **2011**
- **GWS Giants** enters AFL in **2012**
- `W. Bulldogs` = `Western Bulldogs` (naming inconsistency across tables)
- Round labels: `0–24` (regular season) + `EF, QF, SF, PF, GF` (finals)

---

## Outlier Analysis (Per-Game Averages)

| Stat | Max | 99th Pct | Verdict |
|---|---|---|---|
| avg_goals | 9.0 | 3.3 | Legitimate elite seasons (e.g., Jason Dunstall era) |
| avg_disposals | 39.0 | 29.0 | Possible for high-usage midfielders |
| avg_kicks | 27.0 | 17.8 | Within range |
| avg_tackles | 14.0 | 7.0 | Legitimate for defensive specialists |
| avg_hit_outs | 60.0 | 27.6 | Ruckmen only — positionally expected |

---

## Task 2: Prediction Targets Defined (Completed)

See `target_data_dictionary.md` for the exact machine-readable contract.

**6 Prediction Targets Constructed:**
1. **T1: `win_binary`** (Classification): Match result (1 = Win, 0 = Loss). Draws dropped.
2. **T2: `margin_home`** (Regression): `home_score - away_score` (Range: -186 to +186).
3. **T3: `is_top_disposal_getter`** (Classification): Player `avg_disposals >= 90th percentile` of the given season (~10% label rate).
4. **T4: `is_top_goalkicker`** (Classification): Player `avg_goals >= 95th percentile` of the given season (~5% label rate).
5. **T5: `cpi_score`** (Regression/Ranking): Weighted composite scoring offensive and defensive impact (`3.0*goals + 1.5*clearances + 2.0*tackles ...`).
6. **T6: `is_top_cpi_player`** (Classification): Player `cpi_score >= 90th percentile` of the given season.

*(Design Decision: Per-season percentiles MUST be used instead of static thresholds to counter era-drift, e.g., disposals growing 30% from 1983 to 2025).*

---

## Task 3: Exploratory Data Analysis (Completed)

See `Week6_Day1_Task3_EDA.ipynb` (12 charts generated).

### Key Team-Level Findings
- **Home Advantage**: 59.6% all-time home win rate (~10pp advantage). COVID 2020 dropped this to ~43% (neutral venues).
- **Era Dominance**: Hawthorn dominated 2005-14; Richmond rose in 2015-22.

### Key Player-Level Findings
- **Consistency**: Elite players (top 20% CPI) have significantly lower Coefficients of Variation (CV). True talent persists.
- **Position Fingerprints**: 4 clear data-driven archetypes (Forward, Midfielder, Defender, Ruck). Rucks strictly defined by `avg_hit_outs > 10`.

### 6 Predictive Relationships (Contract for Day 2 Features)
| PR | Relationship | Finding / Implication |
|---|---|---|
| **PR1** | Recent Binary Form | Teams winning 80-100% of last 5 matches win ~70% of next matches. |
| **PR2** | Rest Days | Teams on short rest (<=5 days) win ~6pp less than teams off a bye (8-9 days). *Need `rest_differential` feature.* |
| **PR3** | Crowd Size | 12pp home win gap between Q1 (<18k) and Q5 (>47k) crowds. |
| **PR4** | Season Round | Later rounds have slightly tighter margins (competitiveness increases). |
| **PR5** | Rolling Avg Margin | **Stronger than binary form (PR1).** Dominant teams (>30 avg margin) win ~85% of next matches. |
| **PR6** | YoY Persistence | Prior-season Disposals (r=0.69) and CPI (r=0.64) are highly persistent. Prior-season Goals (r=0.57) are more volatile. |

---

## Task 5: Reproducible Train/Hold-Out Split (Completed)

See `train_test_split.py` for the reusable Python module, and `data_splitting_strategy.md` for documentation on time-series leakage and realistic performance ceilings.

**Key Concepts:**
- **Strict Temporal Split:** We hold out the most recent seasons (e.g., `> 2022`). Random splitting would leak future information (e.g., future injuries or tactical shifts) into past predictions, ruining evaluating on the unknown future.
- **Realistic Ceiling:** The realistic ceiling for AFL match prediction is **70-75% accuracy**. Anything approaching perfect accuracy (90%+) is a red flag indicating post-match target leakage (e.g., including final score in the features).
