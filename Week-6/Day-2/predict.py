# -*- coding: utf-8 -*-
"""
afl_predictors.py
=================
Production-ready callable interfaces for both AFL prediction models.

Classes
-------
MatchWinnerPredictor   -- predict_match_winner(home_team, away_team, ...)
TopPlayerPredictor     -- predict_top_players(team, year, ...)

Both classes:
  - Load their respective saved pipelines once at init time.
  - Accept plain-English inputs (team name, position, stat type).
  - Validate all inputs and raise clear AFL-domain-specific errors.
  - Return typed Python dicts / lists ready to be serialised to JSON.
  - Are designed to be thin wrappers -- Day 4 LangChain tools call them
    as simple Python functions.

Usage examples are at the bottom of this file and in HOW_TO_CALL.md.
"""

from __future__ import annotations

import warnings
import sys
import json
import re
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ── Resolve paths ──────────────────────────────────────────────────────────
_HERE    = Path(__file__).parent
_DAY1    = _HERE.parent / "Day-1"
_MATCH_MODEL_PATH  = _HERE / "match_winner_model.pkl"
_PLAYER_MODEL_PATH = _HERE / "top_player_model.pkl"
_MATCH_CSV         = _DAY1 / "match_features_v1.csv"
_PLAYER_CSV        = _DAY1 / "player_features_v1.csv"

sys.path.insert(0, str(_DAY1))
from train_test_split import temporal_train_test_split


# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS — AFL domain knowledge
# ══════════════════════════════════════════════════════════════════════════

# Canonical team names as they appear in match_features_v1.csv (home_team col)
CANONICAL_TEAMS: set[str] = {
    "Adelaide Crows",
    "Brisbane Bears",
    "Brisbane Lions",
    "Carlton Blues",
    "Collingwood Magpies",
    "Essendon Bombers",
    "Fitzroy Lions",
    "Fremantle Dockers",
    "Geelong Cats",
    "Gold Coast Suns",
    "Greater Western Sydney Giants",
    "Hawthorn Hawks",
    "Melbourne Demons",
    "North Melbourne Kangaroos",
    "Port Adelaide Power",
    "Richmond Tigers",
    "St Kilda Saints",
    "Sydney Swans",
    "W. Bulldogs",
    "West Coast Eagles",
    # common aliases
    "Western Bulldogs",
    "GWS Giants",
    "GWS",
    "North Melbourne",
    "Carlton",
    "Collingwood",
    "Essendon",
    "Fremantle",
    "Geelong",
    "Hawthorn",
    "Melbourne",
    "Port Adelaide",
    "Richmond",
    "Brisbane",
    "Adelaide",
    "Sydney",
    "West Coast",
    "Gold Coast",
    "St Kilda",
}

# Alias normaliser: maps common short-names / variants -> canonical match CSV name
_TEAM_ALIASES: dict[str, str] = {
    # Short forms
    "Adelaide":                          "Adelaide Crows",
    "Brisbane":                          "Brisbane Lions",
    "Carlton":                           "Carlton Blues",
    "Collingwood":                       "Collingwood Magpies",
    "Essendon":                          "Essendon Bombers",
    "Fremantle":                         "Fremantle Dockers",
    "Geelong":                           "Geelong Cats",
    "Gold Coast":                        "Gold Coast Suns",
    "GWS":                               "Greater Western Sydney Giants",
    "GWS Giants":                        "Greater Western Sydney Giants",
    "Greater Western Sydney":            "Greater Western Sydney Giants",
    "Hawthorn":                          "Hawthorn Hawks",
    "Melbourne":                         "Melbourne Demons",
    "North Melbourne":                   "North Melbourne Kangaroos",
    "North":                             "North Melbourne Kangaroos",
    "Port Adelaide":                     "Port Adelaide Power",
    "Port":                              "Port Adelaide Power",
    "Richmond":                          "Richmond Tigers",
    "St Kilda":                          "St Kilda Saints",
    "Sydney":                            "Sydney Swans",
    "West Coast":                        "West Coast Eagles",
    "Western Bulldogs":                  "W. Bulldogs",
    "Bulldogs":                          "W. Bulldogs",
    "W. Bulldogs":                       "W. Bulldogs",
    "Fitzroy":                           "Fitzroy Lions",
    "Brisbane Bears":                    "Brisbane Bears",
    # Upper-case variants (player CSV)
    "ADELAIDE CROWS":                    "Adelaide Crows",
    "BRISBANE BEARS":                    "Brisbane Bears",
    "BRISBANE LIONS":                    "Brisbane Lions",
    "CARLTON BLUES":                     "Carlton Blues",
    "COLLINGWOOD MAGPIES":               "Collingwood Magpies",
    "ESSENDON BOMBERS":                  "Essendon Bombers",
    "FITZROY LIONS":                     "Fitzroy Lions",
    "FREMANTLE DOCKERS":                 "Fremantle Dockers",
    "GEELONG CATS":                      "Geelong Cats",
    "GOLD COAST SUNS":                   "Gold Coast Suns",
    "GREATER WESTERN SYDNEY GIANTS":     "Greater Western Sydney Giants",
    "HAWTHORN HAWKS":                    "Hawthorn Hawks",
    "MELBOURNE DEMONS":                  "Melbourne Demons",
    "NORTH MELBOURNE KANGAROOS":         "North Melbourne Kangaroos",
    "PORT ADELAIDE POWER":               "Port Adelaide Power",
    "RICHMOND TIGERS":                   "Richmond Tigers",
    "ST KILDA SAINTS":                   "St Kilda Saints",
    "SYDNEY SWANS":                      "Sydney Swans",
    "WEST COAST EAGLES":                 "West Coast Eagles",
    "WESTERN BULLDOGS":                  "W. Bulldogs",
}

VALID_POSITIONS: set[str] = {"Defender", "Forward", "General", "Midfielder", "Ruck"}
VALID_STAT_TYPES: set[str] = {"cpi", "disposal", "goal"}
DATA_YEAR_MIN: int = 1983
DATA_YEAR_MAX: int = 2025


# ══════════════════════════════════════════════════════════════════════════
# CUSTOM EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════

class AFLValidationError(ValueError):
    """Raised when caller passes invalid AFL-domain inputs."""


class AFLModelNotLoaded(RuntimeError):
    """Raised when the model .pkl file cannot be found or loaded."""


# ══════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def _normalise_team(name: str) -> str:
    """
    Normalise a team name to its canonical form.

    Accepts: full names, short names, uppercase variants, common aliases.
    Raises AFLValidationError if unrecognised.
    """
    if not isinstance(name, str) or not name.strip():
        raise AFLValidationError(
            "team name must be a non-empty string. "
            f"Valid examples: 'Geelong Cats', 'Richmond Tigers', 'GWS Giants'."
        )
    stripped = name.strip()
    # Try exact match first
    if stripped in _TEAM_ALIASES:
        return _TEAM_ALIASES[stripped]
    # Case-insensitive alias lookup
    lower = stripped.lower()
    for alias, canon in _TEAM_ALIASES.items():
        if alias.lower() == lower:
            return canon
    # Fuzzy partial match (last resort)
    candidates = [
        canon for alias, canon in _TEAM_ALIASES.items()
        if lower in alias.lower() or alias.lower() in lower
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise AFLValidationError(
        f"Unknown team name: '{name}'. "
        f"Valid teams include: {sorted({v for v in _TEAM_ALIASES.values()})[:6]}... "
        f"(Use a canonical name like 'Richmond Tigers' or alias like 'Richmond')."
    )


def _validate_year(year: int) -> None:
    if not isinstance(year, int):
        raise AFLValidationError(f"year must be an integer, got {type(year).__name__}.")
    if year < DATA_YEAR_MIN or year > DATA_YEAR_MAX:
        raise AFLValidationError(
            f"year {year} is outside the AFL data range "
            f"({DATA_YEAR_MIN}–{DATA_YEAR_MAX})."
        )


def _validate_stat_type(stat_type: str) -> str:
    """Normalise stat_type to one of: 'cpi', 'disposal', 'goal'."""
    if not isinstance(stat_type, str):
        raise AFLValidationError(
            f"stat_type must be a string. Valid values: {sorted(VALID_STAT_TYPES)}."
        )
    clean = stat_type.strip().lower()
    # Alias map
    aliases = {
        "cpi": "cpi", "impact": "cpi", "composite": "cpi", "performance": "cpi",
        "disposal": "disposal", "disposals": "disposal", "kicks": "disposal",
        "goal": "goal", "goals": "goal", "goalkicker": "goal", "goalkicking": "goal",
    }
    if clean not in aliases:
        raise AFLValidationError(
            f"Unknown stat_type '{stat_type}'. "
            f"Valid options: 'cpi' (composite performance), 'disposal', 'goal'."
        )
    return aliases[clean]


# ══════════════════════════════════════════════════════════════════════════
# 1. MATCH WINNER PREDICTOR
# ══════════════════════════════════════════════════════════════════════════

class MatchWinnerPredictor:
    """
    Predicts the winner and win probability for an AFL match.

    The model (Logistic Regression + Platt calibration) was trained on
    match_features_v1.csv and evaluated on 2024-2025 seasons.
    Reported test-set performance: Accuracy=0.668, ROC-AUC=0.643, Brier=0.231.

    Parameters
    ----------
    model_path : Path-like, optional
        Path to the saved joblib pipeline. Defaults to Day-2 directory.
    data_path : Path-like, optional
        Path to match_features_v1.csv for computing feature context.

    Example
    -------
    >>> predictor = MatchWinnerPredictor()
    >>> result = predictor.predict("Richmond Tigers", "Collingwood Magpies")
    >>> print(result)
    {
        'home_team': 'Richmond Tigers',
        'away_team': 'Collingwood Magpies',
        'predicted_winner': 'Richmond Tigers',
        'home_win_probability': 0.614,
        'away_win_probability': 0.386,
        'confidence': 'moderate',
        'model': 'Logistic Regression (Platt Calibration)',
        'note': 'Probability based on 2023 season median features for home team.'
    }
    """

    # Feature columns required by the pipeline
    _NUMERIC_FEATS = [
        "home_feat_form_5", "home_feat_rolling_margin_5", "home_feat_rolling_score_5",
        "home_feat_rest_days", "home_feat_ladder_pts", "home_feat_h2h_win_5",
        "home_feat_venue_win_10", "diff_form_5", "diff_rolling_margin_5",
        "diff_rolling_score_5", "diff_rest_days", "diff_ladder_pts",
        "diff_h2h_win_5", "diff_venue_win_10", "is_finals", "round_num",
    ]
    _ALL_FEATS = _NUMERIC_FEATS + ["home_team"]

    def __init__(
        self,
        model_path: Optional[Path] = None,
        data_path: Optional[Path]  = None,
    ):
        model_path = Path(model_path) if model_path else _MATCH_MODEL_PATH
        data_path  = Path(data_path)  if data_path  else _MATCH_CSV

        if not model_path.exists():
            raise AFLModelNotLoaded(
                f"Match winner model not found at '{model_path}'. "
                "Run match_winner_model.py first to train and save the model."
            )
        self._pipeline = joblib.load(model_path)
        self._season_medians = self._compute_season_medians(data_path)
        self._latest_year = int(self._season_medians.index.max())

    @staticmethod
    def _compute_season_medians(data_path: Path) -> pd.DataFrame:
        """Compute per-season medians from training data (year <= 2023)."""
        mf = pd.read_csv(data_path)
        train = mf[mf["year"] <= 2023]
        home_feats = [
            "home_feat_form_5", "home_feat_rolling_margin_5",
            "home_feat_rolling_score_5", "home_feat_rest_days",
            "home_feat_ladder_pts", "home_feat_h2h_win_5",
            "home_feat_venue_win_10",
        ]
        meds = train.groupby("year")[home_feats].median()
        meds.columns = [c.replace("home_feat_", "med_") for c in home_feats]
        return meds

    def _build_feature_row(
        self,
        home_team: str,
        away_team: str,
        season_year: int,
        is_finals: bool,
        round_num: int,
        home_form_5: Optional[float],
        home_ladder_pts: Optional[float],
        home_rolling_margin: Optional[float],
        home_h2h_win_5: Optional[float],
        home_venue_win_10: Optional[float],
        home_rest_days: Optional[float],
    ) -> pd.DataFrame:
        """
        Build a single-row DataFrame matching the pipeline's expected input.
        Uses season medians as defaults for any None inputs.
        """
        # Nearest available season medians
        available_years = sorted(self._season_medians.index)
        ref_year = max(y for y in available_years if y <= season_year)
        meds = self._season_medians.loc[ref_year]

        home_form_5          = home_form_5          if home_form_5          is not None else float(meds["med_form_5"])
        home_ladder_pts      = home_ladder_pts      if home_ladder_pts      is not None else float(meds["med_ladder_pts"])
        home_rolling_margin  = home_rolling_margin  if home_rolling_margin  is not None else float(meds["med_rolling_margin_5"])
        home_h2h_win_5       = home_h2h_win_5       if home_h2h_win_5       is not None else float(meds["med_h2h_win_5"])
        home_venue_win_10    = home_venue_win_10    if home_venue_win_10    is not None else float(meds["med_venue_win_10"])
        home_rest_days       = home_rest_days       if home_rest_days       is not None else float(meds["med_rest_days"])
        home_rolling_score   = float(meds["med_rolling_score_5"])   # always use median (no direct input)

        row = {
            "home_feat_form_5":           home_form_5,
            "home_feat_rolling_margin_5": home_rolling_margin,
            "home_feat_rolling_score_5":  home_rolling_score,
            "home_feat_rest_days":        home_rest_days,
            "home_feat_ladder_pts":       home_ladder_pts,
            "home_feat_h2h_win_5":        home_h2h_win_5,
            "home_feat_venue_win_10":     home_venue_win_10,
            # Differential features (home minus season median)
            "diff_form_5":           home_form_5          - float(meds["med_form_5"]),
            "diff_rolling_margin_5": home_rolling_margin  - float(meds["med_rolling_margin_5"]),
            "diff_rolling_score_5":  home_rolling_score   - float(meds["med_rolling_score_5"]),
            "diff_rest_days":        home_rest_days        - float(meds["med_rest_days"]),
            "diff_ladder_pts":       home_ladder_pts       - float(meds["med_ladder_pts"]),
            "diff_h2h_win_5":        home_h2h_win_5        - float(meds["med_h2h_win_5"]),
            "diff_venue_win_10":     home_venue_win_10     - float(meds["med_venue_win_10"]),
            "is_finals":             int(is_finals),
            "round_num":             round_num,
            "home_team":             home_team,
        }
        return pd.DataFrame([row])

    def predict(
        self,
        home_team: str,
        away_team: str,
        season_year: Optional[int] = None,
        is_finals: bool = False,
        round_num: int = 12,
        home_form_5: Optional[float] = None,
        home_ladder_pts: Optional[float] = None,
        home_rolling_margin: Optional[float] = None,
        home_h2h_win_5: Optional[float] = None,
        home_venue_win_10: Optional[float] = None,
        home_rest_days: Optional[float] = None,
    ) -> dict:
        """
        Predict the winner of an AFL match.

        Parameters
        ----------
        home_team : str
            Home team name. Accepts full name or common alias.
            e.g., 'Richmond Tigers', 'Richmond', 'Geelong Cats', 'Geelong'.
        away_team : str
            Away team name. Same format as home_team.
        season_year : int, optional
            Season year for context (default: latest available = 2023).
        is_finals : bool, optional
            Whether this is a finals match (default False).
        round_num : int, optional
            Round number 0-24 for regular season, or 25-29 for finals.
        home_form_5 : float, optional
            Home team's win rate over last 5 games (0.0-1.0).
            Default: season median (0.4-0.6 depending on year).
        home_ladder_pts : float, optional
            Home team's cumulative ladder points this season.
            Default: season median (~20 pts).
        home_rolling_margin : float, optional
            Home team's average point margin over last 5 games.
            Default: season median (~0-3 pts).
        home_h2h_win_5 : float, optional
            Home team's H2H win rate vs this opponent (last 5 meetings).
        home_venue_win_10 : float, optional
            Home team's win rate at this venue (last 10 games there).
        home_rest_days : float, optional
            Days of rest since home team's last match (default: 7).

        Returns
        -------
        dict with keys:
            home_team           : str    -- normalised home team name
            away_team           : str    -- normalised away team name
            predicted_winner    : str    -- team name of predicted winner
            home_win_probability: float  -- calibrated P(home win), 0.0-1.0
            away_win_probability: float  -- 1 - home_win_probability
            confidence          : str    -- 'low' (<55%), 'moderate' (55-70%), 'high' (>70%)
            model               : str    -- model identifier
            features_used       : dict   -- the feature values passed to the model
            note                : str    -- any caveats or warnings

        Raises
        ------
        AFLValidationError  : Unknown team name, invalid year, bad feature values.
        AFLModelNotLoaded   : Model file missing.
        """
        # ── Input validation ───────────────────────────────────────────────
        home_team = _normalise_team(home_team)
        away_team = _normalise_team(away_team)

        if home_team == away_team:
            raise AFLValidationError(
                f"home_team and away_team cannot be the same team ('{home_team}')."
            )

        if season_year is None:
            season_year = self._latest_year
        _validate_year(season_year)

        if not isinstance(round_num, int) or not (-1 <= round_num <= 29):
            raise AFLValidationError(
                f"round_num must be an integer 0-24 (regular season) or "
                f"25-29 (finals). Got {round_num}."
            )

        for name, val in [
            ("home_form_5", home_form_5),
            ("home_h2h_win_5", home_h2h_win_5),
            ("home_venue_win_10", home_venue_win_10),
        ]:
            if val is not None and not (0.0 <= val <= 1.0):
                raise AFLValidationError(
                    f"'{name}' must be between 0.0 and 1.0, got {val}."
                )

        if home_rest_days is not None and not (0 <= home_rest_days <= 30):
            raise AFLValidationError(
                f"home_rest_days must be between 0 and 30, got {home_rest_days}."
            )

        # ── Feature construction ───────────────────────────────────────────
        X = self._build_feature_row(
            home_team=home_team,
            away_team=away_team,
            season_year=season_year,
            is_finals=is_finals,
            round_num=round_num,
            home_form_5=home_form_5,
            home_ladder_pts=home_ladder_pts,
            home_rolling_margin=home_rolling_margin,
            home_h2h_win_5=home_h2h_win_5,
            home_venue_win_10=home_venue_win_10,
            home_rest_days=home_rest_days,
        )

        # ── Prediction ─────────────────────────────────────────────────────
        prob_home = float(self._pipeline.predict_proba(X[self._ALL_FEATS])[0, 1])
        prob_away = 1.0 - prob_home
        winner = home_team if prob_home >= 0.5 else away_team

        if prob_home < 0.55:
            confidence = "low"
        elif prob_home < 0.70:
            confidence = "moderate"
        else:
            confidence = "high"

        # Notes / caveats
        notes = []
        if home_form_5 is None or home_ladder_pts is None:
            notes.append(
                "Some features defaulted to season medians — provide actual "
                "match-week stats for a more accurate prediction."
            )
        if season_year > self._latest_year:
            notes.append(
                f"season_year {season_year} is beyond training data; "
                f"using {self._latest_year} medians as context."
            )
        if is_finals:
            notes.append(
                "Finals note: the model may over-estimate home advantage in finals "
                "(away-team quality not modelled). Apply caution for P > 0.80."
            )

        return {
            "home_team":            home_team,
            "away_team":            away_team,
            "predicted_winner":     winner,
            "home_win_probability": round(prob_home, 4),
            "away_win_probability": round(prob_away, 4),
            "confidence":           confidence,
            "model":                "Logistic Regression (Platt Calibration)",
            "test_set_accuracy":    0.6682,
            "test_set_roc_auc":     0.6432,
            "features_used": {
                k: round(float(v), 4) if isinstance(v, float) else v
                for k, v in X.iloc[0].items()
            },
            "note": " | ".join(notes) if notes else "OK",
        }


# ══════════════════════════════════════════════════════════════════════════
# 2. TOP PLAYER PREDICTOR
# ══════════════════════════════════════════════════════════════════════════

class TopPlayerPredictor:
    """
    Predicts top-performing AFL players for a given team and season.

    Uses HGB Regressor to predict each player's CPI (Composite Performance
    Index) and ranks them. Also flags predicted is_top_cpi / is_top_disposal
    / is_top_goal status per player.

    Model performance: NDCG@10=0.931, Precision@10=0.950, ROC-AUC=0.954.

    Parameters
    ----------
    model_path : Path-like, optional
        Path to saved joblib pipeline.
    data_path : Path-like, optional
        Path to player_features_v1.csv.

    Example
    -------
    >>> predictor = TopPlayerPredictor()
    >>> result = predictor.predict(team='Geelong Cats', year=2025, top_k=5)
    >>> for p in result['ranked_players']:
    ...     print(p)
    {'rank': 1, 'player_id': 12345, 'team': 'Geelong Cats',
     'position': 'Midfielder', 'predicted_cpi': 52.3,
     'is_predicted_top_cpi': True, ...}
    """

    # CPI threshold for "top performer" label (90th percentile of training set)
    _CPI_TOP_THRESHOLD    = 45.5   # 90th pct from training data
    _DISP_TOP_THRESHOLD   = None   # derived dynamically from prev season
    _GOAL_TOP_THRESHOLD   = None   # derived dynamically from prev season

    _NUMERIC_FEATS   = [
        "feat_prev_cpi", "feat_prev_disposals", "feat_prev_goals",
        "feat_prev_games", "feat_career_seasons",
    ]
    _CAT_FEATS = ["feat_position_proxy", "team"]
    _ALL_FEATS = _NUMERIC_FEATS + _CAT_FEATS

    def __init__(
        self,
        model_path: Optional[Path] = None,
        data_path: Optional[Path]  = None,
    ):
        model_path = Path(model_path) if model_path else _PLAYER_MODEL_PATH
        data_path  = Path(data_path)  if data_path  else _PLAYER_CSV

        if not model_path.exists():
            raise AFLModelNotLoaded(
                f"Top player model not found at '{model_path}'. "
                "Run top_player_model.py first."
            )
        self._pipeline = joblib.load(model_path)
        self._player_df = pd.read_csv(data_path)
        # Compute percentile thresholds per year from training data
        train = self._player_df[self._player_df["year"] <= 2023]
        self._year_thresholds = (
            train.groupby("year")["target_cpi_raw"]
            .quantile(0.90)
            .rename("cpi_90th")
        )

    def predict(
        self,
        team: str,
        year: int,
        stat_type: str = "cpi",
        top_k: int = 10,
        position_filter: Optional[str] = None,
    ) -> dict:
        """
        Predict and rank the top players for a team in a given season.

        Parameters
        ----------
        team : str
            Team name. Accepts full name or alias.
            e.g., 'Geelong Cats', 'Geelong', 'Richmond Tigers'.
        year : int
            Season year to predict for (must be in 1984-2025; 1983 players
            have no prior-season features).
        stat_type : str, optional
            Performance metric to rank by:
            - 'cpi'      : Composite Performance Index (default)
            - 'disposal' : Disposal-getter ranking
            - 'goal'     : Goal-kicking ranking
        top_k : int, optional
            Number of top players to return (default 10, max 50).
        position_filter : str, optional
            Restrict results to a single position:
            'Defender', 'Forward', 'General', 'Midfielder', 'Ruck'.

        Returns
        -------
        dict with keys:
            team          : str   -- normalised team name
            year          : int   -- prediction season year
            stat_type     : str   -- the requested ranking metric
            top_k         : int   -- number of players returned
            ranked_players: list  -- sorted list of player prediction dicts
            model         : str   -- model identifier
            note          : str   -- any caveats or warnings

        Each player dict contains:
            rank                  : int
            player_id             : int
            team                  : str
            position              : str
            prev_season_cpi       : float
            prev_season_disposals : float
            prev_season_goals     : float
            career_seasons        : int
            predicted_cpi         : float
            is_predicted_top_cpi  : bool
            is_predicted_top_disposal: bool (if stat_type='disposal')
            is_predicted_top_goal : bool (if stat_type='goal')

        Raises
        ------
        AFLValidationError  : Unknown team, invalid year, invalid position/stat.
        AFLModelNotLoaded   : Model file missing.
        """
        # ── Input validation ───────────────────────────────────────────────
        team_canon = _normalise_team(team)
        _validate_year(year)

        if year < DATA_YEAR_MIN + 1:
            raise AFLValidationError(
                f"year must be >= {DATA_YEAR_MIN + 1} because players need at "
                f"least one prior season of data. Got {year}."
            )

        stat_type = _validate_stat_type(stat_type)

        if not isinstance(top_k, int) or not (1 <= top_k <= 50):
            raise AFLValidationError(
                f"top_k must be an integer between 1 and 50, got {top_k}."
            )

        if position_filter is not None:
            pf_clean = position_filter.strip().title()
            if pf_clean not in VALID_POSITIONS:
                raise AFLValidationError(
                    f"Unknown position_filter '{position_filter}'. "
                    f"Valid options: {sorted(VALID_POSITIONS)}."
                )
        else:
            pf_clean = None

        # ── Filter player data ─────────────────────────────────────────────
        # Match team name against the messy player CSV (mixed case)
        # Use case-insensitive matching against canonical name
        canon_lower = team_canon.lower().replace("w. bulldogs", "western bulldogs")
        # Also build alias set for player CSV team column
        team_variants = {
            team_canon,
            team_canon.upper(),
            team_canon.lower(),
            _TEAM_ALIASES.get(team_canon, team_canon),
        }
        # Add Western Bulldogs variants specifically
        if "bulldogs" in canon_lower:
            team_variants.update({"W. Bulldogs", "Western Bulldogs",
                                   "WESTERN BULLDOGS", "western bulldogs"})

        mask_year = self._player_df["year"] == year
        mask_team = self._player_df["team"].isin(team_variants) | \
                    self._player_df["team"].str.lower().str.contains(
                        canon_lower.split()[0].lower(), na=False
                    )
        mask_career = self._player_df["feat_career_seasons"] > 0

        df_team = self._player_df[mask_year & mask_team & mask_career].copy()

        if df_team.empty:
            # Try fuzzy: any team that partially matches
            all_teams_this_year = self._player_df[mask_year]["team"].unique()
            suggestions = [t for t in all_teams_this_year
                           if canon_lower.split()[0].lower() in t.lower()]
            raise AFLValidationError(
                f"No player data found for team '{team}' in year {year}. "
                f"Similar teams in that year: {suggestions[:5]}. "
                f"Note: some historical teams (e.g., Fitzroy Lions, Brisbane Bears) "
                f"only exist in early seasons."
            )

        if pf_clean is not None:
            df_team = df_team[df_team["feat_position_proxy"] == pf_clean]
            if df_team.empty:
                raise AFLValidationError(
                    f"No {pf_clean} players found for '{team}' in {year}. "
                    f"Try a different position or remove the filter."
                )

        # ── Prediction ─────────────────────────────────────────────────────
        X = df_team[self._ALL_FEATS].copy()
        pred_cpi = self._pipeline.predict(X)
        df_team = df_team.copy()
        df_team["predicted_cpi"] = pred_cpi

        # Sort by requested stat_type
        if stat_type == "disposal":
            df_team = df_team.sort_values("feat_prev_disposals", ascending=False)
            score_col = "feat_prev_disposals"
        elif stat_type == "goal":
            df_team = df_team.sort_values("feat_prev_goals", ascending=False)
            score_col = "feat_prev_goals"
        else:
            df_team = df_team.sort_values("predicted_cpi", ascending=False)
            score_col = "predicted_cpi"

        # CPI threshold for this year's "top" label
        cpi_thresh = float(
            self._year_thresholds.get(year,
            self._year_thresholds.iloc[-1])   # fall back to latest known
        )

        # Build ranked list
        top_players = []
        for rank, (_, row) in enumerate(df_team.head(top_k).iterrows(), start=1):
            player_dict = {
                "rank":                   rank,
                "player_id":              int(row["player_id"]),
                "team":                   row["team"],
                "position":               row["feat_position_proxy"],
                "prev_season_cpi":        round(float(row["feat_prev_cpi"]), 2)
                                          if pd.notna(row["feat_prev_cpi"]) else None,
                "prev_season_disposals":  round(float(row["feat_prev_disposals"]), 2)
                                          if pd.notna(row["feat_prev_disposals"]) else None,
                "prev_season_goals":      round(float(row["feat_prev_goals"]), 2)
                                          if pd.notna(row["feat_prev_goals"]) else None,
                "career_seasons":         int(row["feat_career_seasons"]),
                "predicted_cpi":          round(float(row["predicted_cpi"]), 2),
                "is_predicted_top_cpi":   bool(row["predicted_cpi"] >= cpi_thresh),
            }
            if stat_type == "disposal":
                disp_thresh = float(df_team["feat_prev_disposals"].quantile(0.90))
                player_dict["is_predicted_top_disposal"] = bool(
                    row["feat_prev_disposals"] >= disp_thresh
                )
            if stat_type == "goal":
                goal_thresh = float(df_team["feat_prev_goals"].quantile(0.95))
                player_dict["is_predicted_top_goal"] = bool(
                    row["feat_prev_goals"] >= goal_thresh
                )
            top_players.append(player_dict)

        # Caveats
        notes = []
        if year > 2023:
            notes.append(
                f"Year {year} is in the hold-out test period (2024-2025). "
                "Predictions reflect model generalisation, not training fit."
            )
        if df_team.shape[0] < top_k:
            notes.append(
                f"Only {df_team.shape[0]} eligible players found for "
                f"'{team}' in {year} — fewer than requested top_k={top_k}."
            )
        if stat_type == "goal":
            notes.append(
                "Goal predictions are less reliable (prior-season goals r=0.21 "
                "with future CPI). Use with caution."
            )

        return {
            "team":           team_canon,
            "year":           year,
            "stat_type":      stat_type,
            "top_k":          min(top_k, len(top_players)),
            "ranked_players": top_players,
            "model":          "HGB Regressor (CPI Regression + Ranking)",
            "test_ndcg10":    0.9306,
            "test_precision10": 0.9500,
            "note":           " | ".join(notes) if notes else "OK",
        }

    def predict_by_player_id(
        self,
        player_id: int,
        year: int,
    ) -> dict:
        """
        Look up a specific player's prediction for a given season.

        Parameters
        ----------
        player_id : int
            The player's ID from the player_features_v1.csv.
        year : int
            Season year to predict for.

        Returns
        -------
        dict with same structure as individual entries in ranked_players,
        plus 'actual_cpi' and 'actual_is_top_cpi' if the season is in the data.

        Raises
        ------
        AFLValidationError : player_id not found in year.
        """
        if not isinstance(player_id, int):
            raise AFLValidationError(
                f"player_id must be an integer, got {type(player_id).__name__}."
            )
        _validate_year(year)

        mask = (
            (self._player_df["player_id"] == player_id) &
            (self._player_df["year"] == year) &
            (self._player_df["feat_career_seasons"] > 0)
        )
        rows = self._player_df[mask]
        if rows.empty:
            avail_years = sorted(
                self._player_df[self._player_df["player_id"] == player_id]["year"].unique()
            )
            if not avail_years:
                raise AFLValidationError(
                    f"player_id {player_id} not found in the dataset."
                )
            raise AFLValidationError(
                f"player_id {player_id} has no eligible data for year {year}. "
                f"Available seasons: {avail_years}."
            )

        row  = rows.iloc[0]
        X    = row[self._ALL_FEATS].to_frame().T
        pred = float(self._pipeline.predict(X)[0])
        cpi_thresh = float(
            self._year_thresholds.get(year, self._year_thresholds.iloc[-1])
        )

        result = {
            "player_id":             player_id,
            "year":                  year,
            "team":                  row["team"],
            "position":              row["feat_position_proxy"],
            "prev_season_cpi":       round(float(row["feat_prev_cpi"]), 2)
                                     if pd.notna(row["feat_prev_cpi"]) else None,
            "career_seasons":        int(row["feat_career_seasons"]),
            "predicted_cpi":         round(pred, 2),
            "is_predicted_top_cpi":  bool(pred >= cpi_thresh),
        }
        # If the season is in the dataset, append actual outcomes
        if "target_cpi_raw" in row.index:
            result["actual_cpi"]        = round(float(row["target_cpi_raw"]), 2)
            result["actual_is_top_cpi"] = bool(row["target_is_top_cpi"] == 1)
            result["prediction_error"]  = round(abs(pred - float(row["target_cpi_raw"])), 2)

        return result


# ══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS (thin wrappers — ideal for LangChain tool binding)
# ══════════════════════════════════════════════════════════════════════════

_match_predictor: Optional[MatchWinnerPredictor] = None
_player_predictor: Optional[TopPlayerPredictor]  = None


def _get_match_predictor() -> MatchWinnerPredictor:
    global _match_predictor
    if _match_predictor is None:
        _match_predictor = MatchWinnerPredictor()
    return _match_predictor


def _get_player_predictor() -> TopPlayerPredictor:
    global _player_predictor
    if _player_predictor is None:
        _player_predictor = TopPlayerPredictor()
    return _player_predictor


def predict_match_winner(
    home_team: str,
    away_team: str,
    season_year: Optional[int] = None,
    is_finals: bool = False,
    round_num: int = 12,
    home_form_5: Optional[float] = None,
    home_ladder_pts: Optional[float] = None,
    home_rolling_margin: Optional[float] = None,
    home_h2h_win_5: Optional[float] = None,
    home_venue_win_10: Optional[float] = None,
    home_rest_days: Optional[float] = None,
) -> dict:
    """
    Predict the winner of an AFL match.

    Thin wrapper around MatchWinnerPredictor.predict() using a module-level
    singleton (loaded once, reused). Ideal for LangChain tool binding.

    Returns
    -------
    dict -- see MatchWinnerPredictor.predict() for full schema.

    Examples
    --------
    # Minimal call -- all features default to season medians
    predict_match_winner('Richmond Tigers', 'Collingwood Magpies')

    # With context
    predict_match_winner(
        home_team='Geelong Cats',
        away_team='Hawthorn Hawks',
        season_year=2024,
        home_form_5=0.8,
        home_ladder_pts=36,
        home_rolling_margin=12.5,
        home_h2h_win_5=0.6,
        home_venue_win_10=0.7,
    )
    """
    return _get_match_predictor().predict(
        home_team=home_team,
        away_team=away_team,
        season_year=season_year,
        is_finals=is_finals,
        round_num=round_num,
        home_form_5=home_form_5,
        home_ladder_pts=home_ladder_pts,
        home_rolling_margin=home_rolling_margin,
        home_h2h_win_5=home_h2h_win_5,
        home_venue_win_10=home_venue_win_10,
        home_rest_days=home_rest_days,
    )


def predict_top_players(
    team: str,
    year: int,
    stat_type: str = "cpi",
    top_k: int = 5,
    position_filter: Optional[str] = None,
) -> dict:
    """
    Predict and rank the top players for an AFL team in a given season.

    Thin wrapper around TopPlayerPredictor.predict() using a module-level
    singleton. Ideal for LangChain tool binding.

    Parameters
    ----------
    team        : str  -- team name or alias
    year        : int  -- season year (1984-2025)
    stat_type   : str  -- 'cpi' | 'disposal' | 'goal'  (default: 'cpi')
    top_k       : int  -- number of players to return (default: 5)
    position_filter : str | None -- 'Defender' | 'Forward' | 'Midfielder' | 'Ruck' | 'General'

    Returns
    -------
    dict -- see TopPlayerPredictor.predict() for full schema.

    Examples
    --------
    # Top 5 CPI players for Geelong in 2025
    predict_top_players('Geelong Cats', 2025)

    # Top 3 forwards by goal prediction
    predict_top_players('Brisbane Lions', 2024, stat_type='goal',
                        top_k=3, position_filter='Forward')
    """
    return _get_player_predictor().predict(
        team=team,
        year=year,
        stat_type=stat_type,
        top_k=top_k,
        position_filter=position_filter,
    )


# ══════════════════════════════════════════════════════════════════════════
# QUICK DEMO (run as script)
# ══════════════════════════════════════════════════════════════════════════

def _demo():
    print("=" * 60)
    print("  AFL Predictors — Quick Demo")
    print("=" * 60)

    # --- Match Winner ---
    print("\n[1] predict_match_winner()")
    result = predict_match_winner(
        home_team="Geelong Cats",
        away_team="Hawthorn Hawks",
        season_year=2024,
        home_form_5=0.8,
        home_ladder_pts=36,
        home_rolling_margin=12.5,
        home_h2h_win_5=0.6,
        home_venue_win_10=0.7,
    )
    print(f"  {result['home_team']} vs {result['away_team']}")
    print(f"  Predicted winner    : {result['predicted_winner']}")
    print(f"  P(home win)         : {result['home_win_probability']:.1%}")
    print(f"  Confidence          : {result['confidence']}")
    print(f"  Note                : {result['note']}")

    # --- Top Players ---
    print("\n[2] predict_top_players()")
    result2 = predict_top_players("Richmond Tigers", 2024, top_k=5)
    print(f"  Top {result2['top_k']} players for {result2['team']} ({result2['year']}):")
    for p in result2["ranked_players"]:
        top_flag = " [TOP CPI]" if p["is_predicted_top_cpi"] else ""
        print(f"    #{p['rank']}  ID={p['player_id']}  "
              f"Pos={p['position']:<12}  "
              f"Pred CPI={p['predicted_cpi']:>5.1f}  "
              f"Prev CPI={p['prev_season_cpi']:>5.1f}{top_flag}")

    # --- Input validation demo ---
    print("\n[3] Input validation errors:")
    tests = [
        ("Unknown team",    lambda: predict_match_winner("Fake FC", "Richmond Tigers")),
        ("Same team",       lambda: predict_match_winner("Geelong", "Geelong Cats")),
        ("Bad form value",  lambda: predict_match_winner("Geelong", "Richmond", home_form_5=1.5)),
        ("Bad year",        lambda: predict_top_players("Geelong Cats", 1900)),
        ("Bad stat type",   lambda: predict_top_players("Geelong Cats", 2024, stat_type="rebounds")),
        ("Bad position",    lambda: predict_top_players("Geelong Cats", 2024, position_filter="Striker")),
    ]
    for label, fn in tests:
        try:
            fn()
            print(f"  [{label}] -- No error raised (UNEXPECTED)")
        except AFLValidationError as e:
            print(f"  [{label}] AFLValidationError: {str(e)[:80]}...")
        except AFLModelNotLoaded as e:
            print(f"  [{label}] AFLModelNotLoaded: {str(e)[:80]}...")

    print("\n  Demo complete.")


if __name__ == "__main__":
    _demo()
