# -*- coding: utf-8 -*-
"""
retrieval_layer.py
==================
Week 6 Day 3 Task 2 — Structured and Unstructured Retrieval Layer

Justification for Split:
-------------------------
AFL statistics, scores, records, and predictions are exact numerical operations.
Using fuzzy semantic vector search (like retrieving a paragraph of matches) to answer 
"what was Player X's exact disposal average in 2024?" is prone to hallucination, 
era-drift confusion, and numerical error. Thus, all match and player statistics 
are retrieved via structured pandas lookup tools. 
Conversely, general league rules (e.g. holding the ball), team histories (e.g. Richmond 
premierships), and conceptual terms (e.g. what is CPI?) are unstructured texts. 
These are retrieved semantically via TF-IDF cosine similarity search over our local knowledge base.
"""

from __future__ import annotations
import os
import sys
import warnings
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# --- Path setup ---
_HERE = Path(__file__).parent
_DAY1 = _HERE.parent / "Day-1"
_DAY2 = _HERE.parent / "Day-2"

# CSV files
_MATCH_CSV = _DAY1 / "match_features_v1.csv"
_PLAYER_CSV = _DAY1 / "player_features_v1.csv"
_KB_TXT = _HERE / "afl_knowledge_base.txt"


# ══════════════════════════════════════════════════════════════════════════
# 1. STRUCTURED RETRIEVAL TOOLS
# ══════════════════════════════════════════════════════════════════════════

class StructuredRetrievalEngine:
    """Handles exact pandas lookups for match and player datasets."""
    def __init__(self):
        self.match_df = pd.read_csv(_MATCH_CSV) if _MATCH_CSV.exists() else None
        self.player_df = pd.read_csv(_PLAYER_CSV) if _PLAYER_CSV.exists() else None
        
        # Clean up column strings
        if self.player_df is not None:
            self.player_df["team"] = self.player_df["team"].str.strip()
        if self.match_df is not None:
            self.match_df["home_team"] = self.match_df["home_team"].str.strip()
            self.match_df["away_team"] = self.match_df["away_team"].str.strip()

    def get_team_h2h_record(self, team_a: str, team_b: str, start_year: int = 1983, end_year: int = 2025) -> dict:
        """
        Structured tool: Get head-to-head match stats between two teams.
        
        Parameters
        ----------
        team_a : str
            First team name (e.g., 'Richmond Tigers', 'Richmond').
        team_b : str
            Second team name.
        """
        if self.match_df is None:
            return {"error": "Match database not found."}

        # Resolve canonical team names
        sys.path.insert(0, str(_DAY2))
        import predict
        try:
            canon_a = predict._normalise_team(team_a)
            canon_b = predict._normalise_team(team_b)
        except Exception as e:
            return {"error": str(e)}

        # Filter matches where these two played each other
        df_sub = self.match_df[(self.match_df["year"] >= start_year) & (self.match_df["year"] <= end_year)]
        
        mask_h_a = (df_sub["home_team"] == canon_a) & (df_sub["away_team"] == canon_b)
        mask_a_h = (df_sub["home_team"] == canon_b) & (df_sub["away_team"] == canon_a)
        
        matches = df_sub[mask_h_a | mask_a_h].copy()
        
        if matches.empty:
            return {
                "team_a": canon_a,
                "team_b": canon_b,
                "message": f"No historical head-to-head matches recorded between {canon_a} and {canon_b} from {start_year} to {end_year}."
            }

        # Calculate wins
        a_wins = 0
        b_wins = 0
        draws = 0
        margins_a = []
        margins_b = []
        
        recent_meetings = []
        
        # Sort matches by date to get recent ones
        matches = matches.sort_values("match_date", ascending=False)
        
        for _, row in matches.iterrows():
            yr = int(row["year"])
            rnd = row["round"]
            home = row["home_team"]
            away = row["away_team"]
            h_margin = int(row["home_margin"])
            target_win = int(row["target_win"])
            
            # Identify winner
            if h_margin == 0:
                draws += 1
                winner = "Draw"
            elif (home == canon_a and target_win == 1) or (away == canon_a and target_win == 0):
                a_wins += 1
                winner = canon_a
                margins_a.append(abs(h_margin))
            else:
                b_wins += 1
                winner = canon_b
                margins_b.append(abs(h_margin))
                
            if len(recent_meetings) < 5:
                recent_meetings.append({
                    "date": row["match_date"],
                    "round": rnd,
                    "venue": row["venue"],
                    "scoreline": f"{home} vs {away} (Margin: {abs(h_margin)} pts)",
                    "winner": winner
                })

        total_games = a_wins + b_wins + draws
        avg_margin_a = float(np.mean(margins_a)) if margins_a else 0.0
        avg_margin_b = float(np.mean(margins_b)) if margins_b else 0.0

        return {
            "team_a": canon_a,
            "team_b": canon_b,
            "period": f"{start_year}-{end_year}",
            "total_games": total_games,
            "wins_team_a": a_wins,
            "wins_team_b": b_wins,
            "draws": draws,
            "win_rate_team_a": round(a_wins / total_games, 3) if total_games else 0.0,
            "avg_win_margin_team_a": round(avg_margin_a, 1),
            "avg_win_margin_team_b": round(avg_margin_b, 1),
            "recent_meetings": recent_meetings
        }

    def get_player_season_stats(self, player_id: int, year: int) -> dict:
        """
        Structured tool: Get season statistics for a specific player ID.
        """
        if self.player_df is None:
            return {"error": "Player database not found."}

        # Filter by player ID and year
        mask = (self.player_df["player_id"] == player_id) & (self.player_df["year"] == year)
        rows = self.player_df[mask]
        
        if rows.empty:
            return {
                "player_id": player_id,
                "year": year,
                "error": f"No statistics found for player ID {player_id} in the {year} season."
            }
            
        row = rows.iloc[0]
        
        # Build dictionary response
        stats = {
            "player_id":             player_id,
            "year":                  year,
            "team":                  row["team"],
            "position":              row["feat_position_proxy"],
            "games_played":          int(row["games_played"]) if pd.notna(row["games_played"]) else 0,
            "prior_season_cpi":      round(float(row["feat_prev_cpi"]), 2) if pd.notna(row["feat_prev_cpi"]) else None,
            "prior_avg_disposals":   round(float(row["feat_prev_disposals"]), 2) if pd.notna(row["feat_prev_disposals"]) else None,
            "prior_avg_goals":       round(float(row["feat_prev_goals"]), 2) if pd.notna(row["feat_prev_goals"]) else None,
            "career_seasons":        int(row["feat_career_seasons"]) if pd.notna(row["feat_career_seasons"]) else 0,
            "predicted_is_top_cpi":  bool(row["target_is_top_cpi"] == 1) if "target_is_top_cpi" in row.index else False,
        }
        
        # Add targets if they exist in dataset
        if "target_cpi_raw" in row.index:
            stats["actual_cpi_raw"] = round(float(row["target_cpi_raw"]), 2)
        if "target_is_top_disposal" in row.index:
            stats["is_top_disposal_performer"] = bool(row["target_is_top_disposal"] == 1)
        if "target_is_top_goal" in row.index:
            stats["is_top_goalkicker_performer"] = bool(row["target_is_top_goal"] == 1)
            
        return stats


# ══════════════════════════════════════════════════════════════════════════
# 2. UNSTRUCTURED / SEMANTIC RETRIEVAL TOOL (TF-IDF Vector Store)
# ══════════════════════════════════════════════════════════════════════════

class SemanticRetrievalEngine:
    """Implements local Cosine Similarity TF-IDF search over unstructured text paragraphs."""
    def __init__(self):
        self.corpus: List[str] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        
        # Load corpus
        if _KB_TXT.exists():
            with open(_KB_TXT, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Split text by sections/paragraphs (two newlines)
            paragraphs = content.split("\n\n")
            # Filter empty or header paragraphs
            self.corpus = [p.strip() for p in paragraphs if p.strip() and not p.strip().startswith("# ")]
            
            if self.corpus:
                self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
                
    def retrieve_afl_knowledge(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Unstructured tool: Semantic search over AFL rules, terms, and team trivia.
        """
        if not self.corpus or self.tfidf_matrix is None:
            return [{"error": "Knowledge base corpus is empty or not loaded."}]
            
        # Convert query to TF-IDF vector
        query_vec = self.vectorizer.transform([query])
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top matching paragraph indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            # Return paragraph if score exceeds threshold
            if score > 0.05:
                results.append({
                    "score": round(score, 4),
                    "text": self.corpus[idx]
                })
                
        if not results:
            return [{"message": "No relevant historical rules or trivia found in the knowledge base."}]
            
        return results


# ══════════════════════════════════════════════════════════════════════════
# 3. INTERFACE FUNCTIONS FOR TOOLS
# ══════════════════════════════════════════════════════════════════════════

_str_retriever = StructuredRetrievalEngine()
_sem_retriever = SemanticRetrievalEngine()


def get_team_h2h_record(team_a: str, team_b: str, start_year: int = 1983, end_year: int = 2025) -> str:
    """
    Structured lookup tool: Get head-to-head records between two teams.
    Returns a formatted human-readable summary.
    """
    res = _str_retriever.get_team_h2h_record(team_a, team_b, start_year, end_year)
    if "error" in res:
        return f"Error: {res['error']}"
    if "message" in res:
        return res["message"]
        
    lines = [
        f"Head-to-Head Record: {res['team_a']} vs {res['team_b']} ({res['period']})",
        f"Total Games Played: {res['total_games']}",
        f"  - {res['team_a']} Wins: {res['wins_team_a']} (Win Rate: {res['win_rate_team_a']:.1%})",
        f"  - {res['team_b']} Wins: {res['wins_team_b']}",
        f"  - Draws: {res['draws']}",
        f"Average margins:",
        f"  - {res['team_a']} winning margin: {res['avg_win_margin_team_a']} pts",
        f"  - {res['team_b']} winning margin: {res['avg_win_margin_team_b']} pts",
        "\nRecent Meetings:"
    ]
    for meet in res["recent_meetings"]:
        lines.append(f"  - {meet['date']} (Rnd {meet['round']}): {meet['scoreline']} -> Winner: {meet['winner']}")
        
    return "\n".join(lines)


def get_player_season_stats(player_id: int, year: int) -> str:
    """
    Structured lookup tool: Get season stats for a specific player ID.
    Returns a formatted human-readable summary.
    """
    res = _str_retriever.get_player_season_stats(player_id, year)
    if "error" in res:
        return f"Error: {res['error']}"
        
    lines = [
        f"Grounded Stats for Player ID {res['player_id']} in Season {res['year']}:",
        f"  - Team: {res['team']}",
        f"  - Position: {res['position']}",
        f"  - Games Played: {res['games_played']}",
        f"  - Career Seasons (Prior): {res['career_seasons']}",
        f"  - Prior CPI score: {res['prior_season_cpi']}",
        f"  - Prior Avg Disposals: {res['prior_avg_disposals']}",
        f"  - Prior Avg Goals: {res['prior_avg_goals']}"
    ]
    if "actual_cpi_raw" in res:
        lines.append(f"  - Actual CPI Score: {res['actual_cpi_raw']}")
    if "is_top_disposal_performer" in res:
        lines.append(f"  - Top Disposal Performer (>90th percentile): {res['is_top_disposal_performer']}")
    if "is_top_goalkicker_performer" in res:
        lines.append(f"  - Top Goalkicker Performer (>95th percentile): {res['is_top_goalkicker_performer']}")
        
    return "\n".join(lines)


def retrieve_afl_knowledge(query: str) -> str:
    """
    Unstructured semantic retrieval tool: Search AFL rules and trivia knowledge base.
    Returns a formatted list of top matching paragraphs.
    """
    res = _sem_retriever.retrieve_afl_knowledge(query)
    if not res:
        return "No matching rules or history found in the knowledge base."
        
    if "error" in res[0]:
        return f"Error: {res[0]['error']}"
    if "message" in res[0]:
        return res[0]["message"]
        
    lines = [f"Semantic search results for query: '{query}' (Similarity scores threshold > 0.05)"]
    for i, r in enumerate(res, start=1):
        lines.append(f"\nMatch #{i} (Score: {r['score']}):\n{r['text']}")
        
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# DEMO EXECUTION
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  AFL Retrieval Layer Demo")
    print("=" * 60)
    
    # 1. H2H Structured Lookup
    print("\n[1] get_team_h2h_record('Richmond', 'Collingwood')")
    print(get_team_h2h_record("Richmond", "Collingwood"))
    
    # 2. Player Stats Structured Lookup
    print("\n[2] get_player_season_stats(43266, 2025)")
    print(get_player_season_stats(43266, 2025))
    
    # 3. Semantic Retrieval over Rules and Trivia
    print("\n[3] retrieve_afl_knowledge('Explain holding the ball rules')")
    print(retrieve_afl_knowledge("Explain holding the ball rules"))
    
    print("\n[4] retrieve_afl_knowledge('What is Kardinia Park venue fortress?')")
    print(retrieve_afl_knowledge("What is Kardinia Park venue fortress?"))
