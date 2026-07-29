# -*- coding: utf-8 -*-
"""
afl_agent_tools.py
==================
Week 6 Day 3 Tasks 3 & 4 — LangChain Agent, Tool Integration & Conversation Memory

Implements:
1. LangChain registered tools with schemas.
2. ReAct Agent Routing and Execution Loop with Conversation Memory.
3. Grounding validator that parses numbers in the final response and 
   cross-checks them against the raw tool output.
4. Automated conversational demo showcasing context carrying across 5 turns.
"""

from __future__ import annotations
import os
import sys
import json
import warnings
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

warnings.filterwarnings("ignore")

# --- Path setup ---
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from afl_agent import get_agent
from retrieval_layer import get_team_h2h_record, get_player_season_stats, retrieve_afl_knowledge


# ══════════════════════════════════════════════════════════════════════════
# 1. REGISTER LANGCHAIN TOOLS
# ══════════════════════════════════════════════════════════════════════════

@tool
def get_team_h2h_record_tool(team_a: str, team_b: str, start_year: int = 1983, end_year: int = 2025) -> str:
    """
    Lookup historical head-to-head match statistics between two AFL teams.
    Useful for finding win-loss ratios, draw counts, average margins, and recent meeting scores.
    """
    return get_team_h2h_record(team_a, team_b, start_year, end_year)


@tool
def get_player_season_stats_tool(player_id: int, year: int) -> str:
    """
    Lookup detailed season statistics for a specific player ID in a given year.
    Useful for finding a player's team, position, games played, prior CPI, average disposals, average goals, and top performer status.
    """
    return get_player_season_stats(int(player_id), int(year))


@tool
def retrieve_afl_knowledge_tool(query: str) -> str:
    """
    Semantically search the AFL knowledge base for rules, terms, definitions, and team histories.
    Useful for answering conceptual questions like 'holding the ball rules' or 'scoring a behind'.
    """
    return retrieve_afl_knowledge(query)


# Map tools by name
TOOL_MAP = {
    "get_team_h2h_record_tool": get_team_h2h_record_tool,
    "get_player_season_stats_tool": get_player_season_stats_tool,
    "retrieve_afl_knowledge_tool": retrieve_afl_knowledge_tool
}


# ══════════════════════════════════════════════════════════════════════════
# 2. RECONSTRUCT GROUNDING VALIDATOR
# ══════════════════════════════════════════════════════════════════════════

def verify_grounding(tool_output: str, final_response: str) -> dict:
    """
    Statically cross-references all numbers found in the final response against the tool output.
    Returns validation status and lists of matching or mismatched numbers.
    """
    # Extract numbers (integers and floats)
    num_pattern = re.compile(r'\b\d+(?:\.\d+)?\b')
    
    resp_nums = set(num_pattern.findall(final_response))
    tool_nums = set(num_pattern.findall(tool_output))
    
    mismatches = []
    matches = []
    
    for num in resp_nums:
        # Ignore isolated layout digits
        if len(num) == 1 and num in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            continue
        # Ignore years
        if num.isdigit() and 1983 <= int(num) <= 2026:
            continue
            
        if num in tool_nums:
            matches.append(num)
        else:
            # Check for close float equivalence or percentage mappings
            val = float(num)
            found = False
            for t_num in tool_nums:
                try:
                    if abs(float(t_num) - val) < 1e-4:
                        found = True
                        matches.append(num)
                        break
                    if abs(float(t_num)*100 - val) < 1e-2 or abs(float(t_num)/100 - val) < 1e-2:
                        found = True
                        matches.append(num)
                        break
                except ValueError:
                    continue
            if not found:
                mismatches.append(num)
                
    status = "VERIFIED_GROUNDED" if not mismatches else "HALLUCINATION_WARNING"
    
    return {
        "status": status,
        "matched_stats": sorted(matches),
        "mismatched_stats": sorted(mismatches)
    }


# ══════════════════════════════════════════════════════════════════════════
# 3. AGENT ROUTING LOOP WITH MULTI-TURN MEMORY
# ══════════════════════════════════════════════════════════════════════════

class AFLToolRoutingAgent:
    """Uses LLM to decide tool calls, executes them, and formats final responses with memory."""
    def __init__(self):
        self.agent = get_agent()
        self.llm = self.agent.llm
        
    def run(self, user_query: str, history: List[Dict[str, Any]] = None) -> dict:
        """
        Runs the full tool-routing-execution loop.
        """
        print(f"\nUser Query: '{user_query}'")
        
        # Format conversational history for prompt injection
        history_text = ""
        if history:
            history_text = "\nConversational History:\n"
            for turn in history:
                h_call = f" (Tool called: {turn['tool_called']})" if turn.get("tool_called") else ""
                history_text += f"[User]: {turn['query']}\n"
                history_text += f"[Agent]{h_call}: {turn['final_response']}\n"
            history_text += "\n"

        # --- Step 1: LLM Tool Routing Decision ---
        routing_prompt = f"""You are a tool-router for an AFL agent. Your job is to select the correct tool to answer the user's query.
Available tools:
1. `get_team_h2h_record_tool`: Takes parameters: team_a (str), team_b (str). Use for head-to-head match histories or scorelines between two teams.
2. `get_player_season_stats_tool`: Takes parameters: player_id (int), year (int). Use for exact seasonal statistics of a player.
3. `retrieve_afl_knowledge_tool`: Takes parameter: query (str). Use for general rules, behinds, scoring, venue fortress, or Richmond/Geelong history trivia.

Guidelines:
- Read the Conversational History to resolve pronouns or follow-up references. For example, if the user asks "How does that compare to his prior season cpi?" or "Which team did he play for?", look at the history to find the player ID and team discussed previously and query for that same player.
- Respond EXACTLY in the format:
  TOOL: <tool_name> | ARGS: {{"arg_name": "val", ...}}
- If no tool is needed, respond EXACTLY with:
  TOOL: None

{history_text}
Current User Query: {user_query}
"""
        routing_response = self.llm.invoke([SystemMessage(content=routing_prompt)]).content.strip()
        print(f"  Routing decision: '{routing_response}'")
        
        # --- Step 2: Execute Tool if required ---
        tool_output = ""
        tool_name = None
        
        match = re.search(r'TOOL:\s*(\w+)\s*\|\s*ARGS:\s*(\{.*\})', routing_response)
        if match:
            tool_name = match.group(1).strip()
            args_str = match.group(2).strip()
            try:
                args = json.loads(args_str)
                if tool_name in TOOL_MAP:
                    print(f"  Executing {tool_name} with arguments: {args}")
                    tool_output = TOOL_MAP[tool_name].invoke(args)
                else:
                    tool_output = f"Error: Tool {tool_name} is not registered."
            except Exception as e:
                tool_output = f"Error executing tool {tool_name}: {e}"
        else:
            print("  No tool call requested by router.")
            tool_output = "No tool result available. Answer directly based on history if possible."

        # --- Step 3: Generate Final Answer ---
        final_prompt = f"""You are 'AFL Analyst Bot'. Answer the user's question.
If a tool was executed, you must base all numerical statistics, records, and facts strictly on the provided tool result.
Do not make up or hallucinate any numbers. If a statistic is not in the tool result, state that you don't have records for it.
Utilize the Conversational History to keep track of context, entities, and pronouns.

{history_text}
Current User Query: {user_query}
Tool Output: {tool_output}
"""
        final_response = self.llm.invoke([SystemMessage(content=final_prompt)]).content.strip()
        print(f"  Final Response:\n{final_response}")
        
        # --- Step 4: Grounding Verification ---
        grounding_result = verify_grounding(tool_output, final_response)
        print(f"  Grounding Verification: {grounding_result['status']}")
        if grounding_result["mismatched_stats"]:
            # If no tool call was executed (TOOL: None), then stats should align with prior history
            # In that case, we can pass verification if history contains the numbers
            history_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', history_text))
            still_mismatched = [n for n in grounding_result["mismatched_stats"] if n not in history_nums]
            if not still_mismatched:
                grounding_result["status"] = "VERIFIED_GROUNDED_VIA_HISTORY"
                print(f"    Verified Grounded Stats via history: {grounding_result['matched_stats']}")
            else:
                print(f"    Warning! Hallucinated/unreferenced numbers: {still_mismatched}")
        else:
            print(f"    Verified Grounded Stats: {grounding_result['matched_stats']}")
            
        return {
            "query": user_query,
            "tool_called": tool_name,
            "raw_tool_output": tool_output,
            "final_response": final_response,
            "grounding_check": grounding_result
        }


# ══════════════════════════════════════════════════════════════════════════
# DEMO EXECUTION
# ══════════════════════════════════════════════════════════════════════════

def run_agent_demos():
    agent = AFLToolRoutingAgent()
    
    # Task 3: Standard queries
    print("=" * 60)
    print("  AFL Tool Routing Agent — Task 3 Demos")
    print("=" * 60)
    questions = [
        "How did Richmond perform against Collingwood recently?",
        "What were the stats of player 43269 in 2024?",
        "Explain the rule for holding the ball in AFL."
    ]
    results_t3 = []
    for q in questions:
        res = agent.run(q)
        results_t3.append(res)
        
    report_path = _HERE / "grounding_validation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Grounding & Tool Validation Report — Week 6 Day 3\n\n")
        f.write("This report logs the evaluation of the LangChain AFL agent's tool-calling behavior and grounding validation results.\n\n")
        f.write("## Test Executions\n\n")
        for idx, r in enumerate(results_t3, start=1):
            f.write(f"### Test {idx}: {r['query']}\n\n")
            f.write(f"* **Tool Called:** `{r['tool_called']}`\n")
            f.write(f"* **Grounding Status:** `{r['grounding_check']['status']}`\n")
            f.write(f"* **Matched Stats:** `{r['grounding_check']['matched_stats']}`\n")
            if r['grounding_check']['mismatched_stats']:
                f.write(f"* **Mismatched/Hallucinated Stats:** `{r['grounding_check']['mismatched_stats']}`\n")
            f.write("\n#### Raw Tool Output\n```text\n")
            f.write(r["raw_tool_output"] + "\n```\n")
            f.write("\n#### Final Agent Response\n")
            f.write(r["final_response"] + "\n\n")
            f.write("---\n\n")
    print(f"Grounding validation report written to: {report_path}")

    # Task 4: Multi-Turn Conversation Memory Test
    print("\n" + "=" * 60)
    print("  AFL Conversational Agent — Task 4 Memory Demos")
    print("=" * 60)
    
    multi_turn_questions = [
        # Turn 1: Specific player lookup
        "What were the stats of player 43269 in 2024?",
        # Turn 2: Follow-up comparative query using pronoun reference "his"
        "How does that compare to his prior season cpi?",
        # Turn 3: Context reference "he" to find team
        "Which team did he play for?",
        # Turn 4: Context reference "they" to query H2H against Richmond in 2024
        "How did they perform against Richmond in 2024?",
        # Turn 5: Temporal follow-up "the year after that"
        "What about the year after that?"
    ]
    
    history = []
    for turn_idx, q in enumerate(multi_turn_questions, start=1):
        print(f"\n[Turn {turn_idx}]")
        res = agent.run(q, history=history)
        history.append(res)
        
    # Write memory evaluation log
    memory_report_path = _HERE / "conversation_memory_report.md"
    with open(memory_report_path, "w", encoding="utf-8") as f:
        f.write("# Conversation Memory Report — Week 6 Day 3\n\n")
        f.write("This report documents the verification of the multi-turn conversational memory capabilities of the AFL agent.\n\n")
        f.write("## 5-Turn Conversational Transcript\n\n")
        for idx, r in enumerate(history, start=1):
            f.write(f"### Turn {idx}: {r['query']}\n\n")
            f.write(f"* **Tool Selected by Router:** `{r['tool_called']}`\n")
            f.write(f"* **Grounding Check Status:** `{r['grounding_check']['status']}`\n")
            f.write(f"* **Matched Numbers:** `{r['grounding_check']['matched_stats']}`\n")
            f.write("\n#### Raw Tool Response\n```text\n")
            f.write(r["raw_tool_output"] + "\n```\n")
            f.write("\n#### Agent Answer\n")
            f.write(r["final_response"] + "\n\n")
            f.write("---\n\n")
            
    print(f"Conversation memory report written to: {memory_report_path}")


if __name__ == "__main__":
    run_agent_demos()
