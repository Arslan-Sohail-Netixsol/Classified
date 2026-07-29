# -*- coding: utf-8 -*-
"""
chat_cli.py
===========
Interactive Command Line Interface to chat with the AFL Analyst Agent.
Maintains conversational history, executes retrieval tools, and shows grounding metrics.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# Setup paths
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from afl_agent_tools import AFLToolRoutingAgent

class Colors:
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def main():
    print(Colors.CYAN + "=" * 65 + Colors.END)
    print(Colors.BOLD + "  🏉 Welcome to the AFL Chat Agent Interactive CLI 🏉" + Colors.END)
    print("  Type your queries below. The agent has memory of past turns.")
    print("  Type 'exit' or 'quit' to end the session.")
    print(Colors.CYAN + "=" * 65 + Colors.END)

    agent = AFLToolRoutingAgent()
    history = []

    while True:
        try:
            user_query = input("\n" + Colors.BOLD + "[You]: " + Colors.END).strip()
            if not user_query:
                continue
            if user_query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            print(Colors.BLUE + "  Thinking..." + Colors.END)
            
            # Execute loop
            res = agent.run(user_query, history=history)
            
            # Print routing & grounding details to the user for testing transparency
            print("-" * 50)
            if res["tool_called"]:
                print(Colors.YELLOW + f"  [Router Called Tool]: {res['tool_called']}" + Colors.END)
                print(f"  [Raw Tool Result]:\n{res['raw_tool_output']}")
            else:
                print(Colors.YELLOW + "  [Router Called Tool]: None (Conversational/Refusal)" + Colors.END)
                
            print(Colors.GREEN + f"  [Grounding Status]  : {res['grounding_check']['status']}" + Colors.END)
            if res["grounding_check"]["mismatched_stats"]:
                print(Colors.RED + f"  [Grounding Mismatch]: {res['grounding_check']['mismatched_stats']}" + Colors.END)
            print("-" * 50)
            
            print(Colors.BOLD + "\n[Agent]: " + Colors.END + res["final_response"])
            
            # Append turn to history
            history.append(res)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(Colors.RED + f"Error: {e}" + Colors.END)


if __name__ == "__main__":
    main()
