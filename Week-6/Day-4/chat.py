import sys
from pathlib import Path

# Add necessary paths
_HERE = Path(__file__).parent
_DAY2 = _HERE.parent / "Day-2"
_DAY3 = _HERE.parent / "Day-3"

sys.path.insert(0, str(_DAY2))
sys.path.insert(0, str(_DAY3))

from afl_langgraph_app import E2EPipeline

def chat_loop():
    print("\n" + "=" * 70)
    print("  AFL LangGraph Agent Interactive Chat")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 70 + "\n")
    
    # Initialize pipeline with LLM router enabled
    pipeline = E2EPipeline(router_version=2, use_llm_router=True)
    history = []
    
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ['quit', 'exit']:
                break
            if not query.strip():
                continue
                
            print("\nThinking...")
            # Run the query through the full LangGraph pipeline
            state = pipeline.run(query=query, history=history)
            
            # The pipeline appends the new human message and AI response to the state history
            if "conversation_history" in state:
                history = state["conversation_history"]
                
            response = state.get("final_response") or state.get("clarification_msg") or state.get("fallback_msg")
            
            if response:
                print("\nAgent:\n" + response)
            else:
                print("\nAgent: [No response generated]")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    chat_loop()
