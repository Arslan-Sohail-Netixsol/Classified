import sys
from afl_day3_all_tasks import AFLToolRoutingAgent

def start_chat():
    print("\n" + "="*50)
    print("  AFL Domain-Scoped Chat Agent")
    print("  Type 'exit' or 'quit' to end the conversation.")
    print("="*50 + "\n")
    
    agent = AFLToolRoutingAgent()
    history = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            # The agent.run method prints its internal routing steps automatically
            # which is great for seeing how it thinks!
            response = agent.run(user_input, history=history)
            
            # Append to history for multi-turn memory
            history.append(response)
            
            print(f"\n🤖 AFL Bot: {response['final_response']}")
            
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    start_chat()
