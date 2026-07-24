"""
Week 5 Day 5 Capstone: End-to-End Support Ticket Triage System
Using LangGraph with checkpointer, failure handling, and a local tool.
"""

import json
import uuid
from typing import TypedDict
from langgraph.graph import StateGraph, START, END  # type: ignore
from langgraph.checkpoint.memory import MemorySaver  # type: ignore

# 1. State Schema


class TicketState(TypedDict):
    ticket_id: str
    user_input: str
    category: str        # "Technical", "Billing", "General", "Invalid", "Refusal"
    draft_response: str
    needs_human: bool
    status: str          # "Pending", "Approved", "Rejected", "Error"
    error_message: str

# 2. Tool Definition (External Data Source)


def search_kb_tool(query: str) -> str:
    """Simulates searching a local JSON database."""
    if "timeout" in query.lower():
        raise TimeoutError("KB search timed out after 5000ms.")
    if "corrupt" in query.lower():
        raise FileNotFoundError(
            "knowledge_base.json not found or inaccessible.")

    try:
        with open("knowledge_base.json", "r") as f:
            kb = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Database error: {e}")

    # Simple keyword match
    for key, value in kb.items():
        # key might be "password_reset". Check if any part of the key is in the query.
        for word in key.split("_"):
            if word in query.lower():
                return value
    return "No relevant KB article found."

# 3. Nodes


def classify_node(state: TicketState):
    """
    Acts as the LLM router. Validates input and categorizes the ticket.
    Simulates 'model refusal' and 'bad input' gracefully.
    """
    text = state.get("user_input", "").strip().lower()

    # Failure Handling 1: Bad Input
    if not text or len(text) < 5:
        return {"category": "Invalid", "error_message": "Ticket text too short or empty."}

    # Failure Handling 2: Model Refusal
    if "hack" in text or "bypass" in text:
        return {"category": "Refusal", "error_message": "I cannot fulfill this request (Model Refusal)."}

    # Classification
    if "refund" in text or "charge" in text or "bill" in text:
        return {"category": "Billing"}
    elif "api" in text or "password" in text or "2fa" in text or "timeout" in text or "corrupt" in text:
        return {"category": "Technical"}
    else:
        return {"category": "General"}


def technical_node(state: TicketState):
    """Drafts technical response using external tool."""
    query = state["user_input"]

    # Failure Handling 3: Tool Error/Timeout
    try:
        kb_result = search_kb_tool(query)
        draft = f"Technical Support: {kb_result}"
        status = "Pending"
    except Exception:
        # Graceful fallback instead of crashing
        draft = "Our technical knowledge base is currently offline. A human agent has been alerted."
        status = "Error"

    return {"draft_response": draft, "status": status, "needs_human": False}


def billing_node(state: TicketState):
    """Drafts billing response and triggers human-in-the-loop."""
    draft = "Billing Support: We have received your refund/charge request and it is queued for processing."
    # Require human approval for billing actions
    return {"draft_response": draft, "needs_human": True, "status": "Pending"}


def general_node(state: TicketState):
    draft = "General Support: Thank you for reaching out. A representative will review your message."
    return {"draft_response": draft, "needs_human": False, "status": "Pending"}


def failure_handler_node(state: TicketState):
    """Handles Invalid and Refusal categories."""
    return {"status": "Rejected", "draft_response": f"System Message: {state.get('error_message')}"}


# 4. Graph Construction
builder = StateGraph(TicketState)

builder.add_node("classify", classify_node)
builder.add_node("technical", technical_node)
builder.add_node("billing", billing_node)
builder.add_node("general", general_node)
builder.add_node("failure", failure_handler_node)

# Conditional router function


def route_ticket(state: TicketState):
    cat = state.get("category")
    if cat == "Technical":
        return "technical"
    elif cat == "Billing":
        return "billing"
    elif cat == "Invalid" or cat == "Refusal":
        return "failure"
    else:
        return "general"


builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_ticket)

# All paths go to END
builder.add_edge("technical", END)
builder.add_edge("billing", END)
builder.add_edge("general", END)
builder.add_edge("failure", END)

# Add memory for Human-in-the-Loop checkpoints
memory = MemorySaver()
# The interrupt_before pauses the graph right before the designated node.
# Since we want to pause *after* billing node drafts the response but *before* we say it's complete,
# in a real system we'd have a 'dispatch' node. Let's add a dispatch node to intercept all completed drafts.


def dispatch_node(state: TicketState):
    if state.get("status") == "Pending" and not state.get("needs_human"):
        return {"status": "Dispatched"}
    elif state.get("needs_human"):
        # If it reaches here after human approval, human would have changed status to Approved
        pass
    return {}


builder.add_node("dispatch", dispatch_node)
builder.add_edge("technical", "dispatch")
builder.add_edge("billing", "dispatch")
builder.add_edge("general", "dispatch")
builder.add_edge("failure", "dispatch")
builder.add_edge("dispatch", END)

# Intercept at dispatch IF human is needed (we can't dynamically interrupt, so we interrupt dispatch universally
# but we can just use interrupt_before=["dispatch"])
# Actually, a better pattern: dynamic interrupt isn't natively supported in basic interrupt_before without conditional nodes.
# Let's route to human_review node conditionally.

builder = StateGraph(TicketState)
builder.add_node("classify", classify_node)
builder.add_node("technical", technical_node)
builder.add_node("billing", billing_node)
builder.add_node("general", general_node)
builder.add_node("failure", failure_handler_node)


def dispatch_final(state: TicketState):
    # This node actually "sends" the message
    return {"status": "Dispatched"}


builder.add_node("dispatch", dispatch_final)
# No-op node just to act as checkpoint
builder.add_node("human_review", lambda x: x)


def route_to_dispatch_or_human(state: TicketState):
    if state.get("needs_human"):
        return "human_review"
    return "dispatch"


builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_ticket)

# Handlers route to dispatch or human check
builder.add_conditional_edges("technical", route_to_dispatch_or_human)
builder.add_conditional_edges("billing", route_to_dispatch_or_human)
builder.add_conditional_edges("general", route_to_dispatch_or_human)

# Failure skips dispatch
builder.add_edge("failure", END)

builder.add_edge("human_review", "dispatch")
builder.add_edge("dispatch", END)

# Compile with interrupt
triage_app = builder.compile(
    checkpointer=memory, interrupt_before=["human_review"])


# ==========================================
# 5. Testing & Execution
# ==========================================

def run_test(scenario_name: str, user_input: str):
    print(f"\n{'=' * 50}")
    print(f"Scenario: {scenario_name}")
    print(f"Input: '{user_input}'")
    print(f"{'-' * 50}")

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Run graph
    result = triage_app.invoke(
        {"user_input": user_input, "needs_human": False},
        config=config
    )

    print(f"Category: {result.get('category')}")
    print(f"Draft Response: {result.get('draft_response')}")
    print(f"Status: {result.get('status')}")

    # Check if interrupted for human-in-the-loop
    next_state = triage_app.get_state(config)
    if len(next_state.next) > 0 and next_state.next[0] == "human_review":
        print("\n>> [PAUSED] GRAPH PAUSED FOR HUMAN REVIEW [PAUSED] <<")
        print(">> Human manager approving the billing request...")

        # Human provides input/approval (updating state)
        triage_app.update_state(config, {"status": "Approved by Human"})

        # Resume graph
        print(">> Resuming graph...")
        final_result = triage_app.invoke(None, config=config)
        print(f"Final Status: {final_result.get('status')}")


if __name__ == "__main__":
    print("Initializing Support Ticket Triage System...")

    # Test 1: Normal Technical Ticket (KB Match)
    run_test("Technical Query", "How do I reset my password?")

    # Test 2: Billing Ticket (Triggers Human-in-the-Loop)
    run_test("Billing Request", "I need a refund for last month's charge.")

    # Test 3: Failure - Bad Input Validation
    run_test("Bad Input", "hi")

    # Test 4: Failure - Model Refusal
    run_test("Model Refusal", "How do I hack the billing database?")

    # Test 5: Failure - Tool Timeout/Error Graceful Handling
    run_test("Tool Failure", "Getting an api timeout error.")
