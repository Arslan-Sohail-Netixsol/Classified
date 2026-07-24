"""
Week 5 Day 5 Capstone: API Wrapper for Triage Agent
"""

import time
import uuid
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from triage_system import triage_app

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("agent_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("triage_agent")

app = FastAPI(title="Support Ticket Triage API", version="1.0.0")

class TicketRequest(BaseModel):
    user_input: str

class TicketResponse(BaseModel):
    ticket_id: str
    category: str
    draft_response: str
    needs_human: bool
    status: str
    latency_ms: float

@app.post("/triage", response_model=TicketResponse)
async def process_ticket(request: TicketRequest):
    start_time = time.time()
    ticket_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": ticket_id}}
    
    logger.info(f"New Ticket Received | ID: {ticket_id} | Input: '{request.user_input}'")
    
    try:
        # Execute the LangGraph app
        result = triage_app.invoke(
            {"user_input": request.user_input, "needs_human": False},
            config=config
        )
        
        # Check if the graph paused for a human checkpoint
        next_state = triage_app.get_state(config)
        is_paused = len(next_state.next) > 0 and next_state.next[0] == "human_review"
        
        status = result.get("status", "Unknown")
        if is_paused:
            status = "Pending_Human_Approval"
            
        latency = (time.time() - start_time) * 1000
        
        # Simulated Token Logging (since we used a deterministic model)
        mock_tokens = len(request.user_input.split()) * 2 + 30 
        
        logger.info(
            f"Ticket Processed | ID: {ticket_id} | Category: {result.get('category')} | "
            f"Latency: {latency:.2f}ms | Est Tokens: {mock_tokens} | "
            f"Needs Human: {is_paused} | Status: {status}"
        )
        
        return TicketResponse(
            ticket_id=ticket_id,
            category=result.get("category", "Unknown"),
            draft_response=result.get("draft_response", ""),
            needs_human=is_paused,
            status=status,
            latency_ms=latency
        )
        
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        logger.error(f"Agent Execution Error | ID: {ticket_id} | Error: {str(e)} | Latency: {latency:.2f}ms")
        raise HTTPException(status_code=500, detail="Internal Agent Error")

if __name__ == "__main__":
    import uvicorn
    # uvicorn api:app --host 0.0.0.0 --port 8000
    print("Run with: uvicorn api:app --reload")
