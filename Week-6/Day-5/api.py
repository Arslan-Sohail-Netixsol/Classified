# -*- coding: utf-8 -*-
"""
api.py
======
Week 6 Day 5 — Task 3: Wrap as an API

FastAPI application wrapping the LangGraph E2E Pipeline.
Accepts POST /chat with a conversation_id and user_message.
Returns a JSON response containing the assistant's reply and metadata.
"""

import time
import uuid
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

from afl_assistant_core import E2EPipeline

# Re-use the JSON logger from afl_assistant_core
logger = logging.getLogger("afl_assistant")

app = FastAPI(title="AFL Assistant API", version="1.0.0")

# Initialize the pipeline globally (disable LLM router for extreme speed during automated checks, 
# but normally we'd want it True. We use True here for the real API).
pipeline = E2EPipeline(router_version=2, use_llm_router=True)

# In-memory session store for conversation history (for demo purposes)
# In production, this would be Redis or Postgres.
SESSION_STORE: Dict[str, List[Any]] = {}

class ChatRequest(BaseModel):
    user_message: str
    conversation_id: str = None

class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    intent: str
    latency_ms: float
    error_class: str = None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    start_time = time.time()
    
    cid = req.conversation_id
    if not cid:
        cid = str(uuid.uuid4())
    
    # Retrieve history
    history = SESSION_STORE.get(cid, [])
    
    logger.info("Received query", extra={"extra_data": {"conversation_id": cid, "query": req.user_message}})
    
    try:
        # Run pipeline
        state = pipeline.run(query=req.user_message, history=history)
        
        reply = state.get("final_response") or "Sorry, I could not process that request."
        
        # Append to history store (LangChain message objects)
        from langchain_core.messages import HumanMessage, AIMessage
        history.append(HumanMessage(content=req.user_message))
        history.append(AIMessage(content=reply[:1000]))
        SESSION_STORE[cid] = history
        
        latency = (time.time() - start_time) * 1000
        
        logger.info("Completed query", extra={"extra_data": {
            "conversation_id": cid, 
            "intent": state.get("detected_intent"),
            "latency_ms": latency
        }})
        
        return ChatResponse(
            conversation_id=cid,
            reply=reply,
            intent=str(state.get("detected_intent")),
            latency_ms=latency,
            error_class=str(state.get("error_class"))
        )
        
    except Exception as e:
        logger.error("API Error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
