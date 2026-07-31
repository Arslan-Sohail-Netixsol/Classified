# -*- coding: utf-8 -*-
"""
ui.py
=====
Week 6 Day 5 — Task 3: Simple Chat UI

Streamlit application providing a chat interface for the AFL LangGraph Assistant.
Communicates with api.py via HTTP POST requests.
"""

import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="AFL Assistant Pro", page_icon="🏉", layout="centered")

st.title("🏉 AFL AI Assistant")
st.markdown("Ask me about **Match Predictions**, **Historical H2H Records**, **Player Stats**, or **AFL Rules**!")

# Initialize session state for conversation ID and chat history
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("E.g., Will the Pies beat the Cats this weekend?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                # Call the FastAPI backend
                resp = requests.post(
                    API_URL,
                    json={"user_message": prompt, "conversation_id": st.session_state.session_id},
                    timeout=30.0
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    reply = data.get("reply", "")
                    
                    # Optional: display some debug metadata neatly
                    intent = data.get("intent", "Unknown")
                    latency = data.get("latency_ms", 0)
                    
                    st.markdown(reply)
                    
                    # Add a small caption for API transparency
                    st.caption(f"✓ Routed as `{intent}` in {latency:.0f}ms")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                else:
                    err = f"API Error: {resp.status_code} - {resp.text}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                    
            except requests.exceptions.ConnectionError:
                err = "⚠️ Cannot connect to API backend. Ensure `python api.py` is running on port 8000."
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as e:
                err = f"⚠️ Unexpected error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
