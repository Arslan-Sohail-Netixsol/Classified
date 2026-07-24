# Presentation Outline: Automated Ticket Triage Agent
**Target Audience:** Non-Technical Stakeholders & Product Managers
**Duration:** 5-7 Minutes

## Slide 1: The Problem We're Solving
- **Context:** Support teams spend 40% of their time reading, categorizing, and routing tickets manually.
- **Pain Point:** Humans are answering repetitive password resets while high-value billing issues sit in the queue.
- **Goal:** Automate the "front door" of customer support to reduce initial response times and free up human capital for complex problem-solving.

## Slide 2: The Solution Architecture
- **What it is:** A secure, AI-driven routing pipeline built on LangGraph.
- **How it works:**
  - **Classify:** AI reads the intent of the incoming ticket.
  - **Technical Fast-Track:** Instantly searches our Knowledge Base and replies to known bugs.
  - **Billing & Escalation:** Routes sensitive financial issues to a human queue.

## Slide 3: Why This Framework? (Security & Control)
- **Deterministic Control:** Unlike chatty, free-flowing AI agents, our pipeline uses a strict flowchart (LangGraph). A billing ticket *will* go to the billing department.
- **Human-in-the-Loop:** We built a hard "pause" mechanism. For sensitive actions like issuing refunds, the AI drafts the response, but it *cannot* send it until a human manager clicks "Approve."

## Slide 4: Evaluation & Performance
- **Success Rate:** 87.5% task success across 8 rigorous test scenarios.
- **Speed:** Sub-50ms automated response routing (simulated).
- **Graceful Failure:** When we simulated a database crash, the AI didn't break. It politely informed the user of an outage and alerted our human staff. 

## Slide 5: Known Limitations & Next Steps
- **The Vulnerability:** The agent was tricked by a clever "prompt injection" (a user telling the AI to ignore its rules and process a refund). 
- **The Safety Net:** Our Human-in-the-Loop checkpoint caught it and prevented actual damage, but it wasted human review time.
- **Next Steps:** 
  1. Add an "Input Guardrail" before the AI reads the ticket to filter out bad actors.
  2. Upgrade our simple database to a Vector Search engine for smarter technical troubleshooting.
  3. Build the UI dashboard for support managers to review paused tickets.

## Slide 6: Q&A
- *Open floor for questions regarding deployment timelines, cost, or security protocols.*
