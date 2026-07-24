# Week 5 Day 5 Capstone: System Design

**Use Case:** Support Ticket Triage Agent
**Scenario:** An enterprise support pipeline that categorizes incoming tickets, automatically attempts to draft solutions for technical issues by searching a knowledge base, routes billing issues to a human for compliance, and intercepts outgoing automated responses for a final human review before dispatch.

---

## Architecture Diagram

```mermaid
graph TD
    classDef state fill:#f9f,stroke:#333,stroke-width:2px;
    classDef node fill:#bbf,stroke:#333,stroke-width:2px;
    classDef human fill:#fbb,stroke:#333,stroke-width:2px;
    
    START((START)) --> A
    
    A[Classify Ticket Node]:::node --> |Technical| B[Technical Solver Node\\nTool: KB Search]:::node
    A --> |Billing| C[Billing Handler Node]:::node
    A --> |General| D[General Responder Node]:::node
    
    B --> E[Human Review Checkpoint]:::human
    C --> E
    D --> E
    
    E --> |Approved| F[Dispatch Response Node]:::node
    E --> |Rejected/Revise| A
    
    F --> END((END))
```

---

## Framework Choice & Justification

**Framework Chosen:** LangGraph

A support ticket triage system requires strict, deterministic routing based on ticket category, robust state persistence for asynchronous human-in-the-loop approvals (e.g., for billing authorizations or final draft reviews), and cyclic edges for self-correction if a draft is rejected. LangGraph natively excels at explicit control flow and checkpointer-based persistence, which are critical for a secure and predictable enterprise ticketing pipeline. Relying solely on a framework like CrewAI, which uses emergent, conversational routing among agents, would be too unpredictable and hard to control for strict operational SLAs.
