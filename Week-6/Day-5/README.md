# AFL Assistant Pro — Week 6 Day 5 Capstone Deliverables

Welcome to the **AFL Assistant Pro** production package. This repository contains the complete set of deliverables for the Week 6 Day 5 Capstone Project: a production-grade, multi-agent sports analytics platform combining stateful LangGraph workflow orchestration, scikit-learn machine learning engines, FastAPI REST services, Streamlit UI, rigorous evaluation benchmarks (25+ test cases), a 2-page executive report (PDF), and a stakeholder presentation deck with interactive demo scripts.

---

## 📂 Deliverables Structure

All deliverables reside in `d:\netixsol\Week-6\Day-5`:

```text
Week-6/Day-5/
├── afl_assistant_core.py          # [Deliverable 1] LangGraph multi-agent DAG + ML Models + Guardrails
├── api.py                         # [Deliverable 1] FastAPI REST microservice (/chat, /health)
├── ui.py                          # [Deliverable 1] Streamlit chat user interface
├── combined_evaluation_results.md # [Deliverable 2] 25+ Case Evaluation Master Table (100% Pass Rate)
├── task2_evaluation.py            # [Deliverable 2] Automated 25+ Test Suite Runner & Benchmark
├── task2_evaluation_report.md     # [Deliverable 2] Detailed QA & Category Performance Breakdown
├── executive_report.pdf           # [Deliverable 3] 2-Page Executive Report (PDF)
├── generate_pdf_report.py         # [Deliverable 3] ReportLab PDF Generator (NumberedCanvas)
├── executive_summary.md           # [Deliverable 3] Markdown companion executive summary
├── monitoring_maintenance_plan.md # [Deliverable 3] 5-dimension production monitoring checklist
├── presentation_demo.md           # [Deliverable 4] 10-Slide Stakeholder Deck & 5 Interactive Demo Scripts
└── README.md                      # System documentation & execution guide
```

---

## 🚀 Quickstart & Execution Guide

### 1. Run Automated Evaluation Suite (25+ Scenarios)
Executes all 25 test cases (Factual, Scope Guardrails, Prediction Sanity, Multi-turn Coherence, ML vs. Naive Baseline) and outputs markdown reports:
```bash
python task2_evaluation.py
```

### 2. Generate 2-Page Executive PDF Report
Generates the publication-ready, 2-page executive report with dynamic canvas numbering:
```bash
python generate_pdf_report.py
```

### 3. Launch FastAPI Backend Service
Starts the REST microservice on `http://localhost:8000`:
```bash
python api.py
```
- Interactive API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`
- Chat Endpoint: `POST http://localhost:8000/chat`

### 4. Launch Interactive Streamlit UI
In a separate terminal, launch the web application:
```bash
streamlit run ui.py
```

---

## 🏉 Core System Architecture

```text
               User Query
                   │
                   ▼
            [ RouterNode ] ── (Intent & Entity Extraction)
                   │
    ┌──────────────┼──────────────┬──────────────┐
    ▼              ▼              ▼              ▼
[Prediction]  [Retrieval]   [DirectAnswer]  [ScopeNode]
    │              │              │              │
    └──────────────┼──────────────┴──────────────┘
                   ▼
           [ ValidationNode ]
            /      │       \
  (Pass)   /   (Clarify)    \ (Fallback)
          ▼        ▼           ▼
     [Formatter] [ClarifyNode] [FallbackNode]
          │        │           │
          └────────┴─────┬─────┘
                         ▼
                   Final Response
```

### Key Capabilities
1. **Dual-Tier Intent Routing:** Gemini LLM router backed by instantaneous regex heuristics ensuring 100% routing continuity under upstream rate limits.
2. **Predictive Statistical Grounding:** Calibrated `LogisticRegression` for match forecasting and `Ridge` regression for Composite Performance Index (CPI), disposals, and goals.
3. **Multi-Layer Guardrails:** Traps prompt injections, roleplay jailbreaks, and cross-sport inquiries.
4. **Resilient Self-Correction:** Detects unknown teams and invalid season years, prompting the user for targeted clarifications.

---

## 📊 Summary of Evaluation Results

- **Overall Pass Rate:** **100.0% (25/25 test cases)**
  - *Factual & Domain Knowledge:* 6/6 (100.0%)
  - *Scope & Guardrail Defenses:* 8/8 (100.0%)
  - *Prediction Sanity & Disclaimers:* 6/6 (100.0%)
  - *Multi-Turn Conversational Coherence:* 5/5 (100.0%)
- **Predictive ML Model vs. Naive Baseline:**
  - *ML Model Accuracy:* **100.0% (5/5 holdout fixtures)**
  - *Naive Baseline Accuracy:* **60.0% (3/5)**
  - *Finding:* The model correctly detects form differentials to forecast away upsets where naive heuristics fail.

---
*AFL Assistant Pro — Production Release 2026.*
