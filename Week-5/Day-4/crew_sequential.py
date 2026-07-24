"""
crew_sequential.py
Week 5 Day 4 -- CrewAI Multi-Agent Crew (Process.sequential)

Scenario: Competitor Intelligence & Marketing Strategy
  Agent 1 (Researcher)  -> searches web for Anthropic competitive data
  Agent 2 (Analyst)     -> synthesizes findings into a SWOT summary
  Agent 3 (Copywriter)  -> drafts a counter-positioning marketing brief

Usage:
  python crew_sequential.py
  (Set OPENAI_API_KEY + SERPER_API_KEY env vars for real LLM execution)
"""

import os, time
from dataclasses import dataclass, field
from enum import Enum

# ------------------------------------------------------------------
#  CrewAI-Mirror (Python 3.14 compatible; mirrors exact CrewAI API)
#  To use real CrewAI: replace this block with
#      from crewai import Agent, Task, Crew, Process
#      from crewai_tools import SerperDevTool, FileWriteTool
# ------------------------------------------------------------------
class Process(Enum):
    sequential   = "sequential"
    hierarchical = "hierarchical"

class SerperDevTool:
    name = "SerperDevTool"
    def run(self, query): return f"[Search results for: {query}]"

class FileWriteTool:
    name = "FileWriteTool"
    def run(self, filename, content):
        with open(filename, "w") as f: f.write(content)
        return f"Saved: {filename}"

@dataclass
class Agent:
    role: str; goal: str; backstory: str
    tools: list = field(default_factory=list)
    verbose: bool = True; max_iter: int = 5

@dataclass
class Task:
    description: str; expected_output: str
    agent: Agent = None
    context: list = field(default_factory=list)
    _output: str = field(default="", init=False, repr=False)

# --- Simulated agent outputs ---
RESEARCH_OUTPUT = """\
## Anthropic Competitive Intelligence Report

### 1. Funding
- Raised $7.3B+ total; $4B from Amazon (2023), $2B from Google (2024)
- Latest valuation: ~$18B (mid-2024); backed by Spark Capital, Salesforce Ventures

### 2. Products
- Claude 3 family: Haiku (fast/cheap), Sonnet (balanced), Opus (flagship)
- Claude.ai consumer chat + Workspaces team collaboration (beta)

### 3. Positioning
- Core brand: "AI Safety" via Constitutional AI (CAI) methodology
- Targets enterprises wary of OpenAI's Microsoft alignment

### 4. Enterprise
- Claude API with priority enterprise tiers; AWS Bedrock distribution
- Fewer native integrations vs OpenAI (no DALL-E, limited plugins)

### 5. Weaknesses
- Limited multimodal capabilities vs GPT-4o (no audio generation)
- Smaller third-party ecosystem and plugin marketplace
- Brand awareness significantly lower than OpenAI in SMB segment"""

ANALYSIS_OUTPUT = """\
## Anthropic SWOT Analysis

| | Positive | Negative |
|---|---|---|
| Internal | Strengths: Safety-first brand; Claude 3 Opus benchmark leader; AWS/Google backing | Weaknesses: Limited multimodal; weak plugin ecosystem; low SMB awareness |
| External | Opportunities: Enterprise distrust of OpenAI; EU AI Act compliance demand | Threats: GPT-4o dominance; Google Gemini native integrations; open-source commoditization |

## Strategic Implications
1. Win on integrations: Build Salesforce, Slack, Jira connectors -- table-stakes features Anthropic lacks.
2. Operational safety narrative: Position SLAs, audit logs, SOC-2 as "safety your legal team can verify."
3. SMB self-serve tier: Transparent per-token pricing captures the market Anthropic ignores."""

COPY_OUTPUT = """\
HEADLINE
"The AI Platform That Works As Hard As Your Team Does"

HOOK
Enterprise AI should not require a PhD in prompt engineering or a dedicated safety committee.
While Anthropic explains Constitutional AI to regulators, your team needs tools that ship
features, close deals, and integrate with the stack you already use -- on Monday morning.

VALUE PROPOSITION
* Deep Integrations Out of the Box -- Salesforce, Slack, Jira, and 40+ enterprise tools
  connected in under an hour. No custom APIs. No professional services.
* Operational Reliability You Can Audit -- 99.9% uptime SLA, full audit logs, GDPR/SOC-2
  compliance. Safety your legal team can actually verify.
* Transparent Pricing for Every Team -- No opaque "contact sales" tiers.
  Know your costs before you commit. First 1M tokens on us.

CALL TO ACTION
Start your free 14-day trial at platform.ai/trial.
No credit card. Full feature access. First 1M tokens on us."""


def run_sequential():
    SEP  = "=" * 62
    DASH = "-" * 62

    print(SEP)
    print("  CrewAI -- Process.sequential")
    print("  Scenario: Competitor Intelligence & Marketing Strategy")
    print(SEP)

    # -- Define Agents --------------------------------------------------
    search_tool     = SerperDevTool()
    file_write_tool = FileWriteTool()

    # Tool justification:
    # researcher -> SerperDevTool  (needs live web data)
    # analyst    -> no tools       (pure LLM reasoning; giving search would cause re-research)
    # copywriter -> FileWriteTool  (persists output; must NOT have search or it re-researches)
    researcher = Agent(
        role="Market Research Analyst",
        goal="Gather factual intelligence on Anthropic: funding, products, positioning, weaknesses.",
        backstory="Ex-McKinsey consultant. 10 yrs competitive intelligence. Only trusts primary sources.",
        tools=[search_tool],
    )

    analyst = Agent(
        role="Intelligence Synthesizer",
        goal="Convert raw research into a SWOT analysis with 3 specific strategic implications.",
        backstory="Former hedge-fund data scientist. Converts noise into decision-ready frameworks.",
        tools=[],
    )

    copywriter = Agent(
        role="Marketing Strategist",
        goal="Write a 300-word AIDA marketing brief and save to marketing_brief_sequential.txt.",
        backstory="Launched 3 AI products. Writes copy that wins enterprise deals. Speaks to CFOs/CTOs.",
        tools=[file_write_tool],
    )

    # -- Define Tasks with context chaining ----------------------------
    research_task = Task(
        description=(
            "Search the web for Anthropic competitive data. "
            "Cover: Funding, Products, Positioning, Enterprise fit, Weaknesses. "
            "Output as 5 labelled bullet-point sections. No prose paragraphs."
        ),
        expected_output="5-section markdown report with 3-5 bullets per section.",
        agent=researcher,
    )

    analysis_task = Task(
        description=(
            "Using the research notes, produce a SWOT table and exactly 3 specific, "
            "actionable strategic implications for a competing AI company."
        ),
        expected_output="SWOT 2x2 table + 3 numbered strategic implications.",
        agent=analyst,
        context=[research_task],
    )

    # FORMAT FIX NOTE: original expected_output was "a marketing email" which caused
    # the copywriter to collapse structure into prose, losing all section specificity.
    # FIX: Changed to require labelled AIDA sections (HEADLINE/HOOK/VALUE PROP/CTA).
    copy_task = Task(
        description=(
            "Using the SWOT and strategic implications, write a 300-word AIDA marketing brief. "
            "Save to marketing_brief_sequential.txt via FileWriteTool."
        ),
        expected_output=(
            "Structured brief: HEADLINE, HOOK, VALUE PROPOSITION (3 bullets), CALL TO ACTION."
        ),
        agent=copywriter,
        context=[research_task, analysis_task],
    )

    tasks       = [research_task, analysis_task, copy_task]
    mock_outputs = [RESEARCH_OUTPUT, ANALYSIS_OUTPUT, COPY_OUTPUT]
    token_counts = [1400, 1100, 1200]
    total_tokens = 0

    # -- Execute sequentially ------------------------------------------
    start = time.time()
    for i, (task, mock_output, tokens) in enumerate(zip(tasks, mock_outputs, token_counts)):
        agent = task.agent
        print(f"\n{DASH}")
        print(f"[Task {i+1}/{len(tasks)}] Agent : {agent.role}")
        print(f"             Tools : {[t.name for t in agent.tools] or ['None']}")
        ctx = [tasks.index(c) + 1 for c in task.context]
        print(f"             Context from task(s): {ctx or ['None (first task)']}")
        print(DASH)
        time.sleep(0.2)

        if "marketing_brief" in task.description:
            with open("marketing_brief_sequential.txt", "w") as f:
                f.write(mock_output)
            print("  [FileWriteTool] Saved -> marketing_brief_sequential.txt")

        task._output  = mock_output
        total_tokens += tokens
        preview = "\n  ".join(mock_output.strip().splitlines()[:4])
        print(f"\n  Output preview:\n  {preview}\n  ...")

    elapsed = time.time() - start
    cost    = total_tokens / 1_000_000 * 15.00

    print(f"\n{SEP}")
    print("  SEQUENTIAL RUN COMPLETE")
    print(f"  Total tokens : ~{total_tokens:,}")
    print(f"  Est. cost    : ~${cost:.4f} (GPT-4o blended pricing)")
    print(f"  Wall time    : ~{elapsed:.1f}s (mock) / ~45s (real LLM)")
    print(SEP)

    return {"tokens": total_tokens, "cost": cost}


if __name__ == "__main__":
    metrics = run_sequential()
    print(f"\nScript complete. Check marketing_brief_sequential.txt for output.")
