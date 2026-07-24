"""
crew_hierarchical.py
Week 5 Day 4 -- CrewAI Multi-Agent Crew (Process.hierarchical)

Scenario: Same competitor intelligence task as crew_sequential.py
  Manager Agent         -> delegates, reviews, and may request revisions
  Agent 1 (Researcher)  -> searches web for Anthropic competitive data
  Agent 2 (Analyst)     -> synthesizes findings into a SWOT summary
  Agent 3 (Copywriter)  -> drafts counter-positioning marketing brief

Usage:
  python crew_hierarchical.py
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
    allow_delegation: bool = False

@dataclass
class Task:
    description: str; expected_output: str
    agent: Agent = None
    context: list = field(default_factory=list)
    _output: str = field(default="", init=False, repr=False)

# --- Simulated outputs ---
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

ANALYSIS_V1 = """\
## Strategic Implications (v1 -- REJECTED by manager as too vague)
1. Win on integrations: Build Salesforce, Slack, Jira connectors.
2. Focus on quality: Provide better outputs than Anthropic.    <- VAGUE
3. SMB pricing: Offer transparent per-token pricing."""

ANALYSIS_V2 = """\
## Anthropic SWOT Analysis

| | Positive | Negative |
|---|---|---|
| Internal | Strengths: Safety-first brand; Claude 3 Opus benchmark leader; massive backing | Weaknesses: Limited multimodal; weak plugin ecosystem; low SMB awareness |
| External | Opportunities: Enterprise distrust of OpenAI; EU AI Act compliance; regulated industries | Threats: GPT-4o dominance; Google Gemini native integrations; open-source commoditization |

## Strategic Implications (v2 -- APPROVED by manager)
1. Win on integrations: Deep Salesforce, Slack, Jira connectors -- table-stakes features Anthropic lacks.
2. Target regulated industries: Pre-built HIPAA/SOX/GDPR compliance packs -- something CAI does not address.
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
* Built-in Compliance Packs -- Pre-configured HIPAA, SOX, and GDPR modules. Ready for
  regulated industries from day one, not a future roadmap item.
* Transparent Pricing for Every Team -- No opaque "contact sales" tiers.
  Know your costs before you commit. First 1M tokens on us.

CALL TO ACTION
Start your free 14-day trial at platform.ai/trial.
No credit card. Full feature access. First 1M tokens on us."""


def run_hierarchical():
    SEP  = "=" * 62
    DASH = "-" * 62

    print(SEP)
    print("  CrewAI -- Process.hierarchical")
    print("  Scenario: Competitor Intelligence & Marketing Strategy")
    print(SEP)

    # -- Manager Agent --------------------------------------------------
    manager = Agent(
        role="Strategic Intelligence Director",
        goal=(
            "Oversee the competitor research, analysis, and marketing brief. "
            "Review each agent's output and request revisions where vague or incomplete."
        ),
        backstory=(
            "VP of Strategy, 15 yrs leading cross-functional teams. "
            "Rejects vague outputs. Directs and reviews -- does not do the work."
        ),
        tools=[],
        allow_delegation=True,
    )

    # -- Worker Agents --------------------------------------------------
    search_tool     = SerperDevTool()
    file_write_tool = FileWriteTool()

    researcher = Agent(
        role="Market Research Analyst",
        goal="Gather factual intelligence on Anthropic: funding, products, positioning, weaknesses.",
        backstory="Ex-McKinsey consultant. 10 yrs competitive intelligence. Trusts only primary sources.",
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
        goal="Write a 300-word AIDA marketing brief and save to marketing_brief_hierarchical.txt.",
        backstory="Launched 3 AI products. Writes copy that wins enterprise deals.",
        tools=[file_write_tool],
    )

    # -- Tasks ----------------------------------------------------------
    research_task = Task(
        description=(
            "Search the web for Anthropic competitive data. "
            "Cover: Funding, Products, Positioning, Enterprise fit, Weaknesses."
        ),
        expected_output="5-section markdown report with 3-5 bullets per section.",
        agent=researcher,
    )

    analysis_task = Task(
        description=(
            "Using the research notes, produce a SWOT table and exactly 3 specific, "
            "actionable strategic implications referencing specific Anthropic weaknesses."
        ),
        expected_output="SWOT 2x2 table + 3 numbered, specific, actionable strategic implications.",
        agent=analyst,
        context=[research_task],
    )

    copy_task = Task(
        description=(
            "Using the SWOT and strategic implications, write a 300-word AIDA marketing brief. "
            "Save to marketing_brief_hierarchical.txt."
        ),
        expected_output="Structured brief: HEADLINE, HOOK, VALUE PROPOSITION (3 bullets), CALL TO ACTION.",
        agent=copywriter,
        context=[research_task, analysis_task],
    )

    tasks      = [research_task, analysis_task, copy_task]
    task_names = ["Research", "Analysis", "Copywriting"]
    agents     = [researcher, analyst, copywriter]

    # -- Manager-driven execution ---------------------------------------
    start         = time.time()
    total_tokens  = 0
    manager_tokens = 0

    print(f"\n  Manager : {manager.role}")
    print(f"  Workers : {' | '.join(a.role for a in agents)}\n")

    for i, (task, name, agent) in enumerate(zip(tasks, task_names, agents)):
        print(DASH)
        print(f"[Manager] -> Delegating Task {i+1} ({name}) to: {agent.role}")
        time.sleep(0.2)

        # First pass
        if name == "Analysis":
            first_pass  = ANALYSIS_V1
            task_tokens = 800
        elif name == "Research":
            first_pass  = RESEARCH_OUTPUT
            task_tokens = 1400
        else:
            first_pass  = COPY_OUTPUT
            task_tokens = 1200

        total_tokens   += task_tokens
        manager_tokens += 300  # review overhead per task

        preview = "\n  ".join(first_pass.strip().splitlines()[:4])
        print(f"\n  [{agent.role}] Pass 1:\n  {preview}\n  ...")

        # Manager requests revision on Analysis task
        if name == "Analysis":
            print(f'\n  [Manager] REJECT: Implication #2 is too vague ("focus on quality").')
            print(f"  [Manager] Request revision: must reference a specific Anthropic gap.")
            time.sleep(0.2)

            revision       = ANALYSIS_V2
            total_tokens   += 600
            manager_tokens += 200

            preview2 = "\n  ".join(revision.strip().splitlines()[:5])
            print(f"\n  [{agent.role}] Pass 2 (revised):\n  {preview2}\n  ...")
            print(f"  [Manager] APPROVED.")
            task._output = revision
        else:
            print(f"  [Manager] APPROVED.")
            task._output = first_pass

        # FileWriteTool on copy task
        if name == "Copywriting":
            with open("marketing_brief_hierarchical.txt", "w") as f:
                f.write(task._output)
            print("  [FileWriteTool] Saved -> marketing_brief_hierarchical.txt")

    elapsed     = time.time() - start
    grand_total = total_tokens + manager_tokens
    cost        = grand_total / 1_000_000 * 15.00

    print(f"\n{SEP}")
    print("  HIERARCHICAL RUN COMPLETE")
    print(f"  Worker tokens    : ~{total_tokens:,}")
    print(f"  Manager overhead : ~{manager_tokens:,} tokens ({manager_tokens/grand_total*100:.0f}%)")
    print(f"  Total tokens     : ~{grand_total:,}")
    print(f"  Est. cost        : ~${cost:.4f} (GPT-4o blended pricing)")
    print(f"  Wall time        : ~{elapsed:.1f}s (mock) / ~78s (real LLM)")
    print(f"  Manager revisions: 1  (Analysis -- caught vague implication)")
    print(SEP)

    return {"tokens": grand_total, "cost": cost}


if __name__ == "__main__":
    metrics = run_hierarchical()
    print(f"\nScript complete. Check marketing_brief_hierarchical.txt for output.")
