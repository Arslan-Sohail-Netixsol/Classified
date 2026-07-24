# Comparison Report: Sequential vs. Hierarchical vs. Single-Agent

**Task:** Competitor Intelligence & Marketing Strategy — researching Anthropic as an AI competitor
**Week 5, Day 4**

---

## Approaches Compared

| Approach | Framework | Agents | Process |
|---|---|---|---|
| **Single-Agent** | LangGraph (Day 3) | 1 generalist | Linear graph, no delegation |
| **Sequential Crew** | CrewAI | 3 specialists | Fixed order, context chaining |
| **Hierarchical Crew** | CrewAI | 3 specialists + 1 manager | Manager delegates and reviews |

---

## Token Usage & Cost

Pricing reference: GPT-4o blended @ ~$15.00 / 1M tokens

| System | Tokens | Est. Cost | Real Latency |
|---|---|---|---|
| LangGraph Single-Agent | ~1,800 | ~$0.027 | ~12s |
| CrewAI Sequential | ~3,700 | ~$0.056 | ~45s |
| CrewAI Hierarchical | ~5,100 | ~$0.077 | ~78s |

**Key insight:** Hierarchical adds a 38% token premium over sequential and a 183% premium over single-agent, primarily from manager review and revision passes. On this task the manager triggered 1 revision (caught a vague strategic implication), which measurably improved analysis quality.

---

## Quality Scoring — 3 Runs, Manual, 1-5 Scale

| Criterion | Single-Agent | Sequential R1 | Sequential R2 | Hierarchical |
|---|---|---|---|---|
| Factual Grounding | 3/5 | 4/5 | 4/5 | 5/5 |
| Completeness | 3/5 | 5/5 | 4/5 | 5/5 |
| Tone Consistency | 4/5 | 4/5 | 5/5 | 5/5 |
| **Average** | **3.3/5** | **4.3/5** | **4.3/5** | **5.0/5** |

**Why single-agent scored lower:** A generalist agent conflated research and writing phases — it started drafting before exhausting search depth, yielding shallower competitive coverage and more generic copy with no built-in revision mechanism.

---

## Sequential vs. Hierarchical — Side-by-Side

| Dimension | Process.sequential | Process.hierarchical |
|---|---|---|
| Control Flow | Fixed, pre-defined order in code | Manager dynamically delegates and reviews |
| Token Cost | Lower (~3,700) | Higher (~5,100, +38%) |
| Latency | Lower (~45s real) | Higher (~78s real) |
| Output Quality | Consistent; no revision loop | Higher ceiling; manager catches weak outputs |
| Reliability | High — no dynamic routing to fail | Slightly lower — manager LLM can mis-delegate |
| Debuggability | Easy — task order is explicit | Harder — manager decisions are emergent |
| Best for | Stable, repeatable workflows | Tasks needing QA or expert review loops |
| Avoid when | You need mid-chain error catching | Token budget is tight or task is simple |

---

## When to Use Each

```
Simple task (summarize, translate, classify)
  -> Single-Agent  --  cheapest, fastest, good enough

Multi-step task with clear phase boundaries
  -> CrewAI Sequential  --  best balance of quality, cost, and reliability

Multi-step task where quality is critical and errors are costly
  -> CrewAI Hierarchical  --  worth the premium when the manager catches real mistakes
```

---

## Verdict

For the competitor research and marketing brief task, the 3-agent sequential CrewAI crew delivered a meaningful quality uplift over a single LangGraph agent (avg 4.3 vs 3.3/5) at a 2x token cost, primarily because role specialization forced depth at each stage rather than rushing to an output. The hierarchical crew added a further quality improvement (5.0/5) by catching a vague strategic implication in the analysis — but at a 38% token cost premium over sequential, which is only justified if the manager consistently provides real corrections. For recurring automated workflows (e.g., weekly competitor monitoring), sequential is the right default: lower cost, predictable behavior, and easy to debug. The hierarchical process should be reserved for high-stakes one-off deliverables where output quality directly impacts a business decision.
