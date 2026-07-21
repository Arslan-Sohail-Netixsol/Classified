# Week 5 Day 2: LangChain Agent vs. Raw Python

This document summarizes the core differences between a handcrafted Python ReAct agent (Day 1) and a modern framework-driven agent using LangChain (Day 2).

---

## 1. Annotated Reasoning Trace (`verbose=True`)
*Because LangChain automates the prompt logic, its execution trace relies on the `create_agent` wrapper outputting structured tool payloads natively rather than explicit `[THOUGHT]` text blocks.*

```text
> Entering new AgentExecutor chain...

Invoking: `get_user_info` with `{'name': 'alice'}`
User Alice: Backend Developer in Engineering, Age 28

Invoking: `calculator` with `{'operation': 'add', 'a': 28, 'b': 15}`
43.0

Alice is 28 years old. Adding 15 to her age gives 43.

> Finished chain.
```

### Trace Annotations:
* **[ACT]**: `Invoking: 'get_user_info' with {'name': 'alice'}`
* **[OBSERVE]**: `User Alice: Backend Developer in Engineering, Age 28`
* **[ACT]**: `Invoking: 'calculator' with {'operation': 'add', 'a': 28, 'b': 15}`
* **[OBSERVE]**: `43.0`
* **[REASON / FINAL ANSWER]**: `Alice is 28 years old. Adding 15 to her age gives 43.`

---

## 2. Comparison: Raw-Python vs. LangChain

### 2.1 What Did LangChain Automate / Make Easier?
1. **Tool Definition (`@tool`)**: We abandoned manual JSON Draft-07 schemas. LangChain uses Python's `pydantic` and `inspect` under the hood to automatically infer schemas from type hints (`a: float`) and inject docstrings directly into the LLM’s system prompt to guide agent logic.
2. **State & The ReAct Loop**: We replaced manual `while` loops, iteration caps, and array appends with `create_agent()`. It natively triggers tool calls, executes the local python function, and routes the observations back into the context.
3. **Conversation Memory**: Manually managing prompt history lengths is tedious. By attaching a `MemorySaver()` checkpointer bound to a `thread_id`, the framework automatically snapshots the entire graph state at every turn, enabling multi-turn memory natively without re-triggering past tool calls.
4. **Structured Outputs**: By passing `response_format=AgentResponse` to the constructor, LangChain handles the complex prompt engineering or API parameters required to coerce the final output into a strict Pydantic model.

### 2.2 What is Hidden from You Now? (Abstraction "Magic")
1. **Loss of Intermediate Visibility**: LangChain abstracts raw LLM outputs behind compiled graphs (`CompiledStateGraph`). You no longer have a plain text list where you can print `[THOUGHT]` blocks to debug hallucination loops. You must rely heavily on tracing platforms like LangSmith to peek inside the black box.
2. **Exception Handling Overrides**: In raw Python, unhandled tool exceptions crash the script. LangChain overrides this: if a tool raises an exception, the framework catches it, converts the Python traceback into an `[OBSERVATION]` text string, and feeds it back to the LLM to autonomously try again. This shields your app from crashing, but can mask underlying bugs if not monitored.
3. **Under-the-Hood Chaining (LCEL)**: LangChain Expression Language overloads the `|` Python operator (`__or__`). While chaining `prompt | model | parser` is syntactically elegant, it obscures the complex async, streaming, and data conversion pipelines happening implicitly between components.
