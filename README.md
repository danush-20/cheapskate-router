# Cheapskate Router

## Overview

Cheapskate Router is a Docker-based batch inference agent that processes tasks across eight
categories — factual Q&A, math reasoning, sentiment analysis, summarization, named entity
recognition (NER), code debugging, logic puzzles, and code generation — using Fireworks AI
models via an OpenAI-compatible API. Submissions are scored in two stages: first an accuracy
gate (pass/fail), then ranked by **token efficiency** (total tokens consumed) among all
passing submissions. Every design decision in this agent is made with that two-stage rubric
in mind: accuracy first, then ruthless token frugality.

---

## Architecture

```
raw prompt
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  classify_category()  — local, zero tokens          │
│  scoring over CATEGORY_SIGNALS patterns             │
│  logs: category, score, margin (low margin = warn)  │
└──────────────────────┬──────────────────────────────┘
                       │ category
                       ▼
┌─────────────────────────────────────────────────────┐
│  compress_prompt()  — local, zero tokens            │
│  strips boilerplate, collapses whitespace           │
│  logs: original vs compressed token estimate        │
└──────────────────────┬──────────────────────────────┘
                       │ compressed prompt
                       ▼
┌─────────────────────────────────────────────────────┐
│  resolve_model(category)                            │
│  picks model from CATEGORY_MODEL_MAP                │
│  falls back to allowed_models[0] if not available   │
└──────────────────────┬──────────────────────────────┘
                       │ model id
                       ▼
┌─────────────────────────────────────────────────────┐
│  call_model()                                       │
│  system prompt = SYSTEM_PROMPT + CATEGORY_INSTRUCTIONS│
│  max_tokens = CATEGORY_MAX_TOKENS[category]         │
│  reasoning_effort = CATEGORY_REASONING_MAP[category]│
└──────────────────────┬──────────────────────────────┘
                       │ answer
                       ▼
┌─────────────────────────────────────────────────────┐
│  _maybe_retry()  — local validity check             │
│  sentiment: label regex  │  ner: entity/type pattern│
│  math: number present    │  logic: proper noun match│
│  code: ast.parse()                                  │
│  if check fails → single retry with stricter prompt │
└──────────────────────┬──────────────────────────────┘
                       │ final answer
                       ▼
              {task_id, answer}  →  results.json
```

---

## Category → Model Routing

| Category        | Model                  | Rationale |
|-----------------|------------------------|-----------|
| factual         | gemma-4-26b-a4b-it     | One-sentence answers need no heavy reasoning; the 26B MoE variant is fast and cheap. |
| sentiment       | gemma-4-26b-a4b-it     | Single-label output; a lightweight model is more than sufficient. |
| ner             | gemma-4-26b-a4b-it     | Entity extraction is pattern-following, not reasoning; 26B handles it well. |
| summarization   | gemma-4-26b-a4b-it     | Compression task, not generation; 26B produces fluent one-sentence summaries. |
| math            | gemma-4-31b-it         | Multi-step arithmetic benefits from the denser 31B dense model with medium reasoning. |
| logic           | gemma-4-31b-it         | Deductive chains need stronger reasoning; 31B + medium effort reduces errors. |
| code_debugging  | kimi-k2p7-code         | Code-specialised model; significantly fewer syntax errors than general models. |
| code_generation | kimi-k2p7-code         | Same rationale — code-tuned weights produce valid, idiomatic Python. |

---

## Token Efficiency Techniques

- **Prompt compression (`compress_prompt`)** — strips leading boilerplate phrases ("Could you
  please", "I would like you to", etc.) and collapses redundant whitespace before the prompt
  reaches the API. Numbers, code blocks, and quoted strings are never touched. Saves input
  tokens on every call at zero cost.

- **Per-category token budgets (`CATEGORY_MAX_TOKENS`)** — `max_tokens` is set per category
  rather than using a single large ceiling. Sentiment gets 10 tokens (one label), factual gets
  80 (one sentence), code gets 800. This directly caps verbose over-generation on categories
  that don't need it.

- **Tightened system prompts (`CATEGORY_INSTRUCTIONS`)** — each category carries an explicit
  brevity instruction ("Answer in one sentence", "State the final answer first, then at most
  2 lines of justification"). This steers the model away from preamble and elaboration,
  reducing completion tokens without sacrificing correctness.

- **Local validity checks + single retry (`_maybe_retry`)** — after generation, a cheap
  regex/AST check validates the answer structure. If it fails, one retry is issued with a
  stricter format instruction. The retry costs tokens only when the first answer was wrong,
  which is a net win: a structurally invalid answer that passes the accuracy gate is worth
  the retry cost; a valid answer never triggers a retry.

---

## Design Philosophy

This agent treats every task as a small resource-allocation problem: before a single token is
sent to the API, the task has already been assigned a category (local scoring, zero cost), a
model tier (cheap for simple tasks, stronger for reasoning), a compressed prompt (fewer input
tokens), and a token ceiling (fewer output tokens). After generation, a local structural check
decides whether the allocated budget was well spent or needs a single top-up retry — analogous
to a budget game where you sow a fixed stake across a decision and reclaim based on outcome.
The goal is to pass the accuracy gate with the minimum total token spend, not to maximise
answer quality in isolation.

---

## What Was Intentionally Left Out

- **Learned/adaptive routing** — there is no reward signal available mid-run, so a bandit or
  RL-based router would have nothing to learn from. Left out deliberately.
- **Multi-model ensembling or LLM-based self-judging** — calling two models and picking the
  better answer, or asking a judge model to score outputs, would multiply token spend with no
  guaranteed accuracy payoff under this scoring rubric. Left out deliberately.
- **Task decomposition into sub-calls** — breaking a task into smaller LLM calls costs tokens
  directly against the efficiency ranking. Left out deliberately.

These are engineering trade-offs given the actual scoring rubric, not limitations of the
approach.

---

## How to Run Locally

### Prerequisites

```bash
pip install -r requirements.txt
```

### Environment variables

| Variable            | Description |
|---------------------|-------------|
| `FIREWORKS_API_KEY` | Your Fireworks AI API key |
| `FIREWORKS_BASE_URL`| Fireworks OpenAI-compatible base URL (e.g. `https://api.fireworks.ai/inference/v1`) |
| `ALLOWED_MODELS`    | Comma-separated list of fully-qualified Fireworks model IDs |
| `INPUT_PATH`        | Path to input JSON array (default: `/input/tasks.json`) |
| `OUTPUT_PATH`       | Path to write output JSON array (default: `/output/results.json`) |

### Run the agent

```bash
export FIREWORKS_API_KEY=<your_key>
export FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
export ALLOWED_MODELS="accounts/fireworks/models/gemma-4-26b-a4b-it,accounts/fireworks/models/gemma-4-31b-it,accounts/fireworks/models/kimi-k2p7-code"
export INPUT_PATH=test_input/tasks.json
export OUTPUT_PATH=test_output/results.json

python main.py
```

### Run the classifier and compression tests (no API key needed)

```bash
python test_classifier.py
```

---

## Docker

### Build

```bash
docker build -t cheapskate-router .
```

### Run

```bash
docker run --rm \
  -e FIREWORKS_API_KEY=<your_key> \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS="accounts/fireworks/models/gemma-4-26b-a4b-it,accounts/fireworks/models/gemma-4-31b-it,accounts/fireworks/models/kimi-k2p7-code" \
  -v "$(pwd)/test_input:/input:ro" \
  -v "$(pwd)/test_output:/output" \
  cheapskate-router
```

Results are written to `test_output/results.json`.  
A token usage summary is written to `test_output/run_stats.json` (never part of `results.json`).
