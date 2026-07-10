# Cheapskate Router

A containerized AI agent built for **Track 1 (Hybrid Token-Efficient Routing Agent)** of the AMD Developer Hackathon: ACT II. It handles eight task categories using Fireworks AI models, routing each task to the cheapest model that can still answer it accurately — minimizing token spend while clearing the accuracy gate.

## What it does

The agent reads a batch of tasks spanning eight capability categories:

1. Factual knowledge (Q&A)
2. Mathematical reasoning
3. Sentiment classification
4. Text summarization
5. Named entity recognition (NER)
6. Code debugging
7. Logical / deductive reasoning
8. Code generation

For each task, it:
1. **Classifies the category** using a zero-token, regex-based classifier (no model call needed).
2. **Routes to the cheapest suitable Fireworks model** for that category, with automatic fallback if a preferred model isn't available.
3. **Calls the model** with `reasoning_effort="low"` and a brevity-tuned system prompt to minimize token usage without sacrificing correctness.
4. **Locally validates code answers** (code debugging / code generation) using Python's built-in `ast` parser — a free, zero-token syntax check that triggers a single automatic retry only if the generated code is malformed.
5. **Writes results** to the required output schema, with every task wrapped in error handling so a single failure never crashes the full run.

## Architecture

```
main.py            # Entry point: classification, routing, model calls, validation
Dockerfile          # Container definition
requirements.txt    # Python dependencies
.github/workflows/  # CI: builds and pushes the image to GHCR on push to main
```

### Routing logic

Tasks are mapped to models by category. If a mapped model isn't present in `ALLOWED_MODELS` at runtime, the agent automatically falls back to the first available allowed model — so the same code works whether all 5 allowed models are accessible or only a subset (e.g. only the serverless-tier models).

### Token efficiency measures

- **Zero-token category classification** (pure regex, no API call)
- **`reasoning_effort="low"`** on every Fireworks call to suppress unnecessary hidden reasoning tokens
- **Brevity-tuned system prompt** — no markdown headers, no comparison tables, no restating the question, no unrequested alternative solutions — while explicitly preserving completeness (e.g. NER must return *all* entities, including date/time expressions) and code validity
- **Local, zero-token code validation** via `ast.parse()` before accepting a code answer, avoiding wasted retries on obviously broken output

## Setup and usage

### Environment variables

The container expects these to be injected at runtime (the harness provides them during evaluation):

| Variable | Description |
|---|---|
| `FIREWORKS_API_KEY` | Fireworks API key |
| `FIREWORKS_BASE_URL` | Fireworks API base URL |
| `ALLOWED_MODELS` | Comma-separated list of permitted Fireworks model IDs |

For local development, create a `.env` file (never committed — see `.gitignore`):

```env
FIREWORKS_API_KEY=your_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code
INPUT_PATH=test_input/tasks.json
OUTPUT_PATH=test_output/results.json
```

### Running locally (without Docker)

```bash
pip install -r requirements.txt
python main.py
```

Reads tasks from `test_input/tasks.json` and writes results to `test_output/results.json` (paths controlled by `INPUT_PATH` / `OUTPUT_PATH`).

### Running via Docker

```bash
docker build --platform linux/amd64 -t cheapskate-router .

docker run --rm \
  -e FIREWORKS_API_KEY=your_key_here \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code \
  -v "$(pwd)/test_input:/input" \
  -v "$(pwd)/test_output:/output" \
  cheapskate-router
```

### Pre-built image

A pre-built image is published via GitHub Actions to GitHub Container Registry:

```bash
docker pull ghcr.io/danush-20/cheapskate-router:latest
```

## Input / output contract

**Input** (`/input/tasks.json`):
```json
[
  { "task_id": "t1", "prompt": "..." }
]
```

**Output** (`/output/results.json`):
```json
[
  { "task_id": "t1", "answer": "..." }
]
```

## Notes

- Only 2 of the 5 allowed Fireworks models (`minimax-m3`, `kimi-k2p7-code`) are serverless and callable without a dedicated deployment; the router's fallback logic transparently handles this.
- The agent never crashes on a single bad task — failures are caught, logged, and a safe fallback answer is written instead, guaranteeing valid output on every run.
