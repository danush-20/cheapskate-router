import ast
import json
import os
import re
import sys
import traceback
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # only affects local dev; harness injects real env vars at eval time

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

MODEL_PREFIX = "accounts/fireworks/models/"

CATEGORY_MODEL_MAP = {
    # General tasks — minimax-m3 is the available general-purpose model
    "classifier": "minimax-m3",
    "sentiment": "minimax-m3",
    "ner": "minimax-m3",
    "summarization": "minimax-m3",
    "factual": "minimax-m3",
    "math": "minimax-m3",
    "logic": "minimax-m3",
    # Code tasks — kimi-k2p7-code is code-specialised
    "code_debugging": "kimi-k2p7-code",
    "code_generation": "kimi-k2p7-code",
}

CATEGORY_REASONING_MAP = {
    "classifier": "low",
    "sentiment": "low",
    "ner": "low",
    "summarization": "low",
    "factual": "low",
    "math": "medium",
    "logic": "medium",
    "code_debugging": "medium",
    "code_generation": "medium",
}

# minimax-m3 uses internal reasoning tokens before emitting visible output.
# If max_tokens is exhausted during reasoning, visible content = empty string.
# Caps must be set HIGH enough to clear the reasoning budget + the answer.
# Observed safe minimums from probe_caps.py (run against live API):
#   factual: needs ~150, sentiment: needs ~256+ (heavy reasoner), logic: ~120, ner: ~120
# Rule: cap = observed_completion_tokens * 3 for reasoning models, never below 150.
CATEGORY_MAX_TOKENS = {
    "sentiment": 256,      # heavy internal reasoner — burns tokens before outputting label
    "factual": 150,        # needs ~150 to clear reasoning + one sentence
    "math": 256,           # reasoning model; observed ~50-80 visible but needs headroom
    "logic": 256,          # reasoning model; observed ~75-94 visible but needs headroom
    "ner": 200,            # list output; observed ~47-100 visible
    "summarization": 150,  # one sentence; observed ~55 visible
    "code_debugging": 400, # fix + explanation (observed: ~122-187)
    "code_generation": 500,# complete function; observed ~450 at cap — give slight headroom
}

CODE_CATEGORIES = {"code_debugging", "code_generation"}

# Sent with every call — every token here costs 8× across an 8-task run.
SYSTEM_PROMPT = "Be concise. No headers, no restating the question. Include all items for extraction. Valid indented code only."

CATEGORY_INSTRUCTIONS = {
    "factual": (
        "Answer in one sentence. No elaboration beyond what was asked."
    ),
    "sentiment": (
        "Output ONLY the label. Choose one: Positive, Negative, Neutral, Mixed."
    ),
    "ner": (
        "Extract ALL PERSON, ORGANIZATION, LOCATION, and DATE/TIME entities "
        "(including relative time expressions like 'last March'). "
        "List each as: Entity — TYPE. No prose, no extra commentary."
    ),
    "summarization": (
        "Write exactly one sentence that captures the main point. No preamble."
    ),
    "math": (
        "State the final numeric answer first, then at most 2 lines of working. "
        "No prose introduction."
    ),
    "logic": (
        "State the final answer first (one sentence), then at most 2 lines of reasoning. "
        "No prose introduction."
    ),
    "code_debugging": (
        "Fix the bug. Use the simplest correct fix. Single ```python block. "
        "One sentence outside the block at most."
    ),
    "code_generation": (
        "Write correct Python in a single ```python block, 4-space indent. "
        "Prefer the shortest correct implementation. No placeholder comments."
    ),
}

# Retry instructions — one per category that supports local validation retry
RETRY_INSTRUCTIONS = {
    # Generic: used when the model returned empty content (reasoning exhausted max_tokens)
    "empty": (
        "\n\nIMPORTANT: Respond now. Give a direct answer immediately."
    ),
    "code": (
        "\n\nIMPORTANT: Your previous code had a syntax/indentation error. "
        "Rewrite the code block from scratch with consistent 4-space indentation. "
        "Ensure it has no syntax errors."
    ),
    "sentiment": (
        "\n\nIMPORTANT: Respond with ONLY one of these exact labels: "
        "Positive, Negative, Neutral, Mixed. Nothing else."
    ),
    "ner": (
        "\n\nIMPORTANT: Your answer must list each entity in the format: "
        "Entity — TYPE (e.g. 'Maria Sanchez — PERSON'). No prose."
    ),
    "math": (
        "\n\nIMPORTANT: Your answer must include the numeric result. "
        "State the number first, then brief working."
    ),
    "logic": (
        "\n\nIMPORTANT: Your answer must name the specific answer (person/object) "
        "from the prompt. State it first."
    ),
}

# reasoning_effort fallback order for empty-answer retry
_EFFORT_DOWNGRADE = {"medium": "low", "low": "low"}

# Hard ceiling for the empty-answer retry cap expansion
_EMPTY_RETRY_MAX_CAP = 1024

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

# ---------------------------------------------------------------------------
# Scoring-based classifier — fully local, zero API cost
# ---------------------------------------------------------------------------

# Each category maps to a list of (pattern, weight) tuples.
# Patterns are matched case-insensitively against the full prompt.
# Score = sum of weights for all matching patterns; highest score wins.
CATEGORY_SIGNALS: dict[str, list[tuple[str, int]]] = {
    "sentiment": [
        (r"\bsentiment\b", 4),
        (r"\bclassify the sentiment\b", 5),
        (r"\bpositive|negative|neutral\b", 3),
        (r"\bfeel(ing|s)?\b", 2),
        (r"\bopinion\b", 2),
        (r"\bemotion\b", 2),
        (r"\btone\b", 2),
        (r"\breview\b", 1),
    ],
    "summarization": [
        (r"\bsummar(ize|ise|y|ization)\b", 5),
        (r"\bin (exactly )?one sentence\b", 3),
        (r"\bbrief (summary|overview)\b", 4),
        (r"\bmain (points?|ideas?|takeaway)\b", 3),
        (r"\bcondense\b", 3),
        (r"\bparaphrase\b", 2),
        (r"\bshorten\b", 2),
        (r"\bthe (following|text|passage|article)\b", 1),
    ],
    "ner": [
        (r"\bnamed entit(y|ies)\b", 5),
        (r"\bextract (all )?(entities|names|organizations|locations)\b", 5),
        (r"\bperson|organization|location\b", 3),
        (r"\bNER\b", 5),
        (r"\bidentify (the )?(people|places|companies)\b", 3),
        (r"\bfind (the )?(organizations|locations|names)\b", 3),
    ],
    "code_debugging": [
        (r"\bbug\b", 4),
        (r"\bfix (this|the) (code|function|script)\b", 5),
        (r"\bhas (an? )?(error|bug|issue|problem)\b", 4),
        (r"\bdebug\b", 4),
        (r"\bsyntax error\b", 4),
        (r"\bwhy (does|is) (this|the) (code|function)\b", 3),
        (r"\bdef \w+\(", 2),
        (r"\bfor .* in range\b", 1),
    ],
    "code_generation": [
        (r"\bwrite (a |an )?(python )?(function|script|program|class|method)\b", 5),
        (r"\bimplement (a |an )?\b", 4),
        (r"\bcreate (a |an )?(function|script|class)\b", 4),
        (r"\bgenerate (a |an )?(function|script)\b", 4),
        (r"\bcode (that|to|which)\b", 3),
        (r"\bpython (function|script|program)\b", 3),
        (r"\breturn(s)? (the )?\w+\b", 1),
    ],
    "math": [
        (r"\b\d+\s*[\+\-\*\/\%]\s*\d+\b", 4),
        (r"\bhow (many|much|long|far|old)\b", 3),
        (r"\bcalculate|compute|solve\b", 4),
        (r"\bequation\b", 4),
        (r"\bpercent(age)?\b", 3),
        (r"\btotal|sum|product|difference|quotient\b", 3),
        (r"\bword problem\b", 4),
        (r"\barithmetic\b", 4),
        (r"\bsolve for [a-z]\b", 5),
        (r"\bkm(\/h|ph)?\b", 2),
        (r"\bspeed|distance|time\b", 2),
        (r"\bremain(s|ing)?\b", 2),
    ],
    "logic": [
        (r"\bif .{0,40} then\b", 4),
        (r"\ball \w+ are\b", 4),
        (r"\btherefore\b", 3),
        (r"\bdeduction|deductive\b", 4),
        (r"\bvalid or invalid\b", 5),
        (r"\btrue or false\b", 4),
        (r"\beach (own|has|have)\b", 4),
        (r"\bdifferent (pet|color|house|job)\b", 3),
        (r"\bwho owns\b", 4),
        (r"\bpuzzle\b", 3),
        (r"\bconditional\b", 3),
        (r"\btaller than\b", 3),
        (r"\bfriends?.{0,30}(pet|dog|cat|bird)\b", 3),
    ],
    "factual": [
        (r"\bwhat is (the )?\b", 3),
        (r"\bwho (is|was|were)\b", 3),
        (r"\bwhen (did|was|were|is)\b", 3),
        (r"\bwhere (is|was|did)\b", 3),
        (r"\bhow (is|was|does|do)\b", 2),
        (r"\bwhich\b", 2),
        (r"\bcapital (of|city)\b", 4),
        (r"\bfamous for\b", 3),
        (r"\bhistory\b", 2),
        (r"\bfact\b", 2),
        (r"\bknown as\b", 2),
    ],
}

VALID_CATEGORIES = frozenset(CATEGORY_SIGNALS.keys())


def classify_category(prompt: str) -> tuple[str, int, int]:
    """Score each category against the prompt; return (category, top_score, margin).

    Fully local — zero API calls. margin = top_score - second_score; low margin
    means the classification is ambiguous and worth logging.
    """
    p = prompt.lower()
    scores: dict[str, int] = {}
    for cat, signals in CATEGORY_SIGNALS.items():
        scores[cat] = sum(
            w for pattern, w in signals if re.search(pattern, p, re.IGNORECASE)
        )

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_cat, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    margin = top_score - second_score

    # Tie-break: if top score is 0 (no signals matched at all), default to factual
    if top_score == 0:
        return "factual", 0, 0

    return top_cat, top_score, margin


# ---------------------------------------------------------------------------
# Prompt compression — safe, meaning-preserving, local
# ---------------------------------------------------------------------------

# Boilerplate prefixes that carry no information
_BOILERPLATE_RE = re.compile(
    r"^(could you (please )?|please |i would like you to |i want you to |"
    r"can you (please )?|kindly |you are (a |an )?(helpful )?(assistant|ai)[,.]?\s*)",
    re.IGNORECASE,
)


def compress_prompt(text: str) -> str:
    """Strip redundant whitespace, repeated punctuation, and leading boilerplate.

    Guarantees: numbers, code blocks, and quoted strings are never altered.
    Returns the compressed string (may equal input if nothing to strip).
    """
    # Preserve code blocks verbatim by extracting them first
    code_blocks: list[str] = []
    placeholder_tmpl = "\x00CODE{}\x00"

    def _stash(m: re.Match) -> str:
        code_blocks.append(m.group(0))
        return placeholder_tmpl.format(len(code_blocks) - 1)

    safe = re.sub(r"```[\s\S]*?```|`[^`]+`", _stash, text)

    # Strip leading boilerplate (only at the very start of the prompt)
    safe = _BOILERPLATE_RE.sub("", safe).lstrip()
    # Capitalise first letter if we stripped something
    if safe and safe[0].islower() and text[0].isupper():
        safe = safe[0].upper() + safe[1:]

    # Collapse runs of whitespace/newlines (but not inside preserved blocks)
    safe = re.sub(r"[ \t]{2,}", " ", safe)
    safe = re.sub(r"\n{3,}", "\n\n", safe)

    # Collapse repeated punctuation like "!!!" -> "!" or "..." -> "…" (safe)
    safe = re.sub(r"([!?]){2,}", r"\1", safe)

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        safe = safe.replace(placeholder_tmpl.format(i), block)

    return safe.strip()


# ---------------------------------------------------------------------------
# Validity checks — local, no API call
# ---------------------------------------------------------------------------

_SENTIMENT_LABELS = re.compile(
    r"\b(positive|negative|neutral|mixed)\b", re.IGNORECASE
)
_NER_PATTERN = re.compile(
    r"(—|-|:|\()\s*(PERSON|ORGANIZATION|LOCATION|DATE|TIME|ORG|LOC|PER)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d[\d,\.]*\b")


def _is_valid_sentiment(answer: str) -> bool:
    return bool(_SENTIMENT_LABELS.search(answer))


def _is_valid_ner(answer: str) -> bool:
    return bool(_NER_PATTERN.search(answer))


def _is_valid_math(answer: str) -> bool:
    return bool(_NUMBER_RE.search(answer))


def _is_valid_logic(answer: str, prompt: str) -> bool:
    # Logic answer should contain at least one capitalised word from the prompt
    # (the name/object that is the answer)
    proper_nouns = re.findall(r"\b[A-Z][a-z]+\b", prompt)
    if not proper_nouns:
        return True  # can't check — don't penalise
    return any(noun in answer for noun in proper_nouns)


# ---------------------------------------------------------------------------
# Local model — zero Fireworks tokens for sentiment + NER
# ---------------------------------------------------------------------------
# Model: SmolLM2-1.7B-Instruct Q4_K_M GGUF (~1.1 GB on disk, ~1.3 GB RAM).
# Baked into the image at build time (see Dockerfile). No runtime network needed.
# Lazy-loaded on first use so startup is instant when the local path isn't hit.

LOCAL_MODEL_PATH = os.environ.get(
    "LOCAL_MODEL_PATH",
    "/app/models/smollm2-1.7b-instruct-q4_k_m.gguf",
)

_local_llm = None  # module-level singleton


def _get_local_llm():
    global _local_llm
    if _local_llm is not None:
        return _local_llm
    try:
        from llama_cpp import Llama  # type: ignore
        _local_llm = Llama(
            model_path=LOCAL_MODEL_PATH,
            n_ctx=512,       # short context — sentiment/NER prompts are small
            n_threads=2,     # 2 vCPU in grading container
            verbose=False,
        )
        print(f"[INFO] Local model loaded from {LOCAL_MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Local model unavailable ({e}), will use Fireworks fallback", file=sys.stderr)
        _local_llm = None
    return _local_llm


def _local_infer(system: str, user: str, max_tokens: int = 64) -> str:
    """Run one inference on the local GGUF model. Returns empty string on any error."""
    llm = _get_local_llm()
    if llm is None:
        return ""
    try:
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return (result["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        print(f"[WARN] Local inference error: {e}", file=sys.stderr)
        return ""


def try_local_sentiment(prompt: str) -> str | None:
    """Returns one of Positive/Negative/Neutral/Mixed, or None if invalid."""
    answer = _local_infer(
        system="Output ONLY the label. Choose one: Positive, Negative, Neutral, Mixed.",
        user=prompt,
        max_tokens=16,
    )
    return answer if _is_valid_sentiment(answer) else None


def try_local_ner(prompt: str) -> str | None:
    """Returns 'Entity — TYPE' lines, or None if invalid/empty."""
    answer = _local_infer(
        system=(
            "Extract ALL PERSON, ORGANIZATION, LOCATION, and DATE/TIME entities. "
            "List each as: Entity — TYPE. No prose."
        ),
        user=prompt,
        max_tokens=128,
    )
    return answer if _is_valid_ner(answer) else None


# Categories handled locally (zero Fireworks tokens when local model succeeds)
LOCAL_CATEGORIES = {"sentiment", "ner"}


class TokenTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.retried_count = 0
        self.categories: dict[str, dict[str, int]] = {}
        self.trace: list[dict] = []
        self.local_counts: dict[str, int] = {}  # category -> tasks handled locally

    def record_local(self, category: str):
        self.local_counts[category] = self.local_counts.get(category, 0) + 1

    def record(self, task_id: str, category: str, budget: int, tokens_used: int, captured: str):
        """captured: 'yes' | 'retried' | 'no' | 'local'"""
        self.trace.append({
            "task_id": task_id,
            "category": category,
            "budget_allocated": budget,
            "tokens_used": tokens_used,
            "captured": captured,
            "efficiency": round(tokens_used / budget, 3) if budget else None,
        })

    def add(self, category: str, usage):
        if not usage:
            return
        p_tok = getattr(usage, "prompt_tokens", 0) or 0
        c_tok = getattr(usage, "completion_tokens", 0) or 0
        t_tok = getattr(usage, "total_tokens", 0) or (p_tok + c_tok)

        self.prompt_tokens += p_tok
        self.completion_tokens += c_tok
        self.total_tokens += t_tok

        bucket = self.categories.setdefault(category, {"prompt": 0, "completion": 0, "total": 0})
        bucket["prompt"] += p_tok
        bucket["completion"] += c_tok
        bucket["total"] += t_tok

    def print_report(self):
        print("\n" + "=" * 50)
        print("           TOKEN USAGE REPORT")
        print("=" * 50)
        print(f"{'Category':<22} {'Prompt':>8} {'Completion':>12} {'Total':>8} {'Local':>6}")
        print("-" * 50)
        all_cats = sorted(set(list(self.categories.keys()) + list(self.local_counts.keys())))
        for cat in all_cats:
            t = self.categories.get(cat, {"prompt": 0, "completion": 0, "total": 0})
            loc = self.local_counts.get(cat, 0)
            loc_str = str(loc) if loc else "-"
            print(f"{cat:<22} {t['prompt']:>8} {t['completion']:>12} {t['total']:>8} {loc_str:>6}")
        print("-" * 50)
        total_local = sum(self.local_counts.values())
        print(f"{'TOTAL':<22} {self.prompt_tokens:>8} {self.completion_tokens:>12} {self.total_tokens:>8} {total_local:>6}")
        print(f"Tasks retried: {self.retried_count}  |  Tasks handled locally (0 Fireworks tokens): {total_local}")
        print("=" * 50 + "\n")

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "retried_count": self.retried_count,
            "local_counts": self.local_counts,
            "by_category": self.categories,
            "sowing_capturing_trace": self.trace,
        }


def build_system_prompt(category: str) -> str:
    extra = CATEGORY_INSTRUCTIONS.get(category)
    return f"{SYSTEM_PROMPT}\n\n{extra}" if extra else SYSTEM_PROMPT


def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ["FIREWORKS_BASE_URL"],
    )


def resolve_model(category: str, allowed_models: list[str]) -> str:
    desired_short = CATEGORY_MODEL_MAP.get(category)
    desired_full = f"{MODEL_PREFIX}{desired_short}" if desired_short else None

    if desired_full and desired_full in allowed_models:
        return desired_full
    for m in allowed_models:
        if desired_short and desired_short in m:
            return m
    return allowed_models[0]


def extract_code_blocks(answer: str) -> list[str]:
    return CODE_BLOCK_RE.findall(answer)


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def answer_has_valid_code(answer: str) -> bool:
    blocks = extract_code_blocks(answer)
    return bool(blocks) and all(is_valid_python(b) for b in blocks)


def extract_final_code_block(answer: str) -> str:
    """Return the last valid python block only, stripping chain-of-thought prose.

    minimax-m3 sometimes emits reasoning prose before the final code block.
    Using the last block avoids returning an intermediate/partial attempt.
    """
    blocks = CODE_BLOCK_RE.findall(answer)
    valid = [b for b in blocks if is_valid_python(b)]
    if not valid:
        return answer  # no valid block found — return as-is, retry will handle it
    # Reconstruct as a clean fenced block
    return f"```python\n{valid[-1].strip()}\n```"


def call_model(client, model, system_prompt, user_prompt, reasoning_effort="low", max_tokens=None):
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "reasoning_effort": reasoning_effort,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return client.chat.completions.create(**kwargs)


def _maybe_retry(
    client, model, system_prompt, original_prompt, answer,
    category, reasoning_effort, max_tokens, tracker: TokenTracker
) -> tuple[str, bool]:
    """Run one local validity check; if it fails, retry once. Returns (answer, retried)."""

    # Empty-answer guard: fires before category checks, applies to ALL categories.
    # Happens when a reasoning model exhausts max_tokens on internal reasoning.
    # Fix: double the cap (bounded) + downgrade reasoning_effort to reduce reasoning overhead.
    if not answer.strip():
        retry_cap = min((max_tokens or 256) * 2, _EMPTY_RETRY_MAX_CAP)
        retry_effort = _EFFORT_DOWNGRADE.get(reasoning_effort, "low")
        retry_prompt = original_prompt + RETRY_INSTRUCTIONS["empty"]
        response = call_model(client, model, system_prompt, retry_prompt, retry_effort, retry_cap)
        if hasattr(response, "usage"):
            tracker.add(f"{category}_retry", response.usage)
        retried = response.choices[0].message.content or ""
        return (retried if retried.strip() else answer), True

    retry_key: str | None = None

    if category in CODE_CATEGORIES:
        if not answer_has_valid_code(answer):
            retry_key = "code"
    elif category == "sentiment":
        if not _is_valid_sentiment(answer):
            retry_key = "sentiment"
    elif category == "ner":
        if not _is_valid_ner(answer):
            retry_key = "ner"
    elif category == "math":
        if not _is_valid_math(answer):
            retry_key = "math"
    elif category == "logic":
        if not _is_valid_logic(answer, original_prompt):
            retry_key = "logic"

    if retry_key is None:
        return answer, False

    retry_prompt = original_prompt + RETRY_INSTRUCTIONS[retry_key]
    response = call_model(client, model, system_prompt, retry_prompt, reasoning_effort, max_tokens)
    if hasattr(response, "usage"):
        tracker.add(f"{category}_retry", response.usage)
    retried = response.choices[0].message.content or ""
    return (retried if retried.strip() else answer), True


# ---------------------------------------------------------------------------
# Core task processor
# ---------------------------------------------------------------------------

def process_task(client, task: dict, allowed_models: list[str], tracker: TokenTracker) -> dict:
    """Process one task. Never raises — always returns {task_id, answer}."""
    task_id = task.get("task_id", "unknown")
    raw_prompt = task.get("prompt", "")

    try:
        # 1. Classify (local, zero tokens)
        category, top_score, margin = classify_category(raw_prompt)
        print(
            f"[INFO] {task_id}: category='{category}' score={top_score} margin={margin}"
            + (" [LOW MARGIN — check classification]" if margin <= 2 else "")
        )

        # 2. Compress prompt
        prompt = compress_prompt(raw_prompt)
        orig_est, comp_est = len(raw_prompt) // 4, len(prompt) // 4
        if orig_est != comp_est:
            print(f"[INFO] {task_id}: prompt compressed ~{orig_est} → ~{comp_est} tokens")

        # 3. Resolve model + build system prompt
        model = resolve_model(category, allowed_models)
        system_prompt = build_system_prompt(category)
        reasoning_effort = CATEGORY_REASONING_MAP.get(category, "low")
        max_tokens = CATEGORY_MAX_TOKENS.get(category)

        # 3a. Try local model first for zero-token categories
        if category in LOCAL_CATEGORIES:
            local_fn = try_local_sentiment if category == "sentiment" else try_local_ner
            local_answer = local_fn(prompt)
            if local_answer:
                print(f"[INFO] {task_id}: handled locally, 0 Fireworks tokens")
                tracker.record_local(category)
                tracker.record(task_id, category, 0, 0, "local")
                return {"task_id": task_id, "answer": local_answer}
            print(f"[INFO] {task_id}: local model invalid/unavailable, falling through to Fireworks")

        # 4. First call
        response = call_model(client, model, system_prompt, prompt, reasoning_effort, max_tokens)
        if hasattr(response, "usage"):
            tracker.add(category, response.usage)
        answer = response.choices[0].message.content or ""

        # 5. Local validity check + single retry if needed
        # For code categories: strip chain-of-thought prose, keep last valid block only
        if category in CODE_CATEGORIES:
            answer = extract_final_code_block(answer)
        answer, retried = _maybe_retry(
            client, model, system_prompt, prompt, answer,
            category, reasoning_effort, max_tokens, tracker
        )
        if retried:
            tracker.retried_count += 1
            print(f"[INFO] {task_id}: retried (category={category})")

        if not answer.strip():
            answer = "No answer could be generated for this task."

        return {"task_id": task_id, "answer": answer}

    except Exception as e:
        print(f"[WARN] Task {task_id} failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {"task_id": task_id, "answer": "Error: could not generate an answer for this task."}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    try:
        allowed_models = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
        if not allowed_models:
            raise ValueError("ALLOWED_MODELS is empty")
        client = get_client()
    except Exception as e:
        print(f"[FATAL] Could not initialise client/config: {e}", file=sys.stderr)
        os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump([], f)
        sys.exit(1)

    try:
        with open(INPUT_PATH) as f:
            tasks = json.load(f)
        if not isinstance(tasks, list):
            raise ValueError("tasks.json must be a JSON array")
    except Exception as e:
        print(f"[FATAL] Could not read {INPUT_PATH}: {e}", file=sys.stderr)
        os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump([], f)
        sys.exit(1)

    tracker = TokenTracker()
    results = []
    for task in tasks:
        result = process_task(client, task, allowed_models, tracker)
        results.append(result)
        print(f"{result['task_id']:<15} done")

    out_dir = os.path.dirname(OUTPUT_PATH) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} results to {OUTPUT_PATH}")

    tracker.print_report()

    # Write run_stats.json alongside results if the directory is writable
    try:
        stats_path = os.path.join(out_dir, "run_stats.json")
        with open(stats_path, "w") as f:
            json.dump(tracker.to_dict(), f, indent=2)
        print(f"Run stats written to {stats_path}")
    except Exception as e:
        print(f"[WARN] Could not write run_stats.json: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
