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
    "sentiment": "gemma-4-26b-a4b-it",
    "ner": "gemma-4-26b-a4b-it",
    "summarization": "gemma-4-26b-a4b-it",
    "factual": "gemma-4-26b-a4b-it",
    "math": "gemma-4-31b-it",
    "logic": "gemma-4-31b-it",
    "code_debugging": "kimi-k2p7-code",
    "code_generation": "kimi-k2p7-code",
}

CODE_CATEGORIES = {"code_debugging", "code_generation"}

SYSTEM_PROMPT = (
    "Answer directly and concisely. No markdown headers, no comparison tables, "
    "no restating the question, no alternative solutions unless explicitly asked. "
    "Being brief must never cost correctness or completeness: "
    "if asked to extract multiple items (e.g. entities), include ALL of them, none omitted. "
    "If asked to write or fix code, the code must be complete, syntactically valid, "
    "and correctly indented — never truncate or mix tabs/spaces. "
    "Justify answers briefly only where justification is needed."
)

CATEGORY_INSTRUCTIONS = {
    "ner": (
        "Extract EVERY named entity in the text, including PERSON, ORGANIZATION, "
        "LOCATION, and DATE/TIME expressions (e.g. 'last March', 'next week', 'yesterday'). "
        "Relative or informal time expressions still count as DATE entities — do not omit them. "
        "List each entity with its type."
    ),
}

RETRY_INSTRUCTION = (
    "\n\nIMPORTANT: Your previous code had a syntax/indentation error. "
    "Re-write the ENTIRE code block from scratch with fully consistent indentation "
    "(4 spaces per level, no mixed levels, every statement inside the function body "
    "properly indented). Double-check the code would run without a SyntaxError."
)

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def classify_category(prompt: str) -> str:
    p = prompt.lower()
    if "sentiment" in p or "classify the sentiment" in p:
        return "sentiment"
    if "summar" in p:
        return "summarization"
    if "named entit" in p or "extract all named" in p:
        return "ner"
    if "bug" in p and ("def " in prompt or "function" in p):
        return "code_debugging"
    if "write a" in p and ("function" in p or "python" in p or "code" in p):
        return "code_generation"
    if re.search(r"\d+%|\btotal\b|\bremain\b|how many|percent", p):
        return "math"
    if "each own" in p or "who owns" in p or ("different" in p and "own" in p):
        return "logic"
    return "factual"


def build_system_prompt(category: str) -> str:
    extra = CATEGORY_INSTRUCTIONS.get(category)
    if extra:
        return f"{SYSTEM_PROMPT}\n\n{extra}"
    return SYSTEM_PROMPT


def get_client():
    return OpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ["FIREWORKS_BASE_URL"],
    )


def resolve_model(category: str, allowed_models: list[str]) -> str:
    desired_short_name = CATEGORY_MODEL_MAP.get(category)
    desired_full_id = f"{MODEL_PREFIX}{desired_short_name}" if desired_short_name else None

    if desired_full_id and desired_full_id in allowed_models:
        return desired_full_id

    for m in allowed_models:
        if desired_short_name and desired_short_name in m:
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
    if not blocks:
        return False
    return all(is_valid_python(block) for block in blocks)


def call_model(client, model, system_prompt, user_prompt):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        reasoning_effort="low",
    )


def process_task(client, task, allowed_models):
    """Process a single task. Never raises — always returns a result dict,
    falling back to a safe placeholder answer if anything goes wrong."""
    task_id = task.get("task_id", "unknown")
    prompt = task.get("prompt", "")

    try:
        category = classify_category(prompt)
        model = resolve_model(category, allowed_models)
        system_prompt = build_system_prompt(category)

        response = call_model(client, model, system_prompt, prompt)
        answer = response.choices[0].message.content or ""

        if category in CODE_CATEGORIES and not answer_has_valid_code(answer):
            retry_prompt = prompt + RETRY_INSTRUCTION
            response = call_model(client, model, system_prompt, retry_prompt)
            retried_answer = response.choices[0].message.content
            if retried_answer:
                answer = retried_answer

        if not answer.strip():
            answer = "No answer could be generated for this task."

        return {"task_id": task_id, "answer": answer}

    except Exception as e:
        # Never let a single task's failure crash the whole run.
        print(f"[WARN] Task {task_id} failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {"task_id": task_id, "answer": "Error: could not generate an answer for this task."}


def main():
    try:
        allowed_models = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
        if not allowed_models:
            raise ValueError("ALLOWED_MODELS is empty")
        client = get_client()
    except Exception as e:
        print(f"[FATAL] Could not initialize client/config: {e}", file=sys.stderr)
        # Still try to write an empty-but-valid output before exiting non-zero,
        # in case the harness inspects the file regardless of exit code.
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump([], f)
        sys.exit(1)

    try:
        with open(INPUT_PATH, "r") as f:
            tasks = json.load(f)
        if not isinstance(tasks, list):
            raise ValueError("tasks.json must contain a JSON array")
    except Exception as e:
        print(f"[FATAL] Could not read/parse {INPUT_PATH}: {e}", file=sys.stderr)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump([], f)
        sys.exit(1)

    results = []
    for task in tasks:
        result = process_task(client, task, allowed_models)
        results.append(result)
        print(f"{result['task_id']:<15} done")

    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"[FATAL] Could not write {OUTPUT_PATH}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nWrote {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()