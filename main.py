import ast
import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # only affects local dev; harness injects real env vars at eval time

INPUT_PATH = os.environ.get("INPUT_PATH", "test_input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "test_output/results.json")

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

# Extra instruction appended for specific categories known to need stricter guidance.
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
    return "factual"  # default fallback


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
    """Pick the mapped model for this category, falling back to the first
    allowed model if the mapped one isn't actually in ALLOWED_MODELS."""
    desired_short_name = CATEGORY_MODEL_MAP.get(category)
    desired_full_id = f"{MODEL_PREFIX}{desired_short_name}" if desired_short_name else None

    if desired_full_id and desired_full_id in allowed_models:
        return desired_full_id

    # Fallback: match by short name if allowed_models entries are already full IDs
    for m in allowed_models:
        if desired_short_name and desired_short_name in m:
            return m

    # Last resort: first allowed model
    return allowed_models[0]


def extract_code_blocks(answer: str) -> list[str]:
    """Pull out all fenced code blocks from a markdown-style answer."""
    return CODE_BLOCK_RE.findall(answer)


def is_valid_python(code: str) -> bool:
    """Zero-token, local check: does this code parse as valid Python?"""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def answer_has_valid_code(answer: str) -> bool:
    """For code-category answers, all code blocks found must parse cleanly.
    If no code block is found at all, treat as invalid (something's off)."""
    blocks = extract_code_blocks(answer)
    if not blocks:
        return False
    return all(is_valid_python(block) for block in blocks)


def call_model(client, model, system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        reasoning_effort="low",
    )
    return response


def main():
    client = get_client()
    allowed_models = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",")]

    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    retries_used = 0

    results = []
    for task in tasks:
        category = classify_category(task["prompt"])
        model = resolve_model(category, allowed_models)
        system_prompt = build_system_prompt(category)

        response = call_model(client, model, system_prompt, task["prompt"])
        answer = response.choices[0].message.content

        usage = getattr(response, "usage", None)
        if usage:
            total_input_tokens += usage.prompt_tokens
            total_output_tokens += usage.completion_tokens
            total_tokens += usage.total_tokens

        retried = False
        # Local, zero-token validation: for code tasks, verify syntax and
        # retry once (one extra Fireworks call) if it's broken.
        if category in CODE_CATEGORIES and not answer_has_valid_code(answer):
            retried = True
            retries_used += 1
            retry_prompt = task["prompt"] + RETRY_INSTRUCTION
            response = call_model(client, model, system_prompt, retry_prompt)
            answer = response.choices[0].message.content

            usage = getattr(response, "usage", None)
            if usage:
                total_input_tokens += usage.prompt_tokens
                total_output_tokens += usage.completion_tokens
                total_tokens += usage.total_tokens

        results.append({"task_id": task["task_id"], "answer": answer})

        tag = " [retried]" if retried else ""
        print(f"{task['task_id']:<15} category={category:<15} model={model} "
              f"tokens={usage.total_tokens if usage else 'n/a'}{tag}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Token summary ---")
    print(f"Input tokens:  {total_input_tokens}")
    print(f"Output tokens: {total_output_tokens}")
    print(f"Total tokens:  {total_tokens}")
    print(f"Code retries:  {retries_used}")


if __name__ == "__main__":
    main()