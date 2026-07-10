import json
import re

INPUT_PATH = "test_input/tasks.json"


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


def main():
    with open(INPUT_PATH, "r") as f:
        tasks = json.load(f)

    print(f"{'task_id':<15} {'predicted category':<18} prompt (truncated)")
    print("-" * 90)
    for task in tasks:
        category = classify_category(task["prompt"])
        snippet = task["prompt"][:50].replace("\n", " ")
        print(f"{task['task_id']:<15} {category:<18} {snippet}...")




if __name__ == "__main__":
    main()