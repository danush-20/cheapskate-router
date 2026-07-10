"""Tests for the scoring-based classifier and compress_prompt.

Run with:  python test_classifier.py
No external dependencies required — all checks are local/offline.
"""
import json
import sys
from main import classify_category, compress_prompt

INPUT_PATH = "test_input/tasks.json"

# ---------------------------------------------------------------------------
# Expected classifications for the 8 practice tasks
# ---------------------------------------------------------------------------
PRACTICE_EXPECTED = {
    "practice-01": "factual",
    "practice-02": "math",
    "practice-03": "sentiment",
    "practice-04": "summarization",
    "practice-05": "ner",
    "practice-06": "code_debugging",
    "practice-07": "logic",
    "practice-08": "code_generation",
}

# ---------------------------------------------------------------------------
# Additional edge-case prompts that previously tripped the keyword classifier
# ---------------------------------------------------------------------------
EDGE_CASES = [
    # Math — no obvious "%" or "total/remain" keywords in first pass
    ("Solve for x: 5x + 3 = 18. What is the value of x?", "math"),
    ("A train travels at 80 km/h. How long to travel 240 km?", "math"),
    # Logic — no "each own a different pet" phrasing
    (
        "All mammals are warm-blooded. A whale is a mammal. "
        "Therefore, a whale is warm-blooded. Is this deduction valid or invalid?",
        "logic",
    ),
    (
        "If Alice is taller than Bob and Bob is taller than Charlie, "
        "then Alice is taller than Charlie. True or False?",
        "logic",
    ),
    # Code generation — "script" instead of "function"
    (
        "Write a Python script that reads a CSV file and calculates "
        "the average of the second column.",
        "code_generation",
    ),
    # Code debugging — no "has a bug" phrase
    ("Fix this code: for i in range(10) print(i)", "code_debugging"),
    # NER — "find the organizations and locations"
    (
        "Find the organizations and locations in this sentence: "
        "Microsoft announced new layoffs at its Seattle headquarters.",
        "ner",
    ),
    # Sentiment — phrased as a question
    (
        "Does the following statement express a positive, negative, or neutral sentiment? "
        "'I guess the movie was okay, but I wouldn't watch it again.'",
        "sentiment",
    ),
    # Summarization — no "summarize" verb
    ("Provide a brief summary of the main points of the text below.", "summarization"),
    # Factual — general knowledge question
    (
        "Who is widely considered the father of modern physics, "
        "famous for the theory of relativity?",
        "factual",
    ),
]


def _run_classifier_tests(cases: list[tuple[str, str]], label: str) -> tuple[int, int]:
    passed = failed = 0
    for prompt, expected in cases:
        cat, score, margin = classify_category(prompt)
        ok = cat == expected
        status = "PASS" if ok else "FAIL"
        margin_warn = " [LOW MARGIN]" if margin <= 2 else ""
        snippet = prompt[:55].replace("\n", " ") + ("…" if len(prompt) > 55 else "")
        print(f"  [{status}] expected={expected:<16} got={cat:<16} score={score} margin={margin}{margin_warn}")
        print(f"         {snippet}")
        if ok:
            passed += 1
        else:
            failed += 1
    return passed, failed


def test_classifier():
    print("=" * 65)
    print(" CLASSIFIER TESTS")
    print("=" * 65)

    # Load practice tasks
    try:
        with open(INPUT_PATH) as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load {INPUT_PATH}: {e}")
        tasks = []

    practice_cases = [
        (t["prompt"], PRACTICE_EXPECTED[t["task_id"]])
        for t in tasks
        if t.get("task_id") in PRACTICE_EXPECTED
    ]

    print("\n--- Practice tasks (8 categories) ---")
    p1, f1 = _run_classifier_tests(practice_cases, "practice")

    print("\n--- Edge cases ---")
    p2, f2 = _run_classifier_tests(EDGE_CASES, "edge")

    total_pass = p1 + p2
    total = total_pass + f1 + f2
    print(f"\nClassifier: {total_pass}/{total} passed")
    return f1 + f2  # return failure count


# ---------------------------------------------------------------------------
# compress_prompt tests — assert information-bearing content is never altered
# ---------------------------------------------------------------------------

COMPRESS_CASES = [
    # Boilerplate stripped, content preserved
    (
        "Could you please summarize the following text: The quick brown fox.",
        lambda r: "The quick brown fox" in r and "Could you please" not in r,
        "strips leading boilerplate",
    ),
    (
        "I would like you to calculate 42 * 7.",
        lambda r: "42" in r and "7" in r,
        "numbers preserved after boilerplate strip",
    ),
    # Redundant whitespace collapsed
    (
        "What   is   the   capital   of   France?",
        lambda r: "  " not in r and "France" in r,
        "collapses redundant spaces",
    ),
    # Code block untouched
    (
        "Fix this code:\n```python\ndef foo():\n    return  42\n```",
        lambda r: "def foo():" in r and "return  42" in r,
        "code block content preserved verbatim",
    ),
    # Repeated punctuation collapsed
    (
        "Is this correct???",
        lambda r: r.count("?") == 1,
        "repeated punctuation collapsed",
    ),
    # Numbers never altered
    (
        "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday.",
        lambda r: "240" in r and "15%" in r and "60" in r,
        "numbers in math prompt preserved",
    ),
    # No-op: already clean
    (
        "What is the capital of Australia?",
        lambda r: r == "What is the capital of Australia?",
        "clean prompt unchanged",
    ),
]


def test_compress_prompt():
    print("\n" + "=" * 65)
    print(" COMPRESS_PROMPT TESTS")
    print("=" * 65)
    failures = 0
    for text, check, description in COMPRESS_CASES:
        result = compress_prompt(text)
        ok = check(result)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {description}")
        if not ok:
            print(f"         input:  {text!r}")
            print(f"         output: {result!r}")
            failures += 1
    print(f"\ncompress_prompt: {len(COMPRESS_CASES) - failures}/{len(COMPRESS_CASES)} passed")
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    clf_failures = test_classifier()
    cmp_failures = test_compress_prompt()
    total_failures = clf_failures + cmp_failures
    if total_failures:
        print(f"\n{total_failures} test(s) FAILED")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)
