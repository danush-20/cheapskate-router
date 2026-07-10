"""Probe minimax-m3 to find minimum safe token caps per category.
Run: python probe_caps.py
"""
import os, sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["FIREWORKS_API_KEY"], base_url=os.environ["FIREWORKS_BASE_URL"])
MODEL = "accounts/fireworks/models/minimax-m3"

PROBES = [
    ("factual",      "low",    "What is the capital of Australia, and what body of water is it near?",                          [60, 80, 100, 150]),
    ("sentiment",    "low",    "Classify the sentiment: The battery life is great, but the screen scratches too easily.",        [10, 15, 20, 30]),
    ("math",         "medium", "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many remain?",         [80, 100, 120, 150]),
    ("logic",        "medium", "Three friends Sam, Jo, Lee each own cat/dog/bird. Sam not bird. Jo owns dog. Who owns cat?",     [60, 80, 100, 120]),
    ("ner",          "low",    "Extract all named entities: Maria Sanchez joined Fireworks AI in Berlin last March.",            [80, 100, 120]),
    ("summarization","low",    "Summarize in one sentence: The quick brown fox jumps over the lazy dog near the river bank.",    [40, 60, 80, 100]),
]

SYS = "Be concise. No headers, no restating the question. Include all items for extraction. Valid indented code only."

print(f"{'Category':<14} {'Cap':>5} {'Finish':<12} {'Tokens':>7}  Answer")
print("-" * 80)
for cat, effort, prompt, caps in PROBES:
    for cap in caps:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
            max_tokens=cap,
            reasoning_effort=effort,
        )
        content = (r.choices[0].message.content or "").strip()
        finish = r.choices[0].finish_reason
        ctok = r.usage.completion_tokens
        snippet = content[:60].replace("\n", " ")
        empty = " *** EMPTY ***" if not content else ""
        print(f"{cat:<14} {cap:>5} {finish:<12} {ctok:>7}  {snippet}{empty}")
    print()
