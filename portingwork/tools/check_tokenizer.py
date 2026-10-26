#!/usr/bin/env python3
"""check_tokenizer.py - compare prompt tokenisation against llama.cpp's own tokenizer.

The engine prints "prompt ids (N): a b c ..." on stderr. llama-tokenize prints
one "id -> 'piece'" line per token. Any mismatch here makes every downstream
number differ, so this is checked before blaming the math.
"""
import os, re, subprocess, sys

TOK = r"C:\Users\rr\Downloads\llama-b11887-bin-win-vulkan-x64\llama-tokenize.exe"
MINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kindi_vk_llama.exe")
MODEL = r"C:\x\Llama-3.2-1B-Instruct-Q8_0.gguf"

PROMPTS = [
    "The sky is blue because",
    "The capital of France is",
    "Write a short poem about the sea:",
    "def quicksort(arr):",
    "2 + 2 = 4. 3 * 7 =",
    "In the beginning, there was",
    "The quick brown fox jumps over the lazy dog.",
    "Explain quantum mechanics in one sentence:",
    "Once upon a time",
    "1 2 3 45 678 9012",
    "It's a test, isn't it?",
    "List three primary colors:",
]

fails = 0
for p in PROMPTS:
    r = subprocess.run([TOK, "-m", MODEL, "-p", p], capture_output=True, timeout=900)
    ref = [int(m.group(1)) for m in re.finditer(r"^\s*(\d+) ->", r.stdout.decode("utf-8", "replace"), re.M)]

    m = subprocess.run([MINE, "--model", MODEL, "--prompt", p, "--temp", "0", "-n", "1"],
                       capture_output=True, timeout=900)
    err = m.stderr.decode("utf-8", "replace")
    mm = re.search(r"prompt ids \(\d+\):([\d\s]+)", err)
    mine = [int(x) for x in mm.group(1).split()] if mm else []

    ok = ref == mine
    print(f"=== [{p}]")
    print(f"  ref  ({len(ref)}) : {' '.join(map(str, ref))}")
    print(f"  mine ({len(mine)}) : {' '.join(map(str, mine))}")
    print("  TOKENIZER MATCH" if ok else "  TOKENIZER DIFF")
    if not ok:
        fails += 1

print()
print(f"=== {len(PROMPTS) - fails}/{len(PROMPTS)} prompts tokenise identically")
sys.exit(0 if fails == 0 else 1)
