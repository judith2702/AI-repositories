#!/usr/bin/env python3
"""
flan-t5 small-batch script implementing A-D from the user's spec.
 
Requirements:
    pip install transformers torch
 
Model: google/flan-t5-base
Device: CPU only
 
Notes:
- Tokenization uses padding=True, truncation=True, max_length=64.
  Justification: prompts are short single-sentence instructions; 64 tokens
  gives a comfortable bound while keeping CPU memory/attention costs low.
- Generation uses the same max_new_tokens across strategies so outputs are comparable.
- Beam uses num_beams=5 (chosen for slightly stronger search than 3).
- Sampling uses temperature=0.8 and top_p=0.9.
"""
 
import re
import time
import textwrap
from typing import List, Dict, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
 
# -------------------------
# Config
# -------------------------
MODEL_NAME = "google/flan-t5-base"
DEVICE = torch.device("cpu")
MAX_INPUT_LENGTH = 64        
MAX_NEW_TOKENS = 32            
BEAM_SIZE = 5                
SAMPLING_TEMPERATURE = 0.8
SAMPLING_TOP_P = 0.9
 

MIN_WORDS = 8
MAX_WORDS = 24
 

PROMPTS = [
    
    "Rewrite the sentence in simpler English. End with a period. Sentence: 'Python’s clear syntax helps beginners focus on problem-solving.' Output:",
    "Rewrite the sentence in simpler English. End with a period. Sentence: 'Version control lets teams track changes and work safely together.' Output:",
    "Rewrite the sentence in simpler English. End with a period. Sentence: 'Preprocessing text often includes lowercasing and removing extra spaces.' Output:",
    "Rewrite the sentence in simpler English. End with a period. Sentence: 'Short prompts run faster on CPU because attention scales with length.' Output:",
   
    "Explain in one sentence what a learning rate does. End with a period.",
    "Explain in one sentence what an API key is used for. End with a period.",
    "Explain in one sentence what a unit test checks. End with a period.",
    "Explain in one sentence what a tokenizer does in NLP. End with a period.",
   
    "Summarize in one sentence: 'Pipelines bundle tokenization, the model, and decoding. They are great for quick demos on CPU.' Output:",
    "Summarize in one sentence: 'Batching several prompts can improve throughput. Padding and masks keep shapes compatible.' Output:",
    "Summarize in one sentence: 'Beam search is deterministic and often fluent. Sampling adds creativity but may drift.' Output:",
    "Summarize in one sentence: 'SentencePiece and WordPiece split text into subwords. This keeps vocabulary small and improves coverage.' Output:",
]
 
 
 
print("Loading tokenizer and model ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("Loaded model.\n")
 
# ---------------------------------------------------------------------------
# A) Batch tokenization (padding & truncation)
# ---------------------------------------------------------------------------
print("A) Batch tokenization with padding=True, truncation=True, max_length =", MAX_INPUT_LENGTH)
batch_enc = tokenizer(PROMPTS,
                      padding=True,
                      truncation=True,
                      max_length=MAX_INPUT_LENGTH,
                      return_tensors="pt")
input_ids = batch_enc["input_ids"]
attention_mask = batch_enc["attention_mask"]
print(" input_ids.shape:", tuple(input_ids.shape))
print(" attention_mask.shape:", tuple(attention_mask.shape))
print(" tokenizer.pad_token_id:", tokenizer.pad_token_id)
print(" last attention_mask row:", attention_mask[-1].tolist())
print()
 
# ---------------------------------------------------------------------------
# B) Decode the same batch three ways
# ---------------------------------------------------------------------------
# prepare generation kwargs
common_gen = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
}
 
greedy_kwargs = {**common_gen, "do_sample": False, "num_beams": 1}
beam_kwargs   = {**common_gen, "do_sample": False, "num_beams": BEAM_SIZE}  # BEAM_SIZE=5 chosen
sample_kwargs = {**common_gen, "do_sample": True, "num_beams": 1,
                 "temperature": SAMPLING_TEMPERATURE, "top_p": SAMPLING_TOP_P}
 
print("B) Generating the same batch with three strategies (same tokenized batch):\n")
with torch.no_grad():
    # Greedy
    out_greedy = model.generate(input_ids=input_ids.to(DEVICE),
                                attention_mask=attention_mask.to(DEVICE),
                                **greedy_kwargs)
    dec_greedy = tokenizer.batch_decode(out_greedy, skip_special_tokens=True)
 
    # Beam
    out_beam = model.generate(input_ids=input_ids.to(DEVICE),
                              attention_mask=attention_mask.to(DEVICE),
                              **beam_kwargs)
    dec_beam = tokenizer.batch_decode(out_beam, skip_special_tokens=True)
 
    # Sampling
   
    torch.manual_seed(42)
    out_sample = model.generate(input_ids=input_ids.to(DEVICE),
                                attention_mask=attention_mask.to(DEVICE),
                                **sample_kwargs)
    dec_sample = tokenizer.batch_decode(out_sample, skip_special_tokens=True)
 
# Print per-prompt one-liners
for i, prompt in enumerate(PROMPTS, start=1):
    g = dec_greedy[i-1].strip()
    b = dec_beam[i-1].strip()
    s = dec_sample[i-1].strip()
    print(f"{i:02d}. {textwrap.shorten(prompt, width=80)}")
    print(f"    [Greedy] {textwrap.shorten(g, width=200)}")
    print(f"    [Beam ] {textwrap.shorten(b, width=200)}")
    print(f"    [Sample] {textwrap.shorten(s, width=200)}")
    print()
 
# ---------------------------------------------------------------------------
# C) Automatic checks (programmatic, no prose)
# ---------------------------------------------------------------------------
 
TERMINAL_PUNCT = ".!?"
 
def count_terminal_puncts(s: str) -> int:
    # Count only occurrences of ., !, ? as terminal punctuation characters
    return sum(1 for ch in s if ch in TERMINAL_PUNCT)
 
def ends_with_period(s: str) -> bool:
    return s.strip().endswith('.')
 
def one_sentence_check(s: str) -> bool:
    # "Exactly one terminal punctuation among . ! ?" per requirement.
    # We count occurrences of those characters. This is a simple approximation.
    return count_terminal_puncts(s) == 1
 
def word_count(s: str) -> int:
    # simple whitespace split (strip first)
    return len(s.strip().split())
 
def word_count_window_check(s: str, lo=MIN_WORDS, hi=MAX_WORDS) -> bool:
    wc = word_count(s)
    return (wc >= lo) and (wc <= hi)
 
def repetition_flag(s: str) -> bool:
    # detect repeated bigrams/trigrams (simple)
    tokens = s.strip().split()
    if len(tokens) < 4:
        return False
    bigrams = [" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    trigrams = [" ".join(tokens[i:i+3]) for i in range(len(tokens)-2)]
    # if any bigram/trigram occurs more than once -> repetition
    for seq in (bigrams + trigrams):
        if (bigrams + trigrams).count(seq) > 1:
            return True
    # also simple loop detection: same token 4+ times in a row
    if re.search(r'(\b\w+\b)(?:\s+\1){3,}', s):
        return True
    return False
 
def run_checks(s: str) -> Dict[str, Any]:
    oc = one_sentence_check(s)
    ep = ends_with_period(s)
    wc = word_count(s)
    wc_ok = word_count_window_check(s)
    rep = repetition_flag(s)
    return {
        "one_sentence": oc,
        "ends_with_period": ep,
        "word_count": wc,
        "word_count_ok": wc_ok,
        "repetition": rep
    }
 
 
def summarize_checks_for_strategy(outputs: List[str]) -> Dict[str, Any]:
    checks = [run_checks(o) for o in outputs]
    # Constraint pass-rate = % that pass checks 1+2+3 (one_sentence AND ends_with_period AND word_count_ok)
    passes = [1 if (c["one_sentence"] and c["ends_with_period"] and c["word_count_ok"]) else 0 for c in checks]
    pass_rate = 100.0 * sum(passes) / len(passes)
    word_counts = [c["word_count"] for c in checks]
    avg_wc = float(np.mean(word_counts))
    std_wc = float(np.std(word_counts, ddof=0))
    repeat_flags = [1 if c["repetition"] else 0 for c in checks]
    repeat_rate = 100.0 * sum(repeat_flags) / len(repeat_flags)
    return {
        "pass_rate_pct": pass_rate,
        "avg_word_count": avg_wc,
        "std_word_count": std_wc,
        "repetition_pct": repeat_rate,
        "per_prompt_checks": checks
    }
 
stats_g = summarize_checks_for_strategy(dec_greedy)
stats_b = summarize_checks_for_strategy(dec_beam)
stats_s = summarize_checks_for_strategy(dec_sample)
 
# Print compact summary table
print("C) Compact summary table (strategy | pass% | avg_wc ± std | %repetition )")
print("-"*80)
print(f"{'Strategy':8s} | {'Pass%':6s} | {'AvgWC':10s} | {'StdWC':7s} | {'%Repetition':11s}")
print("-"*80)
print(f"{'Greedy':8s} | {stats_g['pass_rate_pct']:6.1f} | {stats_g['avg_word_count']:6.2f} +/- {stats_g['std_word_count']:4.2f} | {stats_g['repetition_pct']:10.1f}")
print(f"{'Beam':8s}   | {stats_b['pass_rate_pct']:6.1f} | {stats_b['avg_word_count']:6.2f} +/- {stats_b['std_word_count']:4.2f} | {stats_b['repetition_pct']:10.1f}")
print(f"{'Sample':8s} | {stats_s['pass_rate_pct']:6.1f} | {stats_s['avg_word_count']:6.2f} +/- {stats_s['std_word_count']:4.2f} | {stats_s['repetition_pct']:10.1f}")
print("-"*80)
print()
 
# -------------------------
# D) Tiny timing (single vs small-batch)
# -------------------------
print("D) Timing: single input vs small batch (includes tokenization -> generation).")
representative_prompt = PROMPTS[0]
 
# Single input timing
t0 = time.perf_counter()
enc_single = tokenizer([representative_prompt], padding=True, truncation=True, max_length=MAX_INPUT_LENGTH, return_tensors="pt")
with torch.no_grad():
    _ = model.generate(input_ids=enc_single["input_ids"].to(DEVICE),
                       attention_mask=enc_single["attention_mask"].to(DEVICE),
                       **common_gen)
t1 = time.perf_counter()
single_time = t1 - t0
 
# Small batch timing (all 12 prompts)
t0 = time.perf_counter()
enc_batch = tokenizer(PROMPTS, padding=True, truncation=True, max_length=MAX_INPUT_LENGTH, return_tensors="pt")
with torch.no_grad():
    _ = model.generate(input_ids=enc_batch["input_ids"].to(DEVICE),
                       attention_mask=enc_batch["attention_mask"].to(DEVICE),
                       **common_gen)
t1 = time.perf_counter()
batch_time = t1 - t0
 
print(f" Single input (tokenize+generate): {single_time:.3f}s")
print(f" Small batch ({len(PROMPTS)} prompts) (tokenize+generate): {batch_time:.3f}s")
print()
 
# -------------------------
# (Optional) per-prompt check outputs summary (compact)
# -------------------------
print("Per-prompt check summary (Strategy : prompt_idx : wc : one_sentence : ends_with_period : wc_ok : repetition )")
for sname, outputs, stats in [("Greedy", dec_greedy, stats_g), ("Beam", dec_beam, stats_b), ("Sample", dec_sample, stats_s)]:
    for idx, out in enumerate(outputs, start=1):
        c = stats["per_prompt_checks"][idx-1]
        wc = c["word_count"]
        print(f"{sname:6s} : {idx:02d} : {wc:02d} : {int(c['one_sentence'])} : {int(c['ends_with_period'])} : {int(c['word_count_ok'])} : {int(c['repetition'])}")
print("\nDone.")
 
 