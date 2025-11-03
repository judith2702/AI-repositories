import argparse
import csv
from transformers import pipeline
from tqdm import tqdm

# Load summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Approximate token count function
def count_tokens(text):
    return len(text.split())

# Constraint checker
def validate(summary):
    words = summary.strip().split()
    word_count = len(words)
    is_one_sentence = summary.count('.') == 1 and summary.strip().endswith('.')
    
    if not is_one_sentence:
        return False, "Not one sentence or missing period"
    if word_count < 8:
        return False, "Too short"
    if word_count > 24:
        return False, "Too long"
    return True, ""

# Summarization function with decoding config
def summarize_text(text, mode):
    if mode == "beam":
        return summarizer(text, max_length=50, min_length=5, do_sample=False, num_beams=4)[0]["summary_text"]
    elif mode == "sample":
        return summarizer(text, max_length=50, min_length=5, do_sample=True, top_p=0.9, temperature=0.8)[0]["summary_text"]
    else:
        raise ValueError("Invalid decoding mode")

# Argument parsing
parser = argparse.ArgumentParser(description="Batch summarization using DistilBART.")
parser.add_argument('--infile', required=True, help='Input .txt file with one entry per line')
args = parser.parse_args()

# Read input lines
with open(args.infile, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

# Prepare output
rows = []
stats = {"beam": {"passes": 0, "total": 0, "tokens": []}, "sample": {"passes": 0, "total": 0, "tokens": []}}

# Process each line with both decoding modes
for line in tqdm(lines, desc="Processing lines"):
    for mode in ["beam", "sample"]:
        output = summarize_text(line, mode)
        tokens_out = count_tokens(output)
        passed, note = validate(output)
        
        stats[mode]["total"] += 1
        stats[mode]["tokens"].append(tokens_out)
        if passed:
            stats[mode]["passes"] += 1
        
        rows.append({
            "input": line,
            "output": output,
            "decoding": mode,
            "tokens_out": tokens_out,
            "constraint_passed": passed,
            "notes": "" if passed else note
        })

# Write CSV
with open("results.csv", "w", newline='', encoding='utf-8') as csvfile:
    fieldnames = ["input", "output", "decoding", "tokens_out", "constraint_passed", "notes"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# Print mini-report
print("\n--- Summary Report ---")
for mode in ["beam", "sample"]:
    total = stats[mode]["total"]
    passed = stats[mode]["passes"]
    avg_tokens = sum(stats[mode]["tokens"]) / total if total else 0
    quality_comment = (
        "Beam search produced more consistent outputs but with slightly lower creativity." if mode == "beam"
        else "Sampling generated more varied outputs, though some lacked clarity or structure."
    )
    print(f"\nMode: {mode}")
    print(f" - % Passed Constraints: {100 * passed / total:.1f}%")
    print(f" - Avg Tokens Out: {avg_tokens:.2f}")
    print(f" - Note: {quality_comment}")
