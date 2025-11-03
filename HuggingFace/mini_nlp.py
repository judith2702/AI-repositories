import sys
import argparse
from transformers import pipeline

parser = argparse.ArgumentParser(description="Simplify an English sentence.")
parser.add_argument('--text', type=str, help='Sentence to simplify')
args = parser.parse_args()


if args.text:
    sentence = args.text
else:
    sentence = input("Enter an English sentence: ")

simplifier = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")


outputs = simplifier(sentence, max_length=50, do_sample=False, num_beams=4)
simplified = outputs[0]['generated_text'].strip()


word_count = len(simplified.split())
is_one_sentence = simplified.count('.') == 1
ends_with_period = simplified.endswith('.')

if 8 <= word_count <= 24 and is_one_sentence and ends_with_period:
    print(simplified)
else:
    print("Constraint not satisfied.")


######################################################3
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


input_str = input("Enter three sentences : ")
sentences = [s.strip() for s in input_str.split(',') if s.strip()]

if len(sentences) != 3:
    print("Please enter exactly three comma-separated sentences.")
    exit()

print("\nSentiment Analysis Results:")
for i, sentence in enumerate(sentences, 1):
    result = sentiment_analyzer(sentence)[0]
    label = result['label']
    score = round(result['score'], 4)
    print(f"Sentence {i}: {label} (confidence: {score})")




