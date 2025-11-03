# # -----------------------------------------------------------------------------
# # Part 7- Batching, padding & truncation
# # -----------------------------------------------------------------------------

# # (for T5, pad_token_id is 0)

# #    • Shape: [batch_size, sequence_length], same as input_ids.
# #    • Values: 1 means 'this position is a *real* token', 0 means 'this is PAD'.

# # Padding mask (0s)
# # casual mask -> .generate()


# #truncation (max_lentgth = ??token)
# # transformers



# import torch
# from transformers import AutoTokenizer 
# from transformers import AutoModelForSeq2SeqLM
# model_id = "google/flan-t5-base"
# device = torch.device("cpu")
# tok = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

# sentences = [
#     "Show one-sentence summary of why Python is used in education.",
#     "Summarize the benefit of running models locally on CPU for teaching.",
#     "Explain in one sentence what a tokenizer does.",   
# ]

# enc_batch = tok(
#     [f"Responf in one sentence: {s}" for s in sentences],
#     return_tensors="pt", #PyTorch tensors
#     padding = True,
#     truncation = True,
#     max_length = 96 #the cap
# ).to(device)


# print("\n---BATCHING---")
# print("input_ids shape:", enc_batch.input_ids.shape) #[B, L]
# print("attention_mask shape:", enc_batch.attention_mask.shape)
# print("Pad token id :", tok.pad_token_id)

# # torch.Size([3, 20])

# with torch.no_grad():
#     out_batch = model.generate(
#         **enc_batch,  #unpacks input_ids and attention_mask
#         max_new_tokens = 32,
#         num_beams = 4,
#         do_sample = False
#     )

# print("\n---BATCH OUTPUTS---\n")
# for i, out in enumerate(out_batch):
#     print(f"{i+1}.", tok.decode(out, skip_special_tokens=True))



# # -----------------------------------------------------------------------------
# # Part 8 - Decoding strategies: greedy vs beam vs sampling
# # -----------------------------------------------------------------------------


# import torch
# from transformers import AutoTokenizer 
# from transformers import AutoModelForSeq2SeqLM
# model_id = "google/flan-t5-base"
# device = torch.device("cpu")
# tok = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

# def run_decode(prompt: str):
#     enc = tok(prompt, return_tensors="pt").to(device)

#     with torch.no_grad():
#         #Greedy decoding
#         g = model.generate(
#             **enc,
#             max_new_tokens=64,
#             do_sample=False,
#             num_beams=1 #pure greedy
#         )

#         #Beam search
#         b = model.generate(
#             **enc,
#             max_new_tokens = 64,
#             do_sample = False,
#             num_beams=5
#         )

#         # Sampling
#         s = model.generate(
#             **enc,
#             max_new_tokens = 64,
#             do_sample = True,
#             temperature=0.8, #<1 = safer, >1 =more random
#             top_p=0.9, #nucleus sampling
#             num_return_sequences = 1
#         )

#         return(
#             tok.decode(g[0], skip_special_tokens=True), 
#             tok.decode(b[0], skip_special_tokens=True),
#             tok.decode(s[0], skip_special_tokens=True),
#         )

# cmp_prompt = (
#         "Create ONE playful, single-sentence analogy that explains how text is split "
#         "for language models to understand. Do NOT use the words 'token' or 'tokenizer'. "
#         "End with a period."    
#     )

# greedy_text, beam_text, sample_text = run_decode(cmp_prompt)

# print("\n---DECODE COMPARISON---\n")
# print("[Greedy]",greedy_text)
# print("[Beam]",beam_text)
# print("[Sample]",sample_text)



# -----------------------------------------------------------------------------
# Part 9 - Timing (single prompt vs small batch on CPU)
# -----------------------------------------------------------------------------

import torch
from transformers import AutoTokenizer 
from transformers import AutoModelForSeq2SeqLM
model_id = "google/flan-t5-base"
device = torch.device("cpu")
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
import time

sentences = [
    "Show one-sentence summary of why Python is used in education.",
    "Summarize the benefit of running models locally on CPU for teaching.",
    "Explain in one sentence what a tokenizer does.",
]


batch_prompts = [f"Respond in one sentence: {s}" for s in sentences] * 2


# Time: single input


t0 = time.perf_counter()

_ = model.generate(
    **tok("Respond in one sentence: What is a tokenizer?", return_tensors="pt").to(device),
    max_new_tokens=24
)

t1 = time.perf_counter()

# Time: small batch

enc2 = tok(
    batch_prompts,
    return_tensors = "pt",
    padding=True,
    truncation=True,
    max_length=96
).to(device)

_ = model.generate(
    **enc2,
    max_new_tokens=24
)

t2 = time.perf_counter()  #t2-t1



print(f"Single input: ~{(t1-t0):.3f}s")
print(f"Small batch: ~{(t2-t1):.3f}s")

 