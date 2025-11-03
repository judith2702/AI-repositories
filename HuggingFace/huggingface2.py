# vocabulary sheet 

# ————— Core concepts —————
# - **Model**: a trained neural network that maps input text to output text/labels.
# - **Weights / checkpoint**: the large files with the model’s learned parameters.
# - **Inference** vs **training**: using a model to predict vs. updating its weights with data.

# ————— Tokenization —————
# - **Tokenizer**: converts text ↔ tokens (integers). Loaded with `AutoTokenizer`.
# - **Token / Token ID**: a subword unit represented as an integer (what the model actually reads).
# - **Vocabulary**: the set of all tokens a tokenizer knows.
# - **WordPiece**: tokenizer family used by BERT/DistilBERT (often uncased; adds special tokens).
# - **SentencePiece**: tokenizer family used by T5/FLAN & Marian/OPUS-MT (language-agnostic subwords).
# - **Detokenize**: convert token IDs back into human-readable text (optionally hiding special tokens).

# ————— Special tokens you’ll see —————
# - **[CLS]**: “classification” token at the start for BERT-style models (pooled summary for classifiers).
# - **[SEP]**: “separator/end” token at the end (and between paired sentences) for BERT-style models.
# - **</s> (EOS)**: end-of-sequence token used by T5/FLAN to signal “stop generating”.
# - **<pad>**: padding token used to equalize sequence lengths inside a batch.
# - **Pad token id**: the integer ID representing `<pad>` (e.g., 0 for many T5/FLAN models).

# ————— Tensors & shapes —————
# - **Shape [B, L]**: tensors print as `torch.Size([batch_size, sequence_length])`.
# - **Batch size (B)**: how many sequences we process at once.
# - **Sequence length (L)**: number of **tokens** after tokenization (not characters/words).
# - **attention_mask**: `[B, L]` tensor with 1=real token, 0=padding; tells the model to ignore pads.
# - **Padding**: add `<pad>` tokens so all sequences in a batch share the same length.
# - **Truncation**: cut inputs longer than a chosen `max_length` (protects CPU time/memory).

# ————— Pipelines & Auto classes —————
# - **Pipeline**: a prebuilt function (e.g., `pipeline("summarization")`) bundling tokenizer+model+decoding.
# - **Auto classes**: factory loaders that pick correct components, e.g.:
#   `AutoTokenizer.from_pretrained("google/flan-t5-base")`,
#   `AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")`.

# ————— Model families we use —————
# - **FLAN-T5 (encoder–decoder / seq2seq)**: instruction-tuned T5 (great for text-to-text tasks).
# - **DistilBERT (encoder-only)**: lightweight BERT for classification (expects [CLS]…[SEP]).
# - **DistilBART-CNN (encoder–decoder)**: distilled BART fine-tuned for news summarization.
# - **OPUS-MT / Marian (encoder–decoder)**: translation models (e.g., EN→SV), SentencePiece-based.
# - **Distillation**: compressing a large “teacher” into a smaller, faster “student” model.

# ————— Generation (decoding) —————
# - **`.generate(...)`**: turns input token IDs into new output token IDs.
# - **Greedy**: always pick the top next token (deterministic, safe).
# - **Beam search (`num_beams`)**: explore several high-probability paths; often more fluent; can repeat.
# - **Sampling (`do_sample=True`)**: add randomness; tune with **temperature** and **top_p** for creativity.
# - **`max_new_tokens`**: cap on how many **generated** tokens to produce.
# - **`no_repeat_ngram_size`**: discourages repeating short phrases (reduces loops/copypasta).
# - **`length_penalty`**: bias beams toward shorter/longer outputs.
# - **`skip_special_tokens=True`**: hide special tokens when decoding to text.

# ————— Performance & timing —————
# - **`time.perf_counter()`**: high-resolution wall-clock timer for quick benchmarks.
# - **Latency vs throughput**: time per call vs items per second (batching improves throughput).
# - **Warm-up / caches**: first call is slower; later calls are faster (weights loaded, kernels warmed).
# - **Scaling with L²**: attention cost grows roughly with the square of sequence length—keep prompts short on CPU.
 
##############PART 5#################################
from transformers import AutoTokenizer

Text = "transformers make local demos easy. python is great for teaching."

tok_flan = AutoTokenizer.from_pretrained("google/flan-t5-base")

ids_flan = tok_flan.encode(Text)

print("\n-----------Tokens: Flan-T5(SentencePiece)----------\n")

print("Token IDs : ",ids_flan[:20], ".........")  #1 (last ID)
print("Decoded: ",tok_flan.decode(ids_flan))#´<s>´  EOS marker

#######DISTILBERT (wordpiece tokenizer)
tok_bert = AutoTokenizer.from_pretrained("distilbert-base-uncased") #[CLS]=101 [SEP]=102
ids_bert = tok_bert.encode(Text)

print("\n-----------Tokens: DistilBERT (wordPiece)----------\n")


print("Token IDs : ",ids_bert[:20], ".........") 

print("Decoded: ", tok_bert.decode(ids_bert))


#############part 6#############################
##########from pipeline to autoclasses################

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

device = torch.device("cpu")
model_id = "google/flan-t5-base"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

prompt = (
    "Rewite the sentence in simple English.Output with one sentence,end with a period.\n"
    "sentence: 'Transformers let us try modern AI models locally for teaching.'\n"
    "output: "
)

enc = tok(prompt,return_tensors = "pt" ).to(device)


print("\n------Shapes (input)------\n")
print("input_ids: ",enc.input_ids.shape)
print("attention_masks: ",enc.attention_mask.shape)


with torch.no_grad():
    out_ids = model.generate(
        **enc,
        max_new_tokens = 32,
        num_beams = 5,
        no_repeat_ngram_size =3,
        do_sample = False
    )


print("\n--------SHAPES(OUPTPUT)------\n")
print("out_ids:",out_ids.shape)

out_text = tok.decode(out_ids[0], skip_special_tokens = True)
print("\n ----------    MANUAL GENERATION(FLAN T5).-----------\n")
print(out_text)


####################part7###############
################3batching ,padding ,truncation#################
