# - **Model**: a trained neural network that maps input text to output text or labels.
# - **Weights / checkpoint**: the learned parameters of that model (the big files we download).
# - **Tokenizer**: converts text → **tokens** (subwords/IDs) for the model, and back again.
# - **Tokens**: pieces of text (not necessarily full words). Length limits and speeds are
#   measured in tokens, so “max_new_tokens” is about subwords, not words.
# - **Transformer (architecture)**: the neural network design (self-attention, etc.) used
#   by most state-of-the-art language models.
# - **Transformers (library)**: the Hugging Face Python library we import to use models.
# - **Pipeline**: a prebuilt function (e.g., `pipeline("sentiment-analysis")`) that handles
#   tokenizer + model + decoding for a common task.
# - **Inference** vs **training**:
#   * Inference = using a trained model to make predictions.
#   * Training/fine-tuning = updating weights with data.
# - **Encoder–decoder** vs **decoder-only**:
#   * Encoder–decoder (e.g., T5/BART): good for text-to-text tasks (summarize/translate).
#   * Decoder-only (e.g., GPT-style): good for next-token text continuation.
# - **Deterministic decoding** (beam search, no sampling): stable, repeatable outputs.
# - **Sampling** (temperature, top-p): more creative/varied but less predictable.



# "How a pipeline call works (mental model):"
#   1) Your input text → **tokenizer** → token IDs.
#   2) Token IDs → **model** (forward pass on CPU) → output logits.
#   3) **Decoding** turns logits into text (beam search or sampling).
#   4) Output tokens → detokenize → final string.


#Install library:
#Transformers a specific version
#Accelerate
#pip install "transformers==4.41.1" accelerate
#pip install torch --index-url https://download.pytorch.org/whl/cpu
#pip install sentencepiece






#PART 1 - First pipeline/model: Text Generation with FLAN-T5-base(local, CPU)
#pipeline is a high‑level API provided by the Hugging Face transformers library. Its goal is to make it incredibly easy to use state‑of‑the‑art models for a wide variety of natural language processing (NLP) tasks without writing boilerplate code.
#pipeline 是 Hugging Face transformers 库提供的高级 API。它的目标是让用户能够极其轻松地使用最先进的模型来完成各种自然语言处理 (NLP) 任务，而无需编写样板代码。

"""
from transformers import pipeline

gen = pipeline("text2text-generation", model="google/flan-t5-base")

prompt = (
    "Produce exactly ONE family-friendly joke. "
    "One sentence, 10-20 words, end with a period."
)

print("\n--TEXT GENERATION (FLAN-T5-base, deterministic) ---\n")
print(gen(
    prompt,
    max_new_tokens=32,
    num_beams=5,
    no_repeat_ngram_size=3,
    do_sample=False
)[0]["generated_text"])
"""





#PART 2 - second pipeline/model: Sentiment Analysis(DistilBERT model)

#Positive
#Negative

#borderline/ambiguous

from transformers import pipeline

sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

#under the hood: tokenization -> DistilBERT forward pass -> sofemax -> label + score

print(("\n--- SENTIMENT ANALYSIS ---\n"))


examples = [
    "I absolutly love coding in Python!",
    "This bug is driving me crazy.",
    "It's okay, not great, not terrible."
]

for text in examples:
    result = sentiment(text)[0]
    print(f"Text: {text}\n→ label: {result['label']}, Score:{result['score']:.3f}\n ")

#--- SENTIMENT ANALYSIS ---

#Text: I absolutly love coding in Python!
#→ label: POSITIVE, Score:0.922

#Text: This bug is driving me crazy.
#→ label: NEGATIVE, Score:0.999

#Text: It's okay, not great, not terrible.
#→ label: POSITIVE, Score:0.994




"""
#PART 3 - model/Summarization(local, CPU) with DistilBART CNN

from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

article = (
    "Python is a popular programming language known for readability and a rich ecosystem. "
    "Hugging Face Transformers lets developers run state-of-the-art AI models locally. "
    "With pipelines, tasks like text generation, sentiment analysis, and summarization "
    "become easy to prototype."
)

summary = summarizer(article, max_length = 28, min_length=15, do_sample=False)[0]["summary_text"]

print("Original:", article)
print("\nSummary:", summary)

#Original: Python is a popular programming language known for readability and a rich ecosystem. Hugging Face Transformers lets developers run state-of-the-art AI models locally. With pipelines, tasks like text generation, sentiment analysis, and summarization become easy to prototype.

#Summary:  Python is a popular programming language known for readability and a rich ecosystem . Hugging Face Transformers lets developers run state-of

"""






"""
#PART 4 - Translation (EN SV) with OPUS-MT (local, CPU)

from transformers import pipeline

print("\n--- Translation (EN to SV) ---\n")

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-sv")

english = "My name is siying rong."
swedish = translator(english)[0]["translation_text"]

print("EN:", english)
print("SV:", swedish)

#EN: Transformers pipelines make it simple to try models locally.
#SV: Transformatorer rörledningar gör det enkelt att prova modeller lokalt. 
# there is a problem only translate word by word 
"""