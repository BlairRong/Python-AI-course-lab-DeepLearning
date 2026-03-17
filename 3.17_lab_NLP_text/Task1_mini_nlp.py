#Lab: Local NLP Toolkit with Hugging Face
#This lab implements a local NLP toolkit using Hugging Face transformers pipelines, running on CPU. 
#The solution is structured into three tasks as required.


"""
Task 1: Instruction rewrite with FLAN-T5-base and sentiment analysis with DistilBERT.
Run locally on CPU.
"""

import sys
from transformers import pipeline

# ----------------------------------------------------------------------
# 1. Instruction rewrite (FLAN-T5-base model) 指令重写
# ----------------------------------------------------------------------
def rewrite_sentence(original: str) -> str:
    print("\n--- Instruction Rewrite (FLAN-T5-base) ---")
    rewrite_pipe = pipeline("text2text-generation", model="google/flan-t5-base")

    # prompt 指令 需要更明确
    prompt = (
        "Simplify the following sentence. The simplified version must be exactly one sentence, "
    "between 8 and 24 words long, and must end with a period.\n\n"
    f"Original: {user_input}\n\nSimplified:"
        ) 

    result = rewrite_pipe(
        prompt,
        max_new_tokens=64,             # 允许生成足够的长度
        num_beams=5,
        no_repeat_ngram_size=3,
        do_sample=False
    )

    rewritten = result[0]["generated_text"].strip()
    print(f"[DEBUG] Model output: {rewritten}")
    print(f"Generated: {rewritten}")   # 调试输出

    # 验证约束
    words = rewritten.split()
    word_count = len(words)
    ends_with_period = rewritten.endswith('.')
    sentence_count = rewritten.count('.')

    if sentence_count != 1 or not ends_with_period or word_count < 8 or word_count > 24:
        print(f"Constraint not satisfied: {rewritten}")
        print(f"  words: {word_count}, ends with period: {ends_with_period}, sentences: {sentence_count}")
        return None
    else:
        print(f"✓ Rewritten: {rewritten}")
        return rewritten


# ----------------------------------------------------------------------
# 2. Sentiment analysis 情感分析 (DistilBERT SST-2)
# ----------------------------------------------------------------------

def sentiment_analysis(sentences):
    """
    Perform sentiment analysis on a list of three sentences.
    Print label + score for each, then explain why no Neutral exists.
    """
    print("\n--- Sentiment Analysis (DistilBERT SST-2) ---")
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    results = sentiment_pipe(sentences)

    for sent, res in zip(sentences, results):
        print(f"Text: {sent}")
        print(f"  Label: {res['label']}, Score: {res['score']:.4f}")

"""
Now enter three sentences for sentiment analysis (one per line):
Sentence 1: this is not good 
Sentence 2: I love it but sometimes it makes me feel too sweet
Sentence 3: is so hot here maybe we should go inside 
--- Sentiment Analysis (DistilBERT SST-2) ---
Text: this is not good 
Label: NEGATIVE, Score: 0.9998
Text: I love it but sometimes it makes me feel too sweet
Label: NEGATIVE, Score: 0.9792
Text: is so hot here maybe we should go inside
Label: POSITIVE, Score: 0.9987
"""

# Explanation paragraph
# Why no Neutral? 中性
"The DistilBERT SST-2 model is fine-tuned on the Stanford Sentiment Treebank 2, "
"which contains only binary labels: positive and negative. There is no neutral class. "
"This means the model is forced to classify every input as either positive or negative, "
"even when the sentiment is truly neutral or ambiguous. Consequently, its confidence "
"scores may be high for borderline cases, but they do not reflect genuine neutrality."


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Part 1: rewrite
    user_input = input("Enter an English sentence to rewrite: ") # The movie was absolutely fantastic and I loved every minute.
    rewrite_sentence(user_input)

    # Part 2: sentiment
    print("\nNow enter three sentences for sentiment analysis (one per line):")
    s1 = input("Sentence 1: ")
    s2 = input("Sentence 2: ")
    s3 = input("Sentence 3: ")
    sentiment_analysis([s1, s2, s3])