#Goal: Add a simple guardrail layer, retry logic, and neutral band for sentiment.


"""
Task 3: Guardrails, retry policy, and neutral band for sentiment. 情感分析的防护机制、重试策略和中立区间。
Supports three tasks: rewrite, summarization, sentiment.支持三项任务：重写、摘要和情感分析。
"""

import argparse
import csv
from transformers import pipeline

# ----------------------------------------------------------------------
# Constraint validator约束验证器 (same as Task 2)
# ----------------------------------------------------------------------
def check_constraints(text: str) -> (bool, str):
    words = text.split()
    word_count = len(words)
    ends_with_period = text.endswith('.')
    sentence_count = text.count('.')
    
    if sentence_count != 1:
        return False, f"sentence count = {sentence_count}"
    if not ends_with_period:
        return False, "no period"
    if word_count < 8:
        return False, f"too short ({word_count})"
    if word_count > 24:
        return False, f"too long ({word_count})"
    return True, ""


# ----------------------------------------------------------------------
# Generation with retry: Beam search vs Sampling(different temperature/top-p)
# ----------------------------------------------------------------------
def generate_with_retry(prompt, pipe, max_retries=1):
    """
    First try beam search. If constraints fail, retry with sampling.
    Returns (output, decoding_used, retry_fixed, tokens_out, passed, note)
    """
    # Beam attempt
    beam_out = pipe(prompt,
                    max_new_tokens=64,
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    do_sample=False)[0]["generated_text"].strip()
    passed, note = check_constraints(beam_out)
    if passed:
        return beam_out, "beam", False, len(beam_out.split()), True, ""

    # Retry with sampling if failed
    if max_retries > 0:
        sampling_out = pipe(prompt,
                            max_new_tokens=64,
                            do_sample=True,
                            temperature=0.8, #new add 
                            top_p=0.9)[0]["generated_text"].strip()
        passed2, note2 = check_constraints(sampling_out)
        return sampling_out, "sampling (retry)", True, len(sampling_out.split()), passed2, note2

    return beam_out, "beam", False, len(beam_out.split()), False, note



# ----------------------------------------------------------------------
# Sentiment with neutral band 中性情绪
# ----------------------------------------------------------------------
def sentiment_with_neutral(text, pipe, threshold=0.1):
    """
    Run sentiment, then map scores within ±threshold of 0.5 to NEUTRAL.
    Returns (raw_label, raw_score, neutral_label)运行情感分析，然后将得分在 0.5 的 ± 阈值范围内的分数映射到“中立”。
    """
    res = pipe(text)[0]
    raw_label = res['label']
    raw_score = res['score']
    # Note: distilbert returns label 'POSITIVE' or 'NEGATIVE'
    # Convert score: for POSITIVE it's the probability of positive; for NEGATIVE it's probability of negative
    # To decide neutrality, we can look at the raw score; if it's near 0.5, it's uncertain.
    # We'll treat score as confidence for the predicted class.
    # A simple approach: if score is between 0.5-threshold and 0.5+threshold, mark neutral.
    if abs(raw_score - 0.5) <= threshold:
        neutral_label = "NEUTRAL"
    else:
        neutral_label = raw_label
    return raw_label, raw_score, neutral_label



# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True, help="Input file with one sentence per line")
    parser.add_argument("--task", choices=["rewrite", "summarize", "sentiment"], required=True)
    args = parser.parse_args()

    with open(args.infile, 'r', encoding='utf-8') as f:
        inputs = [line.strip() for line in f if line.strip()]

#rewrite重写
    if args.task == "rewrite": 
        pipe = pipeline("text2text-generation", model="google/flan-t5-base")
        prompt_template = (
            "Rewrite the following sentence in simpler English. "
            "The output must be exactly one sentence, between 8 and 24 words long, "
            "and must end with a period. Sentence: {}"
        )
        results = []
        for inp in inputs:
            prompt = prompt_template.format(inp)
            out, dec, fixed, tokens, passed, note = generate_with_retry(prompt, pipe)
            results.append({
                "input": inp,
                "output": out,
                "decoding": dec,
                "retry_fixed": fixed,
                "tokens_out": tokens,
                "constraint_passed": passed,
                "notes": note
            })

#summarize单句摘要
    elif args.task == "summarize":
        pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        # For summarization we want a one-sentence summary (2-3 sentences input, output one sentence)
        # Adjust constraints: one sentence, 8-24 words, period.
        results = []
        for inp in inputs:
            # Use beam first
            beam_out = pipe(inp,
                            max_length=28, min_length=15,
                            do_sample=False)[0]["summary_text"].strip()
            passed, note = check_constraints(beam_out)
            if passed:
                out, dec, fixed, tokens = beam_out, "beam", False, len(beam_out.split())
            else:
                # retry with sampling
                samp_out = pipe(inp,
                                max_length=28, min_length=15,
                                do_sample=True, temperature=0.8, top_p=0.9)[0]["summary_text"].strip()
                passed2, note2 = check_constraints(samp_out)
                out, dec, fixed, tokens = samp_out, "sampling (retry)", True, len(samp_out.split())
                note = note2 if not passed2 else ""
            results.append({
                "input": inp,
                "output": out,
                "decoding": dec,
                "retry_fixed": fixed,
                "tokens_out": tokens,
                "constraint_passed": passed,
                "notes": note
            })

#sentiment情感分析
    elif args.task == "sentiment":
        pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        results = []
        for inp in inputs:
            raw_label, raw_score, neutral_label = sentiment_with_neutral(inp, pipe, threshold=0.1)
            results.append({
                "input": inp,
                "raw_label": raw_label,
                "raw_score": round(raw_score, 4),
                "neutral_label": neutral_label
            })

    # Write CSV
    if args.task in ["rewrite", "summarize"]:
        with open("generation_results.csv", "w", newline='', encoding='utf-8') as f:
            fieldnames = ["input", "output", "decoding", "retry_fixed", "tokens_out", "constraint_passed", "notes"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print("Generation results saved to generation_results.csv")


        # Metrics block
        total = len(results)
        beam_only = [r for r in results if r["decoding"] == "beam"]
        retry_fixed = [r for r in results if r["retry_fixed"]]
        passed = [r for r in results if r["constraint_passed"]]
        print("\n=== Final Metrics ===")
        print(f"Constraint pass rate (beam first): {len(beam_only)/total*100:.1f}%")
        print(f"Retry fixed rate: {len(retry_fixed)/total*100:.1f}%")
        print(f"Overall pass rate (after retry): {len(passed)/total*100:.1f}%")
        avg_tokens = sum(r["tokens_out"] for r in results) / total
        print(f"Average output tokens: {avg_tokens:.1f}")

    else:  # sentiment
        with open("sentiment_results.csv", "w", newline='', encoding='utf-8') as f:
            fieldnames = ["input", "raw_label", "raw_score", "neutral_label"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print("Sentiment results saved to sentiment_results.csv")

        # Distribution before/after neutral band
        raw_counts = {"POSITIVE": 0, "NEGATIVE": 0}
        neutral_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        for r in results:
            raw_counts[r["raw_label"]] += 1
            neutral_counts[r["neutral_label"]] += 1
        print("\n=== Sentiment Distribution ===")
        print("Raw labels:", raw_counts)
        print("After neutral band (threshold ±0.1):", neutral_counts)

if __name__ == "__main__":
    main()


#run:
## For rewrite task with retry
#python Task3_advanced_nlp.py --task rewrite --infile advanced_lines.txt

# For summarization task
#python Task3_advanced_nlp.py --task summarize --infile advanced_summaries.txt

# For sentiment with neutral band
#python Task3_advanced_nlp.py --task sentiment --infile advanced_lines.txt