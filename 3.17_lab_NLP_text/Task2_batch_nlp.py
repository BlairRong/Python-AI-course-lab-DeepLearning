#Goal: batch processing批量处理 + decoding comparison解码比较 + basic measurement基本测量
# Extend the tool to process a batch of inputs from a file, compare beam search vs sampling, and export results as CSV.

#I choose option A: rewrite with FLAN-T5-base (the same task as in Task 1).


"""
Task 2: 
Batch processing 批量处理 with two decoding modes解码模式.
Accepts --infile lines.txt接受这个格式的输入, runs beam search 运行束搜索解码模式 and sampling 采样解码模式,
exports results.csv
prints a mini-report.
"""

import argparse
import csv
from transformers import pipeline

# ----------------------------------------------------------------------
# Constraint validator 约束验证器
# ----------------------------------------------------------------------
def check_constraints(text: str) -> (bool, str):
    """
    Returns (passed, note) where note is a short reason if failed.
    Constraints: exactly one sentence, ends with period, 8-24 words.
    """
    words = text.split()
    word_count = len(words)
    ends_with_period = text.endswith('.')
    sentence_count = text.count('.')   # simple, assumes no other punctuation

    if sentence_count != 1:
        return False, f"sentence count = {sentence_count}"
    if not ends_with_period:
        return False, "no period at end"
    if word_count < 8:
        return False, f"too short ({word_count} words)"
    if word_count > 24:
        return False, f"too long ({word_count} words)"
    return True, ""


# ----------------------------------------------------------------------
# Generation function 生成函数
# ----------------------------------------------------------------------
def generate(prompt: str, pipe, do_sample: bool, temperature=1.0, top_p=1.0):
    """
    Generate text using the given pipeline and decoding strategy.
    Returns generated string. 使用给定的管道和解码策略生成文本。返回生成的字符串。
    """
    kwargs = {
        "max_new_tokens": 64,
        "do_sample": do_sample,
    }
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    else:
        kwargs["num_beams"] = 5
        kwargs["no_repeat_ngram_size"] = 3

    result = pipe(prompt, **kwargs)
    return result[0]["generated_text"].strip()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True, help="Input file with one sentence per line")
    args = parser.parse_args()

    # Read input lines
    with open(args.infile, 'r', encoding='utf-8') as f:
        inputs = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(inputs)} inputs from {args.infile}")

    # Initialize pipeline (same model as Task 1)
    rewrite_pipe = pipeline("text2text-generation", model="google/flan-t5-base")

    # Prepare results list
    results = []

    # Define decoding modes
    modes = [
        {"name": "beam", "do_sample": False},
        {"name": "sampling", "do_sample": True, "temperature": 0.8, "top_p": 0.9}
    ]

    # For each input and each mode, generate and validate
    for inp in inputs:
        prompt =f"Simplify the following sentence into one simple sentence (8-24 words, ending with a period):\n\n{inp}\n\nSimplified:"

        for mode in modes:
            gen_text = generate(prompt, rewrite_pipe, mode["do_sample"],
                                temperature=mode.get("temperature", 1.0),
                                top_p=mode.get("top_p", 1.0))

            # token count approximate: split by whitespace
            tokens_out = len(gen_text.split())
            passed, note = check_constraints(gen_text)

            results.append({
                "input": inp,
                "output": gen_text,
                "decoding": mode["name"],
                "tokens_out": tokens_out,
                "constraint_passed": passed,
                "notes": note if not passed else ""
            })

    # Write CSV
    with open("results.csv", "w", newline='', encoding='utf-8') as f:
        fieldnames = ["input", "output", "decoding", "tokens_out", "constraint_passed", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("Results written to results.csv")

    # Mini‑report
    print("\n=== Mini-Report ===")
    for mode_name in ["beam", "sampling"]:
        mode_results = [r for r in results if r["decoding"] == mode_name]
        total = len(mode_results)
        passed = sum(1 for r in mode_results if r["constraint_passed"])
        avg_tokens = sum(r["tokens_out"] for r in mode_results) / total if total else 0
        pass_rate = passed / total * 100 if total else 0
        print(f"\nDecoding: {mode_name}")
        print(f"  Pass rate: {pass_rate:.1f}% ({passed}/{total})")
        print(f"  Average tokens_out: {avg_tokens:.1f}")


if __name__ == "__main__":
    main()

#run: python Task2_batch_nlp.py --infile lines.txt

# Reflection
"Reflection on quality vs control:"
"Beam search tends to produce more conservative, often shorter rewrites that closely follow the prompt structure. " "束搜索往往会产生更保守、通常更短的重写版本，这些版本紧密遵循提示结构。"
"Sampling introduces variability and can occasionally yield more creative outputs, " " 采样引入了变异性，有时可以产生更具创意的成果，"
"but at the cost of lower constraint satisfaction. The trade-off is between reproducibility (beam) and diversity (sampling)." " 但代价是降低了对约束条件的满足程度。这需要在可复现性（束搜索）和多样性（采样）之间进行权衡 "
