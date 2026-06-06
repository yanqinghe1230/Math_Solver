#!/usr/bin/env python3
"""
从 infer_dpo.py --raw 输出的 CSV 中提取最终答案，生成符合 submit.csv 格式的文件。

Usage:
    python extract_answers.py <input.csv> [--output submit.csv]
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dpo_math.answer_verifier import extract_final_answer


def main():
    parser = argparse.ArgumentParser(description="从 raw CoT CSV 中提取答案")
    parser.add_argument("input", help="raw CoT 的 CSV 文件路径")
    parser.add_argument("--output", "-o", default="submit.csv", help="输出文件路径（默认 submit.csv）")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: 文件不存在: {args.input}")
        sys.exit(1)

    total = 0
    extracted = 0
    fallback = 0

    with open(args.input, encoding="utf-8") as inf, \
         open(args.output, "w", encoding="utf-8") as outf:
        reader = csv.reader(inf)
        for row in reader:
            if len(row) < 2:
                continue
            total += 1
            problem_id = row[0]
            raw_text = row[1]

            answer = extract_final_answer(raw_text)
            if answer is not None:
                extracted += 1
            else:
                fallback += 1
                # 回退：取最后一行非空文本
                lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
                answer = lines[-1] if lines else raw_text

            outf.write(f"{problem_id},{answer}\n")

    print(f"Total: {total}, Extracted: {extracted}, Fallback: {fallback}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
