#!/usr/bin/env python3
"""
从 infer_dpo.py --raw 输出的 CSV 中提取最终答案，生成符合 submit.csv 格式的文件。

Usage:
    python extract_answers.py <input.csv> [--output submit.csv]
"""
import argparse
import csv
import os
import re
import sys


def extract_answer(text: str) -> str:
    """
    从模型原始 CoT 输出中提取最终答案。

    支持的格式：
      ### 最终答案 3920         (同行)
      ### 最终答案\\n3920        (换行)
      答案是/答案为/答：3920
    """
    if not text or not text.strip():
        return ""

    # 1. ### 最终答案 后面跟空格/换行
    m = re.search(r'#*\s*最终答案\s*[\n\s]+(.+?)(?:\n|$)', text)
    if m:
        val = m.group(1).strip()
        if val:
            return val

    # 2. 答案是/答案为/答：
    m = re.search(r'(?:答案(?:是|为)|答)[：:]\s*(.+?)(?:\n|$)', text)
    if m:
        return m.group(1).strip()

    # 3. 最后一个 = 后面的数字
    m = re.findall(r'=\s*(\d+(?:\.\d+)?)', text)
    if m:
        return m[-1]

    # 4. 最后一段非空文本
    parts = re.split(r'[\n#]+', text)
    parts = [p.strip() for p in parts if p.strip()]
    if parts:
        return parts[-1]

    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="从 raw CoT CSV 中提取答案")
    parser.add_argument("input", help="raw CoT 的 CSV 文件路径")
    parser.add_argument("--output", "-o", default="submit.csv", help="输出文件路径（默认 submit.csv）")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: 文件不存在: {args.input}")
        sys.exit(1)

    total = 0
    with open(args.input, encoding="utf-8") as inf, \
         open(args.output, "w", encoding="utf-8") as outf:
        reader = csv.reader(inf)
        for row in reader:
            if len(row) < 2:
                continue
            total += 1
            problem_id = row[0]
            raw_text = row[1]

            answer = extract_answer(raw_text)
            outf.write(f"{problem_id},{answer}\n")

    print(f"Total: {total}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
