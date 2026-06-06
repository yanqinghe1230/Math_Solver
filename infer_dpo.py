import argparse
import json
import os
import sys
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入答案提取工具（与 generate_candidates.py 共用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dpo_math.answer_verifier import extract_final_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./output/Qwen-DPO",
                        help="模型目录（默认根目录为最优模型，也可指定 checkpoint-XXX 子目录）")
    parser.add_argument("--test-path", default="test.json")
    parser.add_argument("--output-path", default="submit_dpo.csv")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批量推理大小（默认 32）")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="最大生成 token 数（安全上限，正常情况模型生成完会自动停）")
    parser.add_argument("--raw", action="store_true",
                        help="输出原始 CoT 文本（不提取答案）")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Batch size: {args.batch_size}, Max new tokens: {args.max_new_tokens}")

    # 加载模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True,
    )
    # 左填充用于生成（避免 padding 影响自回归生成）
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    # 加载测试数据
    with open(args.test_path, encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Test samples: {len(test_data)}")

    # 统计提取失败数
    extraction_failures = 0

    # 批量推理
    with open(args.output_path, "w", encoding="utf-8") as out_file:
        for i in tqdm(range(0, len(test_data), args.batch_size), desc="Inference"):
            batch = test_data[i:i + args.batch_size]

            # 构建批量 messages
            batch_messages = []
            for row in batch:
                batch_messages.append([
                    {"role": "system", "content": row["instruction"]},
                    {"role": "user", "content": row["question"]},
                ])

            # 批量 tokenize
            texts = tokenizer.apply_chat_template(
                batch_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)

            # 批量生成
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # 提取新生成的 token 并解码
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # 写结果
            for row, response in zip(batch, responses):
                problem_id = row["id"]
                if args.raw:
                    # 输出原始 CoT
                    output = response.replace("\n", " ")
                else:
                    # 从 CoT 中提取最终答案
                    answer = extract_final_answer(response)
                    if answer is None:
                        extraction_failures += 1
                        # 回退：取最后一行非空文本
                        lines = [l.strip() for l in response.split("\n") if l.strip()]
                        answer = lines[-1] if lines else response.replace("\n", " ")
                    output = answer
                out_file.write(f"{problem_id},{output}\n")

    if not args.raw and extraction_failures > 0:
        print(f"Warning: {extraction_failures}/{len(test_data)} answers failed extraction, used fallback")


if __name__ == "__main__":
    main()
