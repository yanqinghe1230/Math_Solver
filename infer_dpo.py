import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


test_json_path = "test.json"

with open(test_json_path, encoding="utf-8") as file:
    test_data = json.load(file)

# DPO 模型是完整权重（LoRA 已在训练前 merge），直接加载即可，不需要 PeftModel
model_path = "./output/Qwen-DPO"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

output_path = "submit_dpo.csv"
with open(output_path, "w", encoding="utf-8") as file:
    for row in tqdm(test_data):
        instruction = row["instruction"]
        question = row["question"]
        problem_id = row["id"]

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": question},
        ]
        response = predict(messages, model, tokenizer)
        response = response.replace("\n", " ")
        file.write(f"{problem_id},{response}\n")
