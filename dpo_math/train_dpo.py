#!/usr/bin/env python3
"""
DPO fine-tuning for math solver using TRL DPOTrainer.

Adapted from dpo/3-finetune_with_dpo.ipynb reference implementation.
Fine-tunes the already SFT-ed Qwen2.5-0.5B model on preference pairs
to align reasoning towards correct CoT answers.

Usage:
    python train_dpo.py

Configuration: edit the paths and training args in the CONFIG section below.
"""

import json
import os
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

# ============================================================
# CONFIG — edit these to match your environment
# ============================================================

# Base model (same as used in baseline SFT)
BASE_MODEL_PATH = "./Qwen/Qwen2.5-0.5B-Instruct"

# Baseline SFT LoRA checkpoint to load and merge before DPO
BASELINE_CHECKPOINT_PATH = "./output/Qwen/checkpoint-3750"

# DPO preference pairs
DPO_DATA_PATH = "dpo_pairs.jsonl"

# Output directory for DPO model
OUTPUT_DIR = "./output/Qwen-DPO"

# System instruction (same as train_cot.json; used during inference)
# Set to None to omit system message from DPO prompts
SYSTEM_INSTRUCTION = (
    "这是小学数学1-6年级的校内题目，无需进行分析，请直接输出数字答案，不带单位。"
)

# Train/validation split ratio
VALID_SPLIT = 0.1

# Whether to merge LoRA before DPO (recommended: True)
MERGE_LORA_BEFORE_DPO = True

# Maximum number of DPO pairs to use (None = all)
MAX_PAIRS = None

# ============================================================
# DPO Training Arguments
# ============================================================

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    logging_steps=25,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,          # lower than SFT; DPO is sensitive to LR
    beta=0.1,                    # DPO temperature; higher = closer to reference model
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    report_to="none",            # change to "wandb" or "swanlab" if needed
    remove_unused_columns=False,
    bf16=torch.cuda.is_available(),
    fp16=False,
    gradient_checkpointing=True,
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
)


# ============================================================
# Data Loading & Preparation
# ============================================================

def load_dpo_pairs(path: str, max_pairs: int = None) -> list[dict]:
    """Load DPO pairs from a JSONL file."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    if max_pairs is not None and max_pairs > 0:
        pairs = pairs[:max_pairs]
    print(f"Loaded {len(pairs)} DPO pairs from {path}")
    return pairs


def add_system_message(example: dict, instruction: str) -> dict:
    """
    Prepend a system message to the prompt for consistency with the
    baseline SFT training and inference chat template.

    Before: prompt = [{"role": "user", "content": "..."}]
    After:  prompt = [{"role": "system", "content": "..."},
                      {"role": "user", "content": "..."}]
    """
    system_msg = {"role": "system", "content": instruction}
    example["prompt"] = [system_msg] + example["prompt"]
    return example


def build_dataset(pairs: list[dict], instruction: str = None) -> DatasetDict:
    """
    Convert raw DPO pairs to a HuggingFace DatasetDict with train/valid split.

    Each pair already has the correct DPO format:
        {"prompt": [...], "chosen": [...], "rejected": [...]}
    """
    # Keep only the DPO columns + metadata (metadata is ignored by DPOTrainer)
    records = []
    for p in pairs:
        rec = {
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        # Optionally add system message
        if instruction:
            rec = add_system_message(rec, instruction)
        records.append(rec)

    dataset = Dataset.from_list(records)

    # Shuffle and split
    dataset = dataset.shuffle(seed=42)
    split = dataset.train_test_split(test_size=VALID_SPLIT, seed=42)

    dataset_dict = DatasetDict({
        "train": split["train"],
        "valid": split["test"],
    })

    print(f"Train: {len(dataset_dict['train'])} pairs, Valid: {len(dataset_dict['valid'])} pairs")
    return dataset_dict


# ============================================================
# Model Loading
# ============================================================

def load_model_and_tokenizer(
    base_model_path: str,
    checkpoint_path: str = None,
    merge_lora: bool = True,
):
    """
    Load the base model, optionally apply + merge LoRA adapter, return model & tokenizer.

    Args:
        base_model_path: Path to Qwen2.5-0.5B-Instruct base model.
        checkpoint_path: Path to LoRA adapter checkpoint, or None to skip.
        merge_lora: If True, merge LoRA weights into the base model before returning.

    Returns:
        (model, tokenizer)
    """
    print(f"Loading base model from: {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=False,
        trust_remote_code=True,
    )
    # Qwen tokenizer doesn't have a default pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    # Load LoRA adapter from baseline SFT checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading LoRA adapter from: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)

        if merge_lora:
            print("Merging LoRA weights into base model...")
            model = model.merge_and_unload()
            print("LoRA merged successfully.")
    else:
        if checkpoint_path:
            print(f"WARNING: Checkpoint not found at {checkpoint_path}, using base model only.")

    model.enable_input_require_grads()  # Required for gradient checkpointing

    return model, tokenizer


# ============================================================
# Main
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Output directory: {OUTPUT_DIR}")

    # 1. Load and prepare data
    pairs = load_dpo_pairs(DPO_DATA_PATH, max_pairs=MAX_PAIRS)
    dataset = build_dataset(pairs, instruction=SYSTEM_INSTRUCTION)

    # 2. Load model (base + merged LoRA from baseline SFT)
    model, tokenizer = load_model_and_tokenizer(
        base_model_path=BASE_MODEL_PATH,
        checkpoint_path=BASELINE_CHECKPOINT_PATH,
        merge_lora=MERGE_LORA_BEFORE_DPO,
    )

    # 3. Initialize DPOTrainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
    )

    # 4. Train
    print("\nStarting DPO training...")
    trainer.train()

    # 5. Save final model
    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
