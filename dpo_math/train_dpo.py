#!/usr/bin/env python3
"""
DPO fine-tuning for math solver using TRL DPOTrainer.

Adapted from dpo/3-finetune_with_dpo.ipynb reference implementation.
Fine-tunes the already SFT-ed Qwen2.5-0.5B model on preference pairs
to align reasoning towards correct CoT answers.

Usage:
    # Default: use only API-generated pairs (high diversity, ~284 pairs)
    python train_dpo.py

    # Use API + low-similarity fallback pairs
    python train_dpo.py --filter-source all --max-similarity 0.7

    # Use everything (not recommended)
    python train_dpo.py --filter-source all --max-similarity 1.0

Configuration: edit the paths in CONFIG section or use CLI flags.
"""

import argparse
import difflib
import json
import os
import sys
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

# ============================================================
# CONFIG — edit these to match your environment
# ============================================================

BASE_MODEL_PATH = "./Qwen/Qwen2.5-0.5B-Instruct"
BASELINE_CHECKPOINT_PATH = "./output/Qwen/checkpoint-3750"
DPO_DATA_PATH = "dpo_pairs.jsonl"
OUTPUT_DIR = "./output/Qwen-DPO"

SYSTEM_INSTRUCTION = (
    "这是小学数学1-6年级的校内题目，无需进行分析，请直接输出数字答案，不带单位。"
)

VALID_SPLIT = 0.1
MERGE_LORA_BEFORE_DPO = True


# ============================================================
# Data Loading & Filtering
# ============================================================

def compute_similarity(chosen_text: str, rejected_text: str) -> float:
    """SequenceMatcher ratio: 0.0 = completely different, 1.0 = identical."""
    return difflib.SequenceMatcher(None, chosen_text, rejected_text).ratio()


def load_dpo_pairs(
    path: str,
    filter_source: str = "api",
    max_similarity: float = 0.85,
    max_pairs: int = None,
) -> list[dict]:
    """
    Load and filter DPO pairs from JSONL.

    Args:
        path: Path to dpo_pairs.jsonl.
        filter_source: "api" (only API-generated), "fallback" (only programmatic),
                       "all" (everything).
        max_similarity: Discard pairs with similarity >= this threshold.
                        API pairs avg ~0.24, fallback pairs avg ~0.93.
                        Default 0.85 keeps diverse pairs, drops near-identical ones.
        max_pairs: Hard cap on total pairs.

    Returns:
        Filtered list of valid DPO pair dicts.
    """
    raw = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    # Filter: must have prompt/chosen/rejected
    valid = [p for p in raw if all(k in p for k in ("prompt", "chosen", "rejected"))]

    # Filter by source
    if filter_source != "all":
        valid = [p for p in valid if p.get("metadata", {}).get("source") == filter_source]

    # Filter by similarity
    filtered = []
    dropped_similar = 0
    for p in valid:
        c = p["chosen"][0]["content"]
        r = p["rejected"][0]["content"]
        sim = compute_similarity(c, r)
        if sim < max_similarity:
            p["_similarity"] = round(sim, 4)
            filtered.append(p)
        else:
            dropped_similar += 1

    # Print stats
    source_counts = {}
    sims = []
    for p in filtered:
        src = p.get("metadata", {}).get("source", "?")
        source_counts[src] = source_counts.get(src, 0) + 1
        sims.append(p.get("_similarity", 0))

    print(f"Raw records: {len(raw)}, Valid DPO triples: {len(valid)}")
    print(f"After source filter ({filter_source}): {len(filtered) + dropped_similar}")
    print(f"After similarity filter (<{max_similarity}): {len(filtered)} "
          f"(dropped {dropped_similar} near-identical pairs)")
    if sims:
        print(f"Similarity range: {min(sims):.3f} - {max(sims):.3f}, avg: {sum(sims)/len(sims):.3f}")
    print(f"Source breakdown: {source_counts}")

    # Hard cap
    if max_pairs and len(filtered) > max_pairs:
        filtered = filtered[:max_pairs]
        print(f"Capped to {max_pairs} pairs")

    return filtered


def add_system_message(example: dict, instruction: str) -> dict:
    """Prepend system message to prompt for consistency with inference format."""
    system_msg = {"role": "system", "content": instruction}
    example["prompt"] = [system_msg] + example["prompt"]
    return example


def build_dataset(pairs: list[dict], instruction: str = None) -> DatasetDict:
    """Convert filtered pairs to HuggingFace DatasetDict with train/valid split."""
    records = []
    for p in pairs:
        rec = {
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        if instruction:
            rec = add_system_message(rec, instruction)
        records.append(rec)

    dataset = Dataset.from_list(records)
    dataset = dataset.shuffle(seed=42)
    split = dataset.train_test_split(test_size=VALID_SPLIT, seed=42)

    dataset_dict = DatasetDict({"train": split["train"], "valid": split["test"]})
    print(f"Train: {len(dataset_dict['train'])} pairs, Valid: {len(dataset_dict['valid'])} pairs")
    return dataset_dict


# ============================================================
# Model Loading
# ============================================================

def load_model_and_tokenizer(base_model_path, checkpoint_path=None, merge_lora=True):
    print(f"Loading base model: {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, use_fast=False, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading LoRA adapter: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        if merge_lora:
            print("Merging LoRA into base model...")
            model = model.merge_and_unload()
    elif checkpoint_path:
        print(f"WARNING: checkpoint not found at {checkpoint_path}")

    model.enable_input_require_grads()
    return model, tokenizer


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DPO fine-tuning for math solver")
    parser.add_argument("--filter-source", default="api",
                        choices=["api", "fallback", "all"],
                        help="Which source of pairs to use (default: api)")
    parser.add_argument("--max-similarity", type=float, default=0.85,
                        help="Drop pairs with similarity >= this (default: 0.85)")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Hard cap on total pairs")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Training epochs (default: 1)")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate (default: 5e-6)")
    parser.add_argument("--beta", type=float, default=0.3,
                        help="DPO beta: higher = more conservative (default: 0.3)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Model output directory")
    parser.add_argument("--base-model", default=BASE_MODEL_PATH)
    parser.add_argument("--checkpoint", default=BASELINE_CHECKPOINT_PATH)
    parser.add_argument("--data-path", default=DPO_DATA_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show data stats without training")
    args = parser.parse_args()

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Output: {args.output_dir}")
    print(f"Filter: source={args.filter_source}, max_similarity={args.max_similarity}")
    print(f"Training: epochs={args.epochs}, lr={args.lr}, beta={args.beta}")

    # 1. Load + filter data
    pairs = load_dpo_pairs(
        args.data_path,
        filter_source=args.filter_source,
        max_similarity=args.max_similarity,
        max_pairs=args.max_pairs,
    )

    if len(pairs) == 0:
        print("ERROR: No pairs remain after filtering. Relax --filter-source or --max-similarity.")
        sys.exit(1)

    if args.dry_run:
        print("\nDry run complete. Use --dry-run False to train.")
        return

    dataset = build_dataset(pairs, instruction=SYSTEM_INSTRUCTION)

    # 2. Load model
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.checkpoint, merge_lora=MERGE_LORA_BEFORE_DPO,
    )

    # Free GPU memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f} GiB allocated, "
              f"{torch.cuda.memory_reserved()/1024**3:.1f} GiB reserved")

    # 3. DPO config
    training_args = DPOConfig(
        output_dir=args.output_dir,
        logging_steps=10,
        per_device_train_batch_size=2,   # small batch: DPO keeps policy + ref model in memory
        per_device_eval_batch_size=2,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none",
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        gradient_accumulation_steps=8,   # compensate smaller batch: effective batch = 2 × 8 = 16
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    # 4. Train
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
    )

    print(f"\nStarting DPO training with {len(dataset['train'])} pairs...")
    trainer.train()

    # 5. Save
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
