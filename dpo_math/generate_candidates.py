#!/usr/bin/env python3
"""
DPO Candidate Answer Generator.

Generates variant (wrong) math CoT answers using the Zhipu (智谱) API.
Uses the existing correct CoT from train_cot.json as "chosen", and API-generated
answers with errors as "rejected" candidates for DPO training.

For each problem, the script sends the question + correct CoT to the API with
a prompt that simulates a weaker student making a key error. Each problem is
called exactly once — no retries.

Usage:
    # Dry-run: test with 5 problems
    python generate_candidates.py --dry-run

    # Process all 11,999 problems (default)
    python generate_candidates.py

    # Process N problems
    python generate_candidates.py --sample-size 2000

    # Resume from checkpoint
    python generate_candidates.py --start 847

    # Process sequentially (no shuffle)
    python generate_candidates.py --no-shuffle

    # Use a different Zhipu model
    python generate_candidates.py --model glm-4-flash
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI

from config import (
    TRAIN_COT_PATH,
    CHECKPOINT_PATH,
    CANDIDATES_PATH,
    OUTPUT_PATH,
    SAMPLE_SIZE,
    RANDOM_SEED,
    ZHIPU_BASE_URL,
    ZHIPU_MODEL,
    ZHIPU_API_KEY_ENV,
    MAX_TOKENS,
    MIN_DELAY_BETWEEN_CALLS,
    SAVE_INTERVAL,
    MIN_RESPONSE_LENGTH,
)
from prompts import build_messages, get_strategy_temperature
from answer_verifier import extract_final_answer, extract_ground_truth, is_answer_wrong

# Load .env file
load_dotenv()

# Strategy name (constant — single strategy)
STRATEGY_NAME = "weak_student"


# ── Logging ──

def log(msg: str):
    """Timestamped log to stderr (so stdout can be used for piping)."""
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


# ── Checkpoint ──

def load_checkpoint() -> dict:
    """Load checkpoint if it exists, otherwise return default."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {
        "last_processed_index": -1,
        "total_processed": 0,
        "total_api_calls": 0,
        "total_pairs_generated": 0,
        "total_skipped_no_cot": 0,
        "started_at": None,
        "last_updated": None,
    }


def save_checkpoint(checkpoint: dict):
    """Atomically write checkpoint."""
    checkpoint["last_updated"] = datetime.now(timezone.utc).isoformat()
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CHECKPOINT_PATH)


# ── API Client ──

def create_client() -> OpenAI:
    """Create OpenAI-compatible Zhipu (智谱) client."""
    api_key = os.getenv(ZHIPU_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"API key not found. Set {ZHIPU_API_KEY_ENV} environment variable "
            f"or create a .env file with: {ZHIPU_API_KEY_ENV}=your_key_here"
        )
    return OpenAI(base_url=ZHIPU_BASE_URL, api_key=api_key)


def call_api(
    client: OpenAI,
    messages: list[dict],
    temperature: float,
    model: str = ZHIPU_MODEL,
) -> str:
    """
    Call Zhipu API. No retry — any failure raises immediately and stops the program.

    Returns response text.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ── Question preprocessing ──

def preprocess_question(question) -> str:
    """
    Normalize question text. Handles list-type questions (649 records)
    by joining list elements.
    """
    if isinstance(question, list):
        return "".join(q.strip('"') for q in question)
    return question.strip()


def has_valid_cot(answer: str) -> bool:
    """
    Check if the answer has proper CoT reasoning (推理过程 section).
    Records without reasoning can't serve as good 'chosen'.
    """
    return "推理过程" in answer and len(answer) >= 50


# ── Single API call per problem ──

def try_generate_wrong(
    client: OpenAI,
    question: str,
    correct_cot: str,
    ground_truth: str,
    model: str,
) -> dict:
    """
    Call the API once to generate a wrong-answer variant.

    Sends the question + correct CoT to the model with the weak-student prompt.
    Returns a generation record dict.
    """
    temperature = get_strategy_temperature()
    messages = build_messages(question, correct_cot)

    time.sleep(MIN_DELAY_BETWEEN_CALLS)
    response_text = call_api(client, messages, temperature, model=model)

    extracted = extract_final_answer(response_text)
    is_wrong = None
    status = "unknown"

    if extracted is None:
        status = "unparseable"
    else:
        is_wrong = is_answer_wrong(ground_truth, extracted)
        if is_wrong is True:
            status = "rejected_candidate"
        elif is_wrong is False:
            status = "same_answer"
        else:
            status = "cannot_determine"

    # Quality checks
    if len(response_text) < MIN_RESPONSE_LENGTH:
        status = "too_short"
    if response_text.rstrip().endswith(("，", "（", "(")):
        status = "possibly_truncated"

    return {
        "strategy": STRATEGY_NAME,
        "temperature": temperature,
        "generated_text": response_text,
        "extracted_answer": extracted,
        "is_wrong": is_wrong,
        "status": status,
    }


# ── Main generation loop ──

def generate(
    problems: list[dict],
    client: OpenAI,
    start_index: int = 0,
    model: str = ZHIPU_MODEL,
) -> dict:
    """
    Process problems and generate DPO candidate pairs.

    For each problem, sends one API request with the question + correct CoT
    and the weak-student prompt. Falls back to programmatic wrong-answer
    generation if the API call fails.

    Args:
        problems: List of problem dicts from train_cot.json.
        client: OpenAI-compatible client for Zhipu.
        start_index: Index to resume from.
        model: Zhipu model name.

    Returns:
        Final checkpoint dict with statistics.
    """
    checkpoint = load_checkpoint()
    if checkpoint["started_at"] is None:
        checkpoint["started_at"] = datetime.now(timezone.utc).isoformat()

    # Initialize missing checkpoint fields
    for key in ("total_skipped_no_cot",):
        if key not in checkpoint:
            checkpoint[key] = 0

    candidates_file = open(CANDIDATES_PATH, "a", encoding="utf-8")

    total_pairs = checkpoint["total_pairs_generated"]
    total_api_calls = checkpoint["total_api_calls"]
    processed = checkpoint["total_processed"]
    skipped_no_cot = checkpoint["total_skipped_no_cot"]

    total_problems = len(problems)
    end_index = start_index + total_problems - 1

    log(f"Starting from index {start_index}, {total_problems} problems to process")
    log(f"Resume: {processed} done, {total_api_calls} API calls, {total_pairs} pairs")

    for i, record in enumerate(problems):
        problem_idx = start_index + i

        # Checkpoint resume: skip already-processed
        if problem_idx <= checkpoint["last_processed_index"]:
            continue

        problem_id = record.get("id", str(problem_idx))
        question = preprocess_question(record["question"])
        answer_full = record["answer"]

        # Skip records without proper CoT reasoning
        if not has_valid_cot(answer_full):
            log(f"[{problem_idx}/{end_index}] SKIP {problem_id}: no valid CoT reasoning")
            skipped_no_cot += 1
            processed += 1
            candidates_file.write(json.dumps({
                "id": problem_id, "question": question,
                "ground_truth_answer": None, "ground_truth_cot": answer_full,
                "generations": [], "status": "skipped_no_cot",
            }, ensure_ascii=False) + "\n")
            candidates_file.flush()

            if processed % SAVE_INTERVAL == 0:
                checkpoint.update({
                    "last_processed_index": problem_idx, "total_processed": processed,
                    "total_api_calls": total_api_calls, "total_pairs_generated": total_pairs,
                    "total_skipped_no_cot": skipped_no_cot,
                })
                save_checkpoint(checkpoint)
            continue

        ground_truth = extract_ground_truth(answer_full)
        if ground_truth is None:
            log(f"[{problem_idx}/{end_index}] SKIP {problem_id}: cannot extract ground truth")
            skipped_no_cot += 1
            processed += 1
            candidates_file.write(json.dumps({
                "id": problem_id, "question": question,
                "ground_truth_answer": None, "ground_truth_cot": answer_full,
                "generations": [], "status": "skipped_no_gt",
            }, ensure_ascii=False) + "\n")
            candidates_file.flush()
            continue

        log(f"[{problem_idx}/{end_index}] {problem_id}: {question[:60]}...")

        # ── Single API call per problem ──
        total_api_calls += 1
        gen_record = try_generate_wrong(client, question, answer_full, ground_truth, model)
        all_generations = [gen_record]

        status = gen_record["status"]
        if status == "rejected_candidate":
            extracted = gen_record["extracted_answer"]
            log(f"  ✓ WRONG: {extracted} (gt: {ground_truth})")
        elif status == "same_answer":
            log(f"  answer correct but COT may be unreliable — keeping as rejected")
        elif status == "unparseable":
            log(f"  unparseable answer — keeping as rejected")
        else:
            log(f"  {status} — keeping as rejected")

        # ── Always use the API response as rejected (COT may be unreliable) ──
        rejected_source = "api"

        # Write intermediate result to JSONL
        candidate_record = {
            "id": problem_id,
            "question": question,
            "ground_truth_answer": ground_truth,
            "ground_truth_cot": answer_full,
            "generations": all_generations,
            "has_rejected": True,
            "rejected_source": rejected_source,
        }
        candidates_file.write(json.dumps(candidate_record, ensure_ascii=False) + "\n")
        candidates_file.flush()

        # Write DPO pair — always from API response
        pair = {
            "prompt": [{"role": "user", "content": question}],
            "chosen": [{"role": "assistant", "content": answer_full}],
            "rejected": [{"role": "assistant", "content": gen_record["generated_text"]}],
            "metadata": {
                "problem_id": problem_id,
                "rejected_strategy": gen_record["strategy"],
                "ground_truth": ground_truth,
                "rejected_answer": gen_record.get("extracted_answer"),
                "source": rejected_source,
            },
        }
        with open(OUTPUT_PATH, "a", encoding="utf-8") as pf:
            pf.write(json.dumps(pair, ensure_ascii=False) + "\n")
        total_pairs += 1

        processed += 1

        # Periodic checkpoint
        if processed % SAVE_INTERVAL == 0:
            checkpoint.update({
                "last_processed_index": problem_idx,
                "total_processed": processed,
                "total_api_calls": total_api_calls,
                "total_pairs_generated": total_pairs,
                "total_skipped_no_cot": skipped_no_cot,
            })
            save_checkpoint(checkpoint)
            log(f"  ── Checkpoint: {processed} processed, {total_api_calls} calls, "
                f"{total_pairs} pairs ──")

    # Final checkpoint
    checkpoint.update({
        "last_processed_index": start_index + total_problems - 1,
        "total_processed": processed,
        "total_api_calls": total_api_calls,
        "total_pairs_generated": total_pairs,
        "total_skipped_no_cot": skipped_no_cot,
    })
    save_checkpoint(checkpoint)

    candidates_file.close()
    return checkpoint


# ── Summary ──

def print_summary(checkpoint: dict):
    """Print generation summary and per-strategy statistics."""
    print("\n" + "=" * 60, file=sys.stderr)
    print("GENERATION SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for key, label in [
        ("total_processed", "Problems processed"),
        ("total_skipped_no_cot", "Skipped (no CoT)"),
        ("total_api_calls", "Total API calls"),
        ("total_pairs_generated", "DPO pairs generated"),
    ]:
        val = checkpoint.get(key, "N/A")
        print(f"  {label}: {val}", file=sys.stderr)
    print(f"  Candidates file: {CANDIDATES_PATH}", file=sys.stderr)
    print(f"  DPO pairs file:  {OUTPUT_PATH}", file=sys.stderr)

    # Per-status stats from candidates file
    if os.path.exists(CANDIDATES_PATH):
        status_counts = {
            "rejected_candidate": 0,
            "same_answer": 0,
            "unparseable": 0,
            "api_failed": 0,
            "too_short": 0,
            "cannot_determine": 0,
            "other": 0,
        }
        total_gens = 0
        api_gens = 0
        with open(CANDIDATES_PATH) as f:
            for line in f:
                record = json.loads(line)
                for gen in record.get("generations", []):
                    s = gen.get("status", "unknown")
                    if gen.get("strategy") == STRATEGY_NAME:
                        api_gens += 1
                    total_gens += 1
                    if s in status_counts:
                        status_counts[s] += 1
                    else:
                        status_counts["other"] += 1

        print("\n--- Generation Statistics ---", file=sys.stderr)
        print(f"  Total generations: {total_gens}", file=sys.stderr)
        print(f"  API generations: {api_gens}", file=sys.stderr)
        for s, count in sorted(status_counts.items()):
            if count > 0:
                print(f"    {s}: {count}", file=sys.stderr)

        if api_gens > 0:
            err_rate = 100 * status_counts["rejected_candidate"] / api_gens
            print(f"\n  API rejected rate: {err_rate:.1f}% "
                  f"({status_counts['rejected_candidate']}/{api_gens})", file=sys.stderr)


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO candidate answers for math problems using Zhipu (智谱) API"
    )
    parser.add_argument(
        "--sample-size", type=int, default=SAMPLE_SIZE,
        help=f"Number of problems to process (0=all, default: {SAMPLE_SIZE})"
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help="Start index for resume (overrides checkpoint)"
    )
    parser.add_argument(
        "--model", type=str, default=ZHIPU_MODEL,
        help=f"Zhipu model name (default: {ZHIPU_MODEL})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process only 5 problems as a test"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="Process problems in original order (don't shuffle)"
    )
    args = parser.parse_args()

    model_name = args.model
    random.seed(args.seed)

    # Dry-run: only 5 problems
    if args.dry_run:
        args.sample_size = 5
        log("=== DRY RUN MODE: 5 problems ===")

    # Load data
    log(f"Loading data from {TRAIN_COT_PATH}")
    with open(TRAIN_COT_PATH, encoding="utf-8") as f:
        all_data = json.load(f)
    log(f"Loaded {len(all_data)} records")

    # Sample: 0 or None means use all data
    sample_size = args.sample_size
    if sample_size <= 0:
        sample_size = len(all_data)

    if not args.no_shuffle:
        random.shuffle(all_data)

    sample = all_data[:sample_size]
    log(f"Processing {len(sample)} problems "
        f"({100*len(sample)/len(all_data):.1f}% of {len(all_data)} total)")

    # Determine start index
    start_index = args.start
    if start_index is None:
        checkpoint = load_checkpoint()
        if checkpoint["last_processed_index"] >= 0:
            start_index = checkpoint["last_processed_index"] + 1
            log(f"Resuming from index {start_index} (checkpoint)")
        else:
            start_index = 0

    if start_index >= len(sample):
        log("All problems already processed!")
        print_summary(load_checkpoint())
        return

    problems_to_process = sample[start_index:]
    log(f"Will process {len(problems_to_process)} problems "
        f"(index {start_index} to {len(sample)-1})")

    # Create API client
    log(f"Connecting to Zhipu API: {ZHIPU_BASE_URL}")
    log(f"Model: {model_name}")
    client = create_client()

    # Generate
    log("Starting generation (1 API call per problem, no retries)...")
    start_time = time.time()
    checkpoint = generate(
        problems_to_process, client,
        start_index=start_index,
        model=model_name,
    )
    elapsed = time.time() - start_time
    log(f"Generation completed in {elapsed/60:.1f} minutes "
        f"({elapsed/3600:.1f} hours)")

    # Summary
    print_summary(checkpoint)

    if args.dry_run:
        log("\nDry-run complete. Check dpo_candidates.jsonl and dpo_pairs.jsonl for output samples.")


if __name__ == "__main__":
    main()
