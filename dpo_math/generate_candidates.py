#!/usr/bin/env python3
"""
DPO Candidate Answer Generator.

Generates variant (potentially wrong) math CoT answers using the DeepSeek API.
Uses the existing correct CoT from train_cot.json as "chosen", and API-generated
answers with errors as "rejected" candidates for DPO training.

For each problem, the script tries multiple prompt strategies to generate wrong
answers. If the first round fails to produce a wrong answer, it retries with
different strategies (up to MAX_RETRY_ROUNDS) to maximize coverage.

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

    # Use a different DeepSeek model
    python generate_candidates.py --model deepseek-chat
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from config import (
    TRAIN_COT_PATH,
    CHECKPOINT_PATH,
    CANDIDATES_PATH,
    OUTPUT_PATH,
    SAMPLE_SIZE,
    RANDOM_SEED,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_API_KEY_ENV,
    MAX_TOKENS,
    MIN_DELAY_BETWEEN_CALLS,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    SAVE_INTERVAL,
    STRATEGIES_PER_PROBLEM,
    STRATEGY_WEIGHTS,
    MAX_RETRY_ROUNDS,
    MIN_RESPONSE_LENGTH,
)
from prompts import build_messages, get_strategy_temperature, STRATEGIES
from answer_verifier import extract_final_answer, extract_ground_truth, is_answer_wrong, generate_wrong_cot

# Load .env file
load_dotenv()


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
        "total_no_rejected": 0,
        "total_fallback_pairs": 0,
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
    """Create OpenAI-compatible DeepSeek client."""
    api_key = os.getenv(DEEPSEEK_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"API key not found. Set {DEEPSEEK_API_KEY_ENV} environment variable "
            f"or create a .env file with: {DEEPSEEK_API_KEY_ENV}=your_key_here"
        )
    return OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=api_key)


def call_api(
    client: OpenAI,
    messages: list[dict],
    temperature: float,
    model: str = DEEPSEEK_MODEL,
    max_retries: int = MAX_RETRIES,
) -> Optional[str]:
    """
    Call DeepSeek API with exponential backoff retry.

    Returns response text, or None on persistent failure.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=temperature,
            )
            return response.choices[0].message.content

        except Exception as e:
            err_msg = str(e)
            # Rate limit — wait and retry
            if "429" in err_msg or "rate" in err_msg.lower():
                wait = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                log(f"  Rate limited, waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            # Server error — wait and retry
            elif err_msg and "5" in err_msg[:10] if len(err_msg) > 10 else False:
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                log(f"  Server error, waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            # Other errors
            else:
                log(f"  API error (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(RETRY_BASE_DELAY)

    return None


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


# ── Strategy selection ──

def select_strategies(n: int, exclude: list[str] = None) -> list[str]:
    """
    Randomly select N strategies weighted by STRATEGY_WEIGHTS.

    Args:
        n: Number of strategies to select.
        exclude: Strategies to exclude from selection (for retry variety).

    Returns selected strategy names.
    """
    strategies = list(STRATEGY_WEIGHTS.keys())
    weights = list(STRATEGY_WEIGHTS.values())

    # Exclude specified strategies
    if exclude:
        filtered = [(s, w) for s, w in zip(strategies, weights) if s not in exclude]
        if not filtered:
            # All excluded — fall back to all strategies
            filtered = list(zip(strategies, weights))
        strategies, weights = zip(*filtered)

    strategies = list(strategies)
    weights = list(weights)

    # Weighted sample without replacement
    chosen = []
    remaining_s = list(strategies)
    remaining_w = list(weights)
    for _ in range(min(n, len(strategies))):
        if not remaining_s:
            break
        total_w = sum(remaining_w)
        probs = [w / total_w for w in remaining_w]
        idx = random.choices(range(len(remaining_s)), weights=probs, k=1)[0]
        chosen.append(remaining_s.pop(idx))
        remaining_w.pop(idx)
    return chosen


# ── Single strategy call ──

def try_strategy(
    client: OpenAI,
    question: str,
    ground_truth: str,
    strategy: str,
    model: str,
) -> dict:
    """
    Call the API with a single strategy and evaluate the result.

    Returns a generation record dict.
    """
    temperature = get_strategy_temperature(strategy)
    messages = build_messages(question, strategy)

    time.sleep(MIN_DELAY_BETWEEN_CALLS)
    response_text = call_api(client, messages, temperature, model=model)

    if response_text is None:
        return {
            "strategy": strategy,
            "temperature": temperature,
            "generated_text": None,
            "extracted_answer": None,
            "is_wrong": None,
            "status": "api_failed",
        }

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
    if response_text and len(response_text) < MIN_RESPONSE_LENGTH:
        status = "too_short"
    if response_text and response_text.rstrip().endswith(("，", "（", "(")):
        status = "possibly_truncated"

    return {
        "strategy": strategy,
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
    model: str = DEEPSEEK_MODEL,
    strategies_per_problem: int = STRATEGIES_PER_PROBLEM,
    max_retry_rounds: int = MAX_RETRY_ROUNDS,
) -> dict:
    """
    Process problems and generate DPO candidate pairs.

    For each problem, tries up to max_retry_rounds to get at least one
    rejected (wrong) answer. Each round uses different strategies.

    Args:
        problems: List of problem dicts from train_cot.json.
        client: OpenAI-compatible client for DeepSeek.
        start_index: Index to resume from.
        model: DeepSeek model name.
        strategies_per_problem: Strategies to try per round.
        max_retry_rounds: Max retry rounds if no rejected answer found.

    Returns:
        Final checkpoint dict with statistics.
    """
    checkpoint = load_checkpoint()
    if checkpoint["started_at"] is None:
        checkpoint["started_at"] = datetime.now(timezone.utc).isoformat()

    # Initialize missing checkpoint fields
    for key in ("total_skipped_no_cot", "total_no_rejected", "total_fallback_pairs"):
        if key not in checkpoint:
            checkpoint[key] = 0

    candidates_file = open(CANDIDATES_PATH, "a", encoding="utf-8")

    total_pairs = checkpoint["total_pairs_generated"]
    total_api_calls = checkpoint["total_api_calls"]
    processed = checkpoint["total_processed"]
    skipped_no_cot = checkpoint["total_skipped_no_cot"]
    no_rejected = checkpoint["total_no_rejected"]
    fallback_pairs = checkpoint["total_fallback_pairs"]

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
            # Still write a minimal candidate record for tracking
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
                    "total_skipped_no_cot": skipped_no_cot, "total_no_rejected": no_rejected,
                    "total_fallback_pairs": fallback_pairs,
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

        # ── Multi-round generation with retry ──
        all_generations = []
        rejected_candidates = []
        tried_strategies = set()
        round_num = 0  # track outside loop

        for round_num in range(max_retry_rounds):
            # Select fresh strategies (exclude already-tried ones within this problem)
            chosen = select_strategies(strategies_per_problem, exclude=list(tried_strategies))
            if not chosen:
                log(f"  All strategies exhausted for this problem")
                break

            tried_strategies.update(chosen)
            round_label = f"R{round_num+1}" if round_num > 0 else "R1"
            log(f"  [{round_label}] Strategies: {chosen}")

            round_rejected = []

            for strategy in chosen:
                total_api_calls += 1
                gen_record = try_strategy(client, question, ground_truth, strategy, model)
                gen_record["round"] = round_num + 1
                all_generations.append(gen_record)

                status = gen_record["status"]
                if status == "rejected_candidate":
                    extracted = gen_record["extracted_answer"]
                    log(f"    [{strategy}] ✓ WRONG: {extracted} (gt: {ground_truth})")
                    round_rejected.append(gen_record)
                elif status == "same_answer":
                    log(f"    [{strategy}] correct (same as gt)")
                elif status == "api_failed":
                    log(f"    [{strategy}] API FAILED")
                else:
                    log(f"    [{strategy}] {status}")

            rejected_candidates.extend(round_rejected)

            # Stop retrying once we have at least one rejected candidate
            if rejected_candidates:
                break

        # ── Determine rejected source ──
        if rejected_candidates:
            rejected_source = "api"
        else:
            # ── Fallback: programmatically generate a wrong CoT ──
            wrong_cot = generate_wrong_cot(answer_full)
            if wrong_cot:
                fallback_gen = {
                    "strategy": "fallback_programmatic",
                    "temperature": 0,
                    "generated_text": wrong_cot,
                    "extracted_answer": extract_ground_truth(wrong_cot),
                    "is_wrong": True,
                    "status": "rejected_candidate",
                    "round": 0,
                }
                rejected_candidates.append(fallback_gen)
                all_generations.append(fallback_gen)
                rejected_source = "fallback"
                fallback_pairs += 1
                log(f"  ⚡ FALLBACK: programmatic wrong answer generated")
            else:
                rejected_source = "none"

        # Write intermediate result to JSONL
        candidate_record = {
            "id": problem_id,
            "question": question,
            "ground_truth_answer": ground_truth,
            "ground_truth_cot": answer_full,
            "generations": all_generations,
            "total_rounds": round_num + 1,
            "has_rejected": len(rejected_candidates) > 0,
            "rejected_source": rejected_source,
        }
        candidates_file.write(json.dumps(candidate_record, ensure_ascii=False) + "\n")
        candidates_file.flush()

        # Write DPO pair
        if rejected_candidates:
            rejected = rejected_candidates[0]
            pair = {
                "prompt": [{"role": "user", "content": question}],
                "chosen": [{"role": "assistant", "content": answer_full}],
                "rejected": [{"role": "assistant", "content": rejected["generated_text"]}],
                "metadata": {
                    "problem_id": problem_id,
                    "rejected_strategy": rejected["strategy"],
                    "ground_truth": ground_truth,
                    "rejected_answer": rejected.get("extracted_answer"),
                    "source": rejected_source,
                },
            }
            with open(OUTPUT_PATH, "a", encoding="utf-8") as pf:
                pf.write(json.dumps(pair, ensure_ascii=False) + "\n")
            total_pairs += 1
        else:
            no_rejected += 1
            log(f"  ⚠ NO REJECTED after {round_num+1} round(s), {len(all_generations)} attempts")

        processed += 1

        # Periodic checkpoint
        if processed % SAVE_INTERVAL == 0:
            checkpoint.update({
                "last_processed_index": problem_idx,
                "total_processed": processed,
                "total_api_calls": total_api_calls,
                "total_pairs_generated": total_pairs,
                "total_skipped_no_cot": skipped_no_cot,
                "total_no_rejected": no_rejected,
                "total_fallback_pairs": fallback_pairs,
            })
            save_checkpoint(checkpoint)
            log(f"  ── Checkpoint: {processed} processed, {total_api_calls} calls, "
                f"{total_pairs} pairs ({fallback_pairs} fb), {no_rejected} no-rej ──")

    # Final checkpoint
    checkpoint.update({
        "last_processed_index": start_index + total_problems - 1,
        "total_processed": processed,
        "total_api_calls": total_api_calls,
        "total_pairs_generated": total_pairs,
        "total_skipped_no_cot": skipped_no_cot,
        "total_no_rejected": no_rejected,
        "total_fallback_pairs": fallback_pairs,
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
        ("total_fallback_pairs", "  of which fallback"),
        ("total_no_rejected", "No rejected found"),
    ]:
        val = checkpoint.get(key, "N/A")
        print(f"  {label}: {val}", file=sys.stderr)
    print(f"  Candidates file: {CANDIDATES_PATH}", file=sys.stderr)
    print(f"  DPO pairs file:  {OUTPUT_PATH}", file=sys.stderr)

    # Per-strategy stats
    if os.path.exists(CANDIDATES_PATH):
        strategy_stats = {}
        total_gens = 0
        with open(CANDIDATES_PATH) as f:
            for line in f:
                record = json.loads(line)
                for gen in record.get("generations", []):
                    s = gen.get("strategy", "unknown")
                    status = gen.get("status", "unknown")
                    if s not in strategy_stats:
                        strategy_stats[s] = {"total": 0, "rejected": 0, "same": 0,
                                             "unparseable": 0, "api_failed": 0, "other": 0}
                    strategy_stats[s]["total"] += 1
                    total_gens += 1
                    if status == "rejected_candidate":
                        strategy_stats[s]["rejected"] += 1
                    elif status == "same_answer":
                        strategy_stats[s]["same"] += 1
                    elif status == "unparseable":
                        strategy_stats[s]["unparseable"] += 1
                    elif status == "api_failed":
                        strategy_stats[s]["api_failed"] += 1
                    else:
                        strategy_stats[s]["other"] += 1

        print("\n--- Per-Strategy Statistics ---", file=sys.stderr)
        for s, stats in sorted(strategy_stats.items()):
            desc = STRATEGIES.get(s, {}).get("description", s)
            err_rate = 100 * stats["rejected"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {s} ({desc}):", file=sys.stderr)
            print(f"    Total={stats['total']}, Rejected={stats['rejected']} ({err_rate:.1f}%), "
                  f"Same={stats['same']}, Unparseable={stats['unparseable']}, "
                  f"Failed={stats['api_failed']}, Other={stats['other']}", file=sys.stderr)

        if total_gens > 0:
            total_rejected = sum(s['rejected'] for s in strategy_stats.values())
            print(f"\n  Overall rejected rate: {100 * total_rejected / total_gens:.1f}% "
                  f"({total_rejected}/{total_gens})", file=sys.stderr)


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO candidate answers for math problems using DeepSeek API"
    )
    parser.add_argument(
        "--sample-size", type=int, default=SAMPLE_SIZE,
        help=f"Number of problems to process (0=all, default: {SAMPLE_SIZE})"
    )
    parser.add_argument(
        "--strategies-per-problem", type=int, default=STRATEGIES_PER_PROBLEM,
        help=f"Strategies per round (default: {STRATEGIES_PER_PROBLEM})"
    )
    parser.add_argument(
        "--max-retry-rounds", type=int, default=MAX_RETRY_ROUNDS,
        help=f"Max retry rounds if no rejected (default: {MAX_RETRY_ROUNDS})"
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help="Start index for resume (overrides checkpoint)"
    )
    parser.add_argument(
        "--model", type=str, default=DEEPSEEK_MODEL,
        help=f"DeepSeek model name (default: {DEEPSEEK_MODEL})"
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
    log(f"Connecting to DeepSeek API: {DEEPSEEK_BASE_URL}")
    log(f"Model: {model_name}")
    client = create_client()

    # Generate
    log(f"Starting generation ({args.strategies_per_problem} strategies/problem, "
        f"max {args.max_retry_rounds} retry rounds)...")
    start_time = time.time()
    checkpoint = generate(
        problems_to_process, client,
        start_index=start_index,
        model=model_name,
        strategies_per_problem=args.strategies_per_problem,
        max_retry_rounds=args.max_retry_rounds,
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
