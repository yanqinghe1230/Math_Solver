#!/usr/bin/env python3
"""
Build DPO pairs from existing dpo_candidates.jsonl.

For records that already have API-generated rejected answers, use those.
For records without rejected answers, use the programmatic fallback.
Output the final dpo_pairs.jsonl.
"""

import json
import os
import sys
from datetime import datetime

# Allow running from project root or dpo_math/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_verifier import generate_wrong_cot, extract_ground_truth

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATES_PATH = os.path.join(BASE_DIR, "dpo_candidates.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "dpo_pairs.jsonl")
CANDIDATES_UPDATED_PATH = os.path.join(BASE_DIR, "dpo_candidates.jsonl")  # overwrite same file


def log(msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def main():
    if not os.path.exists(CANDIDATES_PATH):
        log(f"ERROR: {CANDIDATES_PATH} not found")
        sys.exit(1)

    log(f"Reading {CANDIDATES_PATH}...")

    records = []
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    log(f"Loaded {len(records)} records")

    # Stats
    api_rejected = 0
    fallback_applied = 0
    fallback_failed = 0
    skipped_no_cot = 0

    # Open output
    pairs_file = open(OUTPUT_PATH, "w", encoding="utf-8")
    candidates_out = open(CANDIDATES_UPDATED_PATH, "w", encoding="utf-8")

    for i, rec in enumerate(records):
        problem_id = rec.get("id", "?")
        question = rec.get("question", "")
        correct_cot = rec.get("ground_truth_cot", "")
        ground_truth = rec.get("ground_truth_answer", "")

        # Already skipped (no CoT)
        if rec.get("status") in ("skipped_no_cot", "skipped_no_gt"):
            skipped_no_cot += 1
            pairs_file.write(json.dumps(rec, ensure_ascii=False) + "\n")  # placeholder
            candidates_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue

        # Already has API rejected
        if rec.get("has_rejected"):
            rejected_gens = [g for g in rec.get("generations", [])
                           if g.get("status") == "rejected_candidate"]
            if rejected_gens:
                api_rejected += 1
                rejected = rejected_gens[0]
                pair = {
                    "prompt": [{"role": "user", "content": question}],
                    "chosen": [{"role": "assistant", "content": correct_cot}],
                    "rejected": [{"role": "assistant", "content": rejected["generated_text"]}],
                    "metadata": {
                        "problem_id": problem_id,
                        "rejected_strategy": rejected.get("strategy", "api"),
                        "ground_truth": ground_truth,
                        "rejected_answer": rejected.get("extracted_answer"),
                        "source": "api",
                    },
                }
                pairs_file.write(json.dumps(pair, ensure_ascii=False) + "\n")
                candidates_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

        # ── Apply programmatic fallback ──
        wrong_cot = generate_wrong_cot(correct_cot)

        if wrong_cot:
            fallback_applied += 1
            wrong_gt = extract_ground_truth(wrong_cot)

            # Update the record
            fallback_gen = {
                "strategy": "fallback_programmatic",
                "temperature": 0,
                "generated_text": wrong_cot,
                "extracted_answer": wrong_gt,
                "is_wrong": True,
                "status": "rejected_candidate",
                "round": 0,
            }
            rec.setdefault("generations", []).append(fallback_gen)
            rec["has_rejected"] = True
            rec["rejected_source"] = "fallback"

            # Write DPO pair
            pair = {
                "prompt": [{"role": "user", "content": question}],
                "chosen": [{"role": "assistant", "content": correct_cot}],
                "rejected": [{"role": "assistant", "content": wrong_cot}],
                "metadata": {
                    "problem_id": problem_id,
                    "rejected_strategy": "fallback_programmatic",
                    "ground_truth": ground_truth,
                    "rejected_answer": wrong_gt,
                    "source": "fallback",
                },
            }
            pairs_file.write(json.dumps(pair, ensure_ascii=False) + "\n")
        else:
            fallback_failed += 1
            rec["has_rejected"] = False
            log(f"  FAILED fallback for problem {problem_id}")

        candidates_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if (i + 1) % 500 == 0:
            log(f"  Progress: {i+1}/{len(records)} — "
                f"api={api_rejected}, fallback={fallback_applied}, "
                f"failed={fallback_failed}")

    pairs_file.close()
    candidates_out.close()

    # Summary
    total_pairs = api_rejected + fallback_applied
    log("")
    log("=" * 50)
    log("BUILD COMPLETE")
    log("=" * 50)
    log(f"  Total records:       {len(records)}")
    log(f"  Skipped (no CoT):    {skipped_no_cot}")
    log(f"  API rejected:        {api_rejected}")
    log(f"  Fallback applied:    {fallback_applied}")
    log(f"  Fallback failed:     {fallback_failed}")
    log(f"  Total DPO pairs:     {total_pairs}")
    log(f"  Coverage:            {100*total_pairs/(len(records)-skipped_no_cot):.1f}%")
    log(f"")
    log(f"  Updated candidates:  {CANDIDATES_UPDATED_PATH}")
    log(f"  DPO pairs output:    {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
