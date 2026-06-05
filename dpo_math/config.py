"""
DPO candidate generation configuration.
All paths, model settings, and rate-limiting constants.
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

TRAIN_COT_PATH = os.path.join(PROJECT_ROOT, "train_cot.json")

# Output files (inside dpo_math/)
CHECKPOINT_PATH = os.path.join(BASE_DIR, "dpo_checkpoint.json")
CANDIDATES_PATH = os.path.join(BASE_DIR, "dpo_candidates.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "dpo_pairs.jsonl")

# --- Sampling ---
SAMPLE_SIZE = 0  # Number of problems to process (0 = all)
RANDOM_SEED = 42

# --- Zhipu (智谱) API ---
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
ZHIPU_MODEL = "glm-4-flash"
ZHIPU_API_KEY_ENV = "ZHIPU_API_KEY"
MAX_TOKENS = 512  # Limit COT length (was 1024)
TEMPERATURE = 0.8  # Moderate temperature for plausible student errors

# --- Rate limiting ---
MIN_DELAY_BETWEEN_CALLS = 0.5  # Seconds between API calls (Zhipu is faster)
MAX_RETRIES = 1  # No retry on API failure — one attempt per problem
RETRY_BASE_DELAY = 1.0  # Base delay for backoff

# --- Checkpoint ---
SAVE_INTERVAL = 50  # Save checkpoint every N problems

# --- Generation ---
# Single strategy, single round — one API call per problem as required
STRATEGIES_PER_PROBLEM = 1
MAX_RETRY_ROUNDS = 1

# Quality filters
MIN_RESPONSE_LENGTH = 30  # Minimum characters for a valid generated response
