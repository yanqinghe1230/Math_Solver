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

# --- DeepSeek API ---
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
MAX_TOKENS = 1024
TEMPERATURE = 0.9  # High temperature for variant generation

# --- Rate limiting ---
MIN_DELAY_BETWEEN_CALLS = 1.2  # Seconds between API calls (conservative)
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # Base delay for exponential backoff

# --- Checkpoint ---
SAVE_INTERVAL = 50  # Save checkpoint every N problems

# --- Generation ---
STRATEGIES_PER_PROBLEM = 3  # Strategies tried per problem (increased for higher error yield)
MAX_RETRY_ROUNDS = 2  # If no rejected answer found, retry with new strategies up to this many rounds

# Strategy weights for random selection (Strategy B weighted 2x)
STRATEGY_WEIGHTS = {
    "high_temp": 1,
    "student_mistake": 2,
    "rushed": 1,
    "alternative": 1,
}

# --- Quality filters ---
MIN_RESPONSE_LENGTH = 30  # Minimum characters for a valid generated response
