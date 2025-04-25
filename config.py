import os
from pathlib import Path

# One place for all paths / hyper-params
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "cache"
INDEX_DIR = BASE_DIR / "retrieval" / "index"
HF_CACHE = os.environ.get("HF_HOME", str(BASE_DIR / ".hf_cache"))

VIT_NAME   = "openai/clip-vit-large-patch14"
LLM_NAME   = "meta-llama/Meta-Llama-3-8B-Instruct"

IMG_SIZE = 224            # ViT-L/14 native
MAX_SEQ  = 512
LR       = 5e-5
BATCH    = 4
EPOCHS   = 3

# FastAPI
HOST = "0.0.0.0"
PORT = 8000

# ── Agentic loop ──────────────────────────────────────────────────────────────
MAX_MCP_ITERS      = 6          # stop after N planner→executor cycles
ANSWER_OK_TOKENS   = ["Yes.", "No.", "It is", "They are", "The answer is"]
# ── CUDA optimisation flags ───────────────────────────────────────────────────
TORCH_COMPILE_MODE = "reduce-overhead"   # good default for inference

# ── OpenAI planner settings ──────────────────────────────────────────────
OPENAI = {
    "api_key":   os.getenv("OPENAI_API_KEY"),   # exported before running
    "model":     "gpt-4o",                      
    "temperature": 0.1,
    "max_tokens": 256,
}
