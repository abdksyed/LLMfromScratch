from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "submission_artifacts"

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DATASET_ID = "ryanmarten/OpenThoughts-1k-sample"

SEED = 7
SEQ_LEN = 256
BATCH_SIZE = 1

LANES = ["general", "code", "agentic", "reasoning", "long", "indic"]

FLOORS = {
    "general": 0.01,
    "code": 0.10,
    "agentic": 0.05,
    "reasoning": 0.04,
    "long": 0.02,
    "indic": 0.08,
}

POLICIES = {
    "general": "concat",
    "code": "concat",
    "agentic": "agent_mask",
    "reasoning": "concat",
    "long": "single_doc_chunk",
    "indic": "concat",
}

STAGES = [
    {
        "name": "foundation",
        "n_batches": 6,
        "shares": {
            "general": 0.445,
            "code": 0.180,
            "agentic": 0.080,
            "reasoning": 0.120,
            "long": 0.030,
            "indic": 0.145,
        },
    },
    {
        "name": "capability",
        "n_batches": 6,
        "shares": {
            "general": 0.350,
            "code": 0.220,
            "agentic": 0.100,
            "reasoning": 0.150,
            "long": 0.030,
            "indic": 0.150,
        },
    },
]

TOTAL_BATCHES = sum(s["n_batches"] for s in STAGES)
CRASH_AFTER = 6
FORK_AT = 3
REPLAY_START = 1
REPLAY_END = 4
FORK_BATCHES = 2

MAX_DOC_TOKENS = {
    "general": 96,
    "code": 96,
    "agentic": 128,
    "reasoning": 96,
    "long": 400,
    "indic": 96,
}

DOCS_PER_SHARD = 40
OPUS_TEMPERATURE = 0.35
EXPOSURE_CAP = 4.0
TRAIN_LR = 2e-5
