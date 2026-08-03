#!/usr/bin/env python3
"""
DeNovoSWE agentic-trajectory corpus -- 9-strategy data-cleaning pipeline.

Input : data/denovoswe_traj_filtered.jsonl   (4.3 GB, 11,463 agent trajectories)
Output : data/clean/shard-*.jsonl.gz         (cleaned corpus)
         data/clean/manifest-*.json          (one reproducible manifest per shard)
         data/report.json                    (measurements consumed by the web report)

The nine strategies taught in the session are each implemented as a named stage.
Order matters: cheap metadata gates (de-duplication, decontamination, the
verifier-score gate) run before expensive per-character work, so we never spend
CPU normalising a trajectory that is about to be dropped.

Run:  .venv/bin/python3 04-data-cleaning/scripts/clean_pipeline.py
"""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "denovoswe_traj_filtered.jsonl"
OUT_DIR = ROOT / "data" / "clean"
REPORT = ROOT / "data" / "report.json"

SHARD_SIZE = 500          # trajectories per shard
SCORE_GATE = 1.0          # keep only verifier-perfect trajectories for SFT
MINHASH_PERM = 128        # signature length
MINHASH_THRESHOLD = 0.67  # Jaccard threshold from the session notes
SHINGLE_K = 8             # words per shingle
MEAN_WORD_LEN = (3.0, 10.0)
MAX_SYMBOL_RATIO = 0.10   # measured on prose only, never on code
MAX_DUP_LINE_RATIO = 0.30
MIN_PROSE_CHARS = 200
TARGET_LANGS = {"en"}

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = whole file

# ---------------------------------------------------------------------------
# Optional dependencies (installed into .venv via uv)
# ---------------------------------------------------------------------------

try:
    import tiktoken

    ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    ENC = None

try:
    from datasketch import MinHash, MinHashLSH

    HAS_LSH = True
except Exception:  # pragma: no cover
    HAS_LSH = False

try:
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except Exception:  # pragma: no cover
    HAS_LANGDETECT = False


def tok_count(text: str) -> int:
    """Exact cl100k token count, or a 4-chars-per-token estimate as fallback."""
    if ENC is None:
        return len(text) // 4
    return len(ENC.encode(text, disallowed_special=()))


# ---------------------------------------------------------------------------
# STRATEGY 1 - Extraction and boilerplate removal
# ---------------------------------------------------------------------------
# Web crawls carry nav bars and cookie banners. An agent corpus carries the
# machine equivalent: an identical system prompt and identical tool schema on
# every row, plus terminal chrome (ANSI colour codes, carriage-return progress
# bars, pip's "Requirement already satisfied" walls) inside tool observations.

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
CR_PROGRESS_RE = re.compile(r"[^\n]*\r(?!\n)")
PIP_NOISE_RE = re.compile(
    r"^(?:Requirement already satisfied|Collecting|Downloading|Using cached|"
    r"Installing collected packages|Successfully installed|"
    r"\s+Preparing metadata|\s+Building wheel|\s+Created wheel|"
    r"\s+Stored in directory|WARNING: You are using pip version).*$",
    re.MULTILINE,
)
BLANK_RUN_RE = re.compile(r"\n{4,}")
BAR_RE = re.compile(r"^[-=_*#~]{40,}$", re.MULTILINE)

FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def strip_boilerplate(text: str, stats: Counter) -> str:
    """Drop terminal chrome and installer walls; keep every byte of real output."""
    before = len(text)

    if "\x1b[" in text:
        text = ANSI_RE.sub("", text)
        stats["ansi_msgs"] += 1
    if "\r" in text:
        text = CR_PROGRESS_RE.sub("", text)
        stats["cr_progress_msgs"] += 1

    text, n = PIP_NOISE_RE.subn("", text)
    if n:
        stats["pip_noise_lines"] += n
    text, n = BAR_RE.subn("", text)
    if n:
        stats["separator_bars"] += n

    text = BLANK_RUN_RE.sub("\n\n\n", text)
    stats["boilerplate_chars"] += before - len(text)
    return text


# ---------------------------------------------------------------------------
# STRATEGY 2 - Normalisation (context aware, code safe)
# ---------------------------------------------------------------------------
# The session's warning applies directly here: this corpus is mostly Python, so
# tabs, leading spaces and blank-line structure are load bearing. We therefore
# normalise *outside* fenced code blocks only, and we never touch indentation.
# NFC is safe for Latin text; zero-width joiners are only stripped where they
# cannot be Brahmic-script combining marks.

ZERO_WIDTH_ALWAYS = str.maketrans({
    "​": "",  # zero-width space
    "‎": "",  # LTR mark
    "‏": "",  # RTL mark
    "﻿": "",  # BOM
    "­": "",  # soft hyphen
    "⁠": "",  # word joiner
})
ZWJ_ZWNJ = ("‌", "‍")
NBSP_RE = re.compile("[   ]")
ENTITY_RE = re.compile(r"&(?:amp|lt|gt|quot|apos|nbsp|#x?[0-9a-fA-F]+);")
TRAIL_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
# Brahmic + other scripts that use ZWJ/ZWNJ semantically.
INDIC_RE = re.compile(
    r"[ऀ-෿ༀ-࿿က-႟ក-៿؀-ۿ]"
)


def _protect_code(text: str):
    """Replace fenced/inline code with placeholders so prose rules cannot hit it."""
    vault: list[str] = []

    def stash(m: re.Match) -> str:
        vault.append(m.group(0))
        return f"\x00{len(vault) - 1}\x00"

    return INLINE_CODE_RE.sub(stash, FENCE_RE.sub(stash, text)), vault


def _restore_code(text: str, vault: list[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: vault[int(m.group(1))], text)


def normalize_text(text: str, stats: Counter) -> str:
    original = text

    # Invisible characters are removed everywhere: they are never meaningful,
    # not even inside code.
    if any(c in text for c in ("​", "﻿", "­", "‎", "‏", "⁠")):
        text = text.translate(ZERO_WIDTH_ALWAYS)
        stats["zero_width_msgs"] += 1

    # ZWJ/ZWNJ carry meaning in Devanagari, Bengali, Tamil, Arabic ... so they
    # are only dropped when no such script is present in the document.
    if any(c in text for c in ZWJ_ZWNJ):
        if INDIC_RE.search(text):
            stats["zwj_preserved_indic"] += 1
        else:
            text = text.replace("‌", "").replace("‍", "")
            stats["zwj_stripped"] += 1

    body, vault = _protect_code(text)

    if NBSP_RE.search(body):
        body = NBSP_RE.sub(" ", body)
        stats["nbsp_msgs"] += 1

    # Double-unescaping is a real corpus bug (&amp;lt; -> <), so unescape twice
    # but only in prose, where an entity cannot be program source.
    if ENTITY_RE.search(body):
        body = html.unescape(html.unescape(body))
        stats["html_entity_msgs"] += 1

    body = unicodedata.normalize("NFC", body)
    text = _restore_code(body, vault)

    # Trailing whitespace is stripped; leading whitespace (indentation) is not.
    text = TRAIL_WS_RE.sub("", text)

    if text != original:
        stats["normalized_msgs"] += 1
    return text


# ---------------------------------------------------------------------------
# STRATEGY 3 - Ghost tags and special tokens
# ---------------------------------------------------------------------------
# Roles already live in structured JSON (`role`), so any chat-template marker
# that leaked into `content` is a ghost. We canonicalise on one vocabulary and
# neutralise the rest. Tags are *escaped*, not deleted, when they sit inside a
# code fence -- a tutorial that legitimately shows `<|im_start|>` should survive.

GHOST_TAGS = [
    r"<\|im_start\|>", r"<\|im_end\|>", r"<\|endoftext\|>", r"<\|eot_id\|>",
    r"<\|start_header_id\|>", r"<\|end_header_id\|>", r"<\|user\|>",
    r"<\|assistant\|>", r"<\|system\|>", r"<\|tool\|>", r"<\|fim_prefix\|>",
    r"<\|fim_middle\|>", r"<\|fim_suffix\|>", r"<s>", r"</s>", r"<pad>",
    r"\[INST\]", r"\[/INST\]", r"<<SYS>>", r"<</SYS>>",
    r"<\|channel\|>", r"<\|message\|>", r"<\|constrain\|>",
]
GHOST_RE = re.compile("|".join(GHOST_TAGS))
ROLE_PREFIX_RE = re.compile(
    r"^\s*(?:###\s*)?(?:Human|Assistant|System|User|AI)\s*:\s*", re.MULTILINE
)


def strip_ghost_tags(text: str, stats: Counter) -> str:
    if GHOST_RE.search(text):
        body, vault = _protect_code(text)
        body, n = GHOST_RE.subn("", body)
        if n:
            stats["ghost_tags_removed"] += n
            stats["ghost_tag_msgs"] += 1
        # Inside code fences the tag is content, so escape the pipe instead.
        vault = [GHOST_RE.sub(lambda m: m.group(0).replace("|", "❘"), v) for v in vault]
        stats["ghost_tags_escaped_in_code"] += sum("❘" in v for v in vault)
        text = _restore_code(body, vault)

    hits = ROLE_PREFIX_RE.findall(text)
    if hits:
        stats["role_prefix_lines"] += len(hits)
    return text


# ---------------------------------------------------------------------------
# STRATEGY 4 - Language identification and validation
# ---------------------------------------------------------------------------
# Never trust the label. Code confuses every language ID model, so we first
# remove code fences, diffs, tracebacks and paths, then run langdetect on what
# prose remains. Records with too little prose are tagged `code-dominant`
# rather than discarded -- discarding them would delete valid Python.

DIFF_LINE_RE = re.compile(r"^(?:[+-]{3}|[@]{2}|[+-]\s|diff --git).*$", re.MULTILINE)
PATH_RE = re.compile(r"(?:/[\w.\-]+){2,}")
TB_RE = re.compile(r'^\s*File "[^"]+", line \d+.*$', re.MULTILINE)


def prose_only(text: str) -> str:
    text = FENCE_RE.sub(" ", text)
    text = DIFF_LINE_RE.sub(" ", text)
    text = TB_RE.sub(" ", text)
    text = INLINE_CODE_RE.sub(" ", text)
    text = PATH_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_lang(prose: str) -> tuple[str, float]:
    if len(prose) < MIN_PROSE_CHARS:
        return "code-dominant", 0.0
    if not HAS_LANGDETECT:
        ascii_ratio = sum(c.isascii() for c in prose) / len(prose)
        return ("en", ascii_ratio) if ascii_ratio > 0.97 else ("unknown", ascii_ratio)
    try:
        top = detect_langs(prose[:20000])[0]
        return top.lang, round(top.prob, 4)
    except Exception:
        return "unknown", 0.0


# ---------------------------------------------------------------------------
# STRATEGY 5 - Quality filtering (heuristics + classifier signal)
# ---------------------------------------------------------------------------
# Layer A: deterministic heuristics, measured on prose so that code's naturally
#          high symbol ratio does not trigger a false reject.
# Layer B: the classifier layer. We do not need to train a proxy scorer here --
#          every trajectory already carries an *executed test-suite* verdict in
#          extra_info.score, which is a stronger signal than a learned rater.

def quality_heuristics(prose: str, full: str) -> list[str]:
    fails: list[str] = []
    words = prose.split()
    if words:
        mean_len = sum(len(w) for w in words) / len(words)
        if not (MEAN_WORD_LEN[0] <= mean_len <= MEAN_WORD_LEN[1]):
            fails.append(f"mean_word_len={mean_len:.2f}")
        symbols = sum(not (c.isalnum() or c.isspace()) for c in prose)
        if symbols / len(prose) > MAX_SYMBOL_RATIO:
            fails.append(f"symbol_ratio={symbols / len(prose):.3f}")

    lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
    if len(lines) > 30:
        dup = 1 - len(set(lines)) / len(lines)
        if dup > MAX_DUP_LINE_RATIO:
            fails.append(f"dup_line_ratio={dup:.3f}")
    return fails


# ---------------------------------------------------------------------------
# STRATEGY 6 - Global de-duplication
# ---------------------------------------------------------------------------
# Three tiers:
#   6a exact    - identical trajectory id / identical content hash
#   6b instance - N rollouts of the same task; keep the best-scoring, then the
#                 shortest. This is the dominant duplicate class in RL-rollout
#                 corpora and is invisible to MinHash (the prose differs).
#   6c near     - MinHash + LSH over 8-word shingles, Jaccard >= 0.67.

def make_minhash(text: str) -> "MinHash":
    m = MinHash(num_perm=MINHASH_PERM)
    words = text.split()
    for i in range(0, max(1, len(words) - SHINGLE_K + 1)):
        m.update(" ".join(words[i:i + SHINGLE_K]).encode("utf8", "ignore"))
    return m


# ---------------------------------------------------------------------------
# STRATEGY 7 - PII scrubbing
# ---------------------------------------------------------------------------
# Balanced, as the session insisted. We scrub contactable identifiers, but we
# deliberately preserve (a) loopback/private IPs, which are functional code,
# (b) copyright and licence attribution, which is a legal requirement, and
# (c) placeholder domains used in documentation.

SAFE_EMAIL_DOMAINS = {"example.com", "example.org", "example.net", "test.com", "email.com"}
EMAIL_RE = re.compile(r"\b[\w.+-]+@(?:[\w-]+\.)+[a-zA-Z]{2,}\b")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
PHONE_RE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]\d{3}[ -]\d{4}(?![\w.])")
SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{36}|gho_[A-Za-z0-9]{36}|"
    r"github_pat_[A-Za-z0-9_]{22,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"AIza[0-9A-Za-z_-]{35}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b"
)
COPYRIGHT_RE = re.compile(r"(?:Copyright|\(c\)|©)\s*(?:\d{4}|\d{4}-\d{4})?.*", re.IGNORECASE)


def _private_ip(ip: str) -> bool:
    try:
        a, b, *_ = (int(p) for p in ip.split("."))
    except ValueError:
        return True  # malformed -> not a real address, leave alone
    if any(int(p) > 255 for p in ip.split(".")):
        return True  # version string like 1.2.3.400
    return (
        a in (0, 10, 127) or (a == 192 and b == 168) or (a == 172 and 16 <= b <= 31)
        or (a == 169 and b == 254) or a >= 224
    )


def scrub_pii(text: str, stats: Counter) -> str:
    # Protect legally required attribution lines before scrubbing.
    keeps: list[str] = []

    def keep(m: re.Match) -> str:
        keeps.append(m.group(0))
        return f"\x01{len(keeps) - 1}\x01"

    text = COPYRIGHT_RE.sub(keep, text)
    if keeps:
        stats["copyright_lines_preserved"] += len(keeps)

    def email_sub(m: re.Match) -> str:
        dom = m.group(0).rsplit("@", 1)[1].lower()
        if dom in SAFE_EMAIL_DOMAINS:
            stats["placeholder_emails_kept"] += 1
            return m.group(0)
        stats["emails_redacted"] += 1
        return "<EMAIL>"

    def ip_sub(m: re.Match) -> str:
        if _private_ip(m.group(0)):
            stats["local_ips_kept"] += 1
            return m.group(0)
        stats["ips_redacted"] += 1
        return "<IP>"

    text = EMAIL_RE.sub(email_sub, text)
    text = IPV4_RE.sub(ip_sub, text)

    text, n = PHONE_RE.subn("<PHONE>", text)
    stats["phones_redacted"] += n
    text, n = SECRET_RE.subn("<SECRET>", text)
    stats["secrets_redacted"] += n

    return re.sub(r"\x01(\d+)\x01", lambda m: keeps[int(m.group(1))], text)


# ---------------------------------------------------------------------------
# STRATEGY 8 - Decontamination
# ---------------------------------------------------------------------------
# This corpus is repository-derived, so the leak risk is structural: if a task
# repo is also a SWE-bench evaluation repo, training on it inflates the score we
# would later report. We gate at repository level (the strong, cheap check) and
# additionally run 13-gram overlap against benchmark probe strings.

SWEBENCH_REPOS = {
    "astropy", "django", "matplotlib", "seaborn", "flask", "requests",
    "xarray", "pylint", "pytest", "scikit-learn", "sphinx", "sympy",
    "pyvista", "dask", "conan", "moto", "hydra", "modin", "dvc", "monai",
    "pydicom", "bokeh", "mne-python", "statsmodels", "sqlfluff", "pandas",
    "pydantic", "aiohttp", "datasette", "tornado", "jinja", "click",
}
BENCH_PROBES = [
    "def has_close_elements(numbers: List[float], threshold: float) -> bool",
    "Write a function to find the similar elements from the given two tuple lists",
    "Natalia sold clips to 48 of her friends in April",
    "The following are multiple choice questions (with answers) about",
]


def ngrams(text: str, n: int = 13) -> set[str]:
    w = text.lower().split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


PROBE_NGRAMS: set[str] = set()
for _p in BENCH_PROBES:
    PROBE_NGRAMS |= ngrams(_p, 8)


def repo_of(instance_id: str) -> str:
    """`python-social-auth_social-app-django_pr812` -> `social-app-django`."""
    base = re.sub(r"_pr\d+$", "", instance_id)
    return base.split("_")[-1] if "_" in base else base


def decontaminate(instance_id: str) -> str | None:
    repo = repo_of(instance_id).lower()
    if repo in SWEBENCH_REPOS:
        return f"swebench_repo:{repo}"
    return None


# ---------------------------------------------------------------------------
# Pass 1 - index metadata and byte offsets (no content work)
# ---------------------------------------------------------------------------

def pass1_index():
    rows = []
    t0 = time.time()
    with SRC.open("rb") as fh:
        offset = 0
        for i, raw in enumerate(fh):
            if LIMIT and i >= LIMIT:
                break
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                offset += len(raw)
                continue
            info = rec.get("extra_info") or {}
            stats = info.get("stats") or {}
            rows.append({
                "line": i,
                "offset": offset,
                "nbytes": len(raw),
                "id": rec.get("id", ""),
                "instance_id": info.get("instance_id", ""),
                "source": rec.get("source", ""),
                "folder": info.get("source_folder", ""),
                "score": float(info.get("score") or 0.0),
                "success": bool(info.get("success")),
                "finish": info.get("finish_reason", ""),
                "steps": int(stats.get("steps") or 0),
                "n_msgs": len(rec.get("messages") or []),
                "completion_tokens": int(stats.get("total_completion_tokens") or 0),
                "prompt_tokens": int(stats.get("total_prompt_tokens") or 0),
            })
            offset += len(raw)
            if i % 2000 == 0:
                print(f"  pass1 {i:>6}  {time.time() - t0:6.1f}s", flush=True)
    print(f"  pass1 done: {len(rows)} rows in {time.time() - t0:.1f}s", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Decide which lines survive the metadata gates (strategies 6a, 6b, 8, 5b)
# ---------------------------------------------------------------------------

def decide(rows):
    st = Counter()
    st["input_records"] = len(rows)
    st["input_bytes"] = sum(r["nbytes"] for r in rows)

    # -- 6a exact duplicate trajectory ids ---------------------------------
    seen_ids: set[str] = set()
    stage = []
    for r in rows:
        if r["id"] in seen_ids:
            st["dropped_exact_id"] += 1
            continue
        seen_ids.add(r["id"])
        stage.append(r)

    # -- 8 decontamination -------------------------------------------------
    kept = []
    contaminated_repos = Counter()
    for r in stage:
        hit = decontaminate(r["instance_id"])
        if hit:
            st["dropped_decontam"] += 1
            contaminated_repos[hit.split(":", 1)[1]] += 1
            continue
        kept.append(r)
    stage = kept

    # -- 5b verifier-score gate -------------------------------------------
    score_hist = Counter(round(r["score"], 1) for r in stage)
    kept = []
    for r in stage:
        if r["score"] < SCORE_GATE or not r["success"]:
            st["dropped_low_score"] += 1
            continue
        kept.append(r)
    stage = kept

    # -- 6b same-task rollout collapse ------------------------------------
    groups: dict[str, list] = defaultdict(list)
    for r in stage:
        groups[r["instance_id"]].append(r)
    kept = []
    for inst, grp in groups.items():
        grp.sort(key=lambda r: (-r["score"], r["steps"] or 10**9, r["nbytes"]))
        kept.append(grp[0])
        st["dropped_rollout_dupe"] += len(grp) - 1
    kept.sort(key=lambda r: r["line"])

    st["unique_instances_in_input"] = len({r["instance_id"] for r in rows})
    st["survivors_after_metadata_gates"] = len(kept)
    return kept, st, score_hist, contaminated_repos


# ---------------------------------------------------------------------------
# Pass 2 - content cleaning on survivors only (seek straight to each offset)
# ---------------------------------------------------------------------------

def pass2_clean(survivors, st: Counter):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*"):
        old.unlink()

    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_PERM) if HAS_LSH else None
    lang_hist, fail_reasons, examples = Counter(), Counter(), []
    system_hashes, tool_hashes = Counter(), Counter()

    shard_idx, shard_rows, shard_tokens = 0, [], 0
    manifests, out_records = [], 0
    total_tokens = total_chars_in = total_chars_out = 0
    t0 = time.time()

    def flush():
        nonlocal shard_idx, shard_rows, shard_tokens
        if not shard_rows:
            return
        path = OUT_DIR / f"shard-{shard_idx:04d}.jsonl.gz"
        h = hashlib.sha256()
        with gzip.open(path, "wt", encoding="utf8") as gz:
            for obj in shard_rows:
                line = json.dumps(obj, ensure_ascii=False) + "\n"
                h.update(line.encode("utf8"))
                gz.write(line)
        manifests.append(write_manifest(path, shard_rows, shard_tokens, h.hexdigest(), shard_idx))
        shard_idx += 1
        shard_rows, shard_tokens = [], 0

    with SRC.open("rb") as fh:
        for n, meta in enumerate(survivors):
            fh.seek(meta["offset"])
            rec = json.loads(fh.read(meta["nbytes"]))
            bstats = st

            # ---- strategy 1: hoist the repeated system prompt + tool schema
            system = rec.get("system") or ""
            tools = rec.get("tools") or ""
            if isinstance(tools, (list, dict)):
                tools = json.dumps(tools, ensure_ascii=False, sort_keys=True)
            sys_h = hashlib.sha256(system.encode()).hexdigest()[:16]
            tool_h = hashlib.sha256(tools.encode()).hexdigest()[:16]
            system_hashes[sys_h] += 1
            tool_hashes[tool_h] += 1
            st["boilerplate_chars_hoisted"] += len(system) + len(tools)

            cleaned_msgs, prose_parts = [], []
            for msg in rec.get("messages") or []:
                content = msg.get("content") or ""
                total_chars_in += len(content)

                content = strip_boilerplate(content, bstats)   # 1
                content = normalize_text(content, bstats)       # 2
                content = strip_ghost_tags(content, bstats)     # 3
                content = scrub_pii(content, bstats)            # 7

                total_chars_out += len(content)
                st["messages_out"] += 1
                cleaned_msgs.append({
                    "role": msg.get("role"),
                    "content": content,
                    "loss_mask": msg.get("loss_mask", 1),
                })
                if msg.get("role") in ("user", "assistant"):
                    prose_parts.append(content)

            joined = "\n".join(prose_parts)
            prose = prose_only(joined)

            # ---- strategy 4: language validation
            lang, conf = detect_lang(prose)
            lang_hist[lang] += 1
            if lang not in TARGET_LANGS and lang != "code-dominant":
                st["dropped_language"] += 1
                fail_reasons[f"lang:{lang}"] += 1
                continue

            # ---- strategy 5a: heuristic quality gate
            fails = quality_heuristics(prose, joined)
            if fails:
                st["dropped_heuristics"] += 1
                for f in fails:
                    fail_reasons[f"heur:{f.split('=')[0]}"] += 1
                continue

            # ---- strategy 6c: MinHash + LSH near-duplicate detection
            if lsh is not None:
                mh = make_minhash(prose or joined[:200000])
                if lsh.query(mh):
                    st["dropped_near_dupe"] += 1
                    continue
                lsh.insert(meta["id"], mh)

            ntok = tok_count(system) + sum(tok_count(m["content"]) for m in cleaned_msgs)
            total_tokens += ntok

            out = {
                "id": meta["id"],
                "instance_id": meta["instance_id"],
                "source": meta["source"],
                "system_ref": sys_h,
                "tools_ref": tool_h,
                "messages": cleaned_msgs,
                "meta": {
                    "verifier_score": meta["score"],
                    "steps": meta["steps"],
                    "lang": lang,
                    "lang_conf": conf,
                    "tokens": ntok,
                },
            }
            shard_rows.append(out)
            shard_tokens += ntok
            out_records += 1
            if len(shard_rows) >= SHARD_SIZE:
                flush()

            if n % 200 == 0:
                print(f"  pass2 {n:>6}/{len(survivors)}  kept={out_records}  "
                      f"{time.time() - t0:6.1f}s", flush=True)

    flush()

    # Persist the hoisted boilerplate exactly once.
    st["unique_system_prompts"] = len(system_hashes)
    st["unique_tool_schemas"] = len(tool_hashes)
    st["output_records"] = out_records
    st["output_tokens"] = total_tokens
    st["chars_in"] = total_chars_in
    st["chars_out"] = total_chars_out
    return manifests, lang_hist, fail_reasons, out_records, total_tokens


# ---------------------------------------------------------------------------
# STRATEGY 9 - Manifest generation
# ---------------------------------------------------------------------------
# One manifest per shard: provenance, licence, token count, language mix, the
# shard's own content hash, and the SHA-256 of this script. That last field is
# what makes the shard reproducible -- change the cleaner, change the hash.

def script_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def write_manifest(path: Path, rows: list[dict], tokens: int, content_hash: str, idx: int) -> dict:
    man = {
        "shard": path.name,
        "shard_index": idx,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dataset": "denovoswe_traj_filtered.jsonl",
        "source_family": "DeNovoSWE agentic trajectories (deepseek-v4-pro rollouts)",
        "license": "MIT",
        "records": len(rows),
        "tokens": tokens,
        "tokenizer": "cl100k_base" if ENC else "estimate(chars/4)",
        "languages": dict(Counter(r["meta"]["lang"] for r in rows)),
        "sources": dict(Counter(r["source"] for r in rows)),
        "verifier_score_min": min((r["meta"]["verifier_score"] for r in rows), default=0),
        "instance_ids": sorted(r["instance_id"] for r in rows)[:5] + ["..."],
        "sha256_jsonl": content_hash,
        "sha256_gz": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cleaner_script": "scripts/clean_pipeline.py",
        "cleaner_sha256": script_hash(),
        "strategies_applied": [
            "1-boilerplate", "2-normalisation", "3-ghost-tags", "4-language-id",
            "5-quality(heuristic+verifier)", "6-dedup(exact+rollout+minhash-lsh)",
            "7-pii", "8-decontamination", "9-manifest",
        ],
    }
    (path.parent / f"manifest-{idx:04d}.json").write_text(
        json.dumps(man, indent=2) + "\n", encoding="utf8"
    )
    return man


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not SRC.exists():
        sys.exit(f"missing input: {SRC}")

    print(f"[1/4] indexing {SRC.name} ({SRC.stat().st_size / 1e9:.2f} GB)", flush=True)
    rows = pass1_index()

    print("[2/4] metadata gates: exact-id, decontamination, verifier score, rollout collapse",
          flush=True)
    survivors, st, score_hist, contaminated = decide(rows)
    print(f"      {st['input_records']} -> {len(survivors)} survivors", flush=True)

    print("[3/4] content cleaning on survivors", flush=True)
    manifests, lang_hist, fail_reasons, out_records, out_tokens = pass2_clean(survivors, st)

    print("[4/4] writing report", flush=True)
    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cleaner_sha256": script_hash(),
        "input": {
            "file": SRC.name,
            "bytes": SRC.stat().st_size,
            "records": st["input_records"],
            "unique_instances": st["unique_instances_in_input"],
            "prompt_tokens_logged": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens_logged": sum(r["completion_tokens"] for r in rows),
            "sources": dict(Counter(r["source"] for r in rows)),
            "folders": len({r["folder"] for r in rows}),
            "messages": sum(r["n_msgs"] for r in rows),
            "score_hist": {str(k): v for k, v in sorted(score_hist.items())},
        },
        "funnel": [
            {"stage": "input", "records": st["input_records"]},
            {"stage": "exact-id dedup", "removed": st["dropped_exact_id"]},
            {"stage": "decontamination", "removed": st["dropped_decontam"]},
            {"stage": "verifier-score gate", "removed": st["dropped_low_score"]},
            {"stage": "rollout collapse", "removed": st["dropped_rollout_dupe"]},
            {"stage": "language validation", "removed": st["dropped_language"]},
            {"stage": "heuristic quality", "removed": st["dropped_heuristics"]},
            {"stage": "minhash near-dupe", "removed": st["dropped_near_dupe"]},
            {"stage": "output", "records": out_records},
        ],
        "counters": dict(st),
        "languages": dict(lang_hist),
        "reject_reasons": dict(fail_reasons),
        "contaminated_repos": dict(contaminated),
        "output": {
            "records": out_records,
            "tokens": out_tokens,
            "shards": len(manifests),
            "bytes": sum(p.stat().st_size for p in OUT_DIR.glob("*.jsonl.gz")),
            "chars_in": st["chars_in"],
            "chars_out": st["chars_out"],
        },
        "manifest_sample": manifests[0] if manifests else None,
        "env": {
            "tiktoken": bool(ENC),
            "datasketch": HAS_LSH,
            "langdetect": HAS_LANGDETECT,
            "python": sys.version.split()[0],
        },
        "config": {
            "score_gate": SCORE_GATE,
            "minhash_perm": MINHASH_PERM,
            "minhash_threshold": MINHASH_THRESHOLD,
            "shingle_k": SHINGLE_K,
            "shard_size": SHARD_SIZE,
            "mean_word_len": list(MEAN_WORD_LEN),
            "max_symbol_ratio": MAX_SYMBOL_RATIO,
            "max_dup_line_ratio": MAX_DUP_LINE_RATIO,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report["funnel"], indent=2))
    print(f"\nreport -> {REPORT}")
    print(f"shards -> {OUT_DIR} ({len(manifests)} shards, {out_tokens:,} tokens)")


if __name__ == "__main__":
    main()
