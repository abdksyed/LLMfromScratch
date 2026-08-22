from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from .config import ARTIFACTS, MODEL_ID, TRAIN_LR
from .core import LOG, dump, sha256_bytes


class SimulatedCrash(Exception):
    pass


class Jsonl:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def append(self, rec: dict):
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Engine:
    def __init__(self, stream, ledger_dir: Path | None = None):
        self.stream = stream
        self.device = pick_device()
        self.model = None
        self.opt = None
        self.ledger_dir = Path(ledger_dir or (ARTIFACTS / "ledgers"))
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.consumption = Jsonl(self.ledger_dir / "consumption.jsonl")
        self.learning = Jsonl(self.ledger_dir / "learning.jsonl")
        self.train_seconds = 0.0
        self.train_loss_tokens = 0
        self.steps = 0

    def close(self):
        self.consumption.close()
        self.learning.close()

    def load_model(self):
        LOG.write(f"loading {MODEL_ID} on {self.device}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float32, trust_remote_code=True
        )
        model.config.use_cache = False
        for name, p in model.named_parameters():
            p.requires_grad = "layers.27" in name
        model.to(self.device)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        self.model = model
        self.opt = torch.optim.AdamW(params, lr=TRAIN_LR)
        n = sum(p.numel() for p in params)
        LOG.write(f"trainable parameters: {n}")

    def _trainable(self):
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items() if "layers.27" in k}

    def save_checkpoint(self, tag: str, extra: dict | None = None) -> Path:
        ckpt_dir = ARTIFACTS / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "tag": tag,
            "next_batch_id": self.stream.batch_id,
            "stream": self.stream.state_dict(),
            "trainable": self._trainable(),
            "optim": self.opt.state_dict(),
        }
        if extra:
            payload.update(extra)
        pt = ckpt_dir / f"{tag}.pt"
        torch.save(payload, pt)
        meta = {
            "tag": tag,
            "path": str(pt.relative_to(ARTIFACTS)),
            "next_batch_id": self.stream.batch_id,
            "ledger_offset": self.stream.batch_id,
        }
        if extra:
            for k, v in extra.items():
                if k in ("expected_next_batch_hash", "expected_next_batch_id", "expected_spans"):
                    meta[k] = v
        dump(ckpt_dir / f"{tag}.json", meta)
        LOG.event("checkpoint saved")
        LOG.ok("checkpoint_saved")
        return pt

    def restore(self, tag: str):
        pt = ARTIFACTS / "checkpoints" / f"{tag}.pt"
        payload = torch.load(pt, map_location=self.device, weights_only=False)
        self.stream.load_state_dict(payload["stream"])
        self.model.load_state_dict(payload["trainable"], strict=False)
        self.opt.load_state_dict(payload["optim"])
        LOG.write(f"restored {tag} next_batch_id={self.stream.batch_id}")
        return payload

    def train_batch(self, batch) -> dict:
        t0 = time.perf_counter()
        ids = torch.tensor(batch.input_ids, device=self.device)
        attn = torch.tensor(batch.attention_mask, device=self.device)
        pos = torch.tensor(batch.position_ids, device=self.device)
        lm = torch.tensor(batch.loss_mask, device=self.device, dtype=torch.float32)
        out = self.model(input_ids=ids, attention_mask=attn, position_ids=pos)
        logits = out.logits[:, :-1].contiguous()
        targets = ids[:, 1:].contiguous()
        mask = lm[:, 1:].contiguous()
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        ce = ce * mask
        denom = mask.sum().clamp(min=1.0)
        loss = ce.sum() / denom
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        token_losses = ce.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        span_losses = []
        for sp in batch.spans:
            s = max(sp["start"] - 1, 0)
            e = max(sp["end"] - 1, 0)
            sl = token_losses[sp["seq"], s:e]
            sm = mask_np[sp["seq"], s:e]
            n = float(sm.sum())
            mean = float((sl.sum() / n) if n else 0.0)
            span_losses.append(
                {
                    "doc_id": sp["doc_id"],
                    "content_hash": sp["content_hash"],
                    "lane": sp["lane"],
                    "src_start": sp["src_start"],
                    "src_end": sp["src_end"],
                    "mean_loss": mean,
                    "n": int(n),
                }
            )
        rec = {
            "batch_id": batch.batch_id,
            "stage": batch.stage,
            "mean_loss": float(loss.detach().cpu()),
            "n_loss_bearing": int(denom.detach().cpu()),
            "token_loss_hash": sha256_bytes(np.ascontiguousarray(token_losses).tobytes()),
            "span_losses": span_losses,
        }
        dt = time.perf_counter() - t0
        self.train_seconds += dt
        self.train_loss_tokens += rec["n_loss_bearing"]
        self.steps += 1
        self.consumption.append(batch.to_meta())
        self.learning.append(rec)
        LOG.write(
            f"batch {batch.batch_id} stage={batch.stage} loss={rec['mean_loss']:.4f} "
            f"loss_tokens={rec['n_loss_bearing']} util={batch.utilization:.3f} hash={batch.hash()[:12]}"
        )
        return rec
