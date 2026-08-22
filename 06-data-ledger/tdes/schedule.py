from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .config import (
    ARTIFACTS,
    BATCH_SIZE,
    EXPOSURE_CAP,
    FLOORS,
    LANES,
    OPUS_TEMPERATURE,
    POLICIES,
    SEQ_LEN,
    STAGES,
)
from .core import LOG, dump


def _int_share(fracs: dict[str, float], total: int) -> dict[str, int]:
    raw = {k: frac * total for k, frac in fracs.items()}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    remain = total - sum(out.values())
    order = sorted(raw, key=lambda k: (raw[k] - out[k], k), reverse=True)
    for i in range(remain):
        out[order[i % len(order)]] += 1
    return out


def compile_mixture():
    T = BATCH_SIZE * SEQ_LEN
    floor_total = int(round(sum(FLOORS.values()) * T))
    norm_floors = {k: v / sum(FLOORS.values()) for k, v in FLOORS.items()}
    floor_tokens = _int_share(norm_floors, floor_total)
    stages = []
    for st in STAGES:
        residual = T - sum(floor_tokens.values())
        residual_tokens = _int_share(st["shares"], residual)
        planned = {l: floor_tokens[l] + residual_tokens[l] for l in LANES}
        stages.append(
            {
                "name": st["name"],
                "n_batches": st["n_batches"],
                "shares": st["shares"],
                "floor_tokens": floor_tokens,
                "residual_tokens": residual_tokens,
                "planned_tokens": planned,
                "seq_len": SEQ_LEN,
                "batch_size": BATCH_SIZE,
            }
        )
    mix = {"token_budget_per_batch": T, "floors": FLOORS, "stages": stages}
    dump(ARTIFACTS / "manifests" / "mixture.json", mix)
    LOG.event("mixture compiled")
    return mix


def sketch(token_ids, dim=64) -> np.ndarray:
    s = np.zeros(dim, np.float64)
    for t in token_ids:
        h = hashlib.sha256(int(t).to_bytes(8, "little", signed=False)).digest()
        idx = int.from_bytes(h[:2], "little") % dim
        sign = 1.0 if h[2] % 2 == 0 else -1.0
        s[idx] += sign
    n = np.linalg.norm(s)
    return s / n if n else s


class OPUS:
    def __init__(self, proxy_docs, rng: np.random.Generator, temperature=OPUS_TEMPERATURE):
        self.rng = rng
        self.T = temperature
        sk = [sketch(d.token_ids) for d in proxy_docs if d.n_tokens]
        self.proxy = np.mean(sk, axis=0) if sk else np.zeros(64)
        n = np.linalg.norm(self.proxy)
        if n:
            self.proxy = self.proxy / n
        self.exposure = Counter()
        self.decisions = []

    def utility(self, doc, in_batch) -> float:
        align = float(sketch(doc.token_ids) @ self.proxy)
        uniq = len(set(doc.token_ids)) / max(1, doc.n_tokens)
        red = 0.6 if doc.cluster_id in in_batch else 0.0
        return align * uniq / (1.0 + self.exposure[doc.cluster_id]) - red

    def decide(self, doc, *, floor=False, lane_full=False, in_batch=()) -> str:
        u = self.utility(doc, in_batch)
        rec = {
            "decision_id": len(self.decisions),
            "doc_id": doc.doc_id,
            "lane": doc.lane,
            "cluster_id": doc.cluster_id,
            "utility": round(u, 6),
            "exposure": int(self.exposure[doc.cluster_id]),
            "floor": floor,
        }
        if lane_full:
            rec["decision"] = "defer"
            rec["reason"] = "lane_budget_full"
            self.decisions.append(rec)
            return "defer"
        if self.exposure[doc.cluster_id] >= EXPOSURE_CAP and not floor:
            rec["decision"] = "reject"
            rec["reason"] = "exposure_cap"
            self.decisions.append(rec)
            return "reject"
        p = 1.0 / (1.0 + math.exp(-u / max(self.T, 1e-6)))
        draw = float(self.rng.random())
        rec["p"] = round(p, 6)
        rec["draw"] = round(draw, 6)
        if floor and (draw > p or self.exposure[doc.cluster_id] >= EXPOSURE_CAP):
            rec["decision"] = "floor_override"
            rec["reason"] = "protected_floor"
            self.decisions.append(rec)
            return "floor_override"
        if draw <= p:
            rec["decision"] = "accept"
            rec["reason"] = "boltzmann"
            self.decisions.append(rec)
            return "accept"
        rec["decision"] = "reject"
        rec["reason"] = "low_utility"
        self.decisions.append(rec)
        return "reject"

    def mark(self, cluster_id: str) -> None:
        self.exposure[cluster_id] += 1


@dataclass
class Piece:
    doc: object
    src_start: int
    src_end: int
    policy: str


@dataclass
class PackedBatch:
    batch_id: int
    stage: str
    input_ids: np.ndarray
    attention_mask: np.ndarray
    position_ids: np.ndarray
    loss_mask: np.ndarray
    spans: list
    lane_tokens: dict
    lane_loss_tokens: dict
    planned_tokens: dict
    opus_decision_ids: list
    utilization: float
    loss_bearing: int

    def hash(self) -> str:
        h = hashlib.sha256()
        for a in (self.input_ids, self.attention_mask, self.position_ids, self.loss_mask):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def to_meta(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "stage": self.stage,
            "batch_hash": self.hash(),
            "n_input_tokens": int(self.input_ids.size),
            "n_real_tokens": int(self.attention_mask.sum()),
            "n_loss_bearing": int(self.loss_bearing),
            "lane_tokens": self.lane_tokens,
            "lane_loss_tokens": self.lane_loss_tokens,
            "planned_tokens": self.planned_tokens,
            "spans": self.spans,
            "opus_decision_ids": self.opus_decision_ids,
            "packing_utilization": self.utilization,
        }


def pack_pieces(pieces: list[Piece], pad_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    B, T = BATCH_SIZE, SEQ_LEN
    ids = np.full((B, T), pad_id, np.int64)
    attn = np.zeros((B, T), np.int64)
    pos = np.zeros((B, T), np.int64)
    loss = np.zeros((B, T), np.int64)
    spans = []
    b = 0
    cur = 0
    for piece in pieces:
        doc = piece.doc
        src = doc.token_ids[piece.src_start : piece.src_end]
        lm = doc.loss_mask[piece.src_start : piece.src_end]
        n = len(src)
        if n == 0:
            continue
        if cur >= T:
            b += 1
            cur = 0
        if b >= B:
            break
        n = min(n, T - cur)
        ids[b, cur : cur + n] = src[:n]
        attn[b, cur : cur + n] = 1
        pos[b, cur : cur + n] = np.arange(n)
        loss[b, cur : cur + n] = lm[:n]
        loss[b, cur] = 0
        spans.append(
            {
                "doc_id": doc.doc_id,
                "shard_id": doc.shard_id,
                "content_hash": doc.content_hash,
                "lane": doc.lane,
                "cluster_id": doc.cluster_id,
                "seq": b,
                "start": cur,
                "end": cur + n,
                "src_start": piece.src_start,
                "src_end": piece.src_start + n,
                "loss_bearing": int(loss[b, cur : cur + n].sum()),
                "policy": piece.policy,
            }
        )
        cur += n
    return ids, attn, pos, loss, spans


class LaneCursor:
    def __init__(self, docs):
        self.docs = docs
        self.i = 0
        self.off = 0
        self.epoch = 0

    def state(self):
        return {"i": self.i, "off": self.off, "epoch": self.epoch}

    def load(self, s):
        self.i, self.off, self.epoch = s["i"], s["off"], s["epoch"]

    def _wrap(self):
        if self.i >= len(self.docs):
            self.i = 0
            self.off = 0
            self.epoch += 1

    def peek(self):
        if not self.docs:
            return None
        self._wrap()
        return self.docs[self.i]

    def skip(self):
        self.i += 1
        self.off = 0
        self._wrap()

    def consume(self, n: int) -> list[Piece]:
        out = []
        left = n
        while left > 0 and self.docs:
            self._wrap()
            doc = self.docs[self.i]
            avail = doc.n_tokens - self.off
            if avail <= 0:
                self.skip()
                continue
            take = min(left, avail)
            out.append(Piece(doc, self.off, self.off + take, POLICIES[doc.lane]))
            self.off += take
            left -= take
            if self.off >= doc.n_tokens:
                self.i += 1
                self.off = 0
        return out


class BatchStream:
    def __init__(self, train_docs, mix, pad_id: int, rng: np.random.Generator, opus: OPUS, firewall):
        self.pad_id = pad_id
        self.rng = rng
        self.opus = opus
        self.firewall = firewall
        self.mix = mix
        self.batch_id = 0
        self.stage_idx = 0
        by_lane = defaultdict(list)
        for d in train_docs:
            if firewall.admit_doc(d, "train"):
                by_lane[d.lane].append(d)
        self.cursors = {lane: LaneCursor(by_lane[lane]) for lane in LANES}

    def stage(self):
        return self.mix["stages"][self.stage_idx]

    def state_dict(self):
        return {
            "batch_id": self.batch_id,
            "stage_idx": self.stage_idx,
            "cursors": {l: c.state() for l, c in self.cursors.items()},
            "rng": self.rng.bit_generator.state,
            "exposure": dict(self.opus.exposure),
            "n_decisions": len(self.opus.decisions),
            "temperature": self.opus.T,
        }

    def load_state_dict(self, s):
        self.batch_id = s["batch_id"]
        self.stage_idx = s["stage_idx"]
        for l, st in s["cursors"].items():
            self.cursors[l].load(st)
        self.rng.bit_generator.state = s["rng"]
        self.opus.exposure = Counter(s["exposure"])
        self.opus.decisions = self.opus.decisions[: s["n_decisions"]]
        self.opus.T = s.get("temperature", self.opus.T)

    def _take(self, lane, need, floor, in_batch) -> list[Piece]:
        pieces = []
        got = 0
        attempts = 0
        while got < need and attempts < 48:
            attempts += 1
            cur = self.cursors[lane]
            doc = cur.peek()
            if doc is None:
                break
            if not self.firewall.admit_doc(doc, "train"):
                cur.skip()
                continue
            if cur.off > 0:
                take = min(need - got, doc.n_tokens - cur.off)
                pieces.extend(cur.consume(take))
                got += take
                continue
            decision = self.opus.decide(doc, floor=floor, lane_full=False, in_batch=in_batch)
            if decision in ("accept", "floor_override"):
                take = min(need - got, doc.n_tokens)
                pieces.extend(cur.consume(take))
                got += take
                self.opus.mark(doc.cluster_id)
                in_batch.add(doc.cluster_id)
            else:
                cur.skip()
        extra = self.cursors[lane].peek()
        if extra is not None and got >= need:
            self.opus.decide(extra, floor=False, lane_full=True, in_batch=in_batch)
        return pieces

    def next_batch(self) -> PackedBatch:
        st = self.stage()
        in_batch = set()
        selected: list[Piece] = []
        start_dec = len(self.opus.decisions)
        for lane in LANES:
            selected.extend(self._take(lane, st["floor_tokens"][lane], True, in_batch))
        for lane in LANES:
            selected.extend(self._take(lane, st["residual_tokens"][lane], False, in_batch))
        ids, attn, pos, loss, spans = pack_pieces(selected, self.pad_id)
        lane_tokens = {l: 0 for l in LANES}
        lane_loss = {l: 0 for l in LANES}
        for sp in spans:
            n = sp["end"] - sp["start"]
            lane_tokens[sp["lane"]] += n
            lane_loss[sp["lane"]] += sp["loss_bearing"]
        real = int(attn.sum())
        lb = int(loss.sum())
        batch = PackedBatch(
            batch_id=self.batch_id,
            stage=st["name"],
            input_ids=ids,
            attention_mask=attn,
            position_ids=pos,
            loss_mask=loss,
            spans=spans,
            lane_tokens=lane_tokens,
            lane_loss_tokens=lane_loss,
            planned_tokens=st["planned_tokens"],
            opus_decision_ids=list(range(start_dec, len(self.opus.decisions))),
            utilization=real / ids.size if ids.size else 0.0,
            loss_bearing=lb,
        )
        self.batch_id += 1
        done = sum(s["n_batches"] for s in self.mix["stages"][: self.stage_idx + 1])
        if self.batch_id >= done and self.stage_idx + 1 < len(self.mix["stages"]):
            self.stage_idx += 1
        return batch

    def peek(self) -> PackedBatch:
        snap = self.state_dict()
        batch = self.next_batch()
        self.load_state_dict(snap)
        return batch
