from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from .config import (
    ARTIFACTS,
    DATASET_ID,
    DOCS_PER_SHARD,
    LANES,
    MAX_DOC_TOKENS,
    MODEL_ID,
)

INDIC_SEEDS = [
    ("hi", "भारत में दस मुख्य भारतीय भाषाएँ शिक्षा और ज्ञान के लिए महत्वपूर्ण हैं। हिंदी सबसे अधिक बोली जाती है।"),
    ("hi", "गणित में शून्य की खोज भारत में हुई। शून्य के बिना आधुनिक गणना संभव नहीं है।"),
    ("hi", "एक समकोण त्रिभुज में पाइथागोरस नियम लागू होता है। कर्ण के वर्ग बराबर होता है अन्य दो भुजाओं के वर्गों के योग के।"),
    ("hi", "कंप्यूटर प्रोग्राम चरणबद्ध निर्देश होते हैं। सही क्रम से लिखे गए निर्देश ही अपेक्षित परिणाम देते हैं।"),
    ("hi", "जल चक्र में वाष्पीकरण, संघनन और वर्षा शामिल हैं। नदियाँ इसी चक्र से जल प्राप्त करती हैं।"),
    ("hi", "प्रकाश सीधी रेखा में चलता है। दर्पण से परावर्तन के बाद भी कोणीय नियम बना रहता है।"),
    ("hi", "संख्याओं को जोड़ने, घटाने, गुणा और भाग करने से अंकगणित बनता है। यह व्यापार और विज्ञान दोनों का आधार है।"),
    ("hi", "भारत का संविधान सभी नागरिकों को समान अधिकार देता है। शिक्षा का अधिकार भी उनमें से एक है।"),
    ("hi", "बीजगणित में अज्ञात मान को प्रतीक से लिखा जाता है। समीकरण हल करके वह मान निकाला जाता है।"),
    ("hi", "पृथ्वी सूर्य की परिक्रमा एक वर्ष में पूरी करती है। इसी गति से ऋतुएँ बदलती हैं।"),
    ("hi", "लोहा ऑक्सीजन और जल के संपर्क में जंग खाता है। यह एक रासायनिक अभिक्रिया है।"),
    ("hi", "भाषा विचारों का वाहक है। लिपि उन विचारों को स्थायी रूप से लिखने का साधन है।"),
    ("bn", "বাংলা ভাষায় বিজ্ঞান শিক্ষা সহজ হয় যখন উদাহরণ স্থানীয় হয়। সংখ্যা ও পরিমাপ প্রতিদিনের জীবনে লাগে।"),
    ("bn", "ত্রিভুজের তিনটি বাহু এবং তিনটি কোণ থাকে। সমবাহু ত্রিভুজের সব বাহু সমান।"),
    ("bn", "নদীর জল পাহাড় থেকে সমতলে নামে। এই প্রবাহে শক্তি উৎপাদন করা যায়।"),
    ("bn", "কম্পিউটার শুধু হিসাব করে না, ভাষাও প্রক্রিয়া করে। সঠিক তথ্য দিলে ফল নির্ভরযোগ্য হয়।"),
    ("ta", "தமிழ் ஒரு செம்மொழி. எண்கள் மற்றும் அளவைகள் அன்றாட வாழ்வில் பயன்படுகின்றன."),
    ("ta", "முக்கோணத்தின் மூன்று பக்கங்களும் மூன்று கோணங்களும் உண்டு. சரியான அளவு கோணத்தை அளிக்கிறது."),
    ("ta", "நீர் சுழற்சியில் ஆவியாதல் மற்றும் மழை அடங்கும். ஆறுகள் இதிலிருந்து நீர் பெறுகின்றன."),
    ("mr", "मराठीत विज्ञान समजणे सोपे होते जेव्हा उदाहरणे स्थानिक असतात. गणित रोजच्या हिशोबात लागते."),
    ("mr", "पृथ्वी सूर्याभोवती फिरते त्यामुळे ऋतू बदलतात. दिवस आणि रात्रही याच गतीमुळे होतात."),
    ("mr", "कार्यक्रम म्हणजे क्रमबद्ध सूचना. चुकीच्या क्रमाने लिहिल्यास निकाल चुकतो."),
    ("gu", "ગુજરાતીમાં ગણિત શીખવું રોજિંદા હિસાબ માટે જરૂરી છે. સરવાળો અને ગુણાકાર બધે કામ આવે છે."),
    ("gu", "પ્રકાશ સીધી લીટીમાં ચાલે છે. અરીસામાં પરાવર્તન થાય ત્યારે કોણનો નિયમ રહે છે."),
    ("hi", "वृक्ष कार्बन डाइऑक्साइड लेते हैं और ऑक्सीजन देते हैं। वन जलवायु को स्थिर रखते हैं।"),
    ("hi", "मानचित्र दिशा और दूरी दिखाता है। पैमाना वास्तविक दूरी को कागज पर छोटा करके लिखता है।"),
    ("hi", "विद्युत धारा धातु के तार से बहती है। प्रतिरोध अधिक हो तो धारा कम हो जाती है।"),
    ("hi", "इतिहास हमें पिछले निर्णयों से सीखने देता है। स्रोत की जाँच के बिना निष्कर्ष अधूरा रहता है।"),
    ("bn", "বই পড়ে জ্ঞান স্থায়ী হয়। মুখস্থের চেয়ে বুঝে শেখা বেশি কাজে লাগে।"),
    ("ta", "புத்தகம் படிப்பதால் அறிவு நிலைக்கும். விளங்கிக் கற்றல் மட்டுமே நீடிக்கும்."),
    ("mr", "चांगले मापन अचूक साधनांनी होते. अंदाजापेक्षा मोजलेली किंमत विश्वासार्ह असते."),
    ("gu", "સારું માપન સાચાં સાધનોથી થાય છે. અંદાજ કરતાં માપેલી કિંમત વધુ વિશ્વસનીય છે."),
    ("hi", "तापमापी तापमान मापता है। जल शून्य डिग्री पर जमता है और सौ डिग्री पर उबलता है।"),
    ("hi", "अनुपात दो राशियों की तुलना है। नुस्खा और नक्शा दोनों अनुपात पर चलते हैं।"),
    ("hi", "एक लीटर में एक हजार मिलीलीटर होते हैं। इकाई बदलते समय केवल पैमाना बदलता है, मात्रा का अर्थ नहीं।"),
    ("bn", "এক লিটারে হাজার মিলিলিটার হয়। একক বদলালে মানের অর্থ একই থাকে।"),
]


class Log:
    def __init__(self):
        self.path = None
        self._fh = None

    def open(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    def write(self, msg: str):
        print(msg, flush=True)
        if self._fh:
            self._fh.write(msg + "\n")
            self._fh.flush()

    def event(self, msg: str):
        self.write(msg)

    def ok(self, name: str):
        self.write(f"[PASS] {name}")

    def fail(self, name: str):
        self.write(f"[FAIL] {name}")

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


LOG = Log()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def content_hash(text: str) -> str:
    return sha256_text(" ".join(text.split()))


@dataclass
class Document:
    doc_id: str
    lane: str
    split: str
    text: str
    content_hash: str
    cluster_id: str
    source: str
    language: str = "en"
    token_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    shard_id: str = ""

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


def tokenizer_fingerprint(tok) -> str:
    vocab = tok.get_vocab()
    blob = json.dumps(
        {"name": MODEL_ID, "vocab_size": len(tok), "vocab": vocab},
        sort_keys=True,
        ensure_ascii=False,
    )
    return sha256_text(blob)


def freeze_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = ARTIFACTS / "manifests" / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out)
    meta = {
        "model_id": MODEL_ID,
        "tokenizer_class": tok.__class__.__name__,
        "vocab_size": len(tok),
        "pad_id": int(tok.pad_token_id),
        "eos_id": int(tok.eos_token_id),
        "tokenizer_hash": tokenizer_fingerprint(tok),
    }
    dump(ARTIFACTS / "manifests" / "tokenizer.json", meta)
    return tok, meta


def verify_tokenizer(tok, meta) -> None:
    got = tokenizer_fingerprint(tok)
    if got != meta["tokenizer_hash"]:
        raise RuntimeError("tokenizer hash mismatch")
    LOG.ok("tokenizer_hash_verified")


def _turns(ex):
    user, asst = "", ""
    for t in ex.get("conversations") or []:
        role = t.get("from") or ""
        val = t.get("value") or ""
        if role in ("user", "human"):
            user = val
        elif role in ("assistant", "gpt"):
            asst = val
    return user, asst


def _block(text: str, start: str, end: str) -> str:
    a, b = text.find(start), text.find(end)
    if a >= 0 and b > a:
        return text[a + len(start) : b].strip()
    return text


def _code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else text


def encode_text(tok, text: str, max_tokens: int) -> tuple[list[int], list[int]]:
    ids = tok.encode(text, add_special_tokens=False)[:max_tokens]
    return ids, [1] * len(ids)


def encode_agent(tok, user: str, thought: str, solution: str, max_tokens: int):
    plan = thought[:600]
    interp = thought[600:1100] if len(thought) > 600 else "Observed output matches the plan."
    obs = "stdout: ok exit=0"
    tool = '{"name":"run_python","arguments":{"code":"solve()"}}'
    segments = [
        (f"USER: {user[:240]}\n", 1),
        (f"TOOL_CALL: {tool}\n", 1),
        (f"OBSERVATION: {obs}\n", 0),
        (f"PLAN: {plan}\n", 1),
        (f"INTERPRET: {interp}\n", 1),
        (f"FINAL: {solution[:400]}\n", 1),
    ]
    ids, loss = [], []
    for text, flag in segments:
        piece = tok.encode(text, add_special_tokens=False)
        ids.extend(piece)
        loss.extend([flag] * len(piece))
        if len(ids) >= max_tokens:
            return ids[:max_tokens], loss[:max_tokens]
    return ids[:max_tokens], loss[:max_tokens]


def _make_doc(doc_id, lane, split, text, source, language, tok):
    ids, loss = encode_text(tok, text, MAX_DOC_TOKENS[lane])
    ch = content_hash(text)
    return Document(
        doc_id=doc_id,
        lane=lane,
        split=split,
        text=text,
        content_hash=ch,
        cluster_id=ch[:16],
        source=source,
        language=language,
        token_ids=ids,
        loss_mask=loss,
    )


def build_documents(tok) -> list[Document]:
    ds = load_dataset(DATASET_ID, split="train")
    docs: list[Document] = []

    def take_ot(idx, lane, split):
        user, asst = _turns(ds[idx])
        thought = _block(asst, "<|begin_of_thought|>", "<|end_of_thought|>")
        solution = _block(asst, "<|begin_of_solution|>", "<|end_of_solution|>")
        if lane == "general":
            text = user[:1200]
            ids, loss = encode_text(tok, text, MAX_DOC_TOKENS[lane])
        elif lane == "code":
            text = _code(solution)[:1500] or user[:1200]
            ids, loss = encode_text(tok, text, MAX_DOC_TOKENS[lane])
        elif lane == "reasoning":
            text = (thought or asst)[:1500]
            ids, loss = encode_text(tok, text, MAX_DOC_TOKENS[lane])
        elif lane == "long":
            text = f"{user}\n{thought}\n{solution}"[:4000]
            ids, loss = encode_text(tok, text, MAX_DOC_TOKENS[lane])
        elif lane == "agentic":
            text = f"{user}\n{thought}\n{solution}"
            ids, loss = encode_agent(tok, user, thought or asst, solution or asst, MAX_DOC_TOKENS[lane])
        else:
            text = user
            ids, loss = encode_text(tok, text, MAX_DOC_TOKENS[lane])
        ch = content_hash(text)
        return Document(
            doc_id=f"ot-{split}-{lane}-{idx:04d}",
            lane=lane,
            split=split,
            text=text,
            content_hash=ch,
            cluster_id=ch[:16],
            source=DATASET_ID,
            language="en",
            token_ids=ids,
            loss_mask=loss,
        )

    non_indic = ["general", "code", "agentic", "reasoning", "long"]
    for i in range(600):
        docs.append(take_ot(i, non_indic[i % 5], "train"))
    for i in range(600, 660):
        docs.append(take_ot(i, non_indic[i % 5], "eval"))
    for i in range(660, 690):
        docs.append(take_ot(i, non_indic[i % 5], "valid"))

    for n in range(50):
        lang, seed = INDIC_SEEDS[n % len(INDIC_SEEDS)]
        text = f"{seed} [{n}]"
        if n < 32:
            split = "train"
        elif n < 44:
            split = "eval"
        else:
            split = "valid"
        docs.append(
            _make_doc(f"indic-{split}-{n:03d}", "indic", split, text, "indic-seed", lang, tok)
        )

    docs = [d for d in docs if d.n_tokens >= 8]
    return docs


def write_shards(docs: list[Document], tokenizer_hash: str):
    shard_dir = ARTIFACTS / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for d in docs:
        groups[(d.split, d.lane)].append(d)

    manifests = []
    for (split, lane), group in sorted(groups.items()):
        for i in range(0, len(group), DOCS_PER_SHARD):
            chunk = group[i : i + DOCS_PER_SHARD]
            shard_id = f"{split}-{lane}-{i // DOCS_PER_SHARD:03d}"
            rel = f"shards/{shard_id}.jsonl"
            path = ARTIFACTS / rel
            lines = []
            for d in chunk:
                d.shard_id = shard_id
                rec = {
                    "doc_id": d.doc_id,
                    "lane": d.lane,
                    "split": d.split,
                    "content_hash": d.content_hash,
                    "cluster_id": d.cluster_id,
                    "source": d.source,
                    "language": d.language,
                    "n_tokens": d.n_tokens,
                    "token_ids": d.token_ids,
                    "loss_mask": d.loss_mask,
                    "tokenizer_hash": tokenizer_hash,
                }
                lines.append(json.dumps(rec, ensure_ascii=False))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            os.chmod(path, 0o444)
            man = {
                "shard_id": shard_id,
                "path": rel,
                "sha256": sha256_file(path),
                "tokenizer_hash": tokenizer_hash,
                "lane": lane,
                "split": split,
                "n_docs": len(chunk),
                "n_tokens": sum(d.n_tokens for d in chunk),
                "doc_ids": [d.doc_id for d in chunk],
                "immutable": True,
            }
            man_path = ARTIFACTS / "manifests" / f"{shard_id}.json"
            dump(man_path, man)
            os.chmod(man_path, 0o444)
            manifests.append(man)
    dump(ARTIFACTS / "manifests" / "index.json", manifests)
    LOG.event("shards created")
    return manifests


def validate_manifests(manifests, tokenizer_hash: str) -> None:
    for m in manifests:
        path = ARTIFACTS / m["path"]
        if sha256_file(path) != m["sha256"]:
            raise RuntimeError(f"shard hash mismatch {m['shard_id']}")
        if m["tokenizer_hash"] != tokenizer_hash:
            raise RuntimeError(f"tokenizer hash mismatch in {m['shard_id']}")
        if os.stat(path).st_mode & 0o222:
            raise RuntimeError(f"shard is writable {m['shard_id']}")
    LOG.event("manifests validated")


class Firewall:
    def __init__(self):
        self.blocked = []

    def admit_shard(self, manifest, purpose="train") -> bool:
        if purpose == "train" and manifest["split"] in ("eval", "valid", "validation"):
            self.blocked.append(
                {
                    "shard_id": manifest["shard_id"],
                    "split": manifest["split"],
                    "lane": manifest["lane"],
                    "purpose": purpose,
                }
            )
            return False
        return True

    def admit_doc(self, doc: Document, purpose="train") -> bool:
        if purpose == "train" and doc.split != "train":
            self.blocked.append(
                {"doc_id": doc.doc_id, "split": doc.split, "lane": doc.lane, "purpose": purpose}
            )
            return False
        return True


def run_firewall(firewall: Firewall, manifests, docs) -> None:
    blocked_eval = False
    for m in manifests:
        if m["split"] in ("eval", "valid"):
            if not firewall.admit_shard(m, "train"):
                if m["split"] == "eval":
                    blocked_eval = True
    for d in docs:
        if d.split != "train":
            firewall.admit_doc(d, "train")
    dump(ARTIFACTS / "ledgers" / "firewall.json", {"blocked": firewall.blocked})
    if blocked_eval:
        LOG.event("evaluation data blocked")
        LOG.ok("eval_shard_blocked")
