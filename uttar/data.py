"""KVME cases, tokenisers and cached frozen image features for UTTAR."""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from tokenizers import AddedToken
from transformers import AutoModel, AutoTokenizer

from schema import QUESTIONS, extract

ORGANS = [("Breast", "new_Mammary2.json", "Mammary_report"),
          ("Thyroid", "new_Thyroid2.json", "Thyroid_report"),
          ("Liver", "new_Liver2.json", "Liver_report")]
# Frozen text encoder f_t: SapBERT-XLMR, a clinical (UMLS) cross-lingual encoder that reads Chinese and English.
TEXT_MODEL = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
# SapBERT-XLMR uses the xlm-roberta-base sentencepiece vocabulary unchanged (checked piece by piece); the official
# fast tokenizer of xlm-roberta-base is loaded because the repo ships only the slow sentencepiece file.
TEXT_TOKENIZER = "FacebookAI/xlm-roberta-base"
PLACEHOLDERS = ["_2DS_", "_3DS_", "_SCM_", "_SMM_", "_Loc_", "_LocR_", "_r_"]
QSEP = "; "  # separator between planned questions (paper: ";"-separated list)
MAX_REPORT = 256
MAX_PLAN = 320


def load_cases(root):
    """All KVME cases with their official split, report, image paths and extracted QA targets."""
    root = Path(root)
    cases = []
    for organ, js, folder in ORGANS:
        j = json.load(open(root / js, encoding="utf-8-sig"))
        for split in ("train", "val", "test"):
            for e in j[split]:
                report = re.sub(r"\s+", "", e["finding"])  # KVME reports have stray trailing spaces
                cases.append({"organ": organ, "split": split, "uid": e["uid"], "report": report,
                              "paths": [str(root / folder / p) for p in e["image_path"]],
                              "qa": extract(report, organ)})
    return cases


def ft_tokenizer():
    """Tokenizer of the frozen text encoder f_t (unchanged xlm-roberta-base vocabulary)."""
    return AutoTokenizer.from_pretrained(TEXT_TOKENIZER)


def ft_model():
    return AutoModel.from_pretrained(TEXT_MODEL)


def plan_target(qa):
    """Serialised question list C* (schema order) for one case's instantiated questions."""
    return QSEP.join(q.en for q in QUESTIONS if q.qid in qa)


class DecTokenizer:
    """Shared f_dec / f_plan vocabulary.

    Base: the f_t (xlm-roberta-base) vocabulary. Added tokens (matched before normalisation): KVME's
    placeholders, every character of the training reports that the tokenizer's NFKC normalisation would rewrite
    (full-width ，；：（）＞ would otherwise come back half-width) and any training-report string mapped to <unk>.
    The softmax is restricted to the tokens that occur in the training reports and the question templates
    (250k -> ~2k); everything else can never be a target, so this only removes dead output rows.
    """

    def __init__(self, train_reports):
        self.tok = ft_tokenizer()
        self.tok.add_tokens([AddedToken(p, normalized=False) for p in PLACEHOLDERS])
        nfkc = sorted({c for r in train_reports for c in r if unicodedata.normalize("NFKC", c) != c})
        self.tok.add_tokens([AddedToken(c, normalized=False) for c in nfkc])
        unk = set()
        for r in train_reports:
            enc = self.tok(r, add_special_tokens=False, return_offsets_mapping=True)
            for i, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
                if i == self.tok.unk_token_id:
                    unk.add(r[a:b])
        self.added = PLACEHOLDERS + nfkc + sorted(unk)
        self.tok.add_tokens([AddedToken(w, normalized=False) for w in sorted(unk)])
        used = {self.tok.pad_token_id, self.tok.bos_token_id, self.tok.eos_token_id, self.tok.unk_token_id}
        for r in train_reports:
            used.update(self.tok(r, add_special_tokens=False)["input_ids"])
        for q in QUESTIONS:
            used.update(self.tok(q.en + QSEP, add_special_tokens=False)["input_ids"])
        self.c2f = sorted(used)
        self.f2c = {f: c for c, f in enumerate(self.c2f)}
        self.pad, self.bos, self.eos = (self.f2c[self.tok.pad_token_id], self.f2c[self.tok.bos_token_id],
                                        self.f2c[self.tok.eos_token_id])
        self.unk = self.f2c[self.tok.unk_token_id]
        self.vocab_size = len(self.c2f)

    def encode(self, text, max_len):
        ids = self.tok(text, add_special_tokens=False)["input_ids"][: max_len - 2]
        return [self.bos] + [self.f2c.get(i, self.unk) for i in ids] + [self.eos]

    def decode(self, ids, report=True):
        out = []
        for i in ids:
            if i == self.eos:
                break
            if i in (self.pad, self.bos):
                continue
            out.append(self.c2f[i])
        text = self.tok.decode(out, skip_special_tokens=True)
        return text.replace(" ", "") if report else text.strip()  # KVME reports contain no spaces


def image_index(cases):
    paths = sorted({p for c in cases for p in c["paths"]})
    return {p: i for i, p in enumerate(paths)}


def load_patch_tokens(cache_dir):
    cache_dir = Path(cache_dir)
    feats = np.load(cache_dir / "patch_tokens.npy", mmap_mode="r")
    index = json.loads((cache_dir / "patch_index.json").read_text())
    return feats, index
