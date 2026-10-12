"""Evaluation metrics shared by training (validation selection) and evaluate.py.

NLG follows the KVME code (jieba + key_technical_words.txt user dictionary, space-joined, pycocoevalcap), the same
recipe as baselines.py. CE follows the KVME multi-label entity strategy with the schema
extractor E as the deterministic entity matcher. QA compares Yes/No answers with the extracted (pseudo) labels.
"""
from pathlib import Path

import jieba
import numpy as np
from sklearn.metrics import f1_score

from schema import ORGAN_QIDS

_DICT_LOADED = False


def init_jieba(kvme_root):
    global _DICT_LOADED
    if not _DICT_LOADED:
        jieba.setLogLevel(60)
        jieba.load_userdict(str(Path(kvme_root) / "key_technical_words.txt"))
        _DICT_LOADED = True


def tok(s):
    return " ".join(w for w in jieba.lcut(s) if w.strip())


def nlg(refs, hyps, full=True):
    from pycocoevalcap.bleu.bleu import Bleu
    gts = {i: [tok(r)] for i, r in enumerate(refs)}
    res = {i: [tok(h) if h.strip() else "<empty>"] for i, h in enumerate(hyps)}
    out = {}
    b, _ = Bleu(4).compute_score(gts, res, verbose=0)
    out.update({f"BLEU-{n + 1}": b[n] for n in range(4)})
    if full:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        out["METEOR"], _ = Meteor().compute_score(gts, res)
        out["ROUGE-L"], _ = Rouge().compute_score(gts, res)
        out["CIDEr"], _ = Cider().compute_score(gts, res)
    return {k: round(float(v) * 100, 2) for k, v in out.items()}


def rouge_l_per_case(refs, hyps):
    from pycocoevalcap.rouge.rouge import Rouge
    r = Rouge()
    return np.array([r.calc_score([tok(h) if h.strip() else "<empty>"], [tok(g)]) for g, h in zip(refs, hyps)])


def ce(ref_qas, hyp_qas, organs):
    """KVME-style multi-label CE. Entities = (question, Yes) and (question, No) for the organ's schema questions;
    y_e(R) = 1 iff E finds that statement in R. Opposite values are separate entities (never merged)."""
    tp = fp = fn = tn = 0
    per = {}
    case_f1 = []
    for ref, hyp, organ in zip(ref_qas, hyp_qas, organs):
        r = {(q, a) for q, a in ref.items()}
        h = {(q, a) for q, a in hyp.items()}
        ctp = cfp = cfn = 0
        for qid in ORGAN_QIDS[organ]:
            for a in (1, 0):
                e = (qid, a)
                yr, yh = e in r, e in h
                s = per.setdefault(e, [0, 0, 0])
                if yr and yh:
                    tp += 1; ctp += 1; s[0] += 1
                elif yh:
                    fp += 1; cfp += 1; s[1] += 1
                elif yr:
                    fn += 1; cfn += 1; s[2] += 1
                else:
                    tn += 1
        case_f1.append(2 * ctp / (2 * ctp + cfp + cfn) if ctp + cfp + cfn else 1.0)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    f1s = [2 * a / (2 * a + b + c) for a, b, c in per.values() if a + b + c]
    out = {"CE-P": P, "CE-R": R, "CE-F1": F, "CE-Acc": (tp + tn) / (tp + fp + fn + tn), "CE-macroF1": float(np.mean(f1s))}
    return {k: round(v * 100, 2) for k, v in out.items()}, np.array(case_f1)


def qa(items):
    """items: list of (qid, gold, pred) with gold/pred in {0, 1}. Yes is the positive class.
    Macro-F1 = mean over question types of the per-type macro-F1 over the labels present (sklearn default)."""
    g = np.array([x[1] for x in items]); p = np.array([x[2] for x in items])
    tp = int(((g == 1) & (p == 1)).sum()); fp = int(((g == 0) & (p == 1)).sum()); fn = int(((g == 1) & (p == 0)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    by = {}
    for q, a, b in items:
        by.setdefault(q, ([], []))
        by[q][0].append(a); by[q][1].append(b)
    macro = float(np.mean([f1_score(a, b, average="macro") for a, b in by.values()]))
    out = {"QA-Acc": float((g == p).mean()), "QA-P": P, "QA-R": R, "QA-F1": 2 * P * R / (P + R) if P + R else 0.0,
           "QA-macroF1": macro}
    return {k: round(v * 100, 2) for k, v in out.items()} | {"n": len(items)}
