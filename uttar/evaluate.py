"""Evaluate UTTAR predictions (and copy-a-report baselines) on a KVME split, following the protocol.

NLG   : BLEU-1..4, METEOR, ROUGE-L, CIDEr (KVME jieba recipe) + BERTScore, on R̂int; also on test references that are /
        are not verbatim training reports ("duplicate" / "novel").
CE    : KVME-style multi-label entity P/R/F1/Accuracy (+ macro-F1); entities found by the schema extractor E,
        identically for predictions and references. Scored for R̂int (text) and R̂QA (verified answers).
QA    : verifier answers vs the extracted (pseudo) labels: on the reference's instantiated questions (oracle
        questions) and on the planned questions; planning precision/recall; QA-consistency before the filter.
NLI   : entity-level entailment (protocol §7, KVME analysis): for every schema question instantiated by E in both the
        reference and the generated report, the entity statement is the sentence(s) that instantiate it; a frozen
        multilingual DeBERTa-v3 NLI model (premise = reference statement, hypothesis = generated statement) decides
        entailment (argmax label = entailment). Entailment rate = entailing / evaluable pairs.
Stats : paired t-test UTTAR vs the nearest-report baseline on per-case ROUGE-L and per-case CE-F1;
        bootstrap 95% CIs (1,000 case resamples) for CE-F1 and QA-F1.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_cases
from metrics import ce, init_jieba, nlg, qa, rouge_l_per_case
from schema import PARENT_CHILD, apply_parent_filter, extract

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  # mDeBERTa-v3-base fine-tuned on MNLI + XNLI (incl. Chinese)
BERTSCORE_MODEL = "google-bert/bert-base-chinese"  # bert_score's default model for Chinese (lang="zh"), layer 8


def model_path(repo):
    """Local path of a Hugging Face model (downloaded on first use; weights + tokenizer files only)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])


def bertscore(refs, hyps):
    from bert_score import score
    path = model_path(BERTSCORE_MODEL)
    _, _, f = score([h if h else "无" for h in hyps], refs, model_type=path, num_layers=8, lang="zh", batch_size=64,
                    device="cuda")
    return round(float(f.mean()) * 100, 2)


def statements(report, organ):
    """question id -> the sentence(s) of the report that instantiate it (E run sentence by sentence)."""
    out = {}
    for sent in [x for x in report.split("。") if x]:
        for q in extract(sent, organ):
            out.setdefault(q, []).append(sent)
    return {q: "。".join(v) + "。" for q, v in out.items()}


class NLI:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = model_path(NLI_MODEL)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).cuda().eval()
        self.ent = {v: int(k) for k, v in self.model.config.id2label.items()}["entailment"]
        self.torch = torch

    def __call__(self, refs, hyps, organs, bs=128):
        pairs = []  # (qid, premise, hypothesis)
        for r, h, o in zip(refs, hyps, organs):
            sr, sh = statements(r, o), statements(h, o)
            pairs += [(q, sr[q], sh[q]) for q in sr if q in sh]
        lab = []
        with self.torch.no_grad():
            for i in range(0, len(pairs), bs):
                b = pairs[i:i + bs]
                enc = self.tok([x[1] for x in b], [x[2] for x in b], truncation=True, max_length=256, padding=True,
                               return_tensors="pt").to("cuda")
                lab += (self.model(**enc).logits.argmax(-1) == self.ent).tolist()
        per = {}
        for (q, _, _), e in zip(pairs, lab):
            per.setdefault(q, [0, 0]); per[q][0] += e; per[q][1] += 1
        return {"entailment_rate": round(100 * sum(lab) / max(len(lab), 1), 2), "correct_entailments": int(sum(lab)),
                "evaluable_pairs": len(lab), "per_question": {q: {"entailed": a, "evaluable": n, "rate": round(100 * a / n, 1)}
                                                              for q, (a, n) in sorted(per.items())}}


def gold_items(recs):
    items = []
    for r in recs:
        ans = apply_parent_filter({q: int(p >= 0.5) for q, p in r["gold_p"].items()})
        items += [(q, r["gold_qa"][q], ans[q]) for q in r["gold_p"]]
    return items


def planned_stats(recs):
    hit = n_plan = n_gold = 0
    items, cons, n_pairs = [], 0, 0
    for r in recs:
        g, pl = r["gold_qa"], r["planned"]
        n_plan += len(pl); n_gold += len(g); hit += len(set(pl) & set(g))
        items += [(q, g[q], r["answers"][q]) for q in pl if q in g]
        for p, c in PARENT_CHILD:
            if p in r["p"] and c in r["p"]:
                n_pairs += 1
                cons += not (r["p"][c] >= 0.5 and r["p"][p] < 0.5)
    out = {"plan-P": round(100 * hit / max(n_plan, 1), 2), "plan-R": round(100 * hit / max(n_gold, 1), 2),
           "questions/case": round(n_plan / len(recs), 2), "QA-consistency(before filter)": round(100 * cons / max(n_pairs, 1), 2),
           "parent-child pairs": n_pairs}
    return out, qa(items) if items else {}


def bootstrap(fn, n, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return [round(float(np.percentile(vals, 2.5)), 2), round(float(np.percentile(vals, 97.5)), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kvme", required=True)
    ap.add_argument("--pred", required=True, help="predictions_<split>.jsonl from infer.py")
    ap.add_argument("--baseline", default="", help="predictions jsonl of retrieval_baseline.py (same split)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-bertscore", action="store_true")
    ap.add_argument("--no-nli", action="store_true")
    args = ap.parse_args()
    init_jieba(args.kvme)
    recs = [json.loads(l) for l in open(args.pred, encoding="utf-8")]
    train_cases = [c for c in load_cases(args.kvme) if c["split"] == "train"]
    train_reports = {c["report"] for c in train_cases}
    prior = {}  # most frequent training answer per question (trivial QA baseline)
    for c in train_cases:
        for q, a in c["qa"].items():
            prior.setdefault(q, [0, 0])[a] += 1
    prior = {q: int(v[1] >= v[0]) for q, v in prior.items()}
    for r in recs:
        r["is_dup"] = r["report"] in train_reports
    systems = {"UTTAR R̂int": [r["report_int"] for r in recs]}
    if "report_qa_zh" in recs[0]:  # verified answers rendered in the reference language (render_qa.py)
        systems["UTTAR R̂QA rendered"] = [r["report_qa_zh"] for r in recs]
    if args.baseline:
        base = {(b["organ"], b["uid"]): b for b in map(json.loads, open(args.baseline, encoding="utf-8"))}
        for key in [k for k in base[(recs[0]["organ"], recs[0]["uid"])] if k.startswith("pred::")]:
            systems[key[len("pred::"):]] = [base[(r["organ"], r["uid"])][key] for r in recs]
    refs = [r["report"] for r in recs]
    organs = [r["organ"] for r in recs]
    res = {"n": len(recs), "NLG": {}, "CE": {}, "NLI": {}, "QA": {}, "stats": {}}

    # ---------------- NLG
    groups = {"ALL": list(range(len(recs)))}
    for o in ("Breast", "Thyroid", "Liver"):
        groups[o] = [i for i, r in enumerate(recs) if r["organ"] == o]
    groups["ALL/novel"] = [i for i, r in enumerate(recs) if not r["is_dup"]]
    groups["ALL/duplicate"] = [i for i, r in enumerate(recs) if r["is_dup"]]
    for name, hyps in systems.items():
        res["NLG"][name] = {g: nlg([refs[i] for i in ix], [hyps[i] for i in ix]) | {"n": len(ix)} for g, ix in groups.items()}
        if not args.no_bertscore:
            res["NLG"][name]["ALL"]["BERTScore-F1"] = bertscore(refs, hyps)
        print(name, json.dumps(res["NLG"][name]["ALL"], ensure_ascii=False), flush=True)

    # ---------------- CE
    ref_qa = [r["gold_qa"] for r in recs]
    hyp_sets = {name: [extract(h, o) for h, o in zip(hyps, organs)] for name, hyps in systems.items()}
    hyp_sets["UTTAR R̂QA (verified answers)"] = [r["answers"] for r in recs]
    case_f1 = {}
    for name, hq in hyp_sets.items():
        res["CE"][name] = {}
        for g in ("ALL", "Breast", "Thyroid", "Liver"):
            ix = groups[g]
            m, f = ce([ref_qa[i] for i in ix], [hq[i] for i in ix], [organs[i] for i in ix])
            res["CE"][name][g] = m
            if g == "ALL":
                case_f1[name] = f
        res["CE"][name]["ALL"]["CE-F1 95% CI"] = bootstrap(
            lambda s: ce([ref_qa[i] for i in s], [hq[i] for i in s], [organs[i] for i in s])[0]["CE-F1"], len(recs), B=300)
        print("CE", name, json.dumps(res["CE"][name]["ALL"], ensure_ascii=False), flush=True)

    # ---------------- entity-level NLI (text outputs)
    if not args.no_nli:
        nli = NLI()
        for name, hyps in systems.items():
            res["NLI"][name] = nli(refs, hyps, organs)
            print("NLI", name, {k: v for k, v in res["NLI"][name].items() if k != "per_question"}, flush=True)

    # ---------------- QA
    gi = gold_items(recs)
    res["QA"]["oracle questions (reference-instantiated)"] = {"ALL": qa(gi)}
    for o in ("Breast", "Thyroid", "Liver"):
        res["QA"]["oracle questions (reference-instantiated)"][o] = qa(gold_items([r for r in recs if r["organ"] == o]))
    per_case = [gold_items([r]) for r in recs]
    res["QA"]["oracle questions (reference-instantiated)"]["ALL"]["QA-F1 95% CI"] = bootstrap(
        lambda s: qa([x for i in s for x in per_case[i]])["QA-F1"], len(recs))
    res["QA"]["baselines on the same oracle questions"] = {
        "prior (most frequent training answer per question)": qa([(q, r["gold_qa"][q], prior.get(q, 1)) for r in recs for q in r["gold_qa"]])}
    for name, hq in hyp_sets.items():
        if name.startswith("UTTAR R̂QA"):
            continue  # R̂QA answers only the planned questions; its oracle-question QA is the verifier row above
        res["QA"]["baselines on the same oracle questions"][f"{name}: E(text), prior if unstated"] = qa(
            [(q, r["gold_qa"][q], h.get(q, prior.get(q, 1))) for r, h in zip(recs, hq) for q in r["gold_qa"]])
    ps, pq = planned_stats(recs)
    res["QA"]["planned questions"] = ps | {"answers on planned ∩ reference": pq}
    print("QA", json.dumps(res["QA"], ensure_ascii=False), flush=True)

    # ---------------- paired tests vs nearest-report baseline
    near = [k for k in systems if k.startswith("nearest")]
    if near:
        ours = [k for k in systems if k.startswith("UTTAR")]
        rl = {k: rouge_l_per_case(refs, systems[k]) for k in ours + [near[0]]}
        for k in ours:
            t = stats.ttest_rel(rl[k], rl[near[0]])
            res["stats"][f"ROUGE-L per case: {k} vs {near[0]}"] = {
                "mean_diff": round(float((rl[k] - rl[near[0]]).mean() * 100), 2), "t": round(float(t.statistic), 3),
                "p": float(t.pvalue)}
        for name in ours + ["UTTAR R̂QA (verified answers)"]:
            t = stats.ttest_rel(case_f1[name], case_f1[near[0]])
            res["stats"][f"CE-F1 per case: {name} vs {near[0]}"] = {
                "mean_diff": round(float((case_f1[name] - case_f1[near[0]]).mean() * 100), 2), "t": round(float(t.statistic), 3),
                "p": float(t.pvalue)}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
