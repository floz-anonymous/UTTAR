"""UTTAR inference (Algorithm 1, Appendix E) on a KVME split with the validation-selected checkpoint.

Per case: Z_v from the image(s) -> R̂int = f_dec(Z_v) (greedy) -> C = f_plan(f_t(R̂int)) (greedy) -> for each c_k:
p_k = f_ver([CLS; V_QA; q~_k; E_k]), a_k = Yes iff p_k >= 0.5 -> parent-child filter -> template realizer R̂QA with
evidence e_k: the image regions of the queries with the highest attention mass in A_k (via the Q-Former
cross-attention map) and the KG triples with the highest attention from h_CLS.
The verifier is also run on the reference's instantiated (gold) questions for the QA metrics with oracle questions.
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import MAX_PLAN, DecTokenizer, ft_model, ft_tokenizer, load_cases
from model import Stage1
from schema import PARENT_CHILD, QMAP, QUESTIONS, apply_parent_filter
from train import (FT_REPORT_LEN, KVME, QIDX, UTTAR, build_text_cache, frozen_question_tokens, ft_states,
                   generate_reports, gold_question_probs)

EN = {q.en.lower().rstrip("?").strip(): q.qid for q in QUESTIONS}


def map_question(s):
    """Planner output -> schema question: exact template match, else nearest template."""
    k = s.lower().strip().rstrip("?").strip()
    if not k:
        return None
    if k in EN:
        return EN[k]
    m = difflib.get_close_matches(k, list(EN), n=1, cutoff=0.5)
    return EN[m[0]] if m else None


def realize(items):
    """Deterministic template realizer: one line per question with its answer and evidence."""
    lines = []
    for n, it in enumerate(items, 1):
        regions = "; ".join(f"image {r['image']}, patch row {r['row']} col {r['col']}" for r in it["regions"])
        kg = "; ".join(f"{h} — {r} — {t}" for h, r, t in it["triples"]) or "none"
        lines.append(f"Q{n}. {QMAP[it['qid']].en} — {'Yes' if it['answer'] else 'No'} (p = {it['p']:.2f})\n"
                     f"    image evidence: {regions}\n    knowledge: {kg}")
    return "\n".join(lines)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kvme", required=True)
    ap.add_argument("--patch-cache", required=True)
    ap.add_argument("--run", required=True, help="training output folder (phaseB_best.pt, text_cache.pt)")
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()
    run = Path(args.run)
    cases = load_cases(args.kvme)
    dec = DecTokenizer([c["report"] for c in cases if c["split"] == "train"])
    cache = build_text_cache(cases, run)
    D = KVME(cases, cache, dec, args.patch_cache)
    model = UTTAR(Stage1(), dec.vocab_size).cuda()
    ck = torch.load(run / "phaseB_best.pt", weights_only=True)
    model.load_state_dict(ck["model"]); model.eval()
    qt, et = frozen_question_tokens(model, D)
    tok, ft = ft_tokenizer(), ft_model().cuda().eval()
    ev_triple = cache["ev_triple"].cuda()
    evid = cache["evid"]

    idx = D.idx(args.split)
    r_int = generate_reports(model, D, idx, dec)
    gold = gold_question_probs(model, D, idx, qt, et)
    H, M = ft_states(r_int, tok, ft, FT_REPORT_LEN)  # planner input at inference: f_t(R̂int)
    out = open(run / f"predictions_{args.split}.jsonl", "w", encoding="utf-8")
    for s in range(0, len(idx), args.bs):
        b = idx[s:s + args.bs]
        V, slot, vpad = D.V(b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zv, qf_attn = model.s1.visual(V, slot, vpad, return_attn=True)  # qf_attn: [B, M, 392]
            plan_ids = model.fplan.greedy(H[s:s + args.bs].cuda().float(), dec.bos, dec.eos, MAX_PLAN,
                                          mem_pad=~M[s:s + args.bs].cuda())
        for j, gi in enumerate(b.tolist()):
            c = cases[gi]
            raw = dec.decode(plan_ids[j].tolist(), report=False)
            qids = []
            for part in raw.split(";"):
                q = map_question(part)
                if q and q not in qids:
                    qids.append(q)
            items = []
            if qids:
                qj = torch.tensor([QIDX[q] for q in qids], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit, A, cls_e = model.ver(zv[j:j + 1].expand(len(qj), -1, -1), qt[qj], D.QM[qj], et[qj], D.EM[qj],
                                                return_attn=True)
                p = torch.sigmoid(logit.float()).tolist()
                ans = apply_parent_filter({q: int(pp >= 0.5) for q, pp in zip(qids, p)})
                for k, q in enumerate(qids):
                    qm = D.QM[qj[k]].float()
                    mass = (A[k].float() * qm[:, None]).sum(0)  # attention mass per query over the question tokens
                    regions = []
                    for m in mass.topk(2).indices.tolist():
                        patch = int(qf_attn[j, m].float().argmax())
                        regions.append({"query": m, "image": patch // 196 + 1, "row": (patch % 196) // 14, "col": patch % 14})
                    tri = ev_triple[qj[k]]
                    w = torch.zeros(len(evid[q]["triples"]) + 1, device="cuda")
                    valid = (tri >= 0) & D.EM[qj[k]]
                    w.index_add_(0, tri[valid], cls_e[k].float()[valid])
                    top = [t for t in w[:-1].argsort(descending=True).tolist()[:2] if w[t] > 0]
                    items.append({"qid": q, "p": round(p[k], 4), "answer": ans[q], "regions": regions,
                                  "triples": [evid[q]["triples"][t] for t in top]})
            rec = {"organ": c["organ"], "uid": c["uid"], "split": c["split"], "report": c["report"], "report_int": r_int[s + j],
                   "plan_raw": raw, "planned": qids, "answers": {it["qid"]: it["answer"] for it in items},
                   "p": {it["qid"]: it["p"] for it in items}, "gold_qa": c["qa"],
                   "gold_p": {q: round(v[0], 4) for q, v in gold[s + j].items()}, "items": items, "report_qa": realize(items)}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.close()
    print("wrote", run / f"predictions_{args.split}.jsonl", len(idx))


if __name__ == "__main__":
    main()
