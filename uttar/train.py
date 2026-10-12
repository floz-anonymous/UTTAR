"""UTTAR training on KVME (method §3.5, Appendix F).

Phase A: L_align = L_global + λ1 L_token + λ2 L_ITM (Eq. 8). Trainable: Q, Q-Former, g_v, g_t, h_ITM.
Phase B: L_B = L_gen + L_ques + L_verify + λ3 L_cons. Q-Former and g_v at 0.1·η_B; f_dec, f_plan, question-
         conditioned attention and f_ver at η_B; g_t frozen.
f_v (Ultrasound-CLIP) and f_t (SapBERT-XLMR) are frozen; their outputs are pre-computed once.
Checkpoints are chosen on KVME validation only; the test split is never touched here.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (MAX_PLAN, MAX_REPORT, QSEP, DecTokenizer, ft_model, ft_tokenizer, load_cases, load_patch_tokens,
                  plan_target)
from kg import build_graph, build_question_evidence
from metrics import init_jieba, nlg, qa
from model import SeqDecoder, Stage1, Verifier, cons_loss
from schema import ORGAN_QIDS, PARENT_CHILD, QUESTIONS, apply_parent_filter

ORGAN_ID = {"Breast": 0, "Thyroid": 1, "Liver": 2}
QIDX = {q.qid: i for i, q in enumerate(QUESTIONS)}
FT_REPORT_LEN, FT_Q_LEN, LE = 224, 24, 128  # L (max KVME report = 222 tokens), L_c, L_e = 128 (Appendix C)


# --------------------------------------------------------------------------- frozen text features
@torch.no_grad()
def ft_states(texts, tok, model, max_len, bs=128):
    H = torch.zeros(len(texts), max_len, 768, dtype=torch.float16)
    M = torch.zeros(len(texts), max_len, dtype=torch.bool)
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding="max_length", truncation=True, max_length=max_len, return_tensors="pt").to("cuda")
        H[i:i + bs] = model(**enc).last_hidden_state.half().cpu()
        M[i:i + bs] = enc["attention_mask"].bool().cpu()
    return H, M


def evidence_token_triples(evid, tok):
    """For each question: index of the KG triple that each of its L_e evidence tokens belongs to (-1 = passage)."""
    out = torch.full((len(QUESTIONS), LE), -1, dtype=torch.long)
    for q in QUESTIONS:
        enc = tok(evid[q.qid]["text"], truncation=True, max_length=LE, return_offsets_mapping=True)
        for t, (a, b) in enumerate(enc["offset_mapping"]):
            for k, (s, e) in enumerate(evid[q.qid]["spans"]):
                if a < b and s <= a < e:
                    out[QIDX[q.qid], t] = k
    return out


def build_text_cache(cases, out):
    path = out / "text_cache.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    tok, model = ft_tokenizer(), ft_model().cuda().eval()
    train = [c for c in cases if c["split"] == "train"]
    graph = build_graph(train)  # KG uses training reports only
    evid = build_question_evidence(QUESTIONS, graph, tok, model, "cuda")
    RT, RM = ft_states([c["report"] for c in cases], tok, model, FT_REPORT_LEN)
    QS, QM = ft_states([q.en for q in QUESTIONS], tok, model, FT_Q_LEN)
    ES, EM = ft_states([evid[q.qid]["text"] for q in QUESTIONS], tok, model, LE)
    cache = {"RT": RT, "RM": RM, "QS": QS, "QM": QM, "ES": ES, "EM": EM, "evid": evid, "graph": graph,
             "ev_triple": evidence_token_triples(evid, tok), "uids": [(c["organ"], c["uid"]) for c in cases]}
    torch.save(cache, path)
    (out / "kg_graph.json").write_text(json.dumps(graph, indent=1, ensure_ascii=False))
    (out / "kg_question_evidence.json").write_text(json.dumps(evid, indent=1, ensure_ascii=False))
    return cache


# --------------------------------------------------------------------------- data on GPU
class KVME:
    def __init__(self, cases, cache, dec, patch_dir):
        feats, index = load_patch_tokens(patch_dir)
        self.PT = torch.from_numpy(np.ascontiguousarray(feats)).cuda()  # [n_img, 196, 768] fp16
        self.img = torch.tensor([[index[p] for p in c["paths"]] for c in cases]).cuda()
        self.RT, self.RM = cache["RT"].cuda(), cache["RM"].cuda()
        self.organ = torch.tensor([ORGAN_ID[c["organ"]] for c in cases]).cuda()
        rid = {}
        self.rid = torch.tensor([rid.setdefault(c["report"], len(rid)) for c in cases]).cuda()
        self.dec = self._pad([dec.encode(c["report"], MAX_REPORT) for c in cases], dec.pad).cuda()
        self.plan = self._pad([dec.encode(plan_target(c["qa"]), MAX_PLAN) for c in cases], dec.pad).cuda()
        qa_ = torch.full((len(cases), len(QUESTIONS)), -1, dtype=torch.long)
        for i, c in enumerate(cases):
            for q, a in c["qa"].items():
                qa_[i, QIDX[q]] = a
        self.qa = qa_.cuda()
        self.split = np.array([c["split"] for c in cases])
        n_img = self.img.size(1)
        self.slot = torch.arange(n_img).repeat_interleave(196).cuda()
        self.QS, self.QM, self.ES, self.EM = cache["QS"].cuda(), cache["QM"].cuda(), cache["ES"].cuda(), cache["EM"].cuda()
        self.organ_q = {ORGAN_ID[o]: torch.tensor([QIDX[q] for q in qs]).cuda() for o, qs in ORGAN_QIDS.items()}
        self.organ_pc = {}
        for o, qs in ORGAN_QIDS.items():
            pos = {q: i for i, q in enumerate(qs)}
            self.organ_pc[ORGAN_ID[o]] = [(pos[p], pos[c]) for p, c in PARENT_CHILD if p in pos and c in pos]

    @staticmethod
    def _pad(seqs, pad):
        L = max(map(len, seqs))
        return torch.tensor([s + [pad] * (L - len(s)) for s in seqs])

    def V(self, b):
        v = self.PT[self.img[b]].flatten(1, 2).float()  # [B, 2*196, 768]
        return v, self.slot.expand(len(b), -1), torch.zeros(v.shape[:2], dtype=torch.bool, device=v.device)

    def idx(self, split):
        return torch.tensor(np.where(self.split == split)[0]).cuda()


def cosine_lr(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


# --------------------------------------------------------------------------- Phase A
def align_loss(s1, D, b, args, gen):
    V, slot, vpad = D.V(b)
    zv, _ = s1.visual(V, slot, vpad)
    tt, tm = s1.text(D.RT[b].float()), D.RM[b]
    same = D.rid[b][:, None] == D.rid[b][None, :]
    sg = s1.s_global(zv, tt[:, 0]).float()  # t_bar = projected [CLS] (<s>)
    lg = s1.infonce(sg, same)
    lt = s1.infonce(s1.s_token(zv, tt, tm).float(), same)
    # ITM: one negative report per image and one negative image per report, same organ, not an identical report,
    # sampled with probability proportional to exp(s/τ) (Appendix F)
    org = D.organ[b]
    valid = (org[:, None] == org[None, :]) & ~same
    w = torch.exp((sg.detach() - sg.detach().max()) / s1.tau).masked_fill(~valid, 0)
    B = len(b)
    ar = torch.arange(B, device=w.device)
    hi, ht = w.sum(1) > 0, w.t().sum(1) > 0
    neg_t = torch.multinomial(w[hi], 1, generator=gen).squeeze(1)
    neg_i = torch.multinomial(w.t()[ht], 1, generator=gen).squeeze(1)
    img_ix = torch.cat([ar, ar[hi], neg_i])
    txt_ix = torch.cat([ar, neg_t, ar[ht]])
    y = torch.cat([torch.ones(B), torch.zeros(len(neg_t) + len(neg_i))]).to(w.device)
    logit = s1.itm(tt[txt_ix], tm[txt_ix], zv[img_ix]).float()
    li = F.binary_cross_entropy_with_logits(logit, y)
    return lg + args.lam1 * lt + args.lam2 * li, (lg.item(), lt.item(), li.item())


@torch.no_grad()
def eval_align(s1, D, args):
    s1.eval()
    gen = torch.Generator(device="cuda").manual_seed(0)
    va = D.idx("val")
    tot, n = 0.0, 0
    for i in range(0, len(va), args.bs_a):
        b = va[i:i + args.bs_a]
        if len(b) < 8:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l, _ = align_loss(s1, D, b, args, gen)
        tot += l.item() * len(b); n += len(b)
    s1.train()
    return tot / n


def phase_a(D, args, out, log):
    s1 = Stage1().cuda()
    opt = torch.optim.AdamW(s1.parameters(), lr=args.lr_a, weight_decay=args.wd)
    tr = D.idx("train")
    steps_ep = len(tr) // args.bs_a
    total, step, best = steps_ep * args.epochs_a, 0, float("inf")
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    for ep in range(args.epochs_a):
        perm = tr[torch.randperm(len(tr), device="cuda", generator=gen)]
        t0, agg = time.time(), np.zeros(3)
        for k in range(steps_ep):
            b = perm[k * args.bs_a:(k + 1) * args.bs_a]
            for g in opt.param_groups:
                g["lr"] = args.lr_a * cosine_lr(step, total, args.warmup)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = align_loss(s1, D, b, args, gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(s1.parameters(), 1.0)
            opt.step(); step += 1; agg += parts
        val = eval_align(s1, D, args)
        rec = {"phase": "A", "epoch": ep, "train_global_token_itm": (agg / steps_ep).round(4).tolist(), "val_align": round(val, 4),
               "sec": round(time.time() - t0, 1)}
        if val < best:
            best = val
            torch.save({"stage1": s1.state_dict(), "epoch": ep, "val_align": val}, out / "phaseA_best.pt")
            rec["best"] = True
        log(rec)
    s1.load_state_dict(torch.load(out / "phaseA_best.pt", weights_only=True)["stage1"])
    return s1


# --------------------------------------------------------------------------- Phase B
class UTTAR(torch.nn.Module):
    def __init__(self, s1, vocab):
        super().__init__()
        self.s1 = s1
        self.fdec = SeqDecoder(vocab, d_mem=256, max_len=MAX_REPORT)
        self.fplan = SeqDecoder(vocab, d_mem=768, max_len=MAX_PLAN)  # seq2seq: attends to f_t(R) (R̂int at inference)
        self.ver = Verifier(d=256)


def frozen_question_tokens(model, D):
    with torch.no_grad():
        return model.s1.text(D.QS.float()), model.s1.text(D.ES.float())  # q~ = g_t(f_t(c)), E = g_t(f_t(ser(G_c)))


def verifier_batch(model, D, b, zv, qt, et):
    """All schema questions of each case's organ -> (logits, case pos, question idx, parent-child positions)."""
    ci, qj, pcs, off = [], [], [], 0
    for i, o in enumerate(D.organ[b].tolist()):
        qs = D.organ_q[o]
        ci.append(torch.full((len(qs),), i, device=qs.device)); qj.append(qs)
        pcs += [(off + p, off + c) for p, c in D.organ_pc[o]]
        off += len(qs)
    ci, qj = torch.cat(ci), torch.cat(qj)
    logit = model.ver(zv[ci], qt[qj], D.QM[qj], et[qj], D.EM[qj])
    return logit, ci, qj, torch.tensor(pcs, device=ci.device).view(-1, 2)


def phase_b_loss(model, D, b, qt, et, args):
    V, slot, vpad = D.V(b)
    zv, _ = model.s1.visual(V, slot, vpad)
    d = D.dec[b]
    lgen = F.cross_entropy(model.fdec(d[:, :-1], zv).float().transpose(1, 2), d[:, 1:], ignore_index=args.pad)
    p_ = D.plan[b]
    lques = F.cross_entropy(model.fplan(p_[:, :-1], D.RT[b].float(), ~D.RM[b]).float().transpose(1, 2), p_[:, 1:],
                            ignore_index=args.pad)
    logit, ci, qj, pcs = verifier_batch(model, D, b, zv, qt, et)
    lab = D.qa[b][ci, qj]
    m = lab >= 0
    lver = F.binary_cross_entropy_with_logits(logit.float()[m], lab[m].float())
    lcons = cons_loss(torch.sigmoid(logit.float()), pcs)
    return lgen + lques + lver + args.lam3 * lcons, (lgen.item(), lques.item(), lver.item(), lcons.item())


@torch.no_grad()
def generate_reports(model, D, b, dec, bs=256):
    out = []
    for i in range(0, len(b), bs):
        bb = b[i:i + bs]
        V, slot, vpad = D.V(bb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zv, _ = model.s1.visual(V, slot, vpad)
            ids = model.fdec.greedy(zv, dec.bos, dec.eos, MAX_REPORT)
        out += [dec.decode(x) for x in ids.tolist()]
    return out


@torch.no_grad()
def gold_question_probs(model, D, b, qt, et, bs=128):
    """Verifier probability for every instantiated (gold) question of each case."""
    res = [{} for _ in range(len(b))]
    for i in range(0, len(b), bs):
        bb = b[i:i + bs]
        V, slot, vpad = D.V(bb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zv, _ = model.s1.visual(V, slot, vpad)
            logit, ci, qj, _ = verifier_batch(model, D, bb, zv, qt, et)
        p = torch.sigmoid(logit.float())
        lab = D.qa[bb][ci, qj]
        for c, q, pp, l in zip(ci.tolist(), qj.tolist(), p.tolist(), lab.tolist()):
            if l >= 0:
                res[i + c][QUESTIONS[q].qid] = (pp, l)
    return res


def qa_items(probs):
    items = []
    for d in probs:
        ans = apply_parent_filter({q: int(p >= 0.5) for q, (p, _) in d.items()})
        items += [(q, d[q][1], ans[q]) for q in d]
    return items


@torch.no_grad()
def eval_b(model, D, cases, dec, qt, et, args):
    model.eval()
    va = D.idx("val")
    hyps = generate_reports(model, D, va, dec)
    refs = [cases[i]["report"] for i in va.tolist()]
    b4 = nlg(refs, hyps, full=False)["BLEU-4"]
    q = qa(qa_items(gold_question_probs(model, D, va, qt, et)))
    model.train()
    return b4, q


def phase_b(s1, D, cases, dec, args, out, log):
    model = UTTAR(s1, dec.vocab_size).cuda()
    for p in model.s1.g_t.parameters():
        p.requires_grad_(False)
    for p in model.s1.itm.parameters():
        p.requires_grad_(False)  # L_ITM is not part of L_B
    slow = list(model.s1.qformer.parameters()) + list(model.s1.g_v.parameters())
    fast = list(model.fdec.parameters()) + list(model.fplan.parameters()) + list(model.ver.parameters())
    opt = torch.optim.AdamW([{"params": slow, "base": 0.1 * args.lr_b}, {"params": fast, "base": args.lr_b}],
                            lr=args.lr_b, weight_decay=args.wd)
    qt, et = frozen_question_tokens(model, D)
    tr = D.idx("train")
    steps_ep = len(tr) // args.bs_b
    total, step, best = steps_ep * args.epochs_b, 0, -1.0
    gen = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    for ep in range(args.epochs_b):
        perm = tr[torch.randperm(len(tr), device="cuda", generator=gen)]
        t0, agg = time.time(), np.zeros(4)
        for k in range(steps_ep):
            b = perm[k * args.bs_b:(k + 1) * args.bs_b]
            f = cosine_lr(step, total, args.warmup)
            for g in opt.param_groups:
                g["lr"] = g["base"] * f
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = phase_b_loss(model, D, b, qt, et, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step(); step += 1; agg += parts
        rec = {"phase": "B", "epoch": ep, "train_gen_ques_verify_cons": (agg / steps_ep).round(4).tolist(),
               "sec": round(time.time() - t0, 1)}
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs_b - 1:
            b4, q = eval_b(model, D, cases, dec, qt, et, args)
            sel = 0.5 * (b4 + q["QA-macroF1"])
            rec.update({"val_BLEU-4": b4, "val_QA": q, "val_select": round(sel, 2)})
            if sel > best:
                best = sel
                torch.save({"model": model.state_dict(), "epoch": ep, "val_select": sel}, out / "phaseB_best.pt")
                rec["best"] = True
        log(rec)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kvme", required=True)
    ap.add_argument("--patch-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs-a", type=int, default=50)
    ap.add_argument("--epochs-b", type=int, default=60)
    ap.add_argument("--bs-a", type=int, default=128)
    ap.add_argument("--bs-b", type=int, default=64)
    ap.add_argument("--lr-a", type=float, default=1e-4)
    ap.add_argument("--lr-b", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lam1", type=float, default=1.0)
    ap.add_argument("--lam2", type=float, default=1.0)
    ap.add_argument("--lam3", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-a", action="store_true", help="reuse phaseA_best.pt in --out")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    init_jieba(args.kvme)
    logf = open(out / "log.jsonl", "a")

    def log(rec):
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        logf.write(json.dumps(rec, ensure_ascii=False) + "\n"); logf.flush()

    cases = load_cases(args.kvme)
    dec = DecTokenizer([c["report"] for c in cases if c["split"] == "train"])
    args.pad = dec.pad
    cache = build_text_cache(cases, out)
    D = KVME(cases, cache, dec, args.patch_cache)
    (out / "config.json").write_text(json.dumps(vars(args) | {"dec_vocab": dec.vocab_size, "dec_added_tokens": dec.added,
                                                              "c2f": dec.c2f}, ensure_ascii=False))
    log({"cases": len(cases), "train": int((D.split == "train").sum()), "val": int((D.split == "val").sum()),
         "dec_vocab": dec.vocab_size})
    if args.skip_a:
        s1 = Stage1().cuda()
        s1.load_state_dict(torch.load(out / "phaseA_best.pt", weights_only=True)["stage1"])
    else:
        s1 = phase_a(D, args, out, log)
    phase_b(s1, D, cases, dec, args, out, log)


if __name__ == "__main__":
    main()
