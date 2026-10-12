"""UTTAR modules (method §3.2–3.4, Appendix D).

Stage 1: Q-Former (Eq. 3) + g_v (Eq. 4), text projection g_t (Eq. 5), L_global (Eq. 6), L_token (Eq. 7), L_ITM.
Stage 2: report decoder f_dec (Eq. 9) and question planner f_plan (L_ques).
Stage 3: question-conditioned attention (Eq. 11), verifier f_ver (Eq. 12), L_verify (Eq. 13), L_cons (Eq. 14).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, d, mult=4, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * mult), nn.GELU(), nn.Dropout(p), nn.Linear(d * mult, d))

    def forward(self, x):
        return self.net(x)


class QFormerBlock(nn.Module):
    """Q <- Q + SA(LN(Q)); Q <- Q + CA(LN(Q), V); Q <- Q + FFN(LN(Q))   (Eq. 3; CA in every block)."""

    def __init__(self, dq, dv, heads, p=0.1):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(dq), nn.LayerNorm(dq), nn.LayerNorm(dq)
        self.sa = nn.MultiheadAttention(dq, heads, dropout=p, batch_first=True)
        self.ca = nn.MultiheadAttention(dq, heads, dropout=p, kdim=dv, vdim=dv, batch_first=True)
        self.ffn = FFN(dq, p=p)

    def forward(self, q, v, v_pad, need_weights=False):
        h = self.ln1(q)
        q = q + self.sa(h, h, h, need_weights=False)[0]
        a, w = self.ca(self.ln2(q), v, v, key_padding_mask=v_pad, need_weights=need_weights, average_attn_weights=True)
        q = q + a
        q = q + self.ffn(self.ln3(q))
        return q, w


class QFormer(nn.Module):
    def __init__(self, M=32, dq=512, dv=768, layers=6, heads=8, n_img_slots=2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, M, dq) * 0.02)
        # Image-slot embedding: KVME cases have two images; queries can tell them apart.
        self.img_slot = nn.Embedding(n_img_slots, dv)
        nn.init.normal_(self.img_slot.weight, std=0.02)
        self.blocks = nn.ModuleList([QFormerBlock(dq, dv, heads) for _ in range(layers)])
        self.ln_out = nn.LayerNorm(dq)

    def forward(self, V, slot, v_pad, return_attn=False):
        """V: [B, N_tot, dv] patch tokens of all images; slot: [B, N_tot] image index; v_pad: [B, N_tot] True=pad."""
        V = V + self.img_slot(slot)
        q = self.queries.expand(V.size(0), -1, -1)
        w = None
        for i, blk in enumerate(self.blocks):
            q, w_i = blk(q, V, v_pad, need_weights=return_attn and i == len(self.blocks) - 1)
            w = w_i if w_i is not None else w
        return self.ln_out(q), w  # Z: [B, M, dq]; w: last-layer query->patch attention [B, M, N_tot]


class ITMHead(nn.Module):
    """h_ITM: two cross-attention layers (text queries attend to Z_v) + linear classifier on the pooled output."""

    def __init__(self, d, heads=4, layers=2, p=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({"ln1": nn.LayerNorm(d), "ca": nn.MultiheadAttention(d, heads, dropout=p, batch_first=True),
                                              "ln2": nn.LayerNorm(d), "ffn": FFN(d, p=p)}))
        self.cls = nn.Linear(d, 1)

    def forward(self, t, t_mask, zv):
        x = t
        for l in self.layers:
            x = x + l["ca"](l["ln1"](x), zv, zv, need_weights=False)[0]
            x = x + l["ffn"](l["ln2"](x))
        m = t_mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return self.cls(pooled).squeeze(-1)


class Stage1(nn.Module):
    """Q, Q-Former, g_v, g_t and h_ITM; f_v and f_t are frozen and live outside this module."""

    def __init__(self, M=32, dq=512, dv=768, dt=768, d=256, layers=6, heads=8, tau=0.07):
        super().__init__()
        self.qformer = QFormer(M, dq, dv, layers, heads)
        self.g_v = nn.Linear(dq, d, bias=False)
        self.g_t = nn.Linear(dt, d, bias=False)
        self.itm = ITMHead(d)
        self.tau = tau

    def visual(self, V, slot, v_pad, return_attn=False):
        Z, w = self.qformer(V, slot, v_pad, return_attn)
        return self.g_v(Z), w  # Z_v [B, M, d]

    def text(self, T):
        return self.g_t(T)  # T~ [B, L, d]

    # ---- similarity functions (rows l2-normalised inside the losses) ----
    @staticmethod
    def s_global(zv, tbar):
        """s(I_i, R_j) = max_m <z_m, t_bar>  ->  [B, B]."""
        return torch.einsum("imd,jd->ijm", F.normalize(zv, dim=-1), F.normalize(tbar, dim=-1)).max(-1).values

    @staticmethod
    def s_token(zv, tt, t_mask):
        """Late interaction (Eq. 7): 1/2 [ 1/M sum_m max_l S_ml + 1/L sum_l max_m S_ml ]  ->  [B, B]."""
        S = torch.einsum("imd,jld->ijml", F.normalize(zv, dim=-1), F.normalize(tt, dim=-1))
        neg = torch.finfo(S.dtype).min
        mask = t_mask[None, :, None, :]  # [1, B, 1, L]
        q2t = S.masked_fill(~mask, neg).max(-1).values.mean(-1)  # [B, B]
        t2q = S.max(2).values  # [B, B, L]
        L = t_mask.sum(-1).clamp(min=1).float()[None, :]
        t2q = (t2q * t_mask[None].float()).sum(-1) / L
        return 0.5 * (q2t + t2q)

    def infonce(self, s, same_text=None):
        """Symmetric InfoNCE (Eq. 6). Off-diagonal pairs with identical report text are not negatives and are masked."""
        logits = s / self.tau
        if same_text is not None:
            eye = torch.eye(len(s), dtype=torch.bool, device=s.device)
            logits = logits.masked_fill(same_text & ~eye, float("-inf"))
        tgt = torch.arange(len(s), device=s.device)
        return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


class SeqDecoder(nn.Module):
    """Transformer decoder over the shared vocabulary that cross-attends to a memory sequence."""

    def __init__(self, vocab, d_mem, d_model=512, layers=3, heads=8, max_len=400, p=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.mem = nn.Linear(d_mem, d_model)
        layer = nn.TransformerDecoderLayer(d_model, heads, 4 * d_model, p, batch_first=True, norm_first=True, activation="gelu")
        self.dec = nn.TransformerDecoder(layer, layers)
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab)
        self.max_len = max_len

    def forward(self, ids, memory, mem_pad=None):
        L = ids.size(1)
        x = self.emb(ids) + self.pos(torch.arange(L, device=ids.device))[None]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=ids.device), 1)
        h = self.dec(x, self.mem(memory), tgt_mask=causal, memory_key_padding_mask=mem_pad)
        return self.out(self.ln(h))

    @torch.no_grad()
    def greedy(self, memory, bos, eos, max_len, mem_pad=None):
        B = memory.size(0)
        ids = torch.full((B, 1), bos, dtype=torch.long, device=memory.device)
        done = torch.zeros(B, dtype=torch.bool, device=memory.device)
        for _ in range(max_len - 1):
            nxt = self.forward(ids, memory, mem_pad)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            ids = torch.cat([ids, nxt[:, None]], 1)
            done |= nxt == eos
            if done.all():
                break
        return ids[:, 1:]


class Verifier(nn.Module):
    """Question-conditioned attention (Eq. 11) + f_ver over [CLS; V_QA; q~; E] with modality-type embeddings (Eq. 12)."""

    def __init__(self, d=256, heads=8, layers=4, attn_heads=4, p=0.1, use_visual=True, use_kg=True):
        super().__init__()
        self.qattn = nn.MultiheadAttention(d, attn_heads, batch_first=True)  # W_q, W_k, W_v in R^{d x d}
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.type_emb = nn.Embedding(4, d)  # 0 CLS, 1 V_QA, 2 question, 3 evidence
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, p, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.ln = nn.LayerNorm(d)
        self.w = nn.Linear(d, 1)
        self.use_visual, self.use_kg = use_visual, use_kg

    def forward(self, zv, q, q_mask, e, e_mask, return_attn=False):
        """zv [K, M, d] (case Z_v repeated per question); q [K, Lc, d]; e [K, Le, d]. Returns logits [K]."""
        K = q.size(0)
        parts, masks, types = [self.cls.expand(K, -1, -1)], [torch.ones(K, 1, dtype=torch.bool, device=q.device)], [0]
        A = None
        if self.use_visual:
            vqa, A = self.qattn(q, zv, zv, need_weights=True)  # A_k: [K, Lc, M]
            parts.append(vqa); masks.append(q_mask); types.append(1)
        parts.append(q); masks.append(q_mask); types.append(2)
        if self.use_kg:
            parts.append(e); masks.append(e_mask); types.append(3)
        x = torch.cat([p + self.type_emb.weight[t] for p, t in zip(parts, types)], 1)
        m = torch.cat(masks, 1)
        if not return_attn:
            h = self.enc(x, src_key_padding_mask=~m)
            return self.w(self.ln(h[:, 0])).squeeze(-1)
        # same computation layer by layer, keeping the last layer's attention from h_CLS (realizer evidence)
        h, cls_attn = x, None
        for i, layer in enumerate(self.enc.layers):
            if i == len(self.enc.layers) - 1:
                y = layer.norm1(h)
                cls_attn = layer.self_attn(y, y, y, key_padding_mask=~m, need_weights=True)[1][:, 0]  # [K, S]
            h = layer(h, src_key_padding_mask=~m)
        logit = self.w(self.ln(h[:, 0])).squeeze(-1)
        e_start = x.size(1) - (e.size(1) if self.use_kg else 0)
        return logit, A, cls_attn[:, e_start:] if self.use_kg else None


def cons_loss(p, pairs):
    """L_cons (Eq. 14): mean over parent-child pairs of max(0, p_child - p_parent). pairs: LongTensor [P, 2] (parent, child)."""
    if pairs.numel() == 0:
        return p.new_zeros(())
    return F.relu(p[pairs[:, 1]] - p[pairs[:, 0]]).mean()
