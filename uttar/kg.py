"""Clinical knowledge graph G and question-specific GraphRAG retrieval (method §3.4, Appendix C).

Nodes are typed (anatomy, finding, appearance, morphology, evidence). Lexicon entries become typed nodes;
evidence sentences that mention a node become evidence nodes linked by "supported-by" edges; attribute-finding
co-occurrences in the *training* reports add has-appearance / has-morphology edges with frequency weights.

The graph is in English (node names, relations, evidence), like the paper's questions; the Chinese surface form of
each lexicon entry is kept for reference only (co-occurrences are counted from the extractor's QA labels).

Retrieval for question c_k: embed c_k with f_t [CLS]; seed nodes = kappa nearest node-name embeddings;
expand the 2-hop neighbourhood; keep the rho nodes with the highest (edge weight x seed similarity);
serialise triples "head [SEP] relation [SEP] tail", then the evidence passages, truncated to L_e tokens.

NOTE: the evidence passages below are short definitional sentences written for this
implementation, not quotations from guideline documents. Replace them with guideline text when available.
"""
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# (English node name, type, organ or None, Chinese surface form in KVME reports)
LEXICON = [
    # anatomy
    ("breast", "anatomy", "Breast", "乳腺"), ("mammary duct", "anatomy", "Breast", "乳导管"),
    ("axillary lymph node", "anatomy", "Breast", "腋下淋巴结"), ("thyroid", "anatomy", "Thyroid", "甲状腺"),
    ("thyroid parenchyma", "anatomy", "Thyroid", "甲状腺腺体"), ("cervical lymph node", "anatomy", "Thyroid", "颈部淋巴结"),
    ("liver", "anatomy", "Liver", "肝脏"), ("liver capsule", "anatomy", "Liver", "肝包膜"), ("portal vein", "anatomy", "Liver", "门静脉"),
    ("bile duct", "anatomy", "Liver", "胆管"), ("gallbladder", "anatomy", "Liver", "胆囊"), ("gallbladder wall", "anatomy", "Liver", "胆囊壁"),
    ("pancreas", "anatomy", "Liver", "胰腺"), ("pancreatic duct", "anatomy", "Liver", "胰管"), ("spleen", "anatomy", "Liver", "脾脏"),
    ("kidney", "anatomy", "Liver", "肾脏"),
    # findings
    ("nodule", "finding", None, "结节"), ("space-occupying lesion", "finding", None, "占位性病变"),
    ("enlarged lymph node", "finding", None, "肿大淋巴结"), ("cyst", "finding", None, "囊肿"), ("calculus", "finding", "Liver", "结石"),
    ("polyp", "finding", "Liver", "息肉"), ("fatty liver", "finding", "Liver", "脂肪肝"), ("calcification", "finding", None, "钙化"),
    ("duct dilatation", "finding", "Breast", "导管扩张"), ("disordered glandular structure", "finding", "Breast", "腺体结构紊乱"),
    ("bile duct dilatation", "finding", "Liver", "胆管扩张"), ("pancreatic duct dilatation", "finding", "Liver", "胰管扩张"),
    ("post-cholecystectomy", "finding", "Liver", "胆囊切除术后"), ("gallbladder wall thickening", "finding", "Liver", "胆囊壁增厚"),
    # appearance
    ("hypoechoic", "appearance", None, "低回声"), ("anechoic", "appearance", None, "无回声"), ("isoechoic", "appearance", None, "等回声"),
    ("hyperechoic", "appearance", None, "高回声"), ("strong echogenicity", "appearance", None, "强回声"),
    ("mixed cystic-solid echogenicity", "appearance", None, "囊实混合回声"), ("punctate echogenic foci", "appearance", None, "点状强回声"),
    ("homogeneous echotexture", "appearance", None, "回声均匀"), ("heterogeneous echotexture", "appearance", None, "回声不均匀"),
    ("fine increased parenchymal echogenicity", "appearance", "Liver", "实质回声细密增强"),
    ("increased hepatorenal echo contrast", "appearance", "Liver", "肝肾回声对比增强"),
    ("far-field attenuation", "appearance", "Liver", "后场回声衰减"), ("blood flow signal", "appearance", None, "血流信号"),
    ("comet-tail artifact", "appearance", None, "彗星尾征"), ("acoustic shadow", "appearance", None, "声影"),
    # morphology
    ("clear boundary", "morphology", None, "边界清晰"), ("ill-defined boundary", "morphology", None, "边界欠清"),
    ("regular shape", "morphology", None, "形态规整"), ("irregular shape", "morphology", None, "形态欠规整"),
    ("normal size and shape", "morphology", None, "大小形态如常"), ("plump contour", "morphology", None, "形态饱满"),
    ("smooth capsule", "morphology", "Liver", "包膜光滑"), ("smooth wall", "morphology", "Liver", "壁光滑"),
    ("lymph node hilum", "morphology", None, "淋巴门结构"), ("moves with change of position", "morphology", "Liver", "随体位改变而移动"),
    ("poorly visualised portal venous system", "morphology", "Liver", "门静脉系统显示欠清晰"),
]

EVIDENCE = [  # authored definitional passages (see module docstring)
    "Hypoechoic: the lesion is less echogenic than the surrounding tissue; a thyroid nodule is compared with the thyroid parenchyma and a breast nodule with the surrounding fat.",
    "Anechoic: no internal echoes, typical of fluid-filled lesions such as a cyst, often with posterior acoustic enhancement.",
    "Mixed cystic-solid echogenicity: a nodule that contains both anechoic cystic and solid components.",
    "Isoechoic and hyperechoic: the lesion is as echogenic as, or more echogenic than, the surrounding tissue.",
    "Strong echogenicity: an echo much brighter than the surrounding tissue; a calculus or calcification shows strong echogenicity with an acoustic shadow.",
    "Punctate echogenic foci: tiny bright dots inside a nodule that may indicate microcalcification; with a comet-tail artifact they usually represent colloid crystals.",
    "Comet-tail artifact: a tapering reverberation artifact behind a strong echo, typical of colloid or cholesterol crystals.",
    "Clear boundary: the lesion is sharply separated from the surrounding tissue; an ill-defined boundary means the interface is blurred.",
    "Regular shape: the lesion is round or oval with a regular outline; an irregular shape means an irregular outline.",
    "Blood flow signal: flow shown by colour Doppler (CDFI), used to assess the vascularity of a nodule or gland; no detectable blood flow signal means no evident vascularity.",
    "Lymph node hilum: the central echogenic hilum of a lymph node; a preserved lymph node hilum is usually seen in benign or reactive nodes.",
    "Enlarged lymph node: a lymph node of increased size; several hypoechoic node-like structures may be seen in the axilla or neck.",
    "Duct dilatation: widening of the mammary duct lumen; a normal mammary duct is not dilated.",
    "Disordered glandular structure: uneven echo distribution and unclear layering of the breast glandular tissue.",
    "Normal size and shape of the thyroid: thyroid volume and contour are normal; a plump contour or enlargement suggests goitre.",
    "Homogeneous echotexture: the thyroid parenchyma has an even echo pattern; coarse, reduced or reticular echoes (heterogeneous echotexture) suggest diffuse disease.",
    "Fine increased parenchymal echogenicity: diffusely fine and increased echoes of the liver parenchyma, the typical ultrasound sign of fatty liver.",
    "Increased hepatorenal echo contrast: the liver parenchyma is clearly more echogenic than the cortex of the kidney, indicating fatty liver.",
    "Far-field attenuation: echoes fade in the deep liver, common in moderate to severe fatty liver.",
    "Poorly visualised portal venous system: in fatty liver the walls of the intrahepatic portal vein branches appear blurred.",
    "Plump contour of the liver: the liver is enlarged with blunted edges.",
    "Smooth capsule: the liver capsule echo is continuous and even.",
    "Bile duct dilatation: widening of the intra- and extrahepatic bile duct; normally the bile duct is not dilated.",
    "Normal portal vein: the main portal vein is not widened and its diameter is within the normal range.",
    "Gallbladder calculus: a strong echo in the gallbladder lumen with an acoustic shadow that moves with change of position.",
    "Gallbladder polyp: an intermediate or slightly hyperechoic echo attached to the gallbladder wall, without acoustic shadow and not mobile.",
    "Gallbladder wall thickening: increased thickness of the gallbladder wall; a rough, uneven wall suggests chronic inflammation, whereas a smooth wall is normal.",
    "Post-cholecystectomy: no gallbladder is seen in the gallbladder fossa.",
    "Pancreatic duct dilatation: widening of the pancreatic duct; normally the pancreatic duct is not widened.",
    "Space-occupying lesion: a focal nodule or mass within an organ; no definite space-occupying lesion means no focal lesion was found.",
    "Normal size and shape of the spleen: spleen thickness and length are within the normal range.",
    "Cyst: a well-defined anechoic lesion with a clear boundary and posterior acoustic enhancement.",
]

HAS_APP, HAS_MOR, HAS_FIND, SUPPORTED = "has-appearance", "has-morphology", "has-finding", "supported-by"
# schema slot -> (finding node, relation, attribute node) used to count co-occurrences in training reports
SLOT_TRIPLES = {
    "B_nodule_hypo": ("nodule", HAS_APP, "hypoechoic"), "B_nodule_anechoic": ("nodule", HAS_APP, "anechoic"),
    "B_nodule_mixed": ("nodule", HAS_APP, "mixed cystic-solid echogenicity"),
    "B_nodule_boundary_clear": ("nodule", HAS_MOR, "clear boundary"), "B_nodule_shape_regular": ("nodule", HAS_MOR, "regular shape"),
    "B_nodule_flow": ("nodule", HAS_APP, "blood flow signal"), "B_nodule_calc": ("nodule", HAS_APP, "punctate echogenic foci"),
    "B_axillary_ln_hilum": ("enlarged lymph node", HAS_MOR, "lymph node hilum"),
    "T_nodule_hypo": ("nodule", HAS_APP, "hypoechoic"), "T_nodule_anechoic": ("nodule", HAS_APP, "anechoic"),
    "T_nodule_mixed": ("nodule", HAS_APP, "mixed cystic-solid echogenicity"), "T_nodule_iso_hyper": ("nodule", HAS_APP, "hyperechoic"),
    "T_nodule_boundary_clear": ("nodule", HAS_MOR, "clear boundary"), "T_nodule_shape_regular": ("nodule", HAS_MOR, "regular shape"),
    "T_nodule_flow": ("nodule", HAS_APP, "blood flow signal"), "T_nodule_calc": ("nodule", HAS_APP, "punctate echogenic foci"),
    "T_nodule_comet": ("nodule", HAS_APP, "comet-tail artifact"), "T_cervical_ln_hilum": ("enlarged lymph node", HAS_MOR, "lymph node hilum"),
    "L_hepatorenal": ("fatty liver", HAS_APP, "increased hepatorenal echo contrast"),
    "L_attenuation": ("fatty liver", HAS_APP, "far-field attenuation"),
    "L_portal_unclear": ("fatty liver", HAS_MOR, "poorly visualised portal venous system"),
    "L_lesion_anechoic": ("cyst", HAS_APP, "anechoic"), "L_lesion_hyper": ("space-occupying lesion", HAS_APP, "strong echogenicity"),
    "G_lesion_shadow": ("calculus", HAS_APP, "acoustic shadow"), "G_lesion_mobile": ("calculus", HAS_MOR, "moves with change of position"),
    "G_wall_comet": ("gallbladder wall", HAS_APP, "comet-tail artifact"),
}
ANATOMY_FINDING = [("breast", "nodule"), ("breast", "space-occupying lesion"), ("mammary duct", "duct dilatation"),
                   ("breast", "disordered glandular structure"), ("axillary lymph node", "enlarged lymph node"),
                   ("thyroid", "nodule"), ("thyroid", "space-occupying lesion"), ("cervical lymph node", "enlarged lymph node"),
                   ("thyroid parenchyma", "homogeneous echotexture"), ("liver", "fatty liver"), ("liver", "space-occupying lesion"),
                   ("liver", "cyst"), ("bile duct", "bile duct dilatation"), ("gallbladder", "calculus"), ("gallbladder", "polyp"),
                   ("gallbladder", "post-cholecystectomy"), ("gallbladder wall", "gallbladder wall thickening"),
                   ("pancreatic duct", "pancreatic duct dilatation"), ("pancreas", "space-occupying lesion"),
                   ("spleen", "space-occupying lesion"), ("liver capsule", "smooth capsule"),
                   ("portal vein", "poorly visualised portal venous system")]


def build_graph(train_cases):
    nodes = [{"name": n, "type": t, "organ": o, "zh": zh} for n, t, o, zh in LEXICON]
    names = [n["name"] for n in nodes]
    edges = defaultdict(float)  # (head, rel, tail) -> weight
    for a, f in ANATOMY_FINDING:
        edges[(a, HAS_FIND, f)] = 1.0
    counts = defaultdict(int)
    for c in train_cases:
        for qid, ans in c["qa"].items():
            if ans == 1 and qid in SLOT_TRIPLES:
                counts[SLOT_TRIPLES[qid]] += 1
    mx = max(counts.values())
    for t, n in counts.items():
        edges[t] = max(edges[t], n / mx)
    for i, passage in enumerate(EVIDENCE):
        ev = f"evidence_{i}"
        nodes.append({"name": ev, "type": "evidence", "organ": None, "text": passage})
        for n in names:  # whole-word, case-insensitive mention
            if re.search(r"(?<![\w-])" + re.escape(n) + r"(?![\w-])", passage, flags=re.I):
                edges[(n, SUPPORTED, ev)] = 1.0
    return {"nodes": nodes, "edges": [{"head": h, "rel": r, "tail": t, "w": round(w, 4)} for (h, r, t), w in edges.items()]}


@torch.no_grad()
def cls_embed(texts, ft_tok, ft_model, device, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        enc = ft_tok(texts[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        out.append(F.normalize(ft_model(**enc).last_hidden_state[:, 0].float(), dim=-1).cpu())
    return torch.cat(out)


def retrieve(graph, node_emb, q_emb, sep, kappa=3, rho=12):
    """Return (serialised evidence text, kept triples, char spans of the triples) for one question."""
    nodes = graph["nodes"]
    names = [n["name"] for n in nodes]
    idx = {n: i for i, n in enumerate(names)}
    is_ev = np.array([n["type"] == "evidence" for n in nodes])
    sims = (node_emb @ q_emb).numpy()
    sims_seed = np.where(is_ev, -np.inf, sims)  # seeds are linked by node-name embeddings
    seeds = list(np.argsort(-sims_seed)[:kappa])
    adj = defaultdict(list)
    for e in graph["edges"]:
        h, t = idx[e["head"]], idx[e["tail"]]
        adj[h].append((t, e["w"], e)); adj[t].append((h, e["w"], e))
    score = {s: float(sims[s]) for s in seeds}
    for s in seeds:
        for n1, w1, _ in adj[s]:
            score[n1] = max(score.get(n1, -1e9), w1 * float(sims[s]))
            for n2, w2, _ in adj[n1]:
                score[n2] = max(score.get(n2, -1e9), w1 * w2 * float(sims[s]))
    keep = [n for n, _ in sorted(score.items(), key=lambda kv: -kv[1])[:rho]]
    order = seeds + [n for n in keep if n not in seeds]  # deterministic: seeds first, then by relevance
    kept = set(order)
    triples = [e for e in graph["edges"] if idx[e["head"]] in kept and idx[e["tail"]] in kept and e["rel"] != SUPPORTED]
    rank = {n: i for i, n in enumerate(order)}
    triples.sort(key=lambda e: (min(rank[idx[e["head"]]], rank[idx[e["tail"]]]), -e["w"]))
    lines = [f'{e["head"]} {sep} {e["rel"]} {sep} {e["tail"]}' for e in triples]
    passages = [nodes[n]["text"] for n in order if is_ev[n]]
    spans, pos = [], 0
    for ln in lines:
        spans.append((pos, pos + len(ln))); pos += len(ln) + 2
    return "; ".join(lines + passages), triples, spans


def build_question_evidence(questions, graph, ft_tok, ft_model, device, kappa=3, rho=12):
    names = [n["name"] if n["type"] != "evidence" else n["text"] for n in graph["nodes"]]
    node_emb = cls_embed(names, ft_tok, ft_model, device)
    q_emb = cls_embed([q.en for q in questions], ft_tok, ft_model, device)
    out = {}
    for q, qe in zip(questions, q_emb):
        text, triples, spans = retrieve(graph, node_emb, qe, ft_tok.sep_token, kappa, rho)
        out[q.qid] = {"text": text, "triples": [[t["head"], t["rel"], t["tail"]] for t in triples], "spans": spans}
    return out
