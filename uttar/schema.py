"""Question schema and finding-level extractor E for KVME (Chinese) reports.

Each question is a Yes/No clinical question with a fixed bilingual template. Attribute questions have a
parent presence question (the parent-child pairs P used by L_cons and the inference-time filter).

extract(report, organ) -> {qid: 1 (Yes) | 0 (No)} follows Appendix B of the method:
  findings asserted -> Yes, explicitly negated -> No, unmentioned -> not instantiated.
Contradictory statements about the same question inside one report (e.g. two nodules, one with a clear and
one with an unclear boundary) leave that question uninstantiated.
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Q:
    qid: str
    organ: str
    zh: str
    en: str
    parent: str | None = None


QUESTIONS = [
    # ---------------- Breast ----------------
    Q("B_nodule", "Breast", "乳腺内是否可见结节？", "Is a breast nodule present?"),
    Q("B_nodule_hypo", "Breast", "该乳腺结节是否为低回声？", "Is the breast nodule hypoechoic?", "B_nodule"),
    Q("B_nodule_anechoic", "Breast", "该乳腺结节是否为无回声？", "Is the breast nodule anechoic?", "B_nodule"),
    Q("B_nodule_mixed", "Breast", "该乳腺结节是否为混合回声？", "Is the breast nodule of mixed echogenicity?", "B_nodule"),
    Q("B_nodule_boundary_clear", "Breast", "该乳腺结节边界是否清晰？", "Is the boundary of the breast nodule clear?", "B_nodule"),
    Q("B_nodule_shape_regular", "Breast", "该乳腺结节形态是否规整？", "Is the shape of the breast nodule regular?", "B_nodule"),
    Q("B_nodule_flow", "Breast", "该乳腺结节是否可探及血流信号？", "Is blood flow detected in the breast nodule?", "B_nodule"),
    Q("B_nodule_calc", "Breast", "该乳腺结节内是否可见钙化或点状强回声？", "Are calcifications present in the breast nodule?", "B_nodule"),
    Q("B_gland_disorder", "Breast", "乳腺腺体结构是否紊乱？", "Is the glandular structure of the breast disordered?"),
    Q("B_gland_echo_uniform", "Breast", "乳腺腺体回声是否均匀？", "Is the breast glandular echotexture homogeneous?"),
    Q("B_duct_dilated", "Breast", "乳导管是否扩张？", "Are the mammary ducts dilated?"),
    Q("B_axillary_ln", "Breast", "腋下是否探及淋巴结？", "Are axillary lymph nodes detected?"),
    Q("B_axillary_ln_hilum", "Breast", "该腋下淋巴结是否可见淋巴门结构？", "Is a fatty hilum visible in the axillary lymph node?", "B_axillary_ln"),
    # ---------------- Thyroid ----------------
    Q("T_size_normal", "Thyroid", "甲状腺大小形态是否正常？", "Is the thyroid normal in size and shape?"),
    Q("T_parenchyma_uniform", "Thyroid", "甲状腺腺体回声是否均匀？", "Is the thyroid parenchymal echotexture homogeneous?"),
    Q("T_gland_flow_increased", "Thyroid", "甲状腺腺体内血流信号是否增多？", "Is parenchymal blood flow of the thyroid increased?"),
    Q("T_nodule", "Thyroid", "甲状腺内是否可见结节？", "Is a thyroid nodule present?"),
    Q("T_nodule_hypo", "Thyroid", "该甲状腺结节是否为低回声？", "Is the thyroid nodule hypoechoic?", "T_nodule"),
    Q("T_nodule_anechoic", "Thyroid", "该甲状腺结节是否为无回声？", "Is the thyroid nodule anechoic?", "T_nodule"),
    Q("T_nodule_mixed", "Thyroid", "该甲状腺结节是否为囊实混合回声？", "Is the thyroid nodule mixed cystic-solid?", "T_nodule"),
    Q("T_nodule_iso_hyper", "Thyroid", "该甲状腺结节是否为等回声或高回声？", "Is the thyroid nodule iso- or hyperechoic?", "T_nodule"),
    Q("T_nodule_boundary_clear", "Thyroid", "该甲状腺结节边界是否清晰？", "Is the boundary of the thyroid nodule clear?", "T_nodule"),
    Q("T_nodule_shape_regular", "Thyroid", "该甲状腺结节形态是否规整？", "Is the shape of the thyroid nodule regular?", "T_nodule"),
    Q("T_nodule_flow", "Thyroid", "该甲状腺结节是否可探及血流信号？", "Is blood flow detected in the thyroid nodule?", "T_nodule"),
    Q("T_nodule_calc", "Thyroid", "该甲状腺结节内是否可见点状强回声或钙化？", "Are echogenic foci or calcifications present in the thyroid nodule?", "T_nodule"),
    Q("T_nodule_comet", "Thyroid", "该甲状腺结节内是否可见彗星尾征？", "Is a comet-tail artifact present in the thyroid nodule?", "T_nodule"),
    Q("T_cervical_ln", "Thyroid", "颈部是否探及淋巴结？", "Are cervical lymph nodes detected?"),
    Q("T_cervical_ln_hilum", "Thyroid", "该颈部淋巴结是否可见淋巴门结构？", "Is a fatty hilum visible in the cervical lymph node?", "T_cervical_ln"),
    # ---------------- Liver / upper abdomen ----------------
    Q("L_size_normal", "Liver", "肝脏大小形态是否正常？", "Is the liver normal in size and shape?"),
    Q("L_capsule_smooth", "Liver", "肝包膜是否光滑？", "Is the liver capsule smooth?"),
    Q("L_fatty", "Liver", "肝实质回声是否细密增强？", "Is the liver parenchymal echo fine and increased (fatty pattern)?"),
    Q("L_hepatorenal", "Liver", "肝肾回声对比是否增强？", "Is the hepatorenal echo contrast increased?", "L_fatty"),
    Q("L_attenuation", "Liver", "肝脏后场回声是否衰减？", "Is there far-field attenuation in the liver?", "L_fatty"),
    Q("L_portal_unclear", "Liver", "门静脉系统显示是否欠清晰？", "Is the portal venous system poorly visualised?", "L_fatty"),
    Q("L_lesion", "Liver", "肝内是否可见局灶性病变？", "Is a focal liver lesion present?"),
    Q("L_lesion_anechoic", "Liver", "该肝内病变是否为无回声？", "Is the liver lesion anechoic?", "L_lesion"),
    Q("L_lesion_hyper", "Liver", "该肝内病变是否为强回声或高回声？", "Is the liver lesion hyperechoic?", "L_lesion"),
    Q("L_bile_duct_dilated", "Liver", "肝内外胆管是否扩张？", "Are the intra- and extrahepatic bile ducts dilated?"),
    Q("L_portal_vein_normal", "Liver", "门静脉主干内径是否正常？", "Is the main portal vein calibre normal?"),
    Q("G_size_normal", "Liver", "胆囊大小形态是否正常？", "Is the gallbladder normal in size and shape?"),
    Q("G_removed", "Liver", "胆囊是否已切除？", "Has the gallbladder been removed?"),
    Q("G_wall_thick", "Liver", "胆囊壁是否增厚？", "Is the gallbladder wall thickened?"),
    Q("G_wall_smooth", "Liver", "胆囊壁是否光滑？", "Is the gallbladder wall smooth?"),
    Q("G_wall_comet", "Liver", "胆囊壁是否可见彗星尾征？", "Is a comet-tail artifact seen on the gallbladder wall?"),
    Q("G_lesion", "Liver", "胆囊内是否可见异常回声（结石或息肉样病变）？", "Is a gallbladder lesion (stone or polyp-like echo) present?"),
    Q("G_lesion_shadow", "Liver", "该胆囊病变后方是否伴声影？", "Does the gallbladder lesion cast an acoustic shadow?", "G_lesion"),
    Q("G_lesion_mobile", "Liver", "该胆囊病变是否随体位改变而移动？", "Does the gallbladder lesion move with position?", "G_lesion"),
    Q("P_size_normal", "Liver", "胰腺大小形态是否正常？", "Is the pancreas normal in size and shape?"),
    Q("P_duct_dilated", "Liver", "胰管是否扩张？", "Is the pancreatic duct dilated?"),
    Q("P_lesion", "Liver", "胰腺内是否可见占位性病变？", "Is a pancreatic mass present?"),
    Q("S_size_normal", "Liver", "脾脏大小形态是否正常？", "Is the spleen normal in size and shape?"),
    Q("S_lesion", "Liver", "脾内是否可见占位性病变？", "Is a splenic mass present?"),
]
QMAP = {q.qid: q for q in QUESTIONS}
ORGAN_QIDS = {o: [q.qid for q in QUESTIONS if q.organ == o] for o in ("Breast", "Thyroid", "Liver")}
PARENT_CHILD = [(q.parent, q.qid) for q in QUESTIONS if q.parent]

# Mutually exclusive value groups: when one value is asserted the others are implied No.
# Used only to add extra verifier training items (complementary negatives), never as planner targets.
EXCLUSIVE = [
    ["B_nodule_hypo", "B_nodule_anechoic", "B_nodule_mixed"],
    ["T_nodule_hypo", "T_nodule_anechoic", "T_nodule_mixed", "T_nodule_iso_hyper"],
    ["L_lesion_anechoic", "L_lesion_hyper"],
]

SPLIT_SENT = re.compile(r"[。]")
SPLIT_CLAUSE = re.compile(r"[，,；;：:]")
NEG = re.compile(r"未见|未探及|未及|不扩张|不宽|不厚|无明显|没有")


def _neg(clause):
    return NEG.search(clause.replace("无回声", "")) is not None


def _echo(clause):
    c = clause
    if re.search(r"囊实(混合)?|混合回声|混合性回声", c):
        return "mixed"
    if re.search(r"(中低|稍低|低)回声", c):
        return "hypo"
    if re.search(r"无回声", c):
        return "anechoic"
    if re.search(r"(等|高|强)回声", c):
        return "iso_hyper"
    return None


def _boundary(c):
    if re.search(r"边界(欠清|不清|模糊)", c):
        return 0
    if re.search(r"边界(尚)?(清晰|清楚|清)", c):
        return 1
    return None


def _shape(c):
    if re.search(r"形态(欠|不)(规整|规则)", c):
        return 0
    if re.search(r"形态(尚)?(规整|规则)", c):
        return 1
    return None


def _flow(c):
    if not re.search(r"血流", c):
        return None
    if re.search(r"(未探及|未见)[^，。；]{0,8}血流", c):
        return 0
    if re.search(r"(可探及|探及|可见|见)[^，。；]{0,8}血流", c):
        return 1
    return None


PRESENCE = {"B_nodule", "B_axillary_ln", "T_nodule", "T_cervical_ln", "L_lesion", "G_lesion", "P_lesion", "S_lesion"}


class _Acc:
    """Collects votes per question. Presence questions: any Yes wins (e.g. left side clear, right side
    nodule). Other questions: contradictory votes inside one report cancel the question."""

    def __init__(self):
        self.v = {}

    def put(self, qid, val):
        if val is None:
            return
        self.v.setdefault(qid, set()).add(int(val))

    def result(self):
        out = {}
        for k, s in self.v.items():
            if k in PRESENCE and 1 in s:
                out[k] = 1
            elif len(s) == 1:
                out[k] = next(iter(s))
        return out


def _breast_thyroid(report, organ):
    acc = _Acc()
    P = "B" if organ == "Breast" else "T"
    ln_q, ln_kw = ("B_axillary_ln", "腋下") if organ == "Breast" else ("T_cervical_ln", "颈部")
    for sent in SPLIT_SENT.split(report):
        target = None  # "nodule" | "ln"
        ln_sentence = ln_kw in sent or "淋巴" in sent
        for c in SPLIT_CLAUSE.split(sent):
            c = c.strip()
            if not c:
                continue
            # --- lymph nodes ---
            if ln_sentence and (ln_kw in c or "淋巴结" in c or ("结节" in c and ln_kw in sent)):
                if _neg(c) and ("淋巴结" in c or "结节" in c):
                    acc.put(ln_q, 0); target = None
                elif "淋巴结" in c or "结节" in c:
                    acc.put(ln_q, 1); target = "ln"
                continue
            if "淋巴门" in c:
                if target == "ln" or ln_sentence:
                    acc.put(ln_q + "_hilum", 0 if _neg(c) else 1)
                continue
            # --- subject switches back to the gland ---
            if re.search(r"余腺体|腺体回声|腺体结构|腺体内|甲状腺大小|甲状腺形态|乳导管|导管|腺体层|皮下", c):
                target = None
            # --- gland-level findings ---
            if organ == "Breast":
                if re.search(r"腺体结构.{0,2}紊乱|结构.{0,2}紊乱", c):
                    acc.put("B_gland_disorder", 1)
                if re.search(r"导管", c) and "扩张" in c:
                    acc.put("B_duct_dilated", 0 if _neg(c) else 1)
                if target is None and re.search(r"回声(不均匀|欠均匀)", c) and "结节" not in c:
                    acc.put("B_gland_echo_uniform", 0)
                elif target is None and re.search(r"(?<![不欠])均匀", c) and "回声" in c and "结节" not in c:
                    acc.put("B_gland_echo_uniform", 1)
            else:
                if re.search(r"大小形态(如常|尚可|正常)|体积正常", c):
                    acc.put("T_size_normal", 1)
                elif re.search(r"(形态|体积).{0,2}(饱满|增大|肿大)", c):
                    acc.put("T_size_normal", 0)
                if target is None and "腺体" in c and "回声" in c:
                    if re.search(r"不均|欠均|增粗|减低|网格", c):
                        acc.put("T_parenchyma_uniform", 0)
                    elif "均匀" in c:
                        acc.put("T_parenchyma_uniform", 1)
                if "腺体内" in c and "血流" in c:
                    if re.search(r"增多|丰富|增加", c):
                        acc.put("T_gland_flow_increased", 1)
                    elif _neg(c):
                        acc.put("T_gland_flow_increased", 0)
                    continue
            # --- nodule presence ---
            if re.search(r"占位", c) and _neg(c):
                acc.put(f"{P}_nodule", 0); target = None
                continue
            if re.search(r"结节|肿块|占位", c) and not _neg(c):
                acc.put(f"{P}_nodule", 1); target = "nodule"
            # --- nodule attributes ---
            if target == "nodule":
                e = _echo(c)
                if e:
                    key = f"{P}_nodule_{e}"
                    if key in QMAP:
                        acc.put(key, 1)
                acc.put(f"{P}_nodule_boundary_clear", _boundary(c))
                acc.put(f"{P}_nodule_shape_regular", _shape(c))
                acc.put(f"{P}_nodule_flow", _flow(c))
                if re.search(r"钙化|点状强回声|强回声光点|强回声斑", c):
                    acc.put(f"{P}_nodule_calc", 0 if _neg(c) else 1)
                if P == "T" and "彗星尾" in c:
                    acc.put("T_nodule_comet", 1)
    return acc.result()


LIVER_SUBJECTS = [
    (r"^(肝内外胆管|肝外胆管|胆管|门脉主干及肝内胆管)", "bile"),
    (r"^(门静脉|门脉)", "portal"),
    (r"^(胆囊)", "gb"),
    (r"^(胰腺|胰管)", "pancreas"),
    (r"^(脾脏|脾)", "spleen"),
    (r"^(肝脏|肝内|肝肾|肝)", "liver"),
]


def _liver(report):
    acc = _Acc()
    subject = "liver"
    lesion_target = None
    for sent in SPLIT_SENT.split(report):
        for c in SPLIT_CLAUSE.split(sent):
            c = c.strip()
            if not c:
                continue
            for pat, s in LIVER_SUBJECTS:
                if re.search(pat, c):
                    subject = s; lesion_target = None
                    break
            if "胆管" in c and "扩张" in c:
                acc.put("L_bile_duct_dilated", 0 if _neg(c) else 1)
            if re.search(r"门(静)?脉主干", c):
                if re.search(r"正常|不宽|未见扩张", c):
                    acc.put("L_portal_vein_normal", 1)
                elif re.search(r"增宽|扩张", c):
                    acc.put("L_portal_vein_normal", 0)
            if "门静脉系统" in c and "显示" in c:
                acc.put("L_portal_unclear", 1 if re.search(r"欠清|不清", c) else 0)
            if subject == "liver":
                if re.search(r"大小形态(如常|正常)|形态如常", c):
                    acc.put("L_size_normal", 1)
                elif re.search(r"饱满|体积.{0,2}(大|增)|肿大|增大", c):
                    acc.put("L_size_normal", 0)
                if "包膜" in c:
                    acc.put("L_capsule_smooth", 0 if re.search(r"欠光滑|不光滑|不平", c) else (1 if "光滑" in c else None))
                if "细密增强" in c:
                    acc.put("L_fatty", 1)
                elif re.search(r"实质回声均匀|^回声均匀", c):
                    acc.put("L_fatty", 0)
                if "肝肾回声对比" in c:
                    acc.put("L_hepatorenal", 1 if "增强" in c else 0)
                if re.search(r"后场.{0,3}衰减", c):
                    acc.put("L_attenuation", 1)
                if "占位" in c and _neg(c):
                    acc.put("L_lesion", 0); lesion_target = None
                elif re.search(r"(可见|见).{0,12}(无|强|高|低)回声", c):
                    acc.put("L_lesion", 1); lesion_target = "lesion"
                    e = _echo(c)
                    if e == "anechoic":
                        acc.put("L_lesion_anechoic", 1)
                    elif e == "iso_hyper":
                        acc.put("L_lesion_hyper", 1)
            elif subject == "gb":
                if "切除" in c:
                    acc.put("G_removed", 1)
                if re.search(r"大小形态(如常|正常)", c):
                    acc.put("G_size_normal", 1); acc.put("G_removed", 0)
                elif re.search(r"增大|肿大|饱满|缩小", c):
                    acc.put("G_size_normal", 0); acc.put("G_removed", 0)
                if "壁" in c and re.search(r"厚", c):
                    acc.put("G_wall_thick", 0 if re.search(r"不厚|未见增厚", c) else 1)
                if re.search(r"毛糙|欠光滑|不光滑", c):
                    acc.put("G_wall_smooth", 0)
                elif "光滑" in c:
                    acc.put("G_wall_smooth", 1)
                if "彗星尾" in c:
                    acc.put("G_wall_comet", 1)
                if re.search(r"腔内.{0,6}未见.{0,4}异常回声|^未见.{0,4}异常回声|透声(好|佳)", c):
                    acc.put("G_lesion", 0); lesion_target = None
                elif re.search(r"(可见|见).{0,12}(强回声|高回声|中等回声|低回声)|回声(结节|凸起|团|光团)", c):
                    acc.put("G_lesion", 1); lesion_target = "gb"
                if lesion_target == "gb":
                    if "声影" in c:
                        acc.put("G_lesion_shadow", 0 if re.search(r"无声影|不伴|未见声影", c) else 1)
                    if "体位" in c and "移动" in c:
                        acc.put("G_lesion_mobile", 0 if re.search(r"不随|不移动", c) else 1)
            elif subject == "pancreas":
                if re.search(r"大小形态(如常|正常)", c):
                    acc.put("P_size_normal", 1)
                elif re.search(r"增大|肿大|饱满", c):
                    acc.put("P_size_normal", 0)
                if "胰管" in c:
                    acc.put("P_duct_dilated", 0 if re.search(r"不宽|不扩张|未见扩张", c) else (1 if re.search(r"扩张|增宽", c) else None))
                if "占位" in c:
                    acc.put("P_lesion", 0 if _neg(c) else 1)
            elif subject == "spleen":
                if re.search(r"大小形态(如常|正常)", c):
                    acc.put("S_size_normal", 1)
                elif re.search(r"增大|肿大|饱满|厚", c):
                    acc.put("S_size_normal", 0)
                if "占位" in c:
                    acc.put("S_lesion", 0 if _neg(c) else 1)
    return acc.result()


def extract(report, organ):
    raw = _liver(report) if organ == "Liver" else _breast_thyroid(report, organ)
    return {k: v for k, v in raw.items() if k in QMAP and QMAP[k].organ == organ}


def complementary_negatives(answers):
    """Implied No answers for the other values of a single-valued slot (verifier training only)."""
    extra = {}
    for group in EXCLUSIVE:
        yes = [q for q in group if answers.get(q) == 1]
        if len(yes) == 1:
            for q in group:
                if q not in answers:
                    extra[q] = 0
    return extra


def apply_parent_filter(answers):
    """Inference-time hard rule: children of a No parent are No."""
    out = dict(answers)
    for parent, child in PARENT_CHILD:
        if child in out and out.get(parent) == 0:
            out[child] = 0
    return out
