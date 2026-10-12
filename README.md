# UTTAR: Query-Guided Ultrasound Visual–Textual Alignment with Structured Reasoning for QA-Style Report Generation

Code for an anonymous ACL Rolling Review submission. Please do not redistribute during the review period.

## Overview

UTTAR writes an ultrasound report as a set of clinical Yes/No questions, each with a verified answer and cited evidence.

- **Stage 1 (alignment).** A frozen Ultrasound-CLIP image encoder produces patch tokens. A Clinical Q-Former summarises them into query tokens. The query tokens are aligned with the paired report by a global contrastive loss, a token-level loss and an image–text matching loss with same-organ hard negatives.
- **Stage 2 (generation).** A decoder writes an intermediate report from the query tokens. A planner derives the schema questions from that report.
- **Stage 3 (reasoning).** For each question, a multimodal verifier answers Yes/No from question-conditioned visual evidence and a question-specific subgraph retrieved from a clinical knowledge graph. A parent–child consistency constraint links attribute questions to their finding. A template realizer writes the QA-style report.

## Repository structure

```
uttar/
├── data.py            KVME loading, tokenisers, cached-feature access
├── schema.py          52 Yes/No questions, parent–child pairs, rule-based finding extractor for KVME reports
├── kg.py              clinical knowledge graph and question-specific GraphRAG retrieval
├── model.py           Q-Former, projections, ITM head, report decoder / planner, verifier, consistency loss
├── cache_features.py  frozen Ultrasound-CLIP patch tokens for all images (run once)
├── train.py           Phase A (alignment) and Phase B (generation + QA); model selection on validation
├── infer.py           inference: report, questions, verified answers, QA-style report with evidence
├── render_qa.py       verified answers written as report text, for text metrics
├── baselines.py       copy-a-training-report baselines (majority, nearest neighbour)
├── evaluate.py        NLG, clinical efficacy, entity-level NLI, QA, significance tests
└── metrics.py         metric implementations shared by train.py and evaluate.py
scripts/
└── run_kvme.sh        all steps end to end
```

## Requirements

- Linux and one NVIDIA GPU. We used a single 48 GB GPU; the cached image and text features alone occupy about 7 GB of GPU memory.
- Python 3.12, PyTorch 2.x with CUDA.
- Java (any recent JRE) for the METEOR scorer of `pycocoevalcap`.

```bash
conda create -n uttar python=3.12 openjdk=17 -c conda-forge
conda activate uttar
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # pick the wheel for your CUDA
pip install -r requirements.txt
```

## Data: KVME

We use the KVME ultrasound report dataset (Li et al., *IEEE Transactions on Medical Imaging*, 2025), which covers breast, thyroid and liver examinations.
- **Content:** each case has two de-identified B-mode images (512 × 512) and a free-text Chinese findings report.
- **De-identification placeholders:** sizes and locations are replaced by tokens such as `_2DS_`, `_Loc_` and `_SCM_`.
- **Access:** obtain the dataset from its official release under its terms of use. We do not redistribute any data.

Expected layout:

```
<KVME_ROOT>/
├── new_Mammary2.json        # official split with reports (train / val / test)
├── new_Thyroid2.json
├── new_Liver2.json
├── Mammary_report/          # images
├── Thyroid_report/
├── Liver_report/
└── key_technical_words.txt  # jieba user dictionary used by the KVME evaluation
```

### Splits

We use the official KVME split unchanged (cases; 2 images per case, 14,707 images in total).

| Organ | Train | Validation | Test | Total | Schema questions |
|---|---:|---:|---:|---:|---:|
| Breast | 2,458 | 352 | 703 | 3,513 | 13 |
| Thyroid | 1,719 | 246 | 492 | 2,457 | 15 |
| Liver (upper abdomen) | 975 | 140 | 279 | 1,394 | 24 |
| **Total** | **5,152** | **738** | **1,474** | **7,364** | **52** |

- **Selection:** the validation split is used for checkpoint selection only. The test split is evaluated once with the selected checkpoint.
- **Training-only resources:** the knowledge graph and the realizer templates are built from the training split only.
- **Duplicate references:** many KVME reports are templated, and some test reports occur verbatim in the training split (Breast 51.4%, Thyroid 19.7%, Liver 55.2%). `evaluate.py` therefore also reports text metrics separately for *novel* and *duplicate* test references.

### Question supervision

- **Questions:** each organ has a fixed schema of Yes/No questions, 52 in total (`uttar/schema.py`). Attribute questions are children of a finding-presence question, giving 25 parent–child pairs.
- **Labels:** answers are extracted automatically from the reports by a rule-based extractor. A stated finding gives Yes, an explicitly negated finding gives No, and an unmentioned finding is not instantiated. These are automatic labels, not manual annotations.

| Split | Instantiated questions | Yes | No |
|---|---:|---:|---:|
| Train | 42,163 | 23,621 | 18,542 |
| Validation | 6,039 | 3,359 | 2,680 |
| Test | 12,087 | 6,733 | 5,354 |

- **Knowledge graph** (`uttar/kg.py`): 56 typed lexicon nodes (anatomy, finding, appearance, morphology) and 32 short definitional evidence passages. Edges come from finding–attribute co-occurrence in the training reports. In total there are 88 nodes and 128 edges.

## Usage

```bash
export KVME=/path/to/KVME_ROOT
export UCLIP=/path/to/checkpoints.pt
bash scripts/run_kvme.sh          # all steps below; outputs in runs/
```

Step by step:

```bash
1. frozen image features (patch tokens of every image, fp16, ~4.4 GB)
python uttar/cache_features.py --kvme $KVME --ckpt $UCLIP --out runs/cache

2. training: Phase A, then Phase B; checkpoints are selected on validation
python uttar/train.py --kvme $KVME --patch-cache runs/cache --out runs/uttar

3. inference with the selected checkpoint, then the text rendering of the verified answers
for S in val test; do
  python uttar/infer.py --kvme $KVME --patch-cache runs/cache --run runs/uttar --split $S
  python uttar/render_qa.py --kvme $KVME --pred runs/uttar/predictions_$S.jsonl
done

4. copy-a-training-report baselines (optional)
python uttar/baselines.py --kvme $KVME --ckpt $UCLIP --split test --out runs/baselines

5. evaluation
python uttar/evaluate.py --kvme $KVME --pred runs/uttar/predictions_test.jsonl \
       --baseline runs/baselines/predictions_test.jsonl --out runs/uttar/eval_test.json
```

**Outputs:**
- `runs/uttar/log.jsonl`: training curves and validation scores.
- `runs/uttar/phaseA_best.pt`, `runs/uttar/phaseB_best.pt`: selected checkpoints.
- `runs/uttar/kg_graph.json`, `runs/uttar/kg_question_evidence.json`: the frozen knowledge graph and the retrieved evidence per question.
- `runs/uttar/predictions_<split>.jsonl`, one record per case, with:
  - the intermediate report;
  - the planned questions;
  - verifier probabilities and answers;
  - the QA-style report, whose evidence is image regions and knowledge-graph triples;
  - the rendered report text;
  - verifier probabilities on the reference questions.
- `runs/uttar/eval_<split>.json`: all metrics, overall and per organ.

## Hyper-parameters

| | Value |
|---|---|
| Q-Former | M = 32 queries, 6 blocks, width 512, 8 heads, cross-attention in every block, image-slot embedding for the 2 images |
| Shared alignment space | d = 256; temperature τ = 0.07 (fixed) |
| Report decoder / planner | 3-layer Transformer decoders, width 512, shared vocabulary (806 output tokens) |
| Verifier | 4-layer Transformer encoder, width 256; question-conditioned attention with 4 heads |
| GraphRAG | κ = 3 seed nodes, 2-hop expansion, ρ = 12 nodes, evidence truncated to 128 tokens |
| Loss weights | λ1 = λ2 = λ3 = 1 |
| Phase A | 50 epochs, batch 128, AdamW, lr 1e-4, weight decay 0.05, 200 warm-up steps, cosine decay |
| Phase B | 60 epochs, batch 64, AdamW, lr 3e-4 (Q-Former and g_v: 3e-5), same schedule |
| Selection | Phase A: lowest validation alignment loss; Phase B: every 5 epochs, mean of validation BLEU-4 and QA macro-F1 |
| Inference | greedy decoding; answer Yes iff p ≥ 0.5; parent–child filter |
| Other | gradient clipping 1.0, bf16 autocast, seed 0 |

## Compute

| Item | Value |
|---|---|
| Trainable parameters | 59.9M: Q-Former 26.8M, report decoder 13.7M, planner 14.0M, verifier 3.4M, ITM head 1.6M, projections 0.3M |
| Frozen encoders | ViT-B/16 image encoder and SapBERT-XLMR (XLM-R base) text encoder |
| Phase A | about 3 s per epoch on one 48 GB GPU |
| Phase B | about 15 s per epoch, plus about 30 s per validation round |
| Full training | under 20 minutes |
| Inference | about 2.5 min for validation and 5 min for test |
| Evaluation | a few minutes |

### Evaluation details

- **Text metrics:** reports are tokenised with jieba using the KVME dictionary and scored with pycocoevalcap BLEU-1–4, METEOR, ROUGE-L and CIDEr. BERTScore-F1 uses `bert-base-chinese`, layer 8.
- **Clinical efficacy (CE):** follows the KVME multi-label entity evaluation.
  - Each report becomes a binary vector over (question, Yes) and (question, No) statements for its organ.
  - The same extractor is applied to predictions and references.
  - Scores are micro-averaged.
- **Entity-level NLI:** for every question stated in both the reference and the generated report, the reference sentence(s) form the premise and the generated sentence(s) the hypothesis. The pair counts as entailed when the frozen NLI model's top label is "entailment".
- **QA:** Yes is the positive class. Macro-F1 averages the per-question macro-F1.

## Reproducibility notes

- All random seeds are fixed (`--seed 0`). GPU non-determinism can still cause small differences between runs.
- No choice (checkpoint, threshold, loss weight, template) was made on the test split.
- The frozen encoders' outputs are cached once. There is no data augmentation.
