"""Render the verified answers of R̂QA as a report in the reference language (Chinese) for NLG / NLI scoring.

R̂QA itself is a list of English Yes/No lines and cannot be compared with the Chinese free-text references, so its
verified answers A are also written out by a deterministic template realizer whose templates are mined from the
*training* reports:
  * every training sentence is parsed by E into its statements {(question, answer)};
  * the template of a statement pattern is the most frequent training sentence with exactly that pattern;
  * A is covered greedily: repeatedly take the pattern P ⊆ A (every statement in the sentence is a verified answer)
    that covers the most still-uncovered answers (ties: the more frequent sentence); stop when nothing new is covered;
  * sentences are ordered by the schema position of their first question.
The rendered text therefore asserts nothing beyond A (as judged by E). Sentences that state no schema finding
(e.g. skin and fat layers) are never emitted.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_cases
from schema import QUESTIONS, extract

ORDER = {q.qid: i for i, q in enumerate(QUESTIONS)}


def mine_templates(train_cases):
    sig = collections.defaultdict(collections.Counter)
    for c in train_cases:
        for s in [x for x in c["report"].split("。") if x]:
            e = extract(s, c["organ"])
            if e:
                sig[(c["organ"], frozenset(e.items()))][s] += 1
    out = collections.defaultdict(list)  # organ -> [(pattern, sentence, count)]
    for (organ, pat), cnt in sig.items():
        s, n = cnt.most_common(1)[0]
        out[organ].append((pat, s, sum(cnt.values())))
    return out


def render(answers, organ, templates):
    A = frozenset(answers.items())
    left, chosen = set(A), []
    while left:
        best = max(((len(p & left), n, p, s) for p, s, n in templates[organ] if p <= A), default=None,
                   key=lambda x: (x[0], x[1]))
        if best is None or best[0] == 0:
            break
        chosen.append((min(ORDER[q] for q, _ in best[2]), best[3]))
        left -= best[2]
    return "。".join(s for _, s in sorted(chosen)) + ("。" if chosen else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kvme", required=True)
    ap.add_argument("--pred", required=True, help="predictions_<split>.jsonl from infer.py (updated in place)")
    args = ap.parse_args()
    templates = mine_templates([c for c in load_cases(args.kvme) if c["split"] == "train"])
    recs = [json.loads(l) for l in open(args.pred, encoding="utf-8")]
    covered = total = 0
    for r in recs:
        r["report_qa_zh"] = render({q: int(a) for q, a in r["answers"].items()}, r["organ"], templates)
        e = extract(r["report_qa_zh"], r["organ"])
        total += len(r["answers"]); covered += sum(e.get(q) == a for q, a in r["answers"].items())
    with open(args.pred, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"rendered {len(recs)} cases; verified answers expressed in the text (by E): {covered}/{total} = {100 * covered / max(total, 1):.1f}%")


if __name__ == "__main__":
    main()
