"""Copy-a-training-report baselines on KVME (they write no new text, so they show how much template copying scores).

  majority : the most frequent training report of the organ (the same for every case of that organ)
  nearest  : the report of the most similar training case of the same organ; similarity = cosine of the case
             embeddings (mean of the L2-normalised Ultrasound-CLIP image embeddings of the case's images)

Writes predictions_<split>.jsonl with the keys "pred::majority" and "pred::nearest" for evaluate.py --baseline.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_features import MODEL_NAME
from data import load_cases


@torch.no_grad()
def case_embeddings(cases, ckpt, bs=128):
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict({k[len("base_clip."):]: v for k, v in sd.items() if k.startswith("base_clip.")}, strict=True)
    model = model.cuda().eval()
    paths = [p for c in cases for p in c["paths"]]
    feats = []
    for i in range(0, len(paths), bs):
        x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats.append(F.normalize(model.encode_image(x).float(), dim=-1).cpu())
    feats = torch.cat(feats).numpy()
    out, k = [], 0
    for c in cases:
        out.append(feats[k:k + len(c["paths"])].mean(0)); k += len(c["paths"])
    out = np.stack(out)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kvme", required=True)
    ap.add_argument("--ckpt", required=True, help="released Ultrasound-CLIP checkpoints.pt")
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cases = load_cases(args.kvme)
    X = case_embeddings(cases, args.ckpt)
    organ = np.array([c["organ"] for c in cases]); split = np.array([c["split"] for c in cases])
    majority = {o: collections.Counter(c["report"] for c in cases if c["split"] == "train" and c["organ"] == o).most_common(1)[0][0]
                for o in ("Breast", "Thyroid", "Liver")}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / f"predictions_{args.split}.jsonl", "w", encoding="utf-8") as f:
        for i in np.where(split == args.split)[0]:
            c = cases[i]
            tr = np.where((split == "train") & (organ == c["organ"]))[0]
            j = tr[int(np.argmax(X[tr] @ X[i]))]
            f.write(json.dumps({"organ": c["organ"], "uid": c["uid"], "split": c["split"], "report": c["report"],
                                "pred::majority": majority[c["organ"]], "pred::nearest": cases[j]["report"]},
                               ensure_ascii=False) + "\n")
    print("wrote", Path(args.out) / f"predictions_{args.split}.jsonl")


if __name__ == "__main__":
    main()
