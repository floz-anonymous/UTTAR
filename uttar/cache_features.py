"""Pre-compute frozen Ultrasound-CLIP patch tokens V = f_v(I) for every KVME image (fp16, N=196, d_v=768)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import image_index, load_cases

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kvme", required=True)
    ap.add_argument("--ckpt", required=True, help="released Ultrasound-CLIP checkpoints.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)["state_dict"]
    sd = {k[len("base_clip."):]: v for k, v in sd.items() if k.startswith("base_clip.")}
    model.load_state_dict(sd, strict=True)
    trunk = model.visual.trunk.cuda().eval()

    cases = load_cases(args.kvme)
    index = image_index(cases)
    paths = sorted(index, key=index.get)
    feats = np.lib.format.open_memmap(out / "patch_tokens.npy", mode="w+", dtype=np.float16, shape=(len(paths), 196, 768))
    with torch.no_grad():
        for i in range(0, len(paths), args.bs):
            x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + args.bs]]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok = trunk.forward_features(x)[:, 1:]  # drop CLS: the paper uses the N patch tokens
            feats[i:i + len(x)] = tok.float().cpu().numpy().astype(np.float16)
            if i % (args.bs * 20) == 0:
                print(f"{i}/{len(paths)}", flush=True)
    feats.flush()
    (out / "patch_index.json").write_text(json.dumps(index))
    (out / "info.json").write_text(json.dumps({"encoder": "Ultrasound-CLIP released checkpoint (BiomedCLIP ViT-B/16)",
                                               "ckpt": args.ckpt, "n_images": len(paths), "tokens": 196, "dim": 768,
                                               "preprocess": str(preprocess)}, indent=2))
    print("done", len(paths))


if __name__ == "__main__":
    main()
