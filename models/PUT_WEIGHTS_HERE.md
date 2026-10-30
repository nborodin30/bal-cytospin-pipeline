# Production weights

The local delivery contains the two original checkpoints in this directory.
They are intentionally ignored by Git and must be attached to the GitHub Release `thesis-v1.0`.
After cloning, download the two `.pth` files from that release into `models/`, or run:

```bash
python scripts/get_models.py --repo YOUR_GITHUB_USERNAME/bal-cytospin-pipeline
python scripts/verify_models.py
```

SHA-256 values, sizes and exact class orders are in `manifest.json`.
Cellpose's original `cpsam` is obtained by the pinned library at first use, or supplied with `--cellpose /path/to/cpsam` for offline work.
Generic pretrained DinoBloom weights cannot replace either trained classifier.
