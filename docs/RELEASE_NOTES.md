# Production inference and custom weights

This release pairs the portable BAL inference code with two project-trained state dictionaries:

- `dinobloom_bal_production.pth`: eight-class supervised production classifier.
- `neutrophil_expert_finetuned.pth`: final fine-tuned three-class neutrophil expert.

Download both files into `models/` and verify against `SHA256SUMS.txt` or `models/manifest.json`.
The segmentation model is Cellpose's original built-in `cpsam`, obtained separately by pinned Cellpose 4.0.6.
Other experimental checkpoints belong to the separate Zenodo archive; its DOI will be added after publication.

Source images and annotation data are not included. This release is a documented portable adaptation,
not a claim of identical historical thesis outputs. Read the validation and implementation notes before use.
