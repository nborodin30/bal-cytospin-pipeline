# BAL cytospin cell classification

Whole-slide segmentation and classification of cells in bronchoalveolar lavage (BAL) cytospins.
The workflow combines Cellpose-SAM, an eight-class DinoBloom-Base production classifier,
and a fine-tuned three-class neutrophil expert. It exports per-cell predictions, contours,
annotated TIFF images and sample summaries.

Prepared for the master's thesis **Classification of White Blood Cells in Cytospins for Prediction of Lung Diseases**, Mykyta Borodin.

## Models

- `dinobloom_bal_production.pth`: the supervised production classifier used by the original whole-slide analysis.
- `neutrophil_expert_finetuned.pth`: the final fine-tuned expert, applied only to cells classified as neutrophils.
- Cellpose-SAM: the original built-in `cpsam` from pinned Cellpose 4.0.6, downloaded once by that library. A local segmentation checkpoint can also be supplied explicitly.

The two custom weights are prepared for attachment to the **`thesis-v1.0` GitHub Release**, rather than being committed to Git.
The prepared local folder already contains them in `models/`; a fresh clone requires downloading them.
Publication of this local preparation is a separate step; no live GitHub repository URL has yet been assigned.
Checksums and exact class orders are in [`models/manifest.json`](models/manifest.json).

## Quick start

Use 64-bit Python 3.11. Detailed platform instructions are in [`INSTALL.md`](INSTALL.md).

```bash
python3.11 install_lab_environment.py --compute auto
```

Activate the resulting `.venv` before the commands below (`source .venv/bin/activate` on Linux/macOS).
For a fresh GitHub clone, download the two assets manually into `models/`, or use the public-release downloader:

```bash
python scripts/get_models.py --repo YOUR_GITHUB_USERNAME/bal-cytospin-pipeline
python scripts/verify_models.py
python run_pipeline.py input_slides/example.czi --out results --mode preview --preview-tiles 1 --check
python run_pipeline.py input_slides/example.czi --out results --mode preview --preview-tiles 1
python run_pipeline.py input_slides/example.czi --out results --mode full
```

Replace the repository owner and example slide with actual values. `--check` performs preflight only;
it does not download or load model weights. First processing needs internet if `cpsam` is not cached.
Classification uses the supplied local state dictionaries with strict parameter matching and no backbone download.
The notebook offers the same portable engine:

```bash
python -m jupyterlab BAL_End_to_End_Lab.ipynb
```

For a directory of slides and animal-level aggregation:

```bash
python run_pipeline.py input_slides --out results --mode full --metadata metadata_example.csv
```

Replace example metadata rows with real filenames, animal IDs, treatment labels and timepoints.
Counts from preparations belonging to the same animal are pooled before calculating animal proportions.
Without metadata, per-cell and per-slide outputs remain available; animal/group summaries are omitted.

## Outputs and interpretation

Each completed run contains `cells.csv`, `neutrophil_cells.csv`, `slide_summary.csv`,
`external_contours.json`, preview PNGs and an annotated full-resolution BigTIFF, plus a run report
with input/model hashes and software versions. Preview runs are explicitly partial and excluded from full-slide summaries.
Completed runs can be reused after fingerprint and output-checksum verification; interrupted slides restart.

The eight-class order is Macrophage, Macrophage_with_particles, Neutrophil, Erythrocyte,
Lymphocyte, Monocyte, Eosinophil, Junk. Expert categories are Immature_Neu,
Mature_Neu_SiglecF+, Mature_Neu_SiglecF-. Expert scores are image-based assignments to annotation
categories, not direct measurements of Siglec-F expression. Dark-structure counts are an image-derived proxy,
not confirmed individual nanotube counts. The workflow is a research tool, not a validated diagnostic system.

## Reproducibility and scope

This is the portable inference adaptation previously prepared for lab use. Original production settings and
the two original custom checkpoints are retained. The adaptation includes documented implementation changes,
including corrected coordinate handling during deduplication. It does **not** guarantee numerical identity
with the historical thesis outputs. Original scripts are preserved in [`reference/`](reference/).
See [`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md) and [`docs/VALIDATION.md`](docs/VALIDATION.md).

Whole-slide decoding, segmentation and TIFF export require substantial working memory and free disk space.
Preflight estimates disk requirements; CPU processing can be slow. Start with a small preview on the target machine.
CUDA, Apple MPS and CPU are supported by the device-selection code; real performance depends on the environment.

## Code, weights and data availability

This repository contains inference code, an English notebook, dependency pins, synthetic software tests,
original-script references and model metadata. Original microscopy images and annotations are not included.
Additional benchmark and self-training checkpoints will be archived separately on Zenodo; the DOI will be
added after that record is published. See [`docs/ARCHIVAL_MODELS.md`](docs/ARCHIVAL_MODELS.md).

## Attribution and publication

Please acknowledge DinoBloom, DINOv2, Cellpose-SAM and the underlying software when using the workflow.
See [`RIGHTS_AND_CITATIONS.md`](RIGHTS_AND_CITATIONS.md). A new code/weights licence is not assigned by
this preparation. [`docs/PUBLISH.md`](docs/PUBLISH.md) describes uploading the small Git repository and the
two checkpoint assets. No Git LFS is required for that arrangement.
