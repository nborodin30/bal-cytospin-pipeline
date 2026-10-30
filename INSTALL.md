# Installation and first run

This package targets **64-bit Python 3.11** on Windows, Linux and Apple Silicon Macs. The notebook selects CUDA, then Apple MPS, then CPU when `DEVICE="auto"`. MPS requires a compatible Mac/macOS and an MPS-enabled PyTorch build; availability is checked at runtime. Intel Macs require a separately validated older PyTorch environment.

Use a local SSD folder with sufficient free space. The notebook estimates space from the image dimensions. Disk-backed decoding reduces image-array RAM use, but the neural models still need RAM and segmentation needs working memory. Begin with one preview tile. A laboratory workstation with 16 GB RAM is a reasonable starting point to evaluate, not a guaranteed requirement for every scan. CPU whole-slide processing can be slow; use a suitable GPU workstation for large batches.

## Windows

Install Python 3.11 (64-bit) with the Python launcher. Extract the complete package. Open PowerShell in the extracted folder and run:

```powershell
py -3.11 install_lab_environment.py --compute cpu
& .\.venv\Scripts\python.exe -m jupyterlab BAL_End_to_End_Lab.ipynb
```

For a compatible NVIDIA workstation, laboratory IT can use `--compute cuda124` instead. This installs the documented PyTorch 2.6.0 / torchvision 0.21.0 CUDA 12.4 pair. Driver/GPU support must match that build; newer GPU generations may need a different, separately validated environment.

## Linux or Apple Silicon macOS

With Python 3.11 available, open Terminal in the extracted folder:

```bash
python3.11 install_lab_environment.py --compute auto
.venv/bin/python -m jupyterlab BAL_End_to_End_Lab.ipynb
```

On compatible Linux/NVIDIA systems, IT can use `--compute cuda124`. macOS uses the standard PyTorch package, which includes MPS support; `--compute mps` is an explicit macOS installer option. The installer selects a package build, while the notebook's DEVICE setting selects the runtime device. Set `DEVICE="cpu"` to force CPU execution.

On Apple Silicon, start with `DEVICE="auto"`, `CLASSIFIER_DEVICE="same"` and the supplied conservative batch sizes. If a classifier has an unsupported MPS operation, set `CLASSIFIER_DEVICE="cpu"` to keep Cellpose on MPS while moving classification to CPU. If memory is tight, reduce Cellpose batch size to 1 and classifier batch size to 4 or 1. The notebook keeps MPS memory safeguards and does not silently rerun failed jobs on another device. Do not disable the PyTorch MPS high-watermark safeguard to force a large run.

## Prepare the files

Copy these local weights into `models/`:

- `dinobloom_bal_production.pth` — trained eight-class production state dictionary.
- `neutrophil_expert_finetuned.pth` — trained three-class expert state dictionary.
These two custom classifiers must be supplied by the project owner. Generic pretrained DinoBloom weights cannot replace their trained classifier heads. The two custom weights are distributed as GitHub Release assets; run scripts/get_models.py after cloning. Original slide data are not distributed.

Cellpose is separate: leave `CELLPOSE_MODEL="cpsam"` in the notebook. The pinned library downloads the original built-in `cpsam` once into its model cache and reuses it afterwards. First use needs internet unless the cache is already populated. It does not upload slide images. To use an existing project file offline, set `CELLPOSE_MODEL` to that file's full path instead. A missing explicit local path fails rather than silently falling back to another model. The actual segmentation file hash is recorded in the run report.

Put a single initial CZI into `input_slides/`. A single-plane 8-bit RGB TIFF export is also supported. Fluorescence stacks, uint16 images, multiple non-singleton scenes/time/Z planes and TIFF pyramids are rejected rather than silently converted or reduced. Keep the acquisition scale and staining consistent with the trained data. CZI pixel spacing is reported when available; the notebook does not resize slides to a target micrometre resolution.

Open the notebook and select **BAL analysis (Python 3.11)**. Run from the top, select the slide and run preflight. `RUN_MODE="full"` produces the complete annotated TIFF by default. For an initial trial, set `RUN_MODE="preview"`. Use a preview start tile other than zero when the first tile is empty. The tile limit counts **visited tiles**, including empty/background tiles. Set the mode back to full for the finished whole-slide TIFF.

## One real-slide acceptance run

Before handing the workflow over for routine use:

1. Use a representative slide, the two project-trained classifiers, and the explicitly selected original `cpsam` (cached by Cellpose or supplied locally).
2. Confirm the device, input dimensions, pixel spacing when available, weight paths and fixed class order.
3. Inspect several original/contour PNG pairs. Check RGB appearance, coordinates, cell boundaries and class labels.
4. Check the coloured particle-count and neutrophil subtype labels. Use the confidence columns in `cells.csv` to select a sample of low-confidence expert assignments for review.
5. Run the complete slide and check that the eight counts sum to total retained objects; expert category counts must sum to neutrophils when enabled.
6. Open the BigTIFF in a compatible image viewer and confirm full image dimensions. There is no pyramid.
7. Re-run with identical settings. The run should be reused after output checksums pass.
8. Save the run reports together with the results. Record target machine/environment and the human review outcome.

No model-quality or runtime promise is inferred from synthetic software tests.

## Troubleshooting

- **Module not found / wrong kernel**: select the BAL kernel; re-run installation from the package folder. Keep the `.py` module beside the notebook.
- **Missing checkpoint**: copy the exact trained file or edit the weight path in the notebook. An original pretrained backbone alone is not the classifier.
- **Cellpose download failed**: the first `cpsam` download needs internet. Check the lab firewall/proxy, or set `CELLPOSE_MODEL` to an existing local `cpsam` file for offline use.
- **State dictionary mismatch**: stop and verify the file. The runner intentionally rejects unmatched/missing parameters.
- **CUDA unavailable**: CPU mode works without NVIDIA CUDA. Ask IT to check the PyTorch build and GPU driver for GPU use.
- **MPS unavailable**: check `torch.backends.mps.is_available()` in the chosen kernel. Use supported Apple Silicon/macOS and the native arm64 Python/PyTorch build. The notebook falls back to CPU only during automatic selection when MPS is unavailable; an explicitly requested unavailable MPS device raises an error.
- **MPS operation error**: keep Cellpose on MPS but set `CLASSIFIER_DEVICE="cpu"` if the error is in classification. Set `DEVICE="cpu"` if Cellpose itself cannot run on the chosen system. Actual MPS throughput and prediction agreement require a real-slide check.
- **Memory error**: reduce classification batch size and Cellpose batch size to one. Keep the original tile size for comparison with production settings. If you change tile size, the configuration is recorded and segmentation results can differ.
- **No detections in trial**: a preview processes the selected visited tiles only. Choose another start tile and inspect the original image; do not interpret an empty corner as a whole-slide finding.
- **Unsupported CZI axes/codec**: export the intended single 8-bit RGB plane from ZEN with validated settings. Do not guess a channel order or auto-rescale a fluorescence image.
- **Insufficient disk space**: choose a larger local results drive. Decoded image arrays and the full TIFF need temporary and permanent disk space. A preview trial still decodes the scan and does not replace the complete TIFF deliverable.
- **Interrupted run**: rerun the full cell. Previously verified complete slides are reused. An interrupted slide restarts in a new attempt folder. Its failed/interrupted output is not included in the new batch summary. There is no mid-tile or mid-slide resume.
- **No animal/group tables**: fill the metadata CSV with explicit animal IDs, treatment and timepoint. Group SD requires more than one animal. Partial previews never produce animal/group tables.

## Primary implementation references

- [Original DinoBloom-Base timm architecture configuration](https://huggingface.co/1aurent/vit_base_patch14_224.dinobloom/blob/main/config.json).
- [Cellpose 4.0.6 release and installation](https://pypi.org/project/cellpose/4.0.6/) and [versioned Cellpose model API implementation](https://github.com/MouseLand/cellpose/blob/v4.0.6/cellpose/models.py).
- [czifile 2019.7.2.3](https://pypi.org/project/czifile/2019.7.2.3/). It is pinned because the later CZI API changed.
- [Official matching PyTorch/torchvision installation commands](https://pytorch.org/get-started/previous-versions/#v260).
- [PyTorch2.6 MPS backend](https://docs.pytorch.org/docs/2.6/notes/mps.html) and [Cellpose4.0.6 native MPS handling](https://github.com/MouseLand/cellpose/blob/v4.0.6/cellpose/models.py).

Keep the original model/data licenses and cite the underlying methods when reporting scientific results.
