# Implementation notes for the project owner

## Source of inference settings

The portable runner follows `full_slide_biological_analysis_production_model.py` and `apply_neutrophil_expert.py` supplied with the thesis analysis package. It does not use the different v2 contour-preview inference as its analysis engine.

Retained production settings include 2500px tiles, reflection padding 256px, Cellpose flow threshold 0.9 and cell-probability threshold 0.0, minimum segmented area 150px, tile background mean thresholds 10/245, the saturation/contrast crop filter, global 224px centroid crops, and ImageNet channel normalization. The production classifier uses one forward pass. The expert uses four rotations and averages logits before softmax. Class order is fixed in `bal_pipeline.py` and shown in the notebook.

The native timm architecture `vit_base_patch14_dinov2` is instantiated with image size 224 and average pooling, matching the original Hugging Face model configuration plus the production override. All trained parameters are loaded strictly from the provided local state dictionary. No Hub metadata request or substitute generic classifier is used during inference.

Version 1.0.1 makes the original built-in Cellpose `cpsam` automatic by default. The pinned Cellpose 4.0.6 cache helper is called only when processing starts, not during import or preflight. Its resolved local file is hashed and that exact path is passed to CellposeModel. The two custom classifier paths are validated first. Explicit local segmentation paths remain supported for offline use and fail if absent. No historical hash equality is asserted for a fresh download.

Version 1.0.2 adds native MPS selection through Cellpose's existing API. Automatic order is CUDA, MPS, CPU. The original `device` setting controls Cellpose; `classifier_device="same"` follows it and can be overridden independently. MPS forces effective Cellpose float32 even when bfloat16 was requested. Both resolved stage devices and effective precision enter the fingerprint before output reuse, and are recorded in the run report. No MPS memory guard is disabled, no unsupported-operation fallback is enabled implicitly, and no failed inference is silently rerun on CPU. Cellpose 4.0.6 still constructs parts of the masks on CPU. Segmentation grid, thresholds, normalization, particles and TTA are unchanged. MPS-specific hardware speed/accuracy has not been verified with real weights.

Version 1.0.3 changes TIFF and PNG presentation only. A single final-class label uses the corresponding contour colour on a white background. Enabled expert assignments replace the generic neutrophil label with the subtype directly; the neutrophil colour is retained. Percentages and source-class labels are omitted from images, while all probabilities and original/final classes remain in CSV. Every macrophage-with-particles label includes `P:n`, including zero; other eligible cells retain a count when positive. `P` is the existing crop dark-structure proxy. Particle-rule reassignment retains its asterisk. Inference, counting, class colours and biological summaries are unchanged.

Erythrocytes remain class index 3, retain `Ery` contour/text annotations and all-object counts, and never enter the neutrophil expert. Only WBC-specific denominators omit erythrocytes. Existing segmentation/crop filtering is unchanged, so this is a routing/retention guarantee for accepted predictions rather than a claim of complete erythrocyte recall.

Particles retain the production full-crop RGB<75 rule, minimum peak distance2, large connected dark-region contour area>=2000 counting as one aggregate, and the >=3 rule assigning eligible original macrophage/monocyte predictions to `Macrophage_with_particles`. The legacy `Nanotubes_Count` column is included as an alias for `Dark_Structure_Count`, not a stronger physical interpretation.

## Deliberate software changes

- Corrected duplicate checks to use global bounding boxes and consistent spatial bins. The original script indexed pixel tile origins but searched neighbouring indices at +/-1 and compared local boxes. This portable version is explicitly versioned rather than claimed numerically identical to historical outputs.
- Preserved independent reflection-padded core tiles. Masks are trimmed to each core before object extraction. A cell crossing a core boundary can still be fragmented; NMS cannot merge two fragments whose boxes do not overlap. `Touches_Tile_Border` flags such objects for review. A real neighbouring-image halo/ownership scheme would be a separate future algorithmic revision.
- Classification works in bounded crop batches. The decoded slide and optional annotation canvas are disk-backed. This still decodes the whole slide once; it is not an arbitrary-region streaming CZI reader.
- Explicit 8-bit RGB/axis checks replace silent selection of the first scene/Z/time plane and automatic intensity rescaling. Extra singleton axes are removed; non-singleton non-image axes are rejected.
- Float32 Cellpose inference is the default for CPU portability (`cellpose_bfloat16=False`), whereas original Cellpose version defaults may have used bfloat16. Device, precision setting, hashes and software versions are recorded; no numeric equivalence claim is made across environments.
- Safe full schemas handle zero cells, and undefined denominator percentages remain missing. Disabled expert output is distinguished from enabled expert with zero neutrophils.
- External mask polygons are saved with linked Cell_IDs. Holes are not retained in polygon JSON; this is not a full instance-label raster export. No quantitative Dice/IoU validation is inferred from contours.
- Full-resolution tiled annotated BigTIFF is the primary notebook output; full mode is the notebook default. PNG pairs and CSV tables are supporting outputs. Neutrophil labels show the expert subtype directly when enabled. No pyramid is generated. Existing source scans and checkpoints are never edited.
- Preview limits count visited tiles, including blank tiles. Preview outputs always retain partial scope, even when a small test image fits inside the limit. A full mode run is required for biological summaries.
- Resume reuses completed individual-slide runs only after matching input/model/config/software fingerprints and all requested output checksums. An interrupted slide restarts in a fresh attempt; completed previous slides can be reused. Output aggregation uses only the runs explicitly selected by the current invocation.
- No historical treatment mappings/exclusions are hard-coded. Animal aggregation requires explicit metadata and pools counts across preparations before calculating proportions. Group mean/sample SD are based on animals.

## Scope of validation

See docs/VALIDATION.md and tests/test_pipeline.py. Synthetic tests use actual NumPy/OpenCV/scikit-image/tifffile operations plus clearly fake segmentation/classification outputs. The pipeline module and notebook source are syntax-checked. Notebook cells are additionally executed in a synthetic harness with a test configuration. These checks test software flow and exports, not learned segmentation/classification performance, actual CZI decoder compatibility with the lab's scanner files, or real checkpoint loading.

The original lab handover was prepared without the custom weights. The repository now includes local copies of the two custom weights and a separate checkpoint check; see docs/VALIDATION.md for checks actually performed. A real-slide acceptance run remains necessary; INSTALL.md contains the checklist. Target environment installation pins are proposed for handover and were not fully installed with PyTorch/Cellpose during preparation.
