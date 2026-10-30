# Validation

Preparation checks and their actual results are recorded in `validation_report.json` alongside this file.
The two delivered checkpoint copies are SHA-256 checked against the recovered SSD source. The production
hash is additionally checked against the archived original contour run's production hash.

`tests/test_pipeline.py` uses synthetic images and explicitly fake segmentation/classification models to test
coordinate handling, exports, empty images, expert routing, metadata aggregation, device selection and resume.
`tests/test_distribution.py` checks checksum verification and safe handling of mismatched/missing release assets.
These tests do not measure trained model accuracy. CI deliberately does not download neural weights or private images.

The supplied full dependency pins target Python 3.11. The local scientific smoke-test environment and any actual
checkpoint-load environment are reported separately. Passing synthetic tests is not proof that the full pinned
PyTorch/Cellpose environment has been installed and tested on every supported platform.

A target-machine acceptance run on a representative slide is still needed before routine use. Check segmentation,
RGB decoding, global coordinates, class labels, expert routing and summaries. Historical thesis results are distinct
from outputs of this portable adaptation; see IMPLEMENTATION_NOTES.md.
