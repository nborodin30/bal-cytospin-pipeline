# Attribution and rights

The project-specific code and trained heads originate from Mykyta Borodin's thesis work.
This preparation does not assign a new licence to that work or to the fine-tuned weights.
The owner can add the selected code and checkpoint licence before public distribution; absence of a project
licence is not a grant of unrestricted reuse. Upstream software and model terms continue to apply.

## Underlying methods and implementations

- Koch et al. (2024), **DinoBloom: a foundation model for generalizable cell embeddings in hematology**.
  [Official repository](https://github.com/marrlab/DinoBloom) (lists the citation and upstream Apache-2.0 licence).
- [Original DinoBloom-Base timm configuration](https://huggingface.co/1aurent/vit_base_patch14_224.dinobloom/blob/main/config.json).
- [DINOv2 official repository](https://github.com/facebookresearch/dinov2).
- [Cellpose official repository](https://github.com/MouseLand/cellpose) and
  [the versioned 4.0.6 implementation](https://github.com/MouseLand/cellpose/tree/v4.0.6).
  Cellpose segmentation weights are obtained from upstream, not repackaged in this release.
- [PyTorch](https://pytorch.org/), [timm](https://github.com/huggingface/pytorch-image-models),
  [czifile](https://github.com/cgohlke/czifile), NumPy, pandas, scikit-image, OpenCV, Pillow and tifffile.

`CITATION.cff` describes this thesis inference software. It does not replace citations of these underlying methods.
