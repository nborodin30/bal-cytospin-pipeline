#!/usr/bin/env python3
"""Command-line interface to the portable BAL inference engine."""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='+', type=Path, help='CZI/RGB TIFF files or folders of slides')
    p.add_argument('--out', required=True, type=Path, help='Separate results directory')
    p.add_argument('--metadata', type=Path, help='CSV: filename,animal_id,treatment,timepoint')
    p.add_argument('--classifier', type=Path, default=ROOT/'models/dinobloom_bal_production.pth')
    p.add_argument('--expert', type=Path, default=ROOT/'models/neutrophil_expert_finetuned.pth')
    p.add_argument('--cellpose', default='cpsam', help='Original built-in cpsam or an explicit weights file')
    p.add_argument('--no-expert', action='store_true')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    p.add_argument('--classifier-device', choices=['same', 'auto', 'cpu', 'cuda', 'mps'], default='same')
    p.add_argument('--mode', choices=['preview', 'full'], default='preview')
    p.add_argument('--preview-tiles', type=int, default=1)
    p.add_argument('--start-tile', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--cellpose-batch-size', type=int, default=2)
    p.add_argument('--no-tiff', action='store_true', help='Export predictions/contours without the annotated TIFF')
    p.add_argument('--check', action='store_true', help='Preflight only, without loading/downloading weights')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.preview_tiles < 1 or args.start_tile < 0:
        raise ValueError('Preview tiles must be positive and start tile must be non-negative.')
    # Parse --help before importing scientific libraries.
    import bal_pipeline as bp
    paths = []
    for item in args.inputs:
        item = item.expanduser()
        paths.extend(bp.discover_slides(item) if item.is_dir() else [item])
    samples = bp.load_samples(paths, args.metadata)
    config = bp.Config(
        classifier_checkpoint=str(args.classifier), expert_checkpoint=str(args.expert),
        cellpose_checkpoint=args.cellpose, output_dir=str(args.out), enable_expert=not args.no_expert,
        device=args.device, classifier_device=args.classifier_device,
        batch_size=args.batch_size, cellpose_batch_size=args.cellpose_batch_size,
        save_annotated_tiff=not args.no_tiff,
    )
    print(bp.preflight(config, samples).to_string(index=False))
    if args.check:
        return 0
    bp.Pipeline(config).run(samples, mode=args.mode,
                            preview_tiles=args.preview_tiles, start_tile=args.start_tile)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
