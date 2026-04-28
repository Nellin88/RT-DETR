"""
crop the Flower dataset into 640×640 slices with SAHI, keeping COCO format.

Usage:
    python tools/sahi_slice_dataset.py
    python tools/sahi_slice_dataset.py --size 640 --overlap 0.2 --min-area 0.1
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from sahi.slicing import slice_coco


SPLITS = ['train', 'val', 'test']

# input dataset path (original, before slicing)
SRC_ROOT = Path('./dataset/Flower')

# output dataset path (after slicing)
DST_ROOT = Path('./dataset/Flower_SAHI')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size',       type=int,   default=640,  help='slice size (square)')
    parser.add_argument('--overlap',    type=float, default=0.2,  help='slice overlap ratio 0~1')
    parser.add_argument('--min-area',   type=float, default=0.1,  help='minimum annotated area ratio (relative to slice area)')
    parser.add_argument('--ignore-neg', action='store_true',      help='ignore slices without annotations (negative samples)')
    return parser.parse_args()


def main():
    args = parse_args()

    slice_size    = args.size
    overlap       = args.overlap
    min_area      = args.min_area
    ignore_neg    = args.ignore_neg

    print(f'Slice size: {slice_size}×{slice_size}  Overlap: {overlap*100:.0f}%  '
          f'Minimum annotated area: {min_area*100:.0f}%  Ignore negative samples: {ignore_neg}')
    print()

    for split in SPLITS:
        ann_file  = SRC_ROOT / 'annotations' / f'{split}.json'
        img_dir   = SRC_ROOT / 'images' / split

        if not ann_file.exists():
            print(f'[Skipped] {split}: File not found - {ann_file}')
            continue

        out_img_dir = DST_ROOT / 'images' / split
        out_ann_dir = DST_ROOT / 'annotations'
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_ann_dir.mkdir(parents=True, exist_ok=True)

        print(f'[{split}] Processing...')

        coco_dict, _ = slice_coco(
            coco_annotation_file_path = str(ann_file),
            image_dir                 = str(img_dir),
            output_coco_annotation_file_name = f'{split}',
            output_dir                = str(out_img_dir),
            slice_height              = slice_size,
            slice_width               = slice_size,
            overlap_height_ratio      = overlap,
            overlap_width_ratio       = overlap,
            min_area_ratio            = min_area,
            ignore_negative_samples   = ignore_neg,
            verbose                   = False,
        )

        # slice_coco writes the json in output_dir, move it to annotations/
        generated_json = out_img_dir / f'{split}_coco.json'
        target_json    = out_ann_dir / f'{split}.json'
        if generated_json.exists():
            shutil.move(str(generated_json), str(target_json))

        n_images = len(coco_dict.get('images', []))
        n_anns   = len(coco_dict.get('annotations', []))
        print(f'  → Sliced images: {n_images}  Annotations: {n_anns}  '
              f'Saved to: {out_ann_dir}/{split}.json')

    print()
    print('save to:', DST_ROOT)
    print()
    print('In flower_detection.yml, change the paths to:')
    print('  img_folder (train): ./dataset/Flower_SAHI/images/train/')
    print('  ann_file   (train): ./dataset/Flower_SAHI/annotations/train.json')
    print('  img_folder (val):   ./dataset/Flower_SAHI/images/val/')
    print('  ann_file   (val):   ./dataset/Flower_SAHI/annotations/val.json')


if __name__ == '__main__':
    main()
