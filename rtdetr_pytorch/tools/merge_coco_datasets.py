#!/usr/bin/env python3
"""Merge flower_detection and RoseBlomming into one COCO dataset.

Output layout matches flower_detection:
  <output_root>/
    annotations/
      train.json
      test.json
      val.json        # Optional, when --test-to-val-ratio > 0
    images/
      train/
      test/
      val/            # Optional, when --test-to-val-ratio > 0

Notes
- Category IDs, image IDs, and annotation IDs are reassigned to avoid conflicts.
- All source categories are remapped into a single output class (default: flower).
- By default, RoseBlomming valid split is merged into train.
- Optionally, a configurable fraction of merged test is split into validation.
- The script uses only Python standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPLITS = ("train", "test")


@dataclass
class LoadedSource:
    dataset: str
    source_split: str
    output_split: str
    image_dir: Path
    ann_file: Path
    coco: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge flower_detection and RoseBlomming COCO datasets"
    )
    parser.add_argument(
        "--flower-root",
        default="dataset/flower_detection",
        help="Path to flower_detection dataset root",
    )
    parser.add_argument(
        "--rose-root",
        default="dataset/RoseBlomming",
        help="Path to RoseBlomming dataset root",
    )
    parser.add_argument(
        "--output-root",
        default="dataset/flower_detection_merged",
        help="Output dataset root path",
    )
    parser.add_argument(
        "--valid-to",
        choices=("train", "test", "skip"),
        default="train",
        help="Where to merge RoseBlomming valid split",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to place images into output dataset",
    )
    parser.add_argument(
        "--no-prefix-source",
        action="store_true",
        help="Do not prefix output image names with dataset name",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print summary without writing files",
    )
    parser.add_argument(
        "--class-name",
        default="flower",
        help="Single output class name used for all merged annotations",
    )
    parser.add_argument(
        "--test-to-val-ratio",
        type=float,
        default=0.0,
        help="Fraction of merged test split moved to validation (e.g. 0.5)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used for deterministic test-to-validation split",
    )
    return parser.parse_args()


def load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_sources(args: argparse.Namespace) -> list[LoadedSource]:
    flower_root = Path(args.flower_root)
    rose_root = Path(args.rose_root)

    source_defs: list[tuple[str, str, str, Path, Path]] = []

    for split in SPLITS:
        source_defs.append(
            (
                "flower",
                split,
                split,
                flower_root / "images" / split,
                flower_root / "annotations" / f"{split}.json",
            )
        )

    for split in SPLITS:
        source_defs.append(
            (
                "rose",
                split,
                split,
                rose_root / split,
                rose_root / split / f"{split}_annotations.coco.json",
            )
        )

    if args.valid_to != "skip":
        source_defs.append(
            (
                "rose",
                "valid",
                args.valid_to,
                rose_root / "valid",
                rose_root / "valid" / "valid_annotations.coco.json",
            )
        )

    loaded: list[LoadedSource] = []
    for dataset, source_split, output_split, image_dir, ann_file in source_defs:
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        loaded.append(
            LoadedSource(
                dataset=dataset,
                source_split=source_split,
                output_split=output_split,
                image_dir=image_dir,
                ann_file=ann_file,
                coco=load_coco(ann_file),
            )
        )

    return loaded


def prepare_output_root(
    output_root: Path,
    overwrite: bool,
    dry_run: bool,
    output_splits: list[str],
) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. Use --overwrite to replace it."
            )
        if dry_run:
            print(f"[dry-run] Would remove existing output: {output_root}")
        else:
            shutil.rmtree(output_root)

    if not dry_run:
        (output_root / "annotations").mkdir(parents=True, exist_ok=True)
        for split in output_splits:
            (output_root / "images" / split).mkdir(parents=True, exist_ok=True)


def build_category_maps(
    sources: list[LoadedSource],
    class_name: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[int, int]]]:
    source_category_maps: dict[tuple[str, str], dict[int, int]] = {}
    categories_out: list[dict[str, Any]] = [
        {
            "id": 1,
            "name": class_name,
            "supercategory": "",
        }
    ]

    for src in sources:
        cat_map: dict[int, int] = {}
        categories = src.coco.get("categories", [])
        if not categories:
            raise ValueError(f"No categories found in {src.ann_file}")

        for category in categories:
            if "id" not in category:
                raise ValueError(f"Category without id found in {src.ann_file}")
            cat_map[int(category["id"])] = 1

        source_category_maps[(src.dataset, src.source_split)] = cat_map

    return categories_out, source_category_maps


def make_unique_filename(
    dataset: str,
    original_name: str,
    used_names: set[str],
    prefix_source: bool,
) -> str:
    base_name = Path(original_name).name
    candidate = f"{dataset}_{base_name}" if prefix_source else base_name

    stem = Path(candidate).stem
    suffix = Path(candidate).suffix
    index = 1
    while candidate in used_names:
        candidate = f"{stem}__{index}{suffix}"
        index += 1

    used_names.add(candidate)
    return candidate


def materialize_file(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            # Cross-device hardlink is common; fallback to copy.
            shutil.copy2(src, dst)
        return

    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return

    raise ValueError(f"Unsupported mode: {mode}")


def merge_split(
    output_root: Path,
    split: str,
    sources: list[LoadedSource],
    categories_out: list[dict[str, Any]],
    category_maps: dict[tuple[str, str], dict[int, int]],
    mode: str,
    prefix_source: bool,
    dry_run: bool,
) -> dict[str, Any]:
    output_image_dir = output_root / "images" / split

    used_names: set[str] = set()
    images_out: list[dict[str, Any]] = []
    annotations_out: list[dict[str, Any]] = []

    next_image_id = 1
    next_ann_id = 1

    for src in sources:
        anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in src.coco.get("annotations", []):
            anns_by_image[int(ann["image_id"])].append(ann)

        source_cat_map = category_maps[(src.dataset, src.source_split)]

        for image in src.coco.get("images", []):
            old_image_id = int(image["id"])
            file_name = str(image.get("file_name", "")).strip()
            if not file_name:
                raise ValueError(f"Image without file_name in {src.ann_file}")

            src_image_path = src.image_dir / file_name
            if not src_image_path.exists():
                raise FileNotFoundError(
                    f"Missing image referenced in {src.ann_file}: {src_image_path}"
                )

            new_file_name = make_unique_filename(
                src.dataset, file_name, used_names, prefix_source
            )
            dst_image_path = output_image_dir / new_file_name

            if not dry_run:
                materialize_file(src_image_path, dst_image_path, mode)

            new_image = dict(image)
            new_image["id"] = next_image_id
            new_image["file_name"] = new_file_name
            new_image["license"] = 0
            images_out.append(new_image)

            for ann in anns_by_image.get(old_image_id, []):
                old_cat_id = int(ann["category_id"])
                if old_cat_id not in source_cat_map:
                    raise KeyError(
                        f"Unknown category id {old_cat_id} in {src.ann_file}"
                    )

                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = next_image_id
                new_ann["category_id"] = source_cat_map[old_cat_id]
                annotations_out.append(new_ann)
                next_ann_id += 1

            next_image_id += 1

    merged_coco = {
        "licenses": [{"id": 0, "name": "", "url": ""}],
        "info": {
            "description": "Merged flower_detection and RoseBlomming",
            "version": "1.0",
            "year": datetime.now(timezone.utc).year,
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "categories": categories_out,
        "images": images_out,
        "annotations": annotations_out,
    }

    return merged_coco


def write_coco(path: Path, coco: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)


def renumber_coco_split(
    images_in: list[dict[str, Any]],
    anns_in: list[dict[str, Any]],
    template_coco: dict[str, Any],
) -> dict[str, Any]:
    image_id_map: dict[int, int] = {}
    images_out: list[dict[str, Any]] = []
    for new_image_id, image in enumerate(images_in, start=1):
        new_image = dict(image)
        image_id_map[int(image["id"])] = new_image_id
        new_image["id"] = new_image_id
        images_out.append(new_image)

    anns_out: list[dict[str, Any]] = []
    next_ann_id = 1
    for ann in anns_in:
        old_image_id = int(ann["image_id"])
        if old_image_id not in image_id_map:
            continue

        new_ann = dict(ann)
        new_ann["id"] = next_ann_id
        new_ann["image_id"] = image_id_map[old_image_id]
        anns_out.append(new_ann)
        next_ann_id += 1

    return {
        "licenses": [dict(x) for x in template_coco.get("licenses", [])],
        "info": dict(template_coco.get("info", {})),
        "categories": [dict(x) for x in template_coco.get("categories", [])],
        "images": images_out,
        "annotations": anns_out,
    }


def split_test_into_validation(
    test_coco: dict[str, Any], ratio: float, seed: int
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    images = list(test_coco.get("images", []))
    anns = list(test_coco.get("annotations", []))

    if len(images) < 2:
        raise ValueError("Need at least 2 test images to split into validation")

    val_count = int(round(len(images) * ratio))
    val_count = min(max(val_count, 1), len(images) - 1)

    indices = list(range(len(images)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_index_set = set(indices[:val_count])

    val_images = [img for i, img in enumerate(images) if i in val_index_set]
    test_images = [img for i, img in enumerate(images) if i not in val_index_set]

    val_image_ids = {int(img["id"]) for img in val_images}
    val_anns = [ann for ann in anns if int(ann["image_id"]) in val_image_ids]
    test_anns = [ann for ann in anns if int(ann["image_id"]) not in val_image_ids]

    new_test_coco = renumber_coco_split(test_images, test_anns, test_coco)
    new_val_coco = renumber_coco_split(val_images, val_anns, test_coco)
    moved_file_names = [str(img["file_name"]) for img in val_images]

    return new_test_coco, new_val_coco, moved_file_names


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root)
    prefix_source = not args.no_prefix_source
    class_name = args.class_name.strip()
    if not class_name:
        raise ValueError("--class-name cannot be empty")
    if not 0.0 <= args.test_to_val_ratio < 1.0:
        raise ValueError("--test-to-val-ratio must be in [0.0, 1.0)")

    output_splits = ["train", "test"]
    if args.test_to_val_ratio > 0.0:
        output_splits.append("val")

    sources = collect_sources(args)
    prepare_output_root(output_root, args.overwrite, args.dry_run, output_splits)

    categories_out, category_maps = build_category_maps(sources, class_name)

    sources_by_split: dict[str, list[LoadedSource]] = {"train": [], "test": []}
    for src in sources:
        if src.output_split not in sources_by_split:
            raise ValueError(f"Invalid output split: {src.output_split}")
        sources_by_split[src.output_split].append(src)

    merged_by_split: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        merged_by_split[split] = merge_split(
            output_root=output_root,
            split=split,
            sources=sources_by_split[split],
            categories_out=categories_out,
            category_maps=category_maps,
            mode=args.mode,
            prefix_source=prefix_source,
            dry_run=args.dry_run,
        )

    if args.test_to_val_ratio > 0.0:
        new_test_coco, new_val_coco, val_file_names = split_test_into_validation(
            merged_by_split["test"], args.test_to_val_ratio, args.split_seed
        )
        merged_by_split["test"] = new_test_coco
        merged_by_split["val"] = new_val_coco

        if not args.dry_run:
            test_image_dir = output_root / "images" / "test"
            val_image_dir = output_root / "images" / "val"
            for file_name in val_file_names:
                src_path = test_image_dir / file_name
                dst_path = val_image_dir / file_name
                if not src_path.exists():
                    raise FileNotFoundError(
                        f"Cannot move missing test image into validation: {src_path}"
                    )
                shutil.move(src_path, dst_path)

    if not args.dry_run:
        for split in output_splits:
            write_coco(
                output_root / "annotations" / f"{split}.json",
                merged_by_split[split],
            )

    print("Merge completed." if not args.dry_run else "Dry-run completed.")
    print(f"Output root: {output_root}")
    for split in output_splits:
        image_count = len(merged_by_split[split]["images"])
        ann_count = len(merged_by_split[split]["annotations"])
        print(f"  {split}: {image_count} images, {ann_count} annotations")
    print("Categories:")
    for category in categories_out:
        print(
            f"  id={category['id']}: {category['name']} (supercategory={category['supercategory']})"
        )


if __name__ == "__main__":
    main()
