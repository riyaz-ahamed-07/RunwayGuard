from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert FOD-A Pascal VOC annotations to YOLO format.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        help="Optional JSON mapping from final class names to lists of source Pascal VOC labels.",
    )
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating hard links.")
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def discover_classes(annotation_dir: Path) -> list[str]:
    names: set[str] = set()
    for path in annotation_dir.glob("*.xml"):
        root = ET.parse(path).getroot()
        names.update((obj.findtext("name") or "").strip() for obj in root.findall("object"))
    names.discard("")
    return sorted(names)


def link_or_copy(source: Path, destination: Path, copy_images: bool) -> None:
    if destination.exists():
        return
    if copy_images:
        shutil.copy2(source, destination)
        return
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def convert_annotation(
    xml_path: Path,
    source_to_final: dict[str, str],
    class_to_id: dict[str, int],
) -> tuple[list[str], Counter[str]]:
    root = ET.parse(xml_path).getroot()
    width = float(root.findtext("size/width", "0"))
    height = float(root.findtext("size/height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions in {xml_path}")

    lines: list[str] = []
    counts: Counter[str] = Counter()
    for obj in root.findall("object"):
        source_name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if source_name not in source_to_final or box is None:
            raise ValueError(f"Invalid object in {xml_path}")
        name = source_to_final[source_name]
        xmin = max(0.0, min(width, float(box.findtext("xmin", "0"))))
        ymin = max(0.0, min(height, float(box.findtext("ymin", "0"))))
        xmax = max(0.0, min(width, float(box.findtext("xmax", "0"))))
        ymax = max(0.0, min(height, float(box.findtext("ymax", "0"))))
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Invalid bounding box in {xml_path}: {(xmin, ymin, xmax, ymax)}")
        center_x = ((xmin + xmax) / 2) / width
        center_y = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        lines.append(
            f"{class_to_id[name]} {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}"
        )
        counts[name] += 1
    return lines, counts


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    splits_dir = args.splits_dir.resolve()
    output_dir = args.output_dir.resolve()
    annotation_dir = dataset_root / "Annotations"
    image_dir = dataset_root / "JPEGImages"

    if args.taxonomy:
        taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
        classes = list(taxonomy)
        source_to_final = {
            source_name: final_name
            for final_name, source_names in taxonomy.items()
            for source_name in source_names
        }
        source_classes = sorted(source_to_final)
        duplicated = sum(len(names) for names in taxonomy.values()) != len(source_to_final)
        if duplicated:
            raise ValueError(
                "Taxonomy must map every source label exactly once; a source label appears more than once."
            )
    else:
        source_classes = discover_classes(annotation_dir)
        classes = source_classes
        source_to_final = {name: name for name in source_classes}
    class_to_id = {name: index for index, name in enumerate(classes)}
    split_counts: dict[str, Counter[str]] = {}

    for split_name in ("train", "val", "test"):
        ids = read_ids(splits_dir / f"{split_name}.txt")
        output_images = output_dir / "images" / split_name
        output_labels = output_dir / "labels" / split_name
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        counts: Counter[str] = Counter()

        for index, image_id in enumerate(ids, start=1):
            source_image = image_dir / f"{image_id}.jpg"
            source_xml = annotation_dir / f"{image_id}.xml"
            if not source_image.exists() or not source_xml.exists():
                raise FileNotFoundError(f"Missing image or annotation for ID {image_id}")
            link_or_copy(source_image, output_images / source_image.name, args.copy_images)
            lines, annotation_counts = convert_annotation(source_xml, source_to_final, class_to_id)
            (output_labels / f"{image_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts.update(annotation_counts)
            if index % 5000 == 0:
                print(f"Converted {index:,}/{len(ids):,} {split_name} images", flush=True)
        split_counts[split_name] = counts

    names_yaml = "\n".join(f"  {index}: {json.dumps(name)}" for index, name in enumerate(classes))
    data_yaml = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names_yaml}\n"
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    report = {
        "source": str(dataset_root),
        "source_classes": source_classes,
        "classes": classes,
        "source_to_final": source_to_final,
        "class_to_id": class_to_id,
        "split_class_counts": {name: dict(counts) for name, counts in split_counts.items()},
    }
    (output_dir / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Converted {sum(sum(c.values()) for c in split_counts.values()):,} annotations.")
    print(f"YOLO dataset: {output_dir}")
    print(f"Configuration: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
