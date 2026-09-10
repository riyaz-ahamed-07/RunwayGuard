from __future__ import annotations

import argparse
import csv
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


class DisjointSet:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass(frozen=True)
class ImageMetadata:
    labels: tuple[str, ...]
    weather: str
    light: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leakage-resistant FOD-A splits.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data_splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--adjacent-hamming-threshold",
        type=int,
        default=18,
        help="Join consecutive, metadata-matching frames at or below this 64-bit dHash distance.",
    )
    return parser.parse_args()


def difference_hash(path: Path, size: int = 8) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((size + 1, size)).getdata())
    bits: list[bool] = []
    for row in range(size):
        offset = row * (size + 1)
        bits.extend(pixels[offset + col] > pixels[offset + col + 1] for col in range(size))
    return sum(int(bit) << index for index, bit in enumerate(bits))


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def read_environment_metadata(dataset_root: Path) -> dict[str, tuple[str, str]]:
    csv_path = (
        dataset_root
        / "ImageSets"
        / "Main"
        / "CategorizationData"
        / "FOD_categorization_annotations.csv"
    )
    metadata: dict[str, tuple[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            metadata[Path(row["File"]).stem] = (row["Weather"], row["Light"])
    return metadata


def read_annotation_metadata(dataset_root: Path) -> dict[str, tuple[str, ...]]:
    labels_by_id: dict[str, tuple[str, ...]] = {}
    for path in sorted((dataset_root / "Annotations").glob("*.xml")):
        root = ET.parse(path).getroot()
        labels = tuple(sorted((obj.findtext("name") or "").strip() for obj in root.findall("object")))
        if not labels or any(not label for label in labels):
            raise ValueError(f"Missing object label in {path}")
        labels_by_id[path.stem] = labels
    return labels_by_id


def component_class_counts(component: list[str], metadata: dict[str, ImageMetadata]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for image_id in component:
        counts.update(metadata[image_id].labels)
    return counts


def choose_split(
    image_count: int,
    class_counts: Counter[str],
    assigned_images: Counter[str],
    assigned_classes: dict[str, Counter[str]],
    target_images: dict[str, float],
    target_classes: dict[str, Counter[str]],
    total_images: int,
    total_classes: Counter[str],
) -> str:
    best_split = "train"
    best_score = float("-inf")
    for split_name in SPLIT_RATIOS:
        image_deficit = (target_images[split_name] - assigned_images[split_name]) / max(total_images, 1)
        class_deficits = []
        for class_name, count in class_counts.items():
            target = target_classes[split_name][class_name]
            deficit = (target - assigned_classes[split_name][class_name]) / max(
                total_classes[class_name], 1
            )
            class_deficits.extend([deficit] * count)
        class_score = sum(class_deficits) / max(len(class_deficits), 1)
        score = 0.65 * class_score + 0.35 * image_deficit
        if score > best_score:
            best_split = split_name
            best_score = score
    return best_split


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_by_id = read_annotation_metadata(dataset_root)
    environment_by_id = read_environment_metadata(dataset_root)
    image_by_id = {
        path.stem: path
        for path in (dataset_root / "JPEGImages").iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    image_ids = sorted(labels_by_id, key=int)
    if set(image_ids) != set(image_by_id):
        raise ValueError("Image and annotation IDs do not match.")

    metadata = {
        image_id: ImageMetadata(
            labels=labels_by_id[image_id],
            weather=environment_by_id[image_id][0],
            light=environment_by_id[image_id][1],
        )
        for image_id in image_ids
    }

    hashes: dict[str, int] = {}
    hash_groups: dict[int, list[str]] = defaultdict(list)
    for index, image_id in enumerate(image_ids, start=1):
        image_hash = difference_hash(image_by_id[image_id])
        hashes[image_id] = image_hash
        hash_groups[image_hash].append(image_id)
        if index % 2500 == 0:
            print(f"Hashed {index:,}/{len(image_ids):,} images", flush=True)

    groups = DisjointSet(image_ids)

    # dHash largely represents the runway background and can ignore a tiny FOD
    # object. Only join identical hashes when the label and capture conditions
    # also agree; otherwise different debris on the same pavement could be
    # incorrectly fused into a giant component.
    for identical_ids in hash_groups.values():
        by_metadata: dict[ImageMetadata, list[str]] = defaultdict(list)
        for image_id in identical_ids:
            by_metadata[metadata[image_id]].append(image_id)
        for metadata_matched_ids in by_metadata.values():
            anchor = metadata_matched_ids[0]
            for image_id in metadata_matched_ids[1:]:
                groups.union(anchor, image_id)

    # Consecutive frames with matching labels/environment and similar appearance are
    # treated as one capture sequence rather than independent examples.
    adjacent_links = 0
    for previous_id, current_id in zip(image_ids, image_ids[1:]):
        if int(current_id) != int(previous_id) + 1:
            continue
        if metadata[current_id] != metadata[previous_id]:
            continue
        distance = hamming_distance(hashes[current_id], hashes[previous_id])
        if distance <= args.adjacent_hamming_threshold:
            groups.union(previous_id, current_id)
            adjacent_links += 1

    components_by_root: dict[str, list[str]] = defaultdict(list)
    for image_id in image_ids:
        components_by_root[groups.find(image_id)].append(image_id)
    components = list(components_by_root.values())

    total_classes: Counter[str] = Counter()
    for item in metadata.values():
        total_classes.update(item.labels)
    target_images = {name: len(image_ids) * ratio for name, ratio in SPLIT_RATIOS.items()}
    target_classes = {
        name: Counter({class_name: count * ratio for class_name, count in total_classes.items()})
        for name, ratio in SPLIT_RATIOS.items()
    }

    rng = random.Random(args.seed)
    rng.shuffle(components)
    # Place large and rare-class components first; this makes the greedy balancing stable.
    components.sort(
        key=lambda component: (
            -sum(1 / total_classes[label] for image_id in component for label in metadata[image_id].labels),
            -len(component),
        )
    )

    assignments: dict[str, list[str]] = {name: [] for name in SPLIT_RATIOS}
    assigned_images: Counter[str] = Counter()
    assigned_classes: dict[str, Counter[str]] = {name: Counter() for name in SPLIT_RATIOS}

    for component in components:
        counts = component_class_counts(component, metadata)
        split_name = choose_split(
            len(component),
            counts,
            assigned_images,
            assigned_classes,
            target_images,
            target_classes,
            len(image_ids),
            total_classes,
        )
        assignments[split_name].extend(component)
        assigned_images[split_name] += len(component)
        assigned_classes[split_name].update(counts)

    for split_name, ids in assignments.items():
        ids.sort(key=int)
        (output_dir / f"{split_name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    split_by_id = {
        image_id: split_name for split_name, ids in assignments.items() for image_id in ids
    }
    crossing_hash_groups = []
    for ids in hash_groups.values():
        by_metadata: dict[ImageMetadata, list[str]] = defaultdict(list)
        for image_id in ids:
            by_metadata[metadata[image_id]].append(image_id)
        crossing_hash_groups.extend(
            matched_ids
            for matched_ids in by_metadata.values()
            if len({split_by_id[image_id] for image_id in matched_ids}) > 1
        )
    component_sizes = Counter(len(component) for component in components)
    report = {
        "seed": args.seed,
        "ratios": SPLIT_RATIOS,
        "adjacent_hamming_threshold": args.adjacent_hamming_threshold,
        "images": len(image_ids),
        "classes": len(total_classes),
        "components": len(components),
        "largest_component": max(map(len, components)),
        "component_size_distribution": dict(sorted(component_sizes.items())),
        "adjacent_links": adjacent_links,
        "split_image_counts": dict(assigned_images),
        "split_class_counts": {
            name: dict(sorted(counts.items())) for name, counts in assigned_classes.items()
        },
        "identical_dhash_groups_crossing_new_splits": len(crossing_hash_groups),
    }
    (output_dir / "split_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== GROUPED SPLIT COMPLETE ===")
    print(f"Components: {report['components']:,}")
    print(f"Largest component: {report['largest_component']:,} images")
    print(f"Adjacent links: {report['adjacent_links']:,}")
    print(f"Split sizes: {report['split_image_counts']}")
    print(f"Identical dHash groups crossing new splits: {report['identical_dhash_groups_crossing_new_splits']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
