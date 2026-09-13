from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = (
    "#00E5FF",
    "#FFEA00",
    "#FF4081",
    "#69F0AE",
    "#FF9100",
    "#B388FF",
    "#FFFFFF",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a stratified visual audit from prepared YOLO labels."
    )
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--probe-limit",
        type=int,
        default=1500,
        help="Max label files to read while building class coverage.",
    )
    return parser.parse_args()


def load_names(data_yaml: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.isdigit():
            names[int(key)] = value.strip().strip('"')
    return names


def yolo_to_xyxy(parts: list[str], width: int, height: int) -> tuple[float, float, float, float]:
    _, cx, cy, bw, bh = map(float, parts)
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    return x1, y1, x2, y2


def class_ids_in_label(label_path: Path) -> set[int]:
    return {
        int(line.split()[0])
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> None:
    args = parse_args()
    names = load_names(args.prepared_data / "data.yaml")
    label_dir = args.prepared_data / "labels" / args.split
    image_dir = args.prepared_data / "images" / args.split
    cache_dir = args.output_dir / ".image_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    split_file = args.splits_dir / f"{args.split}.txt"
    stem_list = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not stem_list:
        raise SystemExit(f"No image ids found in {split_file}")

    rng = random.Random(args.seed)
    probe_stems = stem_list if len(stem_list) <= args.probe_limit else rng.sample(stem_list, args.probe_limit)
    by_class: dict[int, list[str]] = defaultdict(list)
    for stem in probe_stems:
        label_path = label_dir / f"{stem}.txt"
        if not label_path.exists():
            continue
        for class_id in class_ids_in_label(label_path):
            by_class[class_id].append(stem)

    selected: list[str] = []
    for class_id in sorted(names):
        pool = by_class.get(class_id, [])
        if pool:
            selected.append(rng.choice(pool))
    selected = list(dict.fromkeys(selected))
    remaining = args.samples - len(selected)
    if remaining > 0:
        pool = [stem for stem in probe_stems if stem not in set(selected)]
        selected.extend(rng.sample(pool, min(remaining, len(pool))))
    rng.shuffle(selected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()

    color_for = {class_id: COLORS[class_id % len(COLORS)] for class_id in names}
    tiles_per_page = 12
    tile_size = 320
    for page_index in range(0, len(selected), tiles_per_page):
        page_stems = selected[page_index : page_index + tiles_per_page]
        cols = 4
        rows = (len(page_stems) + cols - 1) // cols
        sheet = Image.new("RGB", (tile_size * cols, tile_size * rows), "#111111")
        for tile_index, stem in enumerate(page_stems):
            source_image = image_dir / f"{stem}.jpg"
            if not source_image.exists():
                source_image = image_dir / f"{stem}.png"
            cached = cache_dir / source_image.name
            if not cached.exists():
                shutil.copy2(source_image, cached)
            image = Image.open(cached).convert("RGB")
            draw = ImageDraw.Draw(image)
            width, height = image.size
            label_path = label_dir / f"{stem}.txt"
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                xyxy = yolo_to_xyxy(parts, width, height)
                color = color_for[class_id]
                draw.rectangle(xyxy, outline=color, width=2)
                label = names.get(class_id, str(class_id))
                text_xy = (xyxy[0], max(0, xyxy[1] - 14))
                text_box = draw.textbbox(text_xy, label, font=font)
                draw.rectangle(text_box, fill="#000000")
                draw.text(text_xy, label, fill=color, font=font)
            tile = Image.new("RGB", (tile_size, tile_size), "#222222")
            tile.paste(image.resize((300, 300)), (10, 10))
            ImageDraw.Draw(tile).text((10, 303), stem, fill="white", font=font)
            col = tile_index % cols
            row = tile_index // cols
            sheet.paste(tile, (col * tile_size, row * tile_size))
            image.save(args.output_dir / f"{stem}_boxed.jpg", quality=92)
        output_path = args.output_dir / f"annotation_audit_{page_index // tiles_per_page + 1:02}.jpg"
        sheet.save(output_path, quality=92)
        print(output_path)


if __name__ == "__main__":
    main()
