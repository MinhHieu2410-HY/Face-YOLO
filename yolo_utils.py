from pathlib import Path
import shutil
import zipfile

import cv2
import yaml
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def predict_frame(model, frame_bgr, imgsz, conf, iou, device, half):
    results = model.predict(
        frame_bgr, imgsz=imgsz, conf=conf, iou=iou,
        device=device, half=half, verbose=False,
    )
    r = results[0]
    return r.plot(), r


def class_counts(r, model) -> dict:
    if r.boxes is None or len(r.boxes) == 0:
        return {}
    names = model.names
    counts = {}
    for c in r.boxes.cls.tolist():
        name = names[int(c)]
        counts[name] = counts.get(name, 0) + 1
    return counts


def process_video(model, in_path, out_path, imgsz, conf, iou, device, half,
                   progress_cb=None, preview_cb=None, preview_every=5):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, _ = predict_frame(model, frame, imgsz, conf, iou, device, half)
        writer.write(annotated)
        idx += 1
        if progress_cb and total > 0:
            progress_cb(min(idx / total, 1.0))
        if preview_cb and idx % preview_every == 0:
            preview_cb(annotated)

    cap.release()
    writer.release()
    return out_path



def load_data_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataset_root(yaml_path: str, data_cfg: dict) -> Path:
    base = Path(yaml_path).parent
    if "path" in data_cfg:
        p = Path(data_cfg["path"])
        return p if p.is_absolute() else (base / p)
    return base


def split_images_dir(root: Path, data_cfg: dict, split: str) -> Path:
    rel = data_cfg.get(split, f"{split}/images")
    p = Path(rel)
    return p if p.is_absolute() else (root / p)


def images_dir_to_labels_dir(img_dir: Path) -> Path:
    parts = ["labels" if p == "images" else p for p in img_dir.parts]
    return Path(*parts)


def count_images(images_dir: Path) -> int:
    if not images_dir.exists():
        return 0
    return sum(1 for f in images_dir.iterdir() if f.suffix.lower() in IMG_EXTS)


def dataset_summary(yaml_path: str) -> dict:
    data_cfg = load_data_yaml(yaml_path)
    root = dataset_root(yaml_path, data_cfg)
    summary = {"nc": data_cfg.get("nc"), "names": data_cfg.get("names"), "root": str(root)}
    for split in ("train", "val", "test"):
        if split in data_cfg:
            img_dir = split_images_dir(root, data_cfg, split)
            summary[f"{split}_images"] = count_images(img_dir)
    return summary


def extract_zip(zip_path: str, extract_to: Path) -> Path:
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    return extract_to


def collect_pairs_from_dir(root: Path):
    image_files, label_files = {}, {}
    for f in root.rglob("*"):
        if f.is_file():
            if f.suffix.lower() in IMG_EXTS:
                image_files[f.stem] = f
            elif f.suffix.lower() == ".txt" and f.stem.lower() != "classes":
                label_files[f.stem] = f
    return image_files, label_files


def validate_label_file(label_path: Path, nc: int) -> list:
    errors = []
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path.name} dòng {i + 1}: cần 5 giá trị, có {len(parts)}")
            continue

        cls_id = parts[0]
        if not cls_id.isdigit() or not (0 <= int(cls_id) < nc):
            errors.append(f"{label_path.name} dòng {i + 1}: class id '{cls_id}' không hợp lệ (nc={nc})")

        for v in parts[1:]:
            try:
                fv = float(v)
                if not (0.0 <= fv <= 1.0):
                    errors.append(f"{label_path.name} dòng {i + 1}: tọa độ {v} phải trong [0,1]")
            except ValueError:
                errors.append(f"{label_path.name} dòng {i + 1}: giá trị không hợp lệ '{v}'")

    return errors


def validate_pairs(image_files: dict, label_files: dict, nc: int):
    valid_pairs, errors = [], []
    for stem, img_path in image_files.items():
        label_path = label_files.get(stem)
        if label_path is None:
            errors.append(f"{img_path.name}: thiếu file label tương ứng")
            continue

        label_errors = validate_label_file(label_path, nc)
        if label_errors:
            errors.extend(label_errors)
            continue

        valid_pairs.append((img_path, label_path))
    return valid_pairs, errors


def merge_pairs(valid_pairs, root: Path, data_cfg: dict, split: str) -> int:
    img_dir = split_images_dir(root, data_cfg, split)
    label_dir = images_dir_to_labels_dir(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    for img_path, label_path in valid_pairs:
        shutil.copy2(img_path, img_dir / img_path.name)
        shutil.copy2(label_path, label_dir / label_path.name)
        added += 1
    return added
