#!/usr/bin/env python3
import argparse
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import Sam3Config, Sam3Model, Sam3Processor

COCO_ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_file(url: str, dest: Path, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)


def ensure_coco_annotations(coco_dir: Path) -> Path:
    ann_zip = coco_dir / "annotations_trainval2017.zip"
    ann_json = coco_dir / "annotations" / "instances_val2017.json"
    if ann_json.exists():
        return ann_json
    download_file(COCO_ANN_ZIP_URL, ann_zip)
    coco_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ann_zip, "r") as zf:
        member = "annotations/instances_val2017.json"
        zf.extract(member, path=coco_dir)
    if not ann_json.exists():
        raise FileNotFoundError(f"Expected {ann_json} after extraction")
    return ann_json


def load_coco(ann_json: Path):
    with open(ann_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["categories"], data["images"], data["annotations"]


def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def find_category_id(categories, prompt: str) -> int:
    prompt_norm = normalize_name(prompt)
    exact = [c for c in categories if normalize_name(c["name"]) == prompt_norm]
    if exact:
        return exact[0]["id"]
    partial = [c for c in categories if prompt_norm in normalize_name(c["name"]) ]
    if len(partial) == 1:
        return partial[0]["id"]
    if len(partial) > 1:
        names = ", ".join(c["name"] for c in partial[:10])
        raise ValueError(
            f"Ambiguous prompt '{prompt}'. Matches: {names}. Use an exact COCO category name."
        )
    raise ValueError(
        f"Category '{prompt}' not found. Use an exact COCO category name (e.g., person, car, dog)."
    )


def build_index(images, annotations, category_id: int):
    images_by_id = {img["id"]: img for img in images}
    anns_by_image = {}
    for ann in annotations:
        if ann.get("category_id") != category_id:
            continue
        if ann.get("iscrowd", 0) == 1:
            continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    image_ids = list(anns_by_image.keys())
    return images_by_id, anns_by_image, image_ids


def ann_to_mask(ann, height: int, width: int) -> np.ndarray:
    seg = ann.get("segmentation")
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(seg, dict) and "counts" in seg:
        rle = seg
    else:
        raise ValueError("Unknown segmentation format")
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(bool)


def build_gt_mask(annotations, height: int, width: int) -> np.ndarray:
    gt = np.zeros((height, width), dtype=bool)
    for ann in annotations:
        gt |= ann_to_mask(ann, height, width)
    return gt


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(pred, gt).sum()
    return float(inter / union)


def load_model_and_processor(model_id: str, image_size: int, device: torch.device):
    if image_size:
        config = Sam3Config.from_pretrained(model_id)
        config.image_size = image_size
        model = Sam3Model.from_pretrained(model_id, config=config)
        processor = Sam3Processor.from_pretrained(
            model_id, size={"height": image_size, "width": image_size}
        )
    else:
        model = Sam3Model.from_pretrained(model_id)
        processor = Sam3Processor.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, processor


def run_inference(model, processor, image: Image.Image, text_prompt: str,
                  score_threshold: float, mask_threshold: float, device: torch.device):
    inputs = processor(images=image, text=[text_prompt], return_tensors="pt")
    inputs = inputs.to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        target_sizes = inputs.get("original_sizes")
        if torch.is_tensor(target_sizes):
            target_sizes = target_sizes.cpu().tolist()
        results = processor.post_process_instance_segmentation(
            outputs=outputs,
            threshold=score_threshold,
            mask_threshold=mask_threshold,
            target_sizes=target_sizes,
        )
    return results[0]


def extract_pred_masks(result) -> np.ndarray:
    masks = result.get("masks") if isinstance(result, dict) else None
    if masks is None:
        return np.zeros((0, 1, 1), dtype=bool)
    if torch.is_tensor(masks):
        masks = masks.cpu().numpy()
    else:
        masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    return masks.astype(bool)


def save_mask(mask: np.ndarray, path: Path) -> None:
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    img.save(path)


def save_overlay(image: Image.Image, gt: np.ndarray, pred: np.ndarray, path: Path) -> None:
    base = np.asarray(image).astype(np.float32)
    overlay = base.copy()
    alpha = 0.5
    gt_color = np.array([255, 0, 0], dtype=np.float32)
    pred_color = np.array([0, 255, 0], dtype=np.float32)
    if gt.any():
        overlay[gt] = overlay[gt] * (1 - alpha) + gt_color * alpha
    if pred.any():
        overlay[pred] = overlay[pred] * (1 - alpha) + pred_color * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(path)


def iter_candidate_image_ids(image_ids, seed: int):
    base = list(image_ids)
    cycle = 0
    while True:
        cycle += 1
        shuffled = list(base)
        random.Random(seed + cycle - 1).shuffle(shuffled)
        for image_id in shuffled:
            yield cycle, image_id


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM3 CPU segmentation + IoU on COCO val")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--text-prompt", default="person", help="COCO category name")
    parser.add_argument("--image-size", type=int, default=560, help="SAM3 image size")
    parser.add_argument("--score-threshold", type=float, default=0.2, help="Score threshold")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask threshold")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Total retry attempts. 0 means unlimited.",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=1e-9,
        help="Minimum IoU to accept. Default is strictly > 0.",
    )
    parser.add_argument("--require-pred", action="store_true", help="Require at least one predicted mask")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-id", default="facebook/sam3", help="HF model id")
    args = parser.parse_args()
    if args.max_attempts < 0:
        print("Error: --max-attempts must be >= 0")
        return 1

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    coco_dir = data_dir / "coco"
    images_dir = coco_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Preparing COCO annotations...")
    ann_json = ensure_coco_annotations(coco_dir)
    categories, images, annotations = load_coco(ann_json)

    try:
        category_id = find_category_id(categories, args.text_prompt)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    images_by_id, anns_by_image, image_ids = build_index(images, annotations, category_id)
    if not image_ids:
        print("No images found for that category.")
        return 1

    print(f"[2/5] Loading SAM3 model on CPU (model: {args.model_id})...")
    device = torch.device("cpu")
    model, processor = load_model_and_processor(args.model_id, args.image_size, device)

    attempt_limit = args.max_attempts if args.max_attempts > 0 else None
    candidate_iter = iter_candidate_image_ids(image_ids, args.seed)

    if attempt_limit is None:
        print(f"[3/5] Running inference with retries... (unlimited, min_iou={args.min_iou})")
    else:
        print(f"[3/5] Running inference with retries... (max_attempts={attempt_limit}, min_iou={args.min_iou})")

    attempt = 0
    current_cycle = 0
    try:
        while attempt_limit is None or attempt < attempt_limit:
            attempt += 1
            cycle, image_id = next(candidate_iter)
            if cycle != current_cycle:
                current_cycle = cycle
                print(f"[cycle {cycle}] trying up to {len(image_ids)} images in shuffled order")
            try:
                info = images_by_id[image_id]
                file_name = info["file_name"]
                coco_url = info.get("coco_url")
                image_path = images_dir / file_name
                if coco_url is None:
                    print(f"[attempt {attempt}] No coco_url for image {image_id}, skipping.")
                    continue
                if not image_path.exists():
                    print(f"[attempt {attempt}] Downloading image {file_name}...")
                    download_file(coco_url, image_path)
                image = Image.open(image_path).convert("RGB")
                height, width = info["height"], info["width"]
                gt_mask = build_gt_mask(anns_by_image[image_id], height, width)
                if gt_mask.sum() == 0:
                    print(f"[attempt {attempt}] Empty GT mask, retrying.")
                    continue

                result = run_inference(
                    model,
                    processor,
                    image,
                    args.text_prompt,
                    args.score_threshold,
                    args.mask_threshold,
                    device,
                )
                pred_masks = extract_pred_masks(result)
                if pred_masks.shape[0] == 0:
                    if args.require_pred:
                        print(f"[attempt {attempt}] No predicted masks, retrying.")
                        continue
                    best_iou = 0.0
                    best_idx = None
                    best_mask = np.zeros_like(gt_mask)
                else:
                    best_iou = -1.0
                    best_idx = None
                    best_mask = None
                    for idx, mask in enumerate(pred_masks):
                        iou = compute_iou(mask, gt_mask)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx
                            best_mask = mask
                    if best_mask is None:
                        print(f"[attempt {attempt}] Failed to select mask, retrying.")
                        continue

                if best_iou < args.min_iou:
                    print(f"[attempt {attempt}] IoU {best_iou:.4f} < min_iou, retrying.")
                    continue

                print("[4/5] Saving outputs...")
                save_mask(gt_mask, out_dir / "gt_mask.png")
                save_mask(best_mask, out_dir / "pred_mask_best.png")
                image.save(out_dir / "image.jpg")
                save_overlay(image, gt_mask, best_mask, out_dir / "overlay.png")

                print("[5/5] Done")
                print(f"Image: {file_name} (id={image_id})")
                print(f"Best mask idx: {best_idx}")
                print(f"IoU: {best_iou:.6f}")
                return 0
            except Exception as e:
                print(f"[attempt {attempt}] Error: {e}. Retrying...")
                time.sleep(1)
                continue
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130

    print(f"Failed to compute IoU >= {args.min_iou} within max attempts.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
