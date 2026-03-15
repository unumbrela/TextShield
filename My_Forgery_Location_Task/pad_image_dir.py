"""
将图片 pad 到 28 的倍数，避免 resize 造成的形变和信息损失。

与 resize_image_dir.py 的区别:
  - resize: 拉伸/压缩图片到 28 倍数 → 改变宽高比，插值模糊
  - pad:    右下角填充像素到 28 倍数 → 保持原始像素不变

用法:
  python pad_image_dir.py --input dataset/test --output dataset/test_28
  python pad_image_dir.py --input dataset/train/Black/Image --output dataset/train/Black/Image_28
"""

import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def ceil_to_mult(x, m=28):
    """向上取整到 m 的倍数"""
    return ((x + m - 1) // m) * m


def pad_to_mult_28(in_dir: str, out_dir: str, pad_value: int = 0,
                   save_meta: bool = True):
    """
    将目录中所有图片 pad 到 28 的倍数。

    Args:
        in_dir: 输入图片目录
        out_dir: 输出目录
        pad_value: 填充像素值 (0=黑色, 114=灰色)
        save_meta: 是否保存 padding 元信息 (用于后处理时裁剪回原始尺寸)
    """
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    files = []
    for root, _, names in os.walk(in_dir):
        for n in names:
            if os.path.splitext(n)[1].lower() in IMG_EXT:
                files.append(os.path.join(root, n))

    meta = {}  # filename -> {orig_w, orig_h, pad_w, pad_h}

    for p in tqdm(files, desc="Padding", unit="img"):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        h, w = img.shape[:2]
        nh = ceil_to_mult(h, 28)
        nw = ceil_to_mult(w, 28)

        if (nh, nw) != (h, w):
            pad_bottom = nh - h
            pad_right = nw - w
            img = cv2.copyMakeBorder(
                img, 0, pad_bottom, 0, pad_right,
                cv2.BORDER_CONSTANT, value=pad_value
            )

        rel = os.path.relpath(p, in_dir)
        out_path = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img)

        fname = os.path.basename(p)
        meta[fname] = {
            "orig_w": w, "orig_h": h,
            "pad_w": nw, "pad_h": nh,
        }

    if save_meta:
        meta_path = os.path.join(out_dir, "_pad_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Padding meta saved to: {meta_path}")

    padded = sum(1 for v in meta.values() if v["orig_w"] != v["pad_w"] or v["orig_h"] != v["pad_h"])
    print(f"Done: {len(meta)} images, {padded} padded, {len(meta) - padded} unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image dir")
    parser.add_argument("--output", required=True, help="Output image dir")
    parser.add_argument("--pad_value", type=int, default=0,
                        help="Pad pixel value (0=black, 114=gray)")
    args = parser.parse_args()

    pad_to_mult_28(args.input, args.output, pad_value=args.pad_value)
