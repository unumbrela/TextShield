"""
对小票/收据类图片进行数据增强。

只做不改变空间布局的增强 (颜色空间 + 噪声), 这样:
  - Caption 中的 bbox 坐标不需要修改
  - Mask 可以直接复制
  - 增加了训练样本的多样性, 避免简单复制导致的过拟合

用法:
  # Black (伪造) 小票增强, 每张生成 4 个变体
  python augment_receipts.py \
    --receipt_dir dataset/train/Black/receipt_of_train_black \
    --image_dir dataset/train/Black/Image \
    --caption_dir dataset/train/Black/Caption \
    --mask_dir dataset/train/Black/Mask \
    --num_aug 4

  # White (真实) 小票增强
  python augment_receipts.py \
    --receipt_dir dataset/train/White/receipt_of_train_white \
    --image_dir dataset/train/White/Image \
    --caption_dir dataset/train/White/Caption \
    --num_aug 4
"""

import argparse
import os
import random
import shutil

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def random_brightness(img, low=0.75, high=1.25):
    """随机亮度调整"""
    factor = random.uniform(low, high)
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def random_contrast(img, low=0.75, high=1.25):
    """随机对比度调整"""
    factor = random.uniform(low, high)
    mean = img.mean()
    return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def random_gaussian_noise(img, sigma_low=3, sigma_high=15):
    """添加随机高斯噪声"""
    sigma = random.uniform(sigma_low, sigma_high)
    noise = np.random.randn(*img.shape) * sigma
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_jpeg_compress(img, quality_low=50, quality_high=85):
    """JPEG 重压缩 (模拟不同传输/保存质量)"""
    quality = random.randint(quality_low, quality_high)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def random_gaussian_blur(img, max_ksize=3):
    """轻微高斯模糊"""
    sigma = random.uniform(0.3, 1.0)
    return cv2.GaussianBlur(img, (max_ksize, max_ksize), sigma)


def random_hsv_jitter(img, h_range=8, s_range=30, v_range=30):
    """HSV 色彩空间抖动"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-h_range, h_range)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.uniform(-s_range, s_range), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.uniform(-v_range, v_range), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# 所有可用的增强操作 (名称, 函数, 被选中的概率)
AUGMENTATIONS = [
    ("brightness",    random_brightness,      0.6),
    ("contrast",      random_contrast,        0.6),
    ("gaussian_noise", random_gaussian_noise,  0.5),
    ("jpeg_compress", random_jpeg_compress,    0.5),
    ("gaussian_blur", random_gaussian_blur,    0.3),
    ("hsv_jitter",    random_hsv_jitter,       0.4),
]


def augment_image(img, min_ops=2):
    """对图片随机应用多种增强, 至少应用 min_ops 种"""
    ops = [(name, fn) for name, fn, prob in AUGMENTATIONS if random.random() < prob]
    # 保证至少应用 min_ops 种
    if len(ops) < min_ops:
        pool = [(name, fn) for name, fn, _ in AUGMENTATIONS if (name, fn) not in ops]
        random.shuffle(pool)
        ops.extend(pool[:min_ops - len(ops)])
    random.shuffle(ops)
    result = img.copy()
    for name, fn in ops:
        result = fn(result)
    return result


def main():
    parser = argparse.ArgumentParser(description="小票/收据数据增强")
    parser.add_argument("--receipt_dir", required=True,
                        help="已挑出的小票原图目录 (只用于确定文件名列表)")
    parser.add_argument("--image_dir", required=True,
                        help="训练图片目录 (增强后的图片也会写入这里)")
    parser.add_argument("--caption_dir", required=True,
                        help="Caption 目录 (增强样本复制对应 caption)")
    parser.add_argument("--mask_dir", default=None,
                        help="Mask 目录 (仅 Black 需要, 增强样本复制对应 mask)")
    parser.add_argument("--num_aug", type=int, default=4,
                        help="每张图片生成的增强变体数量 (默认 4)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    receipt_dir = os.path.abspath(args.receipt_dir)
    image_dir = os.path.abspath(args.image_dir)
    caption_dir = os.path.abspath(args.caption_dir)
    mask_dir = os.path.abspath(args.mask_dir) if args.mask_dir else None

    # 获取小票文件名列表 (stem)
    receipt_stems = set()
    for f in os.listdir(receipt_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMG_EXT:
            receipt_stems.add(os.path.splitext(f)[0])

    print(f"Found {len(receipt_stems)} receipt images")
    print(f"Generating {args.num_aug} augmented variants per image")
    print(f"Total new samples: {len(receipt_stems) * args.num_aug}")

    n_created = 0
    n_skipped = 0

    for stem in tqdm(sorted(receipt_stems), desc="Augmenting"):
        # 找原图
        orig_path = None
        orig_ext = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = os.path.join(image_dir, stem + ext)
            if os.path.exists(candidate):
                orig_path = candidate
                orig_ext = ext
                break

        if orig_path is None:
            print(f"  Warning: no image for {stem} in {image_dir}, skipping")
            n_skipped += 1
            continue

        img = cv2.imread(orig_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  Warning: cannot read {orig_path}, skipping")
            n_skipped += 1
            continue

        # 找 caption
        caption_path = None
        for ext in [".md", ".txt"]:
            candidate = os.path.join(caption_dir, stem + ext)
            if os.path.exists(candidate):
                caption_path = candidate
                break

        if caption_path is None:
            print(f"  Warning: no caption for {stem}, skipping")
            n_skipped += 1
            continue

        caption_ext = os.path.splitext(caption_path)[1]

        # 找 mask (仅 Black)
        mask_path = None
        if mask_dir:
            for ext in [".png", ".jpg"]:
                candidate = os.path.join(mask_dir, stem + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

        # 生成增强变体
        for i in range(args.num_aug):
            aug_stem = f"{stem}_aug{i+1}"

            # 增强图片
            aug_img = augment_image(img)
            aug_img_path = os.path.join(image_dir, aug_stem + orig_ext)
            cv2.imwrite(aug_img_path, aug_img)

            # 复制 caption (内容不变, bbox 不变)
            aug_caption_path = os.path.join(caption_dir, aug_stem + caption_ext)
            shutil.copy2(caption_path, aug_caption_path)

            # 复制 mask (空间不变)
            if mask_path:
                mask_ext = os.path.splitext(mask_path)[1]
                aug_mask_path = os.path.join(mask_dir, aug_stem + mask_ext)
                shutil.copy2(mask_path, aug_mask_path)

            n_created += 1

    print(f"\nDone:")
    print(f"  Created: {n_created} augmented samples")
    print(f"  Skipped: {n_skipped} (missing image/caption)")
    print(f"  New total in {os.path.basename(image_dir)}: "
          f"{len([f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in IMG_EXT])}")


if __name__ == "__main__":
    main()
