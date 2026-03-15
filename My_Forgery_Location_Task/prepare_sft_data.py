"""
将赛题训练数据转换为 ms-swift SFT 训练格式的 jsonl。

用法:
  python prepare_sft_data.py \
    --train_dir dataset/train \
    --image_dir_28 dataset/train/Black/Image_28 \
    --output sft_train.jsonl

注意: 小票数据增强请先运行 augment_receipts.py 生成增强样本,
      增强后的图片会自动被本脚本读取 (通过 Caption 目录发现)。

输出 jsonl 格式:
  {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "<image> ...prompt..."},
      {"role": "assistant", "content": "<think>...</think>\n<answer>...</answer>"}
    ],
    "images": ["/abs/path/to/image_28.jpg"]
  }
"""

import argparse
import json
import math
import os
import re

import cv2
import imagesize
import numpy as np
from tqdm import tqdm

PROMPT = (
    '<image> Is this image real, entirely generated, or tampered? '
    'If it has been tampered, what method was used, and what are the content '
    'and bounding box coordinates of the tampered text? Output the thinking '
    'process in <think> </think> and \n final answer (number) in <answer> '
    '</answer> tags. \n Here is an example answer for a real image: '
    '<answer> This image is real. </answer> Here is an example answer for '
    'an entirely generated image: <answer> This image is entirely generated. '
    '</answer> Here is an example answer for a locally tampered image: '
    '<answer> This image is tampered. It was tampered by copy-paste. '
    'The tampered text reads "small" in the text line "a small yellow flower", '
    'and it is located at ... </answer>'
)

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
BBOX_PAT = re.compile(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]')
# Chinese and English quotes
QUOTE_PAT = re.compile(r'[\u201c"](.*?)[\u201d"]')

# Qwen2.5-VL processor 默认参数
QWEN_MAX_PIXELS = 1_003_520   # 1280 * 28 * 28
QWEN_MIN_PIXELS = 3_136       # 28 * 28 * 4


def smart_resize(height, width, factor=28,
                 min_pixels=QWEN_MIN_PIXELS, max_pixels=QWEN_MAX_PIXELS):
    """复制 Qwen2.5-VL processor 的 smart_resize 逻辑,
    返回 processor 实际将图片缩放到的 (new_h, new_w)。"""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return max(factor, h_bar), max(factor, w_bar)


def scale_bbox_for_processor(bbox_str, orig_w, orig_h, img28_w, img28_h):
    """将原始坐标空间的 bbox 字符串缩放到 processor 实际看到的坐标空间。

    训练时, processor 会用 smart_resize 将 img28 缩放到 (model_h, model_w),
    因此训练标签中的坐标也应该在 (model_h, model_w) 空间, 这样模型才能
    学到准确的坐标映射。

    Args:
        bbox_str: "[x1, y1, x2, y2]" 格式的字符串, 坐标在 orig 空间
        orig_w, orig_h: 原始图片尺寸
        img28_w, img28_h: pad/resize 到 28 倍数后的图片尺寸
    Returns:
        缩放后的 "[x1, y1, x2, y2]" 字符串, 坐标在 model 空间
    """
    model_h, model_w = smart_resize(img28_h, img28_w)

    # 如果 processor 不会缩放, 直接返回原始坐标
    if model_h == img28_h and model_w == img28_w:
        return bbox_str

    # 坐标从 orig 空间 → model 空间
    # orig → img28 的映射: pad 模式下坐标不变, resize 模式下需要缩放
    # img28 → model 的映射: 按 processor 缩放比
    scale_x = model_w / orig_w
    scale_y = model_h / orig_h

    def _scale_match(m):
        x1, y1, x2, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        nx1 = int(round(x1 * scale_x))
        ny1 = int(round(y1 * scale_y))
        nx2 = int(round(x2 * scale_x))
        ny2 = int(round(y2 * scale_y))
        return f'[{nx1}, {ny1}, {nx2}, {ny2}]'

    return BBOX_PAT.sub(_scale_match, bbox_str)


def find_image_28(stem, img28_dir):
    """在 resize 后的目录中找到对应图片"""
    for ext in ['.jpg', '.jpeg', '.png']:
        p = os.path.join(img28_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def find_original_image(stem, img_dir):
    """在原始目录中找到对应图片"""
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def build_tampered_answer(caption, bboxes_from_mask, coord_scale_fn=None):
    """
    为伪造图片构建 <think>...</think><answer>...</answer> 格式的标签。

    caption: 中文描述 (含 bbox 和推理过程)
    bboxes_from_mask: 从 mask 提取的 bbox 列表 [[x1,y1,x2,y2], ...]
    coord_scale_fn: 可选, 将 bbox 字符串缩放到 model 空间的函数
    """
    # 用完整 caption 作为 think 内容
    think = caption.strip()

    # 从 caption 提取 bbox (优先用 caption 中的坐标, 更准确)
    caption_bboxes = BBOX_PAT.findall(caption)

    if caption_bboxes:
        # caption 中有 bbox
        bbox_strs = []
        for b in caption_bboxes:
            bbox_strs.append(f'[{b[0]}, {b[1]}, {b[2]}, {b[3]}]')
        bbox_text = ', '.join(bbox_strs)
    elif bboxes_from_mask:
        # fallback: 从 mask 提取的 bbox
        bbox_strs = []
        for b in bboxes_from_mask:
            bbox_strs.append(f'[{b[0]}, {b[1]}, {b[2]}, {b[3]}]')
        bbox_text = ', '.join(bbox_strs)
    else:
        bbox_text = ''

    # 将 bbox 坐标缩放到 processor 实际看到的空间 (如果需要)
    if coord_scale_fn and bbox_text:
        bbox_text = coord_scale_fn(bbox_text)

    # 从 caption 提取引号内的篡改文本
    quotes = QUOTE_PAT.findall(caption)
    tampered_texts = quotes[:2] if quotes else []

    # 判断篡改方法
    if 'copy-paste' in caption.lower() or '粘贴' in caption or '擦除' in caption:
        method = 'copy-paste'
    else:
        method = 'generation'

    # 构建 answer
    answer_parts = [f'This image is tampered. It was tampered by {method}.']
    if tampered_texts:
        text_desc = ', '.join(f'"{t}"' for t in tampered_texts)
        answer_parts.append(f'The tampered text reads {text_desc}.')
    if bbox_text:
        answer_parts.append(f'The tampered region is located at {bbox_text}.')

    answer = ' '.join(answer_parts)
    return f'<think>{think}</think>\n<answer> {answer} </answer>'


def build_real_answer(caption):
    """为真实图片构建标签"""
    think = caption.strip()
    return f'<think>{think}</think>\n<answer> This image is real. </answer>'


def mask_to_bboxes(mask_path):
    """从二值 mask 提取 bbox"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    binary = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 2 and h > 2:
            bboxes.append([x, y, x + w, y + h])
    return bboxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='dataset/train', help='训练数据根目录')
    parser.add_argument('--image_dir_28', default=None,
                        help='Black resize后的图片目录, 不指定则使用原图')
    parser.add_argument('--white_image_dir_28', default=None,
                        help='White resize后的图片目录, 不指定则使用原图')
    parser.add_argument('--output', default='sft_train.jsonl', help='输出 jsonl 路径')
    parser.add_argument('--scale_coords', action='store_true',
                        help='将 bbox 坐标缩放到 processor 实际看到的空间 '
                             '(修复大图训练时坐标偏移问题)')
    args = parser.parse_args()

    train_dir = os.path.abspath(args.train_dir)
    black_dir = os.path.join(train_dir, 'Black')
    white_dir = os.path.join(train_dir, 'White')

    black_caption_dir = os.path.join(black_dir, 'Caption')
    black_img_dir = os.path.join(black_dir, 'Image')
    black_mask_dir = os.path.join(black_dir, 'Mask')
    white_caption_dir = os.path.join(white_dir, 'Caption')
    white_img_dir = os.path.join(white_dir, 'Image')

    # resize 后的图片目录
    black_img28 = os.path.abspath(args.image_dir_28) if args.image_dir_28 else os.path.join(black_dir, 'Image_28')
    white_img28 = os.path.abspath(args.white_image_dir_28) if args.white_image_dir_28 else None

    samples = []

    # ── Black (伪造) 样本 ──
    print('Processing Black (tampered) samples...')
    caption_files = [f for f in sorted(os.listdir(black_caption_dir))
                     if os.path.splitext(f)[1].lower() in {'.md', '.txt'}]
    for fname in tqdm(caption_files):
        stem = os.path.splitext(fname)[0]

        # 读取 caption
        with open(os.path.join(black_caption_dir, fname), encoding='utf-8') as f:
            caption = f.read().strip()

        # 找到 resize 后的图片
        img_path = find_image_28(stem, black_img28)
        if img_path is None:
            img_path = find_original_image(stem, black_img_dir)
        if img_path is None:
            print(f'  Warning: no image for {stem}, skipping')
            continue

        # 验证图片尺寸是28的倍数
        w, h = imagesize.get(img_path)
        if h % 28 != 0 or w % 28 != 0:
            print(f'  Warning: {img_path} size {w}x{h} not multiple of 28, skipping')
            continue

        # 从 mask 提取 bbox (作为 fallback)
        mask_path = os.path.join(black_mask_dir, stem + '.png')
        bboxes_from_mask = mask_to_bboxes(mask_path) if os.path.exists(mask_path) else []

        # 坐标缩放函数 (如果启用)
        coord_scale_fn = None
        if args.scale_coords:
            # 获取原始图片尺寸 (用于计算缩放比)
            orig_img_path = find_original_image(stem, black_img_dir)
            if orig_img_path:
                ow, oh = imagesize.get(orig_img_path)
                coord_scale_fn = lambda s, ow=ow, oh=oh, w=w, h=h: \
                    scale_bbox_for_processor(s, ow, oh, w, h)

        # 构建标签
        assistant_content = build_tampered_answer(caption, bboxes_from_mask, coord_scale_fn)

        samples.append({
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': PROMPT},
                {'role': 'assistant', 'content': assistant_content},
            ],
            'images': [os.path.relpath(img_path)],
        })

    # ── White (真实) 样本 ──
    print('Processing White (real) samples...')
    white_caption_files = [f for f in sorted(os.listdir(white_caption_dir))
                           if os.path.splitext(f)[1].lower() in {'.md', '.txt'}]
    for fname in tqdm(white_caption_files):
        stem = os.path.splitext(fname)[0]

        with open(os.path.join(white_caption_dir, fname), encoding='utf-8') as f:
            caption = f.read().strip()

        # 找图片 (White 可能没有 resize 目录)
        img_path = None
        if white_img28:
            img_path = find_image_28(stem, white_img28)
        if img_path is None:
            img_path = find_original_image(stem, white_img_dir)
        if img_path is None:
            print(f'  Warning: no image for {stem}, skipping')
            continue

        # 检查尺寸
        w, h = imagesize.get(img_path)
        if h % 28 != 0 or w % 28 != 0:
            print(f'  Warning: {img_path} size {w}x{h} not multiple of 28, skipping')
            continue

        assistant_content = build_real_answer(caption)

        samples.append({
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': PROMPT},
                {'role': 'assistant', 'content': assistant_content},
            ],
            'images': [os.path.relpath(img_path)],
        })

    # 写入 jsonl
    import random
    random.seed(42)
    random.shuffle(samples)
    with open(args.output, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    n_tampered = sum(1 for s in samples if 'tampered' in s['messages'][2]['content'])
    n_real = sum(1 for s in samples if 'is real' in s['messages'][2]['content'])
    print(f'\nTotal: {len(samples)} samples written to {args.output}')
    print(f'  Black (tampered): {n_tampered}')
    print(f'  White (real): {n_real}')
    if args.scale_coords:
        print(f'  Coordinate scaling: ENABLED (bbox coords scaled to processor-seen space)')


if __name__ == '__main__':
    main()
