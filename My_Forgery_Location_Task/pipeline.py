"""
TextShield 赛题推理 Pipeline
=============================
用法 (在 My_Forgery_Location_Task/ 目录下执行):

  第0步: 用 padding 将图片调整为 28 的倍数 (避免 resize 形变)
    python pad_image_dir.py --input dataset/test --output dataset/test_28

  第1步: 生成推理数据集 jsonl
    python pipeline.py prepare --input dataset/test_28 --image_dir dataset/test

  第2步: 用 swift 批量推理 (单独执行)
    CUDA_VISIBLE_DEVICES=0 swift infer \
      --model ../textshield \
      --val_dataset dataset/test_28.jsonl \
      --max_new_tokens 4096 \
      --model_type qwen2_5_vl \
      --template qwen2_5_vl \
      --result_path inference_output.jsonl

  第3步: 解析推理结果, 生成 mask/可视化/提交CSV
    python pipeline.py postprocess \
      --result inference_output.jsonl \
      --image_dir dataset/test \
      --output_dir output_finetuned
"""

import argparse
import csv
import json
import os
import re
import sys
from typing import List, Optional, Tuple

import cv2
import imagesize
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────────
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

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

ANSWER_PAT = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

BBOX_PATTERNS = [
    re.compile(r'<\|box_start\|>\s*\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)\s*<\|box_end\|>'),
    re.compile(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'),
    re.compile(r'\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)'),
]


# ──────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────
def extract_answer(response: str) -> str:
    """从模型输出中提取 <answer>...</answer> 内容"""
    m = ANSWER_PAT.search(response)
    if m:
        return m.group(1).strip()
    return response.strip()


def parse_label(answer: str) -> int:
    """从 answer 文本判断 label: 0=真实, 1=伪造"""
    lower = answer.lower()
    if 'tampered' in lower or '伪造' in lower or '篡改' in lower:
        return 1
    if 'generated' in lower or '生成' in lower:
        return 1
    return 0


def extract_bboxes(answer: str) -> List[List[int]]:
    """从 answer 文本中提取所有 bounding box 坐标 [x1,y1,x2,y2]"""
    bboxes = []
    for pat in BBOX_PATTERNS:
        for m in pat.finditer(answer):
            bbox = [int(x) for x in m.groups()]
            bboxes.append(bbox)
        if bboxes:
            return bboxes
    # fallback: 提取所有数字, 每4个一组
    nums = [int(x) for x in re.findall(r'\d+', answer)]
    for i in range(0, len(nums) - 3, 4):
        bboxes.append(nums[i:i+4])
    return bboxes


def extract_explanation(response: str) -> str:
    """提取可解释归因文本: 优先用 <answer> 内容, 否则用完整 response"""
    answer = extract_answer(response)
    # 移除 think 标签
    answer = re.sub(r'</?think>', '', answer).strip()
    return answer


def clamp_bbox(bbox: List[int], width: int, height: int) -> Optional[List[int]]:
    """归一化并裁剪 bbox 到图片范围"""
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_iou(a: List[int], b: List[int]) -> float:
    """计算两个 bbox 的 IoU"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def nms_bboxes(bboxes: List[List[int]], iou_thresh: float = 0.5) -> List[List[int]]:
    """NMS 去除重叠 bbox, 按面积从大到小保留"""
    if not bboxes:
        return []
    # 按面积降序排列
    scored = sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    keep = []
    for box in scored:
        if all(bbox_iou(box, k) < iou_thresh for k in keep):
            keep.append(box)
    return keep


def filter_bboxes(bboxes: List[List[int]], width: int, height: int,
                  max_count: int = 20, min_area_ratio: float = 1e-5) -> List[List[int]]:
    """
    过滤异常 bbox:
    1. clamp 到图片范围
    2. NMS 去重叠
    3. 去掉面积占比过小的(噪声)
    4. 限制最大数量
    """
    img_area = width * height
    # clamp
    valid = []
    for bbox in bboxes:
        clamped = clamp_bbox(bbox, width, height)
        if clamped:
            area = (clamped[2] - clamped[0]) * (clamped[3] - clamped[1])
            if area / img_area >= min_area_ratio:
                valid.append(clamped)
    # NMS
    valid = nms_bboxes(valid, iou_thresh=0.5)
    # 限制数量 (保留面积最大的)
    if len(valid) > max_count:
        valid = sorted(valid, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:max_count]
    return valid


def bboxes_to_mask(bboxes: List[List[int]], width: int, height: int) -> np.ndarray:
    """将多个 bbox 转为二值 mask"""
    mask = np.zeros((height, width), dtype=np.uint8)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255
    return mask


def mask_to_rle(mask: np.ndarray) -> str:
    """二值 mask → RLE JSON 字符串"""
    binary = (mask > 127).astype(np.uint8)
    fortran = np.asfortranarray(binary)
    rle_dict = mask_utils.encode(fortran)
    if isinstance(rle_dict['counts'], bytes):
        rle_dict['counts'] = rle_dict['counts'].decode('utf-8')
    return json.dumps(rle_dict)


def create_overlay(image: np.ndarray, mask: np.ndarray,
                   bboxes: List[List[int]], alpha: float = 0.4) -> np.ndarray:
    """在原图上叠加半透明 mask(红色) + bbox 矩形框"""
    overlay = image.copy()
    # 半透明红色 mask
    red = np.zeros_like(image)
    red[:, :, 2] = 255  # BGR -> Red channel
    mask_bool = mask > 127
    overlay[mask_bool] = cv2.addWeighted(image, 1 - alpha, red, alpha, 0)[mask_bool]
    # bbox 矩形框
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return overlay


# ──────────────────────────────────────────────────────────────────────
# 第1步: prepare — 生成推理用 jsonl
# ──────────────────────────────────────────────────────────────────────
def cmd_prepare(args):
    in_dir = os.path.abspath(args.input)
    # 原始图片目录(未 resize), 用于记录真实文件名
    orig_dir = os.path.abspath(args.image_dir) if args.image_dir else in_dir

    files = sorted([
        f for f in os.listdir(in_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXT
    ])

    out_path = args.output if args.output else in_dir.rstrip('/') + '.jsonl'
    with open(out_path, 'w', encoding='utf-8') as fout:
        for fname in tqdm(files, desc="Preparing dataset"):
            fpath = os.path.join(in_dir, fname)
            w, h = imagesize.get(fpath)
            assert h % 28 == 0 and w % 28 == 0, \
                f'{fpath}: size {w}x{h} not multiple of 28'

            item = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": PROMPT},
                ],
                "images": [os.path.relpath(fpath)],
                # 自定义字段, 方便后处理时追溯原始文件名
                "image_name": fname,
            }
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f'Dataset saved to: {out_path}  ({len(files)} images)')


# ──────────────────────────────────────────────────────────────────────
# 第2步: postprocess — 解析推理结果, 生成 mask/可视化/CSV
# ──────────────────────────────────────────────────────────────────────
def cmd_postprocess(args):
    result_path = args.result
    image_dir = os.path.abspath(args.image_dir)
    output_dir = os.path.abspath(args.output_dir)

    mask_dir = os.path.join(output_dir, 'masks')
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    with open(result_path, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f if line.strip()]

    csv_path = os.path.join(output_dir, 'submission.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'label', 'location', 'explanation'])

        for item in tqdm(results, desc="Postprocessing"):
            response = item.get('response', '')
            # 获取图片文件名
            image_name = item.get('image_name', '')
            if not image_name:
                # 尝试从 images 字段提取
                images = item.get('images', [])
                if images:
                    img_field = images[0]
                    if isinstance(img_field, dict):
                        img_field = img_field.get('path', '')
                    image_name = os.path.basename(img_field)
                else:
                    # 从 messages 中提取
                    for msg in item.get('messages', []):
                        content = msg.get('content', '')
                        m = re.search(r'<img>(.*?)</img>', content)
                        if m:
                            image_name = os.path.basename(m.group(1))
                            break

            if not image_name:
                print(f'Warning: cannot determine image_name, skipping')
                continue

            # 读取原始图片 (未 resize 的) 获取真实尺寸
            stem = os.path.splitext(image_name)[0]
            orig_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                candidate = os.path.join(image_dir, stem + ext)
                if os.path.exists(candidate):
                    orig_path = candidate
                    image_name = stem + ext
                    break
            if orig_path is None:
                orig_path = os.path.join(image_dir, image_name)

            image = cv2.imread(orig_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f'Warning: cannot read {orig_path}, skipping')
                continue
            h, w = image.shape[:2]

            # 解析结果
            answer = extract_answer(response)
            label = parse_label(answer)
            explanation = extract_explanation(response)

            if label == 1:
                bboxes = extract_bboxes(answer)
                valid_bboxes = filter_bboxes(bboxes, w, h)
                mask = bboxes_to_mask(valid_bboxes, w, h)
            else:
                valid_bboxes = []
                mask = np.zeros((h, w), dtype=np.uint8)

            # 保存 mask
            mask_path = os.path.join(mask_dir, stem + '.png')
            cv2.imwrite(mask_path, mask)

            # 保存可视化
            overlay = create_overlay(image, mask, valid_bboxes)
            vis_path = os.path.join(vis_dir, stem + '.jpg')
            cv2.imwrite(vis_path, overlay)

            # RLE 编码
            rle_str = mask_to_rle(mask) if label == 1 else ''

            writer.writerow([image_name, label, rle_str, explanation])

    print(f'\nResults saved to: {output_dir}')
    print(f'  - submission.csv : {csv_path}')
    print(f'  - masks/         : {mask_dir}')
    print(f'  - visualizations/: {vis_dir}')


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='TextShield Competition Pipeline')
    sub = parser.add_subparsers(dest='command')

    # prepare
    p1 = sub.add_parser('prepare', help='生成推理数据集 jsonl')
    p1.add_argument('--input', required=True, help='resize 后的图片目录 (28倍数)')
    p1.add_argument('--image_dir', default=None, help='原始图片目录 (用于记录文件名)')
    p1.add_argument('--output', default=None, help='输出 jsonl 路径')

    # postprocess
    p2 = sub.add_parser('postprocess', help='解析推理结果, 生成 mask/可视化/CSV')
    p2.add_argument('--result', required=True, help='swift infer 输出的 jsonl 文件')
    p2.add_argument('--image_dir', required=True, help='原始图片目录 (用于获取真实尺寸)')
    p2.add_argument('--output_dir', default='output', help='输出目录')

    args = parser.parse_args()
    if args.command == 'prepare':
        cmd_prepare(args)
    elif args.command == 'postprocess':
        cmd_postprocess(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
