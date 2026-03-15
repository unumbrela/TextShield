"""
TextShield 端到端推理工具
========================
支持单张图片或整个文件夹, 自动完成: 加载模型 → pad → 推理 → 解析 → 生成 mask/可视化/submission

用法:
  # 单张图片推理 (原始模型)
  python inference/infer.py --input path/to/image.jpg

  # 文件夹推理 (原始模型)
  python inference/infer.py --input path/to/image_dir/

  # 使用微调后的模型
  python inference/infer.py --input path/to/image_dir/ --model path/to/finetuned_model

  # 指定输出目录和 GPU
  python inference/infer.py --input path/to/image_dir/ --output_dir results/ --gpu 0

  # 使用 resize 模式 (原项目方式, 四舍五入到 28 倍数) 替代默认 pad 模式
  python inference/infer.py --input path/to/image.jpg --preprocess resize

  # 控制 max_pixels (防止 processor 内部缩放导致 bbox 偏移)
  # 默认不限制 (0), 即不缩放; 设为 1003520 恢复 processor 默认行为
  python inference/infer.py --input path/to/image.jpg --max_pixels 0

输出结构:
  output_dir/
  ├── raw_responses.jsonl   — 模型原始输出 (完整保留 think + answer)
  ├── submission.csv        — 提交格式 (image_name, label, location, explanation)
  ├── masks/                — 二值 mask PNG
  └── visualizations/       — 原图叠加 mask + bbox 的可视化
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from typing import List, Optional

import cv2
import numpy as np
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
THINK_PAT = re.compile(r'<think>(.*?)</think>', re.DOTALL)

BBOX_PATTERNS = [
    re.compile(r'<\|box_start\|>\s*\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)\s*<\|box_end\|>'),
    re.compile(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'),
    re.compile(r'\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)'),
]


# ──────────────────────────────────────────────────────────────────────
# 图像预处理
# ──────────────────────────────────────────────────────────────────────
QWEN_DEFAULT_MAX_PIXELS = 1_003_520   # 1280 * 28 * 28
QWEN_DEFAULT_MIN_PIXELS = 3_136       # 28 * 28 * 4


def smart_resize(height, width, factor=28,
                 min_pixels=QWEN_DEFAULT_MIN_PIXELS,
                 max_pixels=QWEN_DEFAULT_MAX_PIXELS):
    """复制 Qwen2.5-VL processor 的 smart_resize 逻辑,
    返回 processor 实际会将图片缩放到的 (new_h, new_w)。"""
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


def ceil_to_mult(x, m=28):
    return ((x + m - 1) // m) * m


def pad_image(img, pad_value=0):
    """将单张图片 pad 到 28 的倍数, 返回 (processed_img, orig_h, orig_w)"""
    h, w = img.shape[:2]
    nh = ceil_to_mult(h, 28)
    nw = ceil_to_mult(w, 28)
    if (nh, nw) != (h, w):
        img = cv2.copyMakeBorder(
            img, 0, nh - h, 0, nw - w,
            cv2.BORDER_CONSTANT, value=pad_value
        )
    return img, h, w


def resize_image(img):
    """将单张图片 resize 到 28 的倍数 (原项目方式), 返回 (processed_img, orig_h, orig_w)"""
    h, w = img.shape[:2]
    nh = max(28, round(h / 28) * 28)
    nw = max(28, round(w / 28) * 28)
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return img, h, w


# ──────────────────────────────────────────────────────────────────────
# 结果解析 (复用 pipeline.py 的逻辑)
# ──────────────────────────────────────────────────────────────────────
def extract_answer(response: str) -> str:
    m = ANSWER_PAT.search(response)
    return m.group(1).strip() if m else response.strip()


def extract_think(response: str) -> str:
    m = THINK_PAT.search(response)
    return m.group(1).strip() if m else ''


def parse_label(answer: str) -> int:
    lower = answer.lower()
    if 'tampered' in lower or '伪造' in lower or '篡改' in lower:
        return 1
    if 'generated' in lower or '生成' in lower:
        return 1
    return 0


def extract_bboxes(answer: str) -> List[List[int]]:
    bboxes = []
    for pat in BBOX_PATTERNS:
        for m in pat.finditer(answer):
            bboxes.append([int(x) for x in m.groups()])
        if bboxes:
            return bboxes
    nums = [int(x) for x in re.findall(r'\d+', answer)]
    for i in range(0, len(nums) - 3, 4):
        bboxes.append(nums[i:i + 4])
    return bboxes


def clamp_bbox(bbox, width, height):
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


def bbox_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def filter_bboxes(bboxes, width, height, max_count=20, min_area_ratio=1e-5):
    img_area = width * height
    valid = []
    for bbox in bboxes:
        clamped = clamp_bbox(bbox, width, height)
        if clamped:
            area = (clamped[2] - clamped[0]) * (clamped[3] - clamped[1])
            if area / img_area >= min_area_ratio:
                valid.append(clamped)
    # NMS
    scored = sorted(valid, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    keep = []
    for box in scored:
        if all(bbox_iou(box, k) < 0.5 for k in keep):
            keep.append(box)
    return keep[:max_count]


def bboxes_to_mask(bboxes, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in bboxes:
        mask[y1:y2, x1:x2] = 255
    return mask


def mask_to_rle(mask):
    from pycocotools import mask as mask_utils
    binary = (mask > 127).astype(np.uint8)
    fortran = np.asfortranarray(binary)
    rle_dict = mask_utils.encode(fortran)
    if isinstance(rle_dict['counts'], bytes):
        rle_dict['counts'] = rle_dict['counts'].decode('utf-8')
    return json.dumps(rle_dict)


def create_overlay(image, mask, bboxes, alpha=0.4):
    overlay = image.copy()
    red = np.zeros_like(image)
    red[:, :, 2] = 255
    mask_bool = mask > 127
    if mask_bool.any():
        overlay[mask_bool] = cv2.addWeighted(image, 1 - alpha, red, alpha, 0)[mask_bool]
    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return overlay


# ──────────────────────────────────────────────────────────────────────
# 模型加载与推理
# ──────────────────────────────────────────────────────────────────────
def load_model(model_path, gpu_id=0):
    """加载 Qwen2.5-VL 模型"""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from: {model_path}")
    print(f"Device: {device}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)

    print("Model loaded successfully.")
    return model, processor, device


def run_inference(model, processor, device, image_path, max_new_tokens=4096,
                  max_pixels=0):
    """对单张图片运行推理, 返回模型原始输出。

    Args:
        max_pixels: processor 的 max_pixels 参数。
            0 = 不限制 (设为图片实际像素数, 防止 processor 内部缩放);
            >0 = 使用指定值。
    """
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
            {"type": "text", "text": PROMPT.replace('<image> ', '')},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    # 设置 max_pixels 防止 processor 内部缩放导致 bbox 偏移
    if max_pixels == 0:
        # 不限制: 用图片实际像素数 (稍微上浮以确保不触发缩放)
        if image_inputs:
            from PIL import Image
            if isinstance(image_inputs[0], Image.Image):
                iw, ih = image_inputs[0].size
                effective_max = max(iw * ih + 10000, QWEN_DEFAULT_MAX_PIXELS)
            else:
                effective_max = 100_000_000
        else:
            effective_max = QWEN_DEFAULT_MAX_PIXELS
    else:
        effective_max = max_pixels

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        max_pixels=effective_max,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # 只取生成的部分 (去掉 input tokens)
    generated_ids = output_ids[0, inputs.input_ids.shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=False)

    # 清理结束 token
    response = response.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()

    return response


# ──────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────
def collect_images(input_path):
    """收集输入路径中的所有图片, 返回 [(原始路径, 文件名), ...]"""
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in IMG_EXT:
            return [(input_path, os.path.basename(input_path))]
        else:
            print(f"Error: {input_path} is not a supported image format")
            sys.exit(1)
    elif os.path.isdir(input_path):
        files = []
        for f in sorted(os.listdir(input_path)):
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                files.append((os.path.join(input_path, f), f))
        if not files:
            print(f"Error: no images found in {input_path}")
            sys.exit(1)
        return files
    else:
        print(f"Error: {input_path} does not exist")
        sys.exit(1)


def process_single(model, processor, device, orig_path, fname, output_dir,
                   max_new_tokens, preprocess='pad', max_pixels=0):
    """处理单张图片: pad/resize → 推理 → 解析 → 生成输出

    Args:
        max_pixels: 传给 processor 的 max_pixels。0=不限制(防止缩放偏移)。
    """

    # 读取原始图片
    orig_img = cv2.imread(orig_path, cv2.IMREAD_COLOR)
    if orig_img is None:
        print(f"  Warning: cannot read {orig_path}, skipping")
        return None
    orig_h, orig_w = orig_img.shape[:2]

    # 预处理到 28 倍数
    if preprocess == 'resize':
        processed_img, _, _ = resize_image(orig_img)
    else:
        processed_img, _, _ = pad_image(orig_img)

    # 保存预处理后的图片到临时文件 (模型需要读文件路径)
    tmp_dir = os.path.join(output_dir, '.tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    processed_path = os.path.join(tmp_dir, fname)
    cv2.imwrite(processed_path, processed_img)

    # 推理
    response = run_inference(model, processor, device, processed_path,
                             max_new_tokens, max_pixels=max_pixels)

    # 解析
    answer = extract_answer(response)
    think = extract_think(response)
    label = parse_label(answer)
    explanation = re.sub(r'</?think>', '', answer).strip()

    if label == 1:
        bboxes = extract_bboxes(answer)
        valid_bboxes = filter_bboxes(bboxes, orig_w, orig_h)
        mask = bboxes_to_mask(valid_bboxes, orig_w, orig_h)
    else:
        valid_bboxes = []
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    stem = os.path.splitext(fname)[0]

    # 保存 mask
    mask_dir = os.path.join(output_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)
    cv2.imwrite(os.path.join(mask_dir, stem + '.png'), mask)

    # 保存可视化
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    overlay = create_overlay(orig_img, mask, valid_bboxes)
    cv2.imwrite(os.path.join(vis_dir, stem + '.jpg'), overlay)

    # RLE
    rle_str = mask_to_rle(mask) if label == 1 else ''

    result = {
        'image_name': fname,
        'orig_size': [orig_w, orig_h],
        'preprocess': preprocess,
        'label': label,
        'label_text': 'tampered' if label == 1 else 'real',
        'bboxes': valid_bboxes,
        'think': think,
        'answer': answer,
        'response_full': response,
        'explanation': explanation,
        'rle': rle_str,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description='TextShield 端到端推理 (支持单张图片或文件夹)')
    parser.add_argument('--input', required=True,
                        help='输入图片路径或图片文件夹')
    parser.add_argument('--model', default=None,
                        help='模型路径 (默认: textshield 原始模型)')
    parser.add_argument('--output_dir', default=None,
                        help='输出目录 (默认: inference/output_<模型名>/)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (默认: 0)')
    parser.add_argument('--max_new_tokens', type=int, default=4096,
                        help='最大生成 token 数 (默认: 4096)')
    parser.add_argument('--preprocess', choices=['pad', 'resize'], default='resize',
                        help='图片预处理方式: pad=填充黑边, resize=缩放到28倍数(默认)(原项目方式)')
    parser.add_argument('--max_pixels', type=int, default=0,
                        help='Processor max_pixels: 0=不限制(防止大图bbox偏移,默认), '
                             '1003520=processor默认值(省显存但大图坐标可能偏移)')
    args = parser.parse_args()

    # 确定模型路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if args.model:
        model_path = os.path.abspath(args.model)
    else:
        model_path = os.path.join(project_root, 'textshield')

    if not os.path.exists(model_path):
        print(f"Error: model not found at {model_path}")
        sys.exit(1)

    # 确定输出目录
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        model_name = os.path.basename(model_path.rstrip('/'))
        output_dir = os.path.join(script_dir, f'output_{model_name}')

    os.makedirs(output_dir, exist_ok=True)

    # 收集图片
    images = collect_images(args.input)
    print(f"Input: {args.input}")
    print(f"Images: {len(images)}")
    print(f"Model: {model_path}")
    max_pixels_desc = "no limit (prevent downscale)" if args.max_pixels == 0 else str(args.max_pixels)
    print(f"Preprocess: {args.preprocess}")
    print(f"Max pixels: {max_pixels_desc}")
    print(f"Output: {output_dir}")
    print()

    # 加载模型
    model, processor, device = load_model(model_path, args.gpu)
    print()

    # 逐张推理
    results = []
    for orig_path, fname in tqdm(images, desc="Inferring"):
        result = process_single(
            model, processor, device,
            orig_path, fname, output_dir,
            args.max_new_tokens,
            preprocess=args.preprocess,
            max_pixels=args.max_pixels,
        )
        if result is None:
            continue
        results.append(result)

        # 实时打印结果
        label_str = f"{'TAMPERED' if result['label'] == 1 else 'REAL':>8}"
        n_boxes = len(result['bboxes'])
        print(f"  {fname}: {label_str}  boxes={n_boxes}")

    # 保存 raw_responses.jsonl (完整保留所有模型输出)
    raw_path = os.path.join(output_dir, 'raw_responses.jsonl')
    with open(raw_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 保存 submission.csv
    csv_path = os.path.join(output_dir, 'submission.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'label', 'location', 'explanation'])
        for r in results:
            writer.writerow([r['image_name'], r['label'], r['rle'], r['explanation']])

    # 清理临时文件
    tmp_dir = os.path.join(output_dir, '.tmp')
    if os.path.exists(tmp_dir):
        import shutil
        shutil.rmtree(tmp_dir)

    # 打印汇总
    n_tampered = sum(1 for r in results if r['label'] == 1)
    n_real = len(results) - n_tampered
    print(f"\n{'='*50}")
    print(f"Done! Processed {len(results)} images")
    print(f"  TAMPERED: {n_tampered}")
    print(f"  REAL:     {n_real}")
    print(f"\nOutputs:")
    print(f"  {raw_path}")
    print(f"  {csv_path}")
    print(f"  {os.path.join(output_dir, 'masks/')}")
    print(f"  {os.path.join(output_dir, 'visualizations/')}")


if __name__ == '__main__':
    main()
