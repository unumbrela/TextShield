import argparse
import os
import re
from typing import List

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render a binary mask from a predicted text tampering bounding box.'
    )
    parser.add_argument('--image', required=True, help='Input image path.')
    parser.add_argument(
        '--output-mask',
        required=True,
        help='Output path for the binary mask image.',
    )
    parser.add_argument(
        '--output-overlay',
        default=None,
        help='Optional output path for the visualization image.',
    )
    parser.add_argument(
        '--bbox',
        nargs=4,
        type=int,
        default=None,
        metavar=('X1', 'Y1', 'X2', 'Y2'),
        help='Bounding box coordinates.',
    )
    parser.add_argument(
        '--answer',
        default=None,
        help='Model answer text containing bbox coordinates.',
    )
    return parser.parse_args()


def extract_bbox_from_answer(answer: str) -> List[int]:
    patterns = [
        r'<\|box_start\|>\s*\((\d+),\s*(\d+)\),\((\d+),\s*(\d+)\)\s*<\|box_end\|>',
        r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
        r'\((\d+),\s*(\d+)\),\((\d+),\s*(\d+)\)',
    ]
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match:
            return [int(x) for x in match.groups()]
    numbers = [int(x) for x in re.findall(r'\d+', answer)]
    if len(numbers) >= 4:
        return numbers[:4]
    raise ValueError('Failed to extract four bbox coordinates from answer text.')


def normalize_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f'Invalid bbox after clamping: {bbox} -> {[x1, y1, x2, y2]}')
    return [x1, y1, x2, y2]


def main():
    args = parse_args()

    if (args.bbox is None) == (args.answer is None):
        raise ValueError('Provide exactly one of --bbox or --answer.')

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Failed to read image: {args.image}')

    height, width = image.shape[:2]
    bbox = args.bbox if args.bbox is not None else extract_bbox_from_answer(args.answer)
    x1, y1, x2, y2 = normalize_bbox(bbox, width, height)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    os.makedirs(os.path.dirname(os.path.abspath(args.output_mask)), exist_ok=True)
    cv2.imwrite(args.output_mask, mask)

    if args.output_overlay:
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        os.makedirs(os.path.dirname(os.path.abspath(args.output_overlay)), exist_ok=True)
        cv2.imwrite(args.output_overlay, overlay)

    print(f'Image: {args.image}')
    print(f'BBox: {[x1, y1, x2, y2]}')
    print(f'Mask saved to: {args.output_mask}')
    if args.output_overlay:
        print(f'Overlay saved to: {args.output_overlay}')


if __name__ == '__main__':
    main()
