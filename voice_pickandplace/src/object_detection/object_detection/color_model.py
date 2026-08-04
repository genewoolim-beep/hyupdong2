########## ColorModel ##########
# YOLO 없이, 색깔만으로 물체 하나를 찾는다.
#
# HSV 범위와 튜닝 값은 코드가 아니라 resource/color_ranges.json 에 있다.
# 조명이 바뀔 때마다 다시 빌드하지 않고 그 파일만 고쳐서 대응하기 위해서다.
import json
import os

import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory

from object_detection.detection_utils import collect_frames, consensus_box

PACKAGE_NAME = 'object_detection'
CONFIG_PATH = os.path.join(
    get_package_share_directory(PACKAGE_NAME), 'resource', 'color_ranges.json'
)


def _load_config(path):
    """색 범위 정의를 읽어 cv2.inRange 가 바로 쓸 수 있는 형태로 바꾼다."""
    with open(path, encoding='utf-8') as f:
        spec = json.load(f)

    ranges = {}
    for name, entry in spec['colors'].items():
        bands = [
            ((r['h'][0], r['s'][0], r['v'][0]), (r['h'][1], r['s'][1], r['v'][1]))
            for r in entry['ranges']
        ]
        ranges[name] = bands
        if entry.get('en'):
            # 같은 리스트를 가리키게 해서, 위 값만 고치면 영어 이름에도 그대로 반영되게 한다
            ranges[entry['en']] = bands

    return ranges, spec['filters'], spec['frames']


COLOR_HSV_RANGES, _FILTERS, _FRAMES = _load_config(CONFIG_PATH)

MIN_AREA = _FILTERS['min_area']
MAX_AREA_RATIO = _FILTERS['max_area_ratio']
MAX_ASPECT_RATIO = _FILTERS['max_aspect_ratio']

FRAME_DURATION = _FRAMES['duration_sec']
IOU_THRESHOLD = _FRAMES['iou_threshold']
MIN_HIT_RATIO = _FRAMES['min_hit_ratio']


def block_candidates(contours, max_area):
    """블록처럼 보이는(크기가 적당하고 정사각형에 가까운) 뭉치만 골라낸다.

    붙어있는 다른 색 블록까지 morphology로 합쳐지면 박스가 길쭉해지는데,
    그런 경우를 여기서 걸러낸다.
    """
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_AREA <= area <= max_area):
            continue
        _, _, w, h = cv2.boundingRect(c)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > MAX_ASPECT_RATIO:
            continue
        candidates.append(c)
    return candidates


def detect_color_box(hsv, ranges, max_area):
    """HSV 이미지 한 장에서 그 색의 블록 하나를 찾아 (박스, 통계) 를 돌려준다.

    ColorModel 과 color_view 가 같은 결과를 보도록 검출 과정을 여기 한 곳에 둔다.
    함께 돌려주는 통계는 못 찾았을 때 그 이유를 로그로 남기기 위한 것이다.
    """
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    raw_pixels = int(cv2.countNonZero(mask))

    # 작은 점 노이즈는 지운다 (open). 닫기(close)는 너무 크게 하면 옆에 붙은
    # 다른 블록까지 하나로 합쳐버리므로 작은 커널만 살짝 적용한다.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats = {
        'raw_pixels': raw_pixels,
        'areas': sorted((cv2.contourArea(c) for c in contours), reverse=True),
    }

    candidates = block_candidates(contours, max_area)
    if not candidates:
        return None, stats

    best = max(candidates, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)
    return [float(x), float(y), float(x + w), float(y + h)], stats


class ColorModel:
    def get_best_detection(self, img_node, target):
        key = target.strip().lower()
        ranges = COLOR_HSV_RANGES.get(key) or COLOR_HSV_RANGES.get(target.strip())
        if ranges is None:
            print(f"'{target}' is not a known color. known: {sorted(set(COLOR_HSV_RANGES))}")
            return None, None

        # 한 장만 보면 그 순간의 그림자나 반사광에 그대로 속는다.
        # 짧게 여러 장을 모아서 매번 같은 자리에 나오는 것만 인정한다.
        frames = collect_frames(img_node, FRAME_DURATION)
        if not frames:
            print("No frames captured from the camera.")
            return None, None

        boxes = []
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            max_area = hsv.shape[0] * hsv.shape[1] * MAX_AREA_RATIO
            box, _ = detect_color_box(hsv, ranges, max_area)
            if box is not None:
                boxes.append(box)

        box, hit_ratio = consensus_box(boxes, len(frames), IOU_THRESHOLD, MIN_HIT_RATIO)
        if box is None:
            print(f"'{target}' was not seen consistently across {len(frames)} frames.")
            return None, None

        return box, hit_ratio
