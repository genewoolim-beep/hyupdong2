########## ColorModel ##########
# YOLO 없이, 색깔만으로 물체 하나를 찾는다.
#
# HSV 범위와 튜닝 값은 코드가 아니라 resource/color_ranges.json 에 있다.
# 조명이 바뀔 때마다 다시 빌드하지 않고 그 파일만 고쳐서 대응하기 위해서다.
import json
import os
import time

import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory

from object_detection.detection_utils import collect_frames, consensus_boxes

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

# 색마다 다시 찍지 않고 이 시간 동안은 같은 촬영을 돌려 쓴다. 자세한 이유는
# ColorModel._frames() 주석 참고.
# 0.5초로 잡았더니 색 하나 처리에 0.6초가 걸려 매번 만료됐다(6색 3.9초).
# 6색을 한 번의 촬영으로 덮으려면 촬영 0.4초 + 색당 처리 0.2초 × 6 = 1.6초 이상.
# 팔이 움직이면 장면이 바뀌지만, 이동에는 대기까지 2초 넘게 걸려 그 사이 만료된다.
FRAME_CACHE_SEC = float(os.environ.get('FRAME_CACHE_SEC', 0.4))
# 자세가 그대로여도 이만큼 지나면 다시 찍는다. 팔이 멈춰 있는 동안 사람이
# 손으로 블록을 옮길 수 있으므로 무한정 돌려쓰면 안 된다.
FRAME_CACHE_MAX_SEC = float(os.environ.get('FRAME_CACHE_MAX_SEC', 3.0))

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


TOP_BAND = 12   # 윗면으로 볼 깊이 폭 (mm). 블록 높이 35mm 의 1/3 정도


def keep_top_face(depth, mask):
    """색 마스크에서 가장 가까운 평면(= 블록 윗면)만 남긴다.

    색만으로는 윗면과 옆면을 가르지 못한다. 비스듬히 볼수록 옆면이 마스크에
    붙어 박스가 커지고 중심이 밀린다. 실측에서 광축 근처 49.5mm, 화면
    가장자리 59.6mm 로 10mm 차이가 났다 — 중심 오차 5mm 다. 블록이 기울면
    옆면이 둘 보이면서 마스크가 L자로 일그러져 회전각까지 틀어진다.

    옆면은 깊이가 아래로 이어지고 바닥은 훨씬 멀다. 윗면 깊이의 ±TOP_BAND
    안에 있는 화소만 통과시키면 순수한 윗면 사각형이 남는다.
    깊이가 없으면 원래 마스크를 그대로 돌려준다 (동작은 유지).
    """
    if depth is None or depth.shape[:2] != mask.shape[:2]:
        return mask

    # 덩어리마다 따로 자른다. 마스크 전체를 한 번에 자르면 같은 색 블록이 여럿일 때
    # 가장 가까운 것만 남고 나머지가 통째로 사라진다.
    # 실측 2026-08-04: 관측 자세에서 빨간 블록 4개의 깊이가 671~872mm 로 벌어져,
    # 한 번에 자르니 671mm 짜리 하나만 살아남았다. 덩어리별로 자르니 4개 다 남았다.
    n, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    out = np.zeros_like(mask)
    for i in range(1, n):
        comp = np.where(labels == i, np.uint8(255), np.uint8(0))
        vals = depth[(comp > 0) & (depth > 0)]
        if vals.size < 200:
            out |= comp                 # 깊이가 모자라면 판단을 미루고 그대로 살린다
            continue
        # 윗면은 그 덩어리에서 가장 가까운 쪽이다. 잡음에 강하게 하위 20% 로 잡는다.
        top = float(np.percentile(vals, 20))
        band = ((depth > top - TOP_BAND) & (depth < top + TOP_BAND)).astype(np.uint8) * 255
        g = cv2.bitwise_and(comp, band)
        out |= g if cv2.countNonZero(g) >= 200 else comp
    return out if cv2.countNonZero(out) else mask


def detect_color_boxes(hsv, ranges, max_area, depth=None):
    """HSV 이미지 한 장에서 그 색으로 보이는 블록을 **전부** 찾는다.

    돌려주는 것은 ([(박스, 각도), ...], 통계) 이고, 큰 것부터 정렬돼 있다.
    하나만 돌려주면 같은 색 블록이 여럿일 때 부르는 쪽이 위치로 고를 수가 없다.

    ColorModel 과 color_view 가 같은 결과를 보도록 검출 과정을 여기 한 곳에 둔다.
    함께 돌려주는 통계는 못 찾았을 때 그 이유를 로그로 남기기 위한 것이고,
    회전각(`angle`)도 여기 담는다 — 축정렬 박스만으로는 기울어진 블록에
    그리퍼를 맞출 수 없어 모서리를 물게 된다.
    """
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    raw_pixels = int(cv2.countNonZero(mask))

    # 작은 점 노이즈는 지운다 (open). 닫기(close)는 너무 크게 하면 옆에 붙은
    # 다른 블록까지 하나로 합쳐버리므로 작은 커널만 살짝 적용한다.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = keep_top_face(depth, mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats = {
        'raw_pixels': raw_pixels,
        'areas': sorted((cv2.contourArea(c) for c in contours), reverse=True),
        'angle': 0.0,
    }

    found = []
    for c in sorted(block_candidates(contours, max_area), key=cv2.contourArea,
                    reverse=True):
        # 정사각 블록은 90° 대칭이므로 0~90 으로 접어서 쓴다.
        (_, _), (rw, rh), ang = cv2.minAreaRect(c)
        if rw < rh:
            ang += 90
        x, y, w, h = cv2.boundingRect(c)

        # 중심은 **마스크의 무게중심**으로 잡는다. 축정렬 박스의 한가운데를 쓰면
        # 한쪽으로 삐져나온 화소 몇 개가 박스를 늘리고 중심을 그 절반만큼 민다.
        # 무게중심은 모든 화소를 평균하므로 그런 이상치에 거의 흔들리지 않고,
        # 정사각 윗면처럼 대칭인 도형에서는 정확히 한가운데가 된다.
        # 소수점까지 살린다 — 35mm 블록이 약 86픽셀이라 1픽셀이 0.4mm다.
        m = cv2.moments(c)
        if m['m00'] > 0:
            cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
        else:
            cx, cy = x + w / 2.0, y + h / 2.0
        found.append(([float(x), float(y), float(x + w), float(y + h)],
                      float(ang % 90), float(cx), float(cy)))
    if found:
        stats['angle'] = found[0][1]
    return found, stats


def detect_color_box(hsv, ranges, max_area, depth=None):
    """가장 큰 것 하나만. 여러 개가 필요 없는 호출부(color_view 등)를 위해 남긴다."""
    found, stats = detect_color_boxes(hsv, ranges, max_area, depth)
    return (found[0][0] if found else None), stats


class ColorModel:
    def __init__(self):
        self.last_angle = 0.0      # 마지막 검출의 회전각 (도, 0~90)
        self._cache = None         # (시각, 프레임들, 깊이, 자세키) — _frames() 참고

    def get_all_detections(self, img_node, target, pose_key=None):
        """그 색으로 보이는 블록을 전부 [(박스, 각도, 신뢰도), ...] 로 돌려준다.

        신뢰도가 높은 순이라 첫 번째만 쓰면 get_best_detection() 과 같다.
        여러 개가 필요한 이유는 detection_utils.consensus_boxes() 주석 참고.
        """
        key = target.strip().lower()
        ranges = COLOR_HSV_RANGES.get(key) or COLOR_HSV_RANGES.get(target.strip())
        if ranges is None:
            print(f"'{target}' is not a known color. known: {sorted(set(COLOR_HSV_RANGES))}")
            self.last_angle = 0.0
            return []

        # 한 장만 보면 그 순간의 그림자나 반사광에 그대로 속는다.
        # 짧게 여러 장을 모아서 매번 같은 자리에 나오는 것만 인정한다.
        frames, depth = self._frames(img_node, pose_key)
        if not frames:
            print("No frames captured from the camera.")
            return []
        items = []
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            max_area = hsv.shape[0] * hsv.shape[1] * MAX_AREA_RATIO
            found, _ = detect_color_boxes(hsv, ranges, max_area, depth)
            items.extend(found)

        out = consensus_boxes(items, len(frames), IOU_THRESHOLD, MIN_HIT_RATIO)
        if not out:
            print(f"'{target}' was not seen consistently across {len(frames)} frames.")
            self.last_angle = 0.0
            return []
        self.last_angle = out[0][1]
        return out

    def _frames(self, img_node, pose_key=None):
        """촬영을 색마다 새로 하지 않고 돌려 쓴다.

        부르는 쪽은 한 자리에 서서 아는 색을 전부 물어본다(현재 6색). 색마다
        FRAME_DURATION(0.4초)씩 새로 모으면 그것만 2.4초다. 실측 2026-08-04:
        경유점 한 곳에서 6색을 재는 데 11초가 걸렸다.
        같은 순간의 같은 화면을 보는 것이므로 돌려 써도 결과가 달라지지 않는다.

        **무효화는 시간이 아니라 팔 자세로 한다.** 낡은 프레임이 위험한 이유는
        시간이 지나서가 아니라, 이동 전에 찍은 영상을 이동 후 자세로 변환하면
        좌표가 통째로 어긋나기 때문이다(실측: 윗면 z 가 +36mm 튀고 250mm 떨어진
        블록이 잡혔다). 그래서 팔이 그대로인지를 직접 보는 것이 정확하다.

        예전에는 FRAME_CACHE_SEC(0.4초) 으로 막았는데, 6색을 덮으려면 1.6초는
        필요해서 캐시가 매번 만료됐다 — 안전은 얻고 속도는 잃은 구조였다.
        자세로 막으면 한 자리에서는 촬영 1회를 공유하고, 팔이 움직이는 순간
        바로 버린다. 경유점 한 곳이 약 6초 → 4초로 줄어든다.

        pose_key 를 못 받으면(자세를 모르면) 예전처럼 시간으로 막는다 —
        모르는 채로 재사용하는 것보다 안전하다.
        """
        now = time.time()
        if self._cache:
            t0, frames, depth, key0 = self._cache
            if pose_key is not None and key0 is not None:
                # 팔이 그대로면 장면도 그대로다. 다만 조명·물체가 바뀔 수 있으니
                # 상한은 둔다 (사람이 손으로 블록을 옮기는 경우).
                if key0 == pose_key and now - t0 < FRAME_CACHE_MAX_SEC:
                    return frames, depth
            elif now - t0 < FRAME_CACHE_SEC:
                return frames, depth
        frames = collect_frames(img_node, FRAME_DURATION)
        depth = img_node.get_depth_frame()
        self._cache = (now, frames, depth, pose_key)
        return frames, depth

    def get_best_detection(self, img_node, target, pose_key=None):
        """가장 믿을 만한 것 하나. 예전 호출부를 그대로 두기 위해 남긴다."""
        out = self.get_all_detections(img_node, target, pose_key)
        if not out:
            return None, None
        box, ang, hit = out[0]
        self.last_angle = ang
        return box, hit
