########## ColorModel ##########
# YOLO 없이, 색깔만으로 물체 하나를 찾는다.
import cv2
import numpy as np
import rclpy

# OpenCV의 Hue는 0~179 범위. 채도(S)·명도(V) 하한을 둬서 회색/흰색/검정/나무 배경을 제외한다.
# 배경(나무판·체스보드)이 낮은 채도로도 걸리는 걸 막기 위해 기준을 다소 엄격하게 잡는다.
_S_MIN, _V_MIN = 100, 60

COLOR_HSV_RANGES = {
    "빨간색": [((0, _S_MIN, _V_MIN), (8, 255, 255)), ((172, _S_MIN, _V_MIN), (179, 255, 255))],
    "red": [((0, _S_MIN, _V_MIN), (8, 255, 255)), ((172, _S_MIN, _V_MIN), (179, 255, 255))],
    "주황색": [((9, _S_MIN, _V_MIN), (20, 255, 255))],
    "orange": [((9, _S_MIN, _V_MIN), (20, 255, 255))],
    "노란색": [((21, _S_MIN, _V_MIN), (33, 255, 255))],
    "yellow": [((21, _S_MIN, _V_MIN), (33, 255, 255))],
    "초록색": [((34, _S_MIN, _V_MIN), (85, 255, 255))],
    "green": [((34, _S_MIN, _V_MIN), (85, 255, 255))],
    "파란색": [((86, _S_MIN, _V_MIN), (130, 255, 255))],
    "blue": [((86, _S_MIN, _V_MIN), (130, 255, 255))],
    # 보라색: hsv_probe로 실측해보니 이 블록은 조명 때문에 hue가 파란색과 같은 대역(86~130)에
    # 찍히고, 채도만 파란 블록보다 낮다. 그래서 hue가 아니라 "파란색 hue대 + 낮은 채도"로 구분한다.
    # V 상한을 둬서 흰색/회색 물체의 밝은 반사광(하이라이트)까지 걸리는 걸 막는다.
    "보라색": [((86, 45, 40), (130, 95, 200))],
    "purple": [((86, 45, 40), (130, 95, 200))],
}

MIN_AREA = 800  # 이보다 작은 뭉치는 노이즈로 보고 무시한다
MAX_AREA_RATIO = 0.35  # 화면의 이 비율보다 큰 뭉치는 배경 오탐으로 보고 무시한다
MAX_ASPECT_RATIO = 2.2  # 블록은 위에서 보면 거의 정사각형이라, 가로/세로 비율이 이보다 길쭉하면
                         # 옆에 붙어있는 다른 블록까지 합쳐진 것으로 보고 제외한다


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


class ColorModel:
    def get_best_detection(self, img_node, target):
        key = target.strip().lower()
        ranges = COLOR_HSV_RANGES.get(key) or COLOR_HSV_RANGES.get(target.strip())
        if ranges is None:
            print(f"'{target}' is not a known color. known: {sorted(set(COLOR_HSV_RANGES))}")
            return None, None

        frame = self._wait_for_frame(img_node)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

        # 작은 점 노이즈는 지운다 (open). 닫기(close)는 너무 크게 하면 옆에 붙은
        # 다른 블록까지 하나로 합쳐버리므로 작은 커널만 살짝 적용한다.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = hsv.shape[0] * hsv.shape[1]
        max_area = frame_area * MAX_AREA_RATIO

        # 배경 오탐과 다른 블록과 합쳐진 뭉치를 걸러내고, 남은 것 중 가장 큰 걸 고른다
        candidates = block_candidates(contours, max_area)
        if not candidates:
            print(f"No plausible-sized '{target}' colored object found.")
            return None, None

        best = max(candidates, key=cv2.contourArea)
        area = cv2.contourArea(best)

        x, y, w, h = cv2.boundingRect(best)
        box = [float(x), float(y), float(x + w), float(y + h)]
        score = float(area) / frame_area
        return box, score

    def _wait_for_frame(self, img_node):
        frame = img_node.get_color_frame()
        while frame is None:
            rclpy.spin_once(img_node)
            frame = img_node.get_color_frame()
        return frame
