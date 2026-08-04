########## 검출기 공용 유틸 ##########
# 색깔 검출과 YOLO 가 함께 쓰는 계산들. 두 벌로 나눠 두면 한쪽만 고쳐져서 따로 놀기 쉬우므로
# 여기 한 곳에만 둔다.
import time

import numpy as np
import rclpy


def iou(box1, box2):
    """두 박스 [x1, y1, x2, y2] 가 얼마나 겹치는지 (0 = 안 겹침, 1 = 완전히 같음).

    같은 물체를 가리키는 박스인지 판단하거나, 서로 다른 색의 박스가
    같은 자리를 물고 있는지(= HSV 범위가 겹쳤는지) 확인할 때 쓴다.
    """
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def collect_frames(img_node, duration, max_frames=30):
    """duration 초 동안 서로 다른 컬러 프레임을 모은다.

    카메라가 새 프레임을 올리기 전에 여러 번 읽으면 같은 그림을 중복해서 세게 되므로,
    촬영 시각을 키로 써서 한 번씩만 담는다.
    """
    deadline = time.time() + duration
    frames = {}
    while time.time() < deadline and len(frames) < max_frames:
        rclpy.spin_once(img_node, timeout_sec=0.05)
        frame = img_node.get_color_frame()
        stamp = img_node.get_color_frame_stamp()
        if frame is not None and stamp is not None:
            frames[stamp] = frame
    return list(frames.values())


def consensus_box(boxes, frame_count, iou_threshold, min_hit_ratio):
    """여러 프레임에서 나온 박스들 중 가장 일관되게 나타난 하나로 합친다.

    겹치는 박스끼리 묶은 뒤 가장 큰 무리를 고르고, 그 좌표를 평균낸다.
    한 프레임에만 반짝 나타난 것은 그림자나 반사광일 가능성이 높아 걸러진다.

    반환하는 점수는 '전체 프레임 중 몇 번 보였는가' 의 비율이다.
    덩어리 크기와 달리 이 값은 실제로 얼마나 믿을 만한지를 뜻한다.
    """
    if not boxes or frame_count == 0:
        return None, None

    groups = []
    for box in boxes:
        for group in groups:
            if iou(box, group[0]) >= iou_threshold:
                group.append(box)
                break
        else:
            groups.append([box])

    best = max(groups, key=len)
    hit_ratio = len(best) / frame_count
    if hit_ratio < min_hit_ratio:
        return None, None

    return np.mean(best, axis=0).tolist(), hit_ratio
