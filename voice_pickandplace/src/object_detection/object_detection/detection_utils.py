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


def consensus_boxes(items, frame_count, iou_threshold, min_hit_ratio):
    """여러 프레임에서 나온 (박스, 각도, 무게중심x, 무게중심y) 들을 겹치는 것끼리
    묶어, 일관되게 나타난 무리를 **전부** 돌려준다.
    [(박스, 각도, 신뢰도, 중심x, 중심y), ...]

    consensus_box() 는 가장 큰 무리 하나만 골랐다. 같은 색 블록이 여럿이면
    나머지가 통째로 버려져서, 부르는 쪽이 '어느 것'인지 고를 수가 없었다.
    실측 2026-08-04: 인간구역의 노란 블록을 집으려는데 165mm 떨어진 다른
    노란 블록이 더 크게 잡혀 그쪽이 돌아왔다. 위치로 고르려면 후보가 다 필요하다.

    신뢰도가 높은(여러 프레임에 걸쳐 꾸준히 보인) 순으로 정렬해 돌려주므로,
    첫 번째만 쓰면 예전 consensus_box() 와 같은 결과가 된다.
    """
    if not items or frame_count == 0:
        return []

    groups = []
    for it in items:
        for g in groups:
            if iou(it[0], g[0][0]) >= iou_threshold:
                g.append(it)
                break
        else:
            groups.append([it])

    out = []
    for g in groups:
        hit = len(g) / frame_count
        if hit < min_hit_ratio:
            continue
        mean_box = np.mean([b for b, _, _, _ in g], axis=0).tolist()
        # 각도는 프레임마다 흔들리므로 중앙값을 쓴다 (평균은 0/90 경계에서 튄다).
        ang = float(np.median([a for _, a, _, _ in g]))
        # 무게중심도 중앙값으로. 한 프레임이 튀어도 끌려가지 않는다.
        cx = float(np.median([x for _, _, x, _ in g]))
        cy = float(np.median([y for _, _, _, y in g]))
        out.append((mean_box, ang, hit, cx, cy))
    out.sort(key=lambda t: -t[2])
    return out


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
