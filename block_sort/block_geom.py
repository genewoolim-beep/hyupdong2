#!/usr/bin/env python3
"""블록 자세 기하 — 로봇도 ROS 도 필요 없는 부분만 모았다.

block_sort.py 는 import 하는 순간 DSR 에 붙기 때문에 로봇 없이는 한 줄도
확인할 수 없다. 그런데 놓는 각도를 정하는 계산은 순수 기하라서 책상에서
검증할 수 있는 것들이다 — 여기 옮겨 두면 test_copy_angle.py 가 **실제로 쓰는
함수**를 그대로 시험할 수 있다. 손목 회전은 실기에서 틀리면 블록을 모서리로
물거나 옆 블록을 치는 쪽이라, 미리 확인할 값이 있으면 확인하는 게 낫다.
"""
from scipy.spatial.transform import Rotation


def fold90(deg):
    """정사각 블록의 기울기를 -45~+45 로 접는다.

    윗면이 90° 대칭이라 0°, 90°, 180° 는 눈으로 구별되지 않는다. 접어 두면
    손목 회전량이 항상 최소가 된다(45° 를 넘게 돌 일이 없다).
    파지(_detect_all)와 복제 놓기(place)가 **같은 접기**를 써야 한다 — 다르면
    같은 블록을 집을 때와 놓을 때 기준이 갈라진다.
    """
    return ((float(deg) + 45.0) % 90.0) - 45.0


def rotate_tool(posx, deg):
    """공구 자신의 z 축 둘레로 deg 만큼 돌린 자세를 만든다.

    위치는 그대로 두고 자세만 바꾼다. 공구 z 는 아래를 보고 있으므로(ry≈180)
    공구 기준 +회전은 base 기준으로는 반대 방향이 된다 — 하지만 파지와 놓기가
    **같은 함수를 같은 방향으로** 쓰는 한 그 부호는 서로 지워진다.
    그게 검출각(이미지 기준)을 그대로 놓기에 쓸 수 있는 이유다.
    """
    R = Rotation.from_euler("ZYZ", posx[3:], degrees=True).as_matrix()
    Rn = R @ Rotation.from_euler("z", deg, degrees=True).as_matrix()
    e = Rotation.from_matrix(Rn).as_euler("ZYZ", degrees=True)
    return list(posx[:3]) + list(e)


def tool_yaw(att):
    """공구 x 축이 base 에서 향하는 방위(도). 블록 면의 방향이 이것으로 정해진다.

    진단·검증용이다. 놓기 계산은 이 값을 쓰지 않는다 — 교시 자세들의 이 값이
    90° 접었을 때 서로 같다는 사실에 기대고 있을 뿐이다(place 주석 참고).
    """
    import numpy as np
    R = Rotation.from_euler("ZYZ", att, degrees=True).as_matrix()
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


# ── 옆 블록 피하기 ────────────────────────────────────────────────
# 그리퍼는 두 손가락이 **한 축의 양쪽에서** 내려와 안쪽으로 닫힌다. 그 축에 다른
# 블록이 얹혀 있으면 손가락이 그것을 치거나 밀어낸다(실측: 파지 중 오류로 정지).
#
# 정사각 블록은 90° 대칭이라 **손목을 90° 돌려도 파지 품질이 같다** — 물리는 면이
# 옆면에서 옆면으로 바뀔 뿐이다. 그래서 두 축을 견주어 여유가 넓은 쪽을 고르면
# 공짜로 얻는다. 이 판단에 필요한 것은 옆 블록들의 중심 좌표뿐이다.

BLOCK_MM = 35.0          # 블록 한 변
# **손가락 한 개**의 두께(mm) + 약간의 여유. 양쪽이 아니다.
# 틈은 면 대 면으로 재고(중심거리 - 한 변), 그 틈 하나에 들어가는 것은 손가락
# 하나다 — 반대쪽 손가락은 반대쪽 틈으로 내려간다. 양쪽을 함께 보는 일은
# approach_gap 의 min 이 한다(더 좁은 쪽을 돌려주므로, 그 값이 한 개 두께보다
# 크면 양쪽 다 들어간다는 뜻이 된다).
# 실제로 재야 하는 것은 **틈에 들어가는 부분(손가락 끝)의 두께**다.
#
# 실측 25mm (2026-08-07, RG2 손가락 끝) + 여유 2mm. 딱 25 로 두면 스치면서
# 들어가므로 여유를 얹었다. 스치는 게 보이면 더 키운다.
# **이 값이 곧 필요한 블록 간격이다** — 중심거리로는 35 + 27 = 62mm 이상.
FINGER_T = 27.0
# 손가락 패드가 축과 나란히 덮는 폭의 절반. 이보다 옆으로 비켜 있는 블록은
# 손가락이 지나가도 안 닿는다.
LANE_HALF = 29.5         # 블록 절반(17.5) + 패드 절반 길이(12)


def approach_gap(target_xy, axis_deg, neighbors, block_mm=BLOCK_MM,
                 lane_half=LANE_HALF):
    """그 축으로 손가락을 내릴 때 **가장 좁은 틈**(mm). 방해가 없으면 큰 값.

    축 방향으로 재고, 축에서 옆으로 lane_half 넘게 비킨 블록은 무시한다 —
    손가락이 그 옆을 지나가므로 닿지 않는다.
    틈은 두 블록 **표면 사이** 거리다(중심거리 - 한 변). 그 틈에 들어가는 것은
    **손가락 하나**이므로 FINGER_T(한 개 두께)와 견준다 — 반대쪽 손가락은
    반대쪽 틈으로 내려간다. 양쪽을 함께 보는 것은 아래 min 이다.
    """
    import numpy as np
    r = np.radians(float(axis_deg))
    ax = np.array([np.cos(r), np.sin(r)])          # 손가락이 닫히는 축
    perp = np.array([-ax[1], ax[0]])
    best = float("inf")
    for nb in neighbors:
        d = np.array([float(nb[0]) - target_xy[0], float(nb[1]) - target_xy[1]])
        if abs(float(perp @ d)) >= lane_half:
            continue                               # 옆으로 비켜 있다
        gap = abs(float(ax @ d)) - block_mm         # 표면 사이 거리
        best = min(best, gap)
    return best


def best_axis(target_xy, axis_deg, neighbors, finger_t=FINGER_T, **kw):
    """축을 그대로 둘지 90° 돌릴지 고른다. (돌릴 각도, 그 축의 틈, 원래 축의 틈)

    돌릴 각도는 0 또는 90 이다. **같으면 돌리지 않는다** — 이유 없는 회전은
    파지 보정(grasp_offset)만 흔든다.
    finger_t 보다 넓은 쪽을 고르고, 둘 다 좁으면 그래도 넓은 쪽을 준다
    (부르는 쪽이 경고한다).
    """
    g0 = approach_gap(target_xy, axis_deg, neighbors, **kw)
    g90 = approach_gap(target_xy, axis_deg + 90.0, neighbors, **kw)
    if g0 >= finger_t:
        return 0.0, g0, g0                         # 지금도 넉넉하다
    if g90 > g0:
        return 90.0, g90, g0
    return 0.0, g0, g0
