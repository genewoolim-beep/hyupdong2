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
