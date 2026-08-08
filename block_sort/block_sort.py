#!/usr/bin/env python3
"""
관측 → 검출 → 파지 → 구역 배치   (M0609 + RG2 + RealSense)

  python3 block_sort.py observe            관측 자세로 가서 무엇이 보이는지만 보고
  python3 block_sort.py pick <색> <구역>     그 색 블록을 집어 구역에 놓기
  python3 block_sort.py pick <구역> <구역>   앞 구역에 있는 것을 뒤 구역으로 (색 무관)
  python3 block_sort.py run                텍스트로 반복 입력
  python3 block_sort.py scan               경유점을 한 바퀴 돌며 무엇이 있는지만 읽기
  python3 block_sort.py copy               인간구역 배치를 로봇구역에 그대로 복제
  python3 block_sort.py copy-mirror        좌우대칭으로 복제
  python3 block_sort.py sign               수어로 명령 → LLM 해석 → 구역 배치
  python3 block_sort.py home               초기 자세 복귀

기존 음성 pick&place 파이프라인을 그대로 쓴다. 바뀐 것은 두 가지뿐이다.
  · 놓는 자리가 BUCKET_POS 한 곳 → zones.yaml 의 구역 1~4
  · 입력이 음성(/get_keyword) → 텍스트 인자 또는 수어(sign 모드)

sign 모드는 sign_control 에서 학습한 제스처 분류기로 글로스를 읽고, 그 나열을
LLM 이 (색, 구역) 목록으로 바꾼다. 해석 부분만 따로 확인하려면 로봇 없이
`python3 sign_command.py parse "빨강 3번구역 놓다"` 를 쓴다.

sign 모드에서 '모드변경' 을 서명하면 **손동작 조종(제어모드)** 으로 넘어간다.
같은 프로세스 안에서 돈다 — DSR 연결을 하나로 두려는 것이다(teleop_mode.py).

블록 전용 YOLO 모델이 아직 없으므로, 지금은 과일/공구 클래스를 대역으로
써서 파이프라인을 검증한다. 모델이 생기면 대상 이름만 바꾸면 된다.
"""
import json
import os
import sys
import time

import numpy as np
import rclpy
import yaml
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation

import DR_init
from od_msg.srv import SrvDepthPosition

# 순수 기하(손목 회전·기울기 접기)는 여기 두지 않는다. block_sort 는 import 하는
# 순간 DSR 에 붙어서 로봇 없이는 시험할 수 없기 때문이다 — 책상에서 확인할 수
# 있는 계산은 block_geom 에 모아 test_copy_angle.py 가 그대로 시험한다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_geom import (BLOCK_MM, LANE_HALF, RELOCATE_CLEAR_MIN,
                        RELOCATE_WIDE_R, SLIDE_MM, axis_blockers, best_axis,
                        fold90, fold_turn, lane_offsets, pick_order,
                        slide_dests,
                        relocate_candidates, reorder_for_conflicts, rotate_tool,
                        tool_yaw)   # noqa: E402

ROBOT_ID, ROBOT_MODEL = "dsr01", "m0609"

# 첫 실주행 때 50/80 으로 낮춰 뒀다가, 여러 번 무사히 돈 뒤 원래 값으로 되돌렸다.
# 경유점당 7~8.5초 중 이동이 4~5초였다(실측 2026-08-04). 느리면 환경변수로 낮춘다.
VEL_J = float(os.environ.get("VEL_J", 120))    # 관절 deg/s
ACC_J = float(os.environ.get("ACC_J", 120))
VEL_L = float(os.environ.get("VEL_L", 300))    # 직선 mm/s
ACC_L = float(os.environ.get("ACC_L", 300))

JREADY = [0, 0, 90, 0, 90, 0]
GRASP_RETRY = 2

# 파지 높이는 깊이 센서가 아니라 티칭값을 쓴다.
#   판이 평평하고(구역 z 편차 0.5mm) 블록 높이가 일정하므로 z 는 상수다.
#   깊이 측정은 편차가 3mm 나고, 게다가 카메라가 보는 건 '윗면'이라
#   그대로 쓰면 블록 위 허공을 물게 된다.
# 실측: 블록 윗면 base z ≈ +3.9, 티칭 구역 z ≈ -13.0 → 차이 16.9mm
#       35mm 블록의 정확히 중앙이다.
TOP_TO_GRASP = 16.9            # 윗면에서 이만큼 내려가면 파지 높이
Z_SANITY = 15.0                # 측정 윗면이 예상에서 이 이상 벗어나면 경고
REFINE_MAX = 40.0              # 보정량이 이보다 크면 다른 블록으로 보고 거부
# 카메라를 블록 바로 위로 옮겨 다시 보는 일을 수렴할 때까지 되풀이한다.
# 옮길 목표를 '오차가 있는 직전 결과' 로 잡으므로 한 번에 광축 위에 서지
# 못한다. 실측에서 1회 후 6~18mm 가 남았다. 회를 거듭하면 그 잔차가 줄어든다.
# 실측: 1회 보정으로 광축 이탈 212mm → 10mm. 그 뒤로는 더 나아지지 않고
# 오히려 흔들린다(10 → 15.6mm). 남은 잔차는 시점 왜곡이 아니라 검출 잡음이라
# 반복해도 줄지 않는다. 그래서 '나아질 때만' 한 번 더 가고, 아니면 멈춘다.
REFINE_ITERS = 3               # 최대 반복 횟수
REFINE_TOL = 12.0              # 광축에서 이 안(mm)이면 충분히 수직으로 본 것
# 수렴 뒤에도 4mm 안팎이 남는다. 손목 각도(0/45/90°)와 무관하게 같은 크기라
# 계통 오차가 아니라 검출 재현성의 바닥이다 — 깊이가 369↔394 로 흔들리고
# 마스크 경계가 조명에 따라 미세하게 달라지는 것이 누적된 값이다.
# 잡음이므로 여러 번 재서 중앙값을 쓰면 √N 만큼 줄어든다.
DETECT_SAMPLES = int(os.environ.get("DETECT_SAMPLES", 3))   # 최종 자세에서 재는 횟수

# 파지 중심 보정 (mm, base 기준). 핸드아이 잔차와 TCP 정의 차이 때문에
# 계산 위치와 실제 손가락 중심이 조금 어긋난다. calib 모드로 실측해 채운다.
#   측정법: 집었다가 같은 자리에 도로 놓고 다시 검출한다.
#           파지 중심이 e 만큼 밀려 있으면 블록은 P+e 에 놓이므로 e 를 얻는다.
CALIB_OUTLIER = 30.0           # calib 에서 이보다 큰 오차는 오검출로 보고 버린다
# 파지 중심 보정은 두 성분으로 나뉜다.  offset(θ) = OFFSET_BASE + R(θ)·OFFSET_TOOL
#   OFFSET_BASE  비전 사슬(내부파라미터·깊이·핸드아이)에서 오는 것. 손목과 무관.
#   OFFSET_TOOL  TCP 와 실제 손가락 중심의 차이. 손목과 함께 돈다.
# 한 각도에서만 재면 둘을 못 가른다. center 모드로 0° 와 90° 두 번 재서 분리했다.
#   실측 2026-08-04   0° (4.70, 25.70)   90° (4.70, 13.70)
# 고정값 하나만 쓰면 45°/135° 에서 6.5mm, 0° 에서 12mm 어긋난다.
OFFSET_BASE = [10.70, 19.70]
OFFSET_TOOL = [-6.00, 6.00]

# ── 6번 축이 특정 각도일 때만 더하는 파지 보정 ──────────────────────
# OFFSET_BASE/OFFSET_TOOL 로도 안 잡히는, **마지막 관절 각도에 딸린** 치우침을
# 메우는 자리다. 손목이 그 각도로 서면 그리퍼가 블록 중심에서 한쪽으로 밀려
# 물리는 현상이 있어, 그때만 밀어 준다.
#
# **방향은 공구(그리퍼) 기준이다.** 손목이 돌면 보정도 같이 돈다 — 그리퍼가
# 틀어진 만큼 밀어야 할 방향도 같이 틀어지기 때문이다. base 고정으로 두면
# 손목 각도가 달라질 때마다 엉뚱한 쪽으로 밀린다.
# J6_TWEAK_FRAME="base" 로 두면 예전처럼 base 고정 방향이 된다.
#
# J6_TWEAK_AT     이 6번 축 각도(도)에서 적용
# J6_TWEAK_W      ±이 안이면 적용, 벗어나면 0 (계단식 — 중간값을 주지 않는다)
# J6_TWEAK_XY     그때 더할 양 (mm). FRAME 이 tool 이면 **공구 x,y** 기준.
# J6_TWEAK_FRAME  "tool"(기본) 또는 "base"
#
# 실측 2026-08-08: 손목각과 6번 축 사이에 팔 자세에 따라 -1~+11° 오프셋이 있다.
#   손목  -1° -> 6번 축 +10.2°     손목  +4° -> 6번 축 +14.8°    <- 둘 다 "세로"
#   손목 +80° -> 6번 축 +85.8°     손목 +90° -> 6번 축 +100.7°   <- "가로"
# 세로 무리와 가로 무리 사이가 70° 넘게 벌어져 있어 +-40 으로 키워도 가로를
# 잘못 잡지 않는다.
#
# **툴 기준 오른쪽은 +y 다 (2026-08-08 실측).** 처음에 축 이름만 보고
# tool(-15,0) 을 넣었다가 옆 블록을 쳤다 — 그 자세(공구방위 +170°)에서
# tool -x 는 오른쪽이 아니라 **앞**이었다:
#   tool +x -> base (-14.8,  +2.6)  뒤
#   tool -x -> base (+14.8,  -2.6)  앞      <- 잘못 넣었던 것
#   tool +y -> base ( -2.6, -14.8)  오른쪽   <- 이것이 맞다
#   tool -y -> base ( +2.6, +14.8)  왼쪽
# 그때 배치에서 보라색까지 TCP 거리:
#   보정 없이 24.6mm(충돌) / tool(-15,0) 27.3mm(충돌) / tool(0,+15) 39.6mm(안전)
#
# 0,0 으로 두면 이 기능이 꺼진다.
J6_TWEAK_AT = float(os.environ.get("J6_TWEAK_AT", 0.0))
J6_TWEAK_W = float(os.environ.get("J6_TWEAK_W", 40.0))
J6_TWEAK_XY = [float(v) for v in
               os.environ.get("J6_TWEAK_XY", "0,15").replace(" ", "").split(",")]
J6_TWEAK_FRAME = os.environ.get("J6_TWEAK_FRAME", "tool")


def j6_tweak(pose):
    """그 자세로 파지할 때 6번 축이 기준 각도 +-여유 안이면 더할 보정.

    돌려주는 것: (base 기준 보정 np.array, 6번 축 각도 또는 None)

    **움직이기 전에 판단한다.** ikin 으로 목표 자세의 관절값을 미리 풀어 6번
    축을 본다. 보정을 처음부터 좌표에 얹은 채로 한 번에 내려가므로 추가 이동이
    없고, 접근/하강 경로도 그대로다.

    **공구 기준 보정을 base 로 돌려서 돌려준다.** 좌표는 base 로 더해야 하므로
    여기서 한 번에 변환한다. 회전량은 그 자세의 공구 방위(tool_yaw)다 —
    OFFSET_TOOL 이 손목과 함께 도는 것과 같은 기준이다.

    각도 차이는 +-180 으로 접어서 잰다 — 0도와 360도는 같은 자세다.
    180도는 그리퍼 물림 방향은 같지만 **적용하지 않는다.**

    ikin 이 실패하면 보정 없이 간다 — 이것 때문에 파지가 죽으면 안 된다.
    """
    if not any(J6_TWEAK_XY):
        return np.zeros(2), None
    try:
        sol = get_current_posx()[1]
        j6 = float(ikin(list(pose), sol)[5])
    except Exception as e:
        print(f"6번 축을 미리 못 풀어 보정을 건너뜁니다({e})")
        return np.zeros(2), None
    d = (j6 - J6_TWEAK_AT + 180.0) % 360.0 - 180.0
    if abs(d) > J6_TWEAK_W:
        return np.zeros(2), j6
    v = np.array(J6_TWEAK_XY, dtype=float)
    if J6_TWEAK_FRAME == "tool":
        r = np.radians(tool_yaw(pose[3:]))
        R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        v = R @ v
    return v, j6


# 기본 파지 방향. 관측 자세 그대로면 손가락이 base x 로 닫힌다.
# 90 을 주면 base y 로 닫힌다.
GRASP_ROT = 90.0

# 복제(똑같이 / 좌우대칭)에서 **본보기의 기울기까지 따라 놓을지**.
# 다른 명령(색·구역 지정 pick)에는 걸리지 않는다 — 그쪽은 따라할 본보기가 없다.
# 놓을 때 손목을 그만큼 돌리는 것이라 모션이 바뀐다. 이상하면 0 으로 끈다.
COPY_ANGLE = os.environ.get("COPY_ANGLE", "1") != "0"
# 블록이 기울어져 있으면 그만큼 손목을 더 돌려 '면' 을 물게 한다.
# 45° 돌아간 정사각 블록을 고정 각도로 물면 모서리만 잡혀 조일 때 돌아간다.
# 정사각은 90° 대칭이므로 -45~+45 로 접어 회전량을 최소화한다.
ALIGN_TO_BLOCK = True

# 구역 하나가 차지하는 반경. 블록이 이 안에 있으면 '그 구역에 놓인 것'이고,
# 8개 구역(로봇 4 + 인간 4) 어디에도 안 걸리면 프리구역이다. 프리구역은 따로
# 티칭하지 않는다 — 구역을 뺀 나머지 전부이기 때문이다.
#
# 45mm 인 근거 (2026-08-04 실측 기하):
#   블록 반대각선 절반  35 × 1.414 / 2 = 24.7mm   45도 돌아가 있어도 덮는다
#   놓기 오차          약 10mm                   calib-zones 보정 후
#   검출 오차          약 10mm                   광축 정렬 수렴 후 잔차
# 더 크면 구역 사이 통로를 잡아먹고(70mm 면 12.5mm 만 남는다), 더 작으면
# 구역에 제대로 놓인 블록을 프리로 오판해 도로 집어간다.
# 구역 간격이 150mm 라 45mm 면 통로가 62.5mm 열린다 — 35mm 블록이 넉넉히 지난다.
ZONE_RADIUS = float(os.environ.get("ZONE_RADIUS", 45.0))

# 크게 움직인 뒤 검출하기 전에 쉬는 시간. mwait() 는 명령이 끝난 것만 알려주고
# 팔이 완전히 멎은 것은 보장하지 않는다. 자세한 이유는 goto() 주석 참고.
SETTLE_SEC = float(os.environ.get("SETTLE_SEC", 0.6))

# 검출 서비스 한 번을 기다릴 시간(초). 노드가 죽으면 색마다 이만큼 매달리므로
# 너무 길면 안 된다. 정상 응답은 0.2~0.6초다.
SVC_TIMEOUT = float(os.environ.get("SVC_TIMEOUT", 8.0))
SVC_WAIT = float(os.environ.get("SVC_WAIT", 3.0))    # 서비스가 떠 있는지 볼 시간

# 이동이 실제로 반영됐는지 판정하는 허용 오차(mm). movel() 주석 참고.
MOVE_TOL = float(os.environ.get("MOVE_TOL", 8.0))

# 카메라 프레임이 이 시간(초) 넘게 안 오면 끊긴 것으로 본다.
# 30fps 라 정상이면 항상 1초 안쪽이다. USB 가 빠져도 노드는 살아 있으므로
# 이 판정이 없으면 끊긴 걸 알 방법이 없다 (push_hardware 주석 참고).
CAM_STALE_SEC = float(os.environ.get("CAM_STALE_SEC", 3.0))

# 제어모드에 머무를 수 있는 최대 시간(초). 넘으면 스스로 작업모드로 돌아온다.
# 손이 사라지면 속도 0 이 나가지만(데드맨), 그래도 사람이 안 보는 채로 팔이
# 속도 지령을 받는 상태로 남지 않게 상한을 둔다.
CONTROL_MAX_SEC = float(os.environ.get("CONTROL_MAX_SEC", 600.0))

# 컨트롤러가 명령을 거부했을 때 다시 넣기 전에 기다리는 시간(초).
# 재시도마다 이 값의 배수로 늘린다 — 거부는 큐가 비면 풀리므로 조금 기다리는
# 것이 곧바로 다시 넣는 것보다 빨리 성공한다.
MOVE_RETRY_WAIT = float(os.environ.get("MOVE_RETRY_WAIT", 0.5))
# movel 이 '도착했는지' 를 **위치로** 확인할 때 쓰는 값.
# 상태(STANDBY)만 보면 명령 직후에는 모션이 시작 전이라 그냥 통과해서,
# 멀쩡히 가고 있는 이동을 실패로 판정한다(실측 2026-08-08 의 상승 실패).
MOVE_SETTLE_T = float(os.environ.get("MOVE_SETTLE_T", 6.0))   # 최대 관찰 시간(초)
MOVE_SETTLE_N = int(os.environ.get("MOVE_SETTLE_N", 6))       # 이만큼 연속 안 변하면 멎음

# 블록을 물고 xy 로 옮길 때 올라가는 절대 높이(mm). lift_height(150) 로는
# TCP 가 판에서 137mm 인데, 손가락이 그보다 한참 아래로 내려와 옆 블록을 스친다.
# 실측 2026-08-04: 옮기는 도중 다른 블록을 건드리는 일이 있었다.
TRANSIT_Z = float(os.environ.get("TRANSIT_Z", 250.0))
# 이송 높이(TRANSIT_Z)까지 못 올라갈 때 대신 올릴 높이(mm).
# 판 바깥쪽(x가 큰 곳) 블록은 팔을 뻗은 채 250mm 까지 올리려면 작업영역을
# 벗어나 IK 해가 없다 — 컨트롤러가 조용히 거부한다(실측 2026-08-08:
# [641.4, -11.0, 250.0] 상승이 두 곳에서 각각 3회씩 다 무시됐다).
# 그때 통째로 포기하면 물린 채 판을 대각선으로 지나가 다른 블록을 친다.
# 조금이라도 올리는 편이 낫다.
LIFT_FALLBACK = float(os.environ.get("LIFT_FALLBACK", 50.0))

# 빈 손으로 블록 위에 접근할 때의 최소 높이(mm). 검출을 마친 자리가 이보다
# 높으면 그 높이를 그대로 쓰고, 낮으면 여기까지만 올린다.
# 블록 윗면이 판 위 35mm 이므로 120mm 면 손가락 밑으로 여유가 충분하다.
# 이송(TRANSIT_Z=250)과 달리 물린 블록이 없어 여유가 덜 필요하다.
APPROACH_Z = float(os.environ.get("APPROACH_Z", 120.0))

# 이 거리 안에 잡힌 것은 **자기 자신**으로 본다(mm).
# 35mm 블록 둘은 중심이 35mm 보다 가까울 수 없다 — 그보다 가깝게 잡혔다면 같은
# 블록을 다시 본 것이다(검출 흔들림). 이 값을 10mm 로 두었더니 흔들림이 그것을
# 넘는 순간 자기 자신을 옆 블록으로 세어 **틈이 음수**가 되고, 두 축 다 막힌 것으로
# 보여 멀쩡한 블록을 거부했다(실측 2026-08-07: 초록을 못 집었다).
SELF_R = float(os.environ.get("SELF_R", 30.0))


def reach_box():
    """팔을 보내도 되는 xy 범위 (x0, x1, y0, y1). 모르면 None.

    이웃을 치울 자리가 **판 밖이면 안 된다.** 그런데 block_sort 에는 판 경계가
    없었다 — 구역 좌표만 있고, 그 바깥이 어디까지인지는 적힌 데가 없다.
    유일하게 **실측된** 경계는 조종 쪽 교시 상자다(teach_box.py 로 두 모서리를
    찍어 teleop_box.env 에 저장, robot_teleop 이 같은 목적으로 쓴다).
    같은 로봇·같은 판이므로 그것을 빌린다 — 새로 교시하면 양쪽이 함께 따라온다.

    없으면 None 을 돌려주고, 부르는 쪽은 **구역 좌표에서 만든 상자**로 물러난다.
    """
    try:
        gd = os.path.join(os.path.dirname(HERE), "hand_gesture_control")
        if gd not in sys.path:
            sys.path.insert(0, gd)
        import robot_teleop as rt
        # **교시 원본**(X_MIN…)을 쓴다. TX_MIN 은 거기서 15mm 물린 값인데, 그
        # 여유는 '속도 지령이 미끄러지는 것' 을 위한 조종 쪽 몫이다. 여기서 쓰면
        # 판 위 구역까지 밖으로 판정된다 — 실측: 인간구역 y −229 가 TY_MIN(−216)
        # 밖이다. 원본(−231)은 구역 전부를 덮는다.
        return rt.X_MIN, rt.X_MAX, rt.Y_MIN, rt.Y_MAX
    except Exception:
        return None

# 옆 블록을 피하려고 손목을 돌릴지 볼 때, 이 반경 안의 블록만 본다(mm).
# 더 멀면 손가락 근처에 오지 않는다 — 개구 절반(약 55) + 블록 한 변이면 넉넉하다.
NEIGHBOR_R = float(os.environ.get("NEIGHBOR_R", 95.0))
# 블록 사이 틈에 손가락이 들어가려면 이만큼은 있어야 한다(mm).
# **손가락 한 개** 두께다 — 틈 하나에 손가락 하나가 들어가고, 반대쪽 손가락은
# 반대쪽 틈으로 내려간다. 양쪽을 함께 보는 일은 approach_gap 의 min 이 한다.
# block_geom.FINGER_T 와 같은 뜻이다.
FINGER_GAP_MIN = float(os.environ.get("FINGER_GAP_MIN", 27.0))
# 두 축 다 이보다 좁으면 **집지 않는다.**
# 처음엔 '그래도 넓은 쪽으로 간다' 로 두었는데(포기하면 그 칸을 못 만드니까),
# 실제로는 그대로 옆 블록을 치고 오류로 멎었다(실측 2026-08-07). 칠 것을 알면서
# 내려가는 것보다, 멈추고 사람에게 치우라고 하는 편이 낫다.
# 1 이면 예전처럼 그래도 집는다.
PICK_WHEN_TIGHT = os.environ.get("PICK_WHEN_TIGHT", "0") != "0"
# 0 이면 이 기능을 끈다(예전처럼 검출 각도대로만 물린다).
AVOID_NEIGHBORS = os.environ.get("AVOID_NEIGHBORS", "1") != "0"

# 두 축 다 좁아 손목을 돌려도 못 풀 때, 막고 있는 이웃 블록 하나를 옆으로
# 옮겨 자리를 만들지 여부.
#
# **기본을 켬으로 바꿨다(2026-08-08).** 원래는 '원래 시키지 않은 블록을 새로
# 집는 동작' 이라 실기 검증 전까지 꺼 두었고, run_all.sh 만 켜서 검증했다.
# 그 검증이 끝났다 — 오늘 실기에서 여러 번 성공했다(빨간색을 46mm·170mm
# 밖으로 치우고 그 자리를 열었다). 기본이 꺼져 있으면 **단독 실행**
# (python3 block_sort.py 빨강 1) 에서만 조용히 동작이 달라져서, 같은 판인데
# 띄운 방법에 따라 로봇이 다르게 군다 — 그게 더 헷갈린다.
# 0 으로 두면 예전처럼 사람에게 치워 달라고만 한다.
RELOCATE_BLOCKERS = os.environ.get("RELOCATE_BLOCKERS", "1") != "0"
# 원래 목표 하나를 위해 이웃을 몇 개까지 옮겨볼지. 옮겨도 다른 이웃이 같은
# 축을 또 막고 있을 수 있어 한 번으로 안 끝날 수 있다 — 그렇다고 무한정
# 옮기면 판 전체를 재배치하게 되므로 상한을 둔다.
# 2 → 4 로 올렸다(2026-08-08). 손가락은 양쪽에서 내려오므로 한 축을 열려면
# 양쪽을 다 치워야 하고(2회), 거기에 검출 오차로 한 번 헛걸음하면 2회로는
# 모자란다 — 실측에서 십자 포위가 2회 상한에 걸려 포기했다.
RELOCATE_MAX_TRIES = int(os.environ.get("RELOCATE_MAX_TRIES", 4))
# 치우려는 이웃 **자신도** 갇혀 있을 때, 그 이웃의 방해물까지 몇 단계나
# 파고들지. 0 이면 예전처럼 '서로 붙은 뭉치' 를 곧장 사람에게 넘긴다.
# 1 이면 한 단계 — 갇힌 이웃을 집기 위해 그 이웃의 방해물을 먼저 치운다.
# **더 올리지 말 것.** 단계마다 로봇이 새로 집어 옮기는 블록이 늘어나고,
# 그만큼 판이 흐트러지고 시간이 든다. 2×2 뭉치 정도는 1 로 풀린다.
RELOCATE_CHAIN = int(os.environ.get("RELOCATE_CHAIN", 1))
# 이 x 를 넘는 블록은 **뒤로 미룬다**(거르지는 않는다). 판 바깥쪽은 주변이
# 비어 있어 "집기 쉬운 것" 으로 뽑히기 쉬운데 정작 팔이 못 닿는다.
#
# **진짜 한계는 블록이 아니라 카메라다.** 블록을 수직으로 내려다보려면 팔이
# 핸드아이 오프셋(+32.6, +60.1)만큼 **더 바깥으로** 나가야 한다:
#   블록 x=589 -> 관측 x=621   성공
#   블록 x=635 -> 관측 x=667   실패
#   블록 x=640 -> 관측 x=672   실패
# 실측 실패 목표: 648, 650, 667, 668 (z 를 170/120 으로 낮춰도 실패)
#
# **넉넉하게 700 으로 둔다 (2026-08-08 결정).** 실패가 확인된 것은 635 부터
# 지만, 그 판단을 미리 잘라 버리면 될 수도 있는 자리까지 포기하게 된다.
# 여기서 미리 막는 대신 **실제로 해보고 안 되면 넘어가는** 쪽을 택했다 —
# 그 안전망은 이미 다 있다:
#   · 실패한 blocker 는 이번 파지에서 다시 고르지 않는다(_failed_blockers)
#   · 후보가 여럿이면 다음 것으로 넘어가고, 그다음 직각 축으로 전환한다
#   · 상승이 안 되면 LIFT_FALLBACK 만큼만 올려서라도 옮긴다
# 그래도 판 바깥쪽은 **뒤로 미룬다** — 안쪽을 먼저 처리해 판이 헐거워진 뒤에
# 시도하면 성공률이 오른다. 거르는 것이 아니라 순서를 뒤로 돌리는 것뿐이다.
#
# 값을 낮추면 그만큼 일찍 포기한다(REACH_X_MAX=600 등).
# 순회 경유점을 바깥에 더 교시하면(teach_zones.py scan) 실제 도달 범위가
# 늘어나 이 값이 추정이 아니라 실측이 된다.
REACH_X_MAX = float(os.environ.get("REACH_X_MAX", 700.0))
# 막는 블록을 **잡은 채로 끌지** 여부. 들었다 놓는 것보다 상승·이송·하강·상승
# 네 번이 빠지고 블록이 공중에 안 뜬다.
#
# **기본을 껐다 (2026-08-08).** 실기에서 충돌했다 — 노란색을 (609,131) 로
# 끌었는데 거기서 5.7mm 떨어진 곳에 빨간 블록이 있었다(순회 때 (605,135) 로
# 봤던 것이다). 끌기 직전 복도 검사에는 그 빨강이 안 잡혔다. 목표에서 102mm
# 라 '옆 블록 살핌(95mm)' 범위 밖이었고, 그 프레임에서 검출도 안 됐다.
#
# **끌기는 검출이 한 번만 놓쳐도 물리적 충돌이 된다.** 판 바닥 높이로 밀기
# 때문에 경로의 물체를 그냥 들이받는다. 들어서 옮기면 TRANSIT_Z 까지 올라갔다
# 수직으로 내려오므로, 못 본 블록이 있어도 최악이 '겹쳐 놓기' 다 — 이 차이가
# 속도 이득보다 크다고 판단했다.
#
# 다시 켜려면 SLIDE_BLOCKERS=1. 켜기 전에 복도 검사를 믿을 수 있게 만들어야
# 한다 — 지금 보이는 것만이 아니라 순회에서 모아 둔 좌표까지 함께 봐야 하고,
# 여유(SLIDE_CLEAR_EXTRA)도 검출 잡음만큼 넉넉해야 한다.
SLIDE_BLOCKERS = os.environ.get("SLIDE_BLOCKERS", "1") != "0"
# 끌고 가는 복도에 요구할 **추가** 여유(mm). 블록 한 변 위에 더 얹는다.
# 5 로 시작했는데 그걸로는 위 충돌을 못 막았다 — 검출이 흔들리는 만큼 키운다.
SLIDE_CLEAR_EXTRA = float(os.environ.get("SLIDE_CLEAR_EXTRA", 5.0))
# 끌고 갈 **길**에 뭐가 있는지 볼지 여부. 기본은 안 본다(2026-08-08 결정) —
# 잡는 데 성공했으면 그냥 끈다. 길 검사가 멀쩡한 방향까지 막아 "끌 길이
# 막혀 있습니다" 로 죽는 일이 잦았다.
#
# **대가:** 끌기는 판 바닥 높이로 미는 동작이라, 길에 있는 블록을 그대로
# 밀고 간다. 실측 2026-08-08 에 그렇게 빨간 블록을 들이받은 적이 있다.
# 길까지 보게 하려면 SLIDE_CHECK_PATH=1.
#
# 길을 안 봐도 **목적지가 목표를 더는 막지 않는지**와 **팔이 닿는지**는
# 여전히 확인한다 — 그건 끌기의 목적 자체와 도달 가능성이라 뺄 수 없다.
SLIDE_CHECK_PATH = os.environ.get("SLIDE_CHECK_PATH", "0") != "0"
# 끌기가 안 될 때 **들어서 임시 자리로** 옮길지 여부. 기본은 끔(2026-08-08 결정).
# 끌기는 잡은 채로 직각 60mm 만 가면 되는데, 들어서 옮기기는 판 위 아무 데나
# (최대 230mm) 던지므로 배치가 크게 흐트러지고 왕복도 길다.
#
# **끄면 대가가 있다.** 끌 길이 양쪽 다 막힌 경우에 물러설 곳이 없어져,
# 다음 후보나 직각 축으로 넘어가고 그것도 안 되면 사람에게 넘긴다.
# 켜려면 LIFT_RELOCATE=1.
#
# '필요한 색을 제 구역으로 곧장 보내는 것'(relocate_to_zone)은 여기 해당하지
# 않는다 — 그건 임시 대피가 아니라 계획 항목을 한 번에 끝내는 것이라 항상 켜져
# 있다. 그때만 들어서 옮긴다.
LIFT_RELOCATE = os.environ.get("LIFT_RELOCATE", "0") != "0"
# 프리구역에 치울 자리가 없을 때 **구역 안에라도** 치울지 여부.
# 원래는 구역이면 무조건 거부했다 — 그런데 꽉 찬 판에서는 그 탓에 치우기가
# 통째로 무산된다(포기하면 원래 블록도 못 집는다). 임시로 치워 두는 자리이니
# 일단 던져 놓고 보는 편이 낫다는 판단이다.
# **대가가 있다.** 인간구역에 떨어지면 사람이 만들어 둔 본보기가 물리적으로
# 망가지고(copy_human 은 이미 읽어 뒀으니 그 판은 끝나지만 다음 판은 다시
# 만들어야 한다), 로봇구역에 떨어지면 그 칸이 '다른 색이 놓여 있음' 으로 잡혀
# 건너뛰어진다. 그래서 프리구역을 **먼저** 다 보고, 없을 때만 여기로 온다.
# 0 으로 두면 예전처럼 구역이면 포기한다.
RELOCATE_ANYWHERE = os.environ.get("RELOCATE_ANYWHERE", "1") != "0"

# 복제(copy_human) 계획에서, 뒤 순번 색이 앞 순번 색을 막고 있으면 순서를
# 앞당길지 여부. 새 로봇 동작을 추가하는 게 아니라 **이미 하기로 한 이동들의
# 순서만 바꾸는 것**이라(relocate_blocker와 달리 위험이 새로 생기지 않는다)
# 기본을 켜 둔다. 0 이면 끈다(예전처럼 인간구역 번호 순서 그대로).
REORDER_FOR_BLOCKING = os.environ.get("REORDER_FOR_BLOCKING", "1") != "0"

# 순회에서 봐 둔 자리와 다시 봤을 때의 자리가 이보다 벌어지면 다른 블록으로 본다.
# 실측 2026-08-04: 한계가 없어 250mm 떨어진 같은 색을 집으러 갔다.
CACHE_TOLERANCE = 70.0

# 구역 지정 명령에서 '그 구역의 블록' 으로 인정할 최대 거리(mm).
# 구역 반경(45)보다 넉넉히 두되, 옆 구역(간격 150)까지 넘어가면 안 된다.
ZONE_PICK_MAX = float(os.environ.get("ZONE_PICK_MAX", 75.0))

# 특정 색만 찾아 다시 돌 때 건너뛸 앞쪽 경유점 수.
#   1번  인간구역 관측용이라 프리 블록이 없다
#   2번  안쪽 로봇구역 점유 확인용이라 프리 블록이 거의 없다
# 둘 다 '집을 블록을 찾는' 목적이 아니므로 재순회에서는 건너뛴다. 다만 전체
# 순회(copy_human)에서는 반드시 지난다 — 2번이 로봇3·4번을 보는 유일한 자리다.
# 경유점을 지우거나 더하면 이 값도 같이 맞춰야 한다. 안 그러면 재순회 범위가
# 통째로 밀린다.
RESCAN_SKIP = int(os.environ.get("RESCAN_SKIP", 2))

# 봐 뒀던 자리에서 블록이 사라졌을 때 다시 도는 범위. **순서가 곧 정책이다.**
#   None → 기본(RESCAN_SKIP 만큼 앞을 건너뛴다). 빠르지만 앞쪽은 못 본다.
#   0    → 경유점 전부. 사람이 앞쪽으로 옮겼을 때 그것까지 잡는다.
# 빠른 것을 먼저 하고, 실패하면 넓게 한 번 더 본다(rescan 참고).
RESCAN_SKIPS = (None, 0)
RESCAN_TRIES = len(RESCAN_SKIPS)

# 로봇구역 배치를 확인할 때 들르는 경유점(0-based). 전체 순회 대신 여기만 본다.
# 실측 시야중심: 2번 경유점[297,166]이 안쪽(로봇3·4), 3번[518,129]이 바깥(로봇1·2).
# 경유점을 바꾸면 이 값도 같이 봐야 한다.
ZONE_SURVEY_POINTS = [int(v) for v in
                      os.environ.get("ZONE_SURVEY_POINTS", "1,2").split(",") if v.strip()]

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP, TOOLCHARGER_PORT = "192.168.1.1", "502"

HERE = os.path.dirname(os.path.abspath(__file__))
ZONES_YAML = os.path.join(HERE, "zones.yaml")
CENTER_YAML = os.path.join(HERE, "center_calib.yaml")   # 손목각별 실측 보정
# 핸드아이 행렬. 저장소의 calib/ 것을 먼저 쓰고, 없으면 환경변수.
# 이 값은 block_sort 와 hand_gesture_control(파지 AR)이 **함께** 읽는다. 그래서
# 어느 한 프로그램 안이 아니라 저장소 공용 자리에 둔다 — 두 벌이 되면 한쪽만
# 재보정했을 때 화면과 실제 파지가 갈라진다.
_HE_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "calib", "T_gripper2camera.npy")
HANDEYE = os.environ.get("HANDEYE", _HE_DEFAULT)
# 검출 노드가 아는 색 목록. 색 없이 집을 때 하나씩 물어보려면 필요하다.
_CR_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sign_pickandplace", "src", "object_detection", "resource", "color_ranges.json")
COLOR_RANGES_JSON = os.environ.get("COLOR_RANGES", _CR_DEFAULT)

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
rclpy.init()
_dsr = rclpy.create_node("block_sort_dsr", namespace=ROBOT_ID)
DR_init.__dsr__node = _dsr

try:
    # speedl 은 제어모드(손동작 조종) 몫이다. 여기서 함께 가져오는 이유는
    # DSR 연결을 하나로 두기 위해서다 — 조종 모듈이 자기 노드를 만들면 한
    # 로봇에 두 연결이 되어 TCP 가 풀리고 모션이 거부된다(teleop_mode 참고).
    from DSR_ROBOT2 import (movej as _movej_raw, movel as _movel_raw, speedl,
                            get_current_posx, ikin, mwait,
                            get_robot_state, STATE_STANDBY,
                            get_current_tool_flange_posx, get_tcp, set_tcp)
except ImportError as e:
    sys.exit(f"DSR_ROBOT2 임포트 실패: {e}\n로봇 드라이버가 떠 있는지 확인하세요.")


# ───────────────────────── signbot_admin 연동 ─────────────────────────
# --admin 을 붙였을 때만 관리자 대시보드로 상태를 쏜다. 기본은 조용히 아무것도
# 안 한다 — 다른 팀원이 --admin 없이 그대로 써도 동작이 달라지지 않는다.
ADMIN_URL = os.environ.get("SIGN_ADMIN_URL", "http://localhost:5000")
USE_ADMIN = "--admin" in sys.argv


def _admin_post(path, payload):
    if not USE_ADMIN:
        return
    import json
    import threading
    import urllib.request

    def _send():
        try:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{ADMIN_URL}{path}", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=1.0)
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()


# ── 작업모드 ↔ 제어모드 ──────────────────────────────────────────────
# 인터페이스가 둘이다. 수어로 블록을 분류하는 것이 '작업모드', 손동작으로
# 팔을 직접 미는 것이 '제어모드'다. **둘 다 이 프로세스가 돈다.**
#
# 전에는 제어모드가 hand_gesture_control.py 라는 별 프로세스였다. 한 로봇에
# DSR 연결이 둘이 되어 TCP 가 0.0mm 로 풀리고 모션이 거부되고 위치 조회가
# 멎었다(실측 2026-08-06). 그래서 조종을 이 안으로 들여왔다 — teleop_mode.py.
# 대시보드의 /api/mode 는 그대로 쓴다. 지금 어느 인터페이스가 활성인지
# 화면에 보여주고, 대시보드에서 작업모드로 되돌리는 길이 되기 때문이다.
def push_mode(mode):
    """지금 활성 인터페이스를 대시보드에 알린다. work | control"""
    _admin_post("/api/mode", {"mode": mode})


_ctrl_frame = {"jpg": None, "lock": None, "started": False}


def control_frame_sink():
    """제어 화면 한 장을 받아 대시보드(/api/frame/control)로 중계하는 콜백.

    --admin 이 아니면 None — 부르는 쪽은 아무것도 안 보낸다.
    작업모드 화면(/api/frame)과 **버퍼가 다르다.** 같은 것을 쓰면 두 화면이
    서로 덮어써 둘 다 깜빡인다 (signbot_admin/app.py 의 주석과 같은 이유).

    보내는 일은 별 스레드가 자기 페이스로 한다. 조종 루프가 전송을 기다리면
    프레임률이 네트워크에 묶이고, 그러면 팔이 손보다 늦게 선다.
    """
    if not USE_ADMIN:
        return None
    import threading
    import urllib.request
    import cv2

    if _ctrl_frame["lock"] is None:
        _ctrl_frame["lock"] = threading.Lock()

    def _sender():
        while True:
            with _ctrl_frame["lock"]:
                data = _ctrl_frame["jpg"]
            if data is not None:
                try:
                    req = urllib.request.Request(
                        f"{ADMIN_URL}/api/frame/control", data=data,
                        headers={"Content-Type": "image/jpeg"}, method="POST")
                    urllib.request.urlopen(req, timeout=1.0)
                except Exception:
                    pass
            time.sleep(0.08)          # 모니터링용이라 12fps 상한이면 충분하다

    if not _ctrl_frame["started"]:    # 제어모드를 여러 번 들락거려도 스레드는 하나
        _ctrl_frame["started"] = True
        threading.Thread(target=_sender, daemon=True).start()

    def _sink(frame):
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            with _ctrl_frame["lock"]:
                _ctrl_frame["jpg"] = buf.tobytes()

    return _sink


_alive_sent_at = [0.0]      # 이 프로세스가 마지막으로 alive 를 보낸 시각


def push_control_alive():
    """제어모드를 지금 이 프로세스가 잡고 있다고 대시보드에 알린다.

    조종이 별 프로세스였을 때 '받아줄 상대가 있는가' 를 보려고 만든 신호다.
    이제 넘길 상대가 없으니 판단에는 안 쓰지만, 계속 보낸다 — 밖에서
    hand_gesture_control.py 를 띄웠을 때 '이미 누가 잡고 있다' 를 볼
    유일한 창구이고, 그쪽도 같은 신호를 쓴다(control_taken 참고).
    """
    _alive_sent_at[0] = time.time()
    _admin_post("/api/control/alive", {})


# 이 이름의 TCP 가 걸려 있어야 한다. 시작할 때 확인하고 다르면 다시 건다.
# 실측 2026-08-06: 두 프로세스가 각자 DSR 에 붙었을 때 TCP 가 빈 값으로 풀려
# 조회가 0.0mm 로 나왔다. 그 상태로 파지하면 좌표가 통째로 어긋난다 —
# 티칭값이 전부 이 TCP 기준이기 때문이다.
EXPECT_TCP = os.environ.get("EXPECT_TCP", "GripperDA_v1")
# 그 TCP 의 실측 길이(mm)와 허용 오차. 이름이 맞아도 길이가 0 이면
# 오프셋이 풀린 것이다 — 실측 2026-08-06 에 그렇게 깨진 적이 있다.
EXPECT_TCP_MM = float(os.environ.get("EXPECT_TCP_MM", 250.0))
TCP_TOL_MM = float(os.environ.get("TCP_TOL_MM", 5.0))


def ensure_tcp(logger=None):
    """걸린 TCP 가 EXPECT_TCP 인지 보고, 아니면 다시 건다.

    빈 문자열이나 다른 이름이면 set_tcp 로 되돌린다. 실패해도 예외는 안 올린다 —
    알리는 것까지가 여기 몫이고, 판단은 사람이 한다.
    """
    def say(msg):
        (logger.warn if logger else print)(msg)

    def ok_now():
        """이름과 **길이**가 둘 다 맞는가. 판정은 tcp_info 한 곳에서 한다.

        이름만 보면 부족하다. 이름이 남아 있어도 실제 오프셋이 풀려 길이가
        0 으로 읽히는 경우가 있다 — 그 상태로 파지하면 좌표가 250mm 어긋난다.
        길이는 TCP 좌표와 플랜지 좌표의 차이로 잰다(tcp_info 참고).
        """
        info = tcp_info()
        return bool(info.get("ok")), info.get("name"), info.get("length_mm")

    try:
        good, name, length = ok_now()
        if good:
            return True
        say(f"TCP 가 '{name}' / 길이 {length}mm 입니다 — "
            f"'{EXPECT_TCP}'({EXPECT_TCP_MM:.0f}mm) 로 다시 겁니다")
        set_tcp(EXPECT_TCP)
        good, name, length = ok_now()
        if good:
            say(f"TCP 복구됨: {EXPECT_TCP}  {length:.1f}mm")
            return True
        say(f"TCP 를 못 걸었습니다 (지금 '{name}' / {length}mm) "
            "— 티치펜던트에서 확인하세요. 이대로 파지하면 좌표가 어긋납니다.")
        return False
    except Exception as e:
        say(f"TCP 확인 실패({e})")
        return False


def tcp_info():
    """지금 걸린 TCP 의 이름과 **플랜지에서의 거리**(mm).

    컨트롤러는 TCP 이름만 돌려주고(GetCurrentTcp) 수치 오프셋을 주는 서비스가
    없다. 대신 같은 순간의 TCP 좌표와 플랜지 좌표를 함께 읽어 그 차이를 재면
    실제로 걸려 있는 값이 나온다 — 설정 파일이 아니라 로봇이 쓰고 있는 값이다.
    """
    try:
        # 반환 형식이 다르다. get_current_posx 는 (posx, 솔루션공간) 튜플이라
        # [0] 이 필요하고, get_current_tool_flange_posx 는 posx 를 바로 준다.
        # [0] 을 붙이면 스칼라를 집어 'invalid index to scalar variable' 이 난다.
        tcp = get_current_posx()[0]
        flange = get_current_tool_flange_posx()
        d = [tcp[i] - flange[i] for i in range(3)]
        length = float(np.linalg.norm(d))
        try:
            name = get_tcp()
        except Exception:
            name = None
        # **기대값과 판정도 함께 보낸다.** 대시보드가 '250mm 인지' 를 스스로 판단하면
        # 문턱이 두 곳에 적히고, 한쪽만 고쳤을 때 화면이 조용히 거짓말을 한다.
        # 판정은 여기 한 곳에서만 한다(ensure_tcp 도 이 값을 쓴다).
        ok = (isinstance(name, str) and name == EXPECT_TCP
              and abs(length - EXPECT_TCP_MM) <= TCP_TOL_MM)
        # error 를 항상 실어 보낸다. 대시보드가 부분 갱신(update)이라, 성공했을 때
        # 이 키를 빼면 지난 실패의 메시지가 그대로 남아 붙어 있는다.
        return {"name": name if isinstance(name, str) else None,
                "length_mm": round(length, 1),
                "offset_mm": [round(v, 1) for v in d],
                "expect": EXPECT_TCP, "expect_mm": EXPECT_TCP_MM,
                "tol_mm": TCP_TOL_MM, "ok": bool(ok),
                "error": None}
    except Exception as e:
        return {"name": None, "length_mm": None, "offset_mm": None,
                "expect": EXPECT_TCP, "expect_mm": EXPECT_TCP_MM,
                "tol_mm": TCP_TOL_MM, "ok": False,
                "error": str(e)}


def control_taken():
    """**다른** 프로세스가 제어모드를 잡고 있는가.

    전에는 이 신호를 '넘길 상대가 있는가' 로 썼다. 조종이 이 안으로 들어온
    뒤로는 뜻이 뒤집혔다 — 밖에 조종 프로세스가 살아 있으면 그쪽도 DSR 에
    붙어 speedl 을 쏜다. 한 로봇에 지령이 둘이면 위험하므로, 그럴 때는
    제어모드로 들어가지 않는다.

    엔드포인트는 마지막 시각 하나만 들고 있어 누가 보냈는지 구별하지 못한다.
    그래서 **이쪽이 최근에 보냈으면 남의 것으로 보지 않는다.** 이 판단이 없으면
    제어모드에서 막 나온 뒤 10초 안에 다시 '모드변경' 을 하면, 방금 이 프로세스가
    남긴 신호를 남의 것으로 읽어 스스로를 막는다.
    """
    if not USE_ADMIN:
        return False
    # 서버가 신선하다고 보는 창(10초)보다 넉넉히. 이 안에 이쪽이 보낸 게 있으면
    # 지금 신선한 신호는 그것일 수 있다 — 남의 것이라고 단정할 근거가 없다.
    if time.time() - _alive_sent_at[0] < 12.0:
        return False
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(f"{ADMIN_URL}/api/control/alive", timeout=1.0) as r:
            return bool(json.loads(r.read().decode("utf-8")).get("alive"))
    except Exception:
        return False


def get_mode(default="work"):
    """대시보드가 아는 현재 모드. --admin 이 아니거나 못 읽으면 default."""
    if not USE_ADMIN:
        return default
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(f"{ADMIN_URL}/api/mode", timeout=1.0) as r:
            return json.loads(r.read().decode("utf-8")).get("mode", default)
    except Exception:
        return default


# 대시보드(app.py 의 COLOR_HEX)는 짧은 이름을 쓰고, 검출 쪽은 color_ranges.json 의
# 긴 이름을 쓴다. 서로 이름이 달라 색상 칩이 안 그려졌다 — COLOR_HEX.get("빨간색")
# 이 None 이라 구역 데이터는 들어갔는데 색만 안 보였다.
# 검출·구역 판정은 긴 이름을 계속 쓰고, 내보낼 때만 여기서 바꾼다.
# '색'만 떼면 빨간색→빨간, 노란색→노란 이 되어 서버가 못 알아본다. 대응표로 적는다.
ADMIN_COLOR = {
    "빨간색": "빨강", "주황색": "주황", "노란색": "노랑",
    "초록색": "초록", "파란색": "파랑", "보라색": "보라",
}


def _short_color(color):
    if not color:
        return color                      # 빈 구역은 None 그대로
    return ADMIN_COLOR.get(color, color)  # 이미 짧은 이름이면 그대로


def push_zone(space, zone_num, color):
    """구역 하나의 색을 signbot_admin에 반영한다. space: robot | human"""
    _admin_post(f"/api/zones/{space}",
                {"zone": f"{zone_num}번구역", "color": _short_color(color)})


def push_zones(space, zone_colors, all_zone_nums):
    """구역 전체를 한 번에 반영한다. zone_colors에 없는 번호는 빈 구역으로 민다."""
    for i in all_zone_nums:
        push_zone(space, i, zone_colors.get(i))


def push_robot_status(**fields):
    _admin_post("/api/robot/status", fields)


def push_debug(level, source, message):
    _admin_post("/api/debug", {"level": level, "source": source, "message": message})


def push_stock(colors, detail=""):
    """어떤 색의 프리 재고가 없는지 대시보드에 띄운다.

    터미널 로그만 남기면 화면만 보는 사람은 '왜 저 칸이 비었지' 를 알 수 없다.
    재고 부족은 고장이 아니라 **사람이 블록을 놓아주면 풀리는 것**이라, 화면에
    나와야 조치로 이어진다. warn 으로 보내 debug 패널에서 눈에 띄게 한다.
    """
    if not colors:
        return
    names = colors if isinstance(colors, str) else ", ".join(colors)
    push_debug("warn", "재고",
               f"프리구역에 {names} 재고가 없습니다" + (f" — {detail}" if detail else ""))


def wait_idle(timeout=20.0):
    """모션이 끝나 STANDBY 가 될 때까지 기다린다.

    mwait() 만으로는 부족하다. 실측 2026-08-04: copy 한 번에
    "A motion is ongoing so new commands are not accepted." 가 27번 났고,
    그때마다 movel 이 **조용히 무시돼** 팔이 그 자리에 머물렀다.
    hover_over 가 목표 위로 못 가 광축이 196mm 어긋난 것도 이 때문이다.
    거부는 예외가 아니라 로그 한 줄이라 코드가 알아챌 방법이 없어서,
    명령을 넣기 전에 상태로 직접 확인한다.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if get_robot_state() == STATE_STANDBY:
                return True
        except Exception:
            pass
        time.sleep(0.05)
    _mlog("warn", f"{timeout}초 동안 STANDBY 가 안 됐습니다 — 그대로 진행합니다")
    return False


def _mlog(level, msg):
    """movel/wait_idle 의 진단을 **ROS 로그에** 남긴다.

    예전에는 print() 였다. 그러면 run_all.sh 를 띄운 터미널에만 뜨고
    ~/.ros/log 에는 안 남아, 나중에 로그만 보고는 "이동이 실패했는지" 를
    알 수가 없다. 실측 2026-08-08: 파지 후 상승(263mm)이 0.39초 만에
    끝난 것처럼 보였는데(VEL_L 300mm/s 로는 1.1초가 걸려야 한다) 실패
    여부를 확인할 방법이 없었다.
    """
    try:
        import rclpy.logging
        getattr(rclpy.logging.get_logger("block_sort"), level)(msg)
    except Exception:
        print(msg)


def _reached(cur, p, tol):
    return (np.hypot(cur[0] - p[0], cur[1] - p[1]) <= tol
            and abs(cur[2] - p[2]) <= tol)


def movel(p, vel=VEL_L, acc=ACC_L, tol=None, tries=3, timeout=20.0):
    """직선 이동 + 도달 확인. 안 갔으면 다시 넣는다.

    "A motion is ongoing so new commands are not accepted." 는 예외가 아니라
    로그 한 줄이라 코드가 알아챌 방법이 없다. 실측 2026-08-05: hover_over 가
    씹혀 팔이 이전 경유점에 머문 채 검출하는 바람에 블록이 광축에서 188mm
    벗어나 보였고, 그 비스듬한 시야에서 잰 각도가 4° 흔들려(64→63→60)
    큐브가 틀어져 물렸다.

    **movel 서비스는 명령을 접수하면 바로 돌아온다.** 모션이 끝난 것이 아니다.
    _async=0 이라 동기인 줄 알았는데, 실측 2026-08-05 로 아니라는 게 드러났다:
    반환값은 51번 모두 0(성공)인데 바로 뒤에 읽은 위치는 목표가 아니었다.
    즉 '거부' 가 아니라 '아직 가는 중' 이었다.

    그래서 모션 완료는 **mwait() 로 기다린다**. movej 가 원래 그렇게 하고 있었고
    movel 만 빠져 있었다. 완료를 기다린 뒤에 위치를 보면 판정이 정확해진다.

    이 함수는 두 번 잘못 고쳤다. 남겨 두는 이유는 같은 길을 또 가지 않기 위해서다.
      · 위치 폴링 — 출발 전과 정지를 구별 못 해 멀쩡한 이동을 포기했다.
      · 반환값만 확인 — 접수 즉시 0 이 오므로 항상 '안 갔다' 로 읽혔고,
        재시도 대기(0.5→1.0→1.5초)가 그대로 파지 전 공중 대기가 됐다.
    """
    p = list(p)
    tol = MOVE_TOL if tol is None else tol
    for k in range(1, tries + 1):
        cur = get_current_posx()[0]
        if _reached(cur, p, tol):
            return True                      # 이미 목표에 있다
        # 앞 모션이 아직 돌고 있으면 이 명령은 버려진다. 미리 막아 둔다.
        wait_idle()
        if _movel_raw(p, vel=vel, acc=acc) != 0:
            _mlog("warn", f"이동 명령이 거부됐습니다 (컨트롤러가 바쁨). 재시도 {k}/{tries}")
            time.sleep(MOVE_RETRY_WAIT)
            continue
        mwait()                              # 모션이 실제로 끝날 때까지
        wait_idle()
        # **상태 보고에 기대지 않고 위치가 멎을 때까지 본다.**
        # mwait()/wait_idle() 은 명령 직후에 부르면 아직 STANDBY 라 그냥
        # 통과한다 — 모션이 시작도 안 했는데 "끝났다" 고 보는 것이다. 그러면
        # 위치를 읽어도 당연히 제자리라 멀쩡한 이동이 실패로 판정된다
        # (실측 2026-08-08: 263mm 상승이 0.13초씩 3번 다 "안 갔음" 이었다).
        # 목표에 닿거나, 위치가 MOVE_SETTLE_N 번 연속 안 변하면 멈춘 것으로 본다.
        t0 = time.time()
        last, still = None, 0
        while time.time() - t0 < MOVE_SETTLE_T:
            cur = get_current_posx()[0]
            if _reached(cur, p, tol):
                return True
            moved = (float(np.hypot(cur[0] - last[0], cur[1] - last[1]))
                     + abs(cur[2] - last[2])) if last is not None else 999.0
            if moved < 0.5:
                still += 1
                if still >= MOVE_SETTLE_N:
                    break                    # 목표는 아닌데 멎었다 — 안 간 것이다
            else:
                still = 0
            last = cur
            time.sleep(0.05)
        cur = get_current_posx()[0]
        if _reached(cur, p, tol):
            return True
        err = float(np.hypot(cur[0] - p[0], cur[1] - p[1]))
        _mlog("warn", f"이동이 끝났는데 목표에서 xy {err:.0f}mm, "
                      f"z {abs(cur[2] - p[2]):.0f}mm 벗어나 있습니다. 재시도 {k}/{tries}")
        # **여기에도 쉬어야 한다.** 예전에는 이 경로에 대기가 없어 0.13초 간격으로
        # 세 번을 몰아쳤고, 컨트롤러가 바쁜 그 짧은 순간을 그대로 세 번 다 맞았다
        # (실측 2026-08-08: 3회 재시도가 통틀어 0.39초 만에 끝났다).
        time.sleep(MOVE_RETRY_WAIT)
    _mlog("error", f"{tries}회 시도했으나 목표에 못 갔습니다 {[round(v, 1) for v in p[:3]]}")
    return False


def movej(q, vel=VEL_J, acc=ACC_J):
    wait_idle()
    _movej_raw(list(q), vel=vel, acc=acc)
    mwait()
    wait_idle()



sys.path.insert(0, HERE)
from onrobot import RG                                    # noqa: E402

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)


def load_zones():
    if not os.path.exists(ZONES_YAML):
        sys.exit(f"{ZONES_YAML} 이 없습니다. teach_zones.py teach 를 먼저 실행하세요.")
    d = yaml.safe_load(open(ZONES_YAML))
    if not d.get("zones"):
        sys.exit("zones.yaml 에 구역이 없습니다.")
    return d


def load_colors():
    """검출 노드와 같은 파일에서 색 이름을 읽는다.

    여기에 목록을 따로 적어두면 color_ranges.json 에 색을 추가했을 때
    '구역에서 집기'만 그 색을 못 보는 일이 생긴다.
    """
    try:
        with open(COLOR_RANGES_JSON, encoding="utf-8") as f:
            return list(json.load(f)["colors"])
    except Exception as e:
        print(f"{COLOR_RANGES_JSON} 을 못 읽어 기본 목록을 씁니다 ({e})")
        return ["빨간색", "주황색", "노란색", "초록색", "파란색", "보라색"]


KNOWN_COLORS = load_colors()


# ry 를 **정확히** 180 으로 두지 않는 이유. ZYZ 에서 ry=180 은 표현의 특이점
# (짐벌락)이다. 회전 자체는 멀쩡하지만 행렬→오일러 역변환이 유일하지 않아,
# scipy 가 `Gimbal lock detected. Setting third angle to zero` 경고와 함께
# rz2 를 0 으로 몰아버린다. 그러면 rotate_tool() 이 만든 파지 자세가
#   ry=180.0 → rz1 -122.79 (지금 46.46 에서 -169.25도!), rz2 0.00
#   ry=179.9 → rz1   46.46 (변화 0.00도),              rz2 169.25
# 가 된다. 둘은 회전행렬이 완전히 같은데도 컨트롤러에 넘어가는 각도가 다르다.
# movel 은 이 각도를 그대로 보간하므로, 180 쪽은 손목을 반 바퀴 돌리라는
# 명령이 되어 조용히 거부된다 — 실측 2026-08-05 에 TRANSIT_Z 상승과
# hover_over 가 통째로 씹힌 원인이 이것이다(재시도 21회, 포기 7회).
# 0.1도 기울면 손가락 60mm 깊이에서 0.1mm 어긋난다. 놓기 오차 10mm 에 비하면
# 없는 값이고, 대신 표현이 잘 정의된 영역에 머문다.
LEVEL_RY = float(os.environ.get("LEVEL_RY", 179.9))


def level_att(att):
    """자세각의 기울기만 펴서 공구가 (거의) 정확히 아래를 보게 한다. 요는 유지.

    ZYZ 에서 ry≈180 이면 공구 z 축이 -z(아래)를 향한다.
    관측 자세가 기울어져 있으면(observe_free 실측 26.5°) 그 기울기가 파지 자세까지
    그대로 딸려와, 손가락이 판에 비스듬히 내려가 모서리를 물게 된다.
    광축도 같이 기울어 x,y 이동이 화면 중심 이동과 어긋난다.

    정확히 180 이 아니라 LEVEL_RY 를 쓰는 이유는 그 상수 주석 참고.
    """
    return [att[0], LEVEL_RY, att[2]]


def wait_gripper():
    while gripper.get_status()[0]:
        time.sleep(0.3)


def grasped():
    """RG2 상태의 1번 비트가 'grip detected'. 빈손이면 0이다."""
    return bool(gripper.get_status()[1])


class BlockSort(Node):
    def __init__(self):
        super().__init__("block_sort")
        self.cfg = load_zones()
        self.lift = self.cfg.get("lift_height", 150)
        self.cli = self.create_client(SrvDepthPosition, "/get_3d_position")
        self.req = SrvDepthPosition.Request()
        # RealSense 살아있는지 판정용. 영상 대신 camera_info 를 받아 마지막 시각만
        # 기록한다 (수백 바이트). 이유는 push_hardware() 주석 참고.
        self._cam_info_t = None
        from sensor_msgs.msg import CameraInfo
        self.create_subscription(
            CameraInfo, "/camera/camera/color/camera_info",
            lambda _m: setattr(self, "_cam_info_t", time.time()), 1)
        self.gripper2cam = np.load(HANDEYE)
        # 블록도 구역과 같은 판 위에 있으므로 파지 높이는 구역 z 와 같다.
        zs = [p[2] for p in self.cfg["zones"].values()]
        self.pick_z = float(np.mean(zs))
        self.expect_top = self.pick_z + TOP_TO_GRASP
        self.last_top = None       # 마지막으로 측정한 블록 윗면 z (진단용)
        self.last_angle = 0.0      # 마지막 검출의 블록 기울기 (도)
        self.last_cam = [0.0, 0.0, 0.0]   # 마지막 검출의 카메라 좌표
        self.last_rot = GRASP_ROT         # 마지막 검출에서 쓴 손목 각도

    # ── 좌표 변환 (기존 robot_control.py 와 동일) ──
    @staticmethod
    def pose_matrix(x, y, z, rx, ry, rz):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T[:3, 3] = [x, y, z]
        return T

    def to_base(self, cam_xyz, robot_posx):
        base2gripper = self.pose_matrix(*robot_posx)
        base2cam = base2gripper @ self.gripper2cam
        return (base2cam @ np.append(np.array(cam_xyz), 1))[:3]

    # ── 검출 ──
    def detect(self, target, refine=True, near=None, max_dist=None):
        """1차로 찾고, 그 위로 카메라를 옮겨 2차로 정밀 측정한다.

        색 임계값은 블록의 윗면과 옆면을 구분하지 못한다. 비스듬히 볼수록
        옆면이 마스크에 붙어 박스가 커지고, 중심이 그 절반만큼 카메라
        반대쪽으로 밀린다. 실측에서 광축 근처 49.5mm, 가장자리 59.6mm 로
        10mm 차이가 났다 — 중심 오차 5mm 다.

        블록 바로 위에서 보면 광축각이 0 에 가까워 옆면이 안 보인다.
        원인 자체가 사라지므로 기하 보정보다 확실하다.
        """
        cur = self._detect_once(target, near=near, max_dist=max_dist)
        if cur is None or not refine:
            return cur

        best, best_off = cur, float(np.hypot(self.last_cam[0], self.last_cam[1]))
        for it in range(1, REFINE_ITERS + 1):
            if best_off <= REFINE_TOL:
                self.get_logger().info(f"광축에서 {best_off:.1f}mm — 수렴")
                break
            posx = get_current_posx()[0]
            # 기울어진 채로 x,y 만 옮기면 광축은 엉뚱한 데를 본다. 실측에서
            # "블록 위로 이동" 뒤에도 광축 이탈이 327mm 그대로 남았다.
            att = level_att(posx[3:])
            R = Rotation.from_euler("ZYZ", att, degrees=True).as_matrix()
            cam_off = R @ self.gripper2cam[:3, 3]    # 카메라가 그리퍼에서 떨어진 양
            view = list(posx[:3]) + att
            view[0] = best[0] - cam_off[0]           # 카메라를 블록 바로 위로
            view[1] = best[1] - cam_off[1]
            self.get_logger().info(
                f"[{it}] 광축 {best_off:.1f}mm — 블록 위로 이동 "
                f"({view[0]:.0f}, {view[1]:.0f})")
            try:
                movel(view)
                # 옮긴 뒤 반드시 쉰다. 이게 없으면 검출 노드가 **이동 전에 찍어둔**
                # 프레임을 돌려주고(FRAME_CACHE_SEC), 그것을 **이동 후** 팔 자세로
                # 변환해 좌표가 통째로 어긋난다. 실측 2026-08-04: 윗면 z 가 예상에서
                # +36mm 튀고 250mm 떨어진 엉뚱한 블록이 잡혔다.
                time.sleep(SETTLE_SEC)
            except Exception as e:
                self.get_logger().warn(f"이동 실패({e}) — 직전 결과 사용")
                break
            # 시점이 바뀌면 다른 블록이 더 크게 보일 수 있다. 직전에 고른
            # 자리를 기준으로 삼아 같은 블록을 계속 따라간다.
            nxt = self._detect_once(target, near=(best[0], best[1]))
            if nxt is None:
                self.get_logger().warn("재검출 실패 — 직전 결과 사용")
                break
            d = float(np.hypot(nxt[0] - best[0], nxt[1] - best[1]))
            if d > REFINE_MAX:
                # 같은 색이 여럿이면 시점이 바뀐 뒤 다른 개체를 잡을 수 있다.
                self.get_logger().warn(
                    f"보정량 {d:.1f}mm 가 한계({REFINE_MAX:.0f})를 넘습니다 — "
                    "다른 블록으로 판단하고 직전 결과 사용")
                break
            off = float(np.hypot(self.last_cam[0], self.last_cam[1]))
            if off >= best_off:
                # 잡음 구간에 들어왔다. 더 가봐야 나빠지기만 한다.
                self.get_logger().info(
                    f"[{it}] 광축 {off:.1f}mm — 나아지지 않아 중단 "
                    f"(최선 {best_off:.1f}mm 유지)")
                break
            self.get_logger().info(f"[{it}] 보정 {d:.1f}mm  광축 {off:.1f}mm")
            best, best_off = nxt, off

        # 수렴한 자세에서 여러 번 재어 잡음을 눌러준다. 이동은 없으므로 싸다.
        if DETECT_SAMPLES > 1:
            xs, ys, angs = [best[0]], [best[1]], [self.last_angle]
            for _ in range(DETECT_SAMPLES - 1):
                s = self._detect_once(target, quiet=True, near=(best[0], best[1]))
                if s is None:
                    continue
                if np.hypot(s[0] - best[0], s[1] - best[1]) > REFINE_MAX:
                    continue                      # 다른 블록을 본 표본은 버린다
                xs.append(s[0]); ys.append(s[1]); angs.append(self.last_angle)
            if len(xs) > 1:
                mx, my = float(np.median(xs)), float(np.median(ys))
                self.get_logger().info(
                    f"{len(xs)}회 평균 — 산포 ({np.std(xs):.1f}, {np.std(ys):.1f})mm "
                    f"→ ({mx:.1f}, {my:.1f})")
                self.last_angle = float(np.median(angs))
                # 자세(손목 각도)는 마지막 검출의 것을 그대로 두고 x,y 만 갈아끼운다.
                # 표본 간 각도 차이는 1° 안팎이라 다시 만들 실익이 없다.
                best = [mx, my] + list(best[2:])
        return best

    # ── 인간구역 읽기 / 복제 ──
    def human_zones(self):
        z = self.cfg.get("human_zones") or {}
        return {int(k): v for k, v in z.items()}

    def all_zone_xy(self):
        """로봇구역 + 인간구역의 (x, y). 프리구역을 가려내는 기준이 된다."""
        out = [tuple(p[:2]) for p in self.cfg["zones"].values()]
        out += [tuple(p[:2]) for p in self.human_zones().values()]
        return out

    def is_free(self, pose):
        """어느 구역에도 속하지 않은 자리인가. 프리구역 블록만 집기 위한 판정."""
        for zx, zy in self.all_zone_xy():
            if np.hypot(pose[0] - zx, pose[1] - zy) < ZONE_RADIUS:
                return False
        return True

    # ── 순회 관측 ──
    def scan_points(self):
        return list(self.cfg.get("scan_path") or [])

    def goto(self, p, why=""):
        """경유점으로 이동하고 팔이 멎기를 기다린다.

        mwait() 는 명령이 끝난 것만 알려주고 팔이 완전히 멎은 것은 보장하지 않는다.
        검출 노드는 0.4초간 여러 프레임을 모아 합의를 내는데, 그 프레임들이 흔들리는
        동안 잡히면 뒤에 읽는 팔 자세와 안 맞아 좌표가 통째로 밀린다.
        실측 2026-08-04: 대기 없이 재면 노랑이 1번 대신 4번으로 잡히고 빨강이
        42mm 어긋났다. 1초 쉬고 재니 3~6mm 로 들어왔다.
        """
        if why:
            self.get_logger().info(f"{why} {[round(v, 1) for v in p[:3]]}")
        movel(p)
        time.sleep(SETTLE_SEC)

    def look(self, hz=None):
        """지금 자리에서 보이는 것을 인간구역/프리로 갈라 돌려준다.

        (인간구역 {번호: 색}, 프리 {색: [검출, ...]}, 로봇구역 {번호: 색},
         구역 기울기 {("human"|"robot", 번호): 도})

        로봇구역·인간구역 위의 블록은 프리에서 빠진다 — 본보기와 방금 만든
        배치를 도로 집어가면 안 되기 때문이다.
        로봇구역에 이미 놓인 것도 돌려준다. 그 자리에 또 놓으려 하면 부딪힌다.

        **기울기를 따로 담는 이유.** 구역 칸에는 색만 담아 왔다(대시보드도
        비교 코드도 색 문자열을 기대한다). 그런데 복제가 본보기의 기울기까지
        따라 놓으려면 그 값이 필요하다. 칸의 자료형을 바꾸면 부르는 쪽이
        전부 흔들리므로, 같은 열쇠로 찾을 수 있는 별 딕셔너리로 내보낸다.
        """
        hz = self.human_zones() if hz is None else hz
        rz = self.cfg["zones"]
        seen, free, occupied, ang = {}, {}, {}, {}
        for color in KNOWN_COLORS:
            for det in self._detect_all(color, quiet=True):
                p = det["pose"]
                if hz:
                    b = min(hz, key=lambda i: np.hypot(p[0] - hz[i][0], p[1] - hz[i][1]))
                    if np.hypot(p[0] - hz[b][0], p[1] - hz[b][1]) <= ZONE_RADIUS:
                        if b not in seen:      # 먼저 본 것을 남긴다. 색과 기울기가
                            seen[b] = color    # 같은 검출에서 나와야 짝이 맞는다.
                            ang[("human", b)] = det["angle"]
                        continue
                b = min(rz, key=lambda i: np.hypot(p[0] - rz[i][0], p[1] - rz[i][1]))
                if np.hypot(p[0] - rz[b][0], p[1] - rz[b][1]) <= ZONE_RADIUS:
                    if b not in occupied:
                        occupied[b] = color
                        ang[("robot", b)] = det["angle"]
                    continue
                if self.is_free(p):
                    free.setdefault(color, []).append(det)
        return seen, free, occupied, ang

    def patrol(self, want=None, stop_on_find=True, skip=None):
        """관심 영역 위를 경유점 따라 한 바퀴 돈다.

        want 에 색 집합을 주고 stop_on_find 면, 그 색을 프리구역에서 보는 즉시
        멈추고 돌아온다 — 나머지 경유점을 도는 시간을 아낀다.
        want 가 없으면 끝까지 돌며 본 것을 모두 모은다.

        skip 은 앞에서부터 건너뛸 경유점 수다. 기본은 want 가 있으면
        RESCAN_SKIP(앞쪽은 프리 블록이 거의 없다), 없으면 0 이다.
        **0 을 명시하면 want 가 있어도 전부 돈다** — 사람이 앞쪽으로 블록을
        옮겨 두면 건너뛴 경유점에 있어서 못 찾는다(_rescan 참고).

        넓게 한 번에 보는 관측 자세를 없앤 이유: 멀리서 한 장에 다 담으면 블록이
        작게 잡혀 최소 면적에 걸리고, 비스듬히 보여 중심이 밀린다. 가까이서
        나눠 보면 두 문제가 같이 사라진다.

        돌려주는 것: (인간구역, 프리재고, 로봇구역 점유,
                     멈춘 경유점 번호 또는 None, 구역 기울기)
        """
        pts = self.scan_points()
        if not pts:
            self.get_logger().error(
                "순회 경유점이 없습니다. python3 teach_zones.py scan 을 먼저 실행하세요.")
            return {}, {}, {}, None
        if skip is None:
            # 앞쪽 경유점에는 프리 블록이 없다. 자세한 이유는 RESCAN_SKIP 주석 참고.
            skip = RESCAN_SKIP if want else 0
        start = min(max(int(skip), 0), len(pts) - 1)
        pts = pts[start:]
        hz = self.human_zones()
        seen, stock, occ, angles = {}, {}, {}, {}
        for i, p in enumerate(pts, 1):
            self.goto(p, f"[순회 {i + start}/{len(pts) + start}]")
            s_i, f_i, o_i, a_i = self.look(hz)
            for z, c in o_i.items():
                if z not in occ:
                    occ[z] = c
                    angles[("robot", z)] = a_i.get(("robot", z), 0.0)
                    self.get_logger().info(
                        f"  로봇{z}번에 이미 {c} 있음 "
                        f"(기울기 {angles[('robot', z)]:.0f}°)")
            for z, c in s_i.items():
                if z not in seen:
                    seen[z] = c
                    angles[("human", z)] = a_i.get(("human", z), 0.0)
                    self.get_logger().info(
                        f"  인간{z}번 = {c}  기울기 {angles[('human', z)]:.0f}°")
            for c, lst in f_i.items():
                known = stock.setdefault(c, [])
                for det in lst:
                    # 경유점마다 시야가 겹치므로 같은 블록을 두 번 볼 수 있다.
                    if any(np.hypot(det["pose"][0] - k["pose"][0],
                                    det["pose"][1] - k["pose"][1]) < ZONE_RADIUS
                           for k in known):
                        continue
                    known.append(det)
                    self.get_logger().info(
                        f"  프리 {c} ({det['pose'][0]:.0f}, {det['pose'][1]:.0f})")
            if want and stop_on_find and any(stock.get(c) for c in want):
                self.get_logger().info(f"  필요한 블록 발견 — {i}번에서 순회 중단")
                return seen, stock, occ, i, angles
        return seen, stock, occ, None, angles

    def hover_over(self, xy):
        """저장해 둔 자리 위로 카메라를 **수직으로** 내려다보게 옮긴다.

        높이·요는 그 자리에서 가장 가까운 경유점 것을 쓰고, 기울기만 편다.
        기울어진 채로 x,y 만 옮기면 광축이 엉뚱한 곳을 본다 — 실측에서
        "블록 위로 이동" 뒤에도 광축 이탈이 327mm 그대로 남았다.
        """
        pts = self.scan_points()
        if not pts:
            return False
        near = min(pts, key=lambda p: np.hypot(p[0] - xy[0], p[1] - xy[1]))
        att = level_att(near[3:])
        R = Rotation.from_euler("ZYZ", att, degrees=True).as_matrix()
        cam_off = R @ self.gripper2cam[:3, 3]      # 카메라가 그리퍼에서 떨어진 양
        p = [xy[0] - cam_off[0], xy[1] - cam_off[1], near[2]] + att
        try:
            self.goto(p)
        except Exception as e:
            self.get_logger().warn(f"목표 위로 이동 실패({e})")
            return False
        return True

    def copy_human(self, mirror=False):
        """인간구역의 배치를 로봇구역에 그대로(또는 좌우대칭으로) 만든다.

        블록은 프리구역에서만 집는다. 이미 구역에 놓인 것을 집으면 본보기나
        방금 만든 배치가 무너진다.
        """
        self.ready_gripper("복제")
        # 한 바퀴 다 돈다 — 인간구역 4칸을 다 읽어야 계획을 세울 수 있다.
        seen, stock, occ, _, angles = self.patrol()
        # 전체 순회라 인간구역/로봇구역 둘 다 지금 상태 그대로다 — 대시보드에 반영.
        push_zones("human", seen, sorted(self.human_zones()))
        push_zones("robot", occ, sorted(self.cfg["zones"]))
        if not seen:
            self.get_logger().error("인간구역에서 아무 블록도 못 찾았습니다.")
            self.go_home()
            return False

        zones = sorted(self.cfg["zones"])
        plan = []
        for hz_i, color in sorted(seen.items()):
            dst = self.mirror_zone(hz_i) if mirror else hz_i
            if dst not in zones:
                self.get_logger().warn(f"인간 {hz_i}번 → 로봇 {dst}번: 그 구역이 없습니다 — 건너뜀")
                continue
            # 본보기의 기울기. 놓을 때 손목을 이만큼 돌린다(place 참고).
            #
            # **좌우대칭이면 기울기도 뒤집는다.** 거울에 비친 +20° 는 -20° 다.
            # 구역만 대칭으로 옮기고 각도를 그대로 두면, 배치는 대칭인데 블록이
            # 기운 방향만 원본과 같아 눈에 대칭으로 보이지 않는다.
            # mirror_zone 이 x축 대칭(y 부호 반전)이라 부호만 바꾸면 된다.
            ang = angles.get(("human", hz_i), 0.0)
            if mirror:
                ang = -ang
            plan.append((hz_i, dst, color, fold90(ang)))
        if not plan:
            self.go_home()
            return False

        kind = "좌우대칭" if mirror else "그대로"
        self.get_logger().warn(
            f"복제 계획 [{kind}] — " +
            ", ".join(f"인간{h}({c} {a:+.0f}°) → 로봇{d}" for h, d, c, a in plan)
            + ("" if COPY_ANGLE else "   (기울기 따라하기 꺼짐: COPY_ANGLE=0)"))

        # 이미 그 색이 놓여 있는 칸은 건드리지 않는다. 같은 자리에 또 놓으면 부딪힌다.
        # 기울기는 판정에 넣지 않는다 — 색이 맞으면 그대로 둔다. 각도를 맞추려고
        # 이미 제자리에 있는 블록을 다시 집었다 놓으면, 얻는 것(몇 도)보다
        # 잃는 것(파지 실패 위험, 시간)이 크다.
        done = [t for t in plan if occ.get(t[1]) == t[2]]
        if done:
            self.get_logger().info(
                "이미 맞게 놓인 칸: " + ", ".join(f"로봇{d}={c}" for _, d, c, _a in done))
            plan = [t for t in plan if t not in done]
        busy = [t for t in plan if t[1] in occ]
        if busy:
            self.get_logger().warn(
                "다른 색이 놓여 있어 건너뜁니다: "
                + ", ".join(f"로봇{d}에 {occ[d]}(→{c} 필요)" for _, d, c, _a in busy))
            plan = [t for t in plan if t not in busy]
        if not plan:
            self.go_home()
            self.get_logger().warn("옮길 것이 없습니다.")
            return True

        # 재고 부족은 옮기기 전에 알려준다. 반쯤 하다 멈추면 판이 어중간해진다.
        short = [c for c in {t[2] for t in plan}
                 if len(stock.get(c, [])) < sum(1 for t in plan if t[2] == c)]
        if short:
            self.get_logger().warn(
                f"프리구역에 부족한 색: {', '.join(short)} — 그 칸은 건너뜁니다")
            push_stock(short, "해당 칸은 건너뜁니다")

        # 뒤 순번 색이 앞 순번 색을 막고 있으면 앞당긴다. 어차피 이 계획에서
        # 옮겨야 할 색이면, relocate_blocker 로 임시 자리에 잠깐 치웠다가
        # 나중에 제 차례에 또 옮기는(두 번 움직이는) 대신 그 자리에서 곧장
        # 제 목적지로 보내는 편이 한 번으로 끝난다. patrol()로 이미 모아 둔
        # 좌표만으로 판단하므로 로봇을 더 움직이지 않는다 — block_geom.py 에서
        # 로봇 없이 검증된다(test_reorder.py).
        if REORDER_FOR_BLOCKING:
            positions = {c: [(d["pose"][0], d["pose"][1]) for d in dets]
                        for c, dets in stock.items()}
            reordered = reorder_for_conflicts(plan, positions)
            if reordered != plan:
                self.get_logger().warn(
                    "옆 블록이 계획에 있는 다른 색이라 순서를 바꿉니다: "
                    + " → ".join(f"{c}(로봇{d}번)" for _h, d, c, _a in reordered))
                push_debug("info", "복제", "막는 색을 먼저 처리하도록 순서 변경")
                plan = reordered

        # 남은 계획을 relocate_blocker 가 볼 수 있게 걸어 둔다. 막는 블록이
        # **어차피 이 계획에서 옮겨야 할 색**이면 임시 자리로 치우는 대신 곧장
        # 제 구역으로 보내기 위해서다(relocate_to_zone). 실측 2026-08-08:
        # 보라를 (500,230) 임시 자리로 치우려다 실패했는데, 그 보라는 로봇2번에
        # 갈 색이었다 — 바로 놓았으면 초록도 열리고 보라도 끝났다.
        self.copy_pending = list(plan)
        self.copy_stock = stock       # 끌기 복도 검사가 순회 좌표까지 보게
        self.copy_placed = {}
        self.copy_occ = dict(occ)
        ok = True
        try:
            # **집을 수 있는 것부터 전부 로봇구역에 보내고, 막힌 것은 나중에.**
            # 계획 순서대로 밀어붙이면 앞 순번이 막혔을 때 그것부터 풀려고
            # 판을 흔들어 놓고 시작한다. 하나씩 빠질 때마다 판이 헐거워져
            # 막혀 있던 것도 저절로 열리는 일이 잦다. 막힌 것만 남았을 때
            # 비로소 밀기·치우기가 나선다.
            while self.copy_pending:
                item, ready = self.choose_next(stock)
                if item is None:
                    break
                hz_i, dst, color, ang = item
                lst = stock.get(color) or []
                if not lst:
                    self.get_logger().warn(f"인간{hz_i}번의 {color} — 프리구역에 없어 건너뜀")
                    push_stock(color, f"인간{hz_i}번 → 로봇{dst}번 건너뜀")
                    self.copy_pending.remove(item)
                    ok = False
                    continue
                if not ready:
                    self.get_logger().warn(
                        f"남은 것이 다 막혀 있습니다 — 인간{hz_i}번의 {color} 부터 "
                        "밀거나 치워서 풀어 봅니다")
                det = lst.pop(0)          # choose_next 가 집기 쉬운 순으로 정렬해 뒀다
                xy = (det["pose"][0], det["pose"][1])
                # **집기 전에** 뺀다 — 이 블록을 집다가 relocate 가 돌면, 그 안에서
                # 지금 이 항목을 '아직 안 한 것' 으로 보고 또 보내려 할 수 있다.
                self.copy_pending.remove(item)
                self.get_logger().info(
                    f"── 인간{hz_i}번의 {color} {ang:+.0f}° → 로봇{dst}번  "
                    f"(프리 ({xy[0]:.0f}, {xy[1]:.0f}) 에서 집기) ──")
                if self.pick_cached(color, xy, dst, angle=ang):
                    self.copy_placed[dst] = color
                    # 한 개 놓을 때마다 바로 보낸다. 네 개를 다 끝내고 한꺼번에 보내면
                    # 2분 넘게 화면이 그대로여서 진행 중인지 멈춘 건지 알 수가 없다.
                    push_zone("robot", dst, color)
                else:
                    ok = False
        finally:
            # 복제 밖(run_one 등)에서는 이 길이 없어야 한다 — 남겨 두면 다음 명령이
            # 지난 계획을 보고 엉뚱한 구역으로 보낸다.
            placed = self.copy_placed
            self.copy_pending = self.copy_occ = self.copy_stock = None
            self.copy_placed = {}
        if placed:
            # 마무리로 전체를 한 번 더 맞춘다 — 중간에 놓친 갱신이 있어도 여기서 정리된다.
            push_zones("robot", {**occ, **placed}, sorted(self.cfg["zones"]))
        self.go_home()
        self.get_logger().warn(f"복제 {'완료' if ok else '일부 실패'}")
        return ok

    def rescan(self, color, xy):
        """봐 뒀던 자리에서 블록이 사라졌을 때 다시 찾는다. 못 찾으면 None.

        **사람이 그 사이에 옮기기 때문이다.** 복제는 한 색을 놓고 오는 동안
        판을 보지 못하는데, 그 틈에 사람이 다음에 쓸 블록을 다른 곳으로 옮기면
        봐 뒀던 좌표가 헛것이 된다(실측 2026-08-07: 첫 순회에서 본 색을 놓고
        돌아오니 없어져 그대로 건너뛰었다).

        두 번 돈다. **두 번의 범위가 다른 것이 요점이다.**

            1차  그 색만, 뒤쪽 경유점부터(RESCAN_SKIP). 발견 즉시 멈춘다 — 빠르다.
            2차  경유점 **전부**. 사람이 앞쪽(인간구역·안쪽 구역을 보는 자리)으로
                 옮겼으면 1차가 그 자리를 통째로 지나친다. 그 경우를 여기서 잡는다.

        찾은 자리는 프리구역인지 부르는 쪽이 다시 본다(pick_cached) — 사람이
        구역 안에 놓았으면 집지 않는다.
        """
        for i, skip in enumerate(RESCAN_SKIPS, start=1):
            where = "전체 경유점" if skip == 0 else "그 색만"
            msg = (f"블록 사라짐 — {color} 가 ({xy[0]:.0f}, {xy[1]:.0f}) 에 없습니다. "
                   f"재탐색 {i}/{RESCAN_TRIES} ({where})")
            self.get_logger().warn(msg)
            push_debug("warn", "재탐색", msg)
            _, stock, _, _, _ = self.patrol(want={color}, skip=skip)
            lst = stock.get(color) or []
            if not lst:
                continue
            nxy = (lst[0]["pose"][0], lst[0]["pose"][1])
            if not self.hover_over(nxy):
                continue
            pose = self.detect(color, near=nxy)
            if pose is not None:
                found = f"재탐색 {i}/{RESCAN_TRIES} 에서 {color} 발견 ({nxy[0]:.0f}, {nxy[1]:.0f})"
                self.get_logger().warn(found)
                push_debug("info", "재탐색", found)
                return pose
        return None

    def pick_cached(self, color, xy, zone, angle=None):
        """관측 때 봐 둔 자리로 곧장 가서 집고 구역에 놓는다.

        넓은 관측 자세로 되돌아가지 않는다 — 좌표를 이미 알고 있으므로 그 위로
        바로 올라가 정밀화만 하면 된다. 옮길 색 수만큼 왕복이 줄어든다.

        angle 은 **놓을 때** 따라할 본보기 기울기다(복제 전용). 집기에는 쓰지
        않는다 — 집을 때는 그 블록 자기 기울기에 손목을 맞춰야 한다.
        """
        if not self.hover_over(xy):
            self.get_logger().error("관측 자세가 없어 집을 수 없습니다.")
            return False
        pose = self.detect(color, near=xy, max_dist=CACHE_TOLERANCE)
        if pose is None:
            pose = self.rescan(color, xy)
        if pose is None:
            self.get_logger().error(f"{color} 를 다시 못 찾았습니다 — 건너뜁니다.")
            push_stock(color, f"{RESCAN_TRIES}번 다시 돌아도 못 찾았습니다")
            return False
        if not self.is_free(pose):
            # 관측 뒤에 판이 바뀌었거나 다른 개체를 잡은 것이다. 구역 위의 것은
            # 절대 집지 않는다 — 본보기나 방금 만든 배치가 무너진다.
            self.get_logger().error(
                f"{color} 로 다시 찾은 자리가 프리구역이 아닙니다 "
                f"({pose[0]:.0f}, {pose[1]:.0f}) — 건너뜁니다.")
            return False
        self.get_logger().info(f"파지 목표 {[round(v, 1) for v in pose[:3]]}")
        if not self.pick(pose):
            self.get_logger().error("파지 실패")
            self.go_home()
            return False
        self.place(zone, angle=angle)
        self.get_logger().info(f"완료: {color} → 로봇 {zone}번")
        return True

    def mirror_zone(self, i):
        """인간구역 i 를 x축 대칭(y 부호 반전)한 자리에 가장 가까운 로봇구역.

        매핑을 {1:2, 2:1, ...} 처럼 박아두면 구역 번호를 다시 매기거나 판을
        옮겼을 때 조용히 틀린다. 티칭 좌표에서 그때그때 유도한다.
        실측(2026-08-04)에서는 유도값이 {1:2, 2:1, 3:4, 4:3} 으로 나왔고,
        대칭점과 구역 중심의 차이는 6.5~9.0mm 였다.
        """
        H = self.human_zones()
        R = self.cfg["zones"]
        if i not in H or not R:
            return i
        mx, my = H[i][0], -H[i][1]          # x축 대칭 = y 부호 반전
        j = min(R, key=lambda k: np.hypot(R[k][0] - mx, R[k][1] - my))
        d = float(np.hypot(R[j][0] - mx, R[j][1] - my))
        if d > ZONE_RADIUS:
            self.get_logger().warn(
                f"인간{i}번의 x축 대칭점이 로봇{j}번에서 {d:.0f}mm 떨어져 있습니다 "
                "— 두 구역군이 x축을 사이에 두고 마주보게 놓였는지 확인하세요.")
        return j

    def detect_in_zone(self, zone):
        """그 구역에 놓인 블록의 색을 알아낸다. 없으면 None.

        색을 말하지 않은 명령("3번구역 들다 1번구역 놓다")은 '거기 있는 것'을
        집으라는 뜻이다. 그런데 검출 서비스는 색을 지정해야 답한다. 그래서
        아는 색을 전부 물어보고 구역 중심에 가장 가까운 것을 고른다.
        이 단계는 팔이 움직이지 않으므로 여섯 번 물어도 싸다 — 색을 정한 뒤에는
        detect() 의 평소 경로(광축 정렬 → 여러 번 재기)를 그대로 탄다.
        """
        zx, zy = self.cfg["zones"][zone][:2]
        found = []
        for color in KNOWN_COLORS:
            # 같은 색이 여러 곳에 있을 수 있으므로 전부 받아 가장 가까운 것만 남긴다.
            for det in self._detect_all(color, quiet=True):
                p = det["pose"]
                found.append((float(np.hypot(p[0] - zx, p[1] - zy)), color))
        if not found:
            self.get_logger().error(f"{zone}번 구역 근처에서 아무 블록도 못 찾았습니다.")
            return None

        found.sort()
        d, color = found[0]
        rest = "  다음: " + ", ".join(f"{c} {dd:.0f}mm" for dd, c in found[1:3]) \
            if len(found) > 1 else ""
        if d > ZONE_RADIUS:
            self.get_logger().error(
                f"{zone}번 구역이 비어 있습니다 — 가장 가까운 블록이 {color}, "
                f"{d:.0f}mm 떨어져 있습니다 (한계 {ZONE_RADIUS:.0f}mm).{rest}")
            return None
        self.get_logger().info(f"{zone}번 구역의 블록 = {color} ({d:.0f}mm){rest}")
        return color

    def _detect_all(self, target, quiet=False):
        """지금 자세에서 그 색으로 보이는 것을 전부 찾는다.

        검출 노드가 (x, y, z, 각도) 를 이어붙여 보내므로 4개씩 끊어 읽는다.
        믿을 만한 순으로 와 있어 첫 번째가 예전 결과와 같다.
        """
        if getattr(self, "_svc_down", False):
            return []                        # 이번 명령에서 이미 끊겼다
        if not self.cli.wait_for_service(timeout_sec=SVC_WAIT):
            # 여기서도 플래그를 세운다. 안 그러면 색마다 SVC_WAIT 씩 기다려
            # 6색이면 그것만 30초다 (실측).
            self._svc_down = True
            self.get_logger().error("/get_3d_position 서비스가 없습니다. "
                                    "object_detection 노드를 먼저 띄우세요.")
            return []
        self.req.target = target
        fut = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=SVC_TIMEOUT)
        if fut.result() is None:
            # 검출 노드가 죽었거나 멎었다. 색마다 20초씩 기다리면 6색에 2분을
            # 버린다 — 실측에서 명령을 내리고 2분간 아무 일도 안 일어났다.
            # 한 번 끊기면 이 명령은 접는다.
            self._svc_down = True
            self.get_logger().error(
                f"검출 서비스 응답 없음({SVC_TIMEOUT:.0f}초) — "
                "object_detection 노드가 살아 있는지 확인하세요.")
            return []

        self._svc_down = False
        raw = list(fut.result().depth_position)
        if len(raw) < 4 or sum(raw[:3]) == 0:
            # quiet 는 '여러 번 물어보는 중'이라는 뜻이다. detect_in_zone 은 아는 색을
            # 전부 훑으므로 없는 색마다 경고를 찍으면 정작 볼 줄이 묻힌다.
            if not quiet:
                self.get_logger().warn(f"'{target}' 을(를) 찾지 못했습니다.")
            return []

        # 영상과 로봇 자세는 같은 시점이어야 한다. 검출 직후 바로 읽는다.
        posx = get_current_posx()[0]
        out = []
        for k in range(0, len(raw) - 3, 4):     # (x, y, z, 각도) 씩 끊어 읽는다
            cam, ang = raw[k:k + 3], float(raw[k + 3])
            if sum(cam) == 0:
                continue
            xyz = self.to_base(cam, posx)
            top_z = float(xyz[2])
            xyz[2] = self.pick_z            # x,y 는 검출값, z 는 티칭값
            rot = GRASP_ROT
            if ALIGN_TO_BLOCK:
                rot += fold90(ang)
            # 파지 자세는 언제나 수직이어야 한다. 관측 자세의 기울기를 물려받으면
            # 그리퍼가 비스듬히 내려가고, rotate_tool 이 기울어진 축으로 돌아
            # 블록 면에 손가락을 맞추지도 못한다. grasp_offset 실측도 수직 기준이다.
            out.append({"pose": rotate_tool(list(xyz) + level_att(posx[3:]), rot),
                        "cam": list(cam), "angle": ang, "top": top_z, "rot": rot})
        if not quiet:
            self.get_logger().info(
                f"'{target}' {len(out)}개 — " +
                ", ".join(f"({d['pose'][0]:.0f}, {d['pose'][1]:.0f})" for d in out))
        return out

    def _adopt(self, det, quiet=False):
        """고른 검출을 '마지막 검출' 상태로 반영하고 파지 자세를 돌려준다.

        pick() 이 grasp_offset() 에서 self.last_rot 을 쓰므로, 어느 것을 골랐는지
        상태에 남겨야 한다. 고르기와 상태 반영을 한 곳에 묶어 둔다.
        """
        self.last_cam = det["cam"]
        self.last_angle = det["angle"]
        self.last_top = det["top"]
        self.last_rot = det["rot"]
        d = det["top"] - self.expect_top
        if not quiet:
            self.get_logger().info(
                f"윗면 z {det['top']:.1f} (예상 {self.expect_top:.1f}, 차이 {d:+.1f}) "
                f"→ 파지 z {self.pick_z:.1f}   기울기 {det['angle']:.0f}° "
                f"→ 손목 {det['rot']:+.0f}°")
            if abs(d) > Z_SANITY:
                self.get_logger().warn(
                    f"윗면 높이가 예상에서 {d:+.1f}mm 벗어났습니다. "
                    "블록이 쌓였거나 깊이가 튀었을 수 있습니다.")
        return det["pose"]

    def _detect_once(self, target, quiet=False, near=None, max_dist=None):
        """지금 자세에서 하나 고른다. near 를 주면 그 점에 가장 가까운 것.

        near 가 없으면 가장 믿을 만한 것(검출 노드가 앞에 놓은 것)을 쓴다.
        같은 색 블록이 여럿일 때 '어느 것'인지는 부르는 쪽만 알기 때문에
        고르는 기준을 인자로 받는다.
        """
        dets = self._detect_all(target, quiet=quiet)
        if not dets:
            return None
        if near is not None:
            dets = sorted(dets, key=lambda d: np.hypot(d["pose"][0] - near[0],
                                                       d["pose"][1] - near[1]))
            if max_dist is not None:
                d0 = float(np.hypot(dets[0]["pose"][0] - near[0],
                                    dets[0]["pose"][1] - near[1]))
                if d0 > max_dist:
                    # 찾던 자리에 없다. 멀리 있는 같은 색을 집으면 엉뚱한 블록이다.
                    if not quiet:
                        self.get_logger().warn(
                            f"'{target}' 가 기대 자리에서 {d0:.0f}mm 떨어져 있습니다 "
                            f"(한계 {max_dist:.0f}) — 못 찾은 것으로 봅니다")
                    return None
        return self._adopt(dets[0], quiet=quiet)

    # ── 동작 ──
    def rise(self, target_z, why=""):
        """지금 자리에서 **수직으로만** 올린다. xy 는 건드리지 않는다.

        target_z 가 안 되면 **지금 높이 + LIFT_FALLBACK 만큼만** 올려 본다.
        판 바깥쪽에서는 250mm 까지 뻗는 자세에 IK 해가 없어 컨트롤러가 조용히
        거부하는데(실측 2026-08-08), 거기서 포기하면 물린 채로 판을 대각선으로
        지나가 다른 블록을 친다. 조금이라도 올리면 그만큼 안전해진다.

        이미 충분히 높으면 아무것도 하지 않고 True.
        """
        try:
            cur = list(get_current_posx()[0])
        except Exception as e:
            self.get_logger().warn(f"현재 높이를 못 읽어 상승을 건너뜁니다({e})")
            return False
        if cur[2] >= target_z - MOVE_TOL:
            return True
        tag = f"{why} " if why else ""
        full = list(cur); full[2] = target_z
        if movel(full):
            self.get_logger().info(
                f"{tag}수직 상승 {cur[2]:.0f} → {target_z:.0f}mm")
            return True
        # 물러나기 — 지금 높이에서 조금만.
        low = list(cur); low[2] = cur[2] + LIFT_FALLBACK
        if movel(low):
            self.get_logger().warn(
                f"{tag}{target_z:.0f}mm 까지 못 올라가 {LIFT_FALLBACK:.0f}mm 만 "
                f"올립니다 ({cur[2]:.0f} → {low[2]:.0f}mm)")
            return True
        self.get_logger().error(
            f"{tag}수직 상승이 전혀 안 됩니다 ({cur[2]:.0f}mm 에 머무름) "
            "— 이대로 옮기면 판을 대각선으로 지나갑니다")
        return False

    def ready_gripper(self, why=""):
        """**명령을 시작하기 전에 그리퍼를 벌린다.** 물고 있었으면 경고한다.

        닫힌 채로 시작하면 팔이 파지 자세로 내려갈 때 손가락(또는 물고 있던
        블록)이 판 위의 다른 블록을 들이받는다. go_home() 이 끝에서 벌려 주긴
        하지만, 앞 명령이 중간에 끊기거나(사람이 Ctrl+C, 모션 거부, 예외) 조종
        모드를 거쳐 오면 닫힌 상태로 남는다 — 그때가 부딪히는 때다.

        시작할 때 한 번 벌리는 값은 거의 0 인데(그리퍼 동작 한 번), 막아 주는
        것은 판 위 충돌이라 남는 장사다.

        블록을 물고 있었다면 여기서 **떨어뜨린다.** 그것 말고 할 수 있는 일이
        없다 — 어디에 놓아야 할지 모르는 블록이고, 문 채로 움직이는 것이 더
        위험하다. 로그에 남기니 사람이 주워서 판에 되돌려 놓으면 된다.
        """
        try:
            held = grasped()
        except Exception:
            held = False
        if held:
            msg = "명령 시작 — 그리퍼가 블록을 물고 있습니다. 여기서 놓습니다"
            self.get_logger().warn(msg + (f" ({why})" if why else ""))
            push_debug("warn", "파지", "앞 명령에서 물고 있던 블록을 놓습니다")
        try:
            gripper.open_gripper()
            wait_gripper()
        except Exception as e:
            self.get_logger().warn(f"그리퍼를 벌리지 못했습니다({e}) — 그대로 진행합니다")

    def go_home(self):
        gripper.open_gripper()
        wait_gripper()
        movej(JREADY)

    def grasp_offset(self):
        """지금 손목 각도에서의 파지 중심 보정. 공구 성분은 함께 회전시킨다."""
        r = np.radians(self.last_rot)
        R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        return np.array(OFFSET_BASE) + R @ np.array(OFFSET_TOOL)

    def neighbors_xy(self, target_xy, radius=None, self_r=None):
        """지금 시야에서 target 주변에 있는 **다른** 블록들의 중심 (base xy).

        색을 전부 물어본다. 팔이 멈춰 있으므로 검출 노드가 촬영 한 번을 6색이
        나눠 쓴다(프레임 캐시가 자세 기준이라 그렇다 — 실측 6색 2.07초).
        """
        radius = NEIGHBOR_R if radius is None else radius
        # self_r 은 '자기 자신' 으로 볼 반경이다. **빈 자리를 검사할 때는 0 을 준다** —
        # 그 자리엔 자기 자신이 없으므로, 기본값(SELF_R=30mm)을 그대로 쓰면 30mm
        # 안의 진짜 블록을 자기 자신으로 착각해 '비었다' 고 답한다.
        # 그 위에 블록을 내려놓게 되는 자리다(relocate_blocker 의 목적지 검사).
        self_r = SELF_R if self_r is None else self_r
        out = []
        for color in KNOWN_COLORS:
            for det in self._detect_all(color, quiet=True):
                p = det["pose"]
                d = float(np.hypot(p[0] - target_xy[0], p[1] - target_xy[1]))
                if d < self_r:
                    continue                  # 자기 자신 (블록은 이보다 붙을 수 없다)
                if d <= radius:
                    out.append((p[0], p[1], color))
        return out

    def aim_between_neighbors(self, pose, allow_relocate=True, _tries=0, _depth=0):
        """손가락이 내려갈 방향에 블록이 있으면 **손목을 90° 돌린다.**

        그리퍼는 한 축의 양쪽에서 손가락이 내려와 안쪽으로 닫힌다. 그 축에 다른
        블록이 얹혀 있으면 손가락이 그것을 치거나 밀어낸다(실측: 파지 중 오류로
        정지했다). 정사각 블록은 90° 대칭이라 **돌려도 물리는 품질이 같다** —
        물는 면이 옆면에서 옆면으로 바뀔 뿐이다. 그래서 여유가 넓은 쪽을 고르면
        잃는 것이 없다.

        두 축 다 좁으면, RELOCATE_BLOCKERS 가 켜져 있고 allow_relocate 이면
        막고 있는 이웃 하나를 옆으로 옮기고 다시 판단한다(relocate_blocker).
        그래도 안 풀리면(또는 꺼져 있으면) **None 을 돌려 파지를 접는다** — 칠 것을
        알면서 내려가는 것보다 멈추고 사람에게 치우라고 하는 편이 낫다
        (PICK_WHEN_TIGHT 로 바꿀 수 있다).

        allow_relocate=False 는 relocate_blocker 가 이웃 블록 **자신**을 집을 때
        쓴다 — 그 이웃을 집으면서 또 다른 이웃을 옮기기 시작하면 판 전체가
        연쇄적으로 재배치될 수 있어, 한 단계로 막는다.

        판단은 block_geom.best_axis 가 한다(로봇 없이 시험된다 — test_avoid.py).
        여기서는 **좌표를 모아 주고, 고른 결과를 자세와 last_rot 에 반영**한다.
        last_rot 을 같이 돌려야 파지 보정(grasp_offset)이 함께 돌아간다 —
        공구 성분(OFFSET_TOOL)이 손목과 같이 도는 값이기 때문이다.
        """
        if not AVOID_NEIGHBORS:
            return pose
        nb = self.neighbors_xy((pose[0], pose[1]))
        # **없어도 찍는다.** 조용히 지나가면 "옆 블록이 없다고 본 것" 과 "이 기능이
        # 안 돌고 있는 것" 을 구별할 수 없다 — 실측 2026-08-07 에 그걸로 헤맸다.
        self.get_logger().info(
            f"옆 블록 살핌 (반경 {NEIGHBOR_R:.0f}mm): "
            + (", ".join(f"{c}({x:.0f},{y:.0f})" for x, y, c in nb) if nb else "없음"))
        if not nb:
            return pose
        axis = tool_yaw(pose[3:])          # 손가락이 닫히는 축의 base 방위
        turn, gap, gap0 = best_axis((pose[0], pose[1]), axis,
                                    [(x, y) for x, y, _c in nb])
        near = ", ".join(f"{c}({x:.0f},{y:.0f})" for x, y, c in nb)

        def mm(v):   # noqa: E306
            """무한(방해 없음)을 '충분' 으로 읽히게. 'infmm' 은 로그에서 읽기 어렵다."""
            return "충분" if v == float("inf") else f"{v:.0f}mm"

        # 어느 축에서 무엇이 막는지 함께 찍는다 — 이것 없이는 "왜 저 블록을
        # 치우려 하지" 를 로그로 따라갈 수 없다(실측 2026-08-07 에 그걸로 헤맸다).
        used = axis + turn
        lane = []
        for x, y, c in nb:
            al, sd = lane_offsets((pose[0], pose[1]), (x, y), used)
            if abs(sd) < LANE_HALF:
                lane.append((c, round(abs(al) - BLOCK_MM)))
        self.get_logger().info(
            f"파지축 {used % 180:.0f}° — 이 축에서 막는 것: "
            + (", ".join(f"{c} 틈{g}mm" for c, g in lane) if lane else "없음"))
        if turn:
            # 손목이 덜 돌아가는 쪽으로 접는다 — 판단은 block_geom.fold_turn 이
            # 한다(로봇 없이 시험된다). ±90 이 같은 물림인 이유도 거기 적어 두었다.
            folded = fold_turn(self.last_rot, turn)
            if folded != turn:
                self.get_logger().info(
                    f"손목을 {folded:+.0f}° 로 접습니다 (반대로 돌려도 같은 물림 — "
                    f"{self.last_rot + turn:+.0f}° 대신 {self.last_rot + folded:+.0f}°)")
                turn = folded
            self.get_logger().warn(
                f"옆 블록 때문에 손목을 {turn:+.0f}° 돌립니다 — "
                f"틈 {mm(gap0)} → {mm(gap)}  [{near}]")
            push_debug("info", "파지", f"옆 블록 회피 — 손목 {turn:+.0f}° "
                                      f"(틈 {mm(gap0)}→{mm(gap)})")
            pose = rotate_tool(list(pose), turn)
            self.last_rot += turn
        elif gap < float("inf"):
            self.get_logger().info(f"옆 블록 있음, 틈 {mm(gap)} — 그대로 물립니다 [{near}]")
        if gap < FINGER_GAP_MIN:
            # 돌려도 안 풀린다 — 두 축 다 좁다.
            if RELOCATE_BLOCKERS and allow_relocate and _tries < RELOCATE_MAX_TRIES:
                # relocate_blocker 는 안에서 (치울) 이웃을 스스로 detect()/pick() 한다 —
                # 그 둘 다 self.last_rot 을 **그 이웃의** 각도로 덮어쓴다. 여기서
                # 저장해 뒀다가 되돌리지 않으면, 되돌아와 재귀할 때 원래 target 의
                # 회전(turn)이 아니라 방금 치운 이웃의 각도 위에 다시 더해져
                # grasp_offset() 이 엉뚱한 방향으로 보정된다.
                saved_last_rot = self.last_rot
                moved = self.relocate_blocker((pose[0], pose[1]), axis + turn, nb,
                                              depth=_depth)
                if not moved:
                    # **이 축이 안 되면 반대 축의 방해물을 치워 본다.**
                    # best_axis 는 '틈이 넓은 축' 을 고르지만, 그 축의 방해물이
                    # 치울 수 있는 것인지는 보지 않는다. 실측 2026-08-08:
                    # 축 1°(틈15mm)와 축 91°(틈14mm) 중 1mm 차이로 1° 를 골랐는데,
                    # 그 축을 막는 보라색은 x=648 이라 팔이 못 닿아 두 번 다
                    # "블록 위로 접근하지 못해" 로 죽었다. 91° 를 막는 노란색은
                    # x=597 로 멀쩡히 닿았다 — 축만 바꿨으면 끝났을 일이다.
                    # 두 축은 어차피 같은 물림이니(정사각 블록), 열 수 있는 쪽을
                    # 열면 된다.
                    self.get_logger().warn(
                        f"파지축 {(axis + turn) % 180:.0f}° 는 못 열었습니다 — "
                        f"직각 축 {(axis + turn + 90) % 180:.0f}° 의 방해물을 치워 봅니다")
                    moved = self.relocate_blocker((pose[0], pose[1]),
                                                  axis + turn + 90.0, nb,
                                                  depth=_depth)
                self.last_rot = saved_last_rot
                if moved:
                    # **목표 위로 돌아가서 다시 본다.** 치우고 나면 팔이 치운 자리
                    # 위에 있어, 그대로 재검출하면 목표 주변이 시야에 제대로 안
                    # 들어온다 — 방금 치운 것도, 앞서 치운 것도 반영이 안 된다.
                    # 돌아가서 보면 판의 **지금 상태**로 다시 판단한다(치운 블록은
                    # 사라졌고, 남은 방해물만 남는다).
                    if not self.hover_over((pose[0], pose[1])):
                        self.get_logger().warn("목표 위로 돌아가지 못해 재판단을 건너뜁니다")
                        return None
                    return self.aim_between_neighbors(pose, allow_relocate,
                                                      _tries + 1, _depth)
            # 여기까지 왔다는 것은 **로봇이 스스로 할 수 있는 것을 다 했다**는 뜻이다.
            # 사람에게 넘기기 전에 무엇을 시도했는지 남긴다 — 그냥 "치우고 다시
            # 명령하세요" 만 뜨면 로봇이 아무것도 안 해본 것처럼 보인다.
            # **시도했는데 실패한 것과 후보가 없던 것을 구별한다.** 예전 문구는
            # 둘 다 "치울 만한 이웃을 못 찾아" 라고 해서, 로그를 봐야만 실제로
            # 두 개를 시도했다는 걸 알 수 있었다.
            failed = len(getattr(self, "_failed_blockers", None) or ())
            tried = (f"이웃 {_tries}개를 치워 봤지만" if _tries else
                     "치우기가 꺼져 있어(RELOCATE_BLOCKERS=0)" if not RELOCATE_BLOCKERS
                     else f"이웃 {failed}개를 치우려다 다 실패해서" if failed
                     else "치울 만한 이웃을 못 찾아")
            self.get_logger().error(
                f"양쪽 다 좁습니다 (틈 {mm(gap)} < 손가락 {FINGER_GAP_MIN:.0f}mm) — {tried} "
                + ("경고만 하고 그대로 집습니다 (PICK_WHEN_TIGHT=1)"
                   if PICK_WHEN_TIGHT else
                   "**집지 않습니다.** 블록 하나를 손으로 떼어 주고 다시 명령하세요"))
            push_debug("warn", "파지",
                       f"파지 틈 {mm(gap)} — {tried} "
                       + ("그대로 집습니다" if PICK_WHEN_TIGHT
                          else "손으로 떼어 주세요"))
            if not PICK_WHEN_TIGHT:
                return None            # 부르는 쪽(pick)이 파지를 접는다
        return pose

    def in_reach(self, xy):
        """팔이 확실히 닿는 자리인가. **x 한계만** 본다.

        **틈이 넓다고 집을 수 있는 것은 아니다.** 판 바깥쪽 블록은 주변이 뻥
        뚫려 있어 '가장 집기 쉬운 것' 으로 뽑히기 쉬운데 정작 팔이 못 간다.
        실측 2026-08-08: 주황이 (534,186) 과 (751,-155) 두 개였는데 여유가
        크다는 이유로 x=751 을 골랐다.

        **왜 교시 상자(reach_box)를 안 쓰나.** 처음엔 그걸 썼는데 x 한계가
        566 이라 너무 좁았다 — 실기에서 멀쩡히 집히는 x=570, 583, 599, 607,
        624, 626 이 전부 '바깥' 으로 밀려 계획 1번이 맨 뒤로 갔다.
        그 상자는 손동작 조종용으로 교시한 범위지 팔의 도달 한계가 아니다.

        **왜 x 만 보나.** 5일치 로그 185건을 위치별로 갈라 보니 y 는 무관했다:
            광축 정렬 중단 59건   x 중앙값 648   |y| 중앙값 116
            광축 정렬 수렴 126건  x 중앙값 565   |y| 중앙값 133
            중단 중 x>=630 인 비율 78%  /  수렴 중 x>=630 인 비율 5%
        같은 날 실기에서도 x=626 까지는 성공했고 x=646, 648 이 "블록 위로
        접근하지 못해" 로 죽었다. 그 사이에 선을 그었다.

        **거르는 기준이 아니라 고르는 기준**이다 — 안쪽을 먼저 고르고, 안쪽에
        할 것이 하나도 없을 때만 바깥도 시도한다.
        """
        return xy[0] <= REACH_X_MAX

    def choose_next(self, stock):
        """남은 복제 계획 중 **지금 바로 집을 수 있는 것**을 먼저 고른다.

        돌려주는 것: (계획 항목, 지금 집을 수 있나)

        **왜 순서를 그때그때 고르나.** 계획 순서대로 밀어붙이면, 앞 순번이
        막혀 있을 때 그것부터 풀려고 이웃을 밀거나 치운다 — 판을 흔들어 놓고
        시작하는 셈이다. 뒤 순번 중에 지금 그냥 집을 수 있는 것이 있으면
        그걸 먼저 로봇구역에 보내는 편이 낫다. 하나 빠질 때마다 판이 헐거워져
        막혀 있던 것도 저절로 열리는 일이 잦다(무작위 배치 400개 검증:
        정적 순서 88.2% → 그때그때 고르기 89.9%, 더 나빠진 배치는 0개).

        판단은 patrol() 로 이미 모아 둔 좌표만 쓴다 — 로봇을 더 움직이지 않는다.
        틀려도 손해가 없다. 집을 수 있다고 봤는데 아니면 평소대로 치우기가
        나서고, 없다고 봤는데 되면 그냥 집힌다.
        """
        pend = self.copy_pending or []
        all_pts = [(d["pose"][0], d["pose"][1])
                   for lst in stock.values() for d in lst]
        # 두 바퀴 돈다. **첫 바퀴는 팔이 닿는 것만** 본다 — 틈이 넓어도 판
        # 바깥이면 못 집는다(in_reach 주석의 실측). 안쪽에 할 게 하나도 없을
        # 때만 바깥을 본다. 상자 밖이라도 실제로 집히는 자리가 있어서다.
        for reach_only in (True, False):
            for item in pend:
                lst = stock.get(item[2]) or []
                if not lst:
                    continue
                lst[:] = self.graspable_first(lst, stock)   # 개체도 좋은 것부터
                d = lst[0]
                xy = (d["pose"][0], d["pose"][1])
                if reach_only and not self.in_reach(xy):
                    continue
                # SELF_R 안쪽은 **이웃이 아니라 자기 자신**이다 — 실제 파지가
                # 쓰는 neighbors_xy 와 같은 규칙. 이게 없으면 색을 이중 검출한
                # 유령(빨강/주황 14.8mm)이 진짜 방해물로 잡혀, 멀쩡히 집히는
                # 블록이 '완전히 막힘' 으로 밀려난다(실측 2026-08-08).
                others = [p for p in all_pts
                          if np.hypot(p[0] - xy[0], p[1] - xy[1]) >= SELF_R]
                _t, gap, _g0 = best_axis(xy, tool_yaw(d["pose"][3:]), others)
                if gap >= FINGER_GAP_MIN:
                    if not reach_only:
                        self.get_logger().warn(
                            f"인간{item[0]}번의 {item[2]} ({xy[0]:.0f},{xy[1]:.0f}) 는 "
                            "판 바깥이지만 안쪽에 할 것이 없어 시도합니다")
                    return item, True
        return (pend[0], False) if pend else (None, False)

    def graspable_first(self, dets, stock, why=""):
        """같은 색 후보들을 **집기 쉬운 순**으로 다시 늘어놓는다 (판단은 block_geom).

        dets   그 색의 검출들(_detect_all 형식). 원본을 건드리지 않고 새 리스트.
        stock  {색: [검출, ...]} — 순회에서 본 전부. 이웃 판정의 재료다.

        검출 노드가 주는 신뢰도 순은 집기 쉬움과 무관하다. 사방이 포위된 것을
        먼저 집으려 들면 이웃 치우기가 발동하고, 그것마저 실패하면 사람에게
        치워 달라고 한다 — 고를 수 있을 때 고르는 편이 언제나 싸다.
        """
        if len(dets) < 2:
            return list(dets)
        all_pts = [(d["pose"][0], d["pose"][1])
                   for lst in stock.values() for d in lst]
        by_xy = {(round(d["pose"][0], 1), round(d["pose"][1], 1)): d for d in dets}
        pts = [(d["pose"][0], d["pose"][1], tool_yaw(d["pose"][3:])) for d in dets]
        out = []
        for p in pick_order(pts, all_pts, self_r=SELF_R):
            d = by_xy.get((round(p[0], 1), round(p[1], 1)))
            if d is not None and d not in out:
                out.append(d)
        for d in dets:                      # 혹시 빠진 것이 있으면 뒤에 붙인다
            if d not in out:
                out.append(d)
        # **닿는 것을 먼저.** 틈이 넓다고 집을 수 있는 것은 아니다 — 판 바깥쪽
        # 블록은 주변이 뻥 뚫려 '가장 집기 쉬운 것' 으로 뽑히기 쉬운데 팔이
        # 거기까지 못 간다(in_reach 주석의 실측 참고). 여유 순서는 그대로 두고
        # 닿는 것들을 앞으로만 당긴다 — 안정 정렬이라 같은 무리 안 순서는 유지된다.
        far = [d for d in out if not self.in_reach(d["pose"])]
        if far and len(far) < len(out):
            out = [d for d in out if self.in_reach(d["pose"])] + far
            where = ", ".join(f"({d['pose'][0]:.0f},{d['pose'][1]:.0f})" for d in far)
            self.get_logger().info(
                f"{why}판 바깥 {len(far)}개 [{where}] 는 뒤로 미룹니다 "
                "— 팔이 못 닿을 수 있습니다")
        if out[0] is not dets[0]:
            self.get_logger().info(
                f"{why}같은 색 {len(dets)}개 중 집기 쉬운 것을 고릅니다 — "
                f"({dets[0]['pose'][0]:.0f},{dets[0]['pose'][1]:.0f}) 대신 "
                f"({out[0]['pose'][0]:.0f},{out[0]['pose'][1]:.0f})")
        return out

    def pending_target(self, color):
        """이 색이 지금 복제 계획에서 **아직 안 옮긴 항목**이면 그 항목, 아니면 None.

        copy_human() 이 돌 때만 값이 있다(self.copy_pending). 색·구역 지정 명령
        (run_one) 처럼 계획이 없는 길에서는 None 이라, 예전처럼 임시 자리로 치운다.

        목적지가 이미 찬 항목은 고르지 않는다 — copy_human 이 계획을 세울 때
        걸렀지만, 그 뒤 이 실행 중에 우리가 채웠을 수 있다.
        """
        for item in (getattr(self, "copy_pending", None) or []):
            _hz_i, dst, c, _ang = item
            if c != color:
                continue
            if dst in (getattr(self, "copy_placed", None) or {}):
                continue                       # 이번 실행에서 이미 채웠다
            if (getattr(self, "copy_occ", None) or {}).get(dst) not in (None, color):
                continue                       # 다른 색이 놓여 있다
            return item
        return None

    def relocate_to_zone(self, item, bxy, allow_chain=False, depth=0):
        """막고 있는 블록을 임시 자리가 아니라 **제 목적지 구역**으로 보낸다.

        치우기와 계획 수행을 한 번의 이동으로 합친다. 성공하면 그 계획 항목은
        끝난 것으로 표시되어 copy_human 의 차례에서 건너뛴다.

        실패하면 False 를 돌려주고 **임시 자리로 물러나지 않는다** — 같은 블록을
        같은 방법으로 다시 집으려는 것이라 또 실패할 뿐이고, 그 사이 팔만 더
        움직인다. 부르는 쪽(aim_between_neighbors)이 원래대로 파지를 접는다.
        """
        hz_i, dst, color, ang = item
        self.get_logger().warn(
            f"{color}({bxy[0]:.0f},{bxy[1]:.0f})가 막고 있는데 이 계획에서 "
            f"로봇{dst}번에 갈 색입니다 — 옆으로 치우지 않고 곧장 놓습니다")
        push_debug("info", "파지", f"막는 {color} 를 로봇{dst}번으로 바로 보냅니다")
        pose = self.detect(color, near=bxy, max_dist=CACHE_TOLERANCE)
        if pose is None:
            self.get_logger().error(f"막고 있는 {color} 를 다시 못 찾았습니다")
            return False
        # allow_relocate=False — 이 블록 자신의 이웃까지 옮기기 시작하면 연쇄가
        # 끝없이 번진다(임시 치우기와 같은 이유로 한 단계에서 멈춘다).
        if not self.pick(pose, allow_relocate=allow_chain, _depth=depth + 1):
            self.get_logger().error(f"{color} 를 집지 못했습니다")
            return False
        self.place(dst, angle=ang)
        self.copy_pending.remove(item)
        self.copy_placed[dst] = color
        push_zone("robot", dst, color)
        self.get_logger().info(
            f"완료: 인간{hz_i}번의 {color} → 로봇 {dst}번  (막는 것을 치우며 함께 끝냄)")
        return True

    def relocate_blocker(self, target_xy, axis_deg, neighbors, depth=0):
        """axis_deg 축을 막고 있는 이웃 블록 하나를 옆으로 옮겨 자리를 만든다.

        1. 어느 이웃이 막는지 block_geom.blocking_neighbor 로 찾는다 — 판정
           로직이 approach_gap 과 같아야 "막혔다고 본 것" 과 "옮기는 것" 이
           같은 블록을 가리킨다.
        2. block_geom.relocate_step 으로 목적지를 계산한다(순수 기하 — 필요한
           최소 거리만 옮긴다).
        3. 목적지가 **다른** 블록과 안 겹치는지 비전으로 확인한다. 겹치면
           치울 자리도 없다는 뜻이라 포기한다(억지로 밀어 넣지 않는다).
        4. 그 블록을 집어 목적지에 그대로 내려놓는다(자세는 원래 것 그대로 —
           정렬을 새로 맞출 이유가 없다). 집을 때 allow_relocate=False 로 —
           이 블록 자신의 이웃까지 옮기기 시작하면 연쇄가 끝없이 번질 수 있다.

        성공하면 True. 실패해도 예외를 올리지 않는다 — 부르는 쪽이 원래
        하던 대로(경고 후 파지 포기) 계속하면 된다.
        """
        # **집을 수 있는 이웃**을 고른다. 가장 가까운 이웃(blocking_neighbor)을
        # 그냥 집으려 하면 자기 발에 걸린다 — 그 이웃 입장에서는 원래 목표가
        # 자기 이웃이라(붙어 있으니 막은 것이다) 같은 '좁음' 판정에 걸려 파지가
        # 포기되고, 치우기가 시작조차 못 한다(실측 2026-08-07: "파란색을 치우라" 는
        # 말만 반복했다). 그래서 두 축 중 한쪽이라도 열린 이웃을 고른다.
        # **이 축에서 막고 있고, 그 자신을 집을 수 있는** 이웃을 고른다.
        #
        # 두 가지를 다 봐야 한다. 지금 쓰려는 축과 무관한 블록을 치우면 헛수고고
        # (실측: 축 119.8°에서 막는 것은 파랑·보라인데 주황을 치웠다), 집을 수
        # 없는 블록을 고르면 치우기가 시작조차 못 한다(그 이웃 입장에서는 원래
        # 목표가 자기 이웃이라 같은 '좁음' 판정에 걸린다).
        #
        # 손가락은 **양쪽에서** 내려오므로 한 축을 열려면 그 축의 양쪽을 다
        # 치워야 한다. 한 번에 하나씩 치우고, 목표 위로 돌아가 다시 보는 것을
        # RELOCATE_MAX_TRIES 만큼 되풀이한다 — 앞서 치운 것은 그 재검출에
        # 반영된다(치운 블록은 이미 그 자리에 없다).
        pts = [(x, y) for x, y, _c in neighbors]
        cand = axis_blockers(target_xy, axis_deg, pts)
        if not cand:
            return False                   # 이 축을 막는 게 아예 없다

        # **하나 실패했다고 접지 않는다.** 집을 수 있는 것부터 차례로 시도하고,
        # 그것들이 다 안 되면 못 집는 것까지 내려가 **그 블록의 방해물부터**
        # 치운다(연쇄, RELOCATE_CHAIN 단계까지). 예전에는 최선 하나만 골라
        # 그게 실패하면 통째로 포기했고, 서로 붙은 뭉치는 곧장 사람에게
        # 넘겼다 — 실측 2026-08-08: 보라 하나를 못 집어서 초록까지 같이 죽었다.
        # **팔이 못 닿는 블록은 후보에서 뺀다.** 치우려면 그것을 집어야 하는데
        # 못 가면 몇 번을 해도 실패한다. 실측 2026-08-08 (로그 185606):
        # 노란색(660,84) 를 끌기 6회·들어 옮기기 6회, 합쳐 12번 시도하는 동안
        # 20번의 검출에서 좌표가 1mm 도 안 변했다 — 129초를 그렇게 썼다.
        # 여기서 걸러야 남은 후보나 직각 축으로 넘어갈 수 있다.
        # 이번 파지에서 이미 실패한 blocker 는 다시 고르지 않는다. 같은 이유로
        # 또 실패할 뿐이고, 그 사이 팔만 왕복한다(실측: 같은 노란색에 12번).
        tried = getattr(self, "_failed_blockers", None) or set()
        again = [b for b, _r, _ok in cand
                 if any(np.hypot(b[0] - t[0], b[1] - t[1]) < 15.0 for t in tried)]
        if again:
            where = ", ".join(f"({b[0]:.0f},{b[1]:.0f})" for b in again)
            self.get_logger().info(
                f"이미 실패한 {len(again)}개 [{where}] 는 다시 시도하지 않습니다")
        cand = [(b, r, ok) for b, r, ok in cand
                if not any(np.hypot(b[0] - t[0], b[1] - t[1]) < 15.0 for t in tried)]
        if not cand:
            return False

        far = [b for b, _r, _ok in cand if not self.in_reach(b)]
        if far:
            where = ", ".join(f"({b[0]:.0f},{b[1]:.0f})" for b in far)
            self.get_logger().warn(
                f"막는 것 중 {len(far)}개 [{where}] 는 팔이 못 닿아 건너뜁니다 "
                f"(x > {REACH_X_MAX:.0f})")
        cand = [(b, room, ok) for b, room, ok in cand if self.in_reach(b)]
        if not cand:
            self.get_logger().error(
                "이 축을 막는 블록이 전부 팔이 못 닿는 자리에 있습니다 — "
                "손으로 안쪽으로 옮겨 주세요")
            push_debug("warn", "파지", "막는 블록이 팔 닿는 범위 밖입니다")
            return False

        easy = [(b, room) for b, room, ok in cand if ok]
        hard = [(b, room) for b, room, ok in cand if not ok]
        if hard and depth >= RELOCATE_CHAIN:
            hard = []                      # 더 파고들지 않기로 한 깊이
        if not easy and not hard:
            self.get_logger().error(
                "이 축을 막는 블록들이 서로도 붙어 있고 더 파고들 수 없습니다 — "
                f"블록 하나를 손으로 떼어 주세요 (RELOCATE_CHAIN={RELOCATE_CHAIN})")
            push_debug("warn", "파지",
                       "블록들이 서로 붙어 있어 로봇이 풀 수 없습니다 — 하나를 떼어 주세요")
            return False
        if len(easy) + len(hard) > 1:
            self.get_logger().info(
                f"이 축을 막는 것 {len(cand)}개 — 집을 수 있는 것 {len(easy)}개부터 "
                f"차례로 시도합니다" + (f" (안 되면 갇힌 것 {len(hard)}개까지 파고듭니다)"
                                    if hard else ""))
        for k, (bxy, room) in enumerate(easy + hard, start=1):
            chained = k > len(easy)
            color = next((c for x, y, c in neighbors
                          if np.hypot(x - bxy[0], y - bxy[1]) < 1.0), None)
            if color is None:
                continue
            if chained:
                self.get_logger().warn(
                    f"[{k}/{len(easy) + len(hard)}] {color}({bxy[0]:.0f},{bxy[1]:.0f}) "
                    f"도 갇혀 있습니다(여유 {room:.0f}mm) — 그 블록의 방해물부터 "
                    f"치웁니다 (연쇄 {depth + 1}단계)")
            elif len(easy) + len(hard) > 1:
                self.get_logger().info(
                    f"[{k}/{len(easy) + len(hard)}] {color}({bxy[0]:.0f},{bxy[1]:.0f}) "
                    f"시도 (여유 {room:.0f}mm)")
            # **필요한 블록은 밀지 않는다 — 잡아서 제 구역에 놓는다.**
            # 이 검사가 끌기보다 **먼저** 와야 한다. 끌기가 앞에 있으면, 어차피
            # 로봇구역에 갈 색을 판 위 임시 자리로 60mm 밀어 놓고 나중에 제
            # 차례에 또 집게 된다 — 한 번이면 끝날 일을 두 번 한다.
            item = self.pending_target(color)
            if item is not None:
                if self.relocate_to_zone(item, bxy, allow_chain=chained,
                                         depth=depth):
                    return True
                self.get_logger().warn(
                    f"{color} 를 로봇{item[1]}번으로 곧장 보내지 못했습니다 "
                    "— 밀어서 치워 봅니다")

            # 계획에 없는 색이거나 직행이 실패했으면 **끌기**로 치운다. 잡은 채로
            # 직각 60mm 만 옮기면 되므로 상승·이송·하강·상승 네 번이 빠지고,
            # 판을 최소한만 흐트러뜨린다.
            # 들어서 임시 자리로 옮기는 길(relocate_one)은 기본이 꺼져 있다 —
            # LIFT_RELOCATE 주석 참고. 꺼져 있으면 끌기가 안 될 때 곧장 다음
            # 후보나 직각 축으로 넘어간다.
            if SLIDE_BLOCKERS and self.slide_blocker(
                    bxy, color, target_xy, axis_deg,
                    allow_chain=chained, depth=depth):
                return True
            if LIFT_RELOCATE and self.relocate_one(
                    bxy, color, target_xy, axis_deg,
                    allow_chain=chained, depth=depth):
                return True
            if not hasattr(self, "_failed_blockers") or self._failed_blockers is None:
                self._failed_blockers = set()
            self._failed_blockers.add((bxy[0], bxy[1]))
            self.get_logger().warn(
                f"{color}({bxy[0]:.0f},{bxy[1]:.0f}) 는 안 됐습니다"
                + (" — 다음 후보로 넘어갑니다" if k < len(easy) + len(hard) else ""))
        return False

    def slide_blocker(self, bxy, color, target_xy, axis_deg,
                      allow_chain=False, depth=0):
        """막는 블록을 **잡은 채로 파지축 직각으로 끈다.** 성공하면 True.

        들었다 놓는 것(relocate_one)보다 싸다 — 상승·이송·하강·상승 네 번이
        빠지고, 수평 이동 한 번으로 끝난다. 블록이 공중에 뜨지 않으니 떨어뜨릴
        일도 없고, 놓는 높이 오차도 없다.

        **대신 지나가는 길이 비어 있어야 한다.** 끌려가는 블록이 다른 블록을
        만나면 그것까지 밀어 판이 무너진다 — slide_dests 가 복도를 먼저 본다
        (로봇 없이 시험된다).

        끌 방향은 파지축의 직각이다. 손가락 길(레인)에서 옆으로 빼는 것이라
        축을 따라 밀어내는 것보다 짧고, 목표에서 보면 대각선으로 비켜난다.
        """
        # **지금 보이는 것 + 순회에서 모아 둔 것**을 함께 본다. 지금 시야에만
        # 기대면 그 프레임에서 안 잡힌 블록을 들이받는다 — 실측 2026-08-08 의
        # 충돌이 그것이었다(순회 때 (605,135) 로 봤던 빨강을 못 보고 그 위로 끌었다).
        # **지금 보이는 것만** 본다. 순회 좌표까지 넣어 봤더니 이미 옮긴 블록이
        # 유령 장애물로 남아 끌 길을 통째로 막았다(실측 2026-08-08: 노란색·
        # 파란색 둘 다 "끌 길이 막혀 있습니다" 로 죽었다). 오래된 좌표는
        # 안전을 더해 주지 않고 멀쩡한 길만 없앤다.
        nb = self.neighbors_xy(bxy, radius=SLIDE_MM + BLOCK_MM * 2)
        dests = slide_dests(target_xy, bxy, axis_deg,
                            others=[(x, y) for x, y, _c in nb],
                            clear_extra=SLIDE_CLEAR_EXTRA,
                            check_path=SLIDE_CHECK_PATH)
        # 끌고 간 자리도 팔이 닿아야 한다 — 못 가면 끌다 말고 멈춘다.
        dests = [d for d in dests if self.in_reach(d)]
        if not dests:
            self.get_logger().info(
                f"{color}({bxy[0]:.0f},{bxy[1]:.0f}) 는 끌 길이 막혀 있습니다"
                + (" — 들어서 옮기는 쪽으로 갑니다" if LIFT_RELOCATE
                   else " — 들어서 옮기기가 꺼져 있어(LIFT_RELOCATE=0) 다음 후보로"))
            return False
        dest = dests[0]
        dx, dy = dest[0] - bxy[0], dest[1] - bxy[1]
        self.get_logger().warn(
            f"{color}({bxy[0]:.0f},{bxy[1]:.0f})가 막고 있어 "
            f"({dest[0]:.0f},{dest[1]:.0f})로 **끕니다** "
            f"(파지축 {axis_deg % 180:.0f}° 의 직각으로 {SLIDE_MM:.0f}mm)")
        push_debug("warn", "파지",
                   f"이웃 {color} 끄는 중 → ({dest[0]:.0f},{dest[1]:.0f})")
        pose = self.detect(color, near=bxy, max_dist=CACHE_TOLERANCE)
        if pose is None:
            self.get_logger().error(f"끌 {color} 를 다시 못 찾았습니다")
            return False
        # lift=False — 잡은 자리에 그대로 머문다. 여기서 올라가 버리면 끌기가 아니다.
        if not self.pick(pose, allow_relocate=allow_chain, _depth=depth + 1,
                         lift=False):
            self.get_logger().error(f"{color} 를 집지 못했습니다")
            return False
        # **지금 자세에서 xy 만 옮긴다.** 파지 보정이 이미 반영된 실제 TCP 에서
        # 끌 만큼만 더한다 — 보정을 다시 계산하면 어긋난다.
        try:
            cur = list(get_current_posx()[0])
        except Exception as e:
            self.get_logger().error(f"현재 자세를 못 읽어 끌기를 중단합니다({e})")
            return False
        slid = list(cur)
        slid[0] += dx
        slid[1] += dy
        if not movel(slid):
            self.get_logger().error("끌기 이동에 실패했습니다 — 그 자리에 놓습니다")
            gripper.open_gripper(); wait_gripper()
            up = list(cur); up[2] = max(TRANSIT_Z, cur[2] + self.lift)
            movel(up)
            return False
        gripper.open_gripper(); wait_gripper()
        up = list(slid); up[2] = max(TRANSIT_Z, slid[2] + self.lift)
        movel(up)                       # 빈 손으로 물러난다
        self.get_logger().info(
            f"{color} 를 {np.hypot(dx, dy):.0f}mm 끌었습니다 (들지 않음)")
        return True

    def relocate_one(self, bxy, color, target_xy, axis_deg,
                     allow_chain=False, depth=0):
        """후보 하나를 실제로 치운다. 성공하면 True, 안 되면 False(부르는 쪽이 다음 후보).

        allow_chain 이면 이 블록을 집을 때 **그 블록의 방해물도 치우게** 한다
        (연쇄). 갇힌 블록을 고른 경우라서, 그러지 않으면 집을 수가 없다.
        """
        # 직행(pending_target → relocate_to_zone)은 부르는 쪽에서 **이미** 봤다.
        # 여기까지 왔다는 것은 계획에 없는 색이거나 직행이 실패했다는 뜻이라,
        # 임시 자리로 치우는 것이 맞다.

        # 판 경계. 후보를 걸러야 하므로 **먼저** 구한다.
        # block_sort 에는 판 경계가 없었다 — 경계는 조종 쪽 교시 상자를 빌린다
        # (reach_box). 그것도 없으면 구역 좌표를 감싸는 상자로 물러난다.
        box = reach_box()
        if box is None:
            zs = self.all_zone_xy()
            m = ZONE_RADIUS + BLOCK_MM
            box = (min(p[0] for p in zs) - m, max(p[0] for p in zs) + m,
                   min(p[1] for p in zs) - m, max(p[1] for p in zs) + m)
        bx0, bx1, by0, by1 = box

        # 치울 자리를 **여러 곳** 본다. 한 곳만 계산하면 그 자리가 막혔을 때
        # 그대로 포기한다(실측 2026-08-07: 유일한 후보가 51.6mm 옆 블록 때문에
        # 거부돼 치우기가 무산됐다 — 겹치지도 않는 거리였다).
        # 두 단계로 본다. ① 파지 여유가 나는 자리를 먼저 찾고, 없으면
        # ② 겹치지만 않는 자리로 물러난다 — 임시로 치워 두는 자리다.
        # 세 번째 단계('구역무시')는 RELOCATE_ANYWHERE 일 때만 붙는다 — 프리구역을
        # 먼저 다 보고, 거기에 자리가 없을 때만 구역 안을 허용한다. 판 밖(reach_box)은
        # 어느 단계에서도 허용하지 않는다 — 팔이 닿지 않는 것은 취향이 아니라 물리다.
        cands = relocate_candidates(target_xy, bxy, axis_deg)
        tiers = [("파지여유", BLOCK_MM + FINGER_GAP_MIN, True),
                 ("겹침없음", RELOCATE_CLEAR_MIN, True)]
        if RELOCATE_ANYWHERE:
            tiers.append(("구역무시", RELOCATE_CLEAR_MIN, False))
        dest, tier = None, None
        for tname, need, free_only in tiers:
            for c in cands:
                if free_only and not self.is_free(c):
                    continue                        # 구역 안 — 프리구역을 먼저 본다
                if not (bx0 <= c[0] <= bx1 and by0 <= c[1] <= by1):
                    continue                        # 판 밖
                # self_r=0 — 빈 자리를 보는 것이라 '자기 자신' 이 없다.
                # 옮길 블록 자신만 뺀다(곧 그 자리를 비운다).
                if [n for n in self.neighbors_xy(c, radius=need, self_r=0.0)
                        if np.hypot(n[0] - bxy[0], n[1] - bxy[1]) > 5.0]:
                    continue
                dest, tier = c, tname
                break
            if dest is not None:
                break
        if dest is None:
            self.get_logger().warn(
                f"{color}({bxy[0]:.0f},{bxy[1]:.0f})를 치울 자리를 못 찾았습니다 "
                f"— 후보 {len(cands)}곳(사방 {RELOCATE_WIDE_R[-1]:.0f}mm까지 훑음)이 "
                + ("모두 막힘·판밖입니다" if RELOCATE_ANYWHERE
                   else "모두 막힘·구역·판밖입니다 (RELOCATE_ANYWHERE=1 이면 구역도 씁니다)")
                + ". 사람이 치워 주세요")
            push_debug("warn", "파지", f"{color} 를 치울 자리가 없습니다")
            return False
        self.get_logger().info(
            f"치울 자리 후보 {len(cands)}곳 중 ({dest[0]:.0f},{dest[1]:.0f}) 선택 "
            f"[{tier}] — blocker 에서 {np.hypot(dest[0] - bxy[0], dest[1] - bxy[1]):.0f}mm")
        if tier == "구역무시":
            # 조용히 넘어가면 안 된다 — 배치가 왜 흐트러졌는지 나중에 알 수가 없다.
            self.get_logger().warn(
                "프리구역에 자리가 없어 **구역 안**에 치웁니다 — 그 칸의 배치가 "
                "흐트러집니다 (RELOCATE_ANYWHERE=0 이면 대신 포기합니다)")
            push_debug("warn", "파지", "프리구역에 자리가 없어 구역 안에 치웁니다")
        self.get_logger().warn(
            f"{color}({bxy[0]:.0f},{bxy[1]:.0f})가 막고 있어 "
            f"({dest[0]:.0f},{dest[1]:.0f})로 치웁니다")
        push_debug("warn", "파지",
                   f"이웃 {color} 치우는 중 → ({dest[0]:.0f},{dest[1]:.0f})")
        pose = self.detect(color, near=bxy, max_dist=CACHE_TOLERANCE)
        if pose is None:
            self.get_logger().error(f"치울 {color} 를 다시 못 찾았습니다")
            return False
        if not self.pick(pose, allow_relocate=allow_chain, _depth=depth + 1):
            self.get_logger().error(f"{color} 를 집지 못했습니다")
            return False
        dest_pose = list(pose)
        dest_pose[0], dest_pose[1] = dest
        self.place_at(dest_pose, taught=False)
        return True

    def pick(self, pose, allow_relocate=True, _depth=0, lift=True):
        """접근 → 하강 → 파지 → 확인. 성공하면 True.

        allow_relocate=False 는 relocate_blocker 가 치울 이웃 자신을 집을 때
        쓴다 — 연쇄적으로 또 다른 이웃을 옮기기 시작하지 않게 막는다.

        lift=False 는 **잡은 자리에 그대로 머문다**(끌기용). 들어 올리지 않으므로
        부르는 쪽이 곧바로 수평 이동해 블록을 끌 수 있다 — 상승·하강 두 번이
        통째로 빠진다. 끌기가 끝나면 부르는 쪽이 직접 올라와야 한다.
        """
        if _depth == 0:
            # 새 목표를 집기 시작한다 — 지난 목표에서 실패한 blocker 기록을 비운다.
            # (연쇄로 들어온 호출은 _depth>0 이라 기록을 이어받는다)
            self._failed_blockers = set()
        pose = list(pose)
        # 손목 방향을 먼저 정한다 — 파지 보정이 손목 각도에 딸려 있어서,
        # 보정을 더한 뒤에 돌리면 보정이 틀린 방향으로 남는다.
        # None 이면 옆 블록에 막혀 내려갈 자리가 없다는 뜻이다.
        pose = self.aim_between_neighbors(pose, allow_relocate=allow_relocate,
                                          _depth=_depth)
        if pose is None:
            return False
        g = self.grasp_offset()
        self.get_logger().info(
            f"파지 보정 ({g[0]:+.1f}, {g[1]:+.1f}) @ 손목 {self.last_rot:+.0f}°")
        pose[0] += g[0]
        pose[1] += g[1]
        # 6번 축이 기준 각도면 여기서 함께 얹는다 — 좌표에 미리 반영하므로
        # 접근·하강 경로가 그대로고 추가 이동이 없다.
        t, j6 = j6_tweak(pose)
        if t.any():
            self.get_logger().info(
                f"6번 축 {j6:+.1f}° (기준 {J6_TWEAK_AT:+.0f}±{J6_TWEAK_W:.0f}°) "
                f"— 파지 보정 {J6_TWEAK_FRAME} ({J6_TWEAK_XY[0]:+.0f}, "
                f"{J6_TWEAK_XY[1]:+.0f}) → base ({t[0]:+.1f}, {t[1]:+.1f}) 추가 "
                f"[공구방위 {tool_yaw(pose[3:]):+.0f}°]")
            pose[0] += t[0]
            pose[1] += t[1]
        elif j6 is not None:
            self.get_logger().info(
                f"6번 축 {j6:+.1f}° — 기준({J6_TWEAK_AT:+.0f}±{J6_TWEAK_W:.0f}°) "
                "밖이라 추가 보정 없음")

        # 접근과 이송을 나눈다.
        #
        # 접근은 **빈 손**이라 이송 높이(TRANSIT_Z=250)까지 올릴 이유가 없다.
        # 검출을 마친 자리(관측 높이 ≈165)에서 그리퍼만 블록 위로 수평 이동한 뒤
        # 곧게 내리면 된다. 그 xy 이동이 필요한 이유는 hover_over 가 **카메라**를
        # 블록 위에 세우기 때문이다 — 그리퍼는 핸드아이 오프셋(+32.6,+60.1)만큼,
        # 파지 보정까지 더해 66~77mm 떨어져 있다.
        #
        # 예전에는 이 xy 이동을 z=250 으로 올라가면서 함께 했는데, 그 상승이
        # 씹히면(실측 2026-08-06: 복제 4회 중 2회) 팔이 그 자리에 머문 채 다음
        # 명령이 나가 **xy 68mm 와 하강 178mm 를 동시에** 하게 됐다. 판 위를
        # 대각선으로 훑어 옆 블록을 칠 수 있는 경로다.
        # 접근에서 z 를 아예 안 건드리면 그 실패 경로 자체가 없어진다.
        #
        # 이송(블록을 물고 옮길 때)은 TRANSIT_Z 를 그대로 쓴다 — 물린 블록이
        # 그리퍼 아래로 튀어나와 옆 블록을 치기 때문이고, 이건 빈 손과 사정이 다르다.
        up = list(pose); up[2] = max(TRANSIT_Z, pose[2] + self.lift)
        for attempt in range(1, GRASP_RETRY + 1):
            over = list(pose)
            # 지금 높이를 유지한다. 다만 너무 낮은 자리에서 시작했을 수 있으므로
            # 하한을 둔다 — 블록 윗면(판 위 35mm)을 확실히 넘겨야 한다.
            try:
                over[2] = max(get_current_posx()[0][2], APPROACH_Z)
            except Exception:
                over[2] = max(TRANSIT_Z, pose[2] + self.lift)
            if not movel(over):
                # 수평 접근이 안 되면 내려가지 않는다. 그대로 하강하면 블록 위가
                # 아닌 곳으로 내려가 옆 블록을 친다.
                self.get_logger().error("블록 위로 접근하지 못해 파지를 중단합니다")
                return False
            movel(pose)                      # 순수 z 하강
            gripper.close_gripper()
            wait_gripper()
            if grasped():
                self.get_logger().info(f"파지 성공 (시도 {attempt})")
                if lift:
                    # **순수 수직 상승.** up 은 pose 와 xy 가 같고 z 만 다르다.
                    #
                    # 물린 직후라 컨트롤러가 아직 앞 모션을 정리 중일 수 있고,
                    # 그 틈에 넣은 movel 은 조용히 삼켜진다(실측 2026-08-08:
                    # 263mm 상승이 0.13초씩 3번 다 "안 갔음" 으로 끝났다).
                    # 그래서 **넣기 전에 컨트롤러가 멎기를 기다린다** — 이것만으로
                    # 대부분 해결된다. 그래도 안 되면 경고만 남기고 그대로 간다:
                    # 여기서 접으면 물린 블록을 든 채 명령이 끝나 버린다.
                    wait_idle()
                    self.rise(up[2], "파지 후")
                return True
            self.get_logger().warn(f"빈손 — 시도 {attempt}/{GRASP_RETRY}")
            gripper.open_gripper(); wait_gripper()
            movel(over)                      # 빈 손이므로 접근 높이면 충분하다
        return False

    def place_at(self, p, taught=False):
        """주어진 자세에 내려놓는다.

        taught=True  손으로 티칭한 구역 좌표. 그 값에는 실제 파지 기하가
                     이미 녹아 있으므로 보정을 더하면 안 된다. 더하면
                     그만큼(측정치 13.7mm) 밀려서 놓인다.
        taught=False 비전이 계산한 좌표. 집을 때와 같은 보정이 필요하다.
        """
        p = list(p)
        if not taught:
            g = self.grasp_offset()
            p[0] += g[0]
            p[1] += g[1]
        # **수직으로 올리고 → 수평으로 옮기고 → 수직으로 내린다.** 이 순서를
        # 지켜야 물고 있는 블록이 판 위의 다른 블록을 안 친다.
        up = list(p); up[2] = max(TRANSIT_Z, p[2] + self.lift)

        # 올리는 것을 **여기서 한 번 더 보장한다.** pick() 이 이미 올렸어야 하지만
        # 그 movel 이 씹히는 일이 있다(실측 2026-08-08: 기동 후 첫 파지에서
        # 263mm 상승이 3회 다 무시됐다). 그 상태로 아래 movel(up) 을 넣으면
        # 목적지 위 z=250 으로 가면서 **상승과 xy 이동을 동시에** 하게 되어,
        # 판을 대각선으로 훑는 바로 그 경로가 된다.
        # 지금 자리에서 z 만 올리는 것이라 이미 높으면 movel 이 즉시 통과한다.
        self.rise(up[2], "옮기기 전")

        movel(up)                        # 목적지 위로 (수평 이동)
        movel(p)                         # 순수 z 하강
        gripper.open_gripper(); wait_gripper()
        movel(up)

    def place(self, zone, angle=None):
        """구역에 놓는다. angle 을 주면 그만큼 **손목을 돌린 채** 내려놓는다.

        블록은 손가락에 정렬돼 물려 있다(파지 때 블록 면에 손목을 맞췄다).
        그래서 놓기 직전에 손목을 d 만큼 돌리면 블록도 그만큼 돌아간 채 놓인다 —
        본보기의 기울기를 따라 놓는 것이 곧 '손목을 그만큼 돌려 놓기' 다.
        집는 동작은 건드리지 않는다. 그쪽은 원래 블록 자기 기울기에 맞춰야 한다.

        **왜 검출각을 그대로 쓸 수 있나.** 검출각은 이미지(=공구) 기준이라
        보통은 base 기준으로 바꿔야 한다. 그런데 이 판의 교시 자세들은
        경유점·인간구역·로봇구역이 모두 공구 요를 90° 로 접었을 때 2.5° 안에서
        일치한다(실측 2026-08-07: zones.yaml 의 rz-rx 가 전부 ≈0). 본보기를 본
        자세와 놓는 자세의 공구 방향이 같으므로 변환이 필요 없다.
        교시를 다시 하면서 손목 방향을 바꾸면 이 전제가 깨진다.

        angle=None 이면 예전과 똑같다 — 색·구역 지정 명령은 따라할 본보기가
        없으므로 이 길로 들어오지 않는다.
        """
        p = self.cfg["zones"][zone]
        if angle is not None and COPY_ANGLE:
            d = fold90(angle)
            # 손목을 돌리면 손가락 중심이 TCP 둘레로 돈다(OFFSET_TOOL 만큼 떨어져
            # 있다). 티칭 좌표는 티칭 당시 손목 방향 기준이라, 돌린 만큼 놓이는
            # 자리가 최대 6.5mm(45°에서, 보통 ±20°면 3mm) 밀린다. 놓기 오차
            # 10mm 안이라 지금은 보정하지 않는다 — 보정하려면 티칭 당시 손목각을
            # 알아야 하고, 그건 복제 후 survey_zones 실측으로 확인한 뒤에 넣는
            # 것이 맞다. 지금 짐작으로 넣으면 오차가 늘 수도 있다.
            self.get_logger().info(f"놓기 손목 {d:+.0f}° (본보기 {angle:.0f}°)")
            p = rotate_tool(list(p), d)
        # 구역은 티칭값이므로 보정을 더하지 않는다.
        self.place_at(p, taught=True)

    def _detect_raw(self, target):
        """z 를 티칭값으로 바꾸지 않은 날것의 base 좌표. 든 블록을 잴 때 쓴다."""
        self.req.target = target
        fut = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=SVC_TIMEOUT)
        if fut.result() is None:
            return None
        cam = list(fut.result().depth_position)
        if sum(cam) == 0:
            return None
        posx = get_current_posx()[0]
        return self.to_base(cam[:3], posx), posx

    def calib_held(self, target, n=3):
        """블록을 든 채로 카메라로 보고 TCP 와의 차이를 잰다.

        집었다 놓는 방식은 손가락이 미는 축의 오차만 드러난다. 그 수직
        방향은 블록이 밀리지 않아 놓아도 같은 자리에 떨어지기 때문이다.
        든 상태로 직접 보면 두 축을 한 번에 잰다.
        """
        errs = []
        for i in range(1, n + 1):
            self.go_home()
            p = self.detect(target)
            if p is None:
                self.get_logger().error("검출 실패 — 중단"); break
            if not self.pick(p):
                self.get_logger().error("파지 실패 — 중단"); break
            # pick 이 끝나면 파지 지점 바로 위에 떠 있다. 그 자리에서 든 블록을 본다.
            got = self._detect_raw(target)
            if got is None:
                self.get_logger().warn("든 블록을 못 봤습니다 — 건너뜀")
            else:
                blk, tcp = got
                e = (blk[0] - tcp[0], blk[1] - tcp[1])
                errs.append(e)
                self.get_logger().info(
                    f"[{i}/{n}] TCP ({tcp[0]:.1f}, {tcp[1]:.1f})  "
                    f"블록 ({blk[0]:.1f}, {blk[1]:.1f})  오차 ({e[0]:+.1f}, {e[1]:+.1f})")
            self.place_at(p)          # 원래 자리에 돌려놓는다
        self.go_home()
        if not errs:
            print("  측정 실패\n")
            return None
        a = np.array(errs)
        keep = np.linalg.norm(a - np.median(a, 0), axis=1) <= CALIB_OUTLIER
        a = a[keep]
        m, s = a.mean(0), a.std(0)
        print(f"\n  유효 측정 {len(a)}/{len(errs)}회")
        print(f"  블록이 TCP 대비   x {m[0]:+.2f}  y {m[1]:+.2f}  mm 에 물림")
        print(f"  표준편차          x {s[0]:5.2f}  y {s[1]:5.2f}  mm")
        print(f"\n  중앙을 물게 하려면")
        print(f"    (참고) 이 자세에서의 보정에 ({m[0]:+.1f}, {m[1]:+.1f}) 를 더하세요\n")
        return m

    def calib_axis(self, target, deg, n=3):
        """손목을 deg 돌린 채로 집었다 놓고, 블록이 밀린 양을 잰다.

        평행 그리퍼는 '닫히는 축' 으로만 블록을 밀어 정렬시킨다. 그래서
        집었다 놓으면 그 축의 오차만 드러난다. 손목을 90° 돌려 한 번 더
        재면 나머지 축도 얻는다.
        """
        errs = []
        for i in range(1, n + 1):
            self.go_home()
            p = self.detect(target)
            if p is None:
                self.get_logger().error("검출 실패 — 중단"); break
            if not self.pick(rotate_tool(p, deg - GRASP_ROT)):
                self.get_logger().error("파지 실패 — 중단"); break
            self.place_at(rotate_tool(p, deg - GRASP_ROT))
            self.go_home()
            q = self.detect(target)
            if q is None:
                self.get_logger().error("재검출 실패 — 중단"); break
            e = (q[0] - p[0], q[1] - p[1])
            errs.append(e)
            self.get_logger().info(
                f"[{i}/{n}] 손목 {deg:+.0f}°  오차 ({e[0]:+.1f}, {e[1]:+.1f})")
        if not errs:
            return None
        a = np.array(errs)
        a = a[np.linalg.norm(a, axis=1) <= CALIB_OUTLIER]
        if len(a) == 0:
            print("  쓸 수 있는 측정이 없습니다."); return None
        m, s = a.mean(0), a.std(0)
        print(f"\n  손목 {deg:+.0f}°  유효 {len(a)}/{len(errs)}회")
        print(f"    평균 ({m[0]:+.2f}, {m[1]:+.2f})   편차 ({s[0]:.2f}, {s[1]:.2f})\n")
        return m

    def center(self, target, wrist=0.0):
        """카메라를 블록 위로 수렴시킨 뒤, 손으로 미세조정해 상수를 실측한다.

        내부파라미터·깊이·핸드아이·TCP 오차를 하나씩 잡는 대신, 그 누적분을
        상수 하나로 흡수한다. 수렴 후 그리퍼를 정중앙에 맞추면, 그 이동량이
        곧 GRASP_OFFSET 에 더할 값이다.
        """
        import termios
        import tty

        global GRASP_ROT, ALIGN_TO_BLOCK
        keep_rot, keep_align = GRASP_ROT, ALIGN_TO_BLOCK
        GRASP_ROT, ALIGN_TO_BLOCK = wrist, False   # 각도를 고정해야 분리가 된다
        self.go_home()
        p = self.detect(target)
        GRASP_ROT, ALIGN_TO_BLOCK = keep_rot, keep_align
        if p is None:
            self.get_logger().error(f"'{target}' 미검출")
            return
        above = list(p); above[2] += self.lift
        gripper.open_gripper(); wait_gripper()
        movel(above, vel=VEL_L, acc=ACC_L); mwait()
        movel(p, vel=VEL_L, acc=ACC_L); mwait()

        step = 1.0
        dx = dy = 0.0
        print("\n" + "=" * 58)
        print(" 손가락 사이 정중앙에 블록이 오도록 맞추세요")
        print("=" * 58)
        print("   w / s   x  +  / -        a / d   y  -  / +")
        print("   [ / ]   이동 폭 줄이기 / 키우기")
        print("   Enter   확정 (상수 저장)        q  취소\n")

        fd = sys.stdin.fileno()
        old_attr = termios.tcgetattr(fd)
        try:
            while True:
                sys.stdout.write(
                    f"\r  누적 ({dx:+6.1f}, {dy:+6.1f}) mm   이동폭 {step:4.1f}mm   ")
                sys.stdout.flush()
                tty.setraw(fd)
                k = sys.stdin.read(1)
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
                if k == "q":
                    print("\n  취소\n"); break
                if k in ("\r", "\n"):
                    cur = self.grasp_offset()
                    # numpy 스칼라는 yaml.safe_dump 가 직렬화하지 못한다.
                    o = (float(cur[0]) + dx, float(cur[1]) + dy)
                    print(f"\n\n  손목 {wrist:.0f}° 에서의 총 보정 "
                          f"({o[0]:+.1f}, {o[1]:+.1f}) mm")
                    d = {}
                    if os.path.exists(CENTER_YAML):
                        d = yaml.safe_load(open(CENTER_YAML)) or {}
                    d[float(wrist)] = [round(o[0], 2), round(o[1], 2)]
                    with open(CENTER_YAML, "w") as f:
                        yaml.safe_dump(d, f, default_flow_style=None)
                    print(f"  {CENTER_YAML} 에 저장")
                    have = sorted(d)
                    print(f"  측정된 각도: {have}")
                    if 0.0 in d and 90.0 in d:
                        o0, o9 = np.array(d[0.0]), np.array(d[90.0])
                        dd = o0 - o9
                        t = np.array([(dd[0]-dd[1])/2, (dd[0]+dd[1])/2])
                        c = o0 - t
                        print(f"\n  분리 완료")
                        print(f"    c_base (각도 무관)  ({c[0]:+.2f}, {c[1]:+.2f})")
                        print(f"    t_tool (함께 회전)  ({t[0]:+.2f}, {t[1]:+.2f})")
                        print(f"    → block_sort.py 에 아래를 넣으세요")
                        print(f"      OFFSET_BASE = [{c[0]:.2f}, {c[1]:.2f}]")
                        print(f"      OFFSET_TOOL = [{t[0]:.2f}, {t[1]:.2f}]\n")
                    else:
                        need = [a for a in (0.0, 90.0) if a not in d]
                        print(f"  분리하려면 손목 {need} 에서도 재야 합니다\n")
                    break
                d = {"w": (step, 0), "s": (-step, 0),
                     "a": (0, -step), "d": (0, step)}.get(k)
                if k == "[":
                    step = max(0.2, step / 2); continue
                if k == "]":
                    step = min(10.0, step * 2); continue
                if d is None:
                    continue
                tgt = list(p)
                tgt[0] += dx + d[0]; tgt[1] += dy + d[1]
                try:
                    movel(tgt, vel=20, acc=20); mwait()
                    dx += d[0]; dy += d[1]
                except Exception as e:
                    print(f"\n  이동 실패: {e}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
        movel(above, vel=VEL_L, acc=ACC_L); mwait()
        self.go_home()

    def aim(self, target, hold=25.0):
        """그리퍼를 연 채로 파지 자세까지만 가서 멈춘다.

        간접 측정(집었다 놓고 다시 보기)은 손가락이 닫히는 축만 드러낸다.
        그 수직 방향은 눈으로 봐야 한다. 손가락 사이에 블록이 대칭으로
        들어와 있는지 확인하고, 치우쳤다면 어느 쪽으로 몇 mm 인지 본다.
        """
        global GRASP_ROT, ALIGN_TO_BLOCK
        keep_rot, keep_align = GRASP_ROT, ALIGN_TO_BLOCK
        GRASP_ROT, ALIGN_TO_BLOCK = wrist, False   # 각도를 고정해야 분리가 된다
        self.go_home()
        p = self.detect(target)
        GRASP_ROT, ALIGN_TO_BLOCK = keep_rot, keep_align
        if p is None:
            self.get_logger().error(f"'{target}' 미검출")
            return
        above = list(p); above[2] += self.lift
        gripper.open_gripper(); wait_gripper()
        movel(above, vel=VEL_L, acc=ACC_L); mwait()
        movel(p, vel=VEL_L, acc=ACC_L); mwait()
        rot = GRASP_ROT + (((self.last_angle + 45.0) % 90.0) - 45.0
                           if ALIGN_TO_BLOCK else 0.0)
        print(f"\n  파지 자세로 정지했습니다 (그리퍼 열림)")
        print(f"    목표 (x, y) = ({p[0]:.1f}, {p[1]:.1f})   손목 {rot:+.0f}°")
        print(f"    블록이 손가락 사이 중앙에 있습니까?")
        print(f"    치우쳤다면 로봇 기준 어느 축으로 몇 mm 인지 보세요.")
        print(f"    {hold:.0f}초 뒤 물러납니다.\n")
        time.sleep(hold)
        movel(above, vel=VEL_L, acc=ACC_L); mwait()
        self.go_home()

    def calib_zones(self, target, rounds=2):
        """구역에 놓고 검출해 zones.yaml 을 실측으로 다듬는다.

        구역 좌표는 사람이 블록을 손으로 물려 티칭한 값이다. 그때의 파지
        상태와 로봇이 집는 상태가 달라 그 차이만큼 어긋나 놓인다.
        측정: 명령 C 로 놓았더니 블록이 q 에 있었다 → 오차 e = q - C.
        목표는 원래 티칭 위치 T 이므로 다음 명령은 C_new = T - e 다.
        """
        goal = {z: list(p) for z, p in self.cfg["zones"].items()}   # 원래 티칭값
        for r in range(1, rounds + 1):
            print(f"\n  ── {r}회차 ──")
            for z in sorted(goal):
                self.go_home()
                p = self.detect(target)
                if p is None:
                    print(f"  {z}번  검출 실패 — 건너뜀"); continue
                if not self.pick(p):
                    print(f"  {z}번  파지 실패 — 건너뜀"); continue
                cmd = list(self.cfg["zones"][z])
                self.place(z)
                self.go_home()
                q = self.detect(target)
                if q is None:
                    print(f"  {z}번  재검출 실패 — 건너뜀"); continue
                e = (q[0] - cmd[0], q[1] - cmd[1])
                new = [goal[z][0] - e[0], goal[z][1] - e[1]] + list(cmd[2:])
                self.cfg["zones"][z] = new
                print(f"  {z}번  오차 ({e[0]:+6.1f}, {e[1]:+6.1f})  →  "
                      f"명령 ({cmd[0]:.1f}, {cmd[1]:.1f}) → ({new[0]:.1f}, {new[1]:.1f})")
        self.go_home()
        out = dict(self.cfg)
        out["zones"] = {int(k): [round(float(v), 2) for v in p]
                        for k, p in self.cfg["zones"].items()}
        with open(ZONES_YAML, "w") as f:
            yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False,
                           default_flow_style=None)
        print(f"\n  {ZONES_YAML} 갱신 완료\n")

    def calibrate(self, target, n=3):
        """집었다 같은 자리에 도로 놓고 다시 검출해 파지 중심 오차를 잰다.

        파지 중심이 e 만큼 밀려 있으면, 집을 때 블록이 손가락 중심으로
        끌려오고 놓을 때 그 자리에 남으므로 블록은 P+e 에 놓인다.
        따라서 (재검출 위치 − 원래 위치) 가 곧 e 다.
        """
        self.ready_gripper("파지 보정 측정")
        errs = []
        for i in range(1, n + 1):
            self.go_home()
            p1 = self.detect(target)
            if p1 is None:
                self.get_logger().error("검출 실패 — 중단")
                break
            if not self.pick(p1):
                self.get_logger().error("파지 실패 — 중단")
                break
            self.place_at(p1)
            self.go_home()
            p2 = self.detect(target)
            if p2 is None:
                self.get_logger().error("재검출 실패 — 중단")
                break
            e = (p2[0] - p1[0], p2[1] - p1[1])
            errs.append(e)
            self.get_logger().info(
                f"[{i}/{n}] 원래 ({p1[0]:.1f}, {p1[1]:.1f}) → "
                f"재검출 ({p2[0]:.1f}, {p2[1]:.1f})   오차 ({e[0]:+.1f}, {e[1]:+.1f})")
        if not errs:
            return None
        a = np.array(errs)
        # 다른 블록을 봤거나 블록이 튄 경우가 섞인다. 파지 오차가 수십 mm 일
        # 수는 없으므로 걸러낸다. 남은 것만으로 평균을 낸다.
        keep = np.linalg.norm(a, axis=1) <= CALIB_OUTLIER
        if (~keep).any():
            for e in a[~keep]:
                print(f"  이상치 제외  ({e[0]:+.1f}, {e[1]:+.1f})")
        a = a[keep]
        if len(a) == 0:
            print("  쓸 수 있는 측정이 없습니다.")
            return None
        m, s = a.mean(0), a.std(0)
        print(f"\n  유효 측정 {len(a)}/{len(errs)}회")
        print(f"  평균 오차   x {m[0]:+.2f}  y {m[1]:+.2f}  mm")
        print(f"  표준편차    x {s[0]:5.2f}  y {s[1]:5.2f}  mm")
        print(f"\n  block_sort.py 의 GRASP_OFFSET 을 이렇게 바꾸세요")
        print(f"    GRASP_OFFSET = [{-m[0]:.1f}, {-m[1]:.1f}]\n")
        return m

    def survey_zones(self):
        """움직이기 전에 로봇구역 배치를 확인하고 대시보드에 반영한다.

        복제(copy_human)는 전체 순회를 돌아 점유를 알지만, 색 지정/구역 지정
        명령은 그냥 집으러 갔다. 그래서 화면의 구역 색이 지난 명령 시점에 머물러
        실제 판과 어긋났다 — 사람이 손으로 블록을 치우거나 놓으면 특히 그렇다.

        전체 순회(5곳)는 비싸므로 **로봇구역이 보이는 경유점만** 들른다.
        어느 경유점이 어느 구역을 보는지는 핸드아이 오프셋(+32.6,+60.1) 때문에
        팔 위치와 다르다 — 실측 시야중심으로 2번이 안쪽(로봇3·4),
        3번이 바깥쪽(로봇1·2)을 본다.
        """
        self.ready_gripper("구역 확인")
        pts = self.scan_points()
        if not pts:
            return {}
        occ, angles = {}, {}
        for i in ZONE_SURVEY_POINTS:
            if i >= len(pts):
                continue
            self.goto(pts[i], f"[구역확인 {i + 1}/{len(pts)}]")
            _, _, o, a = self.look(hz={})   # 인간구역 판정은 여기서 필요 없다
            occ.update(o)
            angles.update(a)
        zones = sorted(self.cfg["zones"])
        # 기울기까지 찍는다. 복제가 본보기 각도를 따라 놓았는지 여기서 확인한다
        # (복제 직후 이 값이 인간구역에서 읽은 각도와 같아야 한다).
        self.get_logger().info(
            "로봇구역 현황 — " +
            ", ".join(
                f"{z}번=" + (f"{occ[z]} {angles.get(('robot', z), 0.0):.0f}°"
                             if occ.get(z) else "비어있음")
                for z in zones))
        push_zones("robot", occ, zones)     # 안 보인 칸은 비었다고 반영된다
        return occ

    def run_one(self, target, zone):
        """target 은 색 이름(str) 또는 집어올 구역 번호(int).

        구역 번호를 주면 그 구역에 있는 블록을 색과 무관하게 집는다.
        """
        self.ready_gripper(f"{target} → {zone}번")
        if zone not in self.cfg["zones"]:
            self.get_logger().error(f"{zone}번 구역이 없습니다. "
                                    f"가능: {sorted(self.cfg['zones'])}")
            return False
        if isinstance(target, int):
            if target not in self.cfg["zones"]:
                self.get_logger().error(f"{target}번 구역이 없습니다. "
                                        f"가능: {sorted(self.cfg['zones'])}")
                return False
            if target == zone:
                self.get_logger().error(f"{target}번에서 집어 {zone}번에 놓으라는 건 "
                                        "제자리입니다.")
                return False

        # 움직이기 전에 로봇구역 현황을 확인한다. 화면과 실제 판을 맞추기 위해서다.
        self.survey_zones()

        if isinstance(target, int):
            # 그 구역에 놓인 것을 집는 경우. 구역 위에 있으므로 순회로 찾지 않는다.
            self.go_home()
            color = self.detect_in_zone(target)
            if color is None:
                return False
            # **구역 좌표를 near 로 넘긴다.** 안 넘기면 _detect_all 이 신뢰도 순으로
            # 준 첫 번째를 집는다 — 같은 색이 여러 개면 엉뚱한 것을 집는다.
            # 실측 2026-08-06: 1번 구역(454.8, 221.9) 명령에 23mm 짜리를 두고
            # 240mm 떨어진 것을 집었다. detect_in_zone 이 거리로 색을 골라놓고도
            # 위치를 버려서 그 판단이 여기서 무효가 됐다.
            zx, zy = self.cfg["zones"][target][:2]
            pose = self.detect(color, near=(zx, zy), max_dist=ZONE_PICK_MAX)
            if pose is None:
                self.get_logger().error(
                    f"{target}번 구역에서 {color} 를 다시 못 찾았습니다.")
                return False
            self.get_logger().info(f"파지 목표 {[round(v, 1) for v in pose[:3]]}")
            if not self.pick(pose):
                self.get_logger().error("파지 실패 — 중단")
                self.go_home()
                return False
            self.place(zone)
            self.go_home()
            self.get_logger().info(f"완료: {target}번구역의 {color} → {zone}번")
            push_zone("robot", target, None)     # 원래 있던 자리는 비었다
            push_zone("robot", zone, color)
            return True

        # 색을 지정한 경우. 관심 영역을 돌다가 그 색을 프리구역에서 보면 멈춘다.
        _, stock, _, at, _ = self.patrol(want={target})
        lst = stock.get(target) or []
        if not lst:
            self.get_logger().error(f"프리구역에서 {target} 을(를) 못 찾았습니다.")
            push_stock(target, "명령을 수행할 수 없습니다")
            self.go_home()
            return False
        # 같은 색이 여럿 보이면 **집기 쉬운 것**을 고른다. 검출 신뢰도 1등이
        # 사방에 포위돼 있으면, 바로 옆에 뻥 뚫린 같은 색을 두고도 이웃 치우기가
        # 발동한다 — 고를 수 있을 때 고르는 편이 치우기보다 싸다.
        lst = self.graspable_first(lst, stock)
        xy = (lst[0]["pose"][0], lst[0]["pose"][1])
        self.get_logger().info(
            f"{target} 발견 ({xy[0]:.0f}, {xy[1]:.0f})"
            + (f" — {at}번 경유점" if at else ""))
        ok = self.pick_cached(target, xy, zone)
        self.go_home()
        if ok:
            push_zone("robot", zone, target)
        return ok

    def push_hardware(self):
        """TCP 와 카메라 연결 상태를 대시보드로 보낸다.

        RealSense 는 **프레임이 실제로 오는지** 로 판단한다. 발행자 수로 보면
        안 된다 — 실측 2026-08-06: USB 가 빠져도 realsense2_camera_node 는 살아
        남아 발행자가 계속 1 로 잡혔고, 대시보드가 정상이라고 거짓 보고했다.
        정작 이 표시가 필요한 상황을 못 잡은 것이다.
        영상 대신 같은 노드가 같은 주기로 내보내는 camera_info 를 본다 —
        수백 바이트라 구독해도 대역폭에 부담이 없다.

        웹캠은 발행자 수로 충분하다. webcam_publisher 는 장치를 못 열면 아예
        종료되므로(sys.exit) 노드가 살아 있는 것이 곧 장치가 살아 있는 것이다.
        """
        # 여기서 spin 하지 않는다. rclpy.shutdown() 과 겹치면 무효 컨텍스트로
        # 타이머를 만들려다 터진다 (실측 2026-08-06: RCLError, 컨텍스트 무효).
        # camera_info 콜백은 검출 서비스 호출(spin_until_future_complete) 때
        # 함께 처리되므로, 갱신 주기가 조금 늦을 뿐 판정은 유지된다.
        if not rclpy.ok():
            return

        cams = {}
        age = None if self._cam_info_t is None else time.time() - self._cam_info_t
        cams["realsense"] = age is not None and age < CAM_STALE_SEC
        try:
            cams["webcam"] = self.count_publishers("/webcam/image_raw/compressed") > 0
        except Exception:
            cams["webcam"] = False
        # 검출은 카메라가 살아 있어도 노드가 죽으면 못 쓴다. 따로 본다.
        cams["detection"] = bool(self.cli.service_is_ready())
        _admin_post("/api/hardware", {"tcp": tcp_info(), "cameras": cams})

    def run_teleop(self, rec):
        """제어모드 — 손동작으로 팔을 직접 민다. 돌아올 때까지 여기서 머문다.

        **이 프로세스의 DSR 연결과 그리퍼를 그대로 넘긴다.** 조종이 별
        프로세스였을 때 한 로봇에 연결이 둘이 되어 TCP 가 풀리고 모션이
        거부됐다(teleop_mode.py 머리말 참고).

        웹캠도 수어 인식기가 열어둔 것을 그대로 쓴다. V4L2 는 한 프로세스만
        열 수 있고, ROS 토픽이라도 같은 프로세스에서 두 번 구독할 이유가 없다.
        """
        if control_taken():
            # 밖에 조종 프로세스가 살아 있다. 그쪽도 DSR 에 붙어 speedl 을
            # 쏘므로 한 로봇에 지령이 둘이 된다 — 들어가지 않는다.
            self.get_logger().error(
                "밖에서 hand_gesture_control.py 가 돌고 있습니다 — 제어모드는 "
                "이제 이 프로세스 안에서 돕니다. 그 프로세스를 끄고 다시 하세요.")
            push_debug("warn", "모드",
                       "외부 조종 프로세스가 떠 있어 모드변경을 취소했습니다")
            return False

        import threading
        sys.path.insert(0, HERE)
        import teleop_mode

        self.get_logger().warn(
            "⇄ 모드변경 — 제어모드. 돌아오는 방법 셋: "
            "양손 3초 펴기 / 조종창 Q / 대시보드에서 작업모드 전환.")
        push_mode("control")
        push_debug("info", "모드", "제어모드 — 손동작 조종")
        self.go_home()          # 조종은 알려진 자세에서 시작한다
        push_control_alive()    # 이 프로세스가 제어를 잡았다

        # 대시보드 왕복은 **별 스레드**에서 한다. 조종 루프에서 직접 부르면
        # HTTP 타임아웃 1초 동안 프레임 읽기도 경계 판정도 멈춘다 — 속도 지령은
        # 명령을 안 보내도 계속 움직이므로, 그 1초에 팔은 20mm 를 더 간다.
        # 이 스레드는 urllib 만 쓴다. rclpy 를 건드리면 전역 실행기를 다퉈
        # 터진다(실측 2026-08-06).
        watch = {"stop": False, "run": True}

        def _watch():
            while watch["run"]:
                push_control_alive()
                if get_mode(default="control") != "control":
                    watch["stop"] = True     # 대시보드에서 작업모드로 돌렸다
                    return
                time.sleep(1.0)

        threading.Thread(target=_watch, daemon=True).start()

        try:
            why = teleop_mode.run(
                rec._camera(),          # 수어 인식기가 쓰는 그 카메라 (위 docstring)
                dsr={"speedl": speedl, "get_posx": get_current_posx},
                gripper=gripper,
                on_frame=control_frame_sink(),
                should_stop=lambda: watch["stop"],
                max_sec=CONTROL_MAX_SEC,
                log=self.get_logger().info)
        except Exception as e:
            # 조종이 터져도 작업모드는 살아 있어야 한다. 이유를 남기고 복귀한다.
            self.get_logger().error(f"제어모드 오류: {e}")
            push_debug("error", "제어모드", str(e))
            why = f"오류: {e}"
        finally:
            watch["run"] = False
            push_mode("work")

        self.get_logger().warn(f"⇄ 작업모드로 복귀했습니다 — {why}")
        push_debug("info", "모드", f"작업모드 복귀 — {why}")
        return True

    def run_sign(self, once=False):
        """수어 한 문장 → LLM 해석 → 구역 배치. 기본은 반복이다.

        인식기 로딩(torch + mediapipe)이 몇 초 걸리므로 이 모드에서만 import 한다.
        다른 모드까지 그 비용을 물면 calib 처럼 여러 번 돌리는 작업이 답답해진다.
        """
        sys.path.insert(0, HERE)
        import sign_command as sc

        rec = sc.make_recognizer()
        zones = sorted(self.cfg["zones"])
        colors = ", ".join(sorted(sc.GLOSS_TO_COLOR))
        # 이 프로세스가 떴다는 것 자체가 작업모드가 활성이라는 뜻이다.
        # 안 알리면 대시보드가 지난 실행의 control 상태를 그대로 들고 있어,
        # 첫 명령 전에 이미 '제어모드' 로 보인다.
        ensure_tcp(self.get_logger())     # 파지 좌표가 전부 이 TCP 기준이다
        push_mode("work")
        self.push_hardware()               # TCP·카메라 상태를 한 번 올린다
        try:
            return self._sign_loop(sc, rec, zones, colors, once)
        finally:
            rec.close()            # 카메라와 창은 여기서 딱 한 번 닫는다

    def _sign_loop(self, sc, rec, zones, colors, once):
        while True:
            # 명령을 기다리기 직전에 갱신한다. create_timer 로는 안 된다 —
            # 이 루프는 인식·모션에 블로킹되어 있어 실행기가 안 돌고, 그러면
            # 타이머 콜백이 영영 안 불린다. 여기라면 팔이 멎어 있어 DSR 서비스
            # 호출도 안전하다. 대시보드를 재시작해도 다음 명령 때 다시 채워진다.
            self.push_hardware()
            self.get_logger().info(
                f"수어로 명령하세요 — 예: 빨강 → {zones[0]}번구역  "
                f"(색: {colors} / 구역: {zones})")
            got = sc.collect_command(
                rec, zones,
                on_gloss=lambda l, p, b: self.get_logger().info(
                    f"  · {l} ({p * 100:.0f}%){'  [동사 가중]' if b else ''}"),
                hint=f"빨강 → {zones[0]}번구역")
            if got is None:                # 창에서 Q — 사용자가 그만두겠다는 뜻이다
                self.get_logger().info("취소됨")
                return False
            glosses, steps, how = got
            text = " ".join(glosses)
            if sc.MODE_GLOSS in glosses:
                # 손동작 조종으로 넘어간다. **같은 프로세스 안에서** 돈다 —
                # 이 함수가 돌아올 때까지 수어 명령은 받지 않으므로, 제어모드
                # 중에 수어가 오인식돼 팔이 멋대로 나가는 일도 없다.
                self.ready_gripper("조종 모드")
                ok = self.run_teleop(rec)
                if once:
                    return ok
                continue
            if not steps:
                self.get_logger().warn(f"'{text}' 를 해석하지 못했습니다 — 다시 하세요.")
                if once:
                    return False
                continue
            self.get_logger().warn(
                f"[{how}] '{text}' → " +
                ", ".join(f"{sc.describe(s)}→{z}번" for s, z in steps))
            ok = True
            self._svc_down = False        # 명령마다 새로 판단한다
            for src, zone in steps:
                if src == "copy":
                    # zone 자리에 mirror 여부가 실려 온다 (sign_command.COPY_GLOSS)
                    if not self.copy_human(mirror=bool(zone)):
                        ok = False
                    continue
                if not self.run_one(src, zone):
                    # 한 짝이 실패해도 나머지는 계속한다. 블록 하나를 못 찾았다고
                    # 방금 서명한 나머지 명령까지 버리면 다시 다 서명해야 한다.
                    self.get_logger().warn(
                        f"{sc.describe(src)} → {zone}번 실패 — 다음으로 넘어갑니다")
                    ok = False
            if once:
                return ok


def as_target(s):
    """숫자면 '그 구역에 있는 것', 아니면 색 이름. 구역 번호에 색 이름이 없으니 안 겹친다."""
    return int(s) if s.isdigit() else s


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    m = sys.argv[1]
    node = BlockSort()
    push_robot_status(connected=True, model="Doosan M0609",
                       last_command=" ".join(sys.argv[1:]))
    try:
        if m == "home":
            node.go_home()
        elif m == "observe":
            node.go_home()
            t = sys.argv[2] if len(sys.argv) > 2 else "Apple"
            p = node.detect(t)
            print(f"\n  '{t}' → {'못 찾음' if p is None else [round(v,1) for v in p[:3]]}\n")
        elif m == "pick":
            if len(sys.argv) < 4:
                sys.exit("사용법: pick <색|집을구역> <놓을구역>")
            node.run_one(as_target(sys.argv[2]), int(sys.argv[3]))
        elif m == "calib":
            if len(sys.argv) < 3:
                sys.exit("사용법: calib <대상> [횟수]")
            node.calib_held(sys.argv[2],
                            int(sys.argv[3]) if len(sys.argv) > 3 else 3)
        elif m == "center":
            if len(sys.argv) < 3:
                sys.exit("사용법: center <대상>")
            node.center(sys.argv[2],
                        float(sys.argv[3]) if len(sys.argv) > 3 else 0.0)
        elif m == "aim":
            if len(sys.argv) < 3:
                sys.exit("사용법: aim <대상> [정지초]")
            node.aim(sys.argv[2],
                     float(sys.argv[3]) if len(sys.argv) > 3 else 25.0)
        elif m == "calib-zones":
            if len(sys.argv) < 3:
                sys.exit("사용법: calib-zones <대상> [회차]")
            node.calib_zones(sys.argv[2],
                             int(sys.argv[3]) if len(sys.argv) > 3 else 2)
        elif m == "calib-axis":
            if len(sys.argv) < 4:
                sys.exit("사용법: calib-axis <대상> <손목각> [횟수]")
            node.calib_axis(sys.argv[2], float(sys.argv[3]),
                            int(sys.argv[4]) if len(sys.argv) > 4 else 3)
        elif m == "calib-drop":
            if len(sys.argv) < 3:
                sys.exit("사용법: calib-drop <대상> [횟수]")
            node.calibrate(sys.argv[2],
                           int(sys.argv[3]) if len(sys.argv) > 3 else 3)
        elif m in ("scan", "read-human", "read-free"):
            seen, stock, occ, _, angles = node.patrol()
            push_zones("human", seen, sorted(node.human_zones()))
            push_zones("robot", occ, sorted(node.cfg["zones"]))
            node.go_home()

            # 기울기까지 찍는다. 복제가 본보기 각도를 따라 놓았는지 보려면
            # 두 줄(인간/로봇)의 각도를 나란히 봐야 한다.
            def _with_ang(space, d):
                if not d:
                    return None
                return {z: f"{c} {angles.get((space, z), 0.0):+.0f}°"
                        for z, c in sorted(d.items())}

            print(f"\n  인간구역 = {_with_ang('human', seen) or '읽기 실패'}")
            counts = {c: len(v) for c, v in stock.items()}
            print(f"  프리 재고 = {counts}")
            print(f"  로봇구역 점유 = {_with_ang('robot', occ) or '비어 있음'}\n")
        elif m == "copy":
            node.copy_human(mirror=False)
        elif m == "copy-mirror":
            node.copy_human(mirror=True)
        elif m == "sign":
            node.run_sign(once="--once" in sys.argv)
        elif m == "run":
            print("\n  '<대상> <구역>' 입력.  q 로 종료")
            print(f"  구역: {sorted(node.cfg['zones'])}\n")
            while True:
                s = input("  > ").strip().split()
                if not s or s[0] == "q":
                    break
                if len(s) != 2 or not s[1].isdigit():
                    print("    형식: 빨간색 1   또는   3 1 (3번 구역의 것을 1번으로)")
                    continue
                node.run_one(as_target(s[0]), int(s[1]))
        else:
            sys.exit(__doc__)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl-C 나 kill 은 오류가 아니다. 역추적을 쏟아내지 않는다.
        print("\n중단됨")
    except Exception as e:
        push_debug("error", "block_sort", f"'{' '.join(sys.argv[1:])}' 실행 중 오류: {e}")
        raise
    finally:
        push_robot_status(connected=False)
        time.sleep(0.3)                  # 대시보드로 가는 마지막 전송을 기다린다
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
