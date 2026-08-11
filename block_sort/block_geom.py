#!/usr/bin/env python3
"""블록 자세 기하 — 로봇도 ROS 도 필요 없는 부분만 모았다.

block_sort.py 는 import 하는 순간 DSR 에 붙기 때문에 로봇 없이는 한 줄도
확인할 수 없다. 그런데 놓는 각도를 정하는 계산은 순수 기하라서 책상에서
검증할 수 있는 것들이다 — 여기 옮겨 두면 test_copy_angle.py 가 **실제로 쓰는
함수**를 그대로 시험할 수 있다. 손목 회전은 실기에서 틀리면 블록을 모서리로
물거나 옆 블록을 치는 쪽이라, 미리 확인할 값이 있으면 확인하는 게 낫다.
"""
import os

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
# 실측 25mm (2026-08-07, RG2 손가락 끝) + 여유.
# **이 값이 곧 필요한 블록 간격이다** — 중심거리로는 35 + 34 = 69mm 이상.
#
# **27 → 38 → 34 (2026-08-11).** 아래는 38 로 올릴 때의 근거이고, 그 뒤 실기에서
# 34 로 내렸다. 이유는 이 주석 끝의 '38 이 비쌌던 이유' 를 보라.
#
# 27 은 손가락 25mm 에 여유 2mm 였다. 그런데 이 판단이 쓰는 좌표에는 그보다
# 큰 오차가 이미 들어 있다:
#     목표 중심 검출 오차     4mm 안팎  (DETECT_SAMPLES 주석의 '재현성 바닥')
#     이웃 중심 검출 오차     4mm 안팎  (같은 값)
#     이웃 기울기 오차        각도 4° 가 흔들리면 반폭이 1~2mm 움직인다
# 즉 "틈 30mm" 라고 본 자리가 실제로는 22mm 일 수 있다 — 손가락이 안 들어간다.
# 여유 2mm 는 이 오차들 앞에서 있으나 마나였고, 실제로 기울어진 이웃의 모서리를
# 반복해서 쳤다(2026-08-10 로그 python3_12362, 2026-08-11 실기).
# 25 + 4 + 4 + 2 = 33 이 '오차를 다 먹고도 손가락이 들어가는' 선이고,
# **38 은 거기에 5mm 를 더 얹은 값**이다. 오차 추정 자체가 몇 개의 실측을
# 합친 것이라 그 추정이 낙관적일 수 있고, 부딪히면 보호정지로 작업이 끊긴다 —
# 치우기 한 번보다 훨씬 비싸다. 그래서 경계에서는 치우는 쪽으로 기울인다.
# 뜻으로 읽으면: 이웃과의 틈에 **블록 한 변(35mm)보다 넓게** 떨어져 있지
# 않으면 집지 않고 치운다.
#
# 대가는 치우기가 늘어나는 것뿐이다. 무작위 판 400개(블록 6개) 검증:
#     FINGER_T   바로 집기   치우면 됨   사람 호출
#         27       85.2%      14.8%       0.0%
#         31       82.9%      17.1%       0.0%
#         35       79.9%      20.1%       0.0%
#         38       78.0%      22.0%       0.0%
# 38 에서도 **치울 자리를 못 찾는 판은 0개**다 — 더 조심스러워질 뿐, 사람을
# 부르는 일이 늘지는 않는다. 같은 날 실기 로그(python3_96039)에서 실제로 집은
# 틈은 27, 30, 34, 40, 44, 48, 49, 49mm 였으니 **앞의 셋이 치우기로 넘어간다**
# (40mm 부터는 그대로 집는다 — 38 과 40 사이가 다음 경계다).
#
# ── 38 이 비쌌던 이유 (실기 2026-08-11, 로그 python3_414742) ──────────
# 위 표는 '치우기 한 번' 을 한 번의 이동으로 셌는데, 실기에서는 그렇지 않았다.
# 복제 중 파란색(593,86)이 두 축 다 막혔다:
#     축 91°  주황색(569,9,46°)   틈 34.0mm
#     축  1°  빨간색(544,85,88°)  틈 13.4mm
# 38 기준으로는 둘 다 부족이라 치우기로 넘어갔는데, 치우려던 주황색 자신이
# 갇혀(-7mm) 실패했고, 직각 축의 빨간색은 그 자신도 갇혀서 연쇄로 들어갔다.
# 그 결과 **상관없는 보라색까지 끌려 나왔고**, 그 끌어 놓은 자리가 하필
# 원래 목표(파란색) 쪽이었다(69mm → 50mm). 파란색 하나를 집자고 판 절반이
# 움직인 셈이다. 27~34 였다면 34.0mm 를 그냥 집고 끝났을 일이다.
#
# **경계 하나를 잘못 잡으면 이동 한 번이 아니라 연쇄 전체가 붙는다** — 판이
# 빽빽할수록 그렇다(블록 10개면 치우기 39% → 53%). 그래서 '오차를 다 먹고도
# 손가락이 들어가는 선' 인 33 바로 위, **34** 로 둔다. 30mm 짜리(2026-08-11
# 아침에 모서리를 쳤던 그 케이스)는 여전히 걸러진다.
#
# **34.0 은 위 실기 케이스와 소수점까지 붙어 있다.** 그 파란색의 틈이 34.02mm
# 라 지금 값으로 간신히 통과한다 — 검출이 0.1mm만 달라져도 다시 치우기로
# 넘어갈 수 있다. 그 케이스를 확실히 집게 하려면 33 으로 내린다.
#
# 조정은 환경변수 하나로 두 값이 함께 움직인다:
#   FINGER_GAP_MIN=33 ./run_all.sh      # 위 케이스에 여유를 준다
#   FINGER_GAP_MIN=38 ./run_all.sh      # 다시 조심스럽게
#   FINGER_GAP_MIN=27 ./run_all.sh      # 예전 값(여유 2mm)
# **block_sort.FINGER_GAP_MIN 과 반드시 같은 값이어야 한다.** 여기만 올리면
# 판단(치울 이웃 고르기)과 실행(파지 거부)이 갈려서, 그 사이 구간의 블록은
# "집지도 않고 치우지도 않는" 상태가 된다. 그래서 같은 환경변수를 읽는다.
FINGER_T = float(os.environ.get("FINGER_GAP_MIN", 34.0))
# 손가락 패드가 축과 나란히 덮는 폭의 절반. 이보다 옆으로 비켜 있는 블록은
# 손가락이 지나가도 안 닿는다.
LANE_PAD = 12.0          # 패드 절반 길이
LANE_HALF = BLOCK_MM / 2.0 + LANE_PAD    # 29.5 — 이웃 기울기를 모를 때의 값


def nb_half(nb, axis_deg, block_mm=BLOCK_MM):
    """이웃이 axis_deg 방향으로 내미는 **반폭**(mm).

    **정사각형은 돌아가면 그 방향 폭이 커진다.** 한 변 s 인 정사각이 φ 만큼
    돌면 어느 방향으로 내미는 폭은 s(|cosφ| + |sinφ|) 다 — 45° 에서 s√2 로
    최대가 된다. 반폭으로는 17.5 → 24.7mm, **7.2mm** 차이다.

    이걸 안 보면 판정이 그만큼 **낙관적**이 된다. 실측 2026-08-10 (로그
    python3_12362): 목표 노란색(547,-74) 파지축 127° 에서 주황색(503,-21) 의
    틈을 34mm 로 보고 "충분(≥27)" 판정해 그대로 내려갔는데, 그리퍼가 그
    주황색에 부딪혀 컨트롤러가 보호정지했다(STANDBY 20초 대기 실패 → 이동
    거부). 주황색이 30~45° 돌아 있었다면 실제 틈은 27mm 이하다.
    같은 로그에서 집힌 기울기가 12·14·15·16·20·23·24·53·88·89° 였다 —
    이 판의 블록은 축과 나란하지 않다.

    **목표 자신은 이 보정이 필요 없다.** 손목을 블록 면에 맞춰 내려가므로
    (ALIGN_TO_BLOCK) 목표는 축 방향으로 정확히 반폭 17.5mm 를 내민다.
    보정이 필요한 것은 **이웃뿐**이다.

    nb 는 (x, y) 또는 (x, y, 기울기deg). 기울기가 없으면 축 정렬로 본다 —
    예전과 같은 값이라 각도를 안 넘기던 호출부는 동작이 안 바뀐다.
    각도를 아는 쪽(block_sort.neighbors_xy)이 넘기면 그때부터 정확해진다.
    """
    import numpy as np
    if len(nb) < 3 or nb[2] is None:
        return block_mm / 2.0
    r = np.radians(float(nb[2]) - float(axis_deg))
    return block_mm / 2.0 * (abs(np.cos(r)) + abs(np.sin(r)))


def _pt(n):
    """이웃 하나를 (x, y) 또는 (x, y, 기울기deg) 로 정규화한다.

    부르는 쪽이 (x, y, 색) 처럼 세 번째에 다른 것을 담아 오는 일이 없도록
    **각도만** 통과시킨다. 각도가 None 이면 두 값짜리로 줄인다.
    돌려주는 것이 늘 2 또는 3 튜플이라 nb_half/nb_lane 이 그대로 읽는다.
    """
    if len(n) > 2 and n[2] is not None:
        return (float(n[0]), float(n[1]), float(n[2]))
    return (float(n[0]), float(n[1]))


def nb_lane(nb, axis_deg, lane_half=LANE_HALF, block_mm=BLOCK_MM,
            lane_pad=LANE_PAD):
    """이 이웃에게 적용할 레인 반폭(mm). 축의 **직각** 방향 반폭 + 패드 절반.

    레인 판정도 같은 이유로 기울기를 봐야 한다 — 돌아간 블록은 옆으로도 더
    내밀어서, 축 정렬로 재면 '비켜 있다' 고 잘못 넘긴다.
    기울기를 모르면 예전 값(lane_half)을 그대로 쓴다.
    """
    if len(nb) < 3 or nb[2] is None:
        return lane_half
    return nb_half(nb, axis_deg + 90.0, block_mm=block_mm) + lane_pad


def approach_gap(target_xy, axis_deg, neighbors, block_mm=BLOCK_MM,
                 lane_half=LANE_HALF):
    """그 축으로 손가락을 내릴 때 **가장 좁은 틈**(mm). 방해가 없으면 큰 값.

    축 방향으로 재고, 축에서 옆으로 lane_half 넘게 비킨 블록은 무시한다 —
    손가락이 그 옆을 지나가므로 닿지 않는다.
    틈은 두 블록 **표면 사이** 거리다(중심거리 - 목표 반폭 - 이웃 반폭).
    그 틈에 들어가는 것은 **손가락 하나**이므로 FINGER_T(한 개 두께)와
    견준다 — 반대쪽 손가락은 반대쪽 틈으로 내려간다. 양쪽을 함께 보는 것은
    아래 min 이다.

    이웃이 (x, y, 기울기deg) 면 **그 기울기로 반폭을 계산한다**(nb_half).
    돌아간 정사각은 축 방향으로 더 내밀기 때문이다 — 최대 7.2mm. 기울기를
    안 넘기면 예전처럼 축 정렬(반폭 17.5)로 본다.
    """
    import numpy as np
    r = np.radians(float(axis_deg))
    ax = np.array([np.cos(r), np.sin(r)])          # 손가락이 닫히는 축
    perp = np.array([-ax[1], ax[0]])
    best = float("inf")
    for nb in neighbors:
        d = np.array([float(nb[0]) - target_xy[0], float(nb[1]) - target_xy[1]])
        if abs(float(perp @ d)) >= nb_lane(nb, axis_deg, lane_half=lane_half,
                                           block_mm=block_mm):
            continue                               # 옆으로 비켜 있다
        # 목표는 손목을 제 면에 맞췄으므로 반폭이 정확히 block_mm/2 다.
        gap = (abs(float(ax @ d)) - block_mm / 2.0
               - nb_half(nb, axis_deg, block_mm=block_mm))
        best = min(best, gap)
    return best


def fold_turn(last_rot, turn):
    """회피 회전을 **손목이 덜 돌아가는 쪽**으로 접는다. 돌려줄 값은 +turn 또는 -turn.

    +90 과 -90 은 **같은 물림**이다 — 그리퍼가 닫히는 것은 직선(축)이라 180°
    차이는 손가락 좌우만 바꾸고, 정사각 블록은 어차피 90° 대칭이다. 그런데
    best_axis 는 늘 +90 만 준다. 손목이 이미 돌아 있으면 그 +90 이 관절 한계를
    넘긴다.

    실측 2026-08-08: 기울기 41° 인 보라가 손목 +131°(GRASP_ROT 90 + 41) 였는데
    +90 을 더해 **+221°** 가 되어 "블록 위로 접근하지 못해 파지를 중단합니다" 로
    죽었다 — 같은 배치에서 두 번, 초록까지 같이 실패했다. -90 을 골랐으면
    +41° 라 아무 문제가 없었다.

    fold90 이 '45° 넘게 돌 일이 없다' 고 보장하는 것은 **블록 기울기까지**다.
    GRASP_ROT(+90)과 이 회피 회전은 그 위에 더해지므로 그 보장이 깨진다.
    """
    if not turn:
        return turn
    return turn if abs(last_rot + turn) <= abs(last_rot - turn) else -turn


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


# ── 사방이 막혔을 때: 이웃 하나를 옆으로 치운다 ──────────────────────
# best_axis 로도 못 풀리면(두 축 다 좁으면) 손목을 돌려서 해결할 방법이 없다 —
# 자리 자체가 없는 것이다. 이땐 막고 있는 이웃 블록 하나를 옮겨 자리를 만든다.
#
# 옮기는 방향은 target에서 그 이웃으로 이어지는 축 방향, 그대로 더 밀어내는
# 쪽이다(옆으로 레인을 벗어나게 미는 방법도 있지만, 축을 따라 미는 쪽이 계산이
# 한 가지뿐이라 더 예측 가능하다 — 그리고 어차피 최소 이동량만 계산하므로
# 옮기는 거리 자체는 작다). 옮기는 양은 "그 축의 틈을 문턱보다 살짝 더
# 벌리는 데 필요한 최소 거리" 다 — 필요 이상으로 멀리 보내지 않는다.
RELOCATE_MARGIN = 5.0    # 문턱에 딱 맞추면 손 떨림 등으로 다시 걸릴 수 있어 여유를 더 준다


def blocking_neighbor(target_xy, axis_deg, neighbors, block_mm=BLOCK_MM,
                      lane_half=LANE_HALF):
    """approach_gap 과 같은 판정으로, 가장 좁은 틈을 만든 이웃의 좌표 자체를 돌려준다.

    approach_gap 은 가장 좁은 틈의 **크기**만 돌려주고 누가 그 틈을 만들었는지는
    버린다. 이웃을 실제로 옮기려면 어느 이웃인지가 필요해서 따로 둔다 — 판정
    로직 자체는 approach_gap 과 반드시 같아야 하므로(안 그러면 "막혔다고 판단한
    것" 과 "옮기려는 것" 이 다른 블록을 가리킬 수 있다) 여기서 다시 구현한다.
    막는 이웃이 없으면 None.
    """
    import numpy as np
    r = np.radians(float(axis_deg))
    ax = np.array([np.cos(r), np.sin(r)])
    perp = np.array([-ax[1], ax[0]])
    best_gap, best_xy = float("inf"), None
    for nb in neighbors:
        d = np.array([float(nb[0]) - target_xy[0], float(nb[1]) - target_xy[1]])
        if abs(float(perp @ d)) >= nb_lane(nb, axis_deg, lane_half=lane_half,
                                           block_mm=block_mm):
            continue                               # 옆으로 비켜 있다 — 방해 아님
        gap = (abs(float(ax @ d)) - block_mm / 2.0
               - nb_half(nb, axis_deg, block_mm=block_mm))
        if gap < best_gap:
            best_gap, best_xy = gap, (float(nb[0]), float(nb[1]))
    return best_xy


def relocate_step(target_xy, blocker_xy, axis_deg, finger_t=FINGER_T,
                  block_mm=BLOCK_MM, margin=RELOCATE_MARGIN):
    """blocker_xy 를 target_xy 로부터 axis_deg 축 방향으로 밀어낼 목적지.

    이미 틈이 충분하면(옮길 필요가 없으면) None. 다른 블록과 겹치는지는 여기서
    보지 않는다 — 순수 기하 함수라 판 위에 뭐가 더 있는지 모른다. 그건 부르는
    쪽이 비전(neighbors_xy)으로 확인한다.
    """
    import numpy as np
    r = np.radians(float(axis_deg))
    ax = np.array([np.cos(r), np.sin(r)])
    d = np.array([blocker_xy[0] - target_xy[0], blocker_xy[1] - target_xy[1]])
    along = float(ax @ d)                          # 부호 있음 — target 기준 어느 쪽인지
    need = (block_mm + finger_t + margin) - abs(along)
    if need <= 0:
        return None                                # 이미 충분하다
    sign = 1.0 if along >= 0 else -1.0
    dest = np.array(blocker_xy) + sign * need * ax
    return (float(dest[0]), float(dest[1]))


# ── 여러 개를 옮기는 계획(복제)에서: 막는 블록이 그 계획에도 있으면 먼저 처리 ──
# copy_human() 같은 계획은 "인간구역 배치를 로봇구역에 그대로 만든다" 처럼
# 여러 색을 순서대로 옮긴다. 그런데 뒤 순번 색이 앞 순번 색을 막고 있으면,
# 앞 색을 집으려다 relocate_blocker 가 뒤 색을 **임시 자리로 잠깐 치웠다가**
# 나중에 그 계획 차례가 왔을 때 제 목적지로 또 옮기게 된다 — 두 번 움직이는
# 낭비다. 뒤 색이 어차피 이 계획에서 옮겨야 할 색이라면, 그 자리에서 곧장
# **제 목적지로** 보내는 편이 한 번의 이동으로 끝난다. 그러려면 순서를
# 먼저 바꿔야 한다 — 이 판단은 patrol()로 이미 모아 둔 좌표만으로 되므로
# 로봇을 다시 움직이지 않고 계획을 세우는 시점에 할 수 있다.
def _blocking_color(color, xy, axis_deg, all_pts, finger_t=FINGER_T):
    """xy(color)가 다른 블록들 사이에서 두 축 다 막혀 있으면 막는 블록의 색.

    all_pts 는 [(x, y, 색), ...] — 판 위 모든 프리 블록(이 색의 다른 개체
    포함). 자기 자신(좌표가 거의 같은 같은 색 항목)은 제외하고 본다.
    안 막혀 있거나 막는 이웃의 색을 모르면 None.
    """
    # all_pts 는 (x, y, 색) 또는 (x, y, 색, 기울기). 기울기가 있으면 기하로 넘긴다
    # — 세 번째가 색이라 _pt 가 그대로 읽을 수 없어 여기서 자리를 바꾼다.
    others = [_pt((p[0], p[1], p[3] if len(p) > 3 else None)) for p in all_pts
              if not (p[2] == color and abs(p[0] - xy[0]) < 1.0
                      and abs(p[1] - xy[1]) < 1.0)]
    turn, gap, _gap0 = best_axis(xy, axis_deg, others, finger_t=finger_t)
    if gap >= finger_t:
        return None                                # 안 막혔다
    bxy = blocking_neighbor(xy, axis_deg + turn, others)
    if bxy is None:
        return None
    return next((p[2] for p in all_pts
                if abs(p[0] - bxy[0]) < 1.0 and abs(p[1] - bxy[1]) < 1.0), None)


def pick_order(pts, all_pts, finger_t=FINGER_T, self_r=30.0):
    """같은 색 후보들을 **집기 쉬운 순**으로 다시 늘어놓는다.

    pts      [(x, y, 파지축deg), ...] — 그 색의 후보들. 축은 그 블록 자신의
             검출 기울기에서 나온 것이어야 한다(놓을 때 각도가 아니다).
    all_pts  [(x, y), ...] — 지금 판에서 본 **모든** 블록. 후보 자신도 들어
             있어도 된다(자기 자신은 아래 self_r 로 걸러진다).
    self_r   이 반경 안의 점은 **이웃이 아니라 자기 자신**으로 본다(기본 30mm,
             block_sort.SELF_R 과 같은 값). 35mm 블록 둘이 30mm 안에 있을 수는
             없으므로, 그런 점은 같은 물리 블록을 두 번 본 것이다.

             **없으면 유령이 진짜 방해물로 잡힌다.** 실측 2026-08-08: 빨강
             (519,85) 와 주황 (532,78) 이 중심거리 14.8mm 로 보고됐다(색상값이
             인접해 한 블록을 둘로 검출한 것이다). 그 유령 탓에 주황의 틈이
             -22mm(겹침)로 나와 '완전히 막힘' 으로 판정됐고, 계획 1번이던
             주황이 맨 뒤로 밀렸다. 정작 실제 파지는 neighbors_xy 가 SELF_R 로
             그것을 걸러내고 멀쩡히 집었다 — 판단과 실행이 어긋난 것이다.

    **왜 필요한가.** 검출 노드는 신뢰도 순으로 답하고, 부르는 쪽은 그 첫 번째를
    집었다. 그런데 신뢰도와 집기 쉬움은 아무 상관이 없다 — 사방이 포위된 블록이
    1등이면 바로 옆에 뻥 뚫린 같은 색을 두고도 그 포위된 것을 집으려 든다.
    그러면 이웃 치우기가 통째로 발동하고, 실패하면 사람에게 치워 달라고 한다.
    고를 수 있을 때 고르는 것이 치우기보다 언제나 싸다.

    점수는 best_axis 의 틈이다(손목 90° 회전까지 본 뒤의 값). 동점이면 원래
    순서를 지킨다 — 신뢰도 순을 이유 없이 흔들지 않기 위해서다.
    """
    import numpy as np

    def gap_of(p):
        x, y = p[0], p[1]
        axis = p[2] if len(p) > 2 else 0.0
        others = [_pt(o) for o in all_pts
                  if np.hypot(o[0] - x, o[1] - y) >= self_r]
        _turn, gap, _g0 = best_axis((x, y), axis, others, finger_t=finger_t)
        return gap

    scored = [(gap_of(p), i, p) for i, p in enumerate(pts)]
    # 틈이 큰 것부터. 같으면 원래 순서(i)를 지킨다.
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [p for _g, _i, p in scored]


def reorder_for_conflicts(plan, positions, finger_t=FINGER_T):
    """plan을 막힘 관계를 반영해 재정렬한다 — 막는 색을 먼저 오게 한다.

    plan       [(hz_i, dst, color, angle_deg), ...] — copy_human()의 계획.
               색이 세 번째, 손목 각도(축 방향 판단 기준)가 네 번째 자리다.
    positions  {색: [(x, y), ...]} 또는 {색: [(x, y, 기울기), ...]} —
               patrol()로 이미 모아 둔 프리구역 좌표. 기울기가 있으면 막힘
               판정이 그만큼 정확해진다(nb_half 참고).
               plan에 없는 색이 섞여 있어도 된다(막는 쪽 판정에만 쓰인다 —
               plan에 없는 색은 순서를 조정할 대상이 아니므로 자연히 걸러진다).

    각 항목이 두 축 다 막혀 있고 그 막는 이웃이 **plan 안의 다른 색**이면,
    그 색의 항목을 이 항목보다 앞으로 옮긴다. 같은 색이 plan에 여럿이면
    (인간구역 두 칸이 같은 색) 순서 판단은 그 색의 대표 좌표(첫 개체) 하나로
    하되, 실제 순서는 항목 단위로(원래 상대 순서를 유지하며) 옮긴다.

    서로가 서로를 막는 순환 의존이면 풀 수 없다 — 그 경우 원래 순서를
    유지한다(무리하게 순서를 만들지 않는다. relocate_blocker 의 임시 대피가
    그런 경우의 안전망으로 남아 있다).
    """
    # (x, y, 색, 기울기) — _blocking_color 가 세 번째를 색, 네 번째를 기울기로 읽는다.
    all_pts = [(p[0], p[1], c, p[2] if len(p) > 2 else None)
               for c, pts in positions.items() for p in pts]
    plan_colors = {t[2] for t in plan}

    deps = {}                          # color -> plan 안에서 그 색을 막는 색들
    seen_color = set()
    for _hz_i, _dst, color, ang in plan:
        if color in seen_color:        # 같은 색은 대표 좌표 하나로 한 번만 판단
            continue
        seen_color.add(color)
        pts = positions.get(color) or []
        if not pts:
            continue
        blocker = _blocking_color(color, pts[0], ang, all_pts, finger_t=finger_t)
        if blocker and blocker in plan_colors and blocker != color:
            deps[color] = blocker

    if not deps:
        return list(plan)

    # 색 단위로 위상정렬한다 — 막는 색(선행)이 먼저 오도록. 안정 정렬:
    # 매 단계에서 아직 선행이 안 끝난 것은 건너뛰고, 끝난 것부터 확정한다.
    # 진행이 안 되면(순환 의존) 남은 것은 원래 순서 그대로 이어붙인다.
    color_order, placed = [], set()
    remaining = list(dict.fromkeys(t[2] for t in plan))   # 등장 순서, 중복 제거
    while remaining:
        progressed = False
        for c in list(remaining):
            need = deps.get(c)
            if need is not None and need not in placed and need in remaining:
                continue                    # 선행 색이 아직 안 끝났다 — 나중에
            color_order.append(c)
            placed.add(c)
            remaining.remove(c)
            progressed = True
        if not progressed:                  # 순환 의존 — 남은 건 원래 순서로
            color_order += remaining
            break

    by_color = {}
    for t in plan:
        by_color.setdefault(t[2], []).append(t)
    return [t for c in color_order for t in by_color[c]]


def graspable_blocker(target_xy, axis_deg, neighbors, finger_t=FINGER_T,
                      target_deg=None, **kw):
    """막는 이웃 중 **그 자신을 집을 수 있는** 것을 고른다. 없으면 None.

    이것이 없으면 이웃 치우기가 자기 발에 걸린다. 이웃을 치우려면 먼저 그 이웃을
    집어야 하는데, 이웃 입장에서는 **원래 목표가 자기 이웃**이다(붙어 있으니까
    막은 것이다). 그래서 이웃을 집으려 할 때 같은 '좁음' 판정에 걸려 파지가
    포기되고, 치우기가 시작조차 못 한다(실측 2026-08-07: 파란색을 치우라는 말만
    반복했다).

    그러므로 **집을 수 있는 이웃**을 골라야 한다. 판정은 파지와 같은 것을 쓴다 —
    그 이웃을 목표로 놓고, 나머지 전부(원래 목표 포함)를 이웃으로 두고 두 축의
    여유를 본다. 손목을 90° 돌릴 수 있으니 두 축 중 넓은 쪽으로 판단한다.

    여러 개면 **가장 여유가 큰** 것을 고른다 — 집기 쉬운 것부터 치우는 게 낫다.
    아무도 못 집으면 None 이고, 그때는 사람이 손으로 치워야 한다(연쇄로 파고들지
    않는다 — 끝없이 번질 수 있다).
    """
    pts = [_pt(n) for n in neighbors]
    tgt = _pt((target_xy[0], target_xy[1], target_deg))
    best, best_xy = -float("inf"), None
    for i, b in enumerate(pts):
        # ① 이 이웃이 **이 축에서** 막고 있는가.
        #
        # 두 축 중 어느 쪽이든 막으면 후보로 봤더니, 지금 쓰려는 축과 무관한
        # 블록을 치우고 있었다 — 치워도 그 축은 그대로 막혀 있으니 헛수고다
        # (실측 2026-08-07: 축 119.8° 에서 막는 것은 파랑·보라인데 주황을 치웠다).
        # **한 축을 열기로 정했으면 그 축의 방해물만 치운다.** 손가락은 양쪽에서
        # 내려오므로 그 축의 양쪽을 다 치워야 열린다 — 한 번에 하나씩, 부르는 쪽이
        # RELOCATE_MAX_TRIES 만큼 되풀이한다.
        if approach_gap(target_xy, axis_deg, [b], **kw) >= finger_t:
            continue
        # ② 그 이웃 자신을 집을 수 있는가 (원래 목표도 이웃으로 넣는다)
        others = [p for j, p in enumerate(pts) if j != i] + [tgt]
        room = max(approach_gap(b, axis_deg, others, **kw),
                   approach_gap(b, axis_deg + 90.0, others, **kw))
        if room >= finger_t and room > best:
            # 자리만 돌려준다 — 부르는 쪽이 np.asarray 로 2차원 벡터를 만든다.
            best, best_xy = room, (b[0], b[1])
    return best_xy


# 막는 블록을 **잡은 채로 끄는** 거리(mm). 들었다 놓지 않는다.
#
# 왜 파지축의 **직각**인가. 손가락이 지나가는 길(레인)은 파지축을 따라 나 있고
# 반폭이 LANE_HALF 다. 축을 따라 멀리 밀려면 중심거리 67mm 를 만들어야 하지만,
# 옆으로 빼면 레인 밖(29.5mm)으로만 나가면 된다 — 훨씬 짧다. 목표에서 보면
# 그 블록이 **대각선**으로 비켜난 자리가 된다.
#
# 60mm 인 이유: 레인을 벗어나는 데 필요한 최대치가 34mm(축 위에 딱 걸친 경우)
# 인데, 거기서 끝내면 **직각 축**의 레인에 새로 걸릴 수 있다. 60mm 면 두 축
# 모두에서 확실히 벗어난다 — 검증: 축 위/옆 어디에 있든 끌고 나면 두 축 다 inf.
SLIDE_MM = 60.0


def seg_dist(p, a, b):
    """점 p 에서 선분 a→b 까지의 거리. 끌고 가는 길에 뭐가 있는지 볼 때 쓴다."""
    import numpy as np
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 < 1e-9:
        return float(np.hypot(p[0] - ax, p[1] - ay))
    t = max(0.0, min(1.0, ((p[0] - ax) * vx + (p[1] - ay) * vy) / L2))
    return float(np.hypot(p[0] - (ax + t * vx), p[1] - (ay + t * vy)))


def slide_dests(target_xy, blocker_xy, axis_deg, others=(), dist=SLIDE_MM,
                finger_t=FINGER_T, block_mm=BLOCK_MM, lane_half=LANE_HALF,
                clear_extra=5.0, check_path=True):
    """blocker 를 파지축 **직각**으로 dist 만큼 끌 자리 후보들 — 좋은 순.

    들었다 놓는 것(relocate_candidates)과 달리 **잡은 채로 끄는** 것이라,
    지나가는 길이 비어 있어야 한다. 블록이 훑고 가는 복도에 다른 블록이 있으면
    그것까지 밀려 판이 무너진다 — 그런 방향은 빼고 돌려준다.

    두 방향(+직각, -직각) 중 **목표에서 멀어지는 쪽**을 먼저 본다. 목표 쪽으로
    끌면 지나가는 길이 목표를 스칠 수 있다.

    others 에는 blocker 자신을 넣어도 된다(같은 자리는 자기 자신으로 본다).
    목표(target_xy)는 넣지 않아도 여기서 함께 본다 — 끌다가 목표를 치면 안 된다.
    """
    import numpy as np
    b = np.asarray(blocker_xy, float)
    t = np.asarray(target_xy, float)
    perp = np.array([-np.sin(np.radians(axis_deg)), np.cos(np.radians(axis_deg))])
    # blocker 가 목표에서 어느 쪽으로 치우쳐 있나 — 그쪽으로 더 밀어낸다.
    lean = float(perp @ (b - t))
    signs = (1.0, -1.0) if lean >= 0 else (-1.0, 1.0)

    obstacles = [(float(o[0]), float(o[1])) for o in others
                 if not (abs(o[0] - b[0]) < 1.0 and abs(o[1] - b[1]) < 1.0)]
    obstacles.append((float(t[0]), float(t[1])))
    out = []
    for s in signs:
        d = b + s * dist * perp
        dxy = (float(d[0]), float(d[1]))
        # ① 끌고 난 자리가 더는 목표를 막지 않아야 한다 (두 축 모두)
        if (approach_gap(target_xy, axis_deg, [dxy], block_mm=block_mm,
                         lane_half=lane_half) < finger_t
                or approach_gap(target_xy, axis_deg + 90.0, [dxy],
                                block_mm=block_mm, lane_half=lane_half) < finger_t):
            continue
        # ② 훑고 가는 복도가 비어 있어야 한다. 블록 한 변이면 중심이 그만큼
        #    떨어져 있어도 모서리가 스친다 — 여유를 조금 더 준다.
        #
        #    **지금보다 가까워질 때만** 막는다. 막고 있는 블록은 정의상 목표에
        #    바싹 붙어 있어서(그러니까 막은 것이다), 절대 거리로 재면 목표 자신
        #    때문에 모든 방향이 거부된다 — 실제로 38mm 옆 목표 때문에 양쪽 다
        #    막혔다. 끌기가 사이를 **더 좁히지 않으면** 통과시킨다.
        if not check_path:
            out.append(dxy)                # 길은 안 본다 — 잡히면 끈다
            continue
        clear = block_mm + clear_extra
        bad = False
        for o in obstacles:
            d0 = float(np.hypot(o[0] - b[0], o[1] - b[1]))
            if seg_dist(o, blocker_xy, dxy) < min(clear, d0) - 1e-6:
                bad = True
                break
        if bad:
            continue
        out.append(dxy)
    return out


def axis_blockers(target_xy, axis_deg, neighbors, finger_t=FINGER_T,
                  target_deg=None, **kw):
    """이 축을 막는 이웃 **전부**를, 자기 자신이 집기 쉬운 순으로.

    돌려주는 것: [(xy, room, 집을수있나), ...] — room 이 큰 것부터.

    graspable_blocker 는 '집을 수 있는 것 중 최선' 하나만 준다. 그것 하나를
    집다 실패하면 치우기가 통째로 죽는데, 막는 이웃이 여럿이면 **다음 것을
    시도해 볼 수 있다.** 그리고 아무도 못 집는 뭉치라도, 가장 덜 갇힌 것을
    골라 **그 블록의 방해물부터** 치우면 한 단계 더 파고들 수 있다.
    포기하기 전에 해볼 것을 다 해보기 위한 목록이다.
    """
    pts = [_pt(n) for n in neighbors]
    tgt = _pt((target_xy[0], target_xy[1], target_deg))
    out = []
    for i, b in enumerate(pts):
        if approach_gap(target_xy, axis_deg, [b], **kw) >= finger_t:
            continue                       # 이 축을 안 막는다 — 치워도 헛수고
        others = [p for j, p in enumerate(pts) if j != i] + [tgt]
        room = max(approach_gap(b, axis_deg, others, **kw),
                   approach_gap(b, axis_deg + 90.0, others, **kw))
        # 자리만 돌려준다(2튜플) — 부르는 쪽이 그대로 좌표로 쓴다.
        out.append(((b[0], b[1]), room, room >= finger_t))
    out.sort(key=lambda t: -t[1])
    return out


def axis_order(target_xy, axis_deg, neighbors, finger_t=FINGER_T,
               target_deg=None, **kw):
    """직각인 두 축을 **치울 개수가 적은 순**으로 돌려준다.

    돌려주는 것: [(축 각도, 막는 개수), (축 각도, 막는 개수)] — 적은 쪽이 앞.
    개수가 같으면 부른 순서(= best_axis 가 고른 축)를 그대로 둔다.

    best_axis 는 **틈이 넓은 축**을 고르지만, 그 축을 여는 데 몇 개를 치워야
    하는지는 보지 않는다. 손가락은 양쪽에서 내려오므로 그 축의 방해물을
    **다 치워야** 열린다 — 두 개짜리 축을 골라 하나만 치우면 그 이동은
    통째로 헛수고다. 정사각 블록이라 두 축의 물림은 어차피 같으니,
    **적게 치우고 열리는 쪽**을 먼저 잡는 것이 맞다.

    실측 2026-08-11 (로그 python3_5769): 보라색(564,-121)이 십자로 둘러싸였다.
        세로축(90°)  막는 것 1개  틈 14mm     ← 한 번 끌면 열린다
        가로축( 0°)  막는 것 3개  틈  2mm
    세로축이 막히자 가로축으로 넘어가 3개 중 하나만 끌었고, 남은 둘 때문에
    축은 그대로 막혀 있었다 (test_relocate.py 26 참고).
    """
    out = []
    for deg in (axis_deg, axis_deg + 90.0):
        n = len(axis_blockers(target_xy, deg, neighbors, finger_t=finger_t,
                              target_deg=target_deg, **kw))
        out.append((deg, n))
    if out[1][1] < out[0][1]:
        out.reverse()
    return out


# 임시로 치워 두는 자리에 요구할 최소 중심거리(mm). 파지 여유(블록+손가락=62)를
# 요구하면 꽉 찬 판에서는 사실상 자리가 없다 — 실측 2026-08-07: 옮길 자리에서
# 51.6mm 떨어진(즉 16mm 떨어져 겹치지도 않는) 블록 때문에 포기했다.
# 여기서는 **겹치지 않는 것**만 본다. 파지 여유가 나는 자리가 있으면 그쪽을
# 먼저 쓰고(부르는 쪽이 두 단계로 본다), 없을 때 이 값으로 물러난다.
RELOCATE_CLEAR_MIN = BLOCK_MM + 8.0


# 가까운 후보가 다 막혔을 때 **넓게 훑을** 반경들(mm)과 방향 수.
# 최소 이동만 보는 후보 다섯은 꽉 찬 판에서 전부 막히기 쉽다 — 그러면 치우기가
# 통째로 무산된다. 임시로 치워 두는 자리일 뿐이니, 판을 좀 어지럽히더라도
# **멀리라도 던질 수 있으면 던지는** 편이 포기보다 낫다.
RELOCATE_WIDE_R = (80.0, 120.0, 170.0, 230.0)
RELOCATE_WIDE_DIRS = 12          # 30° 간격


def relocate_candidates(target_xy, blocker_xy, axis_deg, finger_t=FINGER_T,
                        block_mm=BLOCK_MM, margin=RELOCATE_MARGIN,
                        lane_half=LANE_HALF, wide=True):
    """blocker 를 치울 자리 후보들을 **좋은 순서로** 돌려준다.

    한 곳만 계산해서 그 자리가 막혀 있으면 포기하던 것을 고치기 위한 것이다
    (실측 2026-08-07: 사방이 포위된 블록에서 유일한 후보가 막혀 치우기가 무산됐다).

    순서의 뜻:
      ① 축 방향으로 밀기        — 판을 가장 덜 어지럽힌다(relocate_step 과 같다)
      ② 축과 **직각**으로 밀기   — 손가락 지나가는 길(lane)에서 빼면 되므로
                                 보통 ①보다 짧게 움직인다
      ③ 더 멀리 / 대각선        — 앞이 다 막혔을 때
      ④ wide 면 **사방으로 넓게** — 위 다섯이 다 막혔을 때. blocker 를 중심으로
                                 반경을 키워 가며 빙 둘러 본다. 가까운 반경부터,
                                 같은 반경에서는 **목표 반대쪽 방향부터** 본다
                                 (목표 주변을 다시 어지럽히지 않는 쪽이다).

    후보는 **그 블록이 더는 목표를 막지 않는 자리**만 남긴다(두 축 모두에서).
    실제로 비었는지·구역인지·판 안인지는 비전이 있는 부르는 쪽이 본다.
    """
    import numpy as np
    r = np.radians(float(axis_deg))
    ax = np.array([np.cos(r), np.sin(r)])
    perp = np.array([-ax[1], ax[0]])
    t = np.array([float(target_xy[0]), float(target_xy[1])])
    b = np.array([float(blocker_xy[0]), float(blocker_xy[1])])
    d = b - t
    along, side = float(ax @ d), float(perp @ d)
    s_along = 1.0 if along >= 0 else -1.0
    s_side = 1.0 if side >= 0 else -1.0

    need_along = (block_mm + finger_t + margin) - abs(along)
    need_side = (lane_half + block_mm / 2.0 + margin) - abs(side)

    out = []
    if need_along > 0:
        out.append(b + s_along * need_along * ax)
    if need_side > 0:
        out.append(b + s_side * need_side * perp)          # 가까운 쪽으로 빼기
        out.append(b - s_side * (need_side + 2 * abs(side)) * perp)   # 반대쪽으로
    if need_along > 0:
        out.append(b + s_along * (need_along * 1.8) * ax)  # 더 멀리
        if need_side > 0:                                   # 대각선
            out.append(b + s_along * need_along * ax + s_side * need_side * perp)

    if wide:
        # 목표에서 blocker 로 향하는 방향 — 여기에 가까운 쪽이 '목표 반대쪽' 이다.
        away = d / (np.linalg.norm(d) or 1.0)
        for rad in RELOCATE_WIDE_R:
            ring = []
            for k in range(RELOCATE_WIDE_DIRS):
                th = 2 * np.pi * k / RELOCATE_WIDE_DIRS
                u = np.array([np.cos(th), np.sin(th)])
                ring.append((-float(away @ u), b + rad * u))
            ring.sort(key=lambda t: t[0])       # 목표 반대쪽 방향부터
            out += [p for _s, p in ring]

    keep, seen = [], []
    for c in out:
        c = (float(c[0]), float(c[1]))
        if any(np.hypot(c[0] - s[0], c[1] - s[1]) < 1.0 for s in seen):
            continue                            # 넓은 훑기는 같은 자리를 다시 낼 수 있다
        seen.append(c)
        if (approach_gap(target_xy, axis_deg, [c], block_mm=block_mm,
                         lane_half=lane_half) >= finger_t
                and approach_gap(target_xy, axis_deg + 90.0, [c], block_mm=block_mm,
                                 lane_half=lane_half) >= finger_t):
            keep.append(c)
    return keep


def lane_offsets(target_xy, xy, axis_deg):
    """(축 방향 거리, 옆으로 비낀 거리). 로그로 "왜 저것이 막는가" 를 보여줄 때 쓴다."""
    import numpy as np
    r = np.radians(float(axis_deg))
    ax = np.array([np.cos(r), np.sin(r)])
    d = np.array([float(xy[0]) - target_xy[0], float(xy[1]) - target_xy[1]])
    return float(ax @ d), float(np.array([-ax[1], ax[0]]) @ d)
