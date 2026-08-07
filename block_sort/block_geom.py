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
        if abs(float(perp @ d)) >= lane_half:
            continue                               # 옆으로 비켜 있다 — 방해 아님
        gap = abs(float(ax @ d)) - block_mm
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
    others = [(x, y) for x, y, c in all_pts
             if not (c == color and abs(x - xy[0]) < 1.0 and abs(y - xy[1]) < 1.0)]
    turn, gap, _gap0 = best_axis(xy, axis_deg, others, finger_t=finger_t)
    if gap >= finger_t:
        return None                                # 안 막혔다
    bxy = blocking_neighbor(xy, axis_deg + turn, others)
    if bxy is None:
        return None
    return next((c for x, y, c in all_pts
                if abs(x - bxy[0]) < 1.0 and abs(y - bxy[1]) < 1.0), None)


def reorder_for_conflicts(plan, positions, finger_t=FINGER_T):
    """plan을 막힘 관계를 반영해 재정렬한다 — 막는 색을 먼저 오게 한다.

    plan       [(hz_i, dst, color, angle_deg), ...] — copy_human()의 계획.
               색이 세 번째, 손목 각도(축 방향 판단 기준)가 네 번째 자리다.
    positions  {색: [(x, y), ...]} — patrol()로 이미 모아 둔 프리구역 좌표.
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
    all_pts = [(x, y, c) for c, pts in positions.items() for x, y in pts]
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


def graspable_blocker(target_xy, axis_deg, neighbors, finger_t=FINGER_T, **kw):
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
    import numpy as np
    pts = [(float(n[0]), float(n[1])) for n in neighbors]
    best, best_xy = -float("inf"), None
    for i, b in enumerate(pts):
        # ① 이 이웃이 실제로 목표를 막고 있는가 (아니면 옮겨도 소용없다)
        blocks = min(approach_gap(target_xy, axis_deg, [b], **kw),
                     approach_gap(target_xy, axis_deg + 90.0, [b], **kw))
        if blocks >= finger_t:
            continue
        # ② 그 이웃 자신을 집을 수 있는가 (원래 목표도 이웃으로 넣는다)
        others = [p for j, p in enumerate(pts) if j != i] + [tuple(target_xy)]
        room = max(approach_gap(b, axis_deg, others, **kw),
                   approach_gap(b, axis_deg + 90.0, others, **kw))
        if room >= finger_t and room > best:
            best, best_xy = room, b
    return best_xy
