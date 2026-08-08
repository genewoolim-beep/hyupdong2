#!/usr/bin/env python3
"""사방이 막힌 블록을 위해 이웃 하나를 옆으로 치우는 계산을 로봇 없이 검증한다.

    python3 test_relocate.py

best_axis 로 손목을 돌려도 두 축 다 좁으면(사방이 막히면) 더 이상 손목만으로는
풀 수 없다 — 자리 자체가 없는 것이다. 이땐 막고 있는 이웃 블록 하나를 옮겨
자리를 만든다. 여기서 확인하는 것은 순수 기하 두 가지다:

  blocking_neighbor  어느 이웃이 막고 있는지 (approach_gap과 같은 판정이어야 한다)
  relocate_step      그 이웃을 얼마나, 어느 쪽으로 옮겨야 틈이 벌어지는지

실제로 옮기는 동작(집기·놓기)과 목적지가 다른 블록과 안 겹치는지 확인하는 것은
block_sort.py의 relocate_blocker() — 로봇/비전이 필요해 여기서는 못 다룬다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from block_geom import (approach_gap, blocking_neighbor, relocate_step,
                        BLOCK_MM, FINGER_T, LANE_HALF, RELOCATE_MARGIN)  # noqa: E402

T = (400.0, 0.0)          # 집을 블록 중심
NEED_CENTER = BLOCK_MM + FINGER_T + RELOCATE_MARGIN   # 여유 있는 중심거리

# ── 1. 막는 이웃이 없으면 None ──
assert blocking_neighbor(T, 0.0, []) is None
print("1 옆 블록 없음 → 막는 이웃 없음")

# ── 2. x 축에 딱 붙은 블록 — 그 좌표 자체를 돌려준다 ──
nb = [(440.0, 0.0)]
assert blocking_neighbor(T, 0.0, nb) == (440.0, 0.0)
print("2 x 축 40mm 옆 블록 → 그 좌표를 막는 이웃으로 지목")

# ── 3. 레인 밖으로 비킨 블록은 막는 이웃이 아니다 (approach_gap과 같은 판정) ──
assert blocking_neighbor(T, 0.0, [(440.0, LANE_HALF + 1.0)]) is None
print(f"3 축에서 {LANE_HALF:.0f}mm 이상 비키면 막는 이웃 아님 (approach_gap과 일치)")

# ── 4. 여럿이면 가장 좁은 틈을 만든 쪽을 고른다 ──
close_far = [(440.0, 0.0), (500.0, 0.0)]      # 틈 5mm, 65mm
assert blocking_neighbor(T, 0.0, close_far) == (440.0, 0.0)
print("4 여러 이웃 중 가장 좁은 틈을 만드는 쪽을 지목 (더 먼 쪽은 무시)")

# ── 5. 이미 충분히 넓으면 옮길 필요 없다 ──
assert relocate_step(T, (400.0 + NEED_CENTER + 1.0, 0.0), 0.0) is None
print(f"5 중심거리가 이미 필요량({NEED_CENTER:.0f}mm)보다 넓으면 옮기지 않음")

# ── 6. 딱 붙은 블록을 밀어내면, 그 결과가 다시 재도 충분해진다 ──
blocker = (440.0, 0.0)                        # 틈 5mm, 문턱(27mm) 못 미침
dest = relocate_step(T, blocker, 0.0)
assert dest is not None
new_gap = approach_gap(T, 0.0, [dest])
assert new_gap >= FINGER_T + RELOCATE_MARGIN - 1e-6, new_gap
# 옮기는 거리 자체는 필요한 만큼만 — 과하게 멀리 보내지 않는다.
moved = abs(dest[0] - blocker[0])
assert abs(moved - (NEED_CENTER - 40.0)) < 1e-6, moved
print(f"6 딱 붙은 이웃({blocker})을 {moved:.0f}mm 밀어내면 틈 {new_gap:.0f}mm "
      f"(문턱 {FINGER_T:.0f}mm) 확보")

# ── 7. target 반대쪽(음의 방향)에 있는 이웃은 반대쪽으로 더 밀려난다 ──
#     (뒤집혀서 target 쪽으로 오면 오히려 더 막는다 — 방향을 안 틀리는지 확인)
blocker_neg = (360.0, 0.0)                    # target 왼쪽, 틈 5mm
dest_neg = relocate_step(T, blocker_neg, 0.0)
assert dest_neg is not None
assert dest_neg[0] < blocker_neg[0], dest_neg  # 더 왼쪽(= target 반대 방향)으로
new_gap_neg = approach_gap(T, 0.0, [dest_neg])
assert new_gap_neg >= FINGER_T + RELOCATE_MARGIN - 1e-6, new_gap_neg
print(f"7 target 반대쪽 이웃은 반대 방향으로 밀려남 (더 막는 방향으로 안 감)")

# ── 8. 기울어진 축에서도 같은 계산이 성립한다 ──
import numpy as np  # noqa: E402
r = np.radians(20.0)
axis = 20.0
blocker_tilt = (T[0] + 40 * np.cos(r), T[1] + 40 * np.sin(r))
dest_tilt = relocate_step(T, blocker_tilt, axis)
assert dest_tilt is not None
new_gap_tilt = approach_gap(T, axis, [dest_tilt])
assert new_gap_tilt >= FINGER_T + RELOCATE_MARGIN - 1e-6, new_gap_tilt
print(f"8 기울어진(20°) 축에서도 같은 계산 성립 (틈 {new_gap_tilt:.0f}mm 확보)")

print("\n전부 통과 — 실기에서는 relocate_blocker()가 목적지 충돌 확인 후 "
      "실제로 집어서 옮기는지, 그 뒤 원래 블록을 집을 수 있게 되는지 본다")


# ── 9. 치울 이웃은 **그 자신을 집을 수 있는** 것으로 고른다 ──
#     이게 없으면 치우기가 자기 발에 걸린다: 이웃을 집으려 할 때 원래 목표가
#     그 이웃의 이웃이라 같은 '좁음' 판정에 걸려 파지가 포기된다
#     (실측 2026-08-07: "파란색을 치우라" 는 말만 반복했다).
from block_geom import graspable_blocker  # noqa: E402

T = (400.0, 0.0)
GAP = BLOCK_MM + 5.0            # 중심거리 40mm = 틈 5mm (막힘)

# (a) ㄱ 자: x 와 y 가 막혔다. y 쪽 이웃은 자기 위가 비어 있어 집을 수 있다.
L = [(T[0] + GAP, T[1]), (T[0], T[1] + GAP)]
b = graspable_blocker(T, 0.0, L)
assert b is not None, "ㄱ 자에서는 하나는 집을 수 있어야 한다"
print(f"9a ㄱ 자 배치 → 치울 이웃 ({b[0]:.0f},{b[1]:.0f}) 선택")

# (b) 2×2 뭉치: 넷이 서로 막아 아무도 못 집는다 → None (사람이 치워야 한다)
Q = [(T[0] + GAP, T[1]), (T[0], T[1] + GAP), (T[0] + GAP, T[1] + GAP)]
assert graspable_blocker(T, 0.0, Q) is None, "2×2 뭉치는 아무도 못 집는다"
print("9b 2×2 뭉치 → None (연쇄로 파고들지 않고 사람에게 넘긴다)")

# (c) 막지 않는 이웃은 고르지 않는다 (옮겨도 소용없다)
far = [(T[0] + 200.0, T[1])]
assert graspable_blocker(T, 0.0, far) is None
print("9c 안 막는 이웃은 고르지 않는다")


# ── 10. 실기에서 실패한 그 배치 (2026-08-07 로그) ──
#     빨간 블록이 네 블록에 **십자로 포위**됐다. 모두 ~49mm 라 틈 14mm.
#     이 배치에서 실패한 이유가 셋이었고 셋 다 고쳤다:
#       ① 지금 쓰는 축과 무관한 블록(주황)을 치우려 했다 → 축의 방해물만 고른다
#       ② 치울 자리를 한 곳만 봤고 그 자리가 51.6mm 옆 블록에 걸렸다 → 여러 곳
#       ③ 임시 자리에 파지 여유(62mm)를 요구했다 → 겹침만 안 하면 되게 두 단계
#     손가락은 양쪽에서 내려오므로 **한 축의 양쪽을 다** 치워야 열린다(2회).
from block_geom import (best_axis, graspable_blocker,   # noqa: E402
                        relocate_candidates, RELOCATE_CLEAR_MIN)

REAL_T = (574.8, 115.7)
REAL_NB = {"주황": (532.0, 92.0), "노랑": (618.0, 142.0),
           "파랑": (599.0, 73.0), "보라": (550.0, 158.0)}
REAL_AXIS = 119.8                      # 로그의 손목 +60° 상황 (역산값)
FULL = BLOCK_MM + FINGER_T


def _one_round(cur, axis):
    """한 번 판단해서 (치울 색, 목적지) 또는 None(파지 가능/불가)."""
    pts = list(cur.values())
    turn, gap, _ = best_axis(REAL_T, axis, pts)
    used = axis + turn
    if gap >= FINGER_T:
        return "가능", None, used
    b = graspable_blocker(REAL_T, used, pts)
    if b is None:
        return "뭉치", None, used
    name = next(k for k, v in cur.items()
                if abs(v[0] - b[0]) < 1 and abs(v[1] - b[1]) < 1)
    others = [v for k, v in cur.items() if k != name] + [REAL_T]
    for need in (FULL, RELOCATE_CLEAR_MIN):
        for c in relocate_candidates(REAL_T, b, used):
            if min(np.hypot(c[0] - x, c[1] - y) for x, y in others) >= need:
                return name, c, used
    return name, None, used


cur = dict(REAL_NB)
moved = []
for _ in range(3):
    name, dest, used = _one_round(cur, REAL_AXIS)
    if name == "가능":
        break
    assert name != "뭉치", "이 배치는 뭉치가 아니다 — 치울 수 있어야 한다"
    assert dest is not None, f"{name} 를 치울 자리를 못 찾았다"
    moved.append(name)
    cur.pop(name)
else:
    raise AssertionError(f"세 번 안에 못 열었다 (치운 것: {moved})")

assert moved == ["파랑", "보라"], f"축을 막는 둘을 치워야 한다 — 치운 것: {moved}"
print(f"10 실기 배치(십자 포위) → {' → '.join(moved)} 치우고 파지 가능  "
      f"(주황·노랑은 이 축을 안 막으므로 건드리지 않는다)")


# ── 11. 가까운 자리가 다 막히면 **사방으로 넓게** 훑는다 ──
#     최소 이동만 보는 후보 다섯은 꽉 찬 판에서 전부 막히기 쉽다. 그때
#     "치울 자리가 없다" 로 포기하면 원래 블록도 못 집는다 — 멀리라도 던진다.
from block_geom import RELOCATE_WIDE_R  # noqa: E402

B = (T[0] + GAP, T[1])                  # 딱 붙은 blocker
near_only = relocate_candidates(T, B, 0.0, wide=False)
wide_all = relocate_candidates(T, B, 0.0)
assert len(wide_all) > len(near_only), (len(near_only), len(wide_all))
far = [c for c in wide_all if np.hypot(c[0] - B[0], c[1] - B[1]) > 70.0]
assert far, "멀리 던질 후보가 있어야 한다"
print(f"11 넓은 훑기 — 가까운 후보 {len(near_only)}곳 → 전체 {len(wide_all)}곳 "
      f"(그중 {len(far)}곳은 70mm 넘게 떨어진 자리, 최대 {RELOCATE_WIDE_R[-1]:.0f}mm)")

# ── 12. 넓게 훑어도 '목표를 안 막는 자리' 라는 조건은 그대로다 ──
for c in wide_all:
    for ax in (0.0, 90.0):
        assert approach_gap(T, ax, [c]) >= FINGER_T, (c, ax)
print("12 넓은 후보도 전부 목표를 더는 막지 않는 자리")

# ── 13. 가까운 것부터 나온다 (멀리 던지는 건 앞이 다 막혔을 때뿐) ──
assert wide_all[:len(near_only)] == near_only, "가까운 후보가 앞에 와야 한다"
d = [round(np.hypot(c[0] - B[0], c[1] - B[1]), 3)
     for c in wide_all[len(near_only):]]
assert d == sorted(d), d                # 반경이 커지는 순서(같은 반경 안은 방향 순)
print("13 가까운 후보가 먼저, 넓은 훑기는 반경이 커지는 순서")

print("\n넓은 훑기 통과 — 구역 안까지 쓸지는 block_sort.py 의 RELOCATE_ANYWHERE "
      "(기본 1). 프리구역을 먼저 다 보고 없을 때만 구역을 쓴다")


# ── 14. 같은 색이 여럿이면 **집을 수 있는 것**을 고른다 ──
#     검출 노드는 신뢰도 순으로 답하는데 그건 집기 쉬움과 무관하다. 포위된
#     것이 1등이면 바로 옆 뻥 뚫린 같은 색을 두고도 이웃 치우기가 발동한다.
from block_geom import pick_order  # noqa: E402

BOXED = (400.0, 0.0)                      # 십자로 포위 — 못 집는다
FREE = (400.0, 300.0)                     # 사방이 비었다
WALL = [(BOXED[0] + GAP, BOXED[1]), (BOXED[0] - GAP, BOXED[1]),
        (BOXED[0], BOXED[1] + GAP), (BOXED[0], BOXED[1] - GAP)]
ALL = [BOXED, FREE] + WALL

# 신뢰도 순이 [포위된 것, 자유로운 것] 으로 와도 자유로운 것을 앞에 놓는다.
order = pick_order([(BOXED[0], BOXED[1], 0.0), (FREE[0], FREE[1], 0.0)], ALL)
assert (order[0][0], order[0][1]) == FREE, order
print(f"14 포위된 것이 1등이어도 자유로운 것을 먼저 고른다 "
      f"({BOXED} → {FREE})")

# ── 15. 둘 다 집을 수 있으면 원래 순서를 지킨다 (신뢰도를 이유 없이 안 흔든다) ──
A, B = (400.0, 300.0), (400.0, 500.0)
same = pick_order([(A[0], A[1], 0.0), (B[0], B[1], 0.0)], [A, B])
assert [(p[0], p[1]) for p in same] == [A, B], same
print("15 둘 다 집을 수 있으면 원래(신뢰도) 순서를 지킨다")

# ── 16. 개수와 내용이 보존된다 (하나도 잃지 않는다) ──
many = [(400.0, 0.0, 0.0), (400.0, 300.0, 0.0), (400.0, 500.0, 30.0)]
got = pick_order(many, ALL + [(400.0, 500.0)])
assert len(got) == len(many) and set(map(tuple, got)) == set(map(tuple, many))
print("16 후보 개수·내용 보존")


# ── 17. 서로 붙은 뭉치도 **한 단계 파고들면** 풀린다 ──
#     예전에는 axis_blockers 가 없어 "집을 수 있는 것 하나" 만 봤고, 아무도
#     못 집으면 곧장 사람에게 넘겼다(9b). 이제 갇힌 것까지 목록에 담아
#     그 블록의 방해물부터 치운다 — block_sort.RELOCATE_CHAIN 단계까지.
from block_geom import axis_blockers  # noqa: E402

T = (400.0, 0.0)
Q = [(T[0] + GAP, T[1]), (T[0], T[1] + GAP), (T[0] + GAP, T[1] + GAP)]
assert graspable_blocker(T, 0.0, Q) is None      # 예전 판정은 그대로 (9b)
cand = axis_blockers(T, 0.0, Q)
assert cand, "이 축을 막는 것이 목록에 나와야 한다"
assert all(not ok for _b, _r, ok in cand), "이 뭉치는 아무도 못 집는 게 맞다"
rooms = [r for _b, r, _ok in cand]
assert rooms == sorted(rooms, reverse=True), rooms
print(f"17 2×2 뭉치 → 후보 {len(cand)}개를 '덜 갇힌 순'으로 내놓는다 "
      f"(여유 {', '.join(f'{r:.0f}mm' for r in rooms)}) — 연쇄로 파고들 수 있다")

# ── 18. 집을 수 있는 것과 없는 것을 갈라서, 쉬운 것부터 준다 ──
#     ㄱ 자 배치: 하나는 집을 수 있고 하나는 아니다. 쉬운 쪽이 먼저 와야
#     한 단계도 안 파고들고 끝난다.
L = [(T[0] + GAP, T[1]), (T[0], T[1] + GAP)]
cand_l = axis_blockers(T, 0.0, L)
easy = [b for b, _r, ok in cand_l if ok]
assert easy, "ㄱ 자에서는 집을 수 있는 것이 있어야 한다"
assert cand_l[0][2] is True, "집을 수 있는 것이 먼저 와야 한다"
print(f"18 ㄱ 자 → 집을 수 있는 것({len(easy)}개)이 목록 앞에 온다 (연쇄 불필요)")

# ── 19. 이 축을 안 막는 것은 목록에 없다 (치워도 헛수고) ──
assert axis_blockers(T, 0.0, [(T[0] + 200.0, T[1])]) == []
print("19 안 막는 이웃은 후보에 넣지 않는다")


# ── 20. 잡은 채로 **파지축 직각 60mm** 끌면 두 축이 다 열린다 ──
#     들었다 놓는 것보다 상승·이송·하강·상승 네 번이 빠진다. 손가락 길(레인)
#     에서 옆으로 빼는 것이라 축을 따라 미는 것(중심거리 67mm)보다 짧다.
from block_geom import SLIDE_MM, seg_dist, slide_dests  # noqa: E402

T = (400.0, 0.0)
for along, side in ((38, 0), (40, 0), (45, 10), (50, -20), (55, 25), (40, 29)):
    b = (T[0] + along, T[1] + side)
    ds = slide_dests(T, b, 0.0)
    assert ds, f"빈 판에서는 끌 자리가 있어야 한다 {b}"
    for d in ds:
        assert approach_gap(T, 0.0, [d]) >= FINGER_T, (b, d)
        assert approach_gap(T, 90.0, [d]) >= FINGER_T, (b, d)
        assert abs(np.hypot(d[0] - b[0], d[1] - b[1]) - SLIDE_MM) < 1e-6
print(f"20 직각 {SLIDE_MM:.0f}mm 끌기 — 6가지 배치 모두 두 축이 열린다")

# ── 21. 끌고 가는 **복도**에 블록이 있으면 그 방향은 안 준다 ──
#     들어서 옮기는 것과 다른 점이다. 끌면 도중의 블록까지 밀어 판이 무너진다.
b = (T[0] + 40.0, T[1])
free = slide_dests(T, b, 0.0)
assert len(free) == 2, "빈 판에서는 양쪽 다 가능"
wall = [(b[0], b[1] + 30.0)]              # +방향 복도 한가운데
one = slide_dests(T, b, 0.0, others=wall)
assert len(one) == 1, one
assert one[0][1] < b[1], "막힌 +쪽 대신 -쪽으로 가야 한다"
both = slide_dests(T, b, 0.0, others=wall + [(b[0], b[1] - 30.0)])
assert both == [], "양쪽 다 막히면 끌 수 없다 → 들어서 옮기는 쪽으로"
print("21 복도가 막힌 방향은 빼고, 양쪽 다 막히면 빈 목록 (들어서 옮기기로 물러남)")

# ── 22. 끌기가 사이를 **더 좁히지는** 않는다 ──
#     막는 블록은 정의상 목표에 바싹 붙어 있다. 절대 거리로 재면 목표 자신
#     때문에 모든 방향이 거부된다(실제로 38mm 목표 때문에 양쪽 다 막혔다).
#     지금보다 가까워질 때만 막는 것이 맞다.
for along in (38.0, 40.0, 45.0):
    nt = (T[0] + along, T[1])
    ds = slide_dests(T, nt, 0.0)
    assert ds, f"목표가 {along:.0f}mm 옆이어도 끌 수 있어야 한다"
    for d in ds:
        assert seg_dist(T, nt, d) >= along - 1e-6, (along, d, seg_dist(T, nt, d))
print("22 끌기 도중 목표와의 거리가 시작보다 가까워지지 않는다 (38/40/45mm)")

# ── 23. 그래도 **더 가까워지는** 방향은 막는다 ──
#     목표를 향해 비스듬히 끌려 하면 도중에 사이가 좁아진다.
side_t = (T[0] + 40.0, T[1] + 50.0)      # 목표에서 대각선으로 떨어진 blocker
ds = slide_dests(side_t, (side_t[0], side_t[1] - 45.0), 90.0, others=[T])
for d in ds:
    assert seg_dist(T, (side_t[0], side_t[1] - 45.0), d) >= \
        min(BLOCK_MM + 5.0, np.hypot(side_t[0] - T[0], side_t[1] - 45.0 - T[1])) - 1e-6
print("23 사이를 더 좁히는 방향은 여전히 거부한다")


# ── 24. 한 축이 안 열리면 **반대 축**의 방해물을 치운다 ──
#     실측 2026-08-08 (로그 62671): best_axis 가 1mm 차이로 축 1°(틈15mm)를
#     골랐는데 그 축을 막는 보라색은 x=648 이라 팔이 못 닿았다. 축 91° 를 막는
#     노란색은 x=597 로 멀쩡히 닿았다 — 축만 바꿨으면 끝났을 일이다.
REAL_G = (598.0, 128.0)                       # 집으려던 초록
REAL_NB2 = {"노란색": (597.0, 177.0), "파란색": (546.0, 186.0),
            "보라색": (648.0, 131.0)}
pts2 = list(REAL_NB2.values())

t2, g2, _ = best_axis(REAL_G, 1.0, pts2)
assert t2 == 0.0, "실기와 같이 축 1° 를 고른다"
b1 = axis_blockers(REAL_G, 1.0, pts2)
b2 = axis_blockers(REAL_G, 91.0, pts2)
assert len(b1) == 1 and abs(b1[0][0][0] - 648.0) < 1, b1
assert len(b2) == 1 and abs(b2[0][0][0] - 597.0) < 1, b2
print(f"24 축 1° 는 보라색(x=648, 도달불가) 하나뿐 / 축 91° 는 노란색(x=597) 하나 "
      f"— 축을 바꾸면 치울 수 있는 것이 나온다")

# ── 25. 그 노란색은 끌 수도 있다 (반대 축으로 넘어가면 끝난다) ──
ds24 = slide_dests(REAL_G, REAL_NB2["노란색"], 91.0,
                   others=[REAL_NB2["파란색"], REAL_NB2["보라색"]])
assert ds24, "노란색은 끌 자리가 있어야 한다"
for d in ds24:
    assert approach_gap(REAL_G, 91.0, [d]) >= FINGER_T
print(f"25 그 노란색은 직각 60mm 끌기로 풀린다 ({len(ds24)}방향 가능)")
