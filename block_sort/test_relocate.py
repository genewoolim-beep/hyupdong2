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
