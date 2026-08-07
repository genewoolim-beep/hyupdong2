#!/usr/bin/env python3
"""옆 블록 피하기(손목 90° 회전)를 로봇 없이 검증한다.

    python3 test_avoid.py

그리퍼는 한 축의 양쪽에서 손가락이 내려와 안쪽으로 닫힌다. 그 축에 다른 블록이
얹혀 있으면 손가락이 그것을 치거나 밀어낸다. 정사각 블록은 90° 대칭이라 손목을
90° 돌리면 물리는 품질은 그대로면서 접근 방향만 바뀐다 — 여유가 넓은 쪽을 고른다.

여기서 확인하는 것은 **엉뚱하게 돌리지 않는가**다. 필요 없을 때 돌리면 파지
보정(grasp_offset)만 흔들리고 얻는 게 없다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from block_geom import approach_gap, best_axis, BLOCK_MM, FINGER_T, LANE_HALF  # noqa: E402

T = (400.0, 0.0)          # 집을 블록 중심

# ── 1. 방해가 없으면 틈은 무한, 돌리지 않는다 ──
assert approach_gap(T, 0.0, []) == float("inf")
turn, g, g0 = best_axis(T, 0.0, [])
assert turn == 0.0
print("1 옆 블록 없음 → 회전 없음")

# ── 2. x 축(0°)에 딱 붙은 블록: x 로는 못 물고 y 로는 물 수 있다 ──
#     중심거리 40mm → 표면 틈 5mm. 손가락(FINGER_T)이 안 들어간다.
nb = [(440.0, 0.0)]
gx = approach_gap(T, 0.0, nb)
gy = approach_gap(T, 90.0, nb)
assert abs(gx - 5.0) < 1e-6, gx
assert gy == float("inf"), gy      # y 축에서 보면 옆으로 비켜 있다
turn, g, g_before = best_axis(T, 0.0, nb)
assert turn == 90.0, (turn, g, g_before)
print(f"2 x 축 40mm 옆에 블록 → 틈 {gx:.0f}mm (손가락 {FINGER_T:.0f}mm 못 들어감) "
      f"→ 90° 회전, 틈 무한")
print(f"   ※ 필요한 최소 중심거리 = 블록 {BLOCK_MM:.0f} + 손가락 {FINGER_T:.0f} "
      f"= {BLOCK_MM + FINGER_T:.0f}mm")

# ── 3. 이미 넉넉하면 돌리지 않는다 ──
#     **문턱 숫자를 박지 않는다.** FINGER_T 는 실측으로 바뀌는 값이고(10 → 27),
#     시험이 그 숫자에 묶이면 값을 고칠 때마다 엉뚱하게 깨진다.
roomy = BLOCK_MM + FINGER_T + 15.0          # 넉넉한 중심거리
turn, g, _ = best_axis(T, 0.0, [(T[0] + roomy, T[1])])
assert turn == 0.0 and abs(g - (FINGER_T + 15.0)) < 1e-6, (turn, g)
print(f"3 틈 {g:.0f}mm (문턱 {FINGER_T:.0f}mm) → 넉넉하니 회전 없음")

# ── 4. 옆으로 비킨 블록은 무시한다 (손가락이 그 옆을 지난다) ──
far = [(440.0, LANE_HALF + 1.0)]
assert approach_gap(T, 0.0, far) == float("inf")
near = [(440.0, LANE_HALF - 1.0)]
assert approach_gap(T, 0.0, near) == 5.0
print(f"4 축에서 옆으로 {LANE_HALF:.0f}mm 이상 비키면 방해 아님")

# ── 5. 네 방향이 다 막히면 돌려도 못 푼다 — 그래도 더 넓은 쪽을 준다 ──
box = [(440.0, 0.0), (360.0, 0.0), (400.0, 42.0), (400.0, -42.0)]
turn, g, g0 = best_axis(T, 0.0, box)
assert g < FINGER_T, g                     # 둘 다 좁다
assert turn in (0.0, 90.0)
assert g >= g0                             # 준 쪽이 원래보다 나쁘지는 않다
print(f"5 사방이 막힘 → 회전 {turn:.0f}°, 틈 {g:.0f}mm (경고 대상)")

# ── 6. 기울어진 블록에서도 축을 따라 잰다 ──
#     축 20° 위에 놓인 방해 블록: 20° 축에서는 방해, 110° 축에서는 아니다
import numpy as np  # noqa: E402
r = np.radians(20.0)
nb20 = [(T[0] + 40 * np.cos(r), T[1] + 40 * np.sin(r))]
assert abs(approach_gap(T, 20.0, nb20) - 5.0) < 1e-6
assert approach_gap(T, 110.0, nb20) == float("inf")
turn, _, _ = best_axis(T, 20.0, nb20)
assert turn == 90.0
print("6 기울어진(20°) 파지축에서도 같은 판단")

print("\n전부 통과 — 실기에서는 손목이 90° 돌아 물리는지, 보정이 함께 돌아가는지 본다")
