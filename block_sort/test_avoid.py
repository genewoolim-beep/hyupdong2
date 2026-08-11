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


# ── 7. 회피 회전을 손목이 덜 도는 쪽으로 접는다 ──
#     실측 2026-08-08: 기울기 41° 인 보라가 손목 +131° 인 상태에서 +90 을 더해
#     **+221°** 가 되어 "블록 위로 접근하지 못해 파지를 중단합니다" 로 죽었다.
#     같은 배치에서 두 번 죽었고, 그 보라가 막고 있던 초록까지 같이 실패했다.
#     ±90 은 같은 물림이므로(그리퍼가 닫히는 것은 직선) -90 을 고르면 +41° 다.
from block_geom import fold_turn  # noqa: E402

assert fold_turn(131.0, 90.0) == -90.0, "실기에서 죽은 그 경우"
assert 131.0 + fold_turn(131.0, 90.0) == 41.0
assert fold_turn(-131.0, 90.0) == 90.0, "반대로 돌아 있으면 +90 이 맞다"
assert fold_turn(0.0, 90.0) == 90.0, "가운데면 그대로 둔다 (이유 없이 안 바꾼다)"
assert fold_turn(45.0, 90.0) == -90.0, "45°+90=135 보다 45°-90=-45 가 낫다"
assert fold_turn(10.0, 0.0) == 0.0, "안 돌릴 때는 건드리지 않는다"
# 어떤 손목각에서도 접은 뒤가 원래보다 나빠지지 않는다.
for lr in range(-180, 181, 5):
    assert abs(lr + fold_turn(float(lr), 90.0)) <= abs(lr + 90.0) + 1e-9, lr
print("7 회피 회전을 손목이 덜 도는 쪽으로 접는다 (131°+90=221° → 131°-90=41°)")

# ── 8. 이웃의 기울기를 봐야 한다 — 돌아간 정사각은 더 내민다 ──
#     실측 2026-08-10 (로그 python3_12362): 목표 노란색(547.4,-73.8) 기울기 53°,
#     파지축 127°. 주황색(503,-21) 의 틈을 **34mm** 로 보고 "충분(≥27)" 판정해
#     그대로 내려갔는데 그리퍼가 그 주황색에 부딪혔다 — 컨트롤러가 보호정지해
#     "20.0초 동안 STANDBY 가 안 됐습니다" → "이동 명령이 거부됐습니다" 로 끝났다.
#     주황색이 돌아 있었던 것이다. 정사각은 φ 만큼 돌면 그 방향 폭이
#     35(|cosφ|+|sinφ|) 로 커진다 — 반폭으로 17.5 → 최대 24.7mm.
from block_geom import approach_gap, nb_half, nb_lane, FINGER_T  # noqa: E402

TGT, AXIS, ORANGE = (547.4, -73.8), 127.0, (503.0, -21.0)

# 각도를 안 넘기면 예전 그대로다 — 기존 호출부의 동작이 안 바뀐다.
flat = approach_gap(TGT, AXIS, [ORANGE])
assert abs(flat - 33.9) < 0.2, f"축 정렬 가정이면 {flat:.1f}mm (그때 로그의 34mm)"
# **그때 쓰던 문턱과 견준다.** 여기서 말하는 것은 '그날 왜 통과했나' 라는
# 지나간 사실이라, 지금 값(FINGER_T)이 아니라 그때 값으로 재야 뜻이 맞다.
FINGER_T_THEN = 27.0                     # 2026-08-10 당시 값 (손가락 25 + 여유 2)
assert flat >= FINGER_T_THEN, "그래서 그때는 '충분' 으로 통과했다"

# 기울기를 넘기면 그만큼 줄어든다. 45° 에서 최대 7.2mm.
worst = approach_gap(TGT, AXIS, [ORANGE + (AXIS + 45.0,)])
assert abs((flat - worst) - 7.2) < 0.1, f"45° 이웃이면 {flat - worst:.1f}mm 줄어야"
assert worst < FINGER_T, f"{worst:.1f}mm — 손가락({FINGER_T})이 안 들어간다. 이제 걸러진다"

# 기울기가 커질수록 단조롭게 좁아진다.
gaps = [approach_gap(TGT, AXIS, [ORANGE + (AXIS + d,)]) for d in (0, 10, 20, 30, 45)]
assert all(a >= b - 1e-9 for a, b in zip(gaps, gaps[1:])), gaps
assert abs(gaps[0] - flat) < 1e-9, "축과 나란한 이웃은 예전 값과 같아야 한다"

# 90° 대칭 — 정사각이므로 φ 와 φ+90 은 같은 폭이다.
for d in (0.0, 17.0, 33.0, 45.0):
    assert abs(nb_half((0, 0, d), 0.0) - nb_half((0, 0, d + 90.0), 0.0)) < 1e-9, d

# 레인 판정도 같이 넓어진다 — 돌아간 블록은 옆으로도 더 내민다.
assert nb_lane((0, 0, 45.0), 0.0) > nb_lane((0, 0, 0.0), 0.0)
assert abs(nb_lane((0, 0), 0.0) - 29.5) < 1e-9, "모르면 예전 값(29.5)"
print("8 이웃 기울기를 반영한다 — 그때 그 배치에서 34mm → 26.8mm 로 걸러진다")

# ── 9. 지금 문턱에서 그 배치가 실제로 걸러지는가 ──
#     **문턱 값 자체를 시험이 정하지는 않는다.** 그건 실기에서 조율하는 값이다
#     (27 → 38 → 34, block_geom.FINGER_T 주석). 여기서 붙잡아 둘 것은 하나다:
#     그때 부딪혔던 그 배치가 지금 값에서 걸러지는가.
assert worst < FINGER_T, f"기울기를 반영한 {worst:.1f}mm 는 걸러져야 한다"
margin_flat = FINGER_T - flat          # 기울기를 몰랐을 때의 여유
print(f"9 문턱 {FINGER_T:.0f}mm — 그 배치는 걸러진다 "
      f"(기울기 반영 {worst:.1f}mm, 여유 {FINGER_T - worst:.1f}mm)")

# ── 10. 기울기 보정에 얼마나 기대고 있나 ──
#     8번의 판정은 기울기 보정 하나에 기대고 있다. 그 보정이 맞으려면 검출
#     각도가 맞아야 하는데, 같은 로그에서 각도가 4° 씩 흔들린 적이 있다.
#     문턱이 축정렬 값(33.9mm)보다 높으면 보정이 통째로 빗나가도 안전한 쪽으로
#     떨어진다 — 그 여유가 지금 몇 mm 인지 눈에 보이게 남긴다.
if margin_flat > 0:
    print(f"10 기울기 보정이 빗나가도 걸러진다 — 축정렬로 본 {flat:.1f}mm 에 "
          f"여유 {margin_flat:.1f}mm"
          + ("  ※ 1mm 미만이라 사실상 경계선이다" if margin_flat < 1.0 else ""))
else:
    print(f"10 기울기 보정에 **의존한다** — 축정렬로 보면 {flat:.1f}mm 라 "
          f"문턱({FINGER_T:.0f}mm)을 통과한다. 검출 각도가 틀리면 그대로 내려간다")

print("\n전부 통과 — 실기에서는 손목이 90° 돌아 물리는지, 보정이 함께 돌아가는지 본다")
