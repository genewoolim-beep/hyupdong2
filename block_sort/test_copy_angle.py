#!/usr/bin/env python3
"""복제(똑같이/좌우대칭)가 본보기 기울기를 재현하는지 로봇 없이 검증한다.

    python3 test_copy_angle.py        # 로봇도 ROS 도 필요 없다

## 무엇을 확인하나

놓기는 이렇게 한다: **집는 동작은 그대로 두고, 놓기 직전에 손목을 본보기
각도만큼 돌린다.** 블록이 손가락에 정렬돼 물려 있으므로 손목을 돌린 만큼
블록이 돌아간 채 놓인다.

검출각은 이미지(=공구) 기준이라 원래는 base 기준으로 바꿔야 한다. 그 변환을
생략할 수 있는 근거는 하나뿐이다 — **교시된 자세들의 공구 방향이 서로 같다.**
그 전제와, 부호가 정말 지워지는지를 실제 zones.yaml 값으로 확인한다.
전제가 깨지면(교시를 다시 하면서 손목을 돌려 잡으면) 여기서 먼저 걸린다.
"""
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from block_geom import fold90, rotate_tool, tool_yaw   # noqa: E402

GRASP_ROT = 90.0          # block_sort 와 같은 값
LEVEL_RY = 179.9
CFG = yaml.safe_load(open(os.path.join(HERE, "zones.yaml")))


def level_att(att):
    return [att[0], LEVEL_RY, att[2]]


def d90(a, b):
    """90° 대칭을 감안한 두 방위의 차이(0~45)."""
    return abs(((a - b + 45.0) % 90.0) - 45.0)


# ── 1. 접기 ──
for raw, want in ((0, 0), (20, 20), (70, -20), (90, 0), (45, -45), (-20, -20), (135, -45)):
    got = fold90(raw)
    assert abs(got - want) < 1e-6, (raw, got, want)
print("1 fold90  0→0  20→20  70→-20  90→0  135→-45   (정사각 90° 대칭)")

# ── 2. 교시 자세들의 공구 방향이 같은가 (이 전제가 깨지면 나머지가 무의미) ──
poses = ([("로봇%d" % k, v) for k, v in CFG["zones"].items()]
         + [("인간%d" % k, v) for k, v in CFG["human_zones"].items()]
         + [("경유%d" % i, v) for i, v in enumerate(CFG["scan_path"], 1)])
yaws = {n: tool_yaw(p[3:]) for n, p in poses}
ref = yaws["경유1"]
worst = max(yaws, key=lambda n: d90(yaws[n], ref))
spread = d90(yaws[worst], ref)
assert spread < 5.0, (
    f"교시 자세들의 공구 방향이 {spread:.1f}° 어긋난다({worst}) — 검출각을 그대로 "
    "놓기에 쓸 수 없다. place() 주석의 전제가 깨졌으니 base 변환을 넣어야 한다.")
print(f"2 교시 자세 공구방향 일치   최대 어긋남 {spread:.1f}° ({worst})  < 5° 기준")

# ── 3. 본보기와 같은 기울기로 놓이는가 ──
#     본보기: 경유점에서 각도 a 로 검출됐다 → 파지는 rotate_tool(관측, 90+fold(a))
#             블록은 손가락에 정렬되므로 그 손목 방향이 곧 블록 방향이다.
#     놓기:   rotate_tool(구역, fold(a)) → 이 손목 방향이 놓인 블록 방향이다.
#     둘이 90° 대칭 안에서 같아야 한다. (부호는 같은 함수를 쓰므로 지워진다)
scan = CFG["scan_path"][0]
for a in (0.0, 12.0, 20.0, 44.0, 67.0, 89.0):
    obs = list(scan[:3]) + level_att(scan[3:])
    grip = tool_yaw(rotate_tool(obs, GRASP_ROT + fold90(a))[3:])   # 물린 블록 방향
    for z, zp in sorted(CFG["zones"].items()):
        put = tool_yaw(rotate_tool(list(zp), fold90(a))[3:])       # 놓인 블록 방향
        err = d90(put, grip)
        assert err < 5.0, (a, z, err)
print("3 놓인 기울기 = 본보기 기울기   검출각 0~89°, 로봇구역 4칸 전부 오차 < 5°")

# ── 4. 좌우대칭이면 기울기도 뒤집힌다 ──
#     거울에 비친 +20° 는 -20° 다. 부호를 안 바꾸면 배치만 대칭이고 블록이
#     기운 방향은 원본과 같아 눈에 대칭으로 보이지 않는다.
for a in (12.0, 20.0, 44.0):
    same = fold90(a)
    mir = fold90(-a)
    assert abs(same + mir) < 1e-6, (a, same, mir)
    assert abs(same - mir) > 1e-6
print("4 좌우대칭은 기울기 부호 반전   +20°→-20°  (똑같이 는 그대로)")

# ── 5. 손목을 돌리면 놓이는 자리가 얼마나 밀리나 (보정 안 하는 근거) ──
#     티칭 좌표는 티칭 당시 손목 방향 기준이다. 손목을 d 돌리면 손가락 중심이
#     TCP 둘레로 돌아 그만큼 밀린다. |OFFSET_TOOL| = 8.49mm.
OFFSET_TOOL = np.array([-6.0, 6.0])
r = float(np.hypot(*OFFSET_TOOL))
for d in (10, 20, 45):
    shift = 2 * r * np.sin(np.radians(d) / 2)
    assert shift < 10.0, (d, shift)      # 놓기 오차(약 10mm) 안이어야 한다
print(f"5 손목 회전이 만드는 위치 밀림   10°:{2*r*np.sin(np.radians(10)/2):.1f}mm  "
      f"20°:{2*r*np.sin(np.radians(20)/2):.1f}mm  45°:{2*r*np.sin(np.radians(45)/2):.1f}mm"
      f"   (놓기 오차 10mm 안)")

print("\n전부 통과 — 다만 실기에서 한 번은 확인해야 한다:")
print("  python3 block_sort.py copy   →  로그의 '인간n번 = 색 기울기 X°' 와")
print("  이어서 python3 block_sort.py scan  →  '로봇구역 점유' 의 각도가 같은지")
