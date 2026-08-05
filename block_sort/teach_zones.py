#!/usr/bin/env python3
"""
구역 좌표 티칭 도구 — M0609 + RG2

  python3 teach_zones.py home          초기 자세로 이동, 그리퍼 열기
  python3 teach_zones.py teach         로봇구역 좌표 기록 (대화형)
  python3 teach_zones.py teach human   인간구역 좌표 기록
  python3 teach_zones.py scan          순회 경유점 4개 기록
  python3 teach_zones.py show          저장값 출력
  python3 teach_zones.py verify        저장된 구역을 저속으로 순회
  python3 teach_zones.py free          순응 모드 on/off (손으로 밀어 움직이기)

티칭 요령
  · 블록을 그리퍼에 물린 채로 위치시킨다.
    빈 손으로 잡으면 실제 놓이는 자리가 블록 두께만큼 어긋난다.
  · 자세(rx,ry,rz)도 함께 저장된다. 매번 같은 방향으로 놓기 위해서다.

구역 세 가지
  로봇구역(zones)       로봇이 블록을 놓는 곳. 자세까지 쓴다.
  인간구역(human_zones) 사람이 배치하는 곳. 로봇은 보기만 해서 x,y 만 쓴다.
  프리구역              둘 중 어디에도 안 속하는 나머지. 따로 티칭하지 않는다 —
                        '어느 구역에도 가깝지 않은 자리'로 정의되기 때문이다.
                        로봇은 여기 있는 블록만 집어간다.
"""
import os
import sys
import time

import rclpy
import yaml

import DR_init

ROBOT_ID, ROBOT_MODEL = "dsr01", "m0609"

# 티칭·검증은 사람이 옆에 있는 작업이라 본 운전보다 느리게 간다.
VEL_J, ACC_J = 40, 40          # 관절 deg/s
VEL_L, ACC_L = 60, 60          # 직선 mm/s

JREADY = [0, 0, 90, 0, 90, 0]
LIFT_HEIGHT = 150              # 접근·이탈 시 z 여유 (mm)
N_ZONES = 4

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP, TOOLCHARGER_PORT = "192.168.1.1", "502"

HERE = os.path.dirname(os.path.abspath(__file__))
ZONES_YAML = os.path.join(HERE, "zones.yaml")

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
rclpy.init()
_node = rclpy.create_node("teach_zones", namespace=ROBOT_ID)
DR_init.__dsr__node = _node

try:
    from DSR_ROBOT2 import (movej, movel, get_current_posx, get_current_posj,
                            mwait, task_compliance_ctrl, release_compliance_ctrl)
except ImportError as e:
    sys.exit(f"DSR_ROBOT2 임포트 실패: {e}\n로봇 드라이버가 떠 있는지 확인하세요.")

sys.path.insert(0, HERE)
from onrobot import RG                                    # noqa: E402

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)


# ───────────────────────── 저장/로드 ─────────────────────────
def load():
    if not os.path.exists(ZONES_YAML):
        return {"home": JREADY, "lift_height": LIFT_HEIGHT, "zones": {}}
    with open(ZONES_YAML) as f:
        return yaml.safe_load(f)


def save(data):
    with open(ZONES_YAML, "w") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False,
                       default_flow_style=None)
    print(f"  → {ZONES_YAML}")


def posx_now():
    """get_current_posx() 는 (좌표, 솔루션공간) 튜플을 준다."""
    p = get_current_posx()
    return [round(float(v), 2) for v in (p[0] if isinstance(p, tuple) else p)]


def wait_gripper():
    while gripper.get_status()[0]:
        time.sleep(0.3)


# ───────────────────────── 모드 ─────────────────────────
def mode_home():
    print(f"\n초기 자세로 이동 — 속도 {VEL_J} (본 운전보다 낮춤)")
    print(f"  목표 관절각 {JREADY}")
    gripper.open_gripper()
    wait_gripper()
    movej(JREADY, vel=VEL_J, acc=ACC_J)
    mwait()
    print(f"  도착 관절각 {[round(float(v),1) for v in get_current_posj()]}")
    print(f"  현재 직교좌표 {posx_now()}")
    print("  그리퍼 열림\n")


def mode_free(on):
    if on:
        # 강성을 낮추면 손으로 밀어 움직일 수 있다. 값이 너무 낮으면 팔이 처진다.
        task_compliance_ctrl(stx=[500, 500, 500, 100, 100, 100])
        print("\n순응 모드 ON — 손으로 팔을 밀어 움직이세요.")
        print("  끝나면:  python3 teach_zones.py free off\n")
    else:
        release_compliance_ctrl()
        print("\n순응 모드 OFF — 팔이 다시 굳었습니다.\n")


def mode_teach(which="robot"):
    """which: robot  로봇이 블록을 놓을 구역 (기존)
              human  사람이 배치하는 구역. 로봇은 보기만 하므로 자세는 안 쓰고
                     x,y 만 쓴다 — 검출된 블록이 어느 칸인지 가리기 위해서다.
    """
    data = load()
    key = "zones" if which == "robot" else "human_zones"
    data.setdefault(key, {})
    title = "로봇구역" if which == "robot" else "인간구역"

    print("\n" + "=" * 56)
    print(f" {title} 좌표 티칭")
    print("=" * 56)
    if which == "robot":
        print("  블록을 그리퍼에 물린 채로 각 구역에 위치시키세요.")
        print("  빈 손으로 잡으면 실제 놓이는 자리가 블록 두께만큼 어긋납니다.")
    else:
        print("  각 칸의 한가운데에 그리퍼 끝을 대세요.")
        print("  여기에 놓지는 않으므로 자세(rx,ry,rz)는 아무래도 좋습니다.")
    print("  Enter  현재 위치 기록 / s  건너뛰기 / q  종료\n")

    for i in range(1, N_ZONES + 1):
        cur = data[key].get(i)
        mark = f"  (기존값 {cur})" if cur else ""
        ans = input(f"  [{title} {i}번] 위치시킨 뒤 Enter{mark} > ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            print("     건너뜀")
            continue
        p = posx_now()
        data[key][i] = p
        print(f"     기록  {p}")

    data["home"] = JREADY
    data["lift_height"] = LIFT_HEIGHT
    save(data)
    print()


N_SCAN = 5                     # 경유점 개수. 1번은 인간구역 관측, 나머지는 프리구역 훑기


def mode_scan():
    """관심 영역 위를 도는 경유점을 기록한다.

    넓게 한 번에 보는 관측 자세를 없애고, 낮게 호버링하며 한 바퀴 도는 방식으로
    바꿨다. 멀리서 한 장에 다 담으면 블록이 작게 잡히고 비스듬히 보여 중심이
    밀린다. 가까이서 나눠 보면 그 두 문제가 같이 사라진다.

    자세는 기울어도 된다 — 검출·파지 때 level_att() 가 수직으로 편다.
    """
    data = load()
    data.setdefault("scan_path", [])
    print("\n" + "=" * 56)
    print(" 순회 경유점 티칭")
    print("=" * 56)
    print("  1번  인간구역 4칸이 한 화면에 다 들어오는 높은 자세")
    print("  2~5번 프리구역 위를 낮게 도는 자세")
    print("  각 점에서 카메라에 보이는 것만 검출됩니다 — 겹치게 잡는 편이 안전합니다.")
    print("  주의: 카메라는 그리퍼에서 (+33, +60)mm 떨어져 있어 팔 위치와 보는 자리가 다릅니다.")
    print("  Enter  현재 위치 기록 / s  건너뛰기 / q  종료\n")

    pts = list(data["scan_path"])
    for i in range(N_SCAN):
        cur = pts[i] if i < len(pts) else None
        mark = f"  (기존값 {cur})" if cur else ""
        what = "인간구역 관측" if i == 0 else "프리구역 훑기"
        ans = input(f"  [{i + 1}번 경유점 — {what}] 자세를 잡은 뒤 Enter{mark} > ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            print("     건너뜀")
            continue
        p = posx_now()
        if i < len(pts):
            pts[i] = p
        else:
            pts.append(p)
        print(f"     기록  {p}")

    data["scan_path"] = pts
    # 관측 자세는 더 이상 쓰지 않는다. 남겨두면 어느 쪽이 쓰이는지 헷갈린다.
    for k in ("observe_human", "observe_free"):
        data.pop(k, None)
    save(data)
    print(f"\n  확인:  python3 block_sort.py scan\n")


def mode_show():
    data = load()
    print(f"\n{ZONES_YAML}")
    print(f"  home        {data.get('home')}")
    print(f"  lift_height {data.get('lift_height')} mm")
    for key, label, hint in [("zones", "로봇구역", "teach"),
                             ("human_zones", "인간구역", "teach human")]:
        z = data.get(key) or {}
        if not z:
            print(f"  {key:12s}(비어 있음 — {hint} 를 실행하세요)")
            continue
        print(f"  {key}   [{label}]")
        for i in sorted(z):
            print(f"    {i}: {z[i]}")
    pts = data.get("scan_path") or []
    if not pts:
        print("  scan_path   (비어 있음 — scan 을 실행하세요)")
    else:
        print("  scan_path   [순회 경유점]")
        for i, p in enumerate(pts, 1):
            print(f"    {i}: {p}")
    print()


def mode_verify():
    data = load()
    z = data.get("zones", {})
    if not z:
        sys.exit("저장된 구역이 없습니다. teach 를 먼저 실행하세요.")
    lift = data.get("lift_height", LIFT_HEIGHT)
    print(f"\n저속 순회 — 그리퍼는 건드리지 않습니다. 속도 {VEL_L} mm/s")
    movej(data.get("home", JREADY), vel=VEL_J, acc=ACC_J)
    mwait()
    for i in sorted(z):
        p = list(z[i])
        above = list(p); above[2] += lift
        print(f"  {i}번 접근")
        movel(above, vel=VEL_L, acc=ACC_L); mwait()
        print(f"  {i}번 하강  {p}")
        movel(p, vel=VEL_L, acc=ACC_L); mwait()
        time.sleep(0.6)
        movel(above, vel=VEL_L, acc=ACC_L); mwait()
    movej(data.get("home", JREADY), vel=VEL_J, acc=ACC_J)
    mwait()
    print("  순회 완료 — 각 구역에 정확히 들어갔는지 확인하세요.\n")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    m = sys.argv[1]
    try:
        if m == "home":
            mode_home()
        elif m == "teach":
            which = sys.argv[2] if len(sys.argv) > 2 else "robot"
            if which not in ("robot", "human"):
                sys.exit("사용법: teach [robot|human]")
            mode_teach(which)
        elif m == "scan":
            mode_scan()
        elif m == "show":
            mode_show()
        elif m == "verify":
            mode_verify()
        elif m == "free":
            mode_free(len(sys.argv) < 3 or sys.argv[2] != "off")
        else:
            sys.exit(__doc__)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
