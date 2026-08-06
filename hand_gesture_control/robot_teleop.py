#!/usr/bin/env python3
"""손동작 조종 값을 실제 로봇 움직임으로 옮긴다.

hand_gesture_control.py 는 손을 읽어 화면에만 그렸다. 이 모듈이 그 값을
로봇 **속도**로 바꿔 팔에 흘려보낸다. `--robot` 없이는 import 되지 않아
기본이 안전(화면만)이다.

## 왜 speedl 인가 — 위치가 아니라 속도를 준다

`movel` 은 "이 점으로 가라". 매 프레임 밀어넣으면 앞 모션이 안 끝나 거부된다
("A motion is ongoing", 실측 한 실행에 23회). 스트리밍에 못 쓴다.

`servol` 은 스트리밍은 되지만 여전히 **위치**를 준다. 목표가 로봇보다 앞서
달아나면 컨트롤러가 그 거리를 메우려고 **가속한다** — 지정 속도를 넘겨
판에 부딪힌 원인이 이것이었다(실측 2026-08-06, 두 번). 목표점 이동량을
제한하고(STEP_MAX) 목표를 실제 위치에 묶어도(LEASH) 근본은 남았다.
제한이 전부 '목표점'에 걸릴 뿐, 로봇이 그 목표까지 얼마나 빨리 가는지는
컨트롤러가 정하기 때문이다.

`speedl` 은 **속도를 직접 지령한다.** 목표점 개념이 없으므로 따라잡기 가속이
원리적으로 생기지 않는다. 손이 중앙이면 0, 벗어나면 일정 속도 — 그뿐이다.

## 십자선 방식

방향만 본다. **얼마나 벗어났는지는 속도를 안 바꾼다.**
속도가 변하지 않아야 예측 가능하고, 그게 이 방식을 고른 이유다.

  중앙(데드존)  →  정지
  위/아래       →  ±x 로 TELEOP_SPEED
  좌/우         →  ±y 로 TELEOP_SPEED (거울)
  z 게이지      →  가리키는 높이로 일정 속도, 도달하면 정지

## 경계

교시한 상자(teach_box.py) 밖으로 나가는 **방향의 속도만 0** 으로 만든다.
위치를 자르는 게 아니라 속도를 막는 것이라, 이번엔 실제 팔에 직접 걸린다.
정지에도 거리가 필요하므로 경계 BRAKE 안에 들어오면 미리 끊는다.

환경변수로 전부 조정한다.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(HERE)

# ── 속도 ─────────────────────────────────────────────────────────
# **속도를 직접 지령한다(speedl).** 목표점을 주고 따라가게 하면(servol) 목표가
# 로봇보다 앞서 달아날 때 컨트롤러가 따라잡으려 가속한다 — 지정 속도를 넘겨
# 판에 부딪힌 원인이 그것이었다(실측 2026-08-06, 두 번).
# 속도를 직접 주면 목표점 개념이 없어 그 가속이 원리적으로 생기지 않는다.
#
# 십자선 방식: 손이 중앙이면 정지, 방향으로 벗어나면 그 축으로 일정 속도.
# 얼마나 벗어났는지는 **속도에 영향을 주지 않는다** — 방향만 본다.
# 속도가 변하지 않아야 예측 가능하고, 그게 이 방식을 고른 이유다.
TELEOP_SPEED = float(os.environ.get("TELEOP_SPEED", 20.0))   # mm/s
TELEOP_ACC = float(os.environ.get("TELEOP_ACC", 100.0))      # mm/s^2 (지령 전환 응답)

# 십자선 중앙의 정지 구역. 화면 비율(0~1)에서 중앙으로부터 이 반경 안이면 정지.
DEADZONE = float(os.environ.get("TELEOP_DEADZONE", 0.08))

# z 게이지가 가리키는 높이와 이만큼 안이면 z 정지. 게이지는 높이를 뜻하므로
# 도달하면 멈춰야 한다 — 방향만 보고 일정 속도로 가되, 다 오면 선다.
Z_TOL = float(os.environ.get("TELEOP_Z_TOL", 3.0))           # mm

# 경계에 이만큼 다가가면 그 축 속도를 0 으로 만든다. 정지에도 거리가 필요하다.
BRAKE = float(os.environ.get("TELEOP_BRAKE", 15.0))          # mm

# 백그라운드가 실제 위치를 읽는 주기(초). 경계 판정에만 쓴다.
# 실제 위치를 몇 프레임마다 읽을지. 경계 판정에만 쓰므로 드물어도 된다.
POLL_EVERY = int(os.environ.get("TELEOP_POLL_EVERY", 10))

# ── 작업영역 (base 좌표, mm) ─────────────────────────────────────
# 2026-08-06 교시값. teach_box.py 로 두 모서리를 찍어 얻었다.
# 이것이 **절대 경계**다 — 로봇의 실제 위치가 여기를 넘으면 안 된다.
X_MIN = float(os.environ.get("TELEOP_X_MIN", 199.2))
X_MAX = float(os.environ.get("TELEOP_X_MAX", 566.2))
Y_MIN = float(os.environ.get("TELEOP_Y_MIN", -231.3))
Y_MAX = float(os.environ.get("TELEOP_Y_MAX", 280.9))
Z_MIN = float(os.environ.get("TELEOP_Z_MIN", -17.5))
Z_MAX = float(os.environ.get("TELEOP_Z_MAX", 218.2))

# 교시 경계에서 이만큼 안쪽을 실제 사용 범위로 삼는다.
# 속도를 막아도 정지까지 약간 미끄러지므로 여유를 둔다.
SAFETY_MARGIN = float(os.environ.get("TELEOP_MARGIN", 15.0))


def _inner(lo, hi, m=None):
    """경계를 안쪽으로 물린 (하한, 상한). 상자가 너무 좁으면 가운데로 붙인다."""
    m = SAFETY_MARGIN if m is None else m
    if hi - lo <= 2 * m:
        c = (lo + hi) / 2.0
        return c, c
    return lo + m, hi - m


# 실제 사용 범위. 이 밖으로 나가는 방향은 속도가 0 이 된다.
TX_MIN, TX_MAX = _inner(X_MIN, X_MAX)
TY_MIN, TY_MAX = _inner(Y_MIN, Y_MAX)
TZ_MIN, TZ_MAX = _inner(Z_MIN, Z_MAX)

# 손이 이만큼 연속으로 안 보이면 속도 0 을 보낸다 (데드맨).
LOST_HOLD = int(os.environ.get("TELEOP_LOST_HOLD", 3))

# 로봇 연결에 이만큼 못 붙으면 조종을 끄고 화면만 돌린다.
# DSR 호출이 타임아웃 없이 멈추는 것을 여기서 막는다 (connect 주석 참고).
CONNECT_TIMEOUT = float(os.environ.get("TELEOP_CONNECT_TIMEOUT", 5.0))

# 걸려 있어야 하는 TCP. 작업모드(block_sort)와 같은 값을 써야 한다.
EXPECT_TCP = os.environ.get("EXPECT_TCP", "GripperDA_v1")

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP, TOOLCHARGER_PORT = "192.168.1.1", "502"
ROBOT_ID, ROBOT_MODEL = "dsr01", "m0609"


# DSR 연결은 **프로세스당 한 번만** 만든다.
#
# DSR_ROBOT2 는 import 시점에 g_node 로 서비스 클라이언트를 만들어 모듈에
# 들고 있는다. 세션마다 노드를 새로 만들고 옛것을 destroy 하면, 그 클라이언트가
# 파괴된 노드를 가리켜 두 번째 진입부터 명령이 안 나간다.
# 실측 2026-08-06: 제어 → 작업 → 제어 로 돌아왔을 때 화면은 떴는데 팔이
# 안 움직였다. 그래서 한 번 만든 것을 계속 재사용한다.
_DSR = {}


def _dsr_connect():
    """DSR 노드와 함수들을 한 번만 만들어 캐시한다."""
    if _DSR:
        return _DSR
    import rclpy
    import DR_init
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("gesture_teleop", namespace=ROBOT_ID)
    # 모듈 수준 대입이어야 한다. 클래스 안에서 쓰면 이름 맹글링으로
    # _RobotTeleop__dsr__node 가 되어 DSR_ROBOT2 가 None 을 읽는다.
    setattr(DR_init, "__dsr__node", node)
    from DSR_ROBOT2 import speedl, get_current_posx, get_tcp, set_tcp
    _DSR.update(node=node, speedl=speedl, get_posx=get_current_posx,
                get_tcp=get_tcp, set_tcp=set_tcp)
    return _DSR


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


class RobotTeleop:
    """조종 값(화면 비율 0~1)을 로봇 속도로 바꿔 speedl 로 흘린다.

    쓰는 쪽은 매 프레임 update() 만 부르면 된다. 로봇 연결이 안 되면
    enabled=False 로 남아 아무 일도 하지 않는다 — 화면 동작은 그대로다.
    """

    def __init__(self):
        self.enabled = False
        self.att = None             # 자세는 진입 시점 것을 유지한다
        self.lost = 0
        self._grip_open = None      # 마지막으로 보낸 그리퍼 상태
        self._err = None
        self._last_cmd = None   # 마지막으로 보낸 속도 지령
        self._tick = 0
        self._pos_cache = None   # 백그라운드가 갱신

    # ── 연결 ──
    def connect(self):
        """DSR 과 그리퍼에 붙는다. 실패해도 예외를 올리지 않는다.

        조종 화면은 로봇 없이도 의미가 있으므로(인식 확인), 연결 실패는
        기능을 끄는 것으로 끝낸다. 이유는 로그로 남긴다.
        """
        try:
            d = _dsr_connect()
            self._speedl = d["speedl"]
            self._get_posx = d["get_posx"]
            self._node = d["node"]

            # TCP 가 풀려 있으면 좌표계가 달라져 상자 판정이 무의미해진다.
            try:
                if d["get_tcp"]() != EXPECT_TCP:
                    print(f"  TCP 가 '{d['get_tcp']()}' — '{EXPECT_TCP}' 로 다시 겁니다")
                    d["set_tcp"](EXPECT_TCP)
            except Exception as e:
                print(f"  TCP 확인 실패({e})")

            sys.path.insert(0, os.path.join(_ROOT, "block_sort"))
            from onrobot import RG
            self._gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

            # **진입 시점의 현재 위치에서 시작한다.** 안 그러면 첫 명령에
            # 팔이 엉뚱한 곳으로 튄다.
            #
            # DSR 의 get_current_posx 는 **타임아웃이 없다**(spin_until_future_complete
            # 를 인자 없이 부른다). block_sort 가 이미 DSR 에 붙어 있으면 응답이
            # 안 와 여기서 영원히 멈춘다 — 실측 2026-08-06: 제어모드로 들어가면
            # 멈췄다가 block_sort 쪽 타임아웃으로만 빠져나왔다.
            # 별도 스레드에서 부르고 시간을 재서, 안 오면 조종만 끈다.
            # 카메라와 화면은 그대로 동작해야 한다 — 로봇 없이도 인식 확인은 된다.
            import threading
            box = {}

            def _read():
                try:
                    box["p"] = self._get_posx()[0]
                except Exception as e:
                    box["e"] = e

            th = threading.Thread(target=_read, daemon=True)
            th.start()
            th.join(timeout=CONNECT_TIMEOUT)
            if "p" not in box:
                raise RuntimeError(
                    f"로봇 위치를 {CONNECT_TIMEOUT:.0f}초 안에 못 읽었습니다 "
                    "— block_sort 가 DSR 을 잡고 있을 수 있습니다")
            cur = box["p"]
            # 진입 시점에 팔이 상자 밖에 있을 수 있다(작업모드가 옮겨둔 자리).
            # 목표를 상자 안으로 넣어 시작한다 — 밖에서 시작하면 첫 명령이
            # 곧바로 경계를 향해 달린다.
            self.att = list(cur[3:])
            self._pos_cache = cur
            self.enabled = True
            print(f"  로봇 조종 연결됨 — 시작 위치 "
                  f"[{cur[0]:.0f}, {cur[1]:.0f}, {cur[2]:.0f}]  "
                  f"속도 {TELEOP_VEL:.0f}mm/s  한틱 최대 {STEP_MAX:.1f}mm")
            print(f"  작업영역(교시)  x {X_MIN:.0f}~{X_MAX:.0f}  "
                  f"y {Y_MIN:.0f}~{Y_MAX:.0f}  z {Z_MIN:.0f}~{Z_MAX:.0f}")
            print(f"  목표 한계(-{SAFETY_MARGIN:.0f}mm)  x {TX_MIN:.0f}~{TX_MAX:.0f}  "
                  f"y {TY_MIN:.0f}~{TY_MAX:.0f}  z {TZ_MIN:.0f}~{TZ_MAX:.0f}")
            return True
        except Exception as e:
            self._err = str(e)
            print(f"  로봇 조종을 켜지 못했습니다({e}) — 화면만 동작합니다.")
            return False

    # ── 매 프레임 ──
    def update(self, pos_xy, z_norm, gripper_open, hand_seen):
        """조종 값을 **속도**로 바꿔 로봇에 보낸다.

        pos_xy       화면 비율 (0~1, 0~1). 십자선 중앙은 (0.5, 0.5)
        z_norm       0~1. 게이지가 가리키는 높이
        gripper_open True 면 열기
        hand_seen    이번 프레임에 손이 보였는가 (데드맨)

        방향만 보고 **일정 속도**를 준다. 얼마나 벗어났는지는 속도를 안 바꾼다 —
        속도가 변하지 않아야 예측 가능하고, 그게 이 방식을 고른 이유다.
        """
        if not self.enabled:
            return

        # 손이 사라지면 즉시 정지 (데드맨)
        if not hand_seen:
            self.lost += 1
            if self.lost >= LOST_HOLD:
                self._send([0.0, 0.0, 0.0])
                return
        else:
            self.lost = 0

        # 실제 위치는 **메인 스레드에서** 드물게 읽는다.
        # 백그라운드 스레드로 읽으면 rclpy 전역 실행기를 프레임 읽기와 다퉈
        # 터진다(실측 2026-08-06: AttributeError __enter__ 로 죽었다).
        # 속도 지령은 방향만 보므로 위치는 경계 판정에만 쓰이고, BRAKE(15mm)
        # 여유가 갱신 지연을 흡수한다.
        self._tick += 1
        if self._tick % POLL_EVERY == 0 or self._pos_cache is None:
            got = self._read_posx()
            if got is not None:
                self._pos_cache = got

        cur = self._pos_cache
        u = float(pos_xy[0]) - 0.5
        v = float(pos_xy[1]) - 0.5

        # ── xy: 십자선. 중앙이면 정지, 벗어난 축으로 일정 속도 ──
        # 화면 위(v<0)가 로봇 앞(+x), 화면 오른쪽(u>0)이 로봇 왼쪽(+y) — 거울.
        vx = -TELEOP_SPEED if v > DEADZONE else (TELEOP_SPEED if v < -DEADZONE else 0.0)
        vy = -TELEOP_SPEED if u > DEADZONE else (TELEOP_SPEED if u < -DEADZONE else 0.0)

        # ── z: 게이지가 가리키는 높이로 일정 속도. 도달하면 정지 ──
        vz = 0.0
        if cur is not None:
            z_want = TZ_MIN + _clamp(float(z_norm), 0.0, 1.0) * (TZ_MAX - TZ_MIN)
            gap = z_want - cur[2]
            if abs(gap) > Z_TOL:
                vz = TELEOP_SPEED if gap > 0 else -TELEOP_SPEED

        # ── 경계: 밖으로 나가는 방향의 속도만 0 으로 ──
        # 정지에도 거리가 필요하므로 경계 BRAKE 안에 들어오면 미리 끊는다.
        # 위치가 아니라 **속도**를 막는 것이라, 이번엔 실제 팔에 직접 걸린다.
        if cur is not None:
            vx = self._gate(vx, cur[0], TX_MIN, TX_MAX)
            vy = self._gate(vy, cur[1], TY_MIN, TY_MAX)
            vz = self._gate(vz, cur[2], TZ_MIN, TZ_MAX)
        else:
            # 위치를 모르면 움직이지 않는다. 경계를 확인할 방법이 없다.
            vx = vy = vz = 0.0

        self._send([vx, vy, vz])
        self._set_gripper(gripper_open)

    @staticmethod
    def _gate(v, pos, lo, hi):
        """경계 밖으로 나가는 방향이면 0. 안으로 들어오는 방향은 그대로 둔다."""
        if v > 0 and pos >= hi - BRAKE:
            return 0.0
        if v < 0 and pos <= lo + BRAKE:
            return 0.0
        return v

    def _send(self, xyz):
        """속도 지령. 회전은 항상 0 — 조종은 자세를 안 바꾼다."""
        if xyz == self._last_cmd:
            return                      # 같은 지령을 반복해 보내지 않는다
        self._last_cmd = list(xyz)
        try:
            self._speedl([xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0],
                         [TELEOP_ACC, TELEOP_ACC], 0.0)
        except Exception as e:
            print(f"  speedl 실패({e}) — 조종을 끕니다.")
            self.enabled = False

    def _read_posx(self):
        """실제 위치. 못 읽으면 None.

        **메인 스레드에서 직접 부른다.** 스레드로 감싸면 rclpy 전역 실행기를
        프레임 읽기와 다퉈 터진다(실측 2026-08-06). 여기가 실행기를 쓰는
        유일한 곳이 되도록 두는 것이 안전하다.
        """
        try:
            return self._get_posx()[0]
        except Exception:
            return None

    def _set_gripper(self, want_open):
        """상태가 바뀔 때만 보낸다. 매 프레임 모드버스를 두드리면 컨트롤러의
        모션 큐와 겹쳐 이동 명령이 거부된다(실측에서 그 조합으로 많이 났다)."""
        if want_open == self._grip_open:
            return
        try:
            if want_open:
                self._gripper.open_gripper()
            else:
                self._gripper.close_gripper()
            self._grip_open = want_open
        except Exception as e:
            print(f"  그리퍼 명령 실패({e})")

    def status(self):
        """화면에 겹쳐 보여줄 한 줄."""
        if not self.enabled:
            return "ROBOT OFF" + (f" ({self._err})" if self._err else "")
        p = self._pos_cache
        v = self._last_cmd or [0, 0, 0]
        pos = f"[{p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}]" if p is not None else "[--]"
        return f"ROBOT {pos}  v=({v[0]:+.0f},{v[1]:+.0f},{v[2]:+.0f}) mm/s"

    def close(self):
        """조종만 끈다. **노드는 파괴하지 않는다.**

        DSR 서비스 클라이언트가 이 노드에 묶여 있어, 파괴하면 다음 세션에서
        명령이 안 나간다(_dsr_connect 주석 참고). 프로세스가 끝날 때 함께 정리된다.
        """
        try:
            self._send([0.0, 0.0, 0.0])    # 나가기 전에 반드시 세운다
        except Exception:
            pass
        self.enabled = False
        self._last_cmd = None
