#!/usr/bin/env python3
"""손동작 조종 값을 실제 로봇 움직임으로 옮긴다.

hand_gesture_control.py 는 손을 읽어 화면에만 그렸다. 이 모듈이 그 값을
로봇 좌표로 바꿔 팔에 흘려보낸다. 파일을 나눈 이유는 두 가지다 —
조종 인식(그쪽)과 로봇 제어(이쪽)는 고쳐야 할 이유가 서로 다르고,
`--robot` 없이는 아예 import 되지 않아 **기본이 안전(화면만)** 이 된다.

## 왜 servol 인가

movel 은 '한 번의 이동' 명령이다. 서비스가 명령을 접수하면 바로 반환하고
실제 이동은 그 뒤에 진행되므로, 매 프레임 새 목표를 밀어넣으면 앞 모션이
안 끝나 거부된다("A motion is ongoing so new commands are not accepted." —
실측 2026-08-06 한 실행에 23회). servol 은 목표를 주기적으로 갱신하는
용도로 만들어진 명령이라 이 문제가 없다.

## 천천히 움직이게 하는 두 겹

속도 상한(TELEOP_VEL)만으로는 부족하다. 그건 "빨리 가지 마라"일 뿐,
손이 튀거나 인식이 한 프레임 흔들리면 **먼 목표**가 그대로 들어간다.
그래서 한 주기에 움직일 수 있는 거리(STEP_MAX)를 따로 막는다 —
이쪽이 실질적인 안전장치다.

## 작업영역

목표는 항상 상자 안으로 clamp 된다. 실측 2026-08-06 에 판 밖 오검출로
로봇이 (116, 413) 으로 간 적이 있는데, 조종 모드에서는 사람이 그렇게
보낼 수도 있다. Z_MIN 은 판보다 위에 둔다 — 판을 뚫는 명령은 아예 못 만든다.

환경변수로 전부 조정한다. 처음에는 기본값(느림)으로 시작해서 익숙해지면 올린다.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(HERE)

# ── 속도 ──────────────────────────────────────────────────────────
# 픽앤플레이스는 300mm/s 로 돈다. 조종은 사람이 보면서 따라가야 하므로
# 1/10 에서 시작한다. 익숙해지면 TELEOP_VEL 을 올린다.
TELEOP_VEL = float(os.environ.get("TELEOP_VEL", 15.0))    # mm/s
TELEOP_ACC = float(os.environ.get("TELEOP_ACC", 15.0))    # mm/s^2

# 한 주기에 목표가 움직일 수 있는 최대 거리(mm).
#
# **속도 상한과 함께 잡아야 한다.** 이 값 × 프레임률이 TELEOP_VEL 을 넘으면
# 목표점이 로봇보다 빨리 달아나고, 그 격차를 컨트롤러가 따라잡으려 하면서
# 지정 속도를 초과한다. 실측 2026-08-06: 5mm × 15fps = 75mm/s 가 vel 30mm/s 를
# 2.5배 앞질러, 손을 멈춰도 로봇이 계속 날아가 판에 부딪혔다.
#   기준:  STEP_MAX ≤ TELEOP_VEL / 프레임률(≈15) = 15/15 = 1.0
STEP_MAX = float(os.environ.get("TELEOP_STEP_MAX", 1.0))

# 목표점이 **실제 로봇 위치**에서 이보다 멀어지지 못하게 묶는다(밧줄).
# 위의 STEP_MAX 만으로는 프레임률이 흔들리면 다시 깨진다. 이 밧줄이 있으면
# 목표가 로봇을 구조적으로 앞지르지 못하고, 손을 멈추면 로봇도 이 거리 안에서
# 멈춘다 — 오버슛의 상한이 곧 이 값이 된다.
LEASH = float(os.environ.get("TELEOP_LEASH", 5.0))

# 실제 위치를 몇 프레임마다 읽을지. 매 프레임 읽으면 서비스 호출이 비싸다.
LEASH_EVERY = int(os.environ.get("TELEOP_LEASH_EVERY", 2))

# ── 작업영역 (base 좌표, mm) ─────────────────────────────────────
# 2026-08-06 교시값. teach_box.py 로 두 모서리를 찍어 얻었다.
# 이것이 **절대 경계**다 — 로봇의 실제 위치가 여기를 넘으면 안 된다.
X_MIN = float(os.environ.get("TELEOP_X_MIN", 199.2))
X_MAX = float(os.environ.get("TELEOP_X_MAX", 566.2))
Y_MIN = float(os.environ.get("TELEOP_Y_MIN", -231.3))
Y_MAX = float(os.environ.get("TELEOP_Y_MAX", 280.9))
Z_MIN = float(os.environ.get("TELEOP_Z_MIN", -13.8))
Z_MAX = float(os.environ.get("TELEOP_Z_MAX", 218.2))

# 경계를 이만큼 **안쪽으로 물려서** 목표를 자른다.
#
# 상자는 목표점에 걸리는데 로봇은 목표를 지나칠 수 있다(밧줄 LEASH 만큼).
# 실측 2026-08-06: 목표 z 를 20mm 로 막아뒀는데 로봇이 지나쳐 판에 부딪혔다.
# 여유를 안 두면 "경계까지" 가 "경계를 넘어서" 가 된다.
# LEASH 보다 크게 잡아 오버슛을 흡수한다 — 교시한 상자 안에 실제 팔이 머문다.
#
# 특히 Z_MIN 교시값(-13.8)은 판 표면이다. 여유 없이 쓰면 그대로 찍는다.
SAFETY_MARGIN = float(os.environ.get("TELEOP_MARGIN", 8.0))


def _inner(lo, hi, m=None):
    """경계를 안쪽으로 물린 (하한, 상한). 상자가 너무 좁으면 가운데로 붙인다."""
    m = SAFETY_MARGIN if m is None else m
    if hi - lo <= 2 * m:
        c = (lo + hi) / 2.0
        return c, c
    return lo + m, hi - m


# 목표를 자를 때 쓰는 값. 실제 팔은 여기에 LEASH 를 더한 범위 안에 머문다.
TX_MIN, TX_MAX = _inner(X_MIN, X_MAX)
TY_MIN, TY_MAX = _inner(Y_MIN, Y_MAX)
TZ_MIN, TZ_MAX = _inner(Z_MIN, Z_MAX)

# 손이 이만큼 연속으로 안 보이면 목표를 그 자리에 묶는다 (데드맨).
# 카메라가 가려지거나 사람이 자리를 뜨면 팔이 멈춰야 한다.
LOST_HOLD = int(os.environ.get("TELEOP_LOST_HOLD", 5))

# 로봇 연결에 이만큼 못 붙으면 조종을 끄고 화면만 돌린다.
# DSR 호출이 타임아웃 없이 멈추는 것을 여기서 막는다 (connect 주석 참고).
CONNECT_TIMEOUT = float(os.environ.get("TELEOP_CONNECT_TIMEOUT", 5.0))

# 걸려 있어야 하는 TCP. 작업모드(block_sort)와 같은 값을 써야 한다.
EXPECT_TCP = os.environ.get("EXPECT_TCP", "GripperDA_v1")

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP, TOOLCHARGER_PORT = "192.168.1.1", "502"
ROBOT_ID, ROBOT_MODEL = "dsr01", "m0609"


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


class RobotTeleop:
    """조종 값(화면 비율 0~1)을 로봇 좌표로 바꿔 servol 로 흘린다.

    쓰는 쪽은 매 프레임 update() 만 부르면 된다. 로봇 연결이 안 되면
    enabled=False 로 남아 아무 일도 하지 않는다 — 화면 동작은 그대로다.
    """

    def __init__(self):
        self.enabled = False
        self.target = None          # 현재 목표 [x, y, z, rx, ry, rz]
        self.att = None             # 자세는 진입 시점 것을 유지한다
        self.lost = 0
        self._grip_open = None      # 마지막으로 보낸 그리퍼 상태
        self._err = None
        self._last = None   # 직전 화면 좌표 (증분 계산용)
        self._tick = 0

    # ── 연결 ──
    def connect(self):
        """DSR 과 그리퍼에 붙는다. 실패해도 예외를 올리지 않는다.

        조종 화면은 로봇 없이도 의미가 있으므로(인식 확인), 연결 실패는
        기능을 끄는 것으로 끝낸다. 이유는 로그로 남긴다.
        """
        try:
            import rclpy
            import DR_init
            DR_init.__dsr__id = ROBOT_ID
            DR_init.__dsr__model = ROBOT_MODEL
            if not rclpy.ok():
                rclpy.init()
            node = rclpy.create_node("gesture_teleop", namespace=ROBOT_ID)
            # 모듈 수준에서 대입해야 한다. 클래스 안에서 쓰면 이름 맹글링으로
            # _RobotTeleop__dsr__node 가 되어 DSR_ROBOT2 가 None 을 읽는다.
            setattr(DR_init, "__dsr__node", node)
            from DSR_ROBOT2 import servol, get_current_posx, get_tcp, set_tcp
            self._servol = servol
            self._get_posx = get_current_posx
            self._node = node

            # TCP 가 풀려 있으면 좌표계가 달라져 상자 판정이 무의미해진다.
            # 조종도 같은 TCP 를 전제로 하므로 여기서도 확인한다.
            try:
                if get_tcp() != EXPECT_TCP:
                    print(f"  TCP 가 '{get_tcp()}' — '{EXPECT_TCP}' 로 다시 겁니다")
                    set_tcp(EXPECT_TCP)
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
            self.target = list(cur)
            self.target[0] = _clamp(cur[0], TX_MIN, TX_MAX)
            self.target[1] = _clamp(cur[1], TY_MIN, TY_MAX)
            self.target[2] = _clamp(cur[2], TZ_MIN, TZ_MAX)
            self.att = list(cur[3:])
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
        """조종 값을 받아 로봇 목표를 갱신하고 명령을 보낸다.

        pos_xy       화면 비율 (0~1, 0~1). hand_gesture_control 의 P.pos
        z_norm       0~1. P.z (0 이 아래, 1 이 위)
        gripper_open True 면 열기
        hand_seen    이번 프레임에 손이 보였는가 (데드맨)
        """
        if not self.enabled:
            return

        # 손이 사라지면 목표를 그대로 둔다 = 팔이 지금 자리에 머문다.
        if not hand_seen:
            self.lost += 1
            if self.lost >= LOST_HOLD:
                self._last = None        # 손이 다시 보일 때 튀지 않게 기준을 버린다
                return
        else:
            self.lost = 0

        # **증분으로 옮긴다.** 절대 매핑을 쓰면 안 된다 — 진입 시점의 팔 위치와
        # 화면 좌표가 가리키는 곳이 다르므로, 손이 데드존에 가만히 있어도 목표가
        # 그 차이만큼 계속 기어간다(실측 2026-08-06: 손을 안 움직여도 로봇이
        # 작업영역 한가운데로 5mm 씩 이동했다).
        # 화면 좌표의 **변화량**만 반영하면 손이 멈추면 로봇도 멈춘다.
        cur = (float(pos_xy[0]), float(pos_xy[1]), float(z_norm))
        if self._last is None:
            self._last = cur
            return
        du, dv = cur[0] - self._last[0], cur[1] - self._last[1]
        self._last = cur

        # xy 는 **증분**. 조이스틱은 "얼마나 움직일까" 를 주는 상대 입력이다.
        # z 는 **절대**. 게이지는 "어느 높이" 를 가리키는 절대 입력이라,
        # 게이지를 맨 아래로 내리면 z 도 최저까지, 맨 위면 최고까지 가야 한다.
        # 증분으로 다루면 게이지 위치와 실제 높이가 따로 놀아, 맨 아래로 내려도
        # 바닥에 안 닿는다(실측 2026-08-06).
        z_want = TZ_MIN + cur[2] * (TZ_MAX - TZ_MIN)
        d = np.array([-dv * (X_MAX - X_MIN),
                      -du * (Y_MAX - Y_MIN),
                      z_want - self.target[2]])

        # 한 주기 이동량 제한. 손이 튀어도 여기서 잘린다.
        n = float(np.linalg.norm(d))
        if n < 0.05:
            return                       # 사실상 정지 — 명령을 보내지 않는다
        if n > STEP_MAX:
            d = d * (STEP_MAX / n)
        want = np.array(self.target[:3]) + d
        self.target[:3] = [_clamp(want[0], TX_MIN, TX_MAX),
                           _clamp(want[1], TY_MIN, TY_MAX),
                           _clamp(want[2], TZ_MIN, TZ_MAX)]

        # ── 밧줄: 목표가 실제 로봇을 앞지르지 못하게 끌어당긴다 ──
        self._tick += 1
        if self._tick % LEASH_EVERY == 0:
            cur = self._read_posx()
            if cur is not None:
                gap = np.array(self.target[:3]) - np.array(cur[:3])
                g = float(np.linalg.norm(gap))
                if g > LEASH:
                    self.target[:3] = list(np.array(cur[:3]) + gap * (LEASH / g))

        # 마지막 방어선. 밧줄이 목표를 끌어당긴 뒤에도 절대 경계는 넘지 않는다.
        self.target[0] = _clamp(self.target[0], X_MIN, X_MAX)
        self.target[1] = _clamp(self.target[1], Y_MIN, Y_MAX)
        self.target[2] = _clamp(self.target[2], Z_MIN, Z_MAX)

        try:
            self._servol(list(self.target), vel=TELEOP_VEL, acc=TELEOP_ACC)
        except Exception as e:
            print(f"  servol 실패({e}) — 조종을 끕니다.")
            self.enabled = False
            return

        self._set_gripper(gripper_open)

    def _read_posx(self):
        """실제 로봇 위치. 못 읽으면 None — 밧줄만 건너뛰고 조종은 계속한다.

        get_current_posx 는 타임아웃이 없어 그냥 부르면 멈출 수 있다(connect 주석
        참고). 짧은 시간만 기다린다.
        """
        import threading
        box = {}

        def _r():
            try:
                box["p"] = self._get_posx()[0]
            except Exception:
                pass

        th = threading.Thread(target=_r, daemon=True)
        th.start()
        th.join(timeout=0.3)
        return box.get("p")

    def _map(self, pos_xy, z_norm):
        """화면 비율 → base 좌표. 항상 작업영역 안으로 clamp 한다."""
        u = _clamp(float(pos_xy[0]), 0.0, 1.0)
        v = _clamp(float(pos_xy[1]), 0.0, 1.0)
        zn = _clamp(float(z_norm), 0.0, 1.0)
        # 화면 오른쪽이 로봇 +y(왼쪽)가 되도록 뒤집는다 — 거울처럼 보게 해야
        # 조작이 직관적이다. 화면 위(v=0)가 로봇 앞쪽(+x)이다.
        x = X_MIN + (1.0 - v) * (X_MAX - X_MIN)
        y = Y_MAX - u * (Y_MAX - Y_MIN)
        z = Z_MIN + zn * (Z_MAX - Z_MIN)
        return [_clamp(x, TX_MIN, TX_MAX),
                _clamp(y, TY_MIN, TY_MAX),
                _clamp(z, TZ_MIN, TZ_MAX)]

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
        t = self.target
        return (f"ROBOT [{t[0]:.0f}, {t[1]:.0f}, {t[2]:.0f}]  "
                f"{TELEOP_VEL:.0f}mm/s")

    def close(self):
        self.enabled = False
        try:
            self._node.destroy_node()
        except Exception:
            pass
