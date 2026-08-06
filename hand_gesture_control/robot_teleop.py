#!/usr/bin/env python3
"""손동작 조종 값을 실제 로봇 움직임으로 옮긴다.

손을 읽어 화면에 그리는 것은 hand_gesture_control.py 가 한다. 이 모듈이 그
값을 로봇 **속도**로 바꿔 팔에 흘려보낸다.

## DSR 연결은 밖에서 받는다 (2026-08-06 이후)

한 로봇에 두 프로세스가 각자 붙으면 안 된다. 실측으로 TCP 조회가 0.0mm 로
깨지고, 모션이 전부 거부되고, get_current_posx() 가 응답 없이 멎었다.
그래서 지금은 **작업모드(block_sort)가 자기 연결을 그대로 넘겨준다.**

    RobotTeleop(dsr={"speedl":…, "get_posx":…}, gripper=gripper)   ← block_sort 안
    RobotTeleop()                                                  ← 단독 실행

인자가 없을 때만 스스로 붙는다(로봇만 있고 작업모드는 안 쓰는 PC용).
같은 프로세스에서 이미 DR_init.__dsr__node 가 잡혀 있으면 그것을 재사용한다 —
노드를 새로 만들면 DSR_ROBOT2 의 서비스 클라이언트가 옛 노드를 가리켜
두 번째 진입부터 명령이 안 나간다.

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

단, **유효시간을 0 으로 주면 안 된다.** DSR_ROBOT2 쪽 검증에 걸려 정지 지령과
-x/-y 지령이 예외로 거부된다 — 팔을 세우는 명령이 안 나간다는 뜻이다.
자세한 것은 TELEOP_CMD_SEC 주석. 회귀 테스트는 test_teleop.py 11번.

## 십자선 방식

방향만 본다. **얼마나 벗어났는지는 속도를 안 바꾼다.**
속도가 변하지 않아야 예측 가능하고, 그게 이 방식을 고른 이유다.

  중앙(데드존)  →  정지
  위/아래       →  ±x 로 TELEOP_SPEED
  좌/우         →  ±y 로 TELEOP_SPEED (거울)
  z 게이지      →  가리키는 높이로 일정 속도, 도달하면 정지

입력은 **지금 손이 십자선 중앙에서 벗어난 방향**이다. 누적된 목표점이 아니다 —
누적값을 쓰면 손을 중앙으로 되돌려도 그 값이 벗어난 채 남아 팔이 계속 기어간다
(실측 2026-08-06: 데드존 안에 손이 있는데도 움직였다). 속도 지령에는 적분값이
아니라 지금 방향만 필요하다.

## 경계

교시한 상자(teach_box.py) 밖으로 나가는 **방향의 속도만 0** 으로 만든다.
위치를 자르는 게 아니라 속도를 막는 것이라, 이번엔 실제 팔에 직접 걸린다.
정지에도 거리가 필요하므로 경계 BRAKE 안에 들어오면 미리 끊는다.

환경변수로 전부 조정한다.
"""
import os
import sys
import time

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

# 한 지령이 유효한 시간(초). 매 프레임 다시 보내 이 시간을 계속 갱신한다.
#
# **0 을 주면 안 된다.** DSR_ROBOT2.speedl 은 time=0 일 때
# _check_valid_vel_acc_task 로 vel[0]·vel[1] 이 양수인지 검사하고, 아니면
# DR_Error 를 올린다(그 검사는 movel 의 (직선속도, 회전속도) 쌍을 전제로 쓰여
# 있는데 speedl 의 6축 속도 벡터에도 그대로 걸린다). 그래서 time=0 으로는
#   · 정지 지령 [0,0,0,0,0,0]        → 예외. **팔을 세우는 명령이 안 나간다**
#   · -x 나 -y 로 가는 지령           → 예외
# 가 된다. 실제로 나갈 수 있는 것은 x,y 가 둘 다 양수인 경우뿐이었다.
#
# 양수 시간을 주면 그 검사를 건너간다. 겸사겸사 **컨트롤러 쪽 데드맨**이 된다 —
# 이 루프가 멎으면 팔도 이 시간 뒤에 스스로 선다. 20mm/s × 0.3초 = 6mm.
# 프레임 주기(15fps → 0.067초)보다 넉넉해야 지령이 끊겨 덜컥거리지 않는다.
TELEOP_CMD_SEC = float(os.environ.get("TELEOP_CMD_SEC", 0.3))

# 십자선 중앙의 정지 구역. 들어오는 벡터(-1~1, 0 이 중앙)의 축 성분이 이보다
# 작으면 그 축은 정지다. 인식 쪽에서 이미 원형 데드존을 뺀 값이 오므로
# (hand_gesture_control.HandController.process), 여기 값은 데드존 경계에서의
# 떨림을 막는 몫이다.
DEADZONE = float(os.environ.get("TELEOP_DEADZONE", 0.08))

# z 게이지가 가리키는 높이와 이만큼 안이면 z 정지. 게이지는 높이를 뜻하므로
# 도달하면 멈춰야 한다 — 방향만 보고 일정 속도로 가되, 다 오면 선다.
Z_TOL = float(os.environ.get("TELEOP_Z_TOL", 3.0))           # mm

# 경계에 이만큼 다가가면 그 축 속도를 0 으로 만든다. 정지에도 거리가 필요하다.
BRAKE = float(os.environ.get("TELEOP_BRAKE", 15.0))          # mm

# 실제 위치를 몇 프레임마다 읽을지. 경계 판정에만 쓴다.
#
# 이 값이 곧 **경계 판정이 늦는 시간**이다. 15fps 에 5프레임이면 0.33초,
# 20mm/s 로 6.7mm — BRAKE(15mm) 여유 안에 들어온다. 10 이었을 때는 13mm 로
# 여유를 거의 다 먹었다. 연결이 하나가 된 뒤로는 위치 조회가 다른 프로세스와
# 다투지 않으므로 더 자주 읽을 수 있다.
# 조회가 프레임률을 갉아먹는 것이 보이면 늘린다(경계 여유가 그만큼 준다).
POLL_EVERY = int(os.environ.get("TELEOP_POLL_EVERY", 5))

# ── 작업영역 (base 좌표, mm) ─────────────────────────────────────
# 이것이 **절대 경계**다 — 로봇의 실제 위치가 여기를 넘으면 안 된다.
# 출처는 셋이고 순서대로 이긴다:
#   ① 환경변수 TELEOP_X_MIN…      그때만 다르게 쓰겠다는 뜻
#   ② teleop_box.env             teach_box.py 가 마지막으로 교시한 값
#   ③ 아래 상수                   2026-08-06 교시값 (파일이 없을 때)
BOX_ENV_FILE = os.environ.get("TELEOP_BOX_ENV", os.path.join(HERE, "teleop_box.env"))


def _load_box_file(path):
    """교시 파일의 TELEOP_* 를 환경변수 기본값으로 깐다.

    교시값이 코드 상수로만 있으면 다시 교시할 때마다 사람이 코드나 명령줄을
    손대야 하고, 그러면 파일에 남은 값과 실제로 쓰이는 값이 갈라진다 —
    조종 경계는 갈라져 있으면 안 되는 값이다(팔이 판에 부딪히는 쪽이다).
    이미 환경변수로 준 것은 건드리지 않는다.
    """
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return {}
    got = {}
    for tok in text.replace("\n", " ").split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k.startswith("TELEOP_") and k not in os.environ:
            os.environ[k] = v
            got[k] = v
    return got


BOX_FROM_FILE = _load_box_file(BOX_ENV_FILE)

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
    """DSR 함수들을 한 번만 붙여 캐시한다. **단독 실행일 때만 쓴다.**

    이미 같은 프로세스에 노드가 잡혀 있으면(작업모드가 만든 것) 그것을 쓴다.
    노드를 또 만들면 DSR_ROBOT2 의 서비스 클라이언트가 어느 노드를 가리키는지가
    바뀌어, 먼저 붙어 있던 쪽 명령이 안 나간다.
    """
    if _DSR:
        return _DSR
    import rclpy
    import DR_init
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    if not rclpy.ok():
        rclpy.init()
    node = getattr(DR_init, "__dsr__node", None)
    if node is None:
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

    dsr      {"speedl":…, "get_posx":…} (선택) — 이미 DSR 에 붙어 있는 쪽이
             자기 함수를 넘긴다. 없으면 스스로 붙는다(단독 실행).
    gripper  onrobot.RG (선택) — 이미 열어둔 모드버스 연결을 그대로 쓴다.
             하나의 그리퍼에 두 연결을 만들지 않으려는 것이다.
    """

    def __init__(self, dsr=None, gripper=None):
        self._dsr = dsr
        self._gripper = gripper
        self.enabled = False
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
            d = self._dsr or _dsr_connect()
            self._speedl = d["speedl"]
            self._get_posx = d["get_posx"]

            # TCP 가 풀려 있으면 좌표계가 달라져 상자 판정이 무의미해진다.
            # 넘겨받은 연결이면 그쪽(block_sort.ensure_tcp)이 이미 확인했다.
            if self._dsr is None:
                try:
                    if d["get_tcp"]() != EXPECT_TCP:
                        print(f"  TCP 가 '{d['get_tcp']()}' — '{EXPECT_TCP}' 로 다시 겁니다")
                        d["set_tcp"](EXPECT_TCP)
                except Exception as e:
                    print(f"  TCP 확인 실패({e})")

            if self._gripper is None:
                sys.path.insert(0, os.path.join(_ROOT, "block_sort"))
                from onrobot import RG
                self._gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

            # 경계 판정에 쓸 **지금 위치**를 읽는다. 못 읽으면 조종을 켜지 않는다 —
            # 위치를 모르면 상자 밖으로 나가는지 알 수 없기 때문이다.
            # 자세(rx,ry,rz)는 따로 들고 있지 않는다. 회전 속도를 항상 0 으로
            # 주므로 진입 시점 자세가 그대로 유지된다.
            cur = self._read_startpos()
            self._pos_cache = cur
            self.enabled = True
            print(f"  로봇 조종 연결됨 — 시작 위치 "
                  f"[{cur[0]:.0f}, {cur[1]:.0f}, {cur[2]:.0f}]  "
                  f"속도 {TELEOP_SPEED:.0f}mm/s (축당 고정)  "
                  f"데드존 {DEADZONE:.2f}")
            src = os.path.basename(BOX_ENV_FILE) if BOX_FROM_FILE else "코드 기본값"
            print(f"  작업영역(교시, {src})  x {X_MIN:.0f}~{X_MAX:.0f}  "
                  f"y {Y_MIN:.0f}~{Y_MAX:.0f}  z {Z_MIN:.0f}~{Z_MAX:.0f}")
            print(f"  목표 한계(-{SAFETY_MARGIN:.0f}mm)  x {TX_MIN:.0f}~{TX_MAX:.0f}  "
                  f"y {TY_MIN:.0f}~{TY_MAX:.0f}  z {TZ_MIN:.0f}~{TZ_MAX:.0f}")
            return True
        except Exception as e:
            self._err = str(e)
            print(f"  로봇 조종을 켜지 못했습니다({e}) — 화면만 동작합니다.")
            return False

    def _read_startpos(self):
        """진입 시점의 위치. 못 읽으면 예외를 올린다(조종을 켜지 않는다).

        DSR 의 get_current_posx 는 **타임아웃이 없다**(spin_until_future_complete
        를 인자 없이 부른다). 다른 프로세스가 DSR 을 잡고 있으면 응답이 안 와
        여기서 영원히 멈춘다 — 실측 2026-08-06: 제어모드로 들어가면 멈췄다가
        block_sort 쪽 타임아웃으로만 빠져나왔다. 연결을 하나로 합쳤으니 그
        상황은 없어야 하지만, 확인하는 값이 싸므로 그대로 둔다.
        별도 스레드에서 부르고 시간을 재서, 안 오면 조종만 끈다 — 카메라와
        화면은 계속 동작해야 한다(로봇 없이도 인식 확인은 된다).
        """
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
            why = box.get("e")
            raise RuntimeError(
                f"로봇 위치를 못 읽었습니다({why})" if why else
                f"로봇 위치를 {CONNECT_TIMEOUT:.0f}초 안에 못 읽었습니다 "
                "— 다른 프로세스가 DSR 을 잡고 있을 수 있습니다")
        return box["p"]

    # ── 매 프레임 ──
    def update(self, vec, z_norm, gripper_open, hand_seen):
        """조종 값을 **속도**로 바꿔 로봇에 보낸다.

        vec          지금 손이 십자선 중앙에서 벗어난 방향 (-1~1, 0 이 중앙).
                     누적된 목표점이 아니다 — 이유는 모듈 docstring 참고.
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
        u = float(vec[0])
        v = float(vec[1])

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
        """속도 지령. 회전은 항상 0 — 조종은 자세를 안 바꾼다.

        **매번 보낸다.** 같은 지령이라고 건너뛰면 안 된다 — 지령에 유효시간
        (TELEOP_CMD_SEC)을 실어 보내므로, 갱신을 멈추면 팔이 그 시간 뒤에 선다.
        그게 컨트롤러 쪽 데드맨이고, 대신 계속 보내야 계속 움직인다.
        speedl 은 서비스가 아니라 토픽(speedl_stream) 발행이라 매 프레임 보내도
        큐가 밀리지 않는다 — 스트리밍이 원래 쓰임새다.
        """
        self._last_cmd = list(xyz)
        try:
            self._speedl([xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0],
                         [TELEOP_ACC, TELEOP_ACC], TELEOP_CMD_SEC)
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
        # 나가기 전에 반드시 세운다. 여러 번 보내는 이유: 토픽 발행이라 마지막
        # 한 장이 유실되면 팔이 마지막 속도로 남는다. 세 번이면 값이 싸다.
        for _ in range(3):
            try:
                self._send([0.0, 0.0, 0.0])
            except Exception:
                break
            time.sleep(0.02)
        self.enabled = False
        self._last_cmd = None
