#!/usr/bin/env python3
"""조종의 속도 지령을 로봇 없이 검증한다.

    python3 test_teleop.py        # 로봇도 ROS 도 필요 없다

가짜 DSR 을 주입해 speedl 로 **무엇이 나가는지** 본다. 조종은 실기에서만 확인할
수 있는 부분이 많지만, 여기서 잡히는 것은 실기에서 잡으면 팔이 부딪힌다.

특히 11번을 지우지 말 것 — DSR_ROBOT2.speedl 은 time=0 일 때 vel[0]·vel[1] 이
양수인지 검사하고 아니면 예외를 올린다(movel 의 (직선,회전) 쌍을 전제로 쓰인 검사가
speedl 의 6축 속도 벡터에도 그대로 걸린다). 그래서 time=0 으로는 **정지 지령과
-x/-y 지령이 아예 안 나간다.** 아래 가짜 speedl 이 그 검사를 흉내내고 있다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import robot_teleop as rt

sent = []
POS = [400.0, 0.0, 100.0, 0.0, 180.0, 0.0]        # 상자 한가운데


class FakeGrip:
    def __init__(self):
        self.log = []

    def open_gripper(self):
        self.log.append("open")

    def close_gripper(self):
        self.log.append("close")


def make(pos=None):
    sent.clear()
    p = list(pos or POS)
    def _speedl(v, a, t):
        # 벤더 검증(_check_valid_vel_acc_task)을 그대로 흉내낸다. time=0 이면
        # 정지 지령([0,0,0,...])과 -x/-y 지령이 예외로 거부된다 — 그것이
        # TELEOP_CMD_SEC 를 양수로 두는 이유다.
        if float(t) == 0.0 or float(t) == -10000.0:
            if v[0] <= 0 or (v[1] != -10000 and v[1] <= 0):
                raise RuntimeError("DR_Error: Invalid value : vel1 (0.0, when time = 0.0)")
        sent.append([round(x, 1) for x in v[:3]])

    dsr = {"speedl": _speedl, "get_posx": lambda: (p, 0)}
    g = FakeGrip()
    t = rt.RobotTeleop(dsr=dsr, gripper=g)
    assert t.connect(), "주입된 연결로 connect 실패"
    return t, g, p


# z 게이지도 xy 와 같은 방향 신호다. 0.5 가 중앙(정지), 위/아래로
# Z_DEADZONE(1/6) 넘게 벗어나면 그 방향. **절대 높이가 아니다** — 로봇의
# 지금 z 와 무관하다(2026-08-06 팀원 변경: 진입 시점 영점 조절이 필요 없어짐).
Z_STOP = 0.5


# ── 1. 중앙(데드존)이면 정지 ──
t, g, p = make()
zc = Z_STOP
t.update((0.0, 0.0), zc, True, True)
assert sent == [[0.0, 0.0, 0.0]], sent
print(f"1 중앙 → 정지            {sent[-1]}")

# ── 2. 방향만 보고 일정 속도. 얼마나 벗어났는지는 속도를 안 바꾼다 ──
t, g, p = make()
zc = Z_STOP
t.update((0.0, -0.2), zc, True, True)      # 화면 위 → 로봇 앞(+x)
a = sent[-1]
t.update((0.0, -1.0), zc, True, True)      # 다섯 배 벗어나도
b = sent[-1]
assert a == [rt.TELEOP_SPEED, 0.0, 0.0], a
assert b == a, (a, b)
# 매 프레임 다시 보낸다. 지령에 유효시간이 실려 있어 갱신을 멈추면 팔이 선다.
assert len(sent) == 2, sent
print(f"2 편차 0.2 와 1.0 이 같은 속도  {a}   (매 프레임 갱신: {len(sent)}건)")

# ── 3. 좌우는 선 자리에 달렸다 (Y_SIGN) ──
#     기본은 '로봇을 마주보고 조종' 이라 내 오른쪽이 로봇의 왼쪽(+y)이다.
#     화면 글자(draw_roi)도 이 부호에서 뽑으므로 둘이 갈라질 수 없다.
t, g, p = make()
zc = Z_STOP
t.update((0.6, 0.0), zc, True, True)          # 화면 오른쪽 = 내 오른쪽
want_y = rt.Y_SIGN * -rt.TELEOP_SPEED
assert sent[-1] == [0.0, want_y, 0.0], sent[-1]
t.update((-0.6, 0.0), zc, True, True)         # 왼쪽은 반대 부호
assert sent[-1] == [0.0, -want_y, 0.0], sent[-1]
assert rt.Y_SIGN in (1.0, -1.0)
print(f"3 화면 오른쪽 → y={want_y:+.0f}   "
      f"(마주보고 조종: {'예' if rt.FACE_ROBOT else '아니오'})")

# ── 4. 대각선은 두 축 동시 ──
t, g, p = make()
zc = Z_STOP
t.update((-0.5, -0.5), zc, True, True)        # 화면 왼쪽 위 = 앞 + 내 왼쪽
assert sent[-1] == [rt.TELEOP_SPEED, -want_y, 0.0], sent[-1]
print(f"4 대각선 → 두 축          {sent[-1]}")

# ── 5. 손이 사라지면 정지 (데드맨) ──
t, g, p = make()
zc = Z_STOP
t.update((0.0, -0.5), zc, True, True)
for _ in range(rt.LOST_HOLD):
    t.update((0.0, -0.5), zc, True, False)
assert sent[-1] == [0.0, 0.0, 0.0], sent
print(f"5 손 사라짐 {rt.LOST_HOLD}프레임 → 정지  {sent[-1]}")

# ── 6. 경계: 밖으로 나가는 방향만 0, 안으로는 그대로 ──
t, g, p = make([rt.TX_MAX - 1.0, 0.0, 100.0, 0.0, 180.0, 0.0])
zc = Z_STOP
t.update((0.0, -0.5), zc, True, True)       # +x = 밖
assert sent[-1][0] == 0.0, sent[-1]
t.update((0.0, 0.5), zc, True, True)        # -x = 안
assert sent[-1][0] == -rt.TELEOP_SPEED, sent[-1]
print(f"6 경계 밖 방향 0 / 안 방향 유지   x={rt.TX_MAX:.0f} 근처")

# ── 7. z 게이지도 십자선. 가운데 정지, 위/아래로 일정 속도 ──
#     로봇의 지금 z 와 무관해야 한다 — 그게 절대 매핑을 버린 이유다.
#     시작 높이는 경계 브레이크 구간(상하 15mm) 밖으로 고른다. 그 안에서는
#     경계 쪽 방향이 0 이 되는 게 정상이다(6번에서 따로 확인한다).
for start_z in (50.0, 100.0, 150.0):
    t, g, p = make([400.0, 0.0, start_z, 0.0, 180.0, 0.0])
    t.update((0.0, 0.0), 1.0, True, True)              # 게이지 최상 → 위로
    assert sent[-1][2] == rt.TELEOP_SPEED, (start_z, sent[-1])
    t.update((0.0, 0.0), 0.0, True, True)              # 최하 → 아래로
    assert sent[-1][2] == -rt.TELEOP_SPEED, (start_z, sent[-1])
    t.update((0.0, 0.0), Z_STOP, True, True)           # 가운데 → 정지
    assert sent[-1][2] == 0.0, (start_z, sent[-1])
# 데드존 경계 바로 안쪽은 정지여야 한다 (높이를 유지하려는 조작)
t, g, p = make()
t.update((0.0, 0.0), 0.5 + rt.Z_DEADZONE * 0.9, True, True)
assert sent[-1][2] == 0.0, sent[-1]
print(f"7 z 십자선 위/정지/아래       데드존 ±{rt.Z_DEADZONE:.3f} "
      f"(시작 높이 50/100/150mm 에서 같은 결과)")

# ── 7b. z 도 경계에서 그 방향만 막힌다 ──
t, g, p = make([400.0, 0.0, rt.TZ_MIN + 1.0, 0.0, 180.0, 0.0])   # 바닥 근처
t.update((0.0, 0.0), 0.0, True, True)        # 아래로 = 밖
assert sent[-1][2] == 0.0, sent[-1]
t.update((0.0, 0.0), 1.0, True, True)        # 위로 = 안
assert sent[-1][2] == rt.TELEOP_SPEED, sent[-1]
print(f"7b 바닥({rt.TZ_MIN:.0f}mm) 근처 → 하강만 차단")

# ── 8. 그리퍼는 바뀔 때만 보낸다 ──
t, g, p = make()
zc = Z_STOP
for want in (True, True, True, False, False, True):
    t.update((0.0, 0.0), zc, want, True)
assert g.log == ["open", "close", "open"], g.log
print(f"8 그리퍼 변화만 전송      {g.log}")

# ── 9. 나갈 때 반드시 속도 0 ──
t, g, p = make()
t.update((0.0, -0.5), Z_STOP, True, True)
t.close()
assert sent[-1] == [0.0, 0.0, 0.0], sent
assert not t.enabled
print(f"9 close() → 속도 0        {sent[-1]}")

# ── 10. 위치를 못 읽으면 안 움직인다 ──
sent.clear()
def boom():
    raise RuntimeError("no posx")
t = rt.RobotTeleop(dsr={"speedl": lambda v, a, tt: sent.append(list(v[:3])),
                        "get_posx": boom}, gripper=FakeGrip())
assert not t.connect(), "위치를 못 읽는데 조종이 켜졌다"
t.update((0.0, -1.0), 0.5, True, True)
assert sent == [], sent
print("10 위치 불명 → 조종 안 켜짐 (지령 0건)")

# ── 11. 정지·역방향 지령이 실제로 나간다 (벤더 검증 통과) ──
t, g, p = make()
zc = Z_STOP
t.update((0.0, 0.5), zc, True, True)        # -x
t.update((0.6, 0.0), zc, True, True)        # y 한쪽
t.update((-0.6, 0.0), zc, True, True)       # y 반대쪽
t.update((0.0, 0.0), zc, True, True)        # 정지
assert [-rt.TELEOP_SPEED, 0.0, 0.0] in sent, sent
assert [0.0, want_y, 0.0] in sent, sent
assert [0.0, -want_y, 0.0] in sent, sent    # 음수 쪽도 실제로 나가야 한다
assert sent[-1] == [0.0, 0.0, 0.0], sent
assert rt.TELEOP_CMD_SEC > 0
print(f"11 -x/±y/정지 전부 발행됨   유효시간 {rt.TELEOP_CMD_SEC}s")

print("\n전부 통과")
