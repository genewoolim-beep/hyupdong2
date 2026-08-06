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


def z_at(pos_z, t):
    """지금 z 에서 게이지를 그 높이로 맞춘 정규값 (z 속도 0 이 되게)."""
    return (pos_z - rt.TZ_MIN) / (rt.TZ_MAX - rt.TZ_MIN)


# ── 1. 중앙(데드존)이면 정지 ──
t, g, p = make()
zc = z_at(p[2], t)
t.update((0.0, 0.0), zc, True, True)
assert sent == [[0.0, 0.0, 0.0]], sent
print(f"1 중앙 → 정지            {sent[-1]}")

# ── 2. 방향만 보고 일정 속도. 얼마나 벗어났는지는 속도를 안 바꾼다 ──
t, g, p = make()
zc = z_at(p[2], t)
t.update((0.0, -0.2), zc, True, True)      # 화면 위 → 로봇 앞(+x)
a = sent[-1]
t.update((0.0, -1.0), zc, True, True)      # 다섯 배 벗어나도
b = sent[-1]
assert a == [rt.TELEOP_SPEED, 0.0, 0.0], a
assert b == a, (a, b)
# 매 프레임 다시 보낸다. 지령에 유효시간이 실려 있어 갱신을 멈추면 팔이 선다.
assert len(sent) == 2, sent
print(f"2 편차 0.2 와 1.0 이 같은 속도  {a}   (매 프레임 갱신: {len(sent)}건)")

# ── 3. 좌우는 거울 ──
t, g, p = make()
zc = z_at(p[2], t)
t.update((0.6, 0.0), zc, True, True)
assert sent[-1] == [0.0, -rt.TELEOP_SPEED, 0.0], sent[-1]
print(f"3 화면 오른쪽 → -y        {sent[-1]}")

# ── 4. 대각선은 두 축 동시 ──
t, g, p = make()
zc = z_at(p[2], t)
t.update((-0.5, -0.5), zc, True, True)
assert sent[-1] == [rt.TELEOP_SPEED, rt.TELEOP_SPEED, 0.0], sent[-1]
print(f"4 대각선 → 두 축          {sent[-1]}")

# ── 5. 손이 사라지면 정지 (데드맨) ──
t, g, p = make()
zc = z_at(p[2], t)
t.update((0.0, -0.5), zc, True, True)
for _ in range(rt.LOST_HOLD):
    t.update((0.0, -0.5), zc, True, False)
assert sent[-1] == [0.0, 0.0, 0.0], sent
print(f"5 손 사라짐 {rt.LOST_HOLD}프레임 → 정지  {sent[-1]}")

# ── 6. 경계: 밖으로 나가는 방향만 0, 안으로는 그대로 ──
t, g, p = make([rt.TX_MAX - 1.0, 0.0, 100.0, 0.0, 180.0, 0.0])
zc = z_at(p[2], t)
t.update((0.0, -0.5), zc, True, True)       # +x = 밖
assert sent[-1][0] == 0.0, sent[-1]
t.update((0.0, 0.5), zc, True, True)        # -x = 안
assert sent[-1][0] == -rt.TELEOP_SPEED, sent[-1]
print(f"6 경계 밖 방향 0 / 안 방향 유지   x={rt.TX_MAX:.0f} 근처")

# ── 7. z 게이지: 가리키는 높이로 가고 도달하면 선다 ──
t, g, p = make()
t.update((0.0, 0.0), 1.0, True, True)       # 게이지 최상 → 위로
assert sent[-1][2] == rt.TELEOP_SPEED, sent[-1]
t.update((0.0, 0.0), 0.0, True, True)       # 게이지 최하 → 아래로
assert sent[-1][2] == -rt.TELEOP_SPEED, sent[-1]
t.update((0.0, 0.0), z_at(p[2], t), True, True)   # 지금 높이 → 정지
assert sent[-1][2] == 0.0, sent[-1]
print(f"7 z 게이지 위/아래/도달정지   {rt.TZ_MIN:.0f}~{rt.TZ_MAX:.0f}mm")

# ── 8. 그리퍼는 바뀔 때만 보낸다 ──
t, g, p = make()
zc = z_at(p[2], t)
for want in (True, True, True, False, False, True):
    t.update((0.0, 0.0), zc, want, True)
assert g.log == ["open", "close", "open"], g.log
print(f"8 그리퍼 변화만 전송      {g.log}")

# ── 9. 나갈 때 반드시 속도 0 ──
t, g, p = make()
t.update((0.0, -0.5), z_at(p[2], t), True, True)
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
zc = z_at(p[2], t)
t.update((0.0, 0.5), zc, True, True)        # -x
t.update((0.6, 0.0), zc, True, True)        # -y
t.update((0.0, 0.0), zc, True, True)        # 정지
assert [-rt.TELEOP_SPEED, 0.0, 0.0] in sent, sent
assert [0.0, -rt.TELEOP_SPEED, 0.0] in sent, sent
assert sent[-1] == [0.0, 0.0, 0.0], sent
assert rt.TELEOP_CMD_SEC > 0
print(f"11 -x/-y/정지 전부 발행됨   유효시간 {rt.TELEOP_CMD_SEC}s")

print("\n전부 통과")
