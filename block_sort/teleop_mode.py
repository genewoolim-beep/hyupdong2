#!/usr/bin/env python3
"""제어모드(손동작 조종)를 작업모드와 **같은 프로세스**에서 돈다.

## 왜 합쳤나

전에는 block_sort(작업모드)와 hand_gesture_control(제어모드)이 각자 DSR 에
붙었다. 한 로봇에 두 연결이라 실측 2026-08-06 에 이렇게 나타났다.

    TCP 조회가 0.0mm 로 깨짐          → 그대로 파지하면 좌표가 250mm 어긋난다
    모션 명령이 전부 거부됨            → STANDBY 20초 경고
    get_current_posx() 무한 대기       → 제어모드 진입 시 화면만 뜨고 멈춤

타임아웃·순서·생존확인으로 우회했지만 전부 증상 차단이었다. 제어모드를 이
프로세스 안으로 들여오면 연결이 하나가 되어 셋이 함께 사라진다.

## 무엇을 공유하는가

    DSR 연결   block_sort 가 만든 노드와 함수(speedl, get_current_posx)
    그리퍼     block_sort 의 모드버스 연결 (RG 를 또 열지 않는다)
    웹캠       수어 인식기가 이미 열어둔 그 카메라 한 장
               (V4L2 는 한 프로세스만 열 수 있고, 토픽이라도 두 번 열 이유가 없다)

인식 로직은 hand_gesture_control.py 에서 그대로 가져온다. 여기에 복사하면
두 벌이 되어 한쪽만 고치는 사고가 난다.

## 돌아가는 방법 셋

    ① 양손 Open_Palm 3초   ② 조종창에서 Q   ③ 대시보드에서 작업모드로 전환
"""
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(HERE)
GESTURE_DIR = os.environ.get("GESTURE_DIR", os.path.join(_ROOT, "hand_gesture_control"))

WIN = "gesture control"

# 프레임이 이만큼 연속으로 안 오면 조종을 접고 작업모드로 돌아간다.
# 웹캠 발행이 죽었는데 화면만 켜둔 채 남아 있으면, 로봇은 마지막 지령대로
# 움직이는 중일 수 있다 — 조종 쪽이 살아 있다고 믿을 근거가 없으면 나간다.
# 데드맨(손 사라짐 → 속도 0)이 먼저 걸리지만, 그 판단도 프레임이 있어야 한다.
BLANK_LIMIT = int(os.environ.get("TELEOP_BLANK_LIMIT", 150))


def _load():
    """손 인식(hand_gesture_control)과 속도 지령(robot_teleop) 모듈.

    ROS 패키지가 아니라 스크립트 폴더라 경로만 붙이면 import 된다.
    hand_gesture_control 은 import 만으로는 카메라도 로봇도 건드리지 않는다
    (모듈 수준에서 sys.argv 를 읽어 상수만 정한다).
    """
    if GESTURE_DIR not in sys.path:
        sys.path.insert(0, GESTURE_DIR)
    import hand_gesture_control as hg
    import robot_teleop as rt
    return hg, rt


def run(cap, dsr, gripper, on_frame=None, should_stop=None,
        max_sec=600.0, log=print):
    """제어모드 한 판. 끝난 이유를 문자열로 돌려준다.

    cap         프레임 공급자 (read() -> (ok, frame)). 수어 인식기가 쓰는 것 그대로.
    dsr         {"speedl":…, "get_posx":…} — 이 프로세스의 DSR 함수
    gripper     onrobot.RG — 이미 열려 있는 연결
    on_frame    화면 한 장을 받는 콜백 (대시보드 중계용). None 이면 안 보낸다.
    should_stop 매 프레임 물어보는 콜백. True 면 나간다 (대시보드에서 전환).
                **막히지 않아야 한다.** 여기서 네트워크를 기다리면 그 시간만큼
                경계 판정이 멈추고, 속도 지령은 그동안에도 팔을 밀고 있다.
    max_sec     이 시간을 넘으면 스스로 나간다. 사람이 안 보는 채로 속도 지령을
                받는 상태로 남지 않게 하는 상한이다.
    """
    hg, rt = _load()

    teleop = rt.RobotTeleop(dsr=dsr, gripper=gripper)
    if not teleop.connect():
        # 여기서 실패하면 팔을 움직일 수 없다. 화면만 띄워두면 사람은 조종이
        # 되는 줄 알고 손을 흔든다 — 그냥 작업모드로 돌려보낸다.
        return f"로봇 조종 연결 실패: {teleop.status()}"

    P = hg.HandController()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 540)
    log(f"제어모드 — 왼손 십자선으로 xy, 오른손 게이지로 z, 주먹/펼침으로 그리퍼. "
        f"돌아가기: 양손 {hg.RETURN_HOLD_SEC:.0f}초 펴기 / Q / 대시보드")

    t0 = t_prev = time.time()
    fps, blank = 0.0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                blank += 1
                if blank >= BLANK_LIMIT:
                    return "웹캠 프레임이 끊겼습니다"
                # 프레임이 없어도 손은 안 보이는 것이다 — 데드맨을 먹인다.
                teleop.update((0.0, 0.0), P.z, P.gripper > 0.5, False)
                time.sleep(0.02)
                continue
            blank = 0
            frame = cv2.flip(frame, 1)

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            xy_hand, gesture_hand = P.process(frame, dt)
            # **지금 벗어난 방향**을 준다. 누적 목표점(P.pos)이 아니다 —
            # 누적값은 손을 중앙으로 되돌려도 남아 팔이 계속 기어간다.
            vec = xy_hand[0] if xy_hand is not None else (0.0, 0.0)
            seen = xy_hand is not None or gesture_hand is not None
            teleop.update(vec, P.z, P.gripper > 0.5, seen)

            # 입력(십자선·3등분 게이지)과 상태(로봇 실제 z)를 나란히 둔다.
            # 단독 실행에서는 상태가 screen 창에 따로 있었지만 여기는 창이
            # 하나뿐이다 — 조종하면서 실제 높이를 못 보면 판에 얼마나 가까운지
            # 알 방법이 없다.
            hg.draw_roi(frame)
            hg.draw_gauge(frame, P.z)
            hg.draw_z_gauge(frame, teleop, (rt.Z_MIN, rt.Z_MAX))
            if xy_hand is not None:
                hg.draw_hand(frame, xy_hand[1], highlight=hg.INDEX_TIP)
            if gesture_hand is not None:
                hg.draw_hand(frame, gesture_hand, highlight=hg.PALM_CENTER)
            hg.draw_return_hold(frame, P.both_open_hold)
            _overlay(frame, P, teleop, fps, hg)

            if on_frame is not None:
                on_frame(frame)
            cv2.imshow(WIN, frame)

            if P.return_triggered():
                return f"양손 {hg.RETURN_HOLD_SEC:.0f}초 펴기"
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                return "조종창 Q"
            if k == ord("r"):
                P.z, P.gripper = hg.Z0, hg.GRIPPER_OPEN
                P.z_hist.clear()

            if should_stop is not None and should_stop():
                return "대시보드에서 작업모드로 전환"
            if now - t0 > max_sec:
                return f"제어모드 상한 {max_sec:.0f}초 초과"
    finally:
        # 나가는 길에 **반드시 속도 0** 을 보낸다. 속도 지령은 명령을 끊는 것으로
        # 멈추지 않는다 — 마지막 지령이 그대로 남는다.
        teleop.close()
        try:
            cv2.destroyWindow(WIN)
        except Exception:
            pass
        cv2.waitKey(1)      # 창이 실제로 닫히도록 한 번 돌린다


def _overlay(frame, P, teleop, fps, hg):
    h = frame.shape[0]
    cv2.putText(frame, teleop.status(), (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if teleop.enabled else (0, 0, 255), 2, cv2.LINE_AA)
    # z 는 값이 아니라 구간(UP/STOP/DOWN)으로 보여준다 — 0~1 숫자는 절대 높이로
    # 읽히기 쉽지만 실제로는 방향 신호다. 구간 판정은 화면·로봇이 같은 것을 쓴다.
    cv2.putText(frame, f"gesture={P.gesture}({P.gesture_score:.2f})  "
                       f"z={hg.z_zone(P.z)}  "
                       f"gripper={'OPEN' if P.gripper > 0.5 else 'CLOSED'}",
                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Q back to work mode   R reset z/gripper",
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1,
                cv2.LINE_AA)
    # fps 를 오른쪽 위에 두면 draw_z_gauge 의 "Z ..mm" 글자와 겹친다.
    cv2.putText(frame, f"{fps:.0f} fps", (12, 82), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (120, 255, 120), 2, cv2.LINE_AA)
