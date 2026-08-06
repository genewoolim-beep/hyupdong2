"""양손 위치/모양으로 로봇팔 xy/z/그리퍼를 조작하는 데모.

왼손: xy 제어. 조이스틱 방식이다 — 절대 위치가 아니라 원 중심에서 벗어난
  방향/양이 신호가 된다. 화면 왼쪽에 원 두 개(ROI_CX/CY, ROI_R)가 있다.
    안쪽 원(반지름 ROI_R의 절반, 데드존) 안에서 검지 끝이 움직이면 "움직임 없음".
    (손목이 아니라 검지 끝 기준이다 — 손목을 꺾지 않고 손가락만 뻗어도 조작된다.)
    바깥 원 밖으로 나가면 그 방향으로, 데드존 경계~바깥 원 사이 거리에 비례한
    크기로 움직인다(0~1로 클램프).
  손 떨림이 그대로 좌표 떨림이 되던 절대 위치 매핑의 문제, 그리고 로봇 가동
  범위가 카메라 앞 특정 구역에 묶이던 문제를 함께 없앤다 — 데드존 밖에서 계속
  버티고 있으면 그 방향으로 계속 움직이면 되므로 화면 밖 범위까지 갈 필요가 없다.

오른손: z 축과 그리퍼 제어. xy 를 담당하는 손과 물리적으로 분리했다 — z/그리퍼
  조작 중 손이 흔들려도 xy 좌표가 왜곡되지 않는다.
  z: 화면 오른쪽의 세로 게이지(GAUGE_CX, GAUGE_Y0~Y1) 안에서 손바닥 높이의
     절대 위치로 직접 정한다. 게이지 위쪽일수록 z가 높고 아래쪽일수록 낮다 —
     엄지를 세워 누르고 있는 것보다 직관적이고 정밀하다.
  그리퍼: MediaPipe GestureRecognizer 의 손 모양 실시간 분류로 연다/닫는다.
  Open_Palm  : 그리퍼 열기
  Closed_Fist: 그리퍼 닫기

주의(핸드니스 반전): 프레임을 cv2.flip 으로 좌우 반전한 뒤 MediaPipe 에 넣으면
  handedness 라벨이 실사용자 기준과 반대로 나온다 (실측 확인됨). 그래서 코드는
  label=="Right" 를 실제 왼손으로, label=="Left" 를 실제 오른손으로 간주해서
  처리한다 (HandController.process 참고). 화면 표시/주석/UI 텍스트는 전부
  "실제 사용자 기준" 왼손/오른손으로 통일한다.

키
  Q 종료          S 시간평균 on/off
  [ ] 감도 조절    SPACE 목표점을 화면 중앙으로 리셋
  R 초기화        F 전체화면
"""

import json
import os
import sys
import threading
import urllib.request
import cv2
import time
import numpy as np
from collections import deque
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (GestureRecognizer, GestureRecognizerOptions,
                                           RunningMode)

CAM_INDEX = int(os.environ.get("GESTURE_CAM", 1))
# 스크립트와 같은 폴더의 모델을 기본으로 쓴다 — 다른 곳에 둔 경우에만 환경변수로 덮어쓴다.
HERE = os.path.dirname(os.path.abspath(__file__))
HAND_TASK = os.environ.get("GESTURE_TASK", os.path.join(HERE, "gesture_recognizer.task"))

# 프레임을 어디서 받을지 (sign_demo.py 와 같은 패턴).
#   v4l2  /dev/videoN 을 직접 연다 (기본). 한 프로세스만 열 수 있다 — sign_demo.py 등
#         다른 프로그램이 이미 v4l2 로 웹캠을 잡고 있으면 여기서 열기가 실패한다.
#   ros   /webcam/image_raw 토픽을 구독한다. 구독자 수 제한이 없어서 sign_demo.py 와
#         동시에 같은 웹캠을 볼 수 있다. 먼저 발행 노드를 띄워야 한다:
#           ros2 run sign_processing webcam_publisher
GESTURE_SOURCE = os.environ.get("GESTURE_SOURCE", "v4l2").lower()
GESTURE_CAM_TOPIC = os.environ.get("GESTURE_CAM_TOPIC", "/webcam/image_raw")

# ───────────────────────── signbot_admin 연동 ─────────────────────────
# --admin 을 붙였을 때만 대시보드로 모드 상태를 보낸다. sign_demo.py/block_sort.py 와
# 같은 패턴(ADMIN_URL, "--admin" in sys.argv) 이다.
ADMIN_URL = os.environ.get("SIGN_ADMIN_URL", "http://localhost:5000")
USE_ADMIN = "--admin" in sys.argv
# --robot 을 붙여야 실제 로봇이 움직인다. 기본은 화면만 — 손동작 인식을
# 먼저 눈으로 확인하고 나서 팔을 붙이는 순서가 안전하다.
# 속도·작업영역은 robot_teleop.py 의 환경변수로 조정한다.
USE_ROBOT = "--robot" in sys.argv


def push_mode(mode):
    """signbot_admin의 /api/mode 로 인터페이스 모드(work/control)를 보고한다."""
    if not USE_ADMIN:
        return
    def _send():
        try:
            body = json.dumps({"mode": mode}).encode("utf-8")
            req = urllib.request.Request(
                f"{ADMIN_URL}/api/mode", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=1.0)
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()


# 제어 화면(screen 창)을 signbot_admin으로 중계한다. sign_demo.py 와 같은 패턴이지만
# 별도 엔드포인트(/api/frame/control)를 쓴다 — 작업모드 카메라(/api/frame)와
# 같은 버퍼를 쓰면 두 화면이 서로 프레임을 덮어써 버리기 때문이다.
_frame_lock = threading.Lock()
_frame_holder = {"jpg": None}


def set_latest_frame(frame_bgr):
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if ok:
        with _frame_lock:
            _frame_holder["jpg"] = buf.tobytes()


def _frame_sender_loop():
    while True:
        with _frame_lock:
            data = _frame_holder["jpg"]
        if data is not None:
            try:
                req = urllib.request.Request(
                    f"{ADMIN_URL}/api/frame/control", data=data,
                    headers={"Content-Type": "image/jpeg"}, method="POST")
                urllib.request.urlopen(req, timeout=1.0)
            except Exception:
                pass
        time.sleep(0.08)   # 모니터링용이라 12fps 상한이면 충분하다


_sender_started = False


def start_frame_sender():
    global _sender_started
    if _sender_started:      # 세션이 반복돼도(대기 -> 제어 -> 대기 -> ...) 스레드는 한 번만
        return
    _sender_started = True
    threading.Thread(target=_frame_sender_loop, daemon=True).start()


def ping_alive():
    """이 프로세스가 살아 있다고 대시보드에 알린다.

    block_sort 는 '모드변경' 을 받으면 여기로 제어권을 넘기고 대기에 들어간다.
    그런데 이 프로세스가 없으면 아무도 work 로 되돌려주지 않아 작업모드가
    멈춘다(2026-08-06 실측). 그래서 block_sort 가 넘기기 전에 이 신호를 본다.
    대기 중에도 보내야 한다 — 대기 중인 것도 '받아줄 준비가 됐다' 는 뜻이다.
    """
    if not USE_ADMIN:
        return
    try:
        req = urllib.request.Request(f"{ADMIN_URL}/api/control/alive",
                                     data=b"{}",
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        urllib.request.urlopen(req, timeout=1.0)
    except Exception:
        pass


def wait_for_mode(target, poll_sec=0.5):
    """USE_ADMIN일 때, signbot_admin의 /api/mode 가 target 이 될 때까지 대기한다."""
    while True:
        ping_alive()          # 대기 중에도 살아 있다고 알린다
        try:
            with urllib.request.urlopen(f"{ADMIN_URL}/api/mode", timeout=1.0) as r:
                data = json.loads(r.read())
            if data.get("mode") == target:
                return
        except Exception:
            pass
        time.sleep(poll_sec)


class RosFrames:
    """ROS2 토픽에서 프레임을 받는다. cv2.VideoCapture 와 같은 모양(isOpened/read/release)으로 쓴다.

    V4L2 장치는 한 프로세스만 열 수 있어서, sign_demo.py 가 이미 웹캠을 잡고 있으면
    이 스크립트가 직접 열 수 없다. webcam_publisher.py 가 장치를 혼자 열고 토픽으로
    뿌리면 구독자 수 제한이 없다. sign_demo.py 의 RosFrames 와 같은 어댑터다.
    """

    def __init__(self, topic=GESTURE_CAM_TOPIC):
        import rclpy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CompressedImage, Image
        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self._node = rclpy.create_node("hand_gesture_control_frames")
        self._bridge = CvBridge()
        self._frame = None
        # 압축과 날것 둘 다 구독한다 — 발행 쪽이 어느 쪽으로 보내든 받는다.
        self._node.create_subscription(
            CompressedImage, f"{topic}/compressed", self._cb_jpeg, 1)
        self._node.create_subscription(Image, topic, self._cb, 1)
        print(f"  프레임 출처: ROS 토픽 {topic}[/compressed]")

    def _cb(self, msg):
        self._frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")

    def _cb_jpeg(self, msg):
        self._frame = self._bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

    def isOpened(self):
        return True

    def read(self):
        self._rclpy.spin_once(self._node, timeout_sec=0.02)
        return (self._frame is not None), self._frame

    def release(self):
        try:
            self._node.destroy_node()
        except Exception:
            pass


def open_cam():
    if GESTURE_SOURCE == "ros":
        return RosFrames()
    cap = cv2.VideoCapture(CAM_INDEX)
    # MJPG 로 열지 않으면 USB2 대역폭 때문에 720p 에서 7fps 밖에 안 나온다
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


WRIST = 0                # MediaPipe Hands: 손목
INDEX_TIP = 8             # MediaPipe Hands: 검지 끝 — xy 조이스틱 기준점(손목보다 조작이 편함)
PALM_CENTER = 9           # MediaPipe Hands: 중지 뿌리 — 손바닥 중앙에 가장 가까운 점, z 게이지 기준점
GAIN0 = 0.8               # 조이스틱 감도. 클수록 같은 편차에도 목표점이 빨리 움직인다
SMOOTH_N = 5
SCREEN_ASPECT = 16 / 9
MOVE_RATE = 0.3            # 최대 편차(magnitude=1)·감도 1일 때 초당 이동량(화면 비율)

Z0 = 0.5
GRIPPER_OPEN, GRIPPER_CLOSED = 1.0, 0.0

# z 게이지: 화면 오른쪽에 놓인 세로로 긴 영역. 손바닥(PALM_CENTER) 높이가
# GAUGE_Y0(위쪽 끝, z=1)~GAUGE_Y1(아래쪽 끝, z=0) 사이 어디인지로 z를 절대
# 위치로 직접 정한다 — xy 조이스틱과 달리 여긴 데드존/속도가 아니라 게이지다.
GAUGE_CX = 0.83           # 화면 오른쪽 끝에 더 가깝게
GAUGE_Y0, GAUGE_Y1 = 0.15, 0.85
GAUGE_HALF_W = 0.05       # 그리는/판정하는 폭의 절반 (프레임 비율)

# z 손떨림 억제: 원시 z를 Z_SMOOTH_N 프레임 평균으로 다듬은 뒤, 그 값이 현재
# self.z 에서 Z_DEADBAND 이상 벌어져야만 실제로 반영한다. 평균만으로는 계속
# 미세하게 흔들리며 서서히 드리프트할 수 있어서, 문턱을 넘을 때만 갱신하는
# 방식(히스테리시스)을 더했다 — 진짜로 움직였을 때는 즉시 반응하고, 손떨림
# 수준의 변화는 아예 무시한다.
Z_SMOOTH_N = 3
Z_DEADBAND = 0.03

# 양손을 동시에 Open_Palm 으로 이 시간(초) 이상 유지하면 작업모드로 복귀한다.
RETURN_HOLD_SEC = 3.0

# xy 조이스틱 영역: 화면(카메라 프레임) 왼쪽에 놓인 원 두 개.
# 중심(ROI_CX, ROI_CY)은 프레임 비율(0~1) 기준. 반지름(ROI_R)은 세로 기준 비율이고,
# 가로는 프레임 종횡비만큼 보정해서 실제 화면에 정원(正圓)으로 그려지고 계산된다.
# 안쪽 데드존 반지름은 바깥 원의 절반이다.
ROI_CX, ROI_CY = 0.22, 0.62   # 화면 왼쪽 아래 — 손을 너무 높이 들지 않아도 되게
ROI_R = 0.25
DEADZONE_R = ROI_R * 0.5

KEY_HELP = [
    ("Q", "종료"),
    ("SPACE", "목표점을 화면 중앙으로 리셋"),
    ("[ / ]", "감도 감소 / 증가"),
    ("S", "시간평균 on/off"),
    ("R", "전체 초기화"),
    ("F", "전체화면 전환"),
]

# MediaPipe Hands 21 랜드마크 연결선 (엄지, 검지, 중지, 약지, 소지, 손바닥)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class HandController:
    def __init__(self):
        self.recognizer = GestureRecognizer.create_from_options(GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=HAND_TASK),
            running_mode=RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5))
        self.ts = 0
        self.hist = deque(maxlen=SMOOTH_N)   # 조이스틱 벡터(v) 노이즈를 줄이는 이동평균
        self.smooth = True
        self.gain = GAIN0                     # 조이스틱 감도 — 속도에 곱해진다
        self.pos = np.array([0.5, 0.5])       # 누적된 목표점(화면 비율 0~1). SPACE로 중앙 리셋
        self.z_hist = deque(maxlen=Z_SMOOTH_N)   # z 손떨림 억제용 이동평균
        self.z = Z0
        self.gripper = GRIPPER_OPEN
        self.gesture = "None"
        self.gesture_score = 0.0
        self.both_open_hold = 0.0     # 양손 Open_Palm 을 동시에 유지 중인 시간(초)

    def return_triggered(self):
        return self.both_open_hold >= RETURN_HOLD_SEC

    def process(self, frame, dt):
        """반환: (xy_hand, gesture_hand)
        xy_hand      = (조이스틱 벡터 v, 랜드마크 픽셀 배열) 또는 None
                       v는 데드존 안이면 (0, 0), 밖이면 방향*크기(0~1) — 속도로 쓰인다.
        gesture_hand = 랜드마크 픽셀 배열 또는 None

        MediaPipe 의 handedness 라벨은 실측상 실제 사용자 기준과 반대로
        나온다 (모듈 docstring "핸드니스 반전" 참고). 그래서 아래에서는
        label=="Right" 를 실제 왼손(xy 제어)으로, label=="Left" 를 실제
        오른손(z/그리퍼 제스처)으로 취급한다.
        """
        h, w = frame.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.ts += 33
        res = self.recognizer.recognize_for_video(img, self.ts)

        xy_hand, gesture_hand = None, None
        self.gesture, self.gesture_score = "None", 0.0
        left_open = right_open = False   # 실제 왼손/오른손이 각각 Open_Palm 인가

        if not res.hand_landmarks:
            self.both_open_hold = 0.0
            return xy_hand, gesture_hand

        for i, lm in enumerate(res.hand_landmarks):
            px = np.array([[p.x * w, p.y * h] for p in lm])
            label = res.handedness[i][0].category_name if res.handedness and res.handedness[i] else None
            top_name = None
            if res.gestures and res.gestures[i]:
                top_name = res.gestures[i][0].category_name

            if label == "Right":  # 실제 왼손 (핸드니스 반전, 모듈 docstring 참고)
                left_open = top_name == "Open_Palm"
                # 검지 끝 좌표를 원 중심 기준 오프셋으로. 손목보다 검지 끝을 기준으로
                # 하면 손목을 꺾지 않고 손가락만 살짝 뻗어도 조작이 되어 훨씬 편하다.
                # 가로는 종횡비로 보정해서 프레임이 16:9 라도 정원(正圓)으로 계산된다.
                dx = (lm[INDEX_TIP].x - ROI_CX) * (w / h)
                dy = (lm[INDEX_TIP].y - ROI_CY)
                dist = float(np.hypot(dx, dy))
                if dist < DEADZONE_R:
                    v = np.zeros(2)                          # 데드존 — 움직임 없음(속도 0)
                else:
                    direction = np.array([dx, dy]) / dist
                    # 데드존 경계에서 0, 바깥 원(또는 그 너머)에서 1이 되도록 편다.
                    magnitude = min((dist - DEADZONE_R) / (ROI_R - DEADZONE_R), 1.0)
                    v = direction * magnitude                # 방향 * 크기(0~1) — 속도 신호
                xy_hand = v, px

            elif label == "Left":  # 실제 오른손 (핸드니스 반전, 모듈 docstring 참고)
                gesture_hand = px
                right_open = top_name == "Open_Palm"
                if top_name:
                    self.gesture, self.gesture_score = top_name, res.gestures[i][0].score

                # z: 손바닥 높이를 세로 게이지 안에서의 절대 위치로 직접 매핑한다
                # (엄지 방향을 누르고 있는 방식 대신). 이미지 y는 아래로 증가하므로
                # 게이지 위쪽(작은 y)이 z=1, 아래쪽(큰 y)이 z=0 이 되도록 뒤집는다.
                py = lm[PALM_CENTER].y
                frac = np.clip((py - GAUGE_Y0) / (GAUGE_Y1 - GAUGE_Y0), 0.0, 1.0)
                raw_z = 1.0 - frac
                self.z_hist.append(raw_z)
                smoothed_z = float(np.mean(self.z_hist))
                if abs(smoothed_z - self.z) >= Z_DEADBAND:   # 문턱 넘을 때만 반영(손떨림 억제)
                    self.z = smoothed_z

                if self.gesture == "Open_Palm":
                    self.gripper = GRIPPER_OPEN
                elif self.gesture == "Closed_Fist":
                    self.gripper = GRIPPER_CLOSED

        # 양손이 동시에 펼쳐져 있어야만 쌓인다 — 한쪽이라도 다른 모양이면 즉시 리셋.
        self.both_open_hold = self.both_open_hold + dt if (left_open and right_open) else 0.0

        return xy_hand, gesture_hand

    def update_pos(self, v, dt):
        """조이스틱 벡터 v를 속도로 삼아 목표점(self.pos)을 누적한다.

        v가 None이면(손이 안 보이면) 즉시 정지한다 — 과거 방향이 남아 손이 사라진
        뒤에도 관성으로 계속 움직이면 안 되므로, 그냥 0을 섞는 게 아니라 이력 자체를
        비운다(그러지 않으면 이동평균이 몇 프레임에 걸쳐 서서히 줄어들며 계속 움직인다).
        """
        if v is None:
            self.hist.clear()
            v = np.zeros(2)
        self.hist.append(v)
        vv = np.mean(np.array(self.hist), axis=0) if self.smooth else self.hist[-1]
        self.pos = np.clip(self.pos + vv * self.gain * MOVE_RATE * dt, 0.0, 1.0)
        return self.pos


def draw_hand(img, px, highlight=WRIST):
    """손 스켈레톤을 그리고, highlight 랜드마크(기본 손목)에 큰 원으로 표시한다.

    xy를 담당하는 손은 실제 기준점(검지 끝)을 highlight로 넘겨서, 화면에서
    뭘 기준으로 움직이는지 헷갈리지 않게 한다.
    """
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(px[a].astype(int)), tuple(px[b].astype(int)),
                 (0, 200, 255), 2, cv2.LINE_AA)
    for p in px:
        cv2.circle(img, tuple(p.astype(int)), 4, (255, 180, 0), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(px[highlight].astype(int)), 8, (0, 255, 120), 2, cv2.LINE_AA)


def draw_z_gauge(screen, z):
    x0, x1 = screen.shape[1] - 90, screen.shape[1] - 50
    y0, y1 = 40, screen.shape[0] - 40
    cv2.rectangle(screen, (x0, y0), (x1, y1), (70, 64, 58), 2, cv2.LINE_AA)
    fy = int(y1 - z * (y1 - y0))
    cv2.rectangle(screen, (x0, fy), (x1, y1), (70, 255, 150), -1, cv2.LINE_AA)
    cv2.putText(screen, f"Z {z:.2f}", (x0 - 6, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1, cv2.LINE_AA)


def draw_gripper(screen, gripper):
    open_ = gripper > 0.5
    txt = "GRIPPER OPEN" if open_ else "GRIPPER CLOSED"
    col = (70, 255, 150) if open_ else (80, 90, 255)
    cv2.putText(screen, txt, (30, screen.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, col, 2, cv2.LINE_AA)


def draw_return_hold(screen, hold_sec):
    """양손을 펼쳐 작업모드 복귀를 기다리는 동안 진행 상황을 보여준다."""
    if hold_sec <= 0:
        return
    w = screen.shape[1]
    frac = min(hold_sec / RETURN_HOLD_SEC, 1.0)
    x0, x1 = w // 2 - 150, w // 2 + 150
    y0, y1 = 60, 84
    cv2.rectangle(screen, (x0, y0), (x1, y1), (70, 64, 58), 2, cv2.LINE_AA)
    cv2.rectangle(screen, (x0, y0), (int(x0 + (x1 - x0) * frac), y1), (80, 200, 255), -1, cv2.LINE_AA)
    cv2.putText(screen, f"returning to work mode... {hold_sec:.1f}/{RETURN_HOLD_SEC:.0f}s",
                (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 255), 1, cv2.LINE_AA)


def draw_roi(frame):
    """xy 조이스틱 원 두 개(바깥 원 + 데드존)를 그린다.

    ROI_R/DEADZONE_R 은 세로 기준 비율이라 픽셀 반지름은 h 를 곱하면 된다
    (process() 의 종횡비 보정과 같은 이유 — h*x, h*y 가 1:1 픽셀 단위가 된다).
    """
    h, w = frame.shape[:2]
    cx, cy = int(ROI_CX * w), int(ROI_CY * h)
    r_outer, r_dead = int(ROI_R * h), int(DEADZONE_R * h)
    cv2.circle(frame, (cx, cy), r_outer, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r_dead, (0, 140, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, (0, 200, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, "xy joystick (left hand)", (cx - r_outer, max(cy - r_outer - 10, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)


def draw_gauge(frame, z):
    """z 게이지(세로 사각형)를 그린다. 지금 z 값만큼 아래에서부터 채운다."""
    h, w = frame.shape[:2]
    cx = int(GAUGE_CX * w)
    y0, y1 = int(GAUGE_Y0 * h), int(GAUGE_Y1 * h)
    half_w = int(GAUGE_HALF_W * h)
    x0, x1 = cx - half_w, cx + half_w
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 255), 2, cv2.LINE_AA)
    fill_y = int(y1 - z * (y1 - y0))
    cv2.rectangle(frame, (x0 + 3, fill_y), (x1 - 3, y1 - 3), (0, 200, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, "z gauge", (x0 - 10, max(y0 - 10, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)


def draw_key_legend(screen, x=20, y=30, line_h=22):
    for i, (key, desc) in enumerate(KEY_HELP):
        line_y = y + i * line_h
        cv2.putText(screen, key, (x, line_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (70, 255, 150), 1, cv2.LINE_AA)
        cv2.putText(screen, desc, (x + 70, line_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (190, 190, 190), 1, cv2.LINE_AA)


def run_session():
    """카메라를 열고, 조종 루프를 한 판(Q 또는 양손 3초 복귀까지) 돈다.

    성공적으로 한 판을 마치면(또는 최소한 카메라를 잡는 데 성공하면) True.
    카메라를 못 열면 False — 이때는 제어권을 실제로 못 가져간 것이므로
    (USE_ADMIN이면) work 로 되돌려 보낸다. 안 그러면 모드는 control 로 남은 채
    카메라 열기만 계속 실패해 대기 루프가 뜨겁게 도는 문제가 생긴다.
    """
    cap = open_cam()
    ok, frame = False, None
    _last_ping = [0.0]
    teleop = None
    if USE_ROBOT:
        from robot_teleop import RobotTeleop
        teleop = RobotTeleop()
        teleop.connect()
    # ROS 토픽은 구독 등록 직후 첫 메시지가 올 때까지 약간 지연될 수 있다 —
    # v4l2 는 보통 첫 시도에 바로 성공하지만 최대 5초까지 같이 기다려준다.
    for _ in range(50):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.1)
    if not ok or frame is None:
        source = GESTURE_CAM_TOPIC if GESTURE_SOURCE == "ros" else CAM_INDEX
        print(f"카메라({source}) 열기 실패")
        if USE_ADMIN:
            push_mode("work")
        return False
    h, w = frame.shape[:2]
    print(f"카메라 {w}x{h}")
    print("손을 카메라에 보이세요. SPACE = 목표점을 화면 중앙으로 리셋")
    if USE_ADMIN:
        print(f"  → signbot_admin 전송: {ADMIN_URL}/api/mode, /api/frame/control")
        push_mode("control")   # 이 스크립트가 카메라를 잡았다는 것 자체가 제어모드가 활성이라는 뜻
        start_frame_sender()

    P = HandController()
    SW = 1280
    SH = int(SW / SCREEN_ASPECT)

    cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("screen", 960, int(960 / SCREEN_ASPECT))
    cv2.resizeWindow("camera", 640, int(640 * h / w))
    cv2.moveWindow("screen", 60, 60)
    cv2.moveWindow("camera", 60, 700)
    full = False
    fps, t_prev = 0.0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        now = time.time()
        dt = now - t_prev
        t_prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        xy_hand, gesture_hand = P.process(frame, dt)
        # 조종 중에도 살아 있다고 알린다. 2초에 한 번이면 충분하다
        # (block_sort 는 10초 안의 신호를 살아있는 것으로 본다).
        # 프레임 번호가 아니라 시각으로 재는 이유: 이 루프에는 tick 이 없다 —
        # 있다고 가정하고 넣었다가 NameError 로 죽었다(2026-08-06).
        if now - _last_ping[0] > 2.0:
            _last_ping[0] = now
            ping_alive()
        if teleop is not None:
            # 손이 하나라도 보이면 살아있는 것으로 본다 (데드맨).
            seen = xy_hand is not None or gesture_hand is not None
            teleop.update(P.pos, P.z, P.gripper > 0.5, seen)
        draw_roi(frame)
        draw_gauge(frame, P.z)
        if xy_hand is not None:
            draw_hand(frame, xy_hand[1], highlight=INDEX_TIP)
        if gesture_hand is not None:
            draw_hand(frame, gesture_hand, highlight=PALM_CENTER)

        pt = P.update_pos(xy_hand[0] if xy_hand else None, dt)

        screen = np.full((SH, SW, 3), (22, 19, 17), np.uint8)
        for gx in range(0, SW, SW // 12):
            cv2.line(screen, (gx, 0), (gx, SH), (40, 36, 33), 1)
        for gy in range(0, SH, SH // 7):
            cv2.line(screen, (0, gy), (SW, gy), (40, 36, 33), 1)
        cv2.line(screen, (SW // 2, 0), (SW // 2, SH), (58, 52, 48), 1)
        cv2.line(screen, (0, SH // 2), (SW, SH // 2), (58, 52, 48), 1)

        # self.pos는 update_pos()에서 이미 0~1로 clamp 되므로 여기선 그대로 쓴다.
        cx = int(pt[0] * (SW - 1))
        cy = int(pt[1] * (SH - 1))
        for rad, col in zip((34, 21, 11), [(0, 70, 30), (0, 150, 65), (70, 255, 150)]):
            cv2.circle(screen, (cx, cy), rad, col, -1, cv2.LINE_AA)
        cv2.circle(screen, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)

        draw_z_gauge(screen, P.z)
        draw_gripper(screen, P.gripper)
        draw_key_legend(screen)
        draw_return_hold(screen, P.both_open_hold)

        txt = (f"gesture={P.gesture}({P.gesture_score:.2f})  gain={P.gain:.2f}  "
               f"{'avg' if P.smooth else 'raw'}  z={P.z:.2f}  "
               f"gripper={'OPEN' if P.gripper > 0.5 else 'CLOSED'}")
        cv2.putText(frame, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Q quit  S smooth  [ ] gain  SPACE center  R reset  F full",
                    (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
        if teleop is not None:
            cv2.putText(frame, teleop.status(), (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if teleop.enabled else (0, 0, 255), 2)
        cv2.putText(frame, f"{fps:.0f} fps", (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 255, 120), 2)

        # 대시보드에는 screen(추상 표시) 대신 camera(실제 영상 + 조이스틱 원 + 손
        # 스켈레톤 + z 게이지)를 보낸다 — 원격에서 손 위치가 안 보이면 조작하기
        # 어렵다는 피드백을 반영했다.
        if USE_ADMIN:
            set_latest_frame(frame)

        cv2.imshow("camera", frame)
        cv2.imshow("screen", screen)

        if P.return_triggered():
            print(f"양손 Open_Palm {RETURN_HOLD_SEC:.0f}초 유지 — 작업모드로 복귀합니다.")
            push_mode("work")
            time.sleep(0.3)   # 대시보드로 가는 마지막 전송을 기다린다 (block_sort.py 와 같은 이유)
            break

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            if USE_ADMIN:
                print("Q — 수동으로 작업모드로 복귀합니다.")
                push_mode("work")
                time.sleep(0.3)
            break
        elif k == ord('s'):
            P.smooth = not P.smooth
        elif k == ord('['):
            P.gain = max(0.1, P.gain - 0.05)
        elif k == ord(']'):
            P.gain = min(5.0, P.gain + 0.05)
        elif k == ord('r'):
            P.pos = np.array([0.5, 0.5]); P.gain = GAIN0; P.hist.clear()
            P.z = Z0; P.z_hist.clear(); P.gripper = GRIPPER_OPEN
            print("초기화")
        elif k == 32:
            P.pos = np.array([0.5, 0.5])      # 목표점을 화면 중앙으로 리셋
            P.hist.clear()
            print("목표점 중앙으로 리셋")
        elif k == ord('f'):
            full = not full
            cv2.setWindowProperty("screen", cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)

    cap.release()
    if teleop is not None:
        teleop.close()
    cv2.destroyAllWindows()
    return True


def main():
    if not USE_ADMIN:
        run_session()   # --admin 없이 쓰면 예전처럼 한 판 하고 끝
        return
    print("대기 중 — signbot_admin에서 '모드변경'으로 제어모드가 되면 자동으로 시작합니다.")
    print("완전히 끄려면 Ctrl+C.")
    while True:
        wait_for_mode("control")
        if run_session():
            print("작업모드로 돌아감 — 다시 제어모드가 되길 기다립니다. (Ctrl+C 로 완전 종료)")
        else:
            time.sleep(1.0)   # 카메라 열기 실패가 반복될 때 뜨겁게 돌지 않도록


if __name__ == "__main__":
    main()
