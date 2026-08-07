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
  z: 화면 오른쪽의 세로 게이지(GAUGE_CX, GAUGE_Y0~Y1) 안에서 손바닥 높이를
     읽되, xy 십자선과 같은 방식이다 — 게이지를 위/가운데/아래로 3등분해서
     **가운데(1/3)면 정지**, 위/아래면 그 방향으로 고정 속도로 움직인다
     (절대 높이로 이동하지 않는다). 가운데를 넓게 잡은 이유는 높이를 그대로
     유지하려는 의도인데 손떨림으로 살짝만 벗어나도 움직여버리면 고정이 안
     되기 때문이다. 제어모드에 들어갈 때마다 손·로봇 시작 위치가 달라 절대
     높이로 매핑하면 진입 즉시 그 차이만큼 갑자기 움직였다 — 상대 방식이라
     그 문제도 함께 없앤다.
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

# 화면을 반시계로 이만큼 돌린다 (0/90/180/270).
#
# **기본이 90(세로)이다.** 지금 설치가 그렇다 — 이 값이 0 이면 조종 화면이
# 눕는다(2026-08-07 현장 확인). 가로로 쓰는 PC 에서는 GESTURE_ROTATE=0.
#
# **인식보다 먼저 돌린다.** 다 그린 뒤 돌리면 십자선·게이지는 돌아가지만 판정은
# 안 돌아가서, 화면에서 위로 벗어난 손이 로봇에게는 옆으로 읽힌다.
# 순서는 회전 → 거울이다. 거울은 조작자 기준 좌우여야 하므로 똑바로 세운 뒤에
# 뒤집어야 한다.
GESTURE_ROTATE = int(os.environ.get("GESTURE_ROTATE", 90)) % 360
_ROT = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE}


def orient(frame):
    """카메라 프레임을 사람이 보는 방향으로 세우고 거울로 만든다.

    조종 창(teleop_mode)과 단독 실행이 **같은 함수**를 쓴다 — 한쪽만 돌리면
    두 화면에서 같은 손짓이 다른 뜻이 된다.
    """
    if GESTURE_ROTATE in _ROT:
        frame = cv2.rotate(frame, _ROT[GESTURE_ROTATE])
    return cv2.flip(frame, 1)

# ───────────────────────── signbot_admin 연동 ─────────────────────────
# --admin 을 붙였을 때만 대시보드로 모드 상태를 보낸다. sign_demo.py/block_sort.py 와
# 같은 패턴(ADMIN_URL, "--admin" in sys.argv) 이다.
ADMIN_URL = os.environ.get("SIGN_ADMIN_URL", "http://localhost:5000")
USE_ADMIN = "--admin" in sys.argv
# --robot 을 붙여야 실제 로봇이 움직인다. 기본은 화면만 — 손동작 인식을
# 먼저 눈으로 확인하고 나서 팔을 붙이는 순서가 안전하다.
# 속도·작업영역은 robot_teleop.py 의 환경변수로 조정한다.
#
# **작업모드(block_sort.py sign)와 함께 쓸 때는 --robot 을 붙이지 않는다.**
# 2026-08-06 이후 제어모드는 block_sort 안에서 돈다(block_sort/teleop_mode.py).
# 여기서도 로봇에 붙으면 한 로봇에 DSR 연결이 둘이 되어, TCP 가 0.0mm 로 풀리고
# 모션이 거부되고 위치 조회가 멎는다 — 합친 이유가 그것이다.
# --robot 이 남아 있는 것은 작업모드를 안 쓰는 PC(로봇 + 웹캠만)를 위해서다.
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
# GAUGE_Y0(위쪽 끝, z=1)~GAUGE_Y1(아래쪽 끝, z=0) 사이 어디인지로 0~1 을 만든다.
# 그 값은 **절대 높이가 아니라 방향 신호**다 — 0.5 가 중앙(정지)이고, 위/아래로
# Z_DEADZONE 넘게 벗어나면 그 방향으로 일정 속도다(robot_teleop.update 참고).
GAUGE_CX = 0.83           # 화면 오른쪽 끝에 더 가깝게
GAUGE_Y0, GAUGE_Y1 = 0.15, 0.85
GAUGE_HALF_W = 0.05       # 그리는/판정하는 폭의 절반 (프레임 비율)

# 게이지를 위/가운데(정지)/아래로 3등분하는 경계. **로봇이 쓰는 값을 그대로
# 가져온다** — 여기에 같은 숫자를 또 적으면 한쪽만 고쳐졌을 때 화면의 STOP 칸과
# 실제로 멈추는 구간이 어긋난다. 그 어긋남은 "가만히 있는데 팔이 내려간다" 로
# 나타나므로 눈으로 알아채기 어렵다.
# robot_teleop 은 import 만으로 로봇도 rclpy 도 건드리지 않는다(상수뿐).
# Y_SIGN 도 같은 이유로 가져온다 — 화면의 좌우 글자와 실제로 나가는 속도 부호가
# 갈라지면 사람이 반대로 밀게 된다(draw_roi 참고).
from robot_teleop import Z_DEADZONE, Y_SIGN               # noqa: E402

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
# 중심(ROI_CX, ROI_CY)은 프레임 비율(0~1) 기준.
#
# 반지름(ROI_R)은 **짧은 변** 기준 비율이다. 가로 기준으로 하면 세로 화면
# (GESTURE_ROTATE=90) 에서 십자선이 화면 밖으로 나가고, 세로 기준으로 하면
# 가로 화면에서 같은 일이 생긴다. 짧은 변에 매어 두면 어느 방향이든 크기가
# 그대로다. 가로 화면에서는 짧은 변이 곧 높이라 예전과 값이 같다.
# 판정(process)과 그리기(draw_roi)가 **같은 단위**를 써야 한다 — 다르면 화면의
# 정지 사각형과 실제로 멈추는 구역이 어긋난다.
ROI_CX, ROI_CY = (float(os.environ.get("GESTURE_ROI_CX", 0.22)),
                  float(os.environ.get("GESTURE_ROI_CY", 0.62)))
ROI_R = float(os.environ.get("GESTURE_ROI_R", 0.25))
DEADZONE_R = ROI_R * 0.5


def unit(frame_or_shape):
    """길이의 기준이 되는 짧은 변(픽셀)."""
    h, w = (frame_or_shape.shape[:2] if hasattr(frame_or_shape, "shape")
            else frame_or_shape[:2])
    return float(min(h, w))


def roi_center(frame):
    """십자선 중심(픽셀). 화면 밖으로 나가지 않게 안쪽으로 물린다.

    ROI_CX/CY 를 비율 그대로 쓰면 세로 화면(GESTURE_ROTATE=90)에서 십자선 팔이
    화면을 넘어간다 — 실측: 720 폭에서 중심 158px, 팔 180px.
    **판정(process)과 그리기(draw_roi)가 이 함수 하나를 써야 한다.** 각자 계산하면
    화면의 정지 사각형과 실제로 멈추는 자리가 어긋나고, 그러면 사람은 멈춘 줄
    알고 손을 두는데 팔은 계속 간다.
    """
    h, w = frame.shape[:2]
    arm = ROI_R * unit(frame) + 8.0        # 8px 은 화살촉·글자 여유
    cx = min(max(ROI_CX * w, arm), w - arm)
    cy = min(max(ROI_CY * h, arm), h - arm)
    return cx, cy

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
                # 검지 끝 좌표를 십자선 중심 기준 오프셋으로. 손목보다 검지 끝을
                # 기준으로 하면 손목을 꺾지 않고 손가락만 살짝 뻗어도 조작이 되어
                # 훨씬 편하다.
                # 두 축 모두 **짧은 변**으로 나눈다(unit). 그래야 가로·세로 어느
                # 화면에서든 데드존이 정사각이고, 그리기(draw_roi)와 같은 단위가 된다.
                s = unit(frame)
                rcx, rcy = roi_center(frame)
                dx = (lm[INDEX_TIP].x * w - rcx) / s
                dy = (lm[INDEX_TIP].y * h - rcy) / s
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


def draw_z_gauge(screen, teleop, z_range):
    """screen 패널에 실제 로봇의 z 높이(mm)를 연속 값으로 보여준다.

    camera 창의 3등분 게이지(draw_gauge)는 조작 **입력** 상태(위/정지/아래
    중 어디 있는지)를 보여주는 것이고, 여기는 반대로 지금 로봇이 실제로
    어디 있는지 보여주는 **상태** 표시다 — 실제 높이는 연속값이므로 여기는
    3등분하지 않는다. --robot 없이는(또는 아직 위치를 못 읽었으면) "Z --"만
    표시한다.
    """
    x0, x1 = screen.shape[1] - 90, screen.shape[1] - 50
    y0, y1 = 40, screen.shape[0] - 40
    cv2.rectangle(screen, (x0, y0), (x1, y1), (70, 64, 58), 2, cv2.LINE_AA)

    pos = teleop.cur_pos() if (teleop is not None and teleop.enabled) else None
    if pos is None:
        cv2.putText(screen, "Z --", (x0 - 10, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (110, 110, 110), 1, cv2.LINE_AA)
        return

    z_mm = pos[2]
    zmin, zmax = z_range
    frac = float(np.clip((z_mm - zmin) / (zmax - zmin), 0.0, 1.0))
    fy = int(y1 - frac * (y1 - y0))
    cv2.rectangle(screen, (x0, fy), (x1, y1), (70, 255, 150), -1, cv2.LINE_AA)
    cv2.putText(screen, f"Z {z_mm:.0f}mm", (x0 - 24, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1, cv2.LINE_AA)


def draw_gripper(screen, gripper):
    open_ = gripper > 0.5
    txt = "GRIPPER OPEN" if open_ else "GRIPPER CLOSED"
    col = (70, 255, 150) if open_ else (80, 90, 255)
    cv2.putText(screen, txt, (30, screen.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, col, 2, cv2.LINE_AA)


def draw_robot_coords(screen, teleop):
    """실제 로봇의 현재 좌표(mm)를 표시한다.

    xy 초록 점/z 게이지는 로컬 입력 상태(조이스틱 목표점, 십자선 구간)일
    뿐 실제 로봇 위치가 아니다 — 실제 좌표는 --robot 으로 연결된 teleop 이
    로봇에서 직접 읽어온 값이라 여기서 따로 보여준다.
    """
    x, y = 30, screen.shape[0] - 66
    if teleop is None:
        cv2.putText(screen, "ROBOT: --robot 없이 실행 중 (연결 안 됨)", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (110, 110, 110), 1, cv2.LINE_AA)
        return
    col = (70, 255, 150) if teleop.enabled else (80, 90, 255)
    cv2.putText(screen, teleop.status(), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, col, 2, cv2.LINE_AA)


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
    """xy 십자선을 그린다.

    로봇 제어가 **속도 지령**이라 방향만 의미가 있다. 중앙 사각(데드존)이면
    정지, 어느 쪽으로 벗어나든 그 축으로 **일정 속도**다 — 얼마나 벗어났는지는
    속도를 안 바꾼다. 원(조이스틱)으로 그리면 "멀리 갈수록 빠르다" 로 읽혀
    실제 동작과 어긋나므로 십자로 바꿨다.

    ROI_R/DEADZONE_R 은 **짧은 변** 기준 비율이다(unit). 판정(process)과 같은
    단위여야 화면의 정지 사각형과 실제로 멈추는 구역이 일치한다.
    """
    s = unit(frame)
    rcx, rcy = roi_center(frame)
    cx, cy = int(rcx), int(rcy)
    arm, dead = int(ROI_R * s), int(DEADZONE_R * s)
    C, D = (0, 200, 255), (0, 140, 255)

    # 십자 축
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), C, 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), C, 2, cv2.LINE_AA)
    # 방향 화살촉
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        tx, ty = cx + dx * arm, cy + dy * arm
        cv2.circle(frame, (tx, ty), 5, C, -1, cv2.LINE_AA)
    # 정지 구역 (데드존)
    cv2.rectangle(frame, (cx - dead, cy - dead), (cx + dead, cy + dead),
                  D, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, C, -1, cv2.LINE_AA)

    # 글자 두 가지를 함께 적는다.
    #   L/R      **조작자 기준**이고 항상 고정이다. 화면이 거울(cv2.flip)이라
    #            내 오른손을 오른쪽으로 옮기면 화면에서도 오른쪽으로 간다.
    #   ±Y       그때 base 로 나가는 부호. 선 자리에 따라 뒤집히므로
    #            **실제로 보내는 값(Y_SIGN)에서 뽑는다.**
    # 축 글자를 손으로 박아두면 매핑을 바꿨을 때 화면만 옛말을 한다 — 조종에서
    # 화면이 거짓말하면 사람이 반대로 밀고, 그건 팔이 엉뚱한 쪽으로 가는 것이다.
    # 좌우 글자는 팔 **안쪽 위**에 붙인다. 팔 바깥에 두면 세로 화면에서 화면을
    # 벗어나고(실측: 중심 188px, 팔 180px → 글자 x 가 -52), 화면 안으로 물리면
    # 팔 끝 점과 겹친다. 안쪽은 십자선이 이미 화면 안이라 항상 자리가 있다.
    h = frame.shape[0]
    right_ax, left_ax = ("+Y", "-Y") if Y_SIGN < 0 else ("-Y", "+Y")
    for txt, org in (("FWD +X", (cx - 30, max(cy - arm - 12, 14))),
                     ("BACK -X", (cx - 34, min(cy + arm + 22, h - 6))),
                     (f"L {left_ax}", (cx - arm + 6, cy - 10)),
                     (f"R {right_ax}", (cx + arm - 52, cy - 10))):
        cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, C, 1, cv2.LINE_AA)
    cv2.putText(frame, "STOP", (cx - 20, cy - dead - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, D, 1, cv2.LINE_AA)


def z_zone(z):
    """게이지 값(0~1)이 지금 어느 구간인가. "UP" | "STOP" | "DOWN"

    화면과 로봇이 **같은 판단**을 쓰게 하는 한 곳이다. 여기와 robot_teleop 의
    vz 계산이 갈라지면 화면에는 STOP 인데 팔이 내려가는 상태가 되고, 그건
    눈으로 알아채기 어렵다. 그래서 경계값(Z_DEADZONE)도 그쪽에서 가져온다.
    """
    off = float(z) - 0.5
    return "UP" if off > Z_DEADZONE else ("DOWN" if off < -Z_DEADZONE else "STOP")


def draw_gauge(frame, z):
    """z 게이지 — 위/가운데(정지)/아래 3등분한 **정적** 위젯이다.

    xy 십자선(draw_roi)과 똑같은 이유로 손 높이를 따라 계속 움직이는 표시를
    두지 않는다. 실제 손 위치는 draw_hand()가 그리는 손 스켈레톤이 이미
    보여주므로, 여기서는 지금 어느 구간(위/정지/아래)이 활성인지만 그 구간을
    밝혀서 보여준다 — 값이 아니라 상태를 표시한다.
    """
    h, w = frame.shape[:2]
    cx = int(GAUGE_CX * w)
    y0, y1 = int(GAUGE_Y0 * h), int(GAUGE_Y1 * h)
    half_w = int(GAUGE_HALF_W * unit(frame))
    x0, x1 = cx - half_w, cx + half_w
    ymid = (y0 + y1) // 2
    dead_px = int(Z_DEADZONE * (y1 - y0))
    up_edge, down_edge = ymid - dead_px, ymid + dead_px   # 3등분 경계

    zone = z_zone(z)

    ON, OFF_UD, OFF_STOP = (0, 200, 255), (55, 50, 46), (45, 60, 58)
    cv2.rectangle(frame, (x0 + 2, y0), (x1 - 2, up_edge),
                  ON if zone == "UP" else OFF_UD, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x0 + 2, up_edge), (x1 - 2, down_edge),
                  (0, 140, 255) if zone == "STOP" else OFF_STOP, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x0 + 2, down_edge), (x1 - 2, y1),
                  ON if zone == "DOWN" else OFF_UD, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x0, y0), (x1, y1), ON, 2, cv2.LINE_AA)
    cv2.line(frame, (x0, up_edge), (x1, up_edge), (30, 30, 30), 1, cv2.LINE_AA)
    cv2.line(frame, (x0, down_edge), (x1, down_edge), (30, 30, 30), 1, cv2.LINE_AA)

    cv2.putText(frame, "UP", (x0 - 8, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ON, 1, cv2.LINE_AA)
    cv2.putText(frame, "DOWN", (x0 - 24, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ON, 1, cv2.LINE_AA)
    cv2.putText(frame, "STOP", (x0 - 60, ymid + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)


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
    z_range = (0.0, 1.0)   # --robot 없으면 draw_z_gauge 는 어차피 "Z --"만 표시하므로 값은 안 씀
    if USE_ROBOT:
        from robot_teleop import RobotTeleop, Z_MIN, Z_MAX
        teleop = RobotTeleop()
        teleop.connect()
        z_range = (Z_MIN, Z_MAX)
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
        frame = orient(frame)          # 회전(세로 세우기) → 거울

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
            # **지금 벗어난 방향**을 준다. 누적 목표점(P.pos)이 아니다 —
            # 누적값은 손을 중앙으로 되돌려도 벗어난 채 남아 팔이 계속 기어간다
            # (실측 2026-08-06: 데드존 안에 손이 있는데 로봇이 움직였다).
            vec = xy_hand[0] if xy_hand is not None else (0.0, 0.0)
            teleop.update(vec, P.z, P.gripper > 0.5, seen)
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

        draw_z_gauge(screen, teleop, z_range)
        draw_gripper(screen, P.gripper)
        draw_robot_coords(screen, teleop)
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
