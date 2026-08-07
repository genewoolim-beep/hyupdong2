#!/usr/bin/env python3
"""로봇의 RealSense 컬러 영상을 signbot_admin 제어 화면으로 중계한다.
**파지 지점을 증강현실로 겹쳐** 보낸다.

로봇을 손동작으로 조종하려면 결국 로봇이 보는 화면을 봐야 한다 — 조종자 앞
웹캠(hand_gesture_control.py)만으로는 로봇이 지금 뭘 보고 있는지 알 수 없다.
그런데 영상만 봐도 "지금 내리면 물릴까" 는 알 수 없다. 그래서 손가락 사이
구역을 그려 준다:

    평소        빨간 네모 + "파지 구역 비어 있음"
    블록 들어옴  초록 네모 + "파지 가능 — 빨간색 (160mm 아래)"

로컬 창은 안 띄운다 — 오직 중계만 한다.

## 왜 여기서 그리나

조종 루프(block_sort 의 제어모드)에 넣지 않는다. 1280x720 을 디코딩하고 그려
다시 인코딩하는 일이 **안전에 직결된 15fps 루프**를 갉아먹는다. 이 프로세스는
모니터링 전용이라 느려도 위험하지 않다.

**로봇 자세도 DSR 연결도 필요 없다.** 카메라가 그리퍼에 붙어 있어서 파지 구역은
화면에서 고정이다(grasp_overlay 머리말). 바뀌는 것은 손가락 벌림(TF)과 블록이
얼마나 아래인지(깊이 영상)뿐이고, 둘 다 토픽이다 — 한 로봇에 DSR 연결을 둘로
만들지 않는다(그게 2026-08-06 사고의 근원이었다).

  python3 realsense_bridge.py

환경변수
  REALSENSE_TOPIC   구독할 컬러 토픽 (기본 /camera/camera/color/image_raw)
  SIGN_ADMIN_URL    signbot_admin 주소 (기본 http://localhost:5000)
  REALSENSE_JPEG    JPEG 품질 (기본 70)
  REALSENSE_FPS     signbot_admin 전송 상한 fps (기본 12 — 모니터링용이라 그 이상 필요 없음)
  GRASP_AR          0 이면 AR 을 끄고 영상만 중계한다 (기본 1)
  GRASP_AR_HZ       블록 검출 주기 (기본 5 — 색 6개를 매 프레임 훑을 필요는 없다)
"""
import os
import sys
import threading
import time
import urllib.request

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TOPIC = os.environ.get("REALSENSE_TOPIC", "/camera/camera/color/image_raw")
DEPTH_TOPIC = os.environ.get(
    "REALSENSE_DEPTH", "/camera/camera/aligned_depth_to_color/image_raw")
INFO_TOPIC = os.environ.get("REALSENSE_INFO", "/camera/camera/color/camera_info")
ADMIN_URL = os.environ.get("SIGN_ADMIN_URL", "http://localhost:5000")
JPEG_QUALITY = int(os.environ.get("REALSENSE_JPEG", 70))
PUSH_INTERVAL = 1.0 / float(os.environ.get("REALSENSE_FPS", 12))
USE_AR = os.environ.get("GRASP_AR", "1") != "0"

# 로봇 시점 화면을 반시계로 이만큼 돌린다 (0/90/180/270). **기본이 90(세로)이다** —
# 판을 내려다보는 시야가 세로로 길어 그쪽이 보기 낫다.
# AR 은 돌린 좌표에 그린다(grasp_overlay.render) — 다 그린 뒤 돌리면 글자도 눕는다.
REALSENSE_ROTATE = int(os.environ.get("REALSENSE_ROTATE", 90)) % 360
AR_INTERVAL = 1.0 / float(os.environ.get("GRASP_AR_HZ", 5))

# 손가락 링크 이름. 벌어진 폭을 여기서 읽는다 — 모드버스로 물어보면 컨트롤러의
# 모션 큐와 겹쳐 이동 명령이 거부된 적이 있다(robot_teleop._set_gripper 주석).
FLANGE_FRAME = os.environ.get("GRASP_FLANGE_FRAME", "link_6")
FINGER_FRAMES = (os.environ.get("GRASP_FINGER_L", "rg2_left_inner_finger"),
                 os.environ.get("GRASP_FINGER_R", "rg2_right_inner_finger"))


class RealsenseBridge(Node):
    def __init__(self):
        super().__init__("realsense_bridge")
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._frame = None
        self._depth = None
        self._blocks = []           # [(카메라좌표 mm, 색이름), ...]
        self._t_ar = 0.0
        # 압축과 날것 둘 다 구독한다 — 어느 쪽이 실제로 발행되든 받는다.
        # 아무도 안 보내는 토픽을 구독하는 비용은 없다 (sign_demo.py의 RosFrames와 같은 이유).
        self.create_subscription(Image, TOPIC, self._cb, 1)
        self.create_subscription(CompressedImage, f"{TOPIC}/compressed", self._cb_jpeg, 1)

        self.view = None
        self.det = None
        if USE_AR:
            import grasp_overlay
            self.go = grasp_overlay
            self.view = grasp_overlay.GraspView()
            # 내부파라미터는 **받아서** 쓴다. 짐작한 값으로 그리면 화면이 거짓말을 한다.
            self.create_subscription(CameraInfo, INFO_TOPIC, self._cb_info, 1)
            self.create_subscription(Image, DEPTH_TOPIC, self._cb_depth, 1)
            self.det, why = grasp_overlay.load_detector()
            if self.det is None:
                self.get_logger().warn(
                    f"색 검출을 못 불러왔습니다({why}) — 파지 구역은 그리지만 "
                    "'파지 가능' 판정은 안 합니다. 워크스페이스를 소싱했는지 보세요.")
            self._tf = self._make_tf()
            self.get_logger().info(
                f"파지 지점 AR 켜짐 — 깊이 {DEPTH_TOPIC}, 손가락 {FINGER_FRAMES[0]}/"
                f"{FINGER_FRAMES[1]} ({int(1/AR_INTERVAL)}Hz 검출, 화면 회전 {REALSENSE_ROTATE}°)")

        threading.Thread(target=self._sender_loop, daemon=True).start()
        self.get_logger().info(f"{TOPIC} 구독 → {ADMIN_URL}/api/frame/realsense 전송 시작")

    def _make_tf(self):
        """손가락 벌림을 읽을 TF 버퍼. 실패해도 AR 은 기본 벌림으로 계속 그린다."""
        try:
            import tf2_ros
            buf = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(buf, self)
            return buf
        except Exception as e:
            self.get_logger().warn(f"TF 를 못 붙였습니다({e}) — 벌림은 기본값으로 그립니다")
            return None

    def _cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        with self._lock:
            self._frame = frame

    def _cb_jpeg(self, msg):
        frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        with self._lock:
            self._frame = frame

    def _cb_depth(self, msg):
        # 정렬된 깊이라 컬러와 픽셀이 1:1 이고 단위는 mm 다.
        d = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        with self._lock:
            self._depth = d

    def _cb_info(self, msg):
        if self.view is not None and self.view.intr is None:
            self.view.intr = {"fx": msg.k[0], "fy": msg.k[4],
                              "ppx": msg.k[2], "ppy": msg.k[5]}
            self.get_logger().info(
                f"내부파라미터 수신 fx={msg.k[0]:.0f} ppx={msg.k[2]:.0f} ppy={msg.k[5]:.0f}")

    # ── AR ──
    def _read_opening(self):
        """지금 손가락 반폭(mm). TF 에서 두 손가락의 x 를 읽어 평균한다."""
        if self._tf is None:
            return None
        try:
            xs = []
            for f in FINGER_FRAMES:
                tr = self._tf.lookup_transform(
                    FLANGE_FRAME, f, rclpy.time.Time()).transform.translation
                xs.append(abs(tr.x) * 1000.0)
            return sum(xs) / len(xs)
        except Exception:
            return None

    def _draw_ar(self, frame):
        """파지 구역을 그리고 **돌린 프레임**을 돌려준다. 못 그리면 그대로 돌려준다.

        그리기는 매 프레임, 블록 검출은 드물게(AR_INTERVAL).
        """
        if self.view is None or self.view.intr is None:
            return self._rotate_only(frame)
        half = self._read_opening()
        if half:
            self.view.set_opening(half)
        now = time.time()
        if now - self._t_ar >= AR_INTERVAL:
            self._t_ar = now
            with self._lock:
                depth = None if self._depth is None else self._depth.copy()
            self._blocks = self.go.find_blocks(self.view, frame, depth, self.det)
            self._last_depth = depth
        # 착지점은 매 프레임 다시 잰다 — 팔이 내려가는 동안 표시가 따라와야 한다.
        # 깊이 한 장을 몇 번 찍어보는 것뿐이라 싸다(landing_z 는 3회 반복).
        with self._lock:
            depth_now = None if self._depth is None else self._depth
        out, _hit, _name = self.go.render(frame, self.view, self._blocks,
                                          depth=depth_now, rotate=REALSENSE_ROTATE)
        return out

    @staticmethod
    def _rotate_only(frame):
        """AR 을 못 그릴 때도 화면 방향은 맞춘다."""
        flag = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_CLOCKWISE}.get(REALSENSE_ROTATE)
        return frame if flag is None else cv2.rotate(frame, flag)

    def _sender_loop(self):
        while rclpy.ok():
            with self._lock:
                frame = None if self._frame is None else self._frame.copy()
            if frame is not None:
                if USE_AR:
                    try:
                        frame = self._draw_ar(frame)
                    except Exception as e:
                        # AR 이 터져도 중계는 계속돼야 한다 — 조종자는 영상이라도
                        # 봐야 한다. 화면 방향은 그래도 맞춘다.
                        self.get_logger().warn(f"AR 그리기 실패({e}) — 영상만 보냅니다",
                                               once=True)
                        frame = self._rotate_only(frame)
                else:
                    frame = self._rotate_only(frame)
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if ok:
                    try:
                        req = urllib.request.Request(
                            f"{ADMIN_URL}/api/frame/realsense", data=buf.tobytes(),
                            headers={"Content-Type": "image/jpeg"}, method="POST")
                        urllib.request.urlopen(req, timeout=1.0)
                    except Exception:
                        pass
            time.sleep(PUSH_INTERVAL)


def main():
    rclpy.init()
    node = RealsenseBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # rclpy.spin()이 SIGINT를 받으면 자체적으로 이미 shutdown을 호출해 둔다.
        # 여기서 다시 부르면 "rcl_shutdown already called"로 죽으므로 확인 후 부른다.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
