#!/usr/bin/env python3
"""로봇의 RealSense 컬러 영상을 signbot_admin 제어 화면으로 중계한다.

로봇을 손동작으로 조종하려면 결국 로봇이 보는 화면을 봐야 한다 — 조종자 앞
웹캠(hand_gesture_control.py)만으로는 로봇이 지금 뭘 보고 있는지 알 수 없다.
이 스크립트는 RealSense 컬러 토픽을 구독해서 JPEG로 인코딩한 뒤
signbot_admin의 /api/frame/realsense 로 올린다. 로컬 창은 안 띄운다 — 오직
중계만 한다(디버깅용으로 보고 싶으면 voice_pickandplace 쪽 뷰어를 쓴다).

  python3 realsense_bridge.py

환경변수
  REALSENSE_TOPIC   구독할 컬러 토픽 (기본 /camera/camera/color/image_raw)
  SIGN_ADMIN_URL    signbot_admin 주소 (기본 http://localhost:5000)
  REALSENSE_JPEG    JPEG 품질 (기본 70)
  REALSENSE_FPS     signbot_admin 전송 상한 fps (기본 12 — 모니터링용이라 그 이상 필요 없음)
"""
import os
import threading
import time
import urllib.request

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

TOPIC = os.environ.get("REALSENSE_TOPIC", "/camera/camera/color/image_raw")
ADMIN_URL = os.environ.get("SIGN_ADMIN_URL", "http://localhost:5000")
JPEG_QUALITY = int(os.environ.get("REALSENSE_JPEG", 70))
PUSH_INTERVAL = 1.0 / float(os.environ.get("REALSENSE_FPS", 12))


class RealsenseBridge(Node):
    def __init__(self):
        super().__init__("realsense_bridge")
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._frame = None
        # 압축과 날것 둘 다 구독한다 — 어느 쪽이 실제로 발행되든 받는다.
        # 아무도 안 보내는 토픽을 구독하는 비용은 없다 (sign_demo.py의 RosFrames와 같은 이유).
        self.create_subscription(Image, TOPIC, self._cb, 1)
        self.create_subscription(CompressedImage, f"{TOPIC}/compressed", self._cb_jpeg, 1)
        threading.Thread(target=self._sender_loop, daemon=True).start()
        self.get_logger().info(f"{TOPIC} 구독 → {ADMIN_URL}/api/frame/realsense 전송 시작")

    def _cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        with self._lock:
            self._frame = frame

    def _cb_jpeg(self, msg):
        frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        with self._lock:
            self._frame = frame

    def _sender_loop(self):
        while rclpy.ok():
            with self._lock:
                frame = None if self._frame is None else self._frame.copy()
            if frame is not None:
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
