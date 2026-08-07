"""웹캠 한 대를 ROS2 토픽으로 뿌린다.

  ros2 run sign_processing webcam_publisher

V4L2 장치는 한 프로세스만 열 수 있다. 그래서 수어 인식과 화면 중계를 각각
따로 돌리면 "화면을 보려면 로봇이 안 움직이고, 로봇을 움직이면 화면이 안
나오는" 상태가 된다. 실제로 그 문제를 겪었다.

이 노드가 장치를 혼자 열고 토픽으로 발행하면 구독자 수에 제한이 없다.
RealSense 가 이미 같은 구조라(`/camera/camera/color/image_raw`), 검출 노드와
뷰어가 같은 영상을 동시에 쓰고 있다. 웹캠도 같은 방식으로 맞춘다.

발행은 **JPEG 압축**으로 한다. 날것(bgr8)으로 보내면 1280x720 한 장이 2.76MB
라, 30fps 에 구독자 둘이면 165MB/s 다. 실측 2026-08-05: 날것으로 보냈더니
카메라와 드라이버는 30fps 인데 토픽에는 14fps 만 나왔다(CPU 는 12% 로 한가했다 —
계산이 아니라 전송이 막힌 것이다). JPEG 로 보내면 한 장이 100~200KB 로 줄어든다.
토픽 이름은 ROS 관례를 따라 `<토픽>/compressed` 다 — rqt_image_view 와
image_transport republish 가 그대로 알아본다.

환경변수
  SIGN_CAM        장치 번호. 안 주면 이름으로 찾는다 (find_webcam 참고).
                  번호를 박아두면 안 되는 이유는 그 함수 주석에 있다.
  SIGN_CAM_TOPIC  발행할 토픽 기준 이름 (기본 /webcam/image_raw)
  SIGN_CAM_FPS    발행 주기 (기본 30)
  SIGN_CAM_JPEG   JPEG 품질 (기본 80)
  SIGN_CAM_RAW    1 이면 날것도 함께 발행한다. 느려지므로 기본은 끈다.
"""
import os
import subprocess
import sys
import threading
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

TOPIC = os.environ.get("SIGN_CAM_TOPIC", "/webcam/image_raw")
FPS = float(os.environ.get("SIGN_CAM_FPS", 30))
JPEG_QUALITY = int(os.environ.get("SIGN_CAM_JPEG", 80))
ALSO_RAW = os.environ.get("SIGN_CAM_RAW", "0") == "1"
WIDTH, HEIGHT = 1280, 720

# 이 이름이 들어간 장치는 수어용 웹캠이 아니다.
#   RealSense  검출용이다. 여기를 V4L2 로 열면 RealSense ROS 드라이버가
#              장치를 못 잡아 검출이 통째로 죽는다 — 실제로 그렇게 됐다.
#   loopback   Iriun 같은 가상 장치. 열려도 프레임이 안 온다.
NOT_WEBCAM = ("realsense", "loopback")


def list_video_devices():
    """`v4l2-ctl --list-devices` 를 파싱해 [(장치이름, [번호, ...]), ...] 로.

    출력은 "이름 (버스):" 한 줄 뒤에 들여쓴 /dev/videoN 들이 오는 형태다.
    """
    try:
        out = subprocess.run(["v4l2-ctl", "--list-devices"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    groups, name = [], None
    for line in out.splitlines():
        if not line.strip():
            continue
        if not line[0].isspace():
            name = line.strip().rstrip(":")
            groups.append((name, []))
        elif groups and "/dev/video" in line:
            groups[-1][1].append(int(line.strip().rsplit("video", 1)[1]))
    return groups


def has_mjpg(index):
    """그 번호가 실제로 영상을 내보내는 노드인가.

    한 카메라가 /dev/video 를 여러 개 만드는데 뒤쪽은 메타데이터 전용이라
    열려도 프레임이 안 온다 (실측: HD Webcam 이 7=영상, 8=메타데이터).
    """
    try:
        out = subprocess.run(["v4l2-ctl", "-d", f"/dev/video{index}",
                              "--list-formats"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "MJPG" in out


def find_webcam():
    """수어용 웹캠의 장치 번호를 **이름으로** 찾는다.

    번호를 박아두면 안 된다. USB 를 다시 꽂거나 부팅 순서가 달라지면 번호가
    통째로 밀린다. 실측 2026-08-05:
        video0      Iriun (가상 loopback)
        video1~6    RealSense D435i      ← 예전에는 여기가 웹캠이었다
        video7,8    HD Webcam            ← 진짜 수어용 (7=영상, 8=메타데이터)
    이 상태에서 예전 기본값(1)로 열면 **RealSense 를 가로채** 검출이 죽는다.
    그래서 이름으로 거르고, 영상이 나오는 노드인지까지 확인한다.
    """
    for name, indices in list_video_devices():
        low = name.lower()
        if any(bad in low for bad in NOT_WEBCAM):
            continue
        for i in indices:
            if has_mjpg(i):
                return i, name
    return None, None


def resolve_index():
    """SIGN_CAM 이 있으면 그것, 없으면 이름으로 찾은 번호."""
    env = os.environ.get("SIGN_CAM")
    if env is not None and env.strip() != "":
        return int(env)
    idx, name = find_webcam()
    if idx is None:
        sys.exit("수어용 웹캠을 못 찾았습니다. `v4l2-ctl --list-devices` 로 확인하고 "
                 "SIGN_CAM=<번호> 로 지정하세요.")
    print(f"  웹캠 자동 선택: /dev/video{idx}  ({name})")
    return idx


def open_cam(index):
    """MJPG 로 열어야 1280x720 에서 30fps 가 나온다. YUYV 는 대역폭이 모자란다."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # 어두우면 카메라가 노출을 늘리려고 스스로 프레임률을 절반으로 떨군다.
    # 이 설정만 끄면 밝기는 그대로인데 30fps 가 나온다. OpenCV 로는 못 건드려서
    # v4l2-ctl 을 직접 부른다. sign_demo.py 의 open_cam() 이 원래 하던 것인데
    # 발행 노드로 옮겨오면서 빠졌다 — 실측에서 30fps 설정에 14fps 만 나왔다.
    try:
        subprocess.run(["v4l2-ctl", "-d", f"/dev/video{index}",
                        "-c", "exposure_dynamic_framerate=0"],
                       capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass
    return cap


def probe():
    """열리는 장치 번호를 훑는다. 번호를 잘못 잡았을 때 헤매지 않도록."""
    alive = []
    for i in range(10):
        c = cv2.VideoCapture(i, cv2.CAP_V4L2)
        ok = c.isOpened() and c.read()[0]
        c.release()
        if ok:
            alive.append(i)
    return alive


class WebcamPublisher(Node):
    def __init__(self):
        super().__init__("webcam_publisher")
        index = resolve_index()
        self.cap = open_cam(index)
        ok = self.cap.isOpened()
        if ok:
            for _ in range(5):          # 첫 몇 장은 노출이 안 잡혀 버린다
                ok, _f = self.cap.read()
        if not ok:
            self.cap.release()
            sys.exit(f"카메라 {index} 에서 프레임을 못 받았습니다.\n"
                     f"  지금 열리는 인덱스: {probe() or '없음'}\n"
                     f"  SIGN_CAM=<번호> 로 지정하세요. RealSense 는 수어용이 아닙니다.")

        self.bridge = CvBridge()
        self.pub = self.create_publisher(CompressedImage, f"{TOPIC}/compressed", 1)
        # 날것은 요청할 때만. 이유는 파일 맨 위 주석 참고.
        self.pub_raw = self.create_publisher(Image, TOPIC, 1) if ALSO_RAW else None
        # 타이머를 쓰지 않고 **카메라 속도에 맞춰** 읽는 전용 스레드를 둔다.
        # create_timer(1/30) 로 돌리면 정확히 절반(15fps)만 나온다: cap.read() 가
        # 다음 프레임까지 33ms 를 막는데 타이머 주기도 33ms 라, 콜백이 끝나는
        # 순간 이미 다음 타이머 시각을 넘겨 한 장씩 건너뛰기 때문이다.
        # 실측 2026-08-05: 타이머 15.5fps → 스레드 30fps (CPU 는 16% 로 한가했다,
        # 즉 인코딩이 아니라 이 박자 문제였다).
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"카메라 {index} → {TOPIC}/compressed "
            f"({WIDTH}x{HEIGHT} @ {FPS:.0f}fps, JPEG q{JPEG_QUALITY})"
            + ("  + 날것" if self.pub_raw else ""))

    def _capture_loop(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)     # 장치가 잠깐 끊겼다. 바쁜 대기는 피한다.
                continue
            self.publish(frame)

    def publish(self, frame):
        stamp = self.get_clock().now().to_msg()
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.header.frame_id = "webcam"
            msg.format = "jpeg"
            msg.data = buf.tobytes()
            self.pub.publish(msg)
        if self.pub_raw is not None:
            raw = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            raw.header.stamp = stamp
            raw.header.frame_id = "webcam"
            self.pub_raw.publish(raw)

    def destroy_node(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = WebcamPublisher()
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
