########## HsvProbe ##########
# 카메라 프레임 한 장을 받아서, 배경(나무판/체스보드)을 뺀 색깔 픽셀들의 실제 hue 분포를
# 로그로 찍어준다. 어떤 색이 실제로 몇 번 hue에 몰려있는지 눈으로 보고 정확한 HSV 범위를
# 추측이 아니라 데이터로 정하기 위한 1회성 진단 도구다.
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

INPUT_TOPIC = '/camera/camera/color/image_raw'
S_MIN, V_MIN = 30, 40  # 배경(나무/회색/흰색)을 뺄 정도로만 느슨하게 잡는다


class HsvProbe(Node):
    def __init__(self):
        super().__init__('hsv_probe_node')
        self.bridge = CvBridge()
        self.done = False
        self.create_subscription(Image, INPUT_TOPIC, self.on_image, 10)
        self.get_logger().info(f"hsv_probe_node started, waiting for one frame from {INPUT_TOPIC}")

    def on_image(self, msg):
        if self.done:
            return
        self.done = True

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        mask = (s > S_MIN) & (v > V_MIN)
        hues = h[mask]

        if hues.size == 0:
            self.get_logger().info("배경 제외하니 남는 픽셀이 없음 (S/V 기준을 더 낮춰야 할 수 있음)")
        else:
            hist = np.bincount(hues, minlength=180)
            # 인접한 hue를 묶어서(5칸 단위) 뭉치 단위로 보여준다 - 단일 hue보다 덩어리로 보는 게 더 명확하다
            bucket = 5
            bucketed = hist.reshape(-1, bucket).sum(axis=1) if 180 % bucket == 0 else None
            self.get_logger().info("=== Hue 분포 (배경 제외, 픽셀 많은 순) ===")
            top = np.argsort(hist)[::-1]
            shown = 0
            for hue in top:
                if hist[hue] < 150:
                    break
                self.get_logger().info(f"  hue={int(hue):3d}  pixels={int(hist[hue])}")
                shown += 1
                if shown >= 20:
                    break

        self.get_logger().info("측정 끝. 노드를 종료합니다.")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = HsvProbe()
    try:
        rclpy.spin(node)
    except Exception:
        pass


if __name__ == '__main__':
    main()
