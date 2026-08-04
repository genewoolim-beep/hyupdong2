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

SMOOTH_WIDTH = 3       # 잔잔한 요철을 없애 구간이 잘게 쪼개지는 걸 막는다
BAND_FLOOR_RATIO = 0.08  # 가장 높은 봉우리의 이 비율 이상인 구간만 색으로 본다


def find_color_bands(hist):
    """픽셀이 몰린 hue 구간을 통째로 찾아낸다. 구간 하나가 대체로 색 하나에 해당한다.

    봉우리 꼭짓점만 보면 어디서 끊어야 할지 알 수 없으므로,
    기준선 위로 이어지는 덩어리를 그대로 구간으로 삼는다.
    """
    smooth = np.convolve(hist, np.ones(SMOOTH_WIDTH) / SMOOTH_WIDTH, mode='same')
    floor = smooth.max() * BAND_FLOOR_RATIO

    bands = []
    start = None
    for h in range(180):
        if smooth[h] >= floor and start is None:
            start = h
        elif smooth[h] < floor and start is not None:
            bands.append((start, h - 1))
            start = None
    if start is not None:
        bands.append((start, 179))
    return bands


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
            self._report_bands(hist, hsv, mask)
            self._report_top_hues(hist)

        self.get_logger().info("측정 끝. 노드를 종료합니다.")
        rclpy.shutdown()

    def _report_bands(self, hist, hsv, mask):
        """색 구간과 그 구간의 채도를 함께 보여준다. 그대로 색 정의에 옮겨 쓸 수 있다."""
        bands = find_color_bands(hist)
        self.get_logger().info("=== 색 구간 (color_ranges.json 의 h/s 에 넣을 값) ===")
        if not bands:
            self.get_logger().info("  뚜렷한 구간이 없음")
            return

        h_ch, s_ch = hsv[:, :, 0], hsv[:, :, 1]
        for lo, hi in bands:
            in_band = mask & (h_ch >= lo) & (h_ch <= hi)
            sat = s_ch[in_band]
            pixels = int(hist[lo:hi + 1].sum())
            peak = int(np.argmax(hist[lo:hi + 1])) + lo
            self.get_logger().info(
                f'  "h": [{lo}, {hi}]  "s": [{max(int(np.percentile(sat, 5)) - 10, 0)}, 255]'
                f'   (픽셀 {pixels}개, 봉우리 hue={peak}, 채도 {sat.min()}~{sat.max()})'
            )

        if len(bands) >= 2 and bands[0][0] <= 2 and bands[-1][1] >= 177:
            self.get_logger().info(
                "  ※ 첫 구간과 마지막 구간은 빨강 하나가 hue 0 을 넘어가며 갈라진 것일 수 있음"
            )

    def _report_top_hues(self, hist):
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


def main(args=None):
    rclpy.init(args=args)
    node = HsvProbe()
    try:
        rclpy.spin(node)
    except Exception:
        pass


if __name__ == '__main__':
    main()
