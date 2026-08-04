########## ColorView ##########
# 카메라 화면에 색깔별 인식 박스를 그려서 새 topic으로 발행한다 (rqt_image_view/rviz2로 확인용)
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage, ImageDraw, ImageFont

from object_detection.color_model import COLOR_HSV_RANGES, MIN_AREA, MAX_AREA_RATIO, block_candidates

# OpenCV의 cv2.putText는 한글을 못 그리므로, 한글 지원 폰트로 PIL을 이용해 그린다
FONT_PATH = '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
FONT_SIZE = 30

# 화면에 그릴 때 쓸 대표색 (BGR)
DISPLAY_COLOR = {
    "빨간색": (0, 0, 255),
    "주황색": (0, 140, 255),
    "노란색": (0, 220, 255),
    "초록색": (0, 200, 0),
    "파란색": (255, 0, 0),
    "보라색": (200, 0, 160),
}

INPUT_TOPIC = '/camera/camera/color/image_raw'
OUTPUT_TOPIC = '/object_detection/color_debug_image'


class ColorView(Node):
    def __init__(self):
        super().__init__('color_view_node')
        self.bridge = CvBridge()
        self.font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        self.pub = self.create_publisher(Image, OUTPUT_TOPIC, 10)
        self.create_subscription(Image, INPUT_TOPIC, self.on_image, 10)
        self.get_logger().info(f"color_view_node started. {INPUT_TOPIC} -> {OUTPUT_TOPIC}")

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        close_kernel = np.ones((3, 3), np.uint8)
        max_area = frame.shape[0] * frame.shape[1] * MAX_AREA_RATIO

        labels = []  # 한글 라벨은 모아뒀다가 PIL로 한번에 그린다
        for name, color_bgr in DISPLAY_COLOR.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in COLOR_HSV_RANGES[name]:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            raw_pixels = int(cv2.countNonZero(mask))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areas = sorted((cv2.contourArea(c) for c in contours), reverse=True)
            self._log_detection_reason(name, raw_pixels, areas, max_area)

            candidates = block_candidates(contours, max_area)
            if not candidates:
                continue
            best = max(candidates, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(best)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 3)
            labels.append((x, max(y - (FONT_SIZE + 10), 0), name, color_bgr))

        if labels:
            frame = self._draw_korean_labels(frame, labels)

        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)

    def _log_detection_reason(self, name, raw_pixels, areas, max_area):
        """색이 왜 안 잡혔는지(혹은 잡혔는지) 원인을 구분해서 5초에 한 번 정도 로그로 남긴다."""
        if raw_pixels == 0:
            reason = "해당 색 픽셀이 화면에 아예 없음 (HSV 범위 밖)"
        elif not areas:
            reason = f"픽셀은 {raw_pixels}개 있었지만 morphology 처리 후 뭉치가 사라짐"
        elif areas[0] < MIN_AREA:
            reason = f"가장 큰 뭉치 area={areas[0]:.0f} < MIN_AREA={MIN_AREA} (너무 작아서 노이즈로 판단)"
        elif areas[0] > max_area:
            reason = f"가장 큰 뭉치 area={areas[0]:.0f} > MAX_AREA={max_area:.0f} (너무 커서 배경으로 판단)"
        else:
            reason = f"정상 인식됨 (area={areas[0]:.0f})"
        self.get_logger().info(f"[{name}] {reason}", throttle_duration_sec=5)

    def _draw_korean_labels(self, frame, labels):
        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        for x, y, text, color_bgr in labels:
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            # 글씨 뒤에 흰색 배경을 깔아서 작은 화면에서도 잘 보이게 한다
            box = draw.textbbox((x, y), text, font=self.font)
            draw.rectangle(box, fill=(255, 255, 255))
            draw.text((x, y), text, font=self.font, fill=color_rgb)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main(args=None):
    rclpy.init(args=args)
    node = ColorView()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
