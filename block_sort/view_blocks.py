#!/usr/bin/env python3
"""
블록 검출 실시간 화면 — 로봇이 실제로 쓰는 경로 그대로 그린다.

  python3 view_blocks.py            회전 박스 + 중심 + 각도
  python3 view_blocks.py --mask     마스크도 함께 (임계값 조정용)
  python3 view_blocks.py --nogate   깊이 게이팅 끄고 비교

팀의 color_view 는 축정렬 박스만 그리고 깊이 게이팅을 거치지 않는다.
이 뷰어는 ColorModel 과 같은 마스크를 써서, 로봇이 집으러 갈 자리와
손목을 얼마나 돌릴지를 눈으로 확인할 수 있게 한다.
  q 종료 / g 깊이 게이팅 토글 / m 마스크 토글
"""
import os
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

# 같은 저장소의 object_detection 패키지를 그대로 쓴다. 로봇이 보는 것과
# 화면이 달라지면 진단 도구로서 의미가 없기 때문이다.
_OD = os.environ.get("OD_SRC", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "voice_pickandplace", "src", "object_detection"))
sys.path.insert(0, _OD)
from object_detection.color_model import (COLOR_HSV_RANGES, MAX_AREA_RATIO,   # noqa: E402
                                          TOP_BAND, block_candidates)

DRAW = {"빨간색": (0, 0, 255), "주황색": (0, 140, 255), "노란색": (0, 220, 255),
        "초록색": (0, 200, 0), "파란색": (255, 0, 0), "보라색": (200, 0, 160)}
FONT = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
WIN = "blocks"


class Viewer(Node):
    def __init__(self, gate, show_mask):
        super().__init__("view_blocks")
        from PIL import ImageFont
        self.font = ImageFont.truetype(FONT, 24)
        self.gate, self.show_mask = gate, show_mask
        self.b = CvBridge()
        self.color = self.depth = None
        self.create_subscription(
            Image, "/camera/camera/color/image_raw", self._c, 10)
        self.create_subscription(
            Image, "/camera/camera/aligned_depth_to_color/image_raw", self._d, 10)
        self.create_timer(0.05, self.draw)
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1280, 720)

    def _c(self, m):
        self.color = self.b.imgmsg_to_cv2(m, "bgr8")

    def _d(self, m):
        self.depth = self.b.imgmsg_to_cv2(m, "passthrough")

    def find(self, name):
        """ColorModel 과 같은 순서로 마스크를 만든다."""
        hsv = cv2.cvtColor(self.color, cv2.COLOR_BGR2HSV)
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in COLOR_HSV_RANGES[name]:
            m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        if self.gate and self.depth is not None and self.depth.shape[:2] == m.shape[:2]:
            v = self.depth[(m > 0) & (self.depth > 0)]
            if v.size >= 200:
                t = float(np.percentile(v, 20))
                band = ((self.depth > t - TOP_BAND)
                        & (self.depth < t + TOP_BAND)).astype(np.uint8) * 255
                g = cv2.bitwise_and(m, band)
                if cv2.countNonZero(g) >= 200:
                    m = g
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cc = block_candidates(cs, self.color.shape[0] * self.color.shape[1] * MAX_AREA_RATIO)
        return (max(cc, key=cv2.contourArea) if cc else None), m

    def draw(self):
        if self.color is None:
            return
        img = self.color.copy()
        masks = np.zeros(img.shape[:2], np.uint8)
        labels = []
        for name, col in DRAW.items():
            c, m = self.find(name)
            masks |= m
            if c is None:
                continue
            rect = cv2.minAreaRect(c)
            (cx, cy), (w, h), ang = rect
            if w < h:
                ang, w, h = ang + 90, h, w
            a = ((ang + 45.0) % 90.0) - 45.0        # 손목 보정량
            cv2.drawContours(img, [np.int0(cv2.boxPoints(rect))], 0, col, 3)
            cv2.drawMarker(img, (int(cx), int(cy)), (255, 255, 255),
                           cv2.MARKER_CROSS, 22, 2)
            z = 0
            if self.depth is not None:
                p = self.depth[max(0, int(cy) - 4):int(cy) + 5,
                               max(0, int(cx) - 4):int(cx) + 5]
                p = p[p > 0]
                z = int(np.median(p)) if p.size else 0
            labels.append((int(cx) - 70, max(int(cy) - h / 2 - 34, 0),
                           f"{name} {a:+.0f}° {z}mm", col))
        # 광축 — 2차 검출은 이 십자에 블록을 올리는 것이 목표다
        cv2.drawMarker(img, (img.shape[1] // 2, img.shape[0] // 2),
                       (0, 255, 255), cv2.MARKER_CROSS, 40, 1)
        if self.show_mask:
            img = cv2.addWeighted(img, 1.0,
                                  cv2.cvtColor(masks, cv2.COLOR_GRAY2BGR), 0.35, 0)
        img = self._ko(img, labels)
        cv2.putText(img, f"gate={'ON' if self.gate else 'OFF'}  q/g/m",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(WIN, img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            raise KeyboardInterrupt
        if k == ord("g"):
            self.gate = not self.gate
        if k == ord("m"):
            self.show_mask = not self.show_mask

    def _ko(self, img, labels):
        from PIL import Image as PI, ImageDraw
        pil = PI.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for x, y, t, col in labels:
            box = d.textbbox((x, y), t, font=self.font)
            d.rectangle(box, fill=(255, 255, 255))
            d.text((x, y), t, font=self.font, fill=(col[2], col[1], col[0]))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    rclpy.init()
    n = Viewer("--nogate" not in sys.argv, "--mask" in sys.argv)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
