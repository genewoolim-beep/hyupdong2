#!/usr/bin/env python3
"""
블록 검출 실시간 화면 — 로봇이 실제로 쓰는 경로 그대로 그린다.

  python3 view_blocks.py            회전 박스 + 중심 + 각도 + 구역
  python3 view_blocks.py --mask     마스크도 함께 (임계값 조정용)
  python3 view_blocks.py --nogate   깊이 게이팅 끄고 비교
  python3 view_blocks.py --nozone   구역 표시 끄기

팀의 color_view 는 축정렬 박스만 그리고 깊이 게이팅을 거치지 않는다.
이 뷰어는 ColorModel 과 같은 마스크를 써서, 로봇이 집으러 갈 자리와
손목을 얼마나 돌릴지를 눈으로 확인할 수 있게 한다.

구역 표시는 zones.yaml 의 티칭 좌표를 지금 팔 자세로 역투영해 그린다.
어떤 블록이 어느 구역에 속하는지(= 로봇이 집을지 말지)를 눈으로 가릴 수 있다.
로봇 드라이버가 없으면 이 부분만 조용히 꺼지고 나머지는 그대로 동작한다.
  q 종료 / g 깊이 게이팅 토글 / m 마스크 토글 / z 구역 토글
"""
import os
import sys

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image

# 같은 저장소의 object_detection 패키지를 그대로 쓴다. 로봇이 보는 것과
# 화면이 달라지면 진단 도구로서 의미가 없기 때문이다.
_OD = os.environ.get("OD_SRC", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sign_pickandplace", "src", "object_detection"))
sys.path.insert(0, _OD)
from object_detection.color_model import (COLOR_HSV_RANGES, MAX_AREA_RATIO,   # noqa: E402
                                          block_candidates, keep_top_face)

DRAW = {"빨간색": (0, 0, 255), "주황색": (0, 140, 255), "노란색": (0, 220, 255),
        "초록색": (0, 200, 0), "파란색": (255, 0, 0), "보라색": (200, 0, 160)}
FONT = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
WIN = "blocks"

HERE = os.path.dirname(os.path.abspath(__file__))
ZONES_YAML = os.path.join(HERE, "zones.yaml")
HANDEYE = os.environ.get("HANDEYE", os.path.join(
    os.path.dirname(HERE), "calib", "T_gripper2camera.npy"))
ZONE_RADIUS = float(os.environ.get("ZONE_RADIUS", 45.0))
ZONE_COLOR = {"robot": (60, 220, 60), "human": (255, 200, 60)}   # BGR


def pose_matrix(x, y, z, rx, ry, rz):
    """block_sort.py 와 같은 ZYZ 오일러 규약. 다르면 구역이 엉뚱한 데 그려진다."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def load_zones():
    """로봇구역·인간구역을 (종류, 번호, x, y, z) 목록으로."""
    if not os.path.exists(ZONES_YAML):
        return []
    d = yaml.safe_load(open(ZONES_YAML)) or {}
    out = []
    for key, kind in (("zones", "robot"), ("human_zones", "human")):
        for i, p in (d.get(key) or {}).items():
            out.append((kind, int(i), float(p[0]), float(p[1]), float(p[2])))
    return out


class Viewer(Node):
    def __init__(self, gate, show_mask, show_zone):
        super().__init__("view_blocks")
        from PIL import ImageFont
        self.font = ImageFont.truetype(FONT, 24)
        self.gate, self.show_mask, self.show_zone = gate, show_mask, show_zone
        self.b = CvBridge()
        self.color = self.depth = self.intr = None
        self.create_subscription(
            Image, "/camera/camera/color/image_raw", self._c, 10)
        self.create_subscription(
            Image, "/camera/camera/aligned_depth_to_color/image_raw", self._d, 10)
        self.create_subscription(
            CameraInfo, "/camera/camera/color/camera_info", self._k, 10)

        # 구역을 그리려면 지금 팔 자세를 알아야 한다. 드라이버가 없으면
        # 이 부분만 끄고 나머지는 그대로 쓴다 — 뷰어는 로봇 없이도 쓸모가 있다.
        self.zones = load_zones()
        self.posx = None
        self._pose_cache, self._pose_t = None, 0.0
        try:
            import DR_init
            # DSR 클라이언트는 반드시 **별도 노드**에 둔다. 이 노드에 두면
            # get_current_posx() 가 그리기 콜백 안에서 자기 실행기를 기다리게 돼
            # 그대로 멎는다 (실측: 첫 프레임 뒤 CPU 275%→9% 로 떨어지며 정지).
            #
            # setattr 을 쓰는 이유: 클래스 안에서 `DR_init.__dsr__node = ...` 라고
            # 쓰면 파이썬 이름 맹글링이 `_Viewer__dsr__node` 로 바꿔버려 정작
            # DSR_ROBOT2 는 None 을 읽는다. block_sort.py 는 모듈 최상위라 무사하다.
            self._dsr = rclpy.create_node("view_blocks_dsr", namespace="dsr01")
            setattr(DR_init, "__dsr__id", "dsr01")
            setattr(DR_init, "__dsr__model", "m0609")
            setattr(DR_init, "__dsr__node", self._dsr)
            from DSR_ROBOT2 import get_current_posx
            self.posx = get_current_posx
            self.gripper2cam = np.load(HANDEYE)
            self.get_logger().info(f"구역 표시 ON — {len(self.zones)}개")
        except Exception as e:
            self.show_zone = False
            self.get_logger().warn(f"구역 표시 OFF (로봇 자세를 못 읽음: {e})")
        # 20Hz 로 그리면 색 6개 마스크 + 덩어리 분할이 매 프레임 돌아 CPU 를
        # 440% 까지 쓴다. 검출 노드와 같은 기계에서 경쟁하므로 10Hz 로 낮춘다.
        self.create_timer(float(os.environ.get("VIEW_HZ_PERIOD", 0.1)), self.draw)
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1280, 720)

    def _c(self, m):
        self.color = self.b.imgmsg_to_cv2(m, "bgr8")

    def _d(self, m):
        self.depth = self.b.imgmsg_to_cv2(m, "passthrough")

    def _k(self, m):
        self.intr = (m.k[0], m.k[4], m.k[2], m.k[5])       # fx, fy, ppx, ppy

    def pose_now(self, ttl=0.3):
        """팔 자세. 화면은 20Hz 라 매 프레임 서비스를 부르면 낭비라 잠깐 캐시한다."""
        import time
        if self._pose_cache is not None and time.time() - self._pose_t < ttl:
            return self._pose_cache
        try:
            self._pose_cache = self.posx()[0]
            self._pose_t = time.time()
        except Exception as e:
            self.get_logger().warn(f"자세 읽기 실패: {e}", once=True)
            return None
        return self._pose_cache

    def project(self, pts_base):
        """베이스 좌표들을 지금 팔 자세 기준으로 픽셀에 되꽂는다.

        검출은 픽셀 → 카메라 → 베이스로 간다. 여기서는 그 반대로 간다.
        같은 핀홀 모델과 같은 핸드아이 행렬을 쓰므로, 화면에 그려진 구역은
        로봇이 판정에 쓰는 그 좌표와 같은 자리다.
        """
        if self.intr is None or self.posx is None:
            return None
        cur = self.pose_now()
        if cur is None:
            return None
        base2cam = pose_matrix(*cur) @ self.gripper2cam
        cam2base = np.linalg.inv(base2cam)
        fx, fy, ppx, ppy = self.intr
        out = []
        for x, y, z in pts_base:
            X, Y, Z = (cam2base @ np.array([x, y, z, 1.0]))[:3]
            if Z <= 1.0:                       # 카메라 뒤쪽이면 그릴 수 없다
                out.append(None)
                continue
            out.append((int(round(X * fx / Z + ppx)), int(round(Y * fy / Z + ppy))))
        return out

    def draw_zones(self, img):
        """구역 중심과 반경을 그린다. 반경 원 = '이 안에 있으면 그 구역의 것'."""
        if not self.show_zone or not self.zones:
            return []
        n = 12
        ring = [(np.cos(t), np.sin(t)) for t in np.linspace(0, 2 * np.pi, n, endpoint=False)]
        pts, meta = [], []
        for kind, i, x, y, z in self.zones:
            pts.append((x, y, z))
            meta.append((kind, i, len(pts) - 1))
            pts += [(x + ZONE_RADIUS * c, y + ZONE_RADIUS * s, z) for c, s in ring]
        uv = self.project(pts)
        if uv is None:
            return []

        labels = []
        for kind, i, base_idx in meta:
            c = uv[base_idx]
            circle = uv[base_idx + 1:base_idx + 1 + n]
            col = ZONE_COLOR[kind]
            if all(p is not None for p in circle):
                cv2.polylines(img, [np.array(circle, np.int32)], True, col, 2)
            if c is not None:
                cv2.drawMarker(img, c, col, cv2.MARKER_TILTED_CROSS, 14, 2)
                labels.append((c[0] + 8, c[1] + 6,
                               f"{'로봇' if kind == 'robot' else '인간'}{i}", col))
        return labels

    def find(self, name):
        """ColorModel 과 같은 순서로 마스크를 만들고 후보를 **전부** 돌려준다.

        예전에는 가장 큰 것 하나만 그렸다. 검출 노드가 여러 개를 내놓게 바뀐
        뒤로는 화면과 로봇이 보는 것이 달라져 진단 도구로 쓸모가 없어진다.
        """
        hsv = cv2.cvtColor(self.color, cv2.COLOR_BGR2HSV)
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in COLOR_HSV_RANGES[name]:
            m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # 깊이 게이팅은 ColorModel 것을 그대로 부른다. 여기 복사해 두면
        # 한쪽만 고쳐져 화면과 로봇이 다른 것을 보게 된다.
        if self.gate:
            m = keep_top_face(self.depth, m)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cc = block_candidates(cs, self.color.shape[0] * self.color.shape[1] * MAX_AREA_RATIO)
        return sorted(cc, key=cv2.contourArea, reverse=True), m

    def to_base(self, u, v, z_mm):
        """픽셀+깊이 → 베이스 좌표. 검출 노드가 쓰는 것과 같은 계산이다."""
        if self.intr is None:
            return None
        cur = self.pose_now()
        if cur is None:
            return None
        fx, fy, ppx, ppy = self.intr
        cam = np.array([(u - ppx) * z_mm / fx, (v - ppy) * z_mm / fy, z_mm, 1.0])
        base2cam = pose_matrix(*cur) @ self.gripper2cam
        return (base2cam @ cam)[:3]

    def zone_of(self, xy):
        """그 자리가 어느 구역인지. 어디에도 안 걸리면 None = 프리구역."""
        best, bd = None, ZONE_RADIUS
        for kind, i, x, y, _z in self.zones:
            d = float(np.hypot(xy[0] - x, xy[1] - y))
            if d < bd:
                best, bd = (kind, i), d
        return best

    def draw(self):
        if self.color is None:
            return
        img = self.color.copy()
        masks = np.zeros(img.shape[:2], np.uint8)
        labels = []
        for name, col in DRAW.items():
            cands, m = self.find(name)
            masks |= m
            for c in cands:
                rect = cv2.minAreaRect(c)
                (_, _), (w, h), ang = rect
                if w < h:
                    ang, w, h = ang + 90, h, w
                a = ((ang + 45.0) % 90.0) - 45.0        # 손목 보정량
                # 중심은 로봇과 같은 기준(마스크 무게중심)으로 그린다.
                # minAreaRect 한가운데를 그리면 화면과 판정이 미묘하게 어긋난다.
                m = cv2.moments(c)
                cx, cy = ((m['m10'] / m['m00'], m['m01'] / m['m00'])
                          if m['m00'] > 0 else rect[0])
                z = 0
                if self.depth is not None:
                    p = self.depth[max(0, int(cy) - 4):int(cy) + 5,
                                   max(0, int(cx) - 4):int(cx) + 5]
                    p = p[p > 0]
                    z = int(np.median(p)) if p.size else 0

                # 로봇이 집을 수 있는 것인지 그대로 판정해 보여준다. 구역 위의
                # 블록은 프리구역 탐색에서 제외되므로 흐리게 그려 구분한다.
                zone = None
                base = self.to_base(cx, cy, z) if z else None
                if base is not None:
                    zone = self.zone_of(base)
                if zone is None:
                    tag, tcol, thick = "프리", col, 3
                else:
                    kind, i = zone
                    tag = f"{'로봇' if kind == 'robot' else '인간'}{i}"
                    tcol = tuple(int(v * 0.45) for v in col)     # 제외 대상은 흐리게
                    thick = 1
                cv2.drawContours(img, [np.int0(cv2.boxPoints(rect))], 0, tcol, thick)
                cv2.drawMarker(img, (int(cx), int(cy)),
                               (255, 255, 255) if zone is None else (140, 140, 140),
                               cv2.MARKER_CROSS, 22, 2)
                labels.append((int(cx) - 70, max(int(cy) - h / 2 - 34, 0),
                               f"{name} {a:+.0f}° {z}mm [{tag}]", tcol))
        # 광축 — 2차 검출은 이 십자에 블록을 올리는 것이 목표다
        cv2.drawMarker(img, (img.shape[1] // 2, img.shape[0] // 2),
                       (0, 255, 255), cv2.MARKER_CROSS, 40, 1)
        labels += self.draw_zones(img)
        if self.show_mask:
            img = cv2.addWeighted(img, 1.0,
                                  cv2.cvtColor(masks, cv2.COLOR_GRAY2BGR), 0.35, 0)
        img = self._ko(img, labels)
        cv2.putText(img, f"gate={'ON' if self.gate else 'OFF'}  "
                         f"zone={'ON' if self.show_zone else 'OFF'}  q/g/m/z",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(WIN, img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            raise KeyboardInterrupt
        if k == ord("g"):
            self.gate = not self.gate
        if k == ord("m"):
            self.show_mask = not self.show_mask
        if k == ord("z") and self.posx is not None:
            self.show_zone = not self.show_zone

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
    n = Viewer("--nogate" not in sys.argv, "--mask" in sys.argv,
               "--nozone" not in sys.argv)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
