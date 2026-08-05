import os
import time

import numpy as np
import rclpy
from rclpy.node import Node

from od_msg.srv import SrvDepthPosition
from object_detection.realsense import ImgNode
from object_detection.yolo import YoloModel
from object_detection.color_model import ColorModel

# 검출에 쓴 컬러 프레임과 거리를 읽을 깊이 프레임이 이만큼 이상 벌어지면 다시 기다린다.
# 두 프레임이 서로 다른 순간의 것이면, 물체나 팔이 움직이는 동안 엉뚱한 거리를 읽게 된다.
MAX_SYNC_GAP_NS = 100_000_000   # 0.1초
MAX_SYNC_RETRY = 10
# 카메라가 멎었을 때 이만큼 기다렸다 포기한다. 무한히 매달리면 노드가 통째로
# 응답 불능이 된다 (_wait_for_valid_data 주석 참고).
DATA_TIMEOUT = float(os.environ.get("DATA_TIMEOUT", 5.0))


class ObjectDetectionNode(Node):
    def __init__(self, model_name = 'color'):
        super().__init__('object_detection_node')
        self.img_node = ImgNode()
        self.model = self._load_model(model_name)
        # 시작할 때는 넉넉히 기다린다. 카메라 없이 뜨면 어차피 아무것도 못 한다.
        self.intrinsics = self._wait_for_valid_data(
            self.img_node.get_camera_intrinsic, "카메라 내부파라미터", timeout=60.0
        )
        if self.intrinsics is None:
            raise SystemExit("카메라 내부파라미터를 못 받았습니다. "
                             "RealSense 가 켜져 있고 USB 3.0 인지 확인하세요.")
        self.create_service(
            SrvDepthPosition,
            'get_3d_position',
            self.handle_get_depth
        )
        self.get_logger().info("ObjectDetectionNode initialized.")

    def _load_model(self, name):
        """모델 이름에 따라 인스턴스를 반환합니다."""
        name = name.lower()
        if name == 'yolo':
            return YoloModel()
        if name == 'color':
            return ColorModel()
        raise ValueError(f"Unsupported model: {name}")

    def handle_get_depth(self, request, response):
        """클라이언트 요청을 처리해 3D 좌표를 반환합니다."""
        self.get_logger().info(f"Received request: {request}")
        coords = self._compute_position(request.target)
        response.depth_position = [float(x) for x in coords]
        return response

    def _compute_position(self, target):
        """그 색으로 보이는 것을 전부 찾아 (x, y, z, 각도) 를 이어붙여 돌려준다.

        하나만 돌려주면 같은 색 블록이 여럿일 때 부르는 쪽이 고를 수가 없다.
        실측 2026-08-04: 인간구역의 노란 블록을 읽으려는데 165mm 떨어진 다른
        노란 블록이 더 크게 잡혀 그쪽이 왔다. 위치로 고르려면 후보가 다 필요하다.

        믿을 만한 순으로 정렬돼 있으므로 앞 4개만 읽으면 예전과 같은 값이다 —
        robot_control 처럼 하나만 쓰는 쪽은 고치지 않아도 그대로 동작한다.
        못 찾으면 예전처럼 0 네 개를 돌려준다.
        """
        rclpy.spin_once(self.img_node)

        if hasattr(self.model, 'get_all_detections'):
            found = self.model.get_all_detections(self.img_node, target)
        else:                                   # YOLO 는 하나만 낸다
            box, score = self.model.get_best_detection(self.img_node, target)
            found = ([] if box is None or score is None else
                     [(box, 0.0, score,
                       (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)])
        if not found:
            self.get_logger().warn("No detection found.")
            return 0.0, 0.0, 0.0, 0.0

        out = []
        for box, ang, score, cx, cy in found:
            # 중심은 마스크 무게중심이다(소수점 유지). 축정렬 박스 한가운데를 쓰면
            # 한쪽으로 삐져나온 화소 몇 개에 중심이 끌려간다 — 35mm 블록이 약
            # 86픽셀이라 5픽셀만 밀려도 2mm 오차다.
            cz = self._get_depth(int(round(cx)), int(round(cy)))
            if cz is None:
                self.get_logger().warn(f"Depth out of range at ({cx:.0f}, {cy:.0f}) — skipped.")
                continue
            x, y, z = self._pixel_to_camera_coords(cx, cy, cz)
            # 회전각도 함께 돌려준다. 기울어진 정사각 블록을 고정 각도로 물면
            # 모서리만 잡혀 조일 때 돌아간다. YOLO 는 각도가 없어 0.
            out += [x, y, z, float(ang)]
            self.get_logger().info(
                f"Detection: box={box}, score={score:.2f}, angle={ang:.0f}")

        if not out:
            self.get_logger().warn("No detection with valid depth.")
            return 0.0, 0.0, 0.0, 0.0
        self.get_logger().info(f"{len(out) // 4} detection(s) for '{target}'")
        return out

    def _get_depth(self, x, y, half=3):
        """그 자리 둘레 작은 창에서 깊이의 중앙값을 읽습니다.

        한 픽셀만 읽으면 그 화소가 깊이 구멍(0)이거나 잡음이면 그대로 속습니다.
        블록 윗면은 평평하므로 둘레 몇 픽셀의 중앙값이 훨씬 안정적입니다.
        0(측정 실패)은 빼고 셉니다.
        """
        frame = self._wait_for_synced_depth()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            self.get_logger().warn(f"Coordinates ({x},{y}) out of range.")
            return None
        patch = frame[max(0, y - half):y + half + 1, max(0, x - half):x + half + 1]
        vals = patch[patch > 0]
        if vals.size == 0:
            return None
        return float(np.median(vals))

    def _wait_for_synced_depth(self):
        """검출에 쓴 컬러 프레임과 촬영 시각이 가까운 깊이 프레임을 기다립니다."""
        color_ns = self.img_node.get_color_frame_stamp()
        gap = None

        frame = None
        for _ in range(MAX_SYNC_RETRY):
            frame = self._wait_for_valid_data(self.img_node.get_depth_frame, "깊이 프레임")
            if frame is None:
                return None
            depth_ns = self.img_node.get_depth_frame_stamp()
            if color_ns is None or depth_ns is None:
                return frame

            gap = abs(color_ns - depth_ns)
            if gap <= MAX_SYNC_GAP_NS:
                return frame
            rclpy.spin_once(self.img_node)

        # 계속 안 맞으면 멈춰 서는 것보다 최신 프레임으로 진행하되, 어긋난 정도는 남긴다
        self.get_logger().warn(
            f"컬러/깊이 프레임 시각이 {gap / 1e6:.0f}ms 벌어진 채로 거리 계산을 진행합니다."
        )
        return frame

    def _wait_for_valid_data(self, getter, description, timeout=DATA_TIMEOUT):
        """유효한 데이터를 기다린다. 못 받으면 None 을 돌려준다.

        예전에는 조건 없는 while 이라 카메라가 멎으면 **영원히** 돌았다.
        실측 2026-08-05: RealSense 가 USB 2.0 으로 재연결돼 프레임이 끊기자
        이 노드가 요청 하나를 받은 채로 멈췄고, 부르는 쪽은 매 명령마다
        타임아웃을 먹었다. 카메라가 죽었으면 매달리지 말고 알려주는 편이 낫다.
        """
        t0 = time.time()
        data = getter()
        while data is None or (isinstance(data, np.ndarray) and not data.any()):
            if time.time() - t0 > timeout:
                self.get_logger().error(
                    f"{description} 를 {timeout:.0f}초 동안 못 받았습니다. "
                    "카메라가 살아 있는지 확인하세요 (RealSense 는 USB 3.0 이어야 합니다).")
                return None
            rclpy.spin_once(self.img_node, timeout_sec=0.1)
            data = getter()
        return data

    def _pixel_to_camera_coords(self, x, y, z):
        """픽셀 좌표와 intrinsics를 이용해 카메라 좌표계로 변환합니다."""
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        ppx = self.intrinsics['ppx']
        ppy = self.intrinsics['ppy']
        return (
            (x - ppx) * z / fx,
            (y - ppy) * z / fy,
            z
        )


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
