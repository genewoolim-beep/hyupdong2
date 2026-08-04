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


class ObjectDetectionNode(Node):
    def __init__(self, model_name = 'color'):
        super().__init__('object_detection_node')
        self.img_node = ImgNode()
        self.model = self._load_model(model_name)
        self.intrinsics = self._wait_for_valid_data(
            self.img_node.get_camera_intrinsic, "camera intrinsics"
        )
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
        """이미지를 처리해 객체의 카메라 좌표를 계산합니다."""
        rclpy.spin_once(self.img_node)

        box, score = self.model.get_best_detection(self.img_node, target)
        if box is None or score is None:
            self.get_logger().warn("No detection found.")
            return 0.0, 0.0, 0.0, 0.0
        
        self.get_logger().info(f"Detection: box={box}, score={score}")
        cx, cy = map(int, [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        cz = self._get_depth(cx, cy)
        if cz is None:
            self.get_logger().warn("Depth out of range.")
            return 0.0, 0.0, 0.0, 0.0

        x, y, z = self._pixel_to_camera_coords(cx, cy, cz)
        # 색 모델이면 회전각도 함께 돌려준다. 기울어진 정사각 블록을 고정
        # 각도로 물면 모서리만 잡혀 조일 때 돌아간다. YOLO 는 각도가 없어 0.
        ang = float(getattr(self.model, 'last_angle', 0.0))
        return x, y, z, ang

    def _get_depth(self, x, y):
        """픽셀 좌표의 depth 값을 안전하게 읽어옵니다."""
        frame = self._wait_for_synced_depth()
        try:
            return frame[y, x]
        except IndexError:
            self.get_logger().warn(f"Coordinates ({x},{y}) out of range.")
            return None

    def _wait_for_synced_depth(self):
        """검출에 쓴 컬러 프레임과 촬영 시각이 가까운 깊이 프레임을 기다립니다."""
        color_ns = self.img_node.get_color_frame_stamp()
        gap = None

        for _ in range(MAX_SYNC_RETRY):
            frame = self._wait_for_valid_data(self.img_node.get_depth_frame, "depth frame")
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

    def _wait_for_valid_data(self, getter, description):
        """getter 함수가 유효한 데이터를 반환할 때까지 spin 하며 재시도합니다."""
        data = getter()
        while data is None or (isinstance(data, np.ndarray) and not data.any()):
            rclpy.spin_once(self.img_node)
            self.get_logger().info(f"Retry getting {description}.")
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
