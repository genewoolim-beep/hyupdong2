# ros2 service call /get_keyword std_srvs/srv/Trigger "{}"
#
# voice_processing 의 /get_keyword 를 대체하는 수어 버전.
# STT/LLM 대신 sign_control 에서 학습한 제스처 분류기로 물체 이름 하나를 인식해 돌려준다.
# robot_control 쪽은 이 노드가 voice_processing 대신 떠 있기만 하면 코드 변경 없이 그대로 동작한다.

import os

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from sign_processing.gesture_recognizer import SignRecognizer

# sign_control 은 voice_pickandplace 와 별도로 관리되는 저장소(hyupdong2) 폴더라
# ROS2 share 디렉터리 밖에 있다. 학습된 모델(model.pt)과 mediapipe .task 파일의
# 단일 소스를 유지하기 위해 복사하지 않고 그대로 참조한다.
# realpath를 쓰는 이유: `colcon build --symlink-install`이면 install 공간의 이 파일은
# 소스 트리로 가는 심볼릭 링크라 realpath로 풀어야 상대경로가 맞는다.
# symlink-install이 아니라면 소스가 복사되므로 이 상대경로는 깨진다 —
# 그 경우 SIGN_CONTROL_DIR 환경변수로 직접 지정한다.
HERE = os.path.dirname(os.path.realpath(__file__))
DEFAULT_SIGN_CONTROL_DIR = os.path.normpath(
    os.path.join(HERE, "..", "..", "..", "..", "sign_control")
)
SIGN_CONTROL_DIR = os.environ.get("SIGN_CONTROL_DIR", DEFAULT_SIGN_CONTROL_DIR)
MODEL_PATH = os.path.join(SIGN_CONTROL_DIR, "sign_data", "model.pt")
HAND_TASK = os.path.join(SIGN_CONTROL_DIR, "hand_landmarker.task")
POSE_TASK = os.path.join(SIGN_CONTROL_DIR, "pose_landmarker_lite.task")
CAM_INDEX = int(os.environ.get("SIGN_CAM", 1))
SHOW_WINDOW = os.environ.get("SIGN_SHOW_WINDOW", "1") != "0"

# sign_control 에 녹화된 수어 라벨 중 로봇이 실제로 집을 수 있는 물체
# (object_detection 의 class_name_tool.json 클래스)만 매핑한다.
# 색깔/동사/구역 라벨은 이번 연동 범위(물체 인식만)에서 제외한다.
# drill/pliers/screwdriver/wrench 에 대한 수어를 추가로 녹화하면 여기에 한 줄씩 추가하면 된다.
GLOSS_TO_OBJECT = {
    "망치": "hammer",
}


class GetKeyword(Node):
    def __init__(self):
        super().__init__("get_keyword_node")
        self.get_logger().info(f"수어 모델 로딩: {MODEL_PATH}")
        self.recognizer = SignRecognizer(
            MODEL_PATH, HAND_TASK, POSE_TASK, CAM_INDEX, show_window=SHOW_WINDOW
        )
        self.get_keyword_srv = self.create_service(
            Trigger, "get_keyword", self.get_keyword
        )
        self.get_logger().info("GetKeyword(sign) initialized.")
        self.get_logger().info("wait for client's request...")

    def get_keyword(self, request, response):
        self.get_logger().info("카메라 앞에서 집을 물체를 수어로 표현하세요.")
        while True:
            label, prob = self.recognizer.recognize_one()
            obj = GLOSS_TO_OBJECT.get(label)
            if obj is None:
                self.get_logger().info(
                    f"'{label}' ({prob * 100:.0f}%) 는 물체 어휘가 아닙니다. 다시 시도하세요."
                )
                continue
            self.get_logger().warn(f"인식된 물체: {label} -> {obj} ({prob * 100:.0f}%)")
            response.success = True
            response.message = obj
            return response


def main():
    rclpy.init()
    node = GetKeyword()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 카메라는 서비스 호출마다 열고 닫지 않는다(창이 깜빡이고 여는 데만
        # 1초 넘게 걸린다). 대신 노드가 내려갈 때 여기서 한 번 정리한다.
        node.recognizer.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
