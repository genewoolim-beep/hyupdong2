import os
import time
import sys
from scipy.spatial.transform import Rotation
import numpy as np
import rclpy
from rclpy.node import Node
import DR_init

from od_msg.srv import SrvDepthPosition
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory
from pick_and_place_text.onrobot import RG

package_path = get_package_share_directory("pick_and_place_text")

# for single robot
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
# movej는 관절 속도(deg/s), movel은 직선 속도(mm/s)라 단위가 달라 값을 나눈다
VELOCITY, ACC = 81, 81
L_VELOCITY, L_ACC = 180, 180
BUCKET_POS = [677.44, 536.04, 128.05, 46.81, 139.63, 17.83]

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
dsr_node = rclpy.create_node("rokey_simple_move", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import movej, movel, get_current_posx, mwait, trans
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()

########### Gripper Setup. Do not modify this area ############

GRIPPER_NAME = "rg2"
TOOLCHANGER_IP = "192.168.1.1"
TOOLCHANGER_PORT = "502"
gripper = RG(GRIPPER_NAME, TOOLCHANGER_IP, TOOLCHANGER_PORT)


########### Robot Controller ############


class RobotController(Node):
    def __init__(self):
        super().__init__("pick_and_place")
        self.init_robot()
        self.depth_client = self.create_client(SrvDepthPosition, "/get_3d_position")
        while not self.depth_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("Waiting for depth position service...")
        self.depth_request = SrvDepthPosition.Request()

        self.get_keyword_client = self.create_client(Trigger, "/get_keyword")
        while not self.get_keyword_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("Waiting for get_keyword service...")
        self.get_keyword_request = Trigger.Request()

        self.robot_control()

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords, gripper2cam_path, robot_pos):
        """
        Converts 3D coordinates from the camera coordinate system
        to the robot's base coordinate system.
        """
        gripper2cam = np.load(gripper2cam_path)
        coord = np.append(np.array(camera_coords), 1)  # Homogeneous coordinate

        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)

        # 좌표 변환 (그리퍼 → 베이스)
        base2cam = base2gripper @ gripper2cam
        td_coord = np.dot(base2cam, coord)

        return td_coord[:3]

    def robot_control(self):
        self.get_logger().info("say 'Hello Rokey' and speak what you want to pick up")
        keyword_future = self.get_keyword_client.call_async(self.get_keyword_request)
        rclpy.spin_until_future_complete(self, keyword_future)
        keyword_result = keyword_future.result()
        if not keyword_result or not keyword_result.success:
            self.get_logger().warn(
                keyword_result.message if keyword_result else "no response from get_keyword"
            )
            self.init_robot()
            return

        for user_input in keyword_result.message.split():
            self.depth_request.target = user_input
            self.get_logger().info("call depth position service with yolo")
            depth_future = self.depth_client.call_async(self.depth_request)
            rclpy.spin_until_future_complete(self, depth_future)

            if depth_future.result():
                result = depth_future.result().depth_position.tolist()
                self.get_logger().info(f"Received depth position: {result}")
                if sum(result) == 0:
                    print("No target position")
                    continue

                gripper2cam_path = os.path.join(
                    package_path, "resource", "T_gripper2camera.npy"
                )
                robot_posx = get_current_posx()[0]
                td_coord = self.transform_to_base(result, gripper2cam_path, robot_posx)

                if td_coord[2] and sum(td_coord) != 0:
                    td_coord[2] += -5  # DEPTH_OFFSET
                    td_coord[2] = max(td_coord[2], 2)  # MIN_DEPTH: float = 2.0

                target_pos = list(td_coord[:3]) + robot_posx[3:]

                self.get_logger().info(f"target position: {target_pos}")
                self.pick_and_place_target(target_pos)
                self.init_robot()
        self.init_robot()

    def init_robot(self):
        JReady = [0, 0, 90, 0, 90, 0]
        movej(JReady, vel=VELOCITY, acc=ACC)
        gripper.open_gripper()
        mwait()

    def pick_and_place_target(self, target_pos):
        # delete
        target_pos[3] += 10

        movel(target_pos, vel=L_VELOCITY, acc=L_ACC)
        mwait()
        gripper.close_gripper()

        while gripper.get_status()[0]:
            time.sleep(0.5)

        # trans()에 리스트를 넘기면 좌표가 깨져서 movel이 무시된다. z를 직접 더한다.
        target_pos_up = list(target_pos)
        target_pos_up[2] += 300

        self.get_logger().info(f"[LIFT] 집은 위치 z={target_pos[2]:.1f} -> 목표 z={target_pos_up[2]:.1f}")
        movel(target_pos_up, vel=L_VELOCITY, acc=L_ACC)
        mwait()
        self.get_logger().info(f"[LIFT] 실제 도달 z={get_current_posx()[0][2]:.1f}")

        movel(BUCKET_POS, vel=L_VELOCITY, acc=L_ACC)
        mwait()

        gripper.open_gripper()
        while gripper.get_status()[0]:
            time.sleep(0.5)


def main(args=None):
    node = RobotController()
    while rclpy.ok():
        node.robot_control()
    rclpy.shutdown()
    node.destroy_node()


if __name__ == "__main__":
    main()
