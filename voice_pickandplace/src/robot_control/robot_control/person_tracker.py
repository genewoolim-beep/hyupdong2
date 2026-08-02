"""J1(베이스 회전)만 사용해 사람 한 명을 계속 바라보게 하는 노드.

팔은 READY 자세에서 펴지 않는다. 방위각 오차만큼 J1을 돌려 카메라가
대상을 정면에 두게 한다. 대상은 tracker ID로 고정한다.

실시간성 관련 설계 (측정값 기준):
  - 이 드라이버의 ROS 서비스는 왕복 100ms대다. get_current_posx/posj 는
    100Hz 로 나오는 /dsr01/joint_states + URDF 정기구학으로 대체했다.
  - amovej 도 서비스 호출이라 비전 루프에서 부르면 루프가 막힌다.
    별도 스레드로 분리해 비전은 카메라 속도(30Hz)로 계속 돌게 한다.
  - 디버그 영상은 원본 1280x720(2.76MB/frame) 대신 축소해서 발행한다.

추적 상황은 /person_tracker/image 로 발행한다 (RViz Image 표시용).
"""

import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation

import DR_init
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image, CameraInfo, JointState
from cv_bridge import CvBridge
from ultralytics import YOLO

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

# J5는 손목 특이점(0도)을 피해 15도 기울인다. J1만 추적으로 바뀐다.
READY_J = [0.0, 0.0, 90.0, 0.0, 15.0, 0.0]

J1_LIMIT = 70.0        # 안전을 위해 좌우 회전 범위를 제한한다
DEADBAND_DEG = 3.0     # 이 안쪽 오차는 무시해 떨림을 막는다
GAIN = 0.6             # 오차의 몇 배를 따라갈지 (1.0이면 즉시 정렬)
VEL, ACC = 15, 15
LOST_TIMEOUT = 2.0     # 대상을 이 시간 이상 놓치면 재획득한다
IMGSZ = 640            # 추론 해상도
IDLE_VEL = 0.02        # 관절 속도가 이보다 작으면 정지로 본다 (rad/s)
DEBUG_WIDTH = 640      # RViz 로 보낼 영상 가로 크기 (원본 1280의 1/4 데이터)

YOLO_MODEL = "/home/gene/Downloads/yolov8n.pt"
PERSON_CLASS = 0

# m0609.urdf 의 revolute 조인트 origin (xyz[m], rpy[rad])
URDF_JOINTS = [
    ([0.0, 0.0, 0.1345], [0.0, 0.0, 0.0]),
    ([0.0, 0.0062, 0.0], [0.0, -1.571, -1.571]),
    ([0.411, 0.0, 0.0], [0.0, 0.0, 1.571]),
    ([0.0, -0.368, 0.0], [1.571, 0.0, 0.0]),
    ([0.0, 0.0, 0.0], [-1.571, 0.0, 0.0]),
    ([0.0, -0.121, 0.0], [1.571, 0.0, 0.0]),
]

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
dsr_node = rclpy.create_node("person_tracker_node", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import movej, amovej, get_current_posx, mwait
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()


def fk_flange(q_deg):
    """관절각(도)으로부터 base->flange 변환을 구한다."""
    T = np.eye(4)
    for (xyz, rpy), q in zip(URDF_JOINTS, q_deg):
        A = np.eye(4)
        A[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        A[:3, 3] = xyz
        B = np.eye(4)
        B[:3, :3] = Rotation.from_euler("z", q, degrees=True).as_matrix()
        T = T @ A @ B
    return T


class PersonTracker(Node):
    def __init__(self):
        super().__init__("person_tracker")
        self.bridge = CvBridge()
        self.frame = None
        self.frame_seq = 0           # 새 프레임이 왔는지 판별용
        self.intrinsics = None
        self.joints = None           # 도 단위, joint_1..joint_6 순
        self.joint_vel = None
        self.R_offset = np.eye(3)

        self.create_subscription(
            Image, "/camera/camera/color/image_raw", self._color_cb, 1)
        self.create_subscription(
            CameraInfo, "/camera/camera/color/camera_info", self._info_cb, 1)
        self.create_subscription(
            JointState, f"/{ROBOT_ID}/joint_states", self._joint_cb, 1)
        self.debug_pub = self.create_publisher(Image, "/person_tracker/image", 1)

        self.gripper2cam = np.load(os.path.join(
            get_package_share_directory("robot_control"),
            "resource", "T_gripper2camera.npy"))

        self.model = YOLO(YOLO_MODEL)
        self.target_id = None
        self.last_seen = 0.0

        # 모션 스레드와 공유하는 목표값
        self._cmd_lock = threading.Lock()
        self._cmd_j1 = None
        self._running = True

    # ---------- 콜백 ----------
    def _color_cb(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.frame_seq += 1

    def _info_cb(self, msg):
        self.intrinsics = {"fx": msg.k[0], "fy": msg.k[4],
                           "ppx": msg.k[2], "ppy": msg.k[5]}

    def _joint_cb(self, msg):
        """joint_states 는 이름 순서가 뒤섞여 오므로 반드시 이름으로 매핑한다."""
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            order = [idx[f"joint_{k}"] for k in range(1, 7)]
        except KeyError:
            return
        self.joints = [np.degrees(msg.position[i]) for i in order]
        if len(msg.velocity) == len(msg.position):
            self.joint_vel = [abs(msg.velocity[i]) for i in order]

    # ---------- 기구학 ----------
    def calibrate_offset(self):
        """FK 플랜지 자세와 posx TCP 자세의 회전 차이를 기동 시 1회만 구한다."""
        posx = get_current_posx()[0]
        R_posx = Rotation.from_euler(
            "ZYZ", [posx[3], posx[4], posx[5]], degrees=True).as_matrix()
        R_fk = fk_flange(self.joints)[:3, :3]
        self.R_offset = R_fk.T @ R_posx
        ang = np.degrees(np.arccos(
            np.clip((np.trace(self.R_offset) - 1) / 2, -1, 1)))
        self.get_logger().info(f"TCP rotation offset = {ang:.2f} deg")

    def base2cam_R(self, joints):
        R_tcp = fk_flange(joints)[:3, :3] @ self.R_offset
        return R_tcp @ self.gripper2cam[:3, :3]

    def is_idle(self):
        if self.joint_vel is None:
            return True
        return max(self.joint_vel) < IDLE_VEL

    # ---------- 대상 선택 ----------
    def pick_target(self, result):
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return None
        people = [
            (int(i), b) for i, b, c in
            zip(boxes.id.tolist(), boxes.xyxy.tolist(), boxes.cls.tolist())
            if int(c) == PERSON_CLASS
        ]
        if not people:
            return None
        for tid, box in people:
            if tid == self.target_id:
                self.last_seen = time.time()
                return box
        if self.target_id is not None and time.time() - self.last_seen < LOST_TIMEOUT:
            return None
        tid, box = max(people, key=lambda p: (p[1][2] - p[1][0]) * (p[1][3] - p[1][1]))
        self.target_id = tid
        self.last_seen = time.time()
        self.get_logger().info(f"target acquired: id={tid}")
        return box

    def azimuth_error(self, box, joints):
        """대상 방향과 카메라 정면 사이의 수평 각도차(도). 깊이 불필요."""
        u = (box[0] + box[2]) / 2.0
        v = box[1] + (box[3] - box[1]) * 0.3      # 가슴 높이

        i = self.intrinsics
        ray_cam = np.array([(u - i["ppx"]) / i["fx"],
                            (v - i["ppy"]) / i["fy"], 1.0])
        ray_cam /= np.linalg.norm(ray_cam)

        R = self.base2cam_R(joints)
        ray_base = R @ ray_cam
        view_base = R @ np.array([0.0, 0.0, 1.0])
        az_t = np.degrees(np.arctan2(ray_base[1], ray_base[0]))
        az_v = np.degrees(np.arctan2(view_base[1], view_base[0]))
        return (az_t - az_v + 180.0) % 360.0 - 180.0

    # ---------- 모션 (별도 스레드) ----------
    def motion_loop(self):
        """amovej 는 서비스 호출이라 100ms대가 걸린다. 비전 루프와 분리한다."""
        while self._running and rclpy.ok():
            with self._cmd_lock:
                target = self._cmd_j1
                self._cmd_j1 = None
            if target is None or not self.is_idle():
                time.sleep(0.01)
                continue
            try:
                amovej([target] + READY_J[1:], vel=VEL, acc=ACC)
            except Exception as e:  # 드라이버가 거부해도 추적은 계속한다
                self.get_logger().warn(f"amovej failed: {e}")
            time.sleep(0.02)

    def request_j1(self, value):
        with self._cmd_lock:
            self._cmd_j1 = value

    # ---------- 디버그 영상 ----------
    def publish_debug(self, frame, result, target_box, err, j1, fps):
        img = frame
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for tid, b, c in zip(boxes.id.tolist(), boxes.xyxy.tolist(),
                                 boxes.cls.tolist()):
                if int(c) != PERSON_CLASS:
                    continue
                x1, y1, x2, y2 = [int(v) for v in b]
                hit = int(tid) == self.target_id
                color = (0, 255, 0) if hit else (140, 140, 140)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if hit else 1)
                cv2.putText(img, f"TARGET id={int(tid)}" if hit else f"id={int(tid)}",
                            (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)
        if target_box is not None:
            cx = int((target_box[0] + target_box[2]) / 2)
            cy = int(target_box[1] + (target_box[3] - target_box[1]) * 0.3)
            cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 30, 3)

        txt = f"J1={j1:+.1f}deg  {fps:.1f}FPS"
        txt += f"  err={err:+.1f}deg" if err is not None else "  no target"
        cv2.putText(img, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2)

        # RViz 부담을 줄이려고 축소해서 보낸다
        if img.shape[1] > DEBUG_WIDTH:
            h = int(img.shape[0] * DEBUG_WIDTH / img.shape[1])
            img = cv2.resize(img, (DEBUG_WIDTH, h), interpolation=cv2.INTER_AREA)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(img, encoding="bgr8"))

    # ---------- 메인 루프 ----------
    def wait_for_data(self):
        while self.frame is None or self.intrinsics is None or self.joints is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.get_logger().info("waiting for camera/joints...",
                                   throttle_duration_sec=2.0)

    def spin_tracking(self):
        self.get_logger().info("tracking started (Ctrl+C to stop)")
        fps = 0.0
        last_seq = -1
        while rclpy.ok():
            # 새 프레임이 올 때까지만 대기한다. 같은 프레임을 두 번 추론하지 않는다.
            while self.frame_seq == last_seq and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.01)
            last_seq = self.frame_seq

            t0 = time.time()
            frame = self.frame            # 영상과 관절각을 같은 시점으로 묶는다
            joints = list(self.joints)

            result = self.model.track(
                frame, persist=True, classes=[PERSON_CLASS],
                conf=0.4, imgsz=IMGSZ, verbose=False)[0]

            box = self.pick_target(result)
            err = None
            if box is not None:
                err = self.azimuth_error(box, joints)
                if abs(err) >= DEADBAND_DEG:
                    self.request_j1(float(np.clip(
                        joints[0] + err * GAIN, -J1_LIMIT, J1_LIMIT)))

            self.publish_debug(frame, result, box, err, joints[0], fps)
            dt = time.time() - t0
            fps = 0.8 * fps + 0.2 * (1.0 / dt if dt > 0 else 0.0)


def main(args=None):
    node = PersonTracker()

    node.get_logger().info(f"moving to READY {READY_J}")
    movej(READY_J, vel=10, acc=10)
    mwait()
    node.get_logger().info("READY reached, camera facing forward")

    node.wait_for_data()
    node.calibrate_offset()

    motion = threading.Thread(target=node.motion_loop, daemon=True)
    motion.start()

    try:
        node.spin_tracking()
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
