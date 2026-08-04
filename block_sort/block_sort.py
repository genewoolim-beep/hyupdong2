#!/usr/bin/env python3
"""
관측 → 검출 → 파지 → 구역 배치   (M0609 + RG2 + RealSense)

  python3 block_sort.py observe            관측 자세로 가서 무엇이 보이는지만 보고
  python3 block_sort.py pick <대상> <구역>   검출해서 집어 그 구역에 놓기
  python3 block_sort.py run                텍스트로 반복 입력
  python3 block_sort.py home               초기 자세 복귀

기존 음성 pick&place 파이프라인을 그대로 쓴다. 바뀐 것은 두 가지뿐이다.
  · 놓는 자리가 BUCKET_POS 한 곳 → zones.yaml 의 구역 1~4
  · 입력이 음성(/get_keyword) → 텍스트 인자

블록 전용 YOLO 모델이 아직 없으므로, 지금은 과일/공구 클래스를 대역으로
써서 파이프라인을 검증한다. 모델이 생기면 대상 이름만 바꾸면 된다.
"""
import os
import sys
import time

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from scipy.spatial.transform import Rotation

import DR_init
from od_msg.srv import SrvDepthPosition

ROBOT_ID, ROBOT_MODEL = "dsr01", "m0609"

# 첫 실주행이라 기존 값(81 / 150)보다 낮춰 둔다. 안정되면 올린다.
VEL_J, ACC_J = 50, 50          # 관절 deg/s
VEL_L, ACC_L = 80, 80          # 직선 mm/s

JREADY = [0, 0, 90, 0, 90, 0]
GRASP_RETRY = 2

# 파지 높이는 깊이 센서가 아니라 티칭값을 쓴다.
#   판이 평평하고(구역 z 편차 0.5mm) 블록 높이가 일정하므로 z 는 상수다.
#   깊이 측정은 편차가 3mm 나고, 게다가 카메라가 보는 건 '윗면'이라
#   그대로 쓰면 블록 위 허공을 물게 된다.
# 실측: 블록 윗면 base z ≈ +3.9, 티칭 구역 z ≈ -13.0 → 차이 16.9mm
#       35mm 블록의 정확히 중앙이다.
TOP_TO_GRASP = 16.9            # 윗면에서 이만큼 내려가면 파지 높이
Z_SANITY = 15.0                # 측정 윗면이 예상에서 이 이상 벗어나면 경고
REFINE_MAX = 40.0              # 보정량이 이보다 크면 다른 블록으로 보고 거부
# 카메라를 블록 바로 위로 옮겨 다시 보는 일을 수렴할 때까지 되풀이한다.
# 옮길 목표를 '오차가 있는 직전 결과' 로 잡으므로 한 번에 광축 위에 서지
# 못한다. 실측에서 1회 후 6~18mm 가 남았다. 회를 거듭하면 그 잔차가 줄어든다.
# 실측: 1회 보정으로 광축 이탈 212mm → 10mm. 그 뒤로는 더 나아지지 않고
# 오히려 흔들린다(10 → 15.6mm). 남은 잔차는 시점 왜곡이 아니라 검출 잡음이라
# 반복해도 줄지 않는다. 그래서 '나아질 때만' 한 번 더 가고, 아니면 멈춘다.
REFINE_ITERS = 3               # 최대 반복 횟수
REFINE_TOL = 12.0              # 광축에서 이 안(mm)이면 충분히 수직으로 본 것
# 수렴 뒤에도 4mm 안팎이 남는다. 손목 각도(0/45/90°)와 무관하게 같은 크기라
# 계통 오차가 아니라 검출 재현성의 바닥이다 — 깊이가 369↔394 로 흔들리고
# 마스크 경계가 조명에 따라 미세하게 달라지는 것이 누적된 값이다.
# 잡음이므로 여러 번 재서 중앙값을 쓰면 √N 만큼 줄어든다.
DETECT_SAMPLES = 5             # 최종 자세에서 이만큼 재어 중앙값을 쓴다

# 파지 중심 보정 (mm, base 기준). 핸드아이 잔차와 TCP 정의 차이 때문에
# 계산 위치와 실제 손가락 중심이 조금 어긋난다. calib 모드로 실측해 채운다.
#   측정법: 집었다가 같은 자리에 도로 놓고 다시 검출한다.
#           파지 중심이 e 만큼 밀려 있으면 블록은 P+e 에 놓이므로 e 를 얻는다.
CALIB_OUTLIER = 30.0           # calib 에서 이보다 큰 오차는 오검출로 보고 버린다
# 파지 중심 보정은 두 성분으로 나뉜다.  offset(θ) = OFFSET_BASE + R(θ)·OFFSET_TOOL
#   OFFSET_BASE  비전 사슬(내부파라미터·깊이·핸드아이)에서 오는 것. 손목과 무관.
#   OFFSET_TOOL  TCP 와 실제 손가락 중심의 차이. 손목과 함께 돈다.
# 한 각도에서만 재면 둘을 못 가른다. center 모드로 0° 와 90° 두 번 재서 분리했다.
#   실측 2026-08-04   0° (4.70, 25.70)   90° (4.70, 13.70)
# 고정값 하나만 쓰면 45°/135° 에서 6.5mm, 0° 에서 12mm 어긋난다.
OFFSET_BASE = [10.70, 19.70]
OFFSET_TOOL = [-6.00, 6.00]

# 기본 파지 방향. 관측 자세 그대로면 손가락이 base x 로 닫힌다.
# 90 을 주면 base y 로 닫힌다.
GRASP_ROT = 90.0
# 블록이 기울어져 있으면 그만큼 손목을 더 돌려 '면' 을 물게 한다.
# 45° 돌아간 정사각 블록을 고정 각도로 물면 모서리만 잡혀 조일 때 돌아간다.
# 정사각은 90° 대칭이므로 -45~+45 로 접어 회전량을 최소화한다.
ALIGN_TO_BLOCK = True

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP, TOOLCHARGER_PORT = "192.168.1.1", "502"

HERE = os.path.dirname(os.path.abspath(__file__))
ZONES_YAML = os.path.join(HERE, "zones.yaml")
CENTER_YAML = os.path.join(HERE, "center_calib.yaml")   # 손목각별 실측 보정
# 핸드아이 행렬. 저장소 안의 robot_control 것을 먼저 쓰고, 없으면 환경변수.
_HE_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "voice_pickandplace", "src", "robot_control", "resource",
    "T_gripper2camera.npy")
HANDEYE = os.environ.get("HANDEYE", _HE_DEFAULT)

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
rclpy.init()
_dsr = rclpy.create_node("block_sort_dsr", namespace=ROBOT_ID)
DR_init.__dsr__node = _dsr

try:
    from DSR_ROBOT2 import movej, movel, get_current_posx, mwait
except ImportError as e:
    sys.exit(f"DSR_ROBOT2 임포트 실패: {e}\n로봇 드라이버가 떠 있는지 확인하세요.")

sys.path.insert(0, HERE)
from onrobot import RG                                    # noqa: E402

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)


def load_zones():
    if not os.path.exists(ZONES_YAML):
        sys.exit(f"{ZONES_YAML} 이 없습니다. teach_zones.py teach 를 먼저 실행하세요.")
    d = yaml.safe_load(open(ZONES_YAML))
    if not d.get("zones"):
        sys.exit("zones.yaml 에 구역이 없습니다.")
    return d


def wait_gripper():
    while gripper.get_status()[0]:
        time.sleep(0.3)


def grasped():
    """RG2 상태의 1번 비트가 'grip detected'. 빈손이면 0이다."""
    return bool(gripper.get_status()[1])


class BlockSort(Node):
    def __init__(self):
        super().__init__("block_sort")
        self.cfg = load_zones()
        self.lift = self.cfg.get("lift_height", 150)
        self.cli = self.create_client(SrvDepthPosition, "/get_3d_position")
        self.req = SrvDepthPosition.Request()
        self.gripper2cam = np.load(HANDEYE)
        # 블록도 구역과 같은 판 위에 있으므로 파지 높이는 구역 z 와 같다.
        zs = [p[2] for p in self.cfg["zones"].values()]
        self.pick_z = float(np.mean(zs))
        self.expect_top = self.pick_z + TOP_TO_GRASP
        self.last_top = None       # 마지막으로 측정한 블록 윗면 z (진단용)
        self.last_angle = 0.0      # 마지막 검출의 블록 기울기 (도)
        self.last_cam = [0.0, 0.0, 0.0]   # 마지막 검출의 카메라 좌표
        self.last_rot = GRASP_ROT         # 마지막 검출에서 쓴 손목 각도

    # ── 좌표 변환 (기존 robot_control.py 와 동일) ──
    @staticmethod
    def pose_matrix(x, y, z, rx, ry, rz):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T[:3, 3] = [x, y, z]
        return T

    def to_base(self, cam_xyz, robot_posx):
        base2gripper = self.pose_matrix(*robot_posx)
        base2cam = base2gripper @ self.gripper2cam
        return (base2cam @ np.append(np.array(cam_xyz), 1))[:3]

    # ── 검출 ──
    def detect(self, target, refine=True):
        """1차로 찾고, 그 위로 카메라를 옮겨 2차로 정밀 측정한다.

        색 임계값은 블록의 윗면과 옆면을 구분하지 못한다. 비스듬히 볼수록
        옆면이 마스크에 붙어 박스가 커지고, 중심이 그 절반만큼 카메라
        반대쪽으로 밀린다. 실측에서 광축 근처 49.5mm, 가장자리 59.6mm 로
        10mm 차이가 났다 — 중심 오차 5mm 다.

        블록 바로 위에서 보면 광축각이 0 에 가까워 옆면이 안 보인다.
        원인 자체가 사라지므로 기하 보정보다 확실하다.
        """
        cur = self._detect_once(target)
        if cur is None or not refine:
            return cur

        best, best_off = cur, float(np.hypot(self.last_cam[0], self.last_cam[1]))
        for it in range(1, REFINE_ITERS + 1):
            if best_off <= REFINE_TOL:
                self.get_logger().info(f"광축에서 {best_off:.1f}mm — 수렴")
                break
            posx = get_current_posx()[0]
            R = Rotation.from_euler("ZYZ", posx[3:], degrees=True).as_matrix()
            cam_off = R @ self.gripper2cam[:3, 3]    # 카메라가 그리퍼에서 떨어진 양
            view = list(posx)
            view[0] = best[0] - cam_off[0]           # 카메라를 블록 바로 위로
            view[1] = best[1] - cam_off[1]
            self.get_logger().info(
                f"[{it}] 광축 {best_off:.1f}mm — 블록 위로 이동 "
                f"({view[0]:.0f}, {view[1]:.0f})")
            try:
                movel(view, vel=VEL_L, acc=ACC_L)
                mwait()
            except Exception as e:
                self.get_logger().warn(f"이동 실패({e}) — 직전 결과 사용")
                break
            nxt = self._detect_once(target)
            if nxt is None:
                self.get_logger().warn("재검출 실패 — 직전 결과 사용")
                break
            d = float(np.hypot(nxt[0] - best[0], nxt[1] - best[1]))
            if d > REFINE_MAX:
                # 같은 색이 여럿이면 시점이 바뀐 뒤 다른 개체를 잡을 수 있다.
                self.get_logger().warn(
                    f"보정량 {d:.1f}mm 가 한계({REFINE_MAX:.0f})를 넘습니다 — "
                    "다른 블록으로 판단하고 직전 결과 사용")
                break
            off = float(np.hypot(self.last_cam[0], self.last_cam[1]))
            if off >= best_off:
                # 잡음 구간에 들어왔다. 더 가봐야 나빠지기만 한다.
                self.get_logger().info(
                    f"[{it}] 광축 {off:.1f}mm — 나아지지 않아 중단 "
                    f"(최선 {best_off:.1f}mm 유지)")
                break
            self.get_logger().info(f"[{it}] 보정 {d:.1f}mm  광축 {off:.1f}mm")
            best, best_off = nxt, off

        # 수렴한 자세에서 여러 번 재어 잡음을 눌러준다. 이동은 없으므로 싸다.
        if DETECT_SAMPLES > 1:
            xs, ys, angs = [best[0]], [best[1]], [self.last_angle]
            for _ in range(DETECT_SAMPLES - 1):
                s = self._detect_once(target, quiet=True)
                if s is None:
                    continue
                if np.hypot(s[0] - best[0], s[1] - best[1]) > REFINE_MAX:
                    continue                      # 다른 블록을 본 표본은 버린다
                xs.append(s[0]); ys.append(s[1]); angs.append(self.last_angle)
            if len(xs) > 1:
                mx, my = float(np.median(xs)), float(np.median(ys))
                self.get_logger().info(
                    f"{len(xs)}회 평균 — 산포 ({np.std(xs):.1f}, {np.std(ys):.1f})mm "
                    f"→ ({mx:.1f}, {my:.1f})")
                self.last_angle = float(np.median(angs))
                # 자세(손목 각도)는 마지막 검출의 것을 그대로 두고 x,y 만 갈아끼운다.
                # 표본 간 각도 차이는 1° 안팎이라 다시 만들 실익이 없다.
                best = [mx, my] + list(best[2:])
        return best

    def _detect_once(self, target, quiet=False):
        """지금 자세에서 한 번 검출한다. 못 찾으면 None."""
        if not self.cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/get_3d_position 서비스가 없습니다. "
                                    "object_detection 노드를 먼저 띄우세요.")
            return None
        self.req.target = target
        fut = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=20.0)
        if fut.result() is None:
            self.get_logger().error("서비스 응답 없음")
            return None

        cam = list(fut.result().depth_position)
        self.last_angle = float(cam[3]) if len(cam) > 3 else 0.0
        self.last_cam = cam[:3]
        if not quiet:
            self.get_logger().info(f"카메라 좌표 {[round(v,1) for v in cam]}")
        if sum(cam[:3]) == 0:
            self.get_logger().warn(f"'{target}' 을(를) 찾지 못했습니다.")
            return None

        # 영상과 로봇 자세는 같은 시점이어야 한다. 검출 직후 바로 읽는다.
        posx = get_current_posx()[0]
        xyz = self.to_base(cam[:3], posx)
        top_z = self.last_top = float(xyz[2])
        # x,y 는 검출값, z 는 티칭값. 이유는 상단 주석 참고.
        xyz[2] = self.pick_z
        d = top_z - self.expect_top
        if not quiet:
            self.get_logger().info(
                f"윗면 z {top_z:.1f} (예상 {self.expect_top:.1f}, 차이 {d:+.1f}) "
                f"→ 파지 z {self.pick_z:.1f}")
        if abs(d) > Z_SANITY and not quiet:
            self.get_logger().warn(
                f"윗면 높이가 예상에서 {d:+.1f}mm 벗어났습니다. "
                "블록이 쌓였거나 깊이가 튀었을 수 있습니다.")
        pose = list(xyz) + list(posx[3:])
        rot = GRASP_ROT
        if ALIGN_TO_BLOCK:
            a = ((self.last_angle + 45.0) % 90.0) - 45.0   # -45~+45 로 접기
            rot += a
            if not quiet:
                self.get_logger().info(
                    f"블록 기울기 {self.last_angle:.0f}° → 손목 {rot:+.0f}°")
        self.last_rot = rot
        return self.rotate_tool(pose, rot)

    # ── 동작 ──
    def go_home(self):
        gripper.open_gripper()
        wait_gripper()
        movej(JREADY, vel=VEL_J, acc=ACC_J)
        mwait()

    def grasp_offset(self):
        """지금 손목 각도에서의 파지 중심 보정. 공구 성분은 함께 회전시킨다."""
        r = np.radians(self.last_rot)
        R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        return np.array(OFFSET_BASE) + R @ np.array(OFFSET_TOOL)

    def pick(self, pose):
        """접근 → 하강 → 파지 → 확인. 성공하면 True."""
        pose = list(pose)
        g = self.grasp_offset()
        self.get_logger().info(
            f"파지 보정 ({g[0]:+.1f}, {g[1]:+.1f}) @ 손목 {self.last_rot:+.0f}°")
        pose[0] += g[0]
        pose[1] += g[1]
        above = list(pose); above[2] += self.lift
        for attempt in range(1, GRASP_RETRY + 1):
            movel(above, vel=VEL_L, acc=ACC_L); mwait()
            movel(pose, vel=VEL_L, acc=ACC_L); mwait()
            gripper.close_gripper()
            wait_gripper()
            if grasped():
                self.get_logger().info(f"파지 성공 (시도 {attempt})")
                movel(above, vel=VEL_L, acc=ACC_L); mwait()
                return True
            self.get_logger().warn(f"빈손 — 시도 {attempt}/{GRASP_RETRY}")
            gripper.open_gripper(); wait_gripper()
            movel(above, vel=VEL_L, acc=ACC_L); mwait()
        return False

    def place_at(self, p, taught=False):
        """주어진 자세에 내려놓는다.

        taught=True  손으로 티칭한 구역 좌표. 그 값에는 실제 파지 기하가
                     이미 녹아 있으므로 보정을 더하면 안 된다. 더하면
                     그만큼(측정치 13.7mm) 밀려서 놓인다.
        taught=False 비전이 계산한 좌표. 집을 때와 같은 보정이 필요하다.
        """
        p = list(p)
        if not taught:
            g = self.grasp_offset()
            p[0] += g[0]
            p[1] += g[1]
        above = list(p); above[2] += self.lift
        movel(above, vel=VEL_L, acc=ACC_L); mwait()
        movel(p, vel=VEL_L, acc=ACC_L); mwait()
        gripper.open_gripper(); wait_gripper()
        movel(above, vel=VEL_L, acc=ACC_L); mwait()

    def place(self, zone):
        # 구역은 티칭값이므로 보정을 더하지 않는다.
        self.place_at(self.cfg["zones"][zone], taught=True)

    def _detect_raw(self, target):
        """z 를 티칭값으로 바꾸지 않은 날것의 base 좌표. 든 블록을 잴 때 쓴다."""
        self.req.target = target
        fut = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=20.0)
        if fut.result() is None:
            return None
        cam = list(fut.result().depth_position)
        if sum(cam) == 0:
            return None
        posx = get_current_posx()[0]
        return self.to_base(cam[:3], posx), posx

    def calib_held(self, target, n=3):
        """블록을 든 채로 카메라로 보고 TCP 와의 차이를 잰다.

        집었다 놓는 방식은 손가락이 미는 축의 오차만 드러난다. 그 수직
        방향은 블록이 밀리지 않아 놓아도 같은 자리에 떨어지기 때문이다.
        든 상태로 직접 보면 두 축을 한 번에 잰다.
        """
        errs = []
        for i in range(1, n + 1):
            self.go_home()
            p = self.detect(target)
            if p is None:
                self.get_logger().error("검출 실패 — 중단"); break
            if not self.pick(p):
                self.get_logger().error("파지 실패 — 중단"); break
            # pick 이 끝나면 파지 지점 바로 위에 떠 있다. 그 자리에서 든 블록을 본다.
            got = self._detect_raw(target)
            if got is None:
                self.get_logger().warn("든 블록을 못 봤습니다 — 건너뜀")
            else:
                blk, tcp = got
                e = (blk[0] - tcp[0], blk[1] - tcp[1])
                errs.append(e)
                self.get_logger().info(
                    f"[{i}/{n}] TCP ({tcp[0]:.1f}, {tcp[1]:.1f})  "
                    f"블록 ({blk[0]:.1f}, {blk[1]:.1f})  오차 ({e[0]:+.1f}, {e[1]:+.1f})")
            self.place_at(p)          # 원래 자리에 돌려놓는다
        self.go_home()
        if not errs:
            print("  측정 실패\n")
            return None
        a = np.array(errs)
        keep = np.linalg.norm(a - np.median(a, 0), axis=1) <= CALIB_OUTLIER
        a = a[keep]
        m, s = a.mean(0), a.std(0)
        print(f"\n  유효 측정 {len(a)}/{len(errs)}회")
        print(f"  블록이 TCP 대비   x {m[0]:+.2f}  y {m[1]:+.2f}  mm 에 물림")
        print(f"  표준편차          x {s[0]:5.2f}  y {s[1]:5.2f}  mm")
        print(f"\n  중앙을 물게 하려면")
        print(f"    (참고) 이 자세에서의 보정에 ({m[0]:+.1f}, {m[1]:+.1f}) 를 더하세요\n")
        return m

    @staticmethod
    def rotate_tool(posx, deg):
        """공구 자신의 z 축 둘레로 deg 만큼 돌린 자세를 만든다."""
        R = Rotation.from_euler("ZYZ", posx[3:], degrees=True).as_matrix()
        Rn = R @ Rotation.from_euler("z", deg, degrees=True).as_matrix()
        e = Rotation.from_matrix(Rn).as_euler("ZYZ", degrees=True)
        return list(posx[:3]) + list(e)

    def calib_axis(self, target, deg, n=3):
        """손목을 deg 돌린 채로 집었다 놓고, 블록이 밀린 양을 잰다.

        평행 그리퍼는 '닫히는 축' 으로만 블록을 밀어 정렬시킨다. 그래서
        집었다 놓으면 그 축의 오차만 드러난다. 손목을 90° 돌려 한 번 더
        재면 나머지 축도 얻는다.
        """
        errs = []
        for i in range(1, n + 1):
            self.go_home()
            p = self.detect(target)
            if p is None:
                self.get_logger().error("검출 실패 — 중단"); break
            if not self.pick(self.rotate_tool(p, deg - GRASP_ROT)):
                self.get_logger().error("파지 실패 — 중단"); break
            self.place_at(self.rotate_tool(p, deg - GRASP_ROT))
            self.go_home()
            q = self.detect(target)
            if q is None:
                self.get_logger().error("재검출 실패 — 중단"); break
            e = (q[0] - p[0], q[1] - p[1])
            errs.append(e)
            self.get_logger().info(
                f"[{i}/{n}] 손목 {deg:+.0f}°  오차 ({e[0]:+.1f}, {e[1]:+.1f})")
        if not errs:
            return None
        a = np.array(errs)
        a = a[np.linalg.norm(a, axis=1) <= CALIB_OUTLIER]
        if len(a) == 0:
            print("  쓸 수 있는 측정이 없습니다."); return None
        m, s = a.mean(0), a.std(0)
        print(f"\n  손목 {deg:+.0f}°  유효 {len(a)}/{len(errs)}회")
        print(f"    평균 ({m[0]:+.2f}, {m[1]:+.2f})   편차 ({s[0]:.2f}, {s[1]:.2f})\n")
        return m

    def center(self, target, wrist=0.0):
        """카메라를 블록 위로 수렴시킨 뒤, 손으로 미세조정해 상수를 실측한다.

        내부파라미터·깊이·핸드아이·TCP 오차를 하나씩 잡는 대신, 그 누적분을
        상수 하나로 흡수한다. 수렴 후 그리퍼를 정중앙에 맞추면, 그 이동량이
        곧 GRASP_OFFSET 에 더할 값이다.
        """
        import termios
        import tty

        global GRASP_ROT, ALIGN_TO_BLOCK
        keep_rot, keep_align = GRASP_ROT, ALIGN_TO_BLOCK
        GRASP_ROT, ALIGN_TO_BLOCK = wrist, False   # 각도를 고정해야 분리가 된다
        self.go_home()
        p = self.detect(target)
        GRASP_ROT, ALIGN_TO_BLOCK = keep_rot, keep_align
        if p is None:
            self.get_logger().error(f"'{target}' 미검출")
            return
        above = list(p); above[2] += self.lift
        gripper.open_gripper(); wait_gripper()
        movel(above, vel=VEL_L, acc=ACC_L); mwait()
        movel(p, vel=VEL_L, acc=ACC_L); mwait()

        step = 1.0
        dx = dy = 0.0
        print("\n" + "=" * 58)
        print(" 손가락 사이 정중앙에 블록이 오도록 맞추세요")
        print("=" * 58)
        print("   w / s   x  +  / -        a / d   y  -  / +")
        print("   [ / ]   이동 폭 줄이기 / 키우기")
        print("   Enter   확정 (상수 저장)        q  취소\n")

        fd = sys.stdin.fileno()
        old_attr = termios.tcgetattr(fd)
        try:
            while True:
                sys.stdout.write(
                    f"\r  누적 ({dx:+6.1f}, {dy:+6.1f}) mm   이동폭 {step:4.1f}mm   ")
                sys.stdout.flush()
                tty.setraw(fd)
                k = sys.stdin.read(1)
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
                if k == "q":
                    print("\n  취소\n"); break
                if k in ("\r", "\n"):
                    cur = self.grasp_offset()
                    # numpy 스칼라는 yaml.safe_dump 가 직렬화하지 못한다.
                    o = (float(cur[0]) + dx, float(cur[1]) + dy)
                    print(f"\n\n  손목 {wrist:.0f}° 에서의 총 보정 "
                          f"({o[0]:+.1f}, {o[1]:+.1f}) mm")
                    d = {}
                    if os.path.exists(CENTER_YAML):
                        d = yaml.safe_load(open(CENTER_YAML)) or {}
                    d[float(wrist)] = [round(o[0], 2), round(o[1], 2)]
                    with open(CENTER_YAML, "w") as f:
                        yaml.safe_dump(d, f, default_flow_style=None)
                    print(f"  {CENTER_YAML} 에 저장")
                    have = sorted(d)
                    print(f"  측정된 각도: {have}")
                    if 0.0 in d and 90.0 in d:
                        o0, o9 = np.array(d[0.0]), np.array(d[90.0])
                        dd = o0 - o9
                        t = np.array([(dd[0]-dd[1])/2, (dd[0]+dd[1])/2])
                        c = o0 - t
                        print(f"\n  분리 완료")
                        print(f"    c_base (각도 무관)  ({c[0]:+.2f}, {c[1]:+.2f})")
                        print(f"    t_tool (함께 회전)  ({t[0]:+.2f}, {t[1]:+.2f})")
                        print(f"    → block_sort.py 에 아래를 넣으세요")
                        print(f"      OFFSET_BASE = [{c[0]:.2f}, {c[1]:.2f}]")
                        print(f"      OFFSET_TOOL = [{t[0]:.2f}, {t[1]:.2f}]\n")
                    else:
                        need = [a for a in (0.0, 90.0) if a not in d]
                        print(f"  분리하려면 손목 {need} 에서도 재야 합니다\n")
                    break
                d = {"w": (step, 0), "s": (-step, 0),
                     "a": (0, -step), "d": (0, step)}.get(k)
                if k == "[":
                    step = max(0.2, step / 2); continue
                if k == "]":
                    step = min(10.0, step * 2); continue
                if d is None:
                    continue
                tgt = list(p)
                tgt[0] += dx + d[0]; tgt[1] += dy + d[1]
                try:
                    movel(tgt, vel=20, acc=20); mwait()
                    dx += d[0]; dy += d[1]
                except Exception as e:
                    print(f"\n  이동 실패: {e}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
        movel(above, vel=VEL_L, acc=ACC_L); mwait()
        self.go_home()

    def aim(self, target, hold=25.0):
        """그리퍼를 연 채로 파지 자세까지만 가서 멈춘다.

        간접 측정(집었다 놓고 다시 보기)은 손가락이 닫히는 축만 드러낸다.
        그 수직 방향은 눈으로 봐야 한다. 손가락 사이에 블록이 대칭으로
        들어와 있는지 확인하고, 치우쳤다면 어느 쪽으로 몇 mm 인지 본다.
        """
        global GRASP_ROT, ALIGN_TO_BLOCK
        keep_rot, keep_align = GRASP_ROT, ALIGN_TO_BLOCK
        GRASP_ROT, ALIGN_TO_BLOCK = wrist, False   # 각도를 고정해야 분리가 된다
        self.go_home()
        p = self.detect(target)
        GRASP_ROT, ALIGN_TO_BLOCK = keep_rot, keep_align
        if p is None:
            self.get_logger().error(f"'{target}' 미검출")
            return
        above = list(p); above[2] += self.lift
        gripper.open_gripper(); wait_gripper()
        movel(above, vel=VEL_L, acc=ACC_L); mwait()
        movel(p, vel=VEL_L, acc=ACC_L); mwait()
        rot = GRASP_ROT + (((self.last_angle + 45.0) % 90.0) - 45.0
                           if ALIGN_TO_BLOCK else 0.0)
        print(f"\n  파지 자세로 정지했습니다 (그리퍼 열림)")
        print(f"    목표 (x, y) = ({p[0]:.1f}, {p[1]:.1f})   손목 {rot:+.0f}°")
        print(f"    블록이 손가락 사이 중앙에 있습니까?")
        print(f"    치우쳤다면 로봇 기준 어느 축으로 몇 mm 인지 보세요.")
        print(f"    {hold:.0f}초 뒤 물러납니다.\n")
        time.sleep(hold)
        movel(above, vel=VEL_L, acc=ACC_L); mwait()
        self.go_home()

    def calib_zones(self, target, rounds=2):
        """구역에 놓고 검출해 zones.yaml 을 실측으로 다듬는다.

        구역 좌표는 사람이 블록을 손으로 물려 티칭한 값이다. 그때의 파지
        상태와 로봇이 집는 상태가 달라 그 차이만큼 어긋나 놓인다.
        측정: 명령 C 로 놓았더니 블록이 q 에 있었다 → 오차 e = q - C.
        목표는 원래 티칭 위치 T 이므로 다음 명령은 C_new = T - e 다.
        """
        goal = {z: list(p) for z, p in self.cfg["zones"].items()}   # 원래 티칭값
        for r in range(1, rounds + 1):
            print(f"\n  ── {r}회차 ──")
            for z in sorted(goal):
                self.go_home()
                p = self.detect(target)
                if p is None:
                    print(f"  {z}번  검출 실패 — 건너뜀"); continue
                if not self.pick(p):
                    print(f"  {z}번  파지 실패 — 건너뜀"); continue
                cmd = list(self.cfg["zones"][z])
                self.place(z)
                self.go_home()
                q = self.detect(target)
                if q is None:
                    print(f"  {z}번  재검출 실패 — 건너뜀"); continue
                e = (q[0] - cmd[0], q[1] - cmd[1])
                new = [goal[z][0] - e[0], goal[z][1] - e[1]] + list(cmd[2:])
                self.cfg["zones"][z] = new
                print(f"  {z}번  오차 ({e[0]:+6.1f}, {e[1]:+6.1f})  →  "
                      f"명령 ({cmd[0]:.1f}, {cmd[1]:.1f}) → ({new[0]:.1f}, {new[1]:.1f})")
        self.go_home()
        out = dict(self.cfg)
        out["zones"] = {int(k): [round(float(v), 2) for v in p]
                        for k, p in self.cfg["zones"].items()}
        with open(ZONES_YAML, "w") as f:
            yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False,
                           default_flow_style=None)
        print(f"\n  {ZONES_YAML} 갱신 완료\n")

    def calibrate(self, target, n=3):
        """집었다 같은 자리에 도로 놓고 다시 검출해 파지 중심 오차를 잰다.

        파지 중심이 e 만큼 밀려 있으면, 집을 때 블록이 손가락 중심으로
        끌려오고 놓을 때 그 자리에 남으므로 블록은 P+e 에 놓인다.
        따라서 (재검출 위치 − 원래 위치) 가 곧 e 다.
        """
        errs = []
        for i in range(1, n + 1):
            self.go_home()
            p1 = self.detect(target)
            if p1 is None:
                self.get_logger().error("검출 실패 — 중단")
                break
            if not self.pick(p1):
                self.get_logger().error("파지 실패 — 중단")
                break
            self.place_at(p1)
            self.go_home()
            p2 = self.detect(target)
            if p2 is None:
                self.get_logger().error("재검출 실패 — 중단")
                break
            e = (p2[0] - p1[0], p2[1] - p1[1])
            errs.append(e)
            self.get_logger().info(
                f"[{i}/{n}] 원래 ({p1[0]:.1f}, {p1[1]:.1f}) → "
                f"재검출 ({p2[0]:.1f}, {p2[1]:.1f})   오차 ({e[0]:+.1f}, {e[1]:+.1f})")
        if not errs:
            return None
        a = np.array(errs)
        # 다른 블록을 봤거나 블록이 튄 경우가 섞인다. 파지 오차가 수십 mm 일
        # 수는 없으므로 걸러낸다. 남은 것만으로 평균을 낸다.
        keep = np.linalg.norm(a, axis=1) <= CALIB_OUTLIER
        if (~keep).any():
            for e in a[~keep]:
                print(f"  이상치 제외  ({e[0]:+.1f}, {e[1]:+.1f})")
        a = a[keep]
        if len(a) == 0:
            print("  쓸 수 있는 측정이 없습니다.")
            return None
        m, s = a.mean(0), a.std(0)
        print(f"\n  유효 측정 {len(a)}/{len(errs)}회")
        print(f"  평균 오차   x {m[0]:+.2f}  y {m[1]:+.2f}  mm")
        print(f"  표준편차    x {s[0]:5.2f}  y {s[1]:5.2f}  mm")
        print(f"\n  block_sort.py 의 GRASP_OFFSET 을 이렇게 바꾸세요")
        print(f"    GRASP_OFFSET = [{-m[0]:.1f}, {-m[1]:.1f}]\n")
        return m

    def run_one(self, target, zone):
        if zone not in self.cfg["zones"]:
            self.get_logger().error(f"{zone}번 구역이 없습니다. "
                                    f"가능: {sorted(self.cfg['zones'])}")
            return False
        self.go_home()
        pose = self.detect(target)
        if pose is None:
            return False
        self.get_logger().info(f"파지 목표 {[round(v,1) for v in pose[:3]]}")
        if not self.pick(pose):
            self.get_logger().error("파지 실패 — 중단")
            self.go_home()
            return False
        self.place(zone)
        self.go_home()
        self.get_logger().info(f"완료: {target} → {zone}번")
        return True


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    m = sys.argv[1]
    node = BlockSort()
    try:
        if m == "home":
            node.go_home()
        elif m == "observe":
            node.go_home()
            t = sys.argv[2] if len(sys.argv) > 2 else "Apple"
            p = node.detect(t)
            print(f"\n  '{t}' → {'못 찾음' if p is None else [round(v,1) for v in p[:3]]}\n")
        elif m == "pick":
            if len(sys.argv) < 4:
                sys.exit("사용법: pick <대상> <구역>")
            node.run_one(sys.argv[2], int(sys.argv[3]))
        elif m == "calib":
            if len(sys.argv) < 3:
                sys.exit("사용법: calib <대상> [횟수]")
            node.calib_held(sys.argv[2],
                            int(sys.argv[3]) if len(sys.argv) > 3 else 3)
        elif m == "center":
            if len(sys.argv) < 3:
                sys.exit("사용법: center <대상>")
            node.center(sys.argv[2],
                        float(sys.argv[3]) if len(sys.argv) > 3 else 0.0)
        elif m == "aim":
            if len(sys.argv) < 3:
                sys.exit("사용법: aim <대상> [정지초]")
            node.aim(sys.argv[2],
                     float(sys.argv[3]) if len(sys.argv) > 3 else 25.0)
        elif m == "calib-zones":
            if len(sys.argv) < 3:
                sys.exit("사용법: calib-zones <대상> [회차]")
            node.calib_zones(sys.argv[2],
                             int(sys.argv[3]) if len(sys.argv) > 3 else 2)
        elif m == "calib-axis":
            if len(sys.argv) < 4:
                sys.exit("사용법: calib-axis <대상> <손목각> [횟수]")
            node.calib_axis(sys.argv[2], float(sys.argv[3]),
                            int(sys.argv[4]) if len(sys.argv) > 4 else 3)
        elif m == "calib-drop":
            if len(sys.argv) < 3:
                sys.exit("사용법: calib-drop <대상> [횟수]")
            node.calibrate(sys.argv[2],
                           int(sys.argv[3]) if len(sys.argv) > 3 else 3)
        elif m == "run":
            print("\n  '<대상> <구역>' 입력.  q 로 종료")
            print(f"  구역: {sorted(node.cfg['zones'])}\n")
            while True:
                s = input("  > ").strip().split()
                if not s or s[0] == "q":
                    break
                if len(s) != 2 or not s[1].isdigit():
                    print("    형식: Apple 1")
                    continue
                node.run_one(s[0], int(s[1]))
        else:
            sys.exit(__doc__)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
