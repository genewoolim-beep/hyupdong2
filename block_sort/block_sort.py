#!/usr/bin/env python3
"""
관측 → 검출 → 파지 → 구역 배치   (M0609 + RG2 + RealSense)

  python3 block_sort.py observe            관측 자세로 가서 무엇이 보이는지만 보고
  python3 block_sort.py pick <색> <구역>     그 색 블록을 집어 구역에 놓기
  python3 block_sort.py pick <구역> <구역>   앞 구역에 있는 것을 뒤 구역으로 (색 무관)
  python3 block_sort.py run                텍스트로 반복 입력
  python3 block_sort.py scan               경유점을 한 바퀴 돌며 무엇이 있는지만 읽기
  python3 block_sort.py copy               인간구역 배치를 로봇구역에 그대로 복제
  python3 block_sort.py copy-mirror        좌우대칭으로 복제
  python3 block_sort.py sign               수어로 명령 → LLM 해석 → 구역 배치
  python3 block_sort.py home               초기 자세 복귀

기존 음성 pick&place 파이프라인을 그대로 쓴다. 바뀐 것은 두 가지뿐이다.
  · 놓는 자리가 BUCKET_POS 한 곳 → zones.yaml 의 구역 1~4
  · 입력이 음성(/get_keyword) → 텍스트 인자 또는 수어(sign 모드)

sign 모드는 sign_control 에서 학습한 제스처 분류기로 글로스를 읽고, 그 나열을
LLM 이 (색, 구역) 목록으로 바꾼다. 해석 부분만 따로 확인하려면 로봇 없이
`python3 sign_command.py parse "빨강 3번구역 놓다"` 를 쓴다.

블록 전용 YOLO 모델이 아직 없으므로, 지금은 과일/공구 클래스를 대역으로
써서 파이프라인을 검증한다. 모델이 생기면 대상 이름만 바꾸면 된다.
"""
import json
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

# 첫 실주행 때 50/80 으로 낮춰 뒀다가, 여러 번 무사히 돈 뒤 원래 값으로 되돌렸다.
# 경유점당 7~8.5초 중 이동이 4~5초였다(실측 2026-08-04). 느리면 환경변수로 낮춘다.
VEL_J = float(os.environ.get("VEL_J", 120))    # 관절 deg/s
ACC_J = float(os.environ.get("ACC_J", 120))
VEL_L = float(os.environ.get("VEL_L", 300))    # 직선 mm/s
ACC_L = float(os.environ.get("ACC_L", 300))

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
DETECT_SAMPLES = int(os.environ.get("DETECT_SAMPLES", 3))   # 최종 자세에서 재는 횟수

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

# 구역 하나가 차지하는 반경. 블록이 이 안에 있으면 '그 구역에 놓인 것'이고,
# 8개 구역(로봇 4 + 인간 4) 어디에도 안 걸리면 프리구역이다. 프리구역은 따로
# 티칭하지 않는다 — 구역을 뺀 나머지 전부이기 때문이다.
#
# 45mm 인 근거 (2026-08-04 실측 기하):
#   블록 반대각선 절반  35 × 1.414 / 2 = 24.7mm   45도 돌아가 있어도 덮는다
#   놓기 오차          약 10mm                   calib-zones 보정 후
#   검출 오차          약 10mm                   광축 정렬 수렴 후 잔차
# 더 크면 구역 사이 통로를 잡아먹고(70mm 면 12.5mm 만 남는다), 더 작으면
# 구역에 제대로 놓인 블록을 프리로 오판해 도로 집어간다.
# 구역 간격이 150mm 라 45mm 면 통로가 62.5mm 열린다 — 35mm 블록이 넉넉히 지난다.
ZONE_RADIUS = float(os.environ.get("ZONE_RADIUS", 45.0))

# 크게 움직인 뒤 검출하기 전에 쉬는 시간. mwait() 는 명령이 끝난 것만 알려주고
# 팔이 완전히 멎은 것은 보장하지 않는다. 자세한 이유는 goto() 주석 참고.
SETTLE_SEC = float(os.environ.get("SETTLE_SEC", 0.6))

# 이동이 실제로 반영됐는지 판정하는 허용 오차(mm). movel() 주석 참고.
MOVE_TOL = float(os.environ.get("MOVE_TOL", 8.0))

# 블록을 물고 xy 로 옮길 때 올라가는 절대 높이(mm). lift_height(150) 로는
# TCP 가 판에서 137mm 인데, 손가락이 그보다 한참 아래로 내려와 옆 블록을 스친다.
# 실측 2026-08-04: 옮기는 도중 다른 블록을 건드리는 일이 있었다.
TRANSIT_Z = float(os.environ.get("TRANSIT_Z", 250.0))

# 순회에서 봐 둔 자리와 다시 봤을 때의 자리가 이보다 벌어지면 다른 블록으로 본다.
# 실측 2026-08-04: 한계가 없어 250mm 떨어진 같은 색을 집으러 갔다.
CACHE_TOLERANCE = 70.0

# 특정 색만 찾아 다시 돌 때 건너뛸 앞쪽 경유점 수.
#   1번  인간구역 관측용이라 프리 블록이 없다
#   2번  로봇구역 쪽이라 프리 블록이 거의 없다 (실측에서 매번 빈손이었다)
# 프리 블록이 실제로 모이는 3번부터 돈다.
RESCAN_SKIP = int(os.environ.get("RESCAN_SKIP", 2))

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
# 검출 노드가 아는 색 목록. 색 없이 집을 때 하나씩 물어보려면 필요하다.
_CR_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "voice_pickandplace", "src", "object_detection", "resource", "color_ranges.json")
COLOR_RANGES_JSON = os.environ.get("COLOR_RANGES", _CR_DEFAULT)

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
rclpy.init()
_dsr = rclpy.create_node("block_sort_dsr", namespace=ROBOT_ID)
DR_init.__dsr__node = _dsr

try:
    from DSR_ROBOT2 import (movej as _movej_raw, movel as _movel_raw,
                            get_current_posx, mwait, get_robot_state, STATE_STANDBY)
except ImportError as e:
    sys.exit(f"DSR_ROBOT2 임포트 실패: {e}\n로봇 드라이버가 떠 있는지 확인하세요.")


# ───────────────────────── signbot_admin 연동 ─────────────────────────
# --admin 을 붙였을 때만 관리자 대시보드로 상태를 쏜다. 기본은 조용히 아무것도
# 안 한다 — 다른 팀원이 --admin 없이 그대로 써도 동작이 달라지지 않는다.
ADMIN_URL = os.environ.get("SIGN_ADMIN_URL", "http://localhost:5000")
USE_ADMIN = "--admin" in sys.argv


def _admin_post(path, payload):
    if not USE_ADMIN:
        return
    import json
    import threading
    import urllib.request

    def _send():
        try:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{ADMIN_URL}{path}", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=1.0)
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()


def push_zone(space, zone_num, color):
    """구역 하나의 색을 signbot_admin에 반영한다. space: robot | human"""
    _admin_post(f"/api/zones/{space}", {"zone": f"{zone_num}번구역", "color": color})


def push_zones(space, zone_colors, all_zone_nums):
    """구역 전체를 한 번에 반영한다. zone_colors에 없는 번호는 빈 구역으로 민다."""
    for i in all_zone_nums:
        push_zone(space, i, zone_colors.get(i))


def push_robot_status(**fields):
    _admin_post("/api/robot/status", fields)


def push_debug(level, source, message):
    _admin_post("/api/debug", {"level": level, "source": source, "message": message})


def wait_idle(timeout=20.0):
    """모션이 끝나 STANDBY 가 될 때까지 기다린다.

    mwait() 만으로는 부족하다. 실측 2026-08-04: copy 한 번에
    "A motion is ongoing so new commands are not accepted." 가 27번 났고,
    그때마다 movel 이 **조용히 무시돼** 팔이 그 자리에 머물렀다.
    hover_over 가 목표 위로 못 가 광축이 196mm 어긋난 것도 이 때문이다.
    거부는 예외가 아니라 로그 한 줄이라 코드가 알아챌 방법이 없어서,
    명령을 넣기 전에 상태로 직접 확인한다.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if get_robot_state() == STATE_STANDBY:
                return True
        except Exception:
            pass
        time.sleep(0.05)
    print(f"경고: {timeout}초 동안 STANDBY 가 안 됐습니다 — 그대로 진행합니다")
    return False


def _reached(cur, p, tol):
    return (np.hypot(cur[0] - p[0], cur[1] - p[1]) <= tol
            and abs(cur[2] - p[2]) <= tol)


def movel(p, vel=VEL_L, acc=ACC_L, tol=None, tries=3, timeout=20.0):
    """직선 이동 + 도달 확인. 안 갔으면 다시 넣는다.

    "A motion is ongoing so new commands are not accepted." 는 예외가 아니라
    로그 한 줄이라 코드가 알아챌 방법이 없다. 실측 2026-08-05: hover_over 가
    씹혀 팔이 이전 경유점에 머문 채 검출하는 바람에 블록이 광축에서 188mm
    벗어나 보였고, 그 비스듬한 시야에서 잰 각도가 4° 흔들려(64→63→60)
    큐브가 틀어져 물렸다.

    상태(STANDBY/MOVING)가 아니라 **위치**를 지켜본다. 상태로 판단하면 두 번
    틀린다. 명령 직후에는 아직 STANDBY 라 '안 움직였다'고 오판하고, 반대로
    짧은 이동은 MOVING 이 관측되기 전에 끝나 고정 타임아웃을 꽉 채운다
    (실측: 배치 한 번이 12초 → 23초). 위치는 그 둘 다에 흔들리지 않는다.
    """
    p = list(p)
    tol = MOVE_TOL if tol is None else tol
    for k in range(1, tries + 1):
        cur = get_current_posx()[0]
        if _reached(cur, p, tol):
            return True                      # 이미 목표에 있다
        _movel_raw(p, vel=vel, acc=acc)

        t0, last, still = time.time(), None, 0
        while time.time() - t0 < timeout:
            cur = get_current_posx()[0]
            if _reached(cur, p, tol):
                return True
            if last is not None and np.hypot(cur[0] - last[0], cur[1] - last[1]) < 0.5 \
                    and abs(cur[2] - last[2]) < 0.5:
                still += 1
                if still >= 6:               # 0.3초 정지 = 더 갈 생각이 없다
                    break
            else:
                still = 0
            last = cur
            time.sleep(0.05)

        err = float(np.hypot(cur[0] - p[0], cur[1] - p[1]))
        print(f"이동이 반영되지 않았습니다 — 목표에서 xy {err:.0f}mm, "
              f"z {abs(cur[2] - p[2]):.0f}mm 어긋남. 재시도 {k}/{tries}")
        time.sleep(0.3)
    print(f"경고: {tries}회 시도했으나 목표에 못 갔습니다 {[round(v, 1) for v in p[:3]]}")
    return False


def movej(q, vel=VEL_J, acc=ACC_J):
    wait_idle()
    _movej_raw(list(q), vel=vel, acc=acc)
    mwait()
    wait_idle()



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


def load_colors():
    """검출 노드와 같은 파일에서 색 이름을 읽는다.

    여기에 목록을 따로 적어두면 color_ranges.json 에 색을 추가했을 때
    '구역에서 집기'만 그 색을 못 보는 일이 생긴다.
    """
    try:
        with open(COLOR_RANGES_JSON, encoding="utf-8") as f:
            return list(json.load(f)["colors"])
    except Exception as e:
        print(f"{COLOR_RANGES_JSON} 을 못 읽어 기본 목록을 씁니다 ({e})")
        return ["빨간색", "주황색", "노란색", "초록색", "파란색", "보라색"]


KNOWN_COLORS = load_colors()


def level_att(att):
    """자세각의 기울기만 펴서 공구가 정확히 아래를 보게 한다. 요(yaw)는 유지.

    ZYZ 에서 ry=180 이면 공구 z 축이 정확히 -z(아래)를 향한다(수치 확인).
    관측 자세가 기울어져 있으면(observe_free 실측 26.5°) 그 기울기가 파지 자세까지
    그대로 딸려와, 손가락이 판에 비스듬히 내려가 모서리를 물게 된다.
    광축도 같이 기울어 x,y 이동이 화면 중심 이동과 어긋난다.
    """
    return [att[0], 180.0, att[2]]


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
    def detect(self, target, refine=True, near=None, max_dist=None):
        """1차로 찾고, 그 위로 카메라를 옮겨 2차로 정밀 측정한다.

        색 임계값은 블록의 윗면과 옆면을 구분하지 못한다. 비스듬히 볼수록
        옆면이 마스크에 붙어 박스가 커지고, 중심이 그 절반만큼 카메라
        반대쪽으로 밀린다. 실측에서 광축 근처 49.5mm, 가장자리 59.6mm 로
        10mm 차이가 났다 — 중심 오차 5mm 다.

        블록 바로 위에서 보면 광축각이 0 에 가까워 옆면이 안 보인다.
        원인 자체가 사라지므로 기하 보정보다 확실하다.
        """
        cur = self._detect_once(target, near=near, max_dist=max_dist)
        if cur is None or not refine:
            return cur

        best, best_off = cur, float(np.hypot(self.last_cam[0], self.last_cam[1]))
        for it in range(1, REFINE_ITERS + 1):
            if best_off <= REFINE_TOL:
                self.get_logger().info(f"광축에서 {best_off:.1f}mm — 수렴")
                break
            posx = get_current_posx()[0]
            # 기울어진 채로 x,y 만 옮기면 광축은 엉뚱한 데를 본다. 실측에서
            # "블록 위로 이동" 뒤에도 광축 이탈이 327mm 그대로 남았다.
            att = level_att(posx[3:])
            R = Rotation.from_euler("ZYZ", att, degrees=True).as_matrix()
            cam_off = R @ self.gripper2cam[:3, 3]    # 카메라가 그리퍼에서 떨어진 양
            view = list(posx[:3]) + att
            view[0] = best[0] - cam_off[0]           # 카메라를 블록 바로 위로
            view[1] = best[1] - cam_off[1]
            self.get_logger().info(
                f"[{it}] 광축 {best_off:.1f}mm — 블록 위로 이동 "
                f"({view[0]:.0f}, {view[1]:.0f})")
            try:
                movel(view)
                # 옮긴 뒤 반드시 쉰다. 이게 없으면 검출 노드가 **이동 전에 찍어둔**
                # 프레임을 돌려주고(FRAME_CACHE_SEC), 그것을 **이동 후** 팔 자세로
                # 변환해 좌표가 통째로 어긋난다. 실측 2026-08-04: 윗면 z 가 예상에서
                # +36mm 튀고 250mm 떨어진 엉뚱한 블록이 잡혔다.
                time.sleep(SETTLE_SEC)
            except Exception as e:
                self.get_logger().warn(f"이동 실패({e}) — 직전 결과 사용")
                break
            # 시점이 바뀌면 다른 블록이 더 크게 보일 수 있다. 직전에 고른
            # 자리를 기준으로 삼아 같은 블록을 계속 따라간다.
            nxt = self._detect_once(target, near=(best[0], best[1]))
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
                s = self._detect_once(target, quiet=True, near=(best[0], best[1]))
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

    # ── 인간구역 읽기 / 복제 ──
    def human_zones(self):
        z = self.cfg.get("human_zones") or {}
        return {int(k): v for k, v in z.items()}

    def all_zone_xy(self):
        """로봇구역 + 인간구역의 (x, y). 프리구역을 가려내는 기준이 된다."""
        out = [tuple(p[:2]) for p in self.cfg["zones"].values()]
        out += [tuple(p[:2]) for p in self.human_zones().values()]
        return out

    def is_free(self, pose):
        """어느 구역에도 속하지 않은 자리인가. 프리구역 블록만 집기 위한 판정."""
        for zx, zy in self.all_zone_xy():
            if np.hypot(pose[0] - zx, pose[1] - zy) < ZONE_RADIUS:
                return False
        return True

    # ── 순회 관측 ──
    def scan_points(self):
        return list(self.cfg.get("scan_path") or [])

    def goto(self, p, why=""):
        """경유점으로 이동하고 팔이 멎기를 기다린다.

        mwait() 는 명령이 끝난 것만 알려주고 팔이 완전히 멎은 것은 보장하지 않는다.
        검출 노드는 0.4초간 여러 프레임을 모아 합의를 내는데, 그 프레임들이 흔들리는
        동안 잡히면 뒤에 읽는 팔 자세와 안 맞아 좌표가 통째로 밀린다.
        실측 2026-08-04: 대기 없이 재면 노랑이 1번 대신 4번으로 잡히고 빨강이
        42mm 어긋났다. 1초 쉬고 재니 3~6mm 로 들어왔다.
        """
        if why:
            self.get_logger().info(f"{why} {[round(v, 1) for v in p[:3]]}")
        movel(p)
        time.sleep(SETTLE_SEC)

    def look(self, hz=None):
        """지금 자리에서 보이는 것을 인간구역/프리로 갈라 돌려준다.

        (인간구역 {번호: 색}, 프리 {색: [검출, ...]}, 로봇구역 {번호: 색})
        로봇구역·인간구역 위의 블록은 프리에서 빠진다 — 본보기와 방금 만든
        배치를 도로 집어가면 안 되기 때문이다.
        로봇구역에 이미 놓인 것도 돌려준다. 그 자리에 또 놓으려 하면 부딪힌다.
        """
        hz = self.human_zones() if hz is None else hz
        rz = self.cfg["zones"]
        seen, free, occupied = {}, {}, {}
        for color in KNOWN_COLORS:
            for det in self._detect_all(color, quiet=True):
                p = det["pose"]
                if hz:
                    b = min(hz, key=lambda i: np.hypot(p[0] - hz[i][0], p[1] - hz[i][1]))
                    if np.hypot(p[0] - hz[b][0], p[1] - hz[b][1]) <= ZONE_RADIUS:
                        seen.setdefault(b, color)
                        continue
                b = min(rz, key=lambda i: np.hypot(p[0] - rz[i][0], p[1] - rz[i][1]))
                if np.hypot(p[0] - rz[b][0], p[1] - rz[b][1]) <= ZONE_RADIUS:
                    occupied.setdefault(b, color)
                    continue
                if self.is_free(p):
                    free.setdefault(color, []).append(det)
        return seen, free, occupied

    def patrol(self, want=None, stop_on_find=True):
        """관심 영역 위를 경유점 따라 한 바퀴 돈다.

        want 에 색 집합을 주고 stop_on_find 면, 그 색을 프리구역에서 보는 즉시
        멈추고 돌아온다 — 나머지 경유점을 도는 시간을 아낀다.
        want 가 없으면 끝까지 돌며 본 것을 모두 모은다.

        넓게 한 번에 보는 관측 자세를 없앤 이유: 멀리서 한 장에 다 담으면 블록이
        작게 잡혀 최소 면적에 걸리고, 비스듬히 보여 중심이 밀린다. 가까이서
        나눠 보면 두 문제가 같이 사라진다.

        돌려주는 것: (인간구역, 프리재고, 로봇구역 점유, 멈춘 경유점 번호 또는 None)
        """
        pts = self.scan_points()
        if not pts:
            self.get_logger().error(
                "순회 경유점이 없습니다. python3 teach_zones.py scan 을 먼저 실행하세요.")
            return {}, {}, {}, None
        start = 0
        if want:
            # 앞쪽 경유점에는 프리 블록이 없다. 자세한 이유는 RESCAN_SKIP 주석 참고.
            start = min(RESCAN_SKIP, len(pts) - 1)
        pts = pts[start:]
        hz = self.human_zones()
        seen, stock, occ = {}, {}, {}
        for i, p in enumerate(pts, 1):
            self.goto(p, f"[순회 {i + start}/{len(pts) + start}]")
            s_i, f_i, o_i = self.look(hz)
            for z, c in o_i.items():
                if z not in occ:
                    occ[z] = c
                    self.get_logger().info(f"  로봇{z}번에 이미 {c} 있음")
            for z, c in s_i.items():
                if z not in seen:
                    seen[z] = c
                    self.get_logger().info(f"  인간{z}번 = {c}")
            for c, lst in f_i.items():
                known = stock.setdefault(c, [])
                for det in lst:
                    # 경유점마다 시야가 겹치므로 같은 블록을 두 번 볼 수 있다.
                    if any(np.hypot(det["pose"][0] - k["pose"][0],
                                    det["pose"][1] - k["pose"][1]) < ZONE_RADIUS
                           for k in known):
                        continue
                    known.append(det)
                    self.get_logger().info(
                        f"  프리 {c} ({det['pose'][0]:.0f}, {det['pose'][1]:.0f})")
            if want and stop_on_find and any(stock.get(c) for c in want):
                self.get_logger().info(f"  필요한 블록 발견 — {i}번에서 순회 중단")
                return seen, stock, occ, i
        return seen, stock, occ, None

    def hover_over(self, xy):
        """저장해 둔 자리 위로 카메라를 **수직으로** 내려다보게 옮긴다.

        높이·요는 그 자리에서 가장 가까운 경유점 것을 쓰고, 기울기만 편다.
        기울어진 채로 x,y 만 옮기면 광축이 엉뚱한 곳을 본다 — 실측에서
        "블록 위로 이동" 뒤에도 광축 이탈이 327mm 그대로 남았다.
        """
        pts = self.scan_points()
        if not pts:
            return False
        near = min(pts, key=lambda p: np.hypot(p[0] - xy[0], p[1] - xy[1]))
        att = level_att(near[3:])
        R = Rotation.from_euler("ZYZ", att, degrees=True).as_matrix()
        cam_off = R @ self.gripper2cam[:3, 3]      # 카메라가 그리퍼에서 떨어진 양
        p = [xy[0] - cam_off[0], xy[1] - cam_off[1], near[2]] + att
        try:
            self.goto(p)
        except Exception as e:
            self.get_logger().warn(f"목표 위로 이동 실패({e})")
            return False
        return True

    def copy_human(self, mirror=False):
        """인간구역의 배치를 로봇구역에 그대로(또는 좌우대칭으로) 만든다.

        블록은 프리구역에서만 집는다. 이미 구역에 놓인 것을 집으면 본보기나
        방금 만든 배치가 무너진다.
        """
        # 한 바퀴 다 돈다 — 인간구역 4칸을 다 읽어야 계획을 세울 수 있다.
        seen, stock, occ, _ = self.patrol()
        # 전체 순회라 인간구역/로봇구역 둘 다 지금 상태 그대로다 — 대시보드에 반영.
        push_zones("human", seen, sorted(self.human_zones()))
        push_zones("robot", occ, sorted(self.cfg["zones"]))
        if not seen:
            self.get_logger().error("인간구역에서 아무 블록도 못 찾았습니다.")
            self.go_home()
            return False

        zones = sorted(self.cfg["zones"])
        plan = []
        for hz_i, color in sorted(seen.items()):
            dst = self.mirror_zone(hz_i) if mirror else hz_i
            if dst not in zones:
                self.get_logger().warn(f"인간 {hz_i}번 → 로봇 {dst}번: 그 구역이 없습니다 — 건너뜀")
                continue
            plan.append((hz_i, dst, color))
        if not plan:
            self.go_home()
            return False

        kind = "좌우대칭" if mirror else "그대로"
        self.get_logger().warn(
            f"복제 계획 [{kind}] — " +
            ", ".join(f"인간{h}({c}) → 로봇{d}" for h, d, c in plan))

        # 이미 그 색이 놓여 있는 칸은 건드리지 않는다. 같은 자리에 또 놓으면 부딪힌다.
        done = [(h, d, c) for h, d, c in plan if occ.get(d) == c]
        if done:
            self.get_logger().info(
                "이미 맞게 놓인 칸: " + ", ".join(f"로봇{d}={c}" for _, d, c in done))
            plan = [t for t in plan if t not in done]
        busy = [(h, d, c) for h, d, c in plan if d in occ]
        if busy:
            self.get_logger().warn(
                "다른 색이 놓여 있어 건너뜁니다: "
                + ", ".join(f"로봇{d}에 {occ[d]}(→{c} 필요)" for _, d, c in busy))
            plan = [t for t in plan if t not in busy]
        if not plan:
            self.go_home()
            self.get_logger().warn("옮길 것이 없습니다.")
            return True

        # 재고 부족은 옮기기 전에 알려준다. 반쯤 하다 멈추면 판이 어중간해진다.
        short = [c for c in {c for _, _, c in plan}
                 if len(stock.get(c, [])) < sum(1 for _, _, x in plan if x == c)]
        if short:
            self.get_logger().warn(
                f"프리구역에 부족한 색: {', '.join(short)} — 그 칸은 건너뜁니다")

        ok = True
        placed = {}
        for hz_i, dst, color in plan:
            lst = stock.get(color) or []
            if not lst:
                self.get_logger().warn(f"인간{hz_i}번의 {color} — 프리구역에 없어 건너뜀")
                ok = False
                continue
            det = lst.pop(0)              # 같은 색이 여럿이면 하나씩 소비한다
            xy = (det["pose"][0], det["pose"][1])
            self.get_logger().info(
                f"── 인간{hz_i}번의 {color} → 로봇{dst}번  "
                f"(프리 ({xy[0]:.0f}, {xy[1]:.0f}) 에서 집기) ──")
            if self.pick_cached(color, xy, dst):
                placed[dst] = color
            else:
                ok = False
        if placed:
            # 다시 순회하지 않고, 이번에 놓은 것만 원래 점유 상태에 얹어서 반영한다.
            push_zones("robot", {**occ, **placed}, sorted(self.cfg["zones"]))
        self.go_home()
        self.get_logger().warn(f"복제 {'완료' if ok else '일부 실패'}")
        return ok

    def pick_cached(self, color, xy, zone):
        """관측 때 봐 둔 자리로 곧장 가서 집고 구역에 놓는다.

        넓은 관측 자세로 되돌아가지 않는다 — 좌표를 이미 알고 있으므로 그 위로
        바로 올라가 정밀화만 하면 된다. 옮길 색 수만큼 왕복이 줄어든다.
        """
        if not self.hover_over(xy):
            self.get_logger().error("관측 자세가 없어 집을 수 없습니다.")
            return False
        pose = self.detect(color, near=xy, max_dist=CACHE_TOLERANCE)
        if pose is None:
            # 그 자리에 없으면 블록이 굴렀거나 애초에 잘못 봤다. 그 색만 찾아
            # 다시 한 바퀴 돌고, 발견하는 즉시 멈춰 거기서 집는다.
            self.get_logger().warn(f"{color} 가 그 자리에 없음 — 그 색만 찾아 재순회")
            _, stock, _, _ = self.patrol(want={color})
            lst = stock.get(color) or []
            if not lst:
                return False
            xy = (lst[0]["pose"][0], lst[0]["pose"][1])
            if not self.hover_over(xy):
                return False
            pose = self.detect(color, near=xy)
        if pose is None:
            self.get_logger().error(f"{color} 를 다시 못 찾았습니다 — 건너뜁니다.")
            return False
        if not self.is_free(pose):
            # 관측 뒤에 판이 바뀌었거나 다른 개체를 잡은 것이다. 구역 위의 것은
            # 절대 집지 않는다 — 본보기나 방금 만든 배치가 무너진다.
            self.get_logger().error(
                f"{color} 로 다시 찾은 자리가 프리구역이 아닙니다 "
                f"({pose[0]:.0f}, {pose[1]:.0f}) — 건너뜁니다.")
            return False
        self.get_logger().info(f"파지 목표 {[round(v, 1) for v in pose[:3]]}")
        if not self.pick(pose):
            self.get_logger().error("파지 실패")
            self.go_home()
            return False
        self.place(zone)
        self.get_logger().info(f"완료: {color} → 로봇 {zone}번")
        return True

    def mirror_zone(self, i):
        """인간구역 i 를 x축 대칭(y 부호 반전)한 자리에 가장 가까운 로봇구역.

        매핑을 {1:2, 2:1, ...} 처럼 박아두면 구역 번호를 다시 매기거나 판을
        옮겼을 때 조용히 틀린다. 티칭 좌표에서 그때그때 유도한다.
        실측(2026-08-04)에서는 유도값이 {1:2, 2:1, 3:4, 4:3} 으로 나왔고,
        대칭점과 구역 중심의 차이는 6.5~9.0mm 였다.
        """
        H = self.human_zones()
        R = self.cfg["zones"]
        if i not in H or not R:
            return i
        mx, my = H[i][0], -H[i][1]          # x축 대칭 = y 부호 반전
        j = min(R, key=lambda k: np.hypot(R[k][0] - mx, R[k][1] - my))
        d = float(np.hypot(R[j][0] - mx, R[j][1] - my))
        if d > ZONE_RADIUS:
            self.get_logger().warn(
                f"인간{i}번의 x축 대칭점이 로봇{j}번에서 {d:.0f}mm 떨어져 있습니다 "
                "— 두 구역군이 x축을 사이에 두고 마주보게 놓였는지 확인하세요.")
        return j

    def detect_in_zone(self, zone):
        """그 구역에 놓인 블록의 색을 알아낸다. 없으면 None.

        색을 말하지 않은 명령("3번구역 들다 1번구역 놓다")은 '거기 있는 것'을
        집으라는 뜻이다. 그런데 검출 서비스는 색을 지정해야 답한다. 그래서
        아는 색을 전부 물어보고 구역 중심에 가장 가까운 것을 고른다.
        이 단계는 팔이 움직이지 않으므로 여섯 번 물어도 싸다 — 색을 정한 뒤에는
        detect() 의 평소 경로(광축 정렬 → 여러 번 재기)를 그대로 탄다.
        """
        zx, zy = self.cfg["zones"][zone][:2]
        found = []
        for color in KNOWN_COLORS:
            # 같은 색이 여러 곳에 있을 수 있으므로 전부 받아 가장 가까운 것만 남긴다.
            for det in self._detect_all(color, quiet=True):
                p = det["pose"]
                found.append((float(np.hypot(p[0] - zx, p[1] - zy)), color))
        if not found:
            self.get_logger().error(f"{zone}번 구역 근처에서 아무 블록도 못 찾았습니다.")
            return None

        found.sort()
        d, color = found[0]
        rest = "  다음: " + ", ".join(f"{c} {dd:.0f}mm" for dd, c in found[1:3]) \
            if len(found) > 1 else ""
        if d > ZONE_RADIUS:
            self.get_logger().error(
                f"{zone}번 구역이 비어 있습니다 — 가장 가까운 블록이 {color}, "
                f"{d:.0f}mm 떨어져 있습니다 (한계 {ZONE_RADIUS:.0f}mm).{rest}")
            return None
        self.get_logger().info(f"{zone}번 구역의 블록 = {color} ({d:.0f}mm){rest}")
        return color

    def _detect_all(self, target, quiet=False):
        """지금 자세에서 그 색으로 보이는 것을 전부 찾는다.

        검출 노드가 (x, y, z, 각도) 를 이어붙여 보내므로 4개씩 끊어 읽는다.
        믿을 만한 순으로 와 있어 첫 번째가 예전 결과와 같다.
        """
        if not self.cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/get_3d_position 서비스가 없습니다. "
                                    "object_detection 노드를 먼저 띄우세요.")
            return []
        self.req.target = target
        fut = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=20.0)
        if fut.result() is None:
            self.get_logger().error("서비스 응답 없음")
            return []

        raw = list(fut.result().depth_position)
        if len(raw) < 4 or sum(raw[:3]) == 0:
            # quiet 는 '여러 번 물어보는 중'이라는 뜻이다. detect_in_zone 은 아는 색을
            # 전부 훑으므로 없는 색마다 경고를 찍으면 정작 볼 줄이 묻힌다.
            if not quiet:
                self.get_logger().warn(f"'{target}' 을(를) 찾지 못했습니다.")
            return []

        # 영상과 로봇 자세는 같은 시점이어야 한다. 검출 직후 바로 읽는다.
        posx = get_current_posx()[0]
        out = []
        for k in range(0, len(raw) - 3, 4):     # (x, y, z, 각도) 씩 끊어 읽는다
            cam, ang = raw[k:k + 3], float(raw[k + 3])
            if sum(cam) == 0:
                continue
            xyz = self.to_base(cam, posx)
            top_z = float(xyz[2])
            xyz[2] = self.pick_z            # x,y 는 검출값, z 는 티칭값
            rot = GRASP_ROT
            if ALIGN_TO_BLOCK:
                rot += ((ang + 45.0) % 90.0) - 45.0     # -45~+45 로 접기
            # 파지 자세는 언제나 수직이어야 한다. 관측 자세의 기울기를 물려받으면
            # 그리퍼가 비스듬히 내려가고, rotate_tool 이 기울어진 축으로 돌아
            # 블록 면에 손가락을 맞추지도 못한다. grasp_offset 실측도 수직 기준이다.
            out.append({"pose": self.rotate_tool(list(xyz) + level_att(posx[3:]), rot),
                        "cam": list(cam), "angle": ang, "top": top_z, "rot": rot})
        if not quiet:
            self.get_logger().info(
                f"'{target}' {len(out)}개 — " +
                ", ".join(f"({d['pose'][0]:.0f}, {d['pose'][1]:.0f})" for d in out))
        return out

    def _adopt(self, det, quiet=False):
        """고른 검출을 '마지막 검출' 상태로 반영하고 파지 자세를 돌려준다.

        pick() 이 grasp_offset() 에서 self.last_rot 을 쓰므로, 어느 것을 골랐는지
        상태에 남겨야 한다. 고르기와 상태 반영을 한 곳에 묶어 둔다.
        """
        self.last_cam = det["cam"]
        self.last_angle = det["angle"]
        self.last_top = det["top"]
        self.last_rot = det["rot"]
        d = det["top"] - self.expect_top
        if not quiet:
            self.get_logger().info(
                f"윗면 z {det['top']:.1f} (예상 {self.expect_top:.1f}, 차이 {d:+.1f}) "
                f"→ 파지 z {self.pick_z:.1f}   기울기 {det['angle']:.0f}° "
                f"→ 손목 {det['rot']:+.0f}°")
            if abs(d) > Z_SANITY:
                self.get_logger().warn(
                    f"윗면 높이가 예상에서 {d:+.1f}mm 벗어났습니다. "
                    "블록이 쌓였거나 깊이가 튀었을 수 있습니다.")
        return det["pose"]

    def _detect_once(self, target, quiet=False, near=None, max_dist=None):
        """지금 자세에서 하나 고른다. near 를 주면 그 점에 가장 가까운 것.

        near 가 없으면 가장 믿을 만한 것(검출 노드가 앞에 놓은 것)을 쓴다.
        같은 색 블록이 여럿일 때 '어느 것'인지는 부르는 쪽만 알기 때문에
        고르는 기준을 인자로 받는다.
        """
        dets = self._detect_all(target, quiet=quiet)
        if not dets:
            return None
        if near is not None:
            dets = sorted(dets, key=lambda d: np.hypot(d["pose"][0] - near[0],
                                                       d["pose"][1] - near[1]))
            if max_dist is not None:
                d0 = float(np.hypot(dets[0]["pose"][0] - near[0],
                                    dets[0]["pose"][1] - near[1]))
                if d0 > max_dist:
                    # 찾던 자리에 없다. 멀리 있는 같은 색을 집으면 엉뚱한 블록이다.
                    if not quiet:
                        self.get_logger().warn(
                            f"'{target}' 가 기대 자리에서 {d0:.0f}mm 떨어져 있습니다 "
                            f"(한계 {max_dist:.0f}) — 못 찾은 것으로 봅니다")
                    return None
        return self._adopt(dets[0], quiet=quiet)

    # ── 동작 ──
    def go_home(self):
        gripper.open_gripper()
        wait_gripper()
        movej(JREADY)

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
        # 이송 높이까지 올린 뒤 xy 로 움직이고, 거기서 곧게 내린다.
        # lift 높이(137mm)로 옮기면 손가락이 옆 블록을 스친다.
        # 중간에 lift 높이를 따로 거치지 않는다 — 같은 수직선 위라 군더더기다.
        up = list(pose); up[2] = max(TRANSIT_Z, pose[2] + self.lift)
        for attempt in range(1, GRASP_RETRY + 1):
            movel(up)
            movel(pose)
            gripper.close_gripper()
            wait_gripper()
            if grasped():
                self.get_logger().info(f"파지 성공 (시도 {attempt})")
                movel(up)
                return True
            self.get_logger().warn(f"빈손 — 시도 {attempt}/{GRASP_RETRY}")
            gripper.open_gripper(); wait_gripper()
            movel(up)
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
        # 목표 위 이송 높이로 먼저 간 뒤 수직으로 내린다. 낮게 가로지르면
        # 물고 있는 블록이 판 위의 다른 블록을 친다.
        up = list(p); up[2] = max(TRANSIT_Z, p[2] + self.lift)
        movel(up)
        movel(p)
        gripper.open_gripper(); wait_gripper()
        movel(up)

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
        """target 은 색 이름(str) 또는 집어올 구역 번호(int).

        구역 번호를 주면 그 구역에 있는 블록을 색과 무관하게 집는다.
        """
        if zone not in self.cfg["zones"]:
            self.get_logger().error(f"{zone}번 구역이 없습니다. "
                                    f"가능: {sorted(self.cfg['zones'])}")
            return False
        if isinstance(target, int):
            if target not in self.cfg["zones"]:
                self.get_logger().error(f"{target}번 구역이 없습니다. "
                                        f"가능: {sorted(self.cfg['zones'])}")
                return False
            if target == zone:
                self.get_logger().error(f"{target}번에서 집어 {zone}번에 놓으라는 건 "
                                        "제자리입니다.")
                return False

        if isinstance(target, int):
            # 그 구역에 놓인 것을 집는 경우. 구역 위에 있으므로 순회로 찾지 않는다.
            self.go_home()
            color = self.detect_in_zone(target)
            if color is None:
                return False
            pose = self.detect(color)
            if pose is None:
                return False
            self.get_logger().info(f"파지 목표 {[round(v, 1) for v in pose[:3]]}")
            if not self.pick(pose):
                self.get_logger().error("파지 실패 — 중단")
                self.go_home()
                return False
            self.place(zone)
            self.go_home()
            self.get_logger().info(f"완료: {target}번구역의 {color} → {zone}번")
            push_zone("robot", target, None)     # 원래 있던 자리는 비었다
            push_zone("robot", zone, color)
            return True

        # 색을 지정한 경우. 관심 영역을 돌다가 그 색을 프리구역에서 보면 멈춘다.
        _, stock, _, at = self.patrol(want={target})
        lst = stock.get(target) or []
        if not lst:
            self.get_logger().error(f"프리구역에서 {target} 을(를) 못 찾았습니다.")
            self.go_home()
            return False
        xy = (lst[0]["pose"][0], lst[0]["pose"][1])
        self.get_logger().info(
            f"{target} 발견 ({xy[0]:.0f}, {xy[1]:.0f})"
            + (f" — {at}번 경유점" if at else ""))
        ok = self.pick_cached(target, xy, zone)
        self.go_home()
        if ok:
            push_zone("robot", zone, target)
        return ok

    def run_sign(self, once=False):
        """수어 한 문장 → LLM 해석 → 구역 배치. 기본은 반복이다.

        인식기 로딩(torch + mediapipe)이 몇 초 걸리므로 이 모드에서만 import 한다.
        다른 모드까지 그 비용을 물면 calib 처럼 여러 번 돌리는 작업이 답답해진다.
        """
        sys.path.insert(0, HERE)
        import sign_command as sc

        rec = sc.make_recognizer()
        zones = sorted(self.cfg["zones"])
        colors = ", ".join(sorted(sc.GLOSS_TO_COLOR))
        try:
            return self._sign_loop(sc, rec, zones, colors, once)
        finally:
            rec.close()            # 카메라와 창은 여기서 딱 한 번 닫는다

    def _sign_loop(self, sc, rec, zones, colors, once):
        while True:
            self.get_logger().info(
                f"수어로 명령하세요 — 예: 빨강 → {zones[0]}번구역  "
                f"(색: {colors} / 구역: {zones})")
            got = sc.collect_command(
                rec, zones,
                on_gloss=lambda l, p, b: self.get_logger().info(
                    f"  · {l} ({p * 100:.0f}%){'  [동사 가중]' if b else ''}"),
                hint=f"빨강 → {zones[0]}번구역")
            if got is None:                # 창에서 Q — 사용자가 그만두겠다는 뜻이다
                self.get_logger().info("취소됨")
                return False
            glosses, steps, how = got
            text = " ".join(glosses)
            if not steps:
                self.get_logger().warn(f"'{text}' 를 해석하지 못했습니다 — 다시 하세요.")
                if once:
                    return False
                continue
            self.get_logger().warn(
                f"[{how}] '{text}' → " +
                ", ".join(f"{sc.describe(s)}→{z}번" for s, z in steps))
            ok = True
            for src, zone in steps:
                if src == "copy":
                    # zone 자리에 mirror 여부가 실려 온다 (sign_command.COPY_GLOSS)
                    if not self.copy_human(mirror=bool(zone)):
                        ok = False
                    continue
                if not self.run_one(src, zone):
                    # 한 짝이 실패해도 나머지는 계속한다. 블록 하나를 못 찾았다고
                    # 방금 서명한 나머지 명령까지 버리면 다시 다 서명해야 한다.
                    self.get_logger().warn(
                        f"{sc.describe(src)} → {zone}번 실패 — 다음으로 넘어갑니다")
                    ok = False
            if once:
                return ok


def as_target(s):
    """숫자면 '그 구역에 있는 것', 아니면 색 이름. 구역 번호에 색 이름이 없으니 안 겹친다."""
    return int(s) if s.isdigit() else s


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    m = sys.argv[1]
    node = BlockSort()
    push_robot_status(connected=True, model="Doosan M0609",
                       last_command=" ".join(sys.argv[1:]))
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
                sys.exit("사용법: pick <색|집을구역> <놓을구역>")
            node.run_one(as_target(sys.argv[2]), int(sys.argv[3]))
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
        elif m in ("scan", "read-human", "read-free"):
            seen, stock, occ, _ = node.patrol()
            push_zones("human", seen, sorted(node.human_zones()))
            push_zones("robot", occ, sorted(node.cfg["zones"]))
            node.go_home()
            print(f"\n  인간구역 = {seen or '읽기 실패'}")
            counts = {c: len(v) for c, v in stock.items()}
            print(f"  프리 재고 = {counts}")
            print(f"  로봇구역 점유 = {occ or '비어 있음'}\n")
        elif m == "copy":
            node.copy_human(mirror=False)
        elif m == "copy-mirror":
            node.copy_human(mirror=True)
        elif m == "sign":
            node.run_sign(once="--once" in sys.argv)
        elif m == "run":
            print("\n  '<대상> <구역>' 입력.  q 로 종료")
            print(f"  구역: {sorted(node.cfg['zones'])}\n")
            while True:
                s = input("  > ").strip().split()
                if not s or s[0] == "q":
                    break
                if len(s) != 2 or not s[1].isdigit():
                    print("    형식: 빨간색 1   또는   3 1 (3번 구역의 것을 1번으로)")
                    continue
                node.run_one(as_target(s[0]), int(s[1]))
        else:
            sys.exit(__doc__)
    except Exception as e:
        push_debug("error", "block_sort", f"'{' '.join(sys.argv[1:])}' 실행 중 오류: {e}")
        raise
    finally:
        push_robot_status(connected=False)
        rclpy.shutdown()


if __name__ == "__main__":
    main()
