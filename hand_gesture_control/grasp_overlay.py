#!/usr/bin/env python3
"""로봇 시점 화면에 **파지 지점**을 겹쳐 그린다 (증강현실).

조종할 때 로봇이 보는 화면만으로는 "지금 내리면 물릴까" 를 알 수 없다.
손가락이 화면 어디에 있는지, 블록이 그 사이에 들어왔는지를 그려 준다.

    평소        빨간 네모  — 손가락 사이(파지 구역)
    블록 들어옴  초록 네모 + "파지 가능"

## 왜 로봇 자세가 필요 없나

카메라가 그리퍼에 붙어 있다(핸드아이). 그리퍼에 고정된 점은 팔이 어디로 가든
**화면의 같은 픽셀**에 찍힌다 — TCP 는 실측 (798, 649) 근처다(1280x720).
그래서 파지 구역을 그리는 데는 로봇 자세도 base 좌표도 필요 없다.
DSR 연결을 또 만들지 않아도 된다는 뜻이다(그게 어제 문제의 근원이었다).

바뀌는 것은 둘뿐이다.
    · 손가락 벌어진 폭   → TF 의 rg2_*_inner_finger 위치에서 그대로 읽는다
    · 블록이 얼마나 아래  → 깊이 영상에서 읽는다

## 파지 가능 판정은 2D 로 하면 안 된다

화면에서 겹쳐 보이는 것과 손가락 사이에 있는 것은 다르다. 카메라와 TCP 가
34mm 어긋나 있어(핸드아이) 깊이가 다르면 같은 픽셀도 실제로는 옆이다 —
검출 쪽에서 광축 정렬에 공들이는 이유가 그 시차다.
그래서 깊이로 블록의 3차원 위치를 구해 **TCP 좌표계에서** 판정한다.

    x  손가락이 닫히는 축   |x| < 벌어진 폭/2 - 여유   (블록이 사이에 들어오는가)
    y  손가락 길이 축       |y| < 패드 절반 길이       (패드에 걸치는가)
    z  하강 방향(아래 +)    0 < z < REACH             (내려가면 닿는 거리인가)
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(HERE)

# 핸드아이 행렬 (TCP 기준 카메라 자세). block_sort 와 **같은 파일**을 쓴다 —
# 두 벌이 되면 한쪽만 재보정했을 때 화면과 실제 파지가 갈라진다.
HANDEYE = os.environ.get("HANDEYE", os.path.join(
    _ROOT, "voice_pickandplace", "src", "robot_control", "resource",
    "T_gripper2camera.npy"))

# TCP 길이(mm). TF 는 플랜지(link_6) 기준으로 손가락 위치를 주고, 핸드아이는
# TCP 기준이라 그 차이만큼 옮겨야 한다. block_sort 의 EXPECT_TCP_MM 과 같은 값.
TCP_LEN = float(os.environ.get("EXPECT_TCP_MM", 250.0))

# 파지 구역 여유(mm). 블록이 손가락 폭에 딱 맞게 들어와도 '가능' 이라 하면
# 실제로는 모서리를 물거나 밀어낸다. 양쪽으로 이만큼 남아야 가능으로 본다.
GRASP_MARGIN = float(os.environ.get("GRASP_MARGIN", 6.0))

# 손가락 패드의 절반 길이(mm). 이 방향으로 블록이 벗어나면 패드 끝에 걸린다.
PAD_HALF = float(os.environ.get("GRASP_PAD_HALF", 12.0))

# 내려가서 닿을 수 있다고 보는 최대 거리(mm). 이보다 멀면 아직 파지 자세가 아니다.
REACH = float(os.environ.get("GRASP_REACH", 350.0))

# 블록 한 변(mm). 사이에 들어오는지 볼 때 이 절반을 쓴다.
BLOCK_MM = float(os.environ.get("BLOCK_MM", 35.0))

RED = (60, 60, 235)
GREEN = (90, 230, 120)
GRAY = (150, 150, 150)

# 한글 글자. cv2.putText 는 한글을 '????' 로 그리므로 PIL 로 얹는다.
# 후보 목록은 sign_processing 의 FONT_CANDIDATES 와 같다 — 한 대에서 되던 것이
# 다른 대에서 안 되면 이 목록에 없는 폰트만 깔린 경우다.
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf",
    "/usr/share/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]


def _font(size=26):
    """한글 폰트 하나. 없으면 None — 그때는 영문으로 그린다."""
    path = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        return None
    try:
        from PIL import ImageFont
        return ImageFont.truetype(path, size)
    except Exception:
        return None


_FONT = [False, None]      # (한 번 찾아봤나, 폰트)


def put_text(frame, text, org, color, size=26, ascii_fallback=None):
    """프레임에 글자를 얹는다. 한글이 되면 한글로, 안 되면 영문으로.

    폰트 찾기는 한 번만 한다 — 매 프레임 파일을 뒤지면 12fps 가 아깝다.
    """
    import cv2
    if not _FONT[0]:
        _FONT[0], _FONT[1] = True, _font(size)
    if _FONT[1] is None:
        import cv2 as _c
        _c.putText(frame, ascii_fallback or text, org, _c.FONT_HERSHEY_SIMPLEX,
                   0.7, color, 2, _c.LINE_AA)
        return frame
    from PIL import Image, ImageDraw
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img).text(org, text, font=_FONT[1],
                             fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame


def load_handeye(path=None):
    """TCP 기준 카메라 자세 T (4x4, mm)."""
    return np.load(path or HANDEYE)


class GraspView:
    """파지 구역을 화면에 그리고, 블록이 들어왔는지 판정한다.

    intrinsics 는 camera_info 에서 받은 {"fx","fy","ppx","ppy"} 다.
    없으면 그리지 않는다 — 짐작한 값으로 그리면 화면이 거짓말을 한다.
    """

    def __init__(self, handeye=None, intr=None):
        T = load_handeye() if handeye is None else handeye
        self.R = np.array(T[:3, :3], float)      # TCP → 카메라 회전
        self.t = np.array(T[:3, 3], float)       # TCP 에서 카메라까지(mm)
        self.intr = intr
        # 손가락 반폭(mm). TF 를 못 읽으면 이 값으로 그린다(RG2 최대 개구 근처).
        self.half_open = 55.0

    # ── 좌표 ──
    def to_cam(self, p_tcp):
        """TCP 좌표(mm) → 카메라 좌표(mm)."""
        return self.R.T @ (np.asarray(p_tcp, float) - self.t)

    def to_tcp(self, p_cam):
        """카메라 좌표(mm) → TCP 좌표(mm)."""
        return self.R @ np.asarray(p_cam, float) + self.t

    def project(self, p_cam):
        """카메라 좌표(mm) → 픽셀. 카메라 뒤(z<=0)면 None."""
        if self.intr is None or p_cam[2] <= 1.0:
            return None
        i = self.intr
        return (i["fx"] * p_cam[0] / p_cam[2] + i["ppx"],
                i["fy"] * p_cam[1] / p_cam[2] + i["ppy"])

    def unproject(self, u, v, depth_mm):
        """픽셀 + 깊이(mm) → 카메라 좌표(mm)."""
        if self.intr is None or depth_mm <= 0:
            return None
        i = self.intr
        return np.array([(u - i["ppx"]) * depth_mm / i["fx"],
                         (v - i["ppy"]) * depth_mm / i["fy"],
                         float(depth_mm)])

    # ── 파지 구역 ──
    def set_opening(self, half_open_mm):
        """지금 손가락 반폭(mm). TF 에서 읽은 값을 넣는다."""
        if half_open_mm and half_open_mm > 0:
            self.half_open = float(half_open_mm)

    def corridor_quad(self, z_mm):
        """하강 z 만큼 내려간 자리의 파지 구역 네 꼭짓점(픽셀).

        구역은 TCP 좌표계에서 손가락 사이 사각형이다. z 를 키우면 아래쪽(판에
        가까운 쪽) 단면이 되고, 카메라와 TCP 가 어긋나 있어 화면에서 위치가
        옮겨간다 — 그 시차가 곧 "지금 높이에서 내리면 어디에 닿는가" 다.
        """
        hw = self.half_open - GRASP_MARGIN
        pts = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            uv = self.project(self.to_cam([sx * hw, sy * PAD_HALF, z_mm]))
            if uv is None:
                return None
            pts.append(uv)
        return pts

    def graspable(self, p_cam):
        """이 점(카메라 좌표)이 지금 손가락 사이에 있는가. (가능?, 이유)"""
        p = self.to_tcp(p_cam)
        half = BLOCK_MM / 2.0
        if not (0.0 < p[2] < REACH):
            return False, f"거리 {p[2]:.0f}mm"
        if abs(p[0]) > self.half_open - GRASP_MARGIN - half:
            return False, f"좌우 {p[0]:+.0f}mm"
        if abs(p[1]) > PAD_HALF + half:
            return False, f"앞뒤 {p[1]:+.0f}mm"
        return True, f"{p[2]:.0f}mm 아래"


def load_detector():
    """검출 노드가 쓰는 **그 색 판정**을 그대로 빌려온다. (도구, 실패이유)

    여기에 HSV 범위를 다시 적으면 조명이 바뀌어 색범위를 고칠 때 화면과 로봇이
    서로 다른 색을 보게 된다. 워크스페이스를 소싱하지 않으면 못 불러오는데,
    그때도 파지 구역은 그려야 한다 — 초록 판정만 빠진다.
    """
    try:
        from object_detection.color_model import (COLOR_HSV_RANGES, MAX_AREA_RATIO,
                                                  detect_color_boxes)
    except Exception as e:
        return None, str(e)
    # 같은 범위가 한글·영문 두 열쇠로 들어 있다. 한글만 쓴다 — 두 번 훑을 이유가 없다.
    colors = [(k, v) for k, v in COLOR_HSV_RANGES.items() if not k.isascii()]
    return (colors, MAX_AREA_RATIO, detect_color_boxes), None


def depth_at(depth, cx, cy, half=4):
    """그 점 주변 깊이의 중앙값(mm). 0(측정 실패)은 뺀다. 없으면 None.

    한 픽셀만 읽으면 구멍에 걸려 0 이 나오거나 옆면 값을 집는다.
    """
    h, w = depth.shape[:2]
    x0, x1 = max(int(cx) - half, 0), min(int(cx) + half + 1, w)
    y0, y1 = max(int(cy) - half, 0), min(int(cy) + half + 1, h)
    patch = depth[y0:y1, x0:x1]
    vals = patch[patch > 0]
    if vals.size < 4:
        return None
    return float(np.median(vals))


def find_blocks(view, frame_bgr, depth, det):
    """화면에 보이는 블록마다 (카메라좌표 mm, 색이름). 깊이를 못 읽은 것은 버린다."""
    import cv2
    if det is None or depth is None or view.intr is None:
        return []
    colors, max_ratio, detect = det
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    max_area = frame_bgr.shape[0] * frame_bgr.shape[1] * max_ratio
    out = []
    for name, ranges in colors:
        try:
            found, _stats = detect(hsv, ranges, max_area, depth)
        except Exception:
            continue
        for _box, _ang, cx, cy in found:
            d = depth_at(depth, cx, cy)
            if d is None:
                continue
            p = view.unproject(cx, cy, d)
            if p is not None:
                out.append((p, name))
    return out


def draw(frame, view, blocks, z_default=200.0):
    """파지 구역과 판정을 프레임에 겹쳐 그린다. (그린 상태, 대상) 을 돌려준다.

    blocks 는 [(카메라좌표, 색이름), ...]. 비어 있으면 기본 거리에 빨간 네모만.
    가능한 블록이 있으면 그 블록 높이에 초록 네모를 그리고 "파지 가능" 을 띄운다.
    """
    import cv2

    hit = None
    for p_cam, name in blocks:
        ok, why = view.graspable(p_cam)
        if ok:
            hit = (p_cam, name, why)
            break

    z = view.to_tcp(hit[0])[2] if hit else z_default
    quad = view.corridor_quad(z)
    if quad is None:
        return False, None

    col = GREEN if hit else RED
    pts = np.array(quad, np.int32)
    cv2.polylines(frame, [pts], True, col, 2, cv2.LINE_AA)
    # 네 귀퉁이만 굵게 — 안이 비어 보여야 블록이 가려지지 않는다
    for (x, y) in pts:
        cv2.circle(frame, (int(x), int(y)), 4, col, -1, cv2.LINE_AA)
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    cv2.line(frame, (cx - 12, cy), (cx + 12, cy), col, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 12), (cx, cy + 12), col, 1, cv2.LINE_AA)

    # 글자는 구역 **위쪽 변 바깥**에 올린다. 안에 쓰면 정작 봐야 하는 블록을
    # 가리고, 아래에 쓰면 판에 놓인 다른 블록을 가린다.
    top = int(min(p[1] for p in pts))
    if hit:
        _, name, why = hit
        txt, alt = f"파지 가능 — {name} ({why})", f"GRASP OK - {name} ({why})"
    else:
        txt, alt = "파지 구역 비어 있음", "no block in grasp zone"
    put_text(frame, txt, (max(cx - 150, 10), max(top - 34, 4)), col,
             ascii_fallback=alt)
    cv2.putText(frame, f"opening {view.half_open * 2:.0f}mm", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY, 1, cv2.LINE_AA)
    return bool(hit), (hit[1] if hit else None)
