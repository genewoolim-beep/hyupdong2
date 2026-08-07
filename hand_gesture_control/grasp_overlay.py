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
    _ROOT, "calib", "T_gripper2camera.npy"))

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
    # 글자 폭을 재서 화면 안으로 물린다. 세로로 돌린 화면은 폭이 720 뿐이라
    # 물리지 않으면 오른쪽이 잘려 나간다("… (149m" 처럼).
    try:
        tw = _FONT[1].getlength(text)
    except Exception:
        tw = len(text) * size * 0.6
    x = min(max(float(org[0]), 4.0), max(frame.shape[1] - tw - 6.0, 4.0))
    ImageDraw.Draw(img).text((x, org[1]), text, font=_FONT[1],
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
        # 핸드아이 T 는 **카메라 좌표를 TCP 좌표로** 옮기는 행렬이다.
        #   p_tcp = R · p_cam + t
        # block_sort 가 base2gripper @ gripper2cam 으로 쓰는 방향이 그것이다.
        # 여기서는 그 반대 방향(TCP → 카메라)도 쓰므로 전치를 취한다(to_cam).
        # 방향을 거꾸로 적으면 파지 구역이 엉뚱한 데 그려지고, 그건 사람이
        # 그 자리로 팔을 가져가게 만든다.
        T = load_handeye() if handeye is None else handeye
        self.R = np.array(T[:3, :3], float)      # 카메라 → TCP 회전
        self.t = np.array(T[:3, 3], float)       # TCP 좌표로 본 카메라 위치(mm)
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


def depth_from_size(box, ang, fx):
    """블록의 화면 크기로 거리를 추정한다(mm). 못 하면 None.

    **깊이 센서가 가까이서 죽는 것을 메운다.** D435i 의 최소 측정거리는
    1280x720 에서 약 280mm 인데, 그리퍼가 파지 높이로 내려가면 블록 윗면은
    카메라에서 200mm 안쪽으로 들어온다 — 깊이가 0 이 되어 블록이 통째로
    사라진다(실측 2026-08-07: 내려가면 파지 구역 추적이 끊겼다).
    한 변이 35mm 인 것을 아니까, 화면에서 몇 픽셀인지 보면 거리가 나온다.
    가까울수록 크게 잡혀 이 추정은 **가까이서 더 정확하다** — 깊이가 약한
    바로 그 구간이다.

    축정렬 박스는 블록이 기울면 커진다(45°에서 1.41배). 검출이 함께 주는
    회전각으로 되돌린다: AABB 한 변 = 실제 한 변 × (|cos|+|sin|).
    """
    import numpy as _np
    x0, y0, x1, y1 = box
    side_px = min(x1 - x0, y1 - y0)
    if side_px <= 2:
        return None
    r = _np.radians(float(ang))
    scale = abs(_np.cos(r)) + abs(_np.sin(r))
    side_px = side_px / max(scale, 1e-6)
    return float(fx * BLOCK_MM / side_px)


def landing_z(view, depth, z_default=200.0, iters=3, tol=400.0):
    """**그대로 수직 하강하면 닿는 지점**까지의 거리(mm, TCP 기준 z). 못 재면 None.

    파지 구역을 여기에 그린다 — 블록을 따라다니는 표시가 아니라 "지금 내리면
    여기 닿는다" 는 표시여야 조종에 쓸 수 있다. 판이든 블록이든 먼저 닿는 것에
    찍힌다.

    푸는 방법: 하강축 위의 점 (0,0,t) 가 찍히는 픽셀은 t 에 따라 옮겨간다
    (카메라와 TCP 가 34mm 어긋난 시차). 그래서 한 번에 못 구하고 되짚는다 —
    기본 거리로 픽셀을 잡고, 그 픽셀의 깊이로 t 를 고치고, 다시 픽셀을 잡는다.
    세 번이면 수렴한다(시차가 작아 첫 보정에서 거의 맞는다).

    깊이가 없으면 None 이다. 가까이서는 센서 최소거리(약 280mm) 밑으로 들어가
    깊이가 통째로 0 이 되므로, 그때는 부르는 쪽이 블록 크기 추정으로 넘어간다.
    """
    if depth is None or view.intr is None:
        return None
    t = float(z_default)
    for _ in range(iters):
        uv = view.project(view.to_cam([0.0, 0.0, t]))
        if uv is None:
            return None
        d = depth_at(depth, uv[0], uv[1])
        if d is None:
            return None
        p = view.unproject(uv[0], uv[1], d)
        if p is None:
            return None
        t_new = float(view.to_tcp(p)[2])
        if not (0.0 < t_new < tol):
            return None
        if abs(t_new - t) < 0.5:
            t = t_new
            break
        t = t_new
    return t


def find_blocks(view, frame_bgr, depth, det):
    """화면에 보이는 블록마다 (카메라좌표 mm, 색이름, 거리출처).

    거리는 깊이 영상이 우선이고, 없으면 블록 크기로 추정한다 — 가까이서 깊이가
    죽는 구간을 메운다(depth_from_size). 둘 다 안 되면 그 블록은 버린다.
    """
    import cv2
    if det is None or view.intr is None:
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
        for box, ang, cx, cy in found:
            d = None if depth is None else depth_at(depth, cx, cy)
            how = "깊이"
            if d is None:
                d = depth_from_size(box, ang, view.intr["fx"])
                how = "크기"
            if d is None:
                continue
            p = view.unproject(cx, cy, d)
            if p is not None:
                out.append((p, name, how))
    return out


def _rot_map(rotate, src_shape):
    """소스 픽셀 (u,v) → 돌린 영상의 픽셀. cv2.rotate 와 같은 규약이다."""
    h, w = src_shape[:2]
    if rotate == 90:                                  # 반시계
        return lambda u, v: (v, (w - 1) - u)
    if rotate == 180:
        return lambda u, v: ((w - 1) - u, (h - 1) - v)
    if rotate == 270:                                 # 시계
        return lambda u, v: ((h - 1) - v, u)
    return lambda u, v: (u, v)


_ROT = {}


def render(frame, view, blocks, depth=None, rotate=0, z_default=200.0):
    """AR 을 겹치고 화면을 rotate 만큼 돌려서 돌려준다. (돌린 프레임, 초록인가, 대상)

    **기하는 소스 방향에서 계산하고, 그리기만 돌린 좌표에서 한다.**
    다 그린 뒤 영상을 돌리면 "파지 가능" 글자까지 같이 눕는다. 반대로 영상을
    먼저 돌리고 기하를 다시 풀면 내부파라미터와 깊이 영상까지 돌려야 해서
    검증된 계산이 통째로 흔들린다. 그래서 **점만 옮긴다**(_rot_map).
    """
    import cv2
    if not _ROT:
        _ROT.update({90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180,
                     270: cv2.ROTATE_90_CLOCKWISE})
    rotate = int(rotate) % 360
    if rotate not in _ROT:
        hit, name = draw(frame, view, blocks, depth, z_default)
        return frame, hit, name
    out = cv2.rotate(frame, _ROT[rotate])
    hit, name = draw(out, view, blocks, depth, z_default,
                     xy=_rot_map(rotate, frame.shape))
    return out, hit, name


def draw(frame, view, blocks, depth=None, z_default=200.0, xy=None):
    """**그대로 내려가면 닿는 지점**에 파지 구역을 그린다. (초록인가, 대상)

    네모는 하강축이 먼저 닿는 표면(판이든 블록이든)에 찍힌다 — 블록을 따라다니는
    표시가 아니다. 색은 그 구역 안에 물 수 있는 블록이 있는가로 갈린다.

        빨강  구역 안에 물 만한 블록이 없다
        초록  있다 → "파지 가능"

    blocks 는 find_blocks() 결과, depth 는 정렬된 깊이(없어도 된다).
    xy 는 소스 픽셀을 그릴 좌표로 옮기는 함수 — 화면을 돌려 그릴 때 쓴다
    (render 참고). frame 은 그때 **이미 돌아간** 영상이고, depth·기하 계산은
    소스 방향 그대로다.
    """
    import cv2

    # 여럿이면 **가장 잘 들어온 것**을 고른다. 목록 순서(색 순서)로 고르면
    # 화면 구석의 빨간 블록 때문에 정작 손가락 밑의 파란 블록을 놓친다.
    hit = None
    best = None
    for b in blocks:
        p_cam, name, how = (b if len(b) == 3 else (b[0], b[1], "깊이"))
        ok, why = view.graspable(p_cam)
        if not ok:
            continue
        p = view.to_tcp(p_cam)
        score = abs(p[0]) + abs(p[1])          # 중심에서 벗어난 정도
        if best is None or score < best:
            best, hit = score, (p_cam, f"{name}·{how}", why)

    # 네모를 그릴 자리 = **그대로 내려가면 닿는 지점.** 블록을 따라다니는 표시가
    # 아니다 — 블록이 없어도 판 위에 찍혀야 "지금 내리면 여기" 가 된다.
    #
    #   ① 깊이로 하강축이 먼저 닿는 표면을 찾는다 (landing_z)
    #   ② 깊이가 죽는 가까운 거리에서는 보이는 블록의 크기 추정으로 대신한다
    #   ③ 둘 다 없으면 기본 거리 — 그래도 네모는 항상 보여야 한다
    z = landing_z(view, depth, z_default)
    src = "깊이"
    if z is None:
        near = [t for t in (view.to_tcp(b[0])[2] for b in blocks) if 0.0 < t < REACH]
        if near:
            z, src = min(near), "크기"
        else:
            z, src = z_default, "추정없음"
    quad = view.corridor_quad(z)
    if quad is None:
        return False, None

    col = GREEN if hit else RED
    # 소스 좌표로 계산한 네 점을 그릴 좌표로 옮긴다(회전 없으면 그대로).
    pts = np.array([xy(u, v) for (u, v) in quad] if xy else quad, np.int32)
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
        # 빨강일 때도 **얼마나 내려가면 닿는지**는 알려준다. 그것이 이 표시의
        # 본래 목적이다 — 초록이 아니어도 조종에 쓰인다.
        d = f"{z:.0f}mm 아래" if src != "추정없음" else "거리 모름"
        txt, alt = f"착지점 {d} — 물 블록 없음", f"landing {d} - no block"
    put_text(frame, txt, (max(cx - 150, 10), max(top - 34, 4)), col,
             ascii_fallback=alt)
    cv2.putText(frame, f"opening {view.half_open * 2:.0f}mm", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY, 1, cv2.LINE_AA)
    return bool(hit), (hit[1] if hit else None)
