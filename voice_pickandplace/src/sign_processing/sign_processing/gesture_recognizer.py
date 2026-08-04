"""수어 한 동작을 인식하는 추론 전용 모듈.

hyupdong2/sign_control/sign_demo.py 의 특징 추출·모델·구간 인식·화면 표시 로직을
서비스 호출 한 번 = 동작 한 번 인식이라는 단위로 옮겨온 것이다.
녹화/학습은 sign_demo.py 쪽 책임으로 남겨두고 여기서는 다루지 않는다.
"""
import os
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

POSE_IDX = [0, 11, 12, 13, 14, 15, 16]        # 코, 양 어깨, 양 팔꿈치, 양 손목
FEAT_DIM = len(POSE_IDX) * 2 + 2 * (2 + 21 * 2 + 1)   # = 104
SEQ_LEN = 32

# 인식 단위는 '슬라이딩 창'이 아니라 '손이 올라와 있는 한 구간'이다.
# 학습 데이터가 한 클립 = 한 동작이므로 인식도 같은 단위로 끊어야 맞는다.
# (자세한 이유는 sign_control/README.md의 '인식 단위' 절 참고)
PRE_PAD = 6           # 손이 보이기 직전 프레임도 포함
END_HOLD = 5           # 손이 이만큼 연속으로 안 보이면 구간 종료
MIN_HAND = 10           # 손이 이보다 적게 보이면 동작으로 안 침
MAX_SEG = 96           # 손을 계속 들고 있어도 여기서 강제 종료
CONF_TH = 0.80          # 이 확률 넘어야 인정
RESULT_HOLD_SEC = 0.6   # 인식 성공 후 결과를 화면에 붙잡아 두는 시간

WIN = "sign_processing"

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
POSE_EDGES = [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6)]   # POSE_IDX 안에서의 인덱스

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]


def resample(seq, n=SEQ_LEN):
    """길이가 제각각인 시퀀스를 고정 길이로. 동작 속도 차이를 흡수한다."""
    seq = np.asarray(seq, np.float32)
    if len(seq) == n:
        return seq
    src = np.linspace(0, len(seq) - 1, len(seq))
    dst = np.linspace(0, len(seq) - 1, n)
    return np.stack([np.interp(dst, src, seq[:, i]) for i in range(seq.shape[1])], 1).astype(np.float32)


class _Net(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.gru = nn.GRU(FEAT_DIM, 128, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(256 * 2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_cls))

    def forward(self, x):
        y, _ = self.gru(x)
        return self.head(torch.cat([y.mean(1), y.max(1).values], -1))


class Extractor:
    """MediaPipe 포즈+손 -> 몸 기준으로 정규화된 특징 벡터."""

    def __init__(self, hand_task, pose_task):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        self.mp = mp

        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=pose_task),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5))
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=hand_task),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5))
        self._t = 0

    def __call__(self, bgr):
        """returns (feat[104] or None, draw_info={"pose": P|None, "hands": [H,...]})"""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        self._t += 33                       # VIDEO 모드는 증가하는 타임스탬프를 요구한다
        pres = self.pose.detect_for_video(img, self._t)
        hres = self.hands.detect_for_video(img, self._t)

        if not pres.pose_landmarks:
            return None, {"pose": None, "hands": []}

        lm = pres.pose_landmarks[0]
        P = np.array([[lm[i].x * w, lm[i].y * h] for i in POSE_IDX], np.float32)

        origin = (P[1] + P[2]) / 2.0
        scale = np.linalg.norm(P[1] - P[2])
        if scale < 1e-3:
            return None, {"pose": None, "hands": []}
        pose_f = ((P - origin) / scale).ravel()

        slots = {"Left": np.zeros(45, np.float32), "Right": np.zeros(45, np.float32)}
        draw_hands = []
        for pts, hd in zip(hres.hand_landmarks, hres.handedness):
            name = hd[0].category_name
            H = np.array([[p.x * w, p.y * h] for p in pts], np.float32)
            if name in slots:
                span = np.linalg.norm(H[0] - H[9]) + 1e-6      # 손목→중지 밑동
                slots[name] = np.concatenate([
                    (H[0] - origin) / scale,
                    ((H - H[0]) / span).ravel(),
                    [1.0],
                ]).astype(np.float32)
            draw_hands.append(H)

        feat = np.concatenate([pose_f, slots["Left"], slots["Right"]]).astype(np.float32)
        return feat, {"pose": P, "hands": draw_hands}


def draw_skeleton(img, info):
    P, hands = info["pose"], info["hands"]
    if P is not None:
        for a, b in POSE_EDGES:
            cv2.line(img, tuple(P[a].astype(int)), tuple(P[b].astype(int)), (90, 200, 90), 2)
        for p in P:
            cv2.circle(img, tuple(p.astype(int)), 4, (60, 255, 60), -1)
    for H in hands:
        for a, b in HAND_EDGES:
            cv2.line(img, tuple(H[a].astype(int)), tuple(H[b].astype(int)), (230, 170, 40), 2)
        for p in H:
            cv2.circle(img, tuple(p.astype(int)), 3, (60, 220, 255), -1)


class Painter:
    """cv2.putText는 한글을 못 그린다. PIL로 우회한다. 폰트가 없으면 None을 반환해 호출부가 영문으로 대체하게 한다."""

    @staticmethod
    def create():
        path = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
        if path is None:
            return None
        return Painter(path)

    def __init__(self, path):
        from PIL import ImageFont
        self._cache = {}
        self._path = path
        self._ImageFont = ImageFont

    def _font(self, size):
        if size not in self._cache:
            self._cache[size] = self._ImageFont.truetype(self._path, size)
        return self._cache[size]

    def text(self, img, xy, s, size=28, color=(255, 255, 255)):
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        f = self._font(size)
        x, y = xy
        d.text((x + 2, y + 2), s, font=f, fill=(0, 0, 0))
        d.text((x, y), s, font=f, fill=color[::-1])
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def open_cam(index, width=1280, height=720):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 {index} 를 열 수 없습니다.")
    for _ in range(5):
        cap.read()
    return cap


class SignRecognizer:
    """모델과 MediaPipe는 한 번만 로드하고, 카메라는 recognize_one() 호출마다 열고 닫는다."""

    def __init__(self, model_path, hand_task, pose_task, cam_index, show_window=True):
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        self.labels = ck["labels"]
        self.net = _Net(len(self.labels))
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.ext = Extractor(hand_task, pose_task)
        self.cam_index = cam_index
        self.show_window = show_window
        self.painter = Painter.create() if show_window else None

    def _classify(self, seg):
        x = torch.tensor(resample(seg)[None])
        with torch.no_grad():
            p = torch.softmax(self.net(x)[0], -1).cpu().numpy()
        i = int(p.argmax())
        return self.labels[i], float(p[i])

    def _status_text(self, frame, text, color):
        if self.painter:
            return self.painter.text(frame, (20, 20), text, 28, color)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame

    def recognize_one(self):
        """손을 올렸다 내리는 동작 한 번을 기다려 (라벨, 확률)을 반환한다.
        확신도가 CONF_TH 미만이면 결과를 버리고 계속 기다린다 — 지나가는 손짓을 걸러낸다."""
        cap = open_cam(self.cam_index)
        try:
            pre = deque(maxlen=PRE_PAD)
            seg, gone, n_hand = [], 0, 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                feat, draw_info = self.ext(frame)
                has_hand = bool(draw_info["hands"])

                if self.show_window:
                    disp = frame.copy()
                    draw_skeleton(disp, draw_info)
                    if seg:
                        disp = self._status_text(disp, f"수집중 {n_hand}", (80, 80, 255))
                    elif has_hand:
                        disp = self._status_text(disp, "손 인식됨", (120, 255, 120))
                    else:
                        disp = self._status_text(disp, "대기 - 손을 올리세요", (200, 200, 200))
                    cv2.imshow(WIN, disp)
                    cv2.waitKey(1)

                if feat is None:
                    continue

                pre.append(feat)
                if not seg:
                    if has_hand:
                        seg, gone, n_hand = list(pre), 0, 1
                    continue

                seg.append(feat)
                if has_hand:
                    gone, n_hand = 0, n_hand + 1
                else:
                    gone += 1

                if gone >= END_HOLD or len(seg) >= MAX_SEG:
                    if n_hand >= MIN_HAND:
                        label, prob = self._classify(seg)
                        if prob >= CONF_TH:
                            if self.show_window:
                                ok_color = (120, 255, 120)
                                shown = self._status_text(frame.copy(), f"{label} {prob*100:.0f}%", ok_color)
                                cv2.imshow(WIN, shown)
                                cv2.waitKey(1)
                                time.sleep(RESULT_HOLD_SEC)
                            return label, prob
                    seg, gone, n_hand = [], 0, 0
        finally:
            cap.release()
            if self.show_window:
                cv2.destroyWindow(WIN)
