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
# 이 확률을 넘어야 한 동작으로 인정한다. 0.80 은 지나가는 손짓을 잘 걸러내지만
# 제대로 한 동작도 자주 흘렸다(실측에서 80~91% 에 몰려 있어 여유가 없었다).
# 0.60 으로 내리면 덜 흘리는 대신 오인식이 늘어난다 — 그 대가는 문장이 될 때까지
# 계속 모으는 방식과 화면의 글로스 패널이 받아낸다. 잘못 들어가면 C 로 지운다.
CONF_TH = float(os.environ.get("SIGN_CONF", 0.60))
RESULT_HOLD_SEC = 0.6   # 인식 성공 후 결과를 화면에 붙잡아 두는 시간

WIN = "sign_processing"

# 품사. 명사 뒤에 동사가 오면 '말이 되는 짝'으로 보고 파랗게 칠한다.
# 여기 없는 단어는 명사로 취급한다 — 구역(1번구역…)과 물체(나사/망치)가 그렇다.
# sign_demo.py 의 WORD_KIND 와 같은 표다. 한쪽만 고치면 데모와 로봇이 달라진다.
WORD_KIND = {
    "빨강": "수식", "주황": "수식", "노랑": "수식",
    "초록": "수식", "파랑": "수식", "보라": "수식",
    "놓다": "동사", "들다": "동사", "조립하다": "동사", "가져오다": "동사",
}
# 명사 다음에는 동사가 올 확률이 높다는 사전지식. 곱셈+재정규화로 하면 이미
# 확률이 몰린 1등이 거의 안 움직여(0.89→0.893) 정작 필요한 경계 구간에서
# 효과가 없다. 그래서 단순 덧셈으로 한다. (sign_demo.py 와 같은 이유·같은 값)
VERB_BONUS = 0.05

BLUE = (255, 170, 60)   # BGR — 짝지어진 명사·동사
PANEL_H = 185           # 하단 패널 높이 (모은 글로스를 여기에 쌓아 보여준다)
PEEK_EVERY = 3          # 수집 중 중간 결과를 몇 프레임마다 갱신할지


def kind_of(word):
    return WORD_KIND.get(word, "명사")


def paired_idx(gl):
    """명사 바로 뒤에 동사가 오는 자리를 찾는다. 그 두 개가 한 덩어리 명령이다."""
    out = set()
    for i in range(len(gl) - 1):
        if kind_of(gl[i]) == "명사" and kind_of(gl[i + 1]) == "동사":
            out.update((i, i + 1))
    return out

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

    def width(self, s, size):
        return self._font(size).getlength(s)

    def render(self, img, items):
        """[(xy, [(글자, 색), ...], 크기, 자간), ...] 를 PIL 왕복 '한 번'으로 전부 그린다.

        1280x720 에서 왕복 1회가 5ms다. 항목마다 따로 부르면 7회 = 36ms 로
        MediaPipe(25ms)보다 비싸져 프레임률이 반토막 난다. (sign_demo.py 와 동일)
        """
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for xy, parts, size, gap in items:
            f = self._font(size)
            x, y = xy
            for s, color in parts:
                d.text((x + 2, y + 2), s, font=f, fill=(0, 0, 0))
                d.text((x, y), s, font=f, fill=color[::-1])
                x += f.getlength(s) + gap
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def text(self, img, xy, s, size=28, color=(255, 255, 255)):
        return self.render(img, [(xy, [(s, color)], size, 0)])


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
        self._cap = None            # 세션 내내 열어 둔다. close() 로만 닫힌다.
        self._verb_mask = np.array(
            [VERB_BONUS if kind_of(l) == "동사" else 0.0 for l in self.labels],
            np.float32)

    def _classify(self, seg, prev=None):
        """prev 는 직전에 인정된 글로스. 명사였으면 동사 쪽에 가산점을 준다."""
        x = torch.tensor(resample(seg)[None])
        with torch.no_grad():
            p = torch.softmax(self.net(x)[0], -1).cpu().numpy()
        boosted = prev is not None and kind_of(prev) == "명사"
        if boosted:
            p = np.minimum(p + self._verb_mask, 1.0)
        i = int(p.argmax())
        return self.labels[i], float(p[i]), boosted

    def _status_text(self, frame, text, color):
        if self.painter:
            return self.painter.text(frame, (20, 20), text, 28, color)
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame

    def recognize_sentence(self, is_end, on_gloss=None, max_glosses=8, hint=""):
        """문장 하나가 될 때까지 카메라 루프 한 번으로 계속 인식한다.

        recognize_one() 을 반복 호출하는 것과 다른 점 셋이다.
          · 카메라를 한 번만 연다. 글로스마다 열고 닫으면 매번 1초 넘게 버려지고,
            그 사이에 한 동작이 통째로 사라진다.
          · 지금까지 모은 글로스를 화면 아래 패널에 쌓아 보여준다. 무엇이
            들어갔는지 안 보이면 왜 실패했는지 알 수가 없다.
          · 직전 글로스가 명사면 동사에 가중치를 준다.
        어디서 끝낼지는 호출자가 is_end(glosses) 로 정한다 — 어휘마다 문장
        구조가 다르기 때문이다.  q 로 취소하면 None 을 돌려준다.
        """
        cap = self._camera()
        try:
            pre = deque(maxlen=PRE_PAD)
            seg, gone, n_hand = [], 0, 0
            glosses = []
            cur, prob, boosted = "", 0.0, False
            peek, tick = "", 0
            fps, t_prev = 0.0, time.time()
            miss = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    # 오래 안 읽다 보면(로봇이 움직이는 동안) 장치가 삐끗할 때가
                    # 있다. 몇 번 연속 실패하면 조용히 다시 연다.
                    miss += 1
                    if miss >= 30:
                        cap = self._camera(reopen=True)
                        miss = 0
                    continue
                miss = 0
                frame = cv2.flip(frame, 1)
                feat, info = self.ext(frame)
                has_hand = bool(info["hands"])
                prev = glosses[-1] if glosses else None

                tick += 1
                if feat is not None:
                    pre.append(feat)
                    if not seg:                      # 손이 올라오면 구간 시작
                        if has_hand:
                            seg, gone, n_hand, peek = list(pre), 0, 1, ""
                    else:
                        seg.append(feat)
                        if has_hand:
                            gone, n_hand = 0, n_hand + 1
                        else:
                            gone += 1                # 손이 내려간 뒤에도 잠깐 더 담는다

                        # 분류가 0.5ms뿐이라 수집 중에도 슬쩍 돌려 중간 결과를 보여준다.
                        # 확정에는 쓰지 않는다 — 화면이 멈춘 듯 보이지 않게 하는 용도다.
                        if n_hand >= MIN_HAND and tick % PEEK_EVERY == 0:
                            peek = self._classify(seg, prev)[0]

                        if gone >= END_HOLD or len(seg) >= MAX_SEG:
                            if n_hand >= MIN_HAND:
                                cur, prob, boosted = self._classify(seg, prev)
                                if prob >= CONF_TH:
                                    glosses.append(cur)
                                    if on_gloss:
                                        on_gloss(cur, prob, boosted)
                            seg, gone, n_hand, peek = [], 0, 0, ""
                            if glosses and (is_end(glosses) or len(glosses) >= max_glosses):
                                if self.show_window:
                                    self._draw(frame, info, glosses, cur, prob,
                                               boosted, seg, n_hand, peek, fps, hint)
                                    cv2.waitKey(1)
                                    time.sleep(RESULT_HOLD_SEC)
                                return glosses

                if self.show_window:
                    self._draw(frame, info, glosses, cur, prob, boosted,
                               seg, n_hand, peek, fps, hint)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        return None
                    if k == ord("c"):
                        glosses, cur, prob = [], "", 0.0

                now = time.time()
                fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
                t_prev = now
        finally:
            # 카메라도 창도 닫지 않는다. 문장마다 닫으면 로봇이 움직이고 돌아올
            # 때마다 창이 사라졌다 다시 뜨고, 여는 데만 1초 넘게 걸린다.
            # 정리는 close() 를 부르는 쪽에서 한 번만 한다.
            pass

    def _camera(self, reopen=False):
        """세션 내내 쓰는 카메라. 처음 부를 때 열고, 그 뒤로는 그대로 돌려준다."""
        if reopen and self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._cap is None:
            self._cap = open_cam(self.cam_index)
            if self.show_window:
                cv2.namedWindow(WIN)
        else:
            # 로봇이 움직이는 동안 아무도 안 읽어 큐에 낡은 장면이 남아 있다.
            # 그대로 쓰면 방금 끝난 동작을 다시 인식할 수 있어 몇 장 버린다.
            for _ in range(5):
                self._cap.read()
        return self._cap

    def close(self):
        """카메라와 창을 정리한다. 세션이 끝날 때 한 번만 부르면 된다."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self.show_window:
            cv2.destroyWindow(WIN)

    def _draw(self, frame, info, glosses, cur, prob, boosted,
              seg, n_hand, peek, fps, hint):
        """cv2 도형을 먼저 다 그리고, 글자는 마지막에 한 번에 (sign_demo 와 같은 순서)."""
        h, w = frame.shape[:2]
        panel = min(PANEL_H, h // 3)     # 작은 해상도에서 화면을 다 덮지 않게
        draw_skeleton(frame, info)
        cv2.rectangle(frame, (0, h - panel), (w, h), (25, 25, 25), -1)

        if self.painter is None:         # 한글 폰트가 없으면 글로스만 영문 좌표로
            cv2.imshow(WIN, frame)
            return

        items = []
        if cur:
            c = (120, 255, 120) if prob >= CONF_TH else (140, 140, 140)
            items.append(((30, 30), [(f"{cur}  {prob * 100:.0f}%", c)], 40, 0))
            if boosted:
                items.append(((32, 84), [(f"동사 가중 +{VERB_BONUS * 100:.0f}%", BLUE)], 22, 0))
        if seg:
            tail = f"  ~{peek}" if peek else ""
            items.append(((w - 360, 34), [(f"● 수집중 {n_hand}{tail}", (80, 80, 255))], 24, 0))
        else:
            items.append(((w - 360, 34), [("대기 — 손을 올리세요", (120, 255, 120))], 24, 0))
        items.append(((w - 130, h - panel - 34), [(f"{fps:.0f} fps", (150, 150, 150))], 22, 0))

        pair = paired_idx(glosses)
        items.append(((30, h - panel + 12),
                      [(g, BLUE if i in pair else (255, 255, 255))
                       for i, g in enumerate(glosses)] or [("…", (110, 110, 110))],
                      34, 14))
        if hint:
            items.append(((30, h - panel + 66), [(hint, (120, 255, 255))], 24, 0))
        items.append(((30, h - panel + 104),
                      [("Q 취소 / C 지우기", (150, 150, 150))], 22, 0))

        cv2.imshow(WIN, self.painter.render(frame, items))

    def recognize_one(self):
        """손을 올렸다 내리는 동작 한 번을 기다려 (라벨, 확률)을 반환한다.
        확신도가 CONF_TH 미만이면 결과를 버리고 계속 기다린다 — 지나가는 손짓을 걸러낸다.

        카메라는 여기서 닫지 않는다. 서비스 호출마다 열고 닫으면 그때마다 창이
        사라졌다 뜨고 여는 데만 1초 넘게 걸린다. 정리는 close() 로 한다."""
        cap = self._camera()
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
                        label, prob, _ = self._classify(seg)
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
            pass                    # 카메라는 close() 에서만 닫는다
