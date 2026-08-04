#!/usr/bin/env python3
"""
수어 인식 테스트 데모  —  녹화 / 학습 / 실시간 인식

  python3 sign_demo.py record 청소     동작 샘플 녹화
  python3 sign_demo.py list            수집 현황
  python3 sign_demo.py train           학습
  python3 sign_demo.py run             실시간 인식 → 텍스트
  python3 sign_demo.py run --llm       + GPT로 자연스러운 문장 변환

수어를 몰라도 됩니다. 모델은 '옳은 수어'가 아니라
'당신이 일관되게 하는 동작'을 학습합니다. 매번 똑같이만 하세요.
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from collections import deque

import cv2
import numpy as np

# ───────────────────────── 설정 ─────────────────────────
CAM_INDEX = int(os.environ.get("SIGN_CAM", 0))       # LG Camera (video1은 메타데이터 전용이라 열리지 않음)
CAP_W, CAP_H = 1280, 720

# 경로는 스크립트 위치를 기준으로 잡는다. 절대경로를 박아두면
# 저장소를 clone 한 사람이 손대야 실행된다.
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("SIGN_MODELS", HERE)     # .task 파일을 둔 곳
HAND_TASK = os.path.join(MODEL_DIR, "hand_landmarker.task")
POSE_TASK = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
DATA_DIR = os.environ.get("SIGN_DATA", os.path.join(HERE, "sign_data"))
MODEL_PT = os.path.join(DATA_DIR, "model.pt")

REC_FRAMES = 36                     # 녹화 길이. C270 실측 24fps 기준 약 1.5초
SEQ_LEN = 32                        # 모델 입력 길이 (리샘플 후)
COUNTDOWN = 3                       # 녹화 전 카운트다운(초)
GAP_SEC = 1.2                       # 배치 녹화에서 샘플 사이 쉼(초)

CONF_TH = 0.80                      # 이 확률 넘어야 인정
VERB_BONUS = 0.05                   # 명사 다음에는 동사가 올 확률이 높다는 사전지식
# 인식 단위는 '슬라이딩 창'이 아니라 '손이 올라와 있는 한 구간'이다.
# 학습 데이터가 한 클립 = 한 동작이므로 인식도 같은 단위로 끊어야 맞는다.
PRE_PAD = 6                         # 손이 보이기 직전 프레임도 포함 (동작 시작부)
END_HOLD = 5                        # 손이 이만큼 연속으로 안 보이면 구간 종료
MIN_HAND = 10                       # 손이 이보다 적게 보이면 동작으로 안 침
MAX_SEG = 96                        # 손을 계속 들고 있어도 여기서 강제 종료

# 포즈 33점 중 상반신만 쓴다. 하반신은 수어와 무관하고 노이즈만 늘린다.
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]   # 코, 양 어깨, 양 팔꿈치, 양 손목

# 한 프레임 특징 = 상반신 7점×2 + (손목위치2 + 손모양21×2 + 유무1)×2손
FEAT_DIM = len(POSE_IDX) * 2 + 2 * (2 + 21 * 2 + 1)     # = 104

# 품사. 명사 뒤에 동사가 오면 '말이 되는 짝'으로 보고 파랗게 칠한다.
# 여기 없는 단어는 명사로 취급한다 — 나중에 단어를 늘려도 그대로 동작한다.
WORD_KIND = {
    "청소하다": "동사", "가져오다": "동사", "정리하다": "동사",
    "파란색": "수식", "빨간색": "수식", "노란색": "수식",
}
BLUE = (255, 170, 60)               # BGR — 짝지어진 명사·동사

WIN = "sign - run"
PANEL_H = 185                       # 하단 패널 높이 (글로스 + 통역문 + 버튼)
BUTTONS = {                         # 이름: (x, y, w, h)
    "지우기": (30, CAP_H - 66, 150, 50),
    "통역": (200, CAP_H - 66, 150, 50),
}

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]


# ───────────────────────── 한글 렌더링 ─────────────────────────
class Painter:
    """cv2.putText는 한글을 못 그린다. PIL로 우회한다."""

    def __init__(self):
        from PIL import ImageFont
        path = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
        if path is None:
            raise RuntimeError("한글 폰트를 찾을 수 없습니다.")
        self._cache = {}
        self._path = path
        self._ImageFont = ImageFont

    def font(self, size):
        if size not in self._cache:
            self._cache[size] = self._ImageFont.truetype(self._path, size)
        return self._cache[size]

    def width(self, s, size):
        return self.font(size).getlength(s)

    def render(self, img, items):
        """[(xy, [(글자, 색), ...], 크기, 자간), ...] 를 PIL 왕복 '한 번'으로 전부 그린다.

        1280x720 에서 왕복 1회가 5ms다. 항목마다 따로 부르면 7회 = 36ms 로
        MediaPipe(25ms)보다 비싸져 프레임률이 반토막 난다."""
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for xy, parts, size, gap in items:
            f = self.font(size)
            x, y = xy
            for s, color in parts:
                d.text((x + 2, y + 2), s, font=f, fill=(0, 0, 0))
                d.text((x, y), s, font=f, fill=color[::-1])
                x += f.getlength(s) + gap
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def text(self, img, xy, s, size=28, color=(255, 255, 255)):
        return self.render(img, [(xy, [(s, color)], size, 0)])


# ───────────────────────── 특징 추출 ─────────────────────────
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
POSE_EDGES = [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6)]   # POSE_IDX 안에서의 인덱스


class Extractor:
    """MediaPipe 포즈+손 → 몸 기준으로 정규화된 특징 벡터."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        self.mp = mp

        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=POSE_TASK),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5))
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=HAND_TASK),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5))
        self._t = 0

    def __call__(self, bgr):
        """returns (feat[104] or None, draw_info)"""
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

        # 어깨 중점을 원점, 어깨너비를 1로. 사람이 움직이거나 멀어져도 같은 값이 나온다.
        origin = (P[1] + P[2]) / 2.0
        scale = np.linalg.norm(P[1] - P[2])
        if scale < 1e-3:
            return None, {"pose": None, "hands": []}
        pose_f = ((P - origin) / scale).ravel()

        # 손은 '어디에 있는가'(위치)와 '어떤 모양인가'(손모양)를 분리해서 넣는다.
        # 수어의 음운 구조가 실제로 그렇게 나뉘어 있다.
        slots = {"Left": np.zeros(45, np.float32), "Right": np.zeros(45, np.float32)}
        draw_hands = []
        for pts, hd in zip(hres.hand_landmarks, hres.handedness):
            name = hd[0].category_name
            if name not in slots:
                continue
            H = np.array([[p.x * w, p.y * h] for p in pts], np.float32)
            draw_hands.append(H)
            span = np.linalg.norm(H[0] - H[9]) + 1e-6      # 손목→중지 밑동
            slots[name] = np.concatenate([
                (H[0] - origin) / scale,                   # 손목의 몸 기준 위치
                ((H - H[0]) / span).ravel(),               # 손 자체 모양
                [1.0],                                     # 검출됨
            ]).astype(np.float32)

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


def kind_of(word):
    return WORD_KIND.get(word, "명사")


def paired_idx(gl):
    """명사 바로 뒤에 동사가 오는 자리를 찾는다. 그 두 개가 한 덩어리 명령이다."""
    out = set()
    for i in range(len(gl) - 1):
        if kind_of(gl[i]) == "명사" and kind_of(gl[i + 1]) == "동사":
            out.update((i, i + 1))
    return out


def resample(seq, n=SEQ_LEN):
    """길이가 제각각인 시퀀스를 고정 길이로. 동작 속도 차이를 흡수한다."""
    seq = np.asarray(seq, np.float32)
    if len(seq) == n:
        return seq
    src = np.linspace(0, len(seq) - 1, len(seq))
    dst = np.linspace(0, len(seq) - 1, n)
    return np.stack([np.interp(dst, src, seq[:, i]) for i in range(seq.shape[1])], 1).astype(np.float32)


# ───────────────────────── 카메라 ─────────────────────────
def open_cam():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    # C270은 기본이 YUYV라 720p에서 6fps밖에 안 나온다. MJPG로 강제해야 한다.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit(f"카메라 {CAM_INDEX} 를 열 수 없습니다.")
    for _ in range(5):
        cap.read()
    # C270은 어두우면 노출을 늘리려고 스스로 프레임률을 절반(15fps)으로 떨군다.
    # 이 설정만 끄면 밝기는 그대로인데 30fps가 나온다. OpenCV로는 못 건드려서
    # v4l2-ctl 을 직접 부른다.
    subprocess.run(["v4l2-ctl", "-d", f"/dev/video{CAM_INDEX}",
                    "-c", "exposure_dynamic_framerate=0"], capture_output=True)
    return cap


def labels_on_disk():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d)))


# ───────────────────────── 모드: 녹화 ─────────────────────────
def mode_record(label, batch=1):
    """batch>1 이면 카운트다운 한 번 뒤 연속으로 여러 개를 찍는다.
    샘플마다 3초씩 세면 30단어 녹화에 2시간이 넘어간다."""
    out_dir = os.path.join(DATA_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    n_have = len(glob.glob(os.path.join(out_dir, "*.npy")))

    ext, paint, cap = Extractor(), Painter(), open_cam()
    print(f"\n[녹화] '{label}'  현재 {n_have}개  배치 {batch}회")
    print("  SPACE 시작 / Q 종료 / U 마지막 삭제 / ESC 배치 중단\n")

    state, buf, t_start, left_in_batch = "idle", [], 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)              # 거울 모드가 따라 하기 쉽다
        feat, info = ext(frame)
        draw_skeleton(frame, info)

        if state == "count":
            left = COUNTDOWN - (time.time() - t_start)
            if left <= 0:
                state, buf = "rec", []
            else:
                frame = paint.text(frame, (CAP_W // 2 - 40, CAP_H // 2 - 60),
                                   str(int(left) + 1), 120, (0, 220, 255))

        elif state == "gap":                    # 배치 중 짧은 쉼 — 손을 내렸다 올릴 시간
            if time.time() - t_start >= GAP_SEC:
                state, buf = "rec", []
            else:
                frame = paint.text(frame, (CAP_W // 2 - 90, CAP_H // 2 - 40),
                                   "준비", 70, (0, 220, 255))

        elif state == "rec":
            if feat is not None:
                buf.append(feat)
            cv2.rectangle(frame, (0, 0), (CAP_W - 1, CAP_H - 1), (0, 0, 255), 8)
            frame = paint.text(frame, (30, 90), f"● 녹화 {len(buf)}/{REC_FRAMES}",
                               34, (80, 80, 255))
            if len(buf) >= REC_FRAMES:
                np.save(os.path.join(out_dir, f"{int(time.time()*1000)}.npy"),
                        np.asarray(buf, np.float32))
                n_have += 1
                left_in_batch -= 1
                print(f"  저장 → {n_have}개" +
                      (f"   (배치 {left_in_batch}회 남음)" if left_in_batch > 0 else ""))
                state, t_start = ("gap", time.time()) if left_in_batch > 0 else ("idle", 0.0)

        bar = "검출됨" if feat is not None else "몸이 안 보입니다"
        col = (120, 255, 120) if feat is not None else (100, 100, 255)
        frame = paint.text(frame, (30, 30), f"[{label}]  {n_have}개   {bar}", 30, col)
        if left_in_batch > 0:
            frame = paint.text(frame, (CAP_W - 220, 34), f"남은 {left_in_batch}", 30,
                               (0, 220, 255))
        if state == "idle":
            frame = paint.text(frame, (30, CAP_H - 50),
                               f"SPACE 녹화(×{batch})   U 마지막삭제   Q 종료",
                               24, (200, 200, 200))

        cv2.imshow("sign - record", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == 27:                              # ESC — 배치 중단
            state, left_in_batch = "idle", 0
        if k == ord(" ") and state == "idle":
            state, t_start, left_in_batch = "count", time.time(), batch
        if k == ord("u") and state == "idle":
            fs = sorted(glob.glob(os.path.join(out_dir, "*.npy")))
            if fs:
                os.remove(fs[-1])
                n_have -= 1
                print(f"  삭제 → {n_have}개")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n'{label}' 총 {n_have}개\n")


# ───────────────────────── 모드: 현황 ─────────────────────────
def mode_list():
    labs = labels_on_disk()
    if not labs:
        print("\n수집된 데이터가 없습니다.  python3 sign_demo.py record <이름>\n")
        return
    print(f"\n{'라벨':<16}{'샘플수':>8}")
    print("-" * 26)
    total = 0
    for l in labs:
        n = len(glob.glob(os.path.join(DATA_DIR, l, "*.npy")))
        total += n
        flag = "" if n >= 20 else "   ← 부족(20개 이상 권장)"
        print(f"{l:<16}{n:>8}{flag}")
    print("-" * 26)
    print(f"{'합계':<16}{total:>8}\n")


# ───────────────────────── 모델 ─────────────────────────
def build_model(n_cls):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(FEAT_DIM, 128, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=0.2)
            self.head = nn.Sequential(
                nn.Linear(256 * 2, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, n_cls))

        def forward(self, x):
            y, _ = self.gru(x)
            # 평균과 최대를 같이 쓴다. 동작 전체 흐름과 결정적 순간을 둘 다 본다.
            return self.head(torch.cat([y.mean(1), y.max(1).values], -1))

    return Net()


def augment(x, rng):
    """데이터가 적으니 증강이 필수다. 사람이 매번 똑같이는 못 하니까."""
    x = x.copy()
    x *= rng.uniform(0.92, 1.08)                             # 몸 크기 차이
    x += rng.normal(0, 0.012, x.shape).astype(np.float32)    # 랜드마크 흔들림
    if rng.random() < 0.5:                                   # 시간 늘이기/줄이기
        k = rng.uniform(0.8, 1.25)
        m = max(8, int(len(x) * k))
        x = resample(resample(x, m), SEQ_LEN)
    return x


def mode_train(epochs=140):
    import torch
    import torch.nn as nn

    labs = labels_on_disk()
    if len(labs) < 2:
        sys.exit("라벨이 2개 이상 필요합니다.")

    X, Y = [], []
    for ci, l in enumerate(labs):
        for f in glob.glob(os.path.join(DATA_DIR, l, "*.npy")):
            X.append(resample(np.load(f)))
            Y.append(ci)
    X, Y = np.stack(X), np.array(Y)
    print(f"\n라벨 {len(labs)}개 / 샘플 {len(X)}개 → {labs}")

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_val = max(len(X) // 5, len(labs))
    val, tr = idx[:n_val], idx[n_val:]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_model(len(labs)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    Xv = torch.tensor(X[val]).to(dev)
    Yv = torch.tensor(Y[val]).to(dev)
    best = 0.0
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(tr))
        for s in range(0, len(order), 32):
            b = tr[order[s:s + 32]]
            xb = torch.tensor(np.stack([augment(X[i], rng) for i in b])).to(dev)
            yb = torch.tensor(Y[b]).to(dev)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
        sch.step()

        net.eval()
        with torch.no_grad():
            acc = (net(Xv).argmax(1) == Yv).float().mean().item()
        if acc >= best:
            best = acc
            torch.save({"labels": labs, "state": net.state_dict()}, MODEL_PT)
        if (ep + 1) % 20 == 0:
            print(f"  ep {ep+1:>3}  loss {loss.item():.3f}  val {acc*100:.1f}%")

    print(f"\n최고 검증 정확도 {best*100:.1f}%  →  {MODEL_PT}")

    # 어떤 단어끼리 헷갈리는지 알려준다. 이걸 모르면 30개 중 뭘 고쳐야 할지 알 수 없다.
    net.load_state_dict(torch.load(MODEL_PT, map_location=dev,
                                   weights_only=False)["state"])
    net.eval()
    with torch.no_grad():
        pred = net(Xv).argmax(1).cpu().numpy()
    true = Y[val]
    conf = {}
    for t, p in zip(true, pred):
        if t != p:
            conf[(labs[t], labs[p])] = conf.get((labs[t], labs[p]), 0) + 1
    if conf:
        print("\n혼동된 쌍 (실제 → 오인):")
        for (t, p), c in sorted(conf.items(), key=lambda kv: -kv[1]):
            print(f"  {t} → {p}   {c}회")
        print("  이 단어들은 동작을 더 다르게 바꾸거나 샘플을 늘리세요.")

    per = {}
    for t, p in zip(true, pred):
        a, b = per.get(labs[t], (0, 0))
        per[labs[t]] = (a + int(t == p), b + 1)
    weak = [(l, c / n) for l, (c, n) in per.items() if n and c / n < 0.9]
    if weak:
        print("\n약한 라벨:")
        for l, r in sorted(weak, key=lambda x: x[1]):
            print(f"  {l:<14}{r*100:>5.0f}%")

    if best < 0.9:
        print("\n  90% 미만입니다. 샘플을 늘리거나, 서로 더 구분되는 동작을 고르세요.")
    print()


# ───────────────────────── 모드: 실시간 인식 ─────────────────────────
def make_llm():
    if not os.environ.get("OPENAI_API_KEY"):
        print("  OPENAI_API_KEY 가 없어 LLM 없이 실행합니다.")
        return None
    from openai import OpenAI
    cli = OpenAI()

    def run(glosses):
        try:
            r = cli.chat.completions.create(
                model="gpt-4o-mini", temperature=0, max_tokens=100,
                messages=[
                    {"role": "system", "content":
                     "한국수어 글로스(단어 나열)를 자연스러운 한국어 한 문장으로 바꾼다. "
                     "수어는 조사가 없고 어순이 한국어와 다르다. 설명 없이 문장만 출력."},
                    {"role": "user", "content": " ".join(glosses)}])
            return r.choices[0].message.content.strip()
        except Exception as e:
            return f"(LLM 오류: {e})"
    return run


def mode_run(use_llm=False):
    import torch

    if not os.path.exists(MODEL_PT):
        sys.exit("학습된 모델이 없습니다.  python3 sign_demo.py train")
    ck = torch.load(MODEL_PT, map_location="cpu", weights_only=False)
    labs = ck["labels"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_model(len(labs)).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()

    llm = make_llm() if use_llm else None
    ext, paint, cap = Extractor(), Painter(), open_cam()
    print(f"\n[인식] 라벨 {labs}")
    print("  Q 종료 / C 문장 지우기" + ("  / ENTER LLM 통역" if llm else ""))
    print("  하단 [지우기] [통역] 버튼 클릭도 됩니다\n")

    ui = {}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, (bx, by, bw, bh) in BUTTONS.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                ui["click"], ui["hit"], ui["hit_t"] = name, name, time.time()

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    verb_mask = np.array([VERB_BONUS if kind_of(l) == "동사" else 0.0
                          for l in labs], np.float32)

    def classify(seg):
        x = torch.tensor(resample(seg)[None]).to(dev)
        with torch.no_grad():
            p = torch.softmax(net(x)[0], -1).cpu().numpy()
        # 직전이 명사면 동사에 가산점을 준다.
        # 곱셈+재정규화로 하면 이미 확률이 몰린 1등은 거의 안 움직여서(0.89→0.893)
        # 정작 필요한 경계 구간에서 효과가 없다. 그래서 단순 덧셈으로 한다.
        b = bool(glosses) and kind_of(glosses[-1]) == "명사"
        if b:
            p = np.minimum(p + verb_mask, 1.0)
        i = int(p.argmax())
        return labs[i], float(p[i]), b

    pre = deque(maxlen=PRE_PAD)
    seg, gone, n_hand = [], 0, 0
    glosses, sentence = [], ""
    cur, prob, boosted = "", 0.0, False
    peek, tick = "", 0
    fps, t_prev = 0.0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        feat, info = ext(frame)
        draw_skeleton(frame, info)
        has_hand = bool(info["hands"])

        tick += 1
        if feat is not None:
            pre.append(feat)
            if not seg:                          # 손이 올라오면 구간 시작
                if has_hand:
                    seg, gone, n_hand, peek = list(pre), 0, 1, ""
            else:
                seg.append(feat)
                if has_hand:
                    gone, n_hand = 0, n_hand + 1
                else:
                    gone += 1                    # 손이 내려간 뒤에도 잠깐 더 담는다

                # 분류가 0.5ms뿐이라, 수집 중에도 슬쩍 돌려 중간 결과를 보여준다.
                # 확정에는 쓰지 않는다 — 화면이 멈춘 듯 보이지 않게 하는 용도다.
                if n_hand >= MIN_HAND and tick % 3 == 0:
                    peek = classify(seg)[0]

                if gone >= END_HOLD or len(seg) >= MAX_SEG:
                    if n_hand >= MIN_HAND:
                        cur, prob, boosted = classify(seg)
                        passed = prob >= CONF_TH
                        print(f"  {'→' if passed else '×'} {cur}  {prob*100:.0f}%"
                              f"   [{n_hand}프레임{', 가중' if boosted else ''}]")
                        if passed:
                            glosses.append(cur)
                    else:
                        print(f"  · 너무 짧음 ({n_hand}프레임) — 무시")
                    seg, gone, n_hand = [], 0, 0

        # ── 화면: cv2 도형을 먼저 다 그리고, 글자는 마지막에 한 번에 ──
        cv2.rectangle(frame, (0, CAP_H - PANEL_H), (CAP_W, CAP_H), (25, 25, 25), -1)
        items = []

        if cur:
            c = (120, 255, 120) if prob >= CONF_TH else (140, 140, 140)
            items.append(((30, 30), [(f"{cur}  {prob*100:.0f}%", c)], 40, 0))
            if boosted:
                items.append(((32, 84), [("동사 가중 +5%", BLUE)], 22, 0))
        if seg:
            tail = f"  ~{peek}" if peek else ""
            items.append(((CAP_W - 360, 34),
                          [(f"● 수집중 {n_hand}{tail}", (80, 80, 255))], 24, 0))
        else:
            items.append(((CAP_W - 360, 34),
                          [("대기 — 손을 올리세요", (120, 255, 120))], 24, 0))
        items.append(((CAP_W - 130, CAP_H - PANEL_H - 34),
                      [(f"{fps:.0f} fps", (150, 150, 150))], 22, 0))

        pair = paired_idx(glosses)
        items.append(((30, CAP_H - PANEL_H + 12),
                      [(g, BLUE if i in pair else (255, 255, 255))
                       for i, g in enumerate(glosses)] or [("…", (110, 110, 110))],
                      34, 14))
        if sentence:
            items.append(((30, CAP_H - PANEL_H + 62),
                          [(sentence, (120, 255, 255))], 30, 0))

        for name, (bx, by, bw, bh) in BUTTONS.items():
            on = ui.get("hit") == name and time.time() - ui.get("hit_t", 0) < 0.15
            enabled = True if name == "지우기" else bool(glosses) and bool(llm)
            fill = (70, 120, 70) if on else ((55, 55, 55) if enabled else (38, 38, 38))
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), fill, -1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (110, 110, 110), 1)
            tw = paint.width(name, 26)
            items.append(((bx + (bw - tw) / 2, by + 11), [(name, (255, 255, 255)
                          if enabled else (120, 120, 120))], 26, 0))

        frame = paint.render(frame, items)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        cv2.imshow(WIN, frame)
        k = cv2.waitKey(1) & 0xFF
        click = ui.pop("click", None)
        if k == ord("q"):
            break
        if k == ord("c") or click == "지우기":
            glosses, sentence = [], ""
        if (k == 13 or click == "통역") and llm and glosses:
            sentence = llm(glosses)
            print(f"  [통역] {sentence}")

    cap.release()
    cv2.destroyAllWindows()


# ───────────────────────── 진입점 ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description="수어 인식 테스트 데모")
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("record", help="샘플 녹화")
    r.add_argument("label")
    r.add_argument("-b", "--batch", type=int, default=1,
                   help="SPACE 한 번에 연속 N개 녹화 (예: -b 20)")
    sub.add_parser("list", help="수집 현황")
    t = sub.add_parser("train", help="학습")
    t.add_argument("--epochs", type=int, default=140)
    u = sub.add_parser("run", help="실시간 인식")
    u.add_argument("--llm", action="store_true", help="GPT로 문장 변환")

    a = ap.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    if a.mode == "record":
        mode_record(a.label, max(1, a.batch))
    elif a.mode == "list":
        mode_list()
    elif a.mode == "train":
        mode_train(a.epochs)
    elif a.mode == "run":
        mode_run(a.llm)


if __name__ == "__main__":
    main()
