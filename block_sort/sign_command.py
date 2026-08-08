#!/usr/bin/env python3
"""
수어 → 글로스 → LLM → (색, 구역) 명령   (로봇 의존성 없음)

  python3 sign_command.py selftest                     해석·수집 자체 검증 (아무것도 불필요)
  python3 sign_command.py cam [번호]                   수어 카메라가 프레임을 주는지
  python3 sign_command.py parse "빨강 3번구역 놓다"   해석만 확인 (카메라·로봇 불필요)
  python3 sign_command.py listen                      수어만 인식해 글로스 출력
  python3 sign_command.py run                          인식 + 해석까지 (로봇은 안 움직임)

로봇을 실제로 움직이는 것은 `block_sort.py sign` 이다. 여기를 따로 떼어 둔 이유는
해석 로직이 로봇·카메라 없이도 고쳐지고 확인돼야 하기 때문이다 —
LLM 프롬프트는 손볼 일이 잦은데 그때마다 팔을 켤 수는 없다.

끝나는 시점
  끝맺음 동사를 기다리지 않는다. 글로스가 하나 늘 때마다 지금까지 모은 것으로
  명령이 되는지 LLM 에게 물어보고, 말이 되는 순간 그대로 실행한다.
    "빨강 3번구역"          → 두 동작만에 실행
    "3번구역 들다 1번구역"   → 중간의 '들다' 는 아직 명령이 아니라 그냥 지나간다
    "조립하다 빨강 3번구역"  → 오인식 글로스가 섞여도 LLM 이 걸러낸다
  대신 한 문장에 명령을 여러 개 넣을 수는 없다 — 첫 명령이 완성되는 즉시
  실행되기 때문이다. 카메라는 계속 열려 있으니 이어서 다음 명령을 하면 된다.

화면
  sign_demo.py 의 run 화면을 그대로 쓴다. 하단 패널에 모인 글로스가 쌓이고,
  명사+동사로 짝이 맞은 자리는 파랗게 칠해진다. 수집 중에는 중간 결과(~단어)가
  같이 보인다. Q 취소 / C 지우기.
"""
import json
import os
import re
import sys
import threading
import time

# ── 글로스 ↔ 시스템 어휘 ────────────────────────────────────────────
# sign_control/sign_data 에 녹화된 라벨이 왼쪽, object_detection 의
# color_ranges.json 키가 오른쪽이다. 검출 노드는 한글 키와 영문 키를 모두
# 받으므로 한글 그대로 넘긴다 — 로그를 사람이 읽을 수 있는 편이 낫다.
GLOSS_TO_COLOR = {
    "빨강": "빨간색",
    "주황": "주황색",
    "노랑": "노란색",
    "초록": "초록색",
    "파랑": "파란색",
    "보라": "보라색",
}
GLOSS_TO_ZONE = {f"{i}번구역": i for i in range(1, 5)}   # 물리적 제약으로 1~4번구역만 운용

# 이 글로스가 나오면 한 문장이 끝난 것으로 본다. 수어는 동사가 문장 끝에 온다.
SENTENCE_END = {"놓다", "가져오다", "들다", "조립하다"}
MAX_GLOSSES = 6          # 끝맺음 동사가 없어도 여기서 끊는다

# 수어용 웹캠 인덱스는 **박아두지 않는다**. USB 를 다시 꽂으면 번호가 통째로
# 밀린다. 실측 2026-08-05 에 RealSense 가 video1~6 을 가져가면서, 예전 기본값
# 1번은 웹캠이 아니라 **RealSense 깊이 노드**가 됐다. 그걸 V4L2 로 열면
# RealSense ROS 드라이버가 장치를 못 잡아 검출이 통째로 죽는다.
# 이름으로 찾는 방법은 webcam_publisher.find_webcam() 에 있다.
DEFAULT_CAM = 0          # 이름으로 못 찾았을 때만 쓰는 최후 기본값

LLM_MODEL = os.environ.get("SIGN_LLM_MODEL", "gpt-4o")
# 저장소에 키를 두지 않으므로 환경변수나 voice_processing 의 .env 에서 읽는다.
# LLM 키를 둘 자리. 없으면 환경변수(OPENAI_API_KEY)에서 읽는다.
# 저장소 최상위를 먼저 본다 — 키는 음성이든 수어든 이 저장소 공용이고,
# 옛 자리(voice_processing/resource)는 음성 코드가 legacy/ 로 빠지면서
# 뜻이 없어졌다. 옛 자리도 뒤에 남겨 둔다(예전 설치를 깨지 않으려고).
_ENV_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 ".env"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "legacy", "voice_processing", "resource", ".env"),
]

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(HERE)
SIGN_CONTROL_DIR = os.environ.get("SIGN_CONTROL_DIR",
                                  os.path.join(_ROOT, "sign_control"))
SIGN_PROCESSING_DIR = os.environ.get(
    "SIGN_PROCESSING_SRC",
    os.path.join(_ROOT, "sign_pickandplace", "src", "sign_processing"))


# ── 수어 인식 ───────────────────────────────────────────────────────
# ── signbot_admin 으로 화면 중계 ────────────────────────────────────
# sign_demo.py 는 인식만 하고 로봇을 안 움직이고, block_sort.py sign 은 로봇을
# 움직이지만 화면을 안 보냈다. 웹캠은 한 프로세스만 열 수 있어서 "화면을 보려면
# 로봇이 안 움직이고, 로봇을 움직이면 화면이 안 나오는" 상태였다.
# 그래서 이쪽에도 프레임 전송을 붙인다. 방식은 sign_demo.py 와 같다.
ADMIN_URL = os.environ.get("SIGN_ADMIN_URL", "http://localhost:5000")
USE_ADMIN = "--admin" in sys.argv

_frame_lock = threading.Lock()
_frame_holder = {"jpg": None}
_sender_started = False


def set_latest_frame(frame_bgr):
    """최신 한 장만 들고 있는다. 전송은 별도 스레드가 자기 페이스로 가져간다.

    인식 루프가 전송을 기다리면 프레임률이 네트워크에 묶인다.
    """
    import cv2
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if ok:
        with _frame_lock:
            _frame_holder["jpg"] = buf.tobytes()


def _frame_sender_loop():
    import urllib.request
    while True:
        with _frame_lock:
            data = _frame_holder["jpg"]
        if data is not None:
            try:
                req = urllib.request.Request(
                    f"{ADMIN_URL}/api/frame", data=data,
                    headers={"Content-Type": "image/jpeg"}, method="POST")
                urllib.request.urlopen(req, timeout=1.0)
            except Exception:
                pass
        time.sleep(0.08)          # 모니터링용이라 12fps 상한이면 충분하다


def start_frame_sender():
    global _sender_started
    if _sender_started or not USE_ADMIN:
        return None
    _sender_started = True
    threading.Thread(target=_frame_sender_loop, daemon=True).start()
    print(f"  → signbot_admin 화면 전송: {ADMIN_URL}/api/frame")
    return set_latest_frame


def make_recognizer(cam_index=None, show_window=None):
    """sign_processing 의 인식기를 그대로 쓴다.

    ROS2 패키지 안에 있지만 gesture_recognizer 자체는 rclpy 를 안 쓰므로
    경로만 잡아주면 colcon 빌드 없이 import 된다. 인식 로직을 여기에
    복사하면 두 벌이 되어 한쪽만 고치는 사고가 난다.
    """
    model = os.path.join(SIGN_CONTROL_DIR, "sign_data", "model.pt")
    hand = os.path.join(SIGN_CONTROL_DIR, "hand_landmarker.task")
    pose = os.path.join(SIGN_CONTROL_DIR, "pose_landmarker_lite.task")
    # torch/mediapipe 를 올리기 전에 파일부터 본다. 없는 걸 몇 초 기다린 뒤
    # 알려주면 원인을 찾는 데 그만큼 더 걸린다.
    for p in (model, hand, pose):
        if not os.path.exists(p):
            sys.exit(f"{p} 가 없습니다.\n"
                     "  model.pt  : sign_control 에서 학습 (python3 sign_demo.py train)\n"
                     "  *.task    : sign_control/README.md 의 wget 두 줄로 내려받기")

    sys.path.insert(0, SIGN_PROCESSING_DIR)
    from sign_processing.gesture_recognizer import SignRecognizer   # noqa: E402

    if cam_index is None:
        cam_index = resolve_cam()
    if show_window is None:
        show_window = os.environ.get("SIGN_SHOW_WINDOW", "1") != "0"
    # ROS 토픽에서 받을 때는 장치를 열지 않으므로 검사도 안 한다.
    # 대신 발행 노드가 떠 있어야 한다 (ros2 run sign_processing webcam_publisher).
    if os.environ.get("SIGN_SOURCE", "v4l2").lower() != "ros":
        check_cam(cam_index)
    rec = SignRecognizer(model, hand, pose, cam_index, show_window=show_window)
    rec.on_frame = start_frame_sender()      # --admin 없으면 None (아무 일 없음)
    return rec


def resolve_cam():
    """SIGN_CAM 이 있으면 그것, 없으면 장치 **이름**으로 찾는다.

    번호를 박아두면 RealSense 를 가로채는 사고가 난다 (DEFAULT_CAM 주석 참고).
    찾는 로직은 webcam_publisher 한 곳에만 둔다 — 두 벌이 되면 한쪽만 고쳐진다.
    """
    env = os.environ.get("SIGN_CAM")
    if env is not None and env.strip() != "":
        return int(env)
    try:
        sys.path.insert(0, SIGN_PROCESSING_DIR)
        from sign_processing.webcam_publisher import find_webcam
        idx, name = find_webcam()
    except Exception:
        idx, name = None, None
    if idx is None:
        return DEFAULT_CAM
    print(f"  웹캠 자동 선택: /dev/video{idx}  ({name})")
    return idx


def check_cam(index, probe_upto=4):
    """수어 카메라가 실제로 프레임을 주는지 미리 본다.

    카메라는 recognize_one() 안에서 열리므로, 확인을 안 하면 모델을 다 올리고
    사용자가 손을 든 뒤에야 실패한다. 게다가 이 PC 에는 RealSense 가 /dev/video3~8
    을 차지하고 가상 루프백(Iriun)이 0번에 잡혀 있어, 인덱스를 잘못 잡기 쉽다.
    """
    import cv2
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    ok = cap.isOpened()
    if ok:
        for _ in range(5):
            ok, _frame = cap.read()
    cap.release()
    if ok:
        return index

    alive = []
    for i in range(probe_upto + 1):
        c = cv2.VideoCapture(i, cv2.CAP_V4L2)
        good = c.isOpened() and c.read()[0]
        c.release()
        if good:
            alive.append(i)
    sys.exit(f"카메라 {index} 에서 프레임을 못 받았습니다.\n"
             f"  지금 열리는 인덱스: {alive or '없음'}\n"
             f"  SIGN_CAM=<번호> 로 지정하세요. RealSense 는 수어용이 아닙니다 "
             f"(v4l2-ctl --list-devices 로 확인).")


# 인간구역 배치를 로봇구역에 복제하라는 명령. 이 글로스 하나면 문장이 끝난다 —
# 색도 구역도 필요 없기 때문이다. sign_control 에 이미 녹화돼 있던 두 단어다.
COPY_GLOSS = {"똑같이": False, "좌우대칭": True}      # 값은 mirror 여부

# 문장의 여닫이. 첫 박수로 열고 두 번째 박수로 닫는다. 그 밖에서 인식된 것은
# 버린다 — 준비 동작이나 지나가는 손짓이 명령에 섞이지 않게 하려는 것이다.
# sign_demo.py 가 쓰는 규약과 같다.
DELIMITER_GLOSS = os.environ.get("SIGN_DELIMITER", "박수")

# 이 글로스가 문장에 들어 있으면 손동작 조종(제어모드)으로 넘어간다.
# 넘기는 것이 아니라 block_sort 가 자기 안에서 돈다 — block_sort/teleop_mode.py.
# 블록 명령이 아니라 인터페이스 전환이므로 steps 로 해석하지 않고 따로 본다.
# sign_demo.py 와 같은 단어를 쓴다 — 다르면 데모와 로봇이 따로 논다.
MODE_GLOSS = os.environ.get("SIGN_MODE_GLOSS", "모드변경")


def _has_command(glosses):
    """지금까지 모은 것으로 명령 하나를 만들 수 있는가.

    두 가지 형태를 인정한다.
      색 + 구역     "빨강 3번구역"          그 색 블록을 그 구역으로
      구역 + 구역   "3번구역 … 1번구역"     앞 구역에 있는 것을 뒤 구역으로 (색 무관)
    """
    colors = [g for g in glosses if g in GLOSS_TO_COLOR]
    zs = [g for g in glosses if g in GLOSS_TO_ZONE]
    return bool(colors and zs) or len(zs) >= 2


def command_ready(glosses, zones):
    """지금까지 모은 글로스가 말이 되는 명령인가. 되면 (steps, 해석방식), 아니면 None.

    끝맺음 동사를 기다리지 않는다. 동사는 어차피 자주 흘리거나 잘못 잡히고,
    "빨강 3번구역" 만으로도 뜻은 이미 분명하다. 대신 글로스가 하나 늘 때마다
    지금 것으로 명령이 되는지 물어보고, 되는 순간 그대로 실행한다.

    LLM 은 구조적으로 가능할 때만 부른다(_has_command). 아직 색도 구역도 없는
    상태까지 매번 물어보면 동작 하나마다 네트워크를 타서 눈에 띄게 굼떠진다.
    """
    # 복제 명령은 글로스 하나로 끝난다. LLM 을 부를 것도 없다.
    for g in reversed(glosses):
        if g in COPY_GLOSS:
            return [("copy", COPY_GLOSS[g])], "복제"
    if not _has_command(glosses):
        return None
    steps, how = parse(" ".join(glosses), zones)
    return (steps, how) if steps else None


def collect_command(recognizer, zones, on_gloss=None, hint="",
                    max_glosses=MAX_GLOSSES, should_stop=None):
    """박수와 박수 사이를 한 문장으로 모아 해석한다.
    (글로스, steps, 해석방식) 또는 None(취소).

    끝나는 시점은 **닫는 박수**가 정한다. 명령이 되는 순간 바로 실행하던 예전
    방식과 다르다 — 사람이 문장을 다 만들고 나서 실행되는 편이 예측 가능하다.
    닫는 박수를 잊어도 max_glosses 에서 끊긴다.

    카메라 루프는 recognizer 가 통째로 돌린다 — 글로스마다 카메라를 열고 닫으면
    그 사이에 한 동작이 통째로 사라지고, 화면에 모인 글로스를 못 쌓아 보여준다.

    should_stop 은 그대로 recognizer.recognize_sentence() 에 넘긴다 — 모드가
    바뀌었을 때 문장이 끝나길 기다리지 않고 바로 빠져나오는 데 쓴다
    (block_sort.py 의 _sign_loop 참고). None 이면 원래 동작 그대로다.
    """
    gl = recognizer.recognize_sentence(
        lambda g: False,               # 끝은 박수가 정한다
        on_gloss=on_gloss, max_glosses=max_glosses, hint=hint,
        delimiter=DELIMITER_GLOSS, should_stop=should_stop)
    if gl is None:
        return None
    r = command_ready(gl, zones)
    steps, how = r if r else ([], "규칙")
    return gl, steps, how


# ── 해석 ────────────────────────────────────────────────────────────
def describe(src):
    """로그용. 집을 대상이 색이면 색 이름, 구역이면 '3번구역'."""
    if src == "copy":
        return "인간구역 복제"
    return f"{src}번구역" if isinstance(src, int) else str(src)


def _valid(steps, zones):
    """쓸 수 있는 값인지 거른다. LLM 이 지어낸 값도 여기서 걸린다.

    step 은 (집을 것, 놓을 구역). 집을 것은 색 이름(str) 이거나 구역 번호(int) 다.
    """
    out = []
    for src, dst in steps:
        if dst not in zones:
            continue
        if isinstance(src, int):
            # 티칭 안 된 구역에서는 집을 수 없고, 제자리로 옮기라는 건 명령이 아니다.
            if src not in zones or src == dst:
                continue
        elif src not in GLOSS_TO_COLOR.values():
            continue
        out.append((src, dst))
    return out


def rule_parse(text, zones):
    """등장 순서대로 짝짓는다.

    LLM 이 없거나(키 미설정) 호출이 실패해도 시스템이 멈추면 안 된다.
    "빨강 3번구역" 같은 곧은 문장은 이것만으로 충분하고, 네트워크도 안 탄다.

    색이 하나도 없으면 구역끼리 짝짓는다 — "3번구역 들다 1번구역 놓다" 처럼
    색을 말하지 않은 명령은 앞 구역에서 집어 뒤 구역에 놓으라는 뜻이다.
    """
    tokens = text.split()
    colors = [GLOSS_TO_COLOR[t] for t in tokens if t in GLOSS_TO_COLOR]
    zs = [GLOSS_TO_ZONE[t] for t in tokens if t in GLOSS_TO_ZONE]
    if not colors and len(zs) >= 2:
        pairs = list(zip(zs[0::2], zs[1::2]))       # (집을 구역, 놓을 구역)
    else:
        pairs = list(zip(colors, zs))
    return _valid(pairs, zones)


_PROMPT = """당신은 수어 인식 결과를 로봇 명령으로 바꾸는 해석기입니다.

<입력>
수어 글로스가 인식된 순서대로 공백으로 이어진 문자열입니다.
수어에는 조사가 없고 동사가 문장 끝에 옵니다. 어순이 흐트러질 수 있고,
인식 오류로 관계없는 글로스가 섞일 수 있습니다.

<가능한 색>
{colors}

<가능한 구역>
{zones}

<할 일>
무엇을 몇 번 구역에 놓으라는 것인지 뽑아냅니다. 명령은 두 가지 형태입니다.
  (가) 색을 말한 경우 — 그 색 블록을 그 구역으로.       {{"color": "빨간색", "zone": 3}}
  (나) 색을 말하지 않고 구역이 둘인 경우 — 앞 구역에 있는 블록을(색은 상관없음)
       뒤 구역으로.                                    {{"from_zone": 3, "zone": 1}}
- 색과 구역을 등장 순서대로 짝지으세요.
- 색이 하나도 없고 구역이 둘 이상이면 (나) 로 보고 앞뒤로 짝지으세요.
- 동사(놓다/가져오다/들다/조립하다)와 수식어(똑같이/좌우대칭), 그 밖의
  글로스(나사/망치 등)는 무시하세요. 이 로봇은 블록을 구역으로 옮기기만 합니다.
- 목록에 없는 구역 번호가 나오면 그 짝은 버리세요.
- 집을 구역과 놓을 구역이 같으면 제자리라 명령이 아닙니다. 버리세요.
- 짝을 지을 수 없으면 빈 배열을 반환하세요. 추측해서 지어내지 마세요.

<출력 형식>
JSON 배열만 출력합니다. 설명·코드펜스 없이 배열 자체만.
색을 말했으면 color, 구역에서 집는 것이면 from_zone 을 씁니다.

<예시>
입력: "빨강 3번구역 놓다"        출력: [{{"color": "빨간색", "zone": 3}}]
입력: "파랑 1번구역 초록 2번구역 놓다"
출력: [{{"color": "파란색", "zone": 1}}, {{"color": "초록색", "zone": 2}}]
입력: "망치 가져오다"            출력: []
입력: "3번구역 빨강 놓다"        출력: [{{"color": "빨간색", "zone": 3}}]
입력: "3번구역 들다 1번구역 놓다" 출력: [{{"from_zone": 3, "zone": 1}}]
입력: "2번구역 들다 4번구역 놓다 1번구역 들다 3번구역 놓다"
출력: [{{"from_zone": 2, "zone": 4}}, {{"from_zone": 1, "zone": 3}}]

<입력>
{text}
"""


def _api_key():
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or "여기에" in key or key.startswith("sk-proj-여기"):
        # .env 는 저장소에 없다(.gitignore). 있으면 읽고, 없으면 규칙 해석으로 간다.
        for path in _ENV_CANDIDATES:
            if os.path.exists(path):
                try:
                    from dotenv import dotenv_values
                    key = dotenv_values(path).get("OPENAI_API_KEY", "") or key
                except ImportError:
                    pass
    if not key or "여기에" in key:
        return None
    return key


def llm_parse(text, zones):
    """LLM 에게 글로스 나열을 (색, 구역) 목록으로 바꾸게 한다. 실패하면 None."""
    key = _api_key()
    if key is None:
        return None
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0, openai_api_key=key)
        prompt = _PROMPT.format(
            colors=", ".join(sorted(set(GLOSS_TO_COLOR.values()))),
            zones=", ".join(f"{z}번구역" for z in zones),
            text=text)
        raw = llm.invoke(prompt).content.strip()
        # 코드펜스를 붙여 오는 경우가 있어 배열만 도려낸다.
        m = re.search(r"\[.*\]", raw, re.S)
        if not m:
            return None
        data = json.loads(m.group(0))
        steps = []
        for d in data:
            if not isinstance(d, dict) or d.get("zone") is None:
                continue
            if d.get("color"):
                steps.append((d["color"], int(d["zone"])))
            elif d.get("from_zone") is not None:
                steps.append((int(d["from_zone"]), int(d["zone"])))
        return _valid(steps, zones)
    except Exception as e:
        print(f"  LLM 해석 실패({e.__class__.__name__}: {e}) — 규칙 해석으로 대체")
        return None


def parse(text, zones=(1, 2, 3, 4)):
    """글로스 문자열 → [(색, 구역), ...] 과 해석 방식.

    LLM 을 먼저 쓰고, 못 쓰거나 빈손이면 규칙 해석으로 내려간다.
    현장에서 키가 없거나 네트워크가 끊겨도 시연이 멈추지 않게 하기 위해서다.
    """
    zones = list(zones)
    steps = llm_parse(text, zones)
    if steps:
        return steps, "LLM"
    return rule_parse(text, zones), "규칙"


# ── 자체 검증 ───────────────────────────────────────────────────────
# 로봇도 카메라도 없이 돌아가는 것들만 넣는다. 실주행 전에 여기가 통과해야
# 팔을 켤 가치가 있다. (글로스 나열, 기대 결과) — 구역은 1~4 기준.
_CASES = [
    ("빨강 3번구역 놓다",              [("빨간색", 3)]),
    ("파랑 1번구역 초록 2번구역 놓다",   [("파란색", 1), ("초록색", 2)]),
    ("3번구역 빨강 놓다",              [("빨간색", 3)]),      # 어순이 뒤바뀌어도
    ("나사 빨강 좌우대칭 4번구역 놓다",  [("빨간색", 4)]),      # 관계없는 글로스는 무시
    ("망치 가져오다",                  []),                  # 색·구역이 없으면 빈손
    ("보라 5번구역 놓다",              []),                  # 티칭 안 된 구역은 거부
    ("노랑 놓다",                      []),                  # 구역이 없으면 짝이 안 된다
    # 색 없이 구역→구역
    ("3번구역 들다 1번구역 놓다",       [(3, 1)]),
    ("2번구역 들다 4번구역 놓다 1번구역 들다 3번구역 놓다", [(2, 4), (1, 3)]),
    ("3번구역 들다 3번구역 놓다",       []),                  # 제자리는 명령이 아니다
    ("3번구역 들다 5번구역 놓다",       []),                  # 티칭 안 된 구역
    ("2번구역 들다",                   []),                  # 놓을 곳이 없다
]

# 복제 명령. parse() 가 아니라 command_ready() 가 처리하므로 따로 검증한다.
_COPY_CASES = [
    (["똑같이"],                 [("copy", False)]),
    (["좌우대칭"],               [("copy", True)]),
    (["조립하다", "똑같이"],      [("copy", False)]),   # 오인식이 앞에 껴도 잡힌다
    (["빨강", "3번구역"],        [("빨간색", 3)]),      # 복제가 아니면 평소대로
    (["나사"],                   None),                # 아직 아무 명령도 아님
]


def _simulate(seq, zones=(1, 2, 3, 4), max_glosses=MAX_GLOSSES):
    """글로스가 이 순서로 들어왔을 때 무엇이 문장으로 남는지. 카메라 없이 확인한다.

    recognize_sentence() 의 여닫이 규칙을 그대로 흉내낸다:
    첫 박수 전은 버리고, 두 번째 박수에서 끊는다.
    """
    glosses, armed = [], False
    for g in seq:
        if g == DELIMITER_GLOSS:
            if armed:
                break                      # 닫는 박수
            armed, glosses = True, []      # 여는 박수 — 앞의 것은 버린다
            continue
        if not armed:
            continue                       # 아직 안 열렸다
        glosses.append(g)
        if len(glosses) >= max_glosses:
            break
    return glosses


def selftest():
    """해석기와 수집 규칙을 로봇·카메라 없이 돌려본다. 실패한 개수를 반환한다."""
    fails = 0
    print(f"\n  해석 ({'LLM' if _api_key() else '규칙 — OPENAI_API_KEY 없음'})")
    for text, want in _CASES:
        got, how = parse(text, [1, 2, 3, 4])
        ok = got == want
        fails += not ok
        print(f"    {'OK ' if ok else '실패'} [{how:2s}] {text:28s} → {got}"
              + ("" if ok else f"   기대: {want}"))

    D = DELIMITER_GLOSS
    print(f"\n  글로스 수집 ({D} 로 열고 {D} 로 닫는다)")
    for name, seq, want in [
        ("여닫이 사이만", [D, "빨강", "3번구역", D, "파랑"], ["빨강", "3번구역"]),
        ("열기 전 것은 버림", ["조립하다", "나사", D, "빨강", "3번구역", D],
         ["빨강", "3번구역"]),
        ("닫은 뒤 것은 안 모음", [D, "빨강", "3번구역", D, "놓다", "파랑"],
         ["빨강", "3번구역"]),
        ("동사가 섞여도 그대로", [D, "3번구역", "들다", "1번구역", "놓다", D],
         ["3번구역", "들다", "1번구역", "놓다"]),
        ("복제 명령", [D, "똑같이", D], ["똑같이"]),
        ("안 열면 아무것도 안 모음", ["빨강", "3번구역", "놓다"], []),
        ("닫기를 잊으면 개수까지", [D] + ["놓다"] * 8, ["놓다"] * MAX_GLOSSES),
    ]:
        got = _simulate(seq)
        ok = got == want
        fails += not ok
        print(f"    {'OK ' if ok else '실패'} {name:16s} → {got}")
        if not ok:
            print(f"         기대: {want}")

    print("\n  복제 명령 (똑같이 / 좌우대칭)")
    for gl, want in _COPY_CASES:
        r = command_ready(gl, [1, 2, 3, 4])
        got = r[0] if r else None
        ok = got == want
        fails += not ok
        print(f"    {'OK ' if ok else '실패'} {str(gl):28s} → {got}"
              + ("" if ok else f"   기대: {want}"))

    print("\n  동사 가중치 표")
    sys.path.insert(0, SIGN_PROCESSING_DIR)
    from sign_processing.gesture_recognizer import kind_of, paired_idx, VERB_BONUS
    for w, want in [("빨강", "수식"), ("3번구역", "명사"), ("망치", "명사"), ("놓다", "동사")]:
        ok = kind_of(w) == want
        fails += not ok
        print(f"    {'OK ' if ok else '실패'} {w:8s} → {kind_of(w)}")
    gl = ["빨강", "3번구역", "놓다"]
    pair = paired_idx(gl)
    ok = pair == {1, 2}                # 3번구역(명사) + 놓다(동사) 가 한 덩어리
    fails += not ok
    print(f"    {'OK ' if ok else '실패'} 짝 {gl} → {sorted(pair)} (가중 +{VERB_BONUS*100:.0f}%)")

    print(f"\n  {'모두 통과' if fails == 0 else f'{fails}건 실패'}\n")
    return fails


# ── 단독 실행 (로봇 없이 확인용) ────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    m = sys.argv[1]
    if m == "selftest":
        sys.exit(1 if selftest() else 0)
    elif m == "cam":
        idx = int(sys.argv[2]) if len(sys.argv) > 2 else resolve_cam()
        print(f"\n  카메라 {check_cam(idx)} 정상 — 프레임 수신 확인\n")
    elif m == "parse":
        if len(sys.argv) < 3:
            sys.exit('사용법: parse "빨강 3번구역 놓다"')
        text = " ".join(sys.argv[2:])
        steps, how = parse(text)
        print(f"\n  '{text}'  →  [{how}]  {steps or '해석 실패'}\n")
    elif m in ("listen", "run"):
        rec = make_recognizer()
        print("\n  수어로 명령하세요 — 예: 빨강 → 3번구역 → 놓다  (창에서 Q 종료)\n")
        try:
            _loop(rec, m)
        finally:
            rec.close()
    else:
        sys.exit(__doc__)


def _loop(rec, m):
    while True:
        got = collect_command(
            rec, [1, 2, 3, 4],
            on_gloss=lambda l, p, b: print(
                f"    · {l} ({p * 100:.0f}%){'  [동사 가중]' if b else ''}"),
            hint="빨강 → 3번구역")
        if got is None:
            print("  취소됨\n")
            return
        gl, steps, how = got
        text = " ".join(gl)
        if m == "listen":
            print(f"  글로스: {text}\n")
            continue
        print(f"  '{text}'  →  [{how}]  {steps or '해석 실패'}\n")


if __name__ == "__main__":
    main()
