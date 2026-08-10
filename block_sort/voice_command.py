#!/usr/bin/env python3
"""
음성 → 문장(STT) → 글로스 정규화 → (색, 구역) 명령   (로봇 의존성 없음)

  python3 voice_command.py selftest                    정규화·해석 자체 검증 (마이크 불필요)
  python3 voice_command.py parse "빨간색 3번 구역에 놔줘"   정규화+해석만 확인 (마이크 불필요)
  python3 voice_command.py listen                       웨이크워드 → 녹음 → STT 텍스트만 출력
  python3 voice_command.py watch                        인식만 계속 보기 (로봇 안 움직임, Ctrl+C 종료)
  python3 voice_command.py watch-gui                     block_sort.py 와 같은 화면(카메라 창)으로 보기 (로봇 안 움직임, Q 종료)

로봇을 움직이는 것은 `block_sort.py`의 `run_voice()`뿐이다. `watch`/`watch-gui`는
그 앞단(인식·정규화·해석)만 반복해서 보여준다 — sign_demo.py의 `run`이 수어
인식만 보여주고 로봇을 안 움직이는 것과 같은 역할이다. `watch-gui`는
`run_voice()`와 똑같은 카메라 창·상태 패널을 쓰므로, 로봇 없이 화면부터
먼저 확인하고 싶을 때 이걸 쓴다 — `sign_control` 모델(model.pt 등)과
카메라는 필요하지만 로봇·RealSense는 필요 없다.

로봇을 실제로 움직이는 것은 `block_sort.py`의 `run_voice()`다. 여기를 따로
떼어 둔 이유는 sign_command.py와 같다 — 정규화 표는 손볼 일이 잦은데
그때마다 로봇·마이크가 있는 자리에 있을 필요는 없다.

## 왜 새 LLM 프롬프트가 없나

이 파일이 하는 일은 "자연스러운 문장을 sign_command.py가 아는 정확한
글로스로 바꾸는 것"뿐이다. 실제 판단(LLM 해석·규칙 해석·복제 명령 처리)은
`sign_command.command_ready()`를 그대로 부른다 — 그 함수는 글로스가
수어에서 왔는지 음성에서 왔는지 모르는 순수 함수다.

## legacy 코드에서 옮겨온 것

마이크(`MicController`)·웨이크워드(`WakeupWord`)·STT는
`legacy/voice_processing/voice_processing/`를 그대로 옮겨왔다. 딱 하나 고친
게 있다 — 원래 코드는 `get_package_share_directory("voice_processing")`로
웨이크워드 모델 파일을 찾는데, legacy가 ROS 워크스페이스 밖으로 나가면서
이 호출이 깨진다(legacy/README.md 참고). `hand_gesture_control.py`의
`GESTURE_TASK` 패턴(같은 폴더 기본값 + 환경변수로 덮어쓰기)을 그대로 썼다.

`device_index`는 legacy 코드에서도 실제로는 안 쓰인다 — `open_stream()`이
`input_device_index`를 안 넘겨서 시스템 기본 입력 장치를 그냥 연다. 그래서
카메라 인덱스처럼 장치를 잘못 잡는 문제 자체가 없다.
"""
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))

# ── 마이크 설정 ──────────────────────────────────────────────────────
RECORD_SECONDS = int(os.environ.get("VOICE_RECORD_SEC", 5))
# 웨이크워드 모델. **파일 경로** 또는 **openwakeword 내장 이름** 둘 다 된다.
#
#   VOICE_WAKE_MODEL=alexa            내장 (alexa / hey_jarvis / hey_mycroft / hey_rhasspy)
#   VOICE_WAKE_MODEL=/path/x.tflite   직접 만든 모델
#
# 내장을 쓸 수 있게 한 이유 (2026-08-10). hello_rokey 가 이 PC·이 목소리에서
# 전혀 안 잡혔다. 같은 녹음으로 세 모델을 비교했더니:
#     hello_rokey_8332_32  최고 0.0013   (합성 잡음 바닥값 0.0009 와 같은 수준)
#     alexa                최고 0.9312   ← 같은 오디오, 같은 마이크
#     hey_jarvis           최고 0.0092
# 즉 특징 추출·마이크·리샘플링은 멀쩡하고(안 그러면 alexa 도 못 터진다),
# hello_rokey 가 이 발음을 학습하지 않은 것이다. 음량을 ×0.15 까지 낮춰
# 포화를 없애도 0.0012 로 그대로였다 — 클리핑 문제도 아니었다.
# 모델 파일 자체는 legacy 두 벌과 md5 가 같아 손상도 아니다.
#
# 기본값은 hello_rokey 그대로 둔다 — 팀원 환경에서는 잡히고 있고(오탐을
# 줄이려 문턱을 올린 커밋이 있다), 기본을 바꾸면 그쪽이 조용히 달라진다.
WAKE_MODEL = os.environ.get(
    "VOICE_WAKE_MODEL", os.path.join(HERE, "hello_rokey_8332_32.tflite"))
WAKE_MODEL_PATH = WAKE_MODEL          # 예전 이름 (다른 곳에서 참조할 수 있다)


def _is_model_file(spec):
    """파일 경로처럼 보이면 True. 내장 이름("alexa")이면 False."""
    return os.sep in spec or spec.endswith((".tflite", ".onnx"))


def _default_phrase(spec):
    """모델 이름에서 '부르는 말' 을 만든다. hello_rokey_8332_32 → hello rokey.

    화면 안내문에 쓴다. 모델을 바꾸면 안내도 같이 바뀌어야 한다 — alexa 를
    쓰면서 "hello rokey 라고 불러주세요" 가 뜨면 아무도 못 부른다.
    뒤에 붙는 학습 일련번호(8332_32)와 버전(v0.1)은 부를 말이 아니라 뗀다.
    """
    base = os.path.basename(spec).split(".")[0] if _is_model_file(spec) else spec
    words = [w for w in base.split("_") if not w.isdigit() and not w.startswith("v")]
    return " ".join(words) or base


WAKE_PHRASE = os.environ.get("VOICE_WAKE_PHRASE", _default_phrase(WAKE_MODEL))
# 웨이크워드("hello rokey") 신뢰도 문턱. 넘으면 듣기 시작한다.
#
# 0.4 → 0.2 (2026-08-10). 0.4 는 오탐을 줄이려고 0.3 에서 올린 값인데,
# 그 조정은 **다른 마이크 기준**이었다. 이 PC 는 노트북 내장 DMIC 라
# 조용한 방에서도 배경 음량이 4~5% 로 깔려 있어(실측) 신뢰도가 그만큼
# 눌린다 — 0.4 로는 불러도 안 잡혔다.
#
# 낮출수록 오탐(옆 대화를 웨이크워드로 오인)이 는다. 다만 음성모드는
# 웨이크워드만으로 로봇이 움직이지 않는다 — 명령을 해석한 뒤 "실행"이라고
# 한 번 더 말해야 한다(run_voice 주석). 그래서 오탐의 대가가 '헛되이
# 5초 녹음' 정도로 작고, 못 잡히는 쪽이 훨씬 답답하다.
#
# 헤드셋처럼 입에 가까운 마이크를 쓰면 다시 올리는 게 낫다:
#   VOICE_WAKE_THRESHOLD=0.4 python3 block_sort.py sign
WAKE_THRESHOLD = float(os.environ.get("VOICE_WAKE_THRESHOLD", 0.2))
LEVEL_GAIN = float(os.environ.get("VOICE_LEVEL_GAIN", 6.0))   # 레벨 바 표시용 증폭 배수

# ── 모드 전환·확인 문구 ──────────────────────────────────────────────
# 발음이 조금씩 달라도 잡히도록 여러 표현을 인정한다. sign_command.py의
# MODE_GLOSS("모드변경", 제어모드 전환)와 헷갈리지 않게 다른 단어를 쓴다.
ENTER_VOICE_PHRASES = ("음성모드", "보이스모드")
EXIT_VOICE_PHRASES = ("작업모드", "수어모드", "돌아가")
CONFIRM_PHRASES = ("실행", "네", "확인")
CONFIRM_TIMEOUT_SEC = float(os.environ.get("VOICE_CONFIRM_TIMEOUT", 8.0))


def _match_any(text, phrases):
    return any(p in text for p in phrases)


def is_enter_voice(text):
    return bool(text) and _match_any(text, ENTER_VOICE_PHRASES)


def is_exit_voice(text):
    return bool(text) and _match_any(text, EXIT_VOICE_PHRASES)


def is_confirm(text):
    return bool(text) and _match_any(text, CONFIRM_PHRASES)


# ── 문장 → 글로스 정규화 ─────────────────────────────────────────────
# LLM을 안 쓴다 — sign_command.parse()가 이미 LLM을 쓰므로, 여기서 또
# 쓰면 명령 하나에 네트워크 왕복이 두 번(정규화 + 해석) 된다. 색·구역·
# 동작어를 사람이 부르는 표현은 몇 가지로 정해져 있어 정규식/사전 매칭만
# 으로 충분하다.
_COLOR_PATTERNS = {
    "빨강": ("빨간색", "빨강", "빨간"),
    "주황": ("주황색", "주황"),
    "노랑": ("노란색", "노랑", "노란"),
    "초록": ("초록색", "초록", "녹색"),
    "파랑": ("파란색", "파랑", "파란"),
    "보라": ("보라색", "보라"),
}
_ACTION_PATTERNS = {
    "들다": ("들어서", "들어", "들다", "집어서", "집어"),
    "놓다": ("놓아줘", "놓아", "놓다", "내려줘", "내려", "놔줘", "놔"),
}
_ZONE_RE = re.compile(r"(\d+)\s*번\s*구역")


def normalize(text):
    """자연스러운 문장 → 글로스 리스트. 등장 순서를 최대한 지킨다.

    "빨간색 블록을 3번 구역에 놓아줘" → ["빨강", "3번구역", "놓다"]

    순서가 완전히 정확할 필요는 없다 — sign_command.parse()의 LLM·규칙
    해석이 순서가 뒤집혀도 짝짓는다("3번구역 빨강 놓다"도 통과, 실측 확인됨
    — sign_command._CASES 참고).
    """
    found = []   # [(문자열 위치, 글로스), ...] — 위치순 정렬용

    for gloss, variants in _COLOR_PATTERNS.items():
        for v in variants:
            i = text.find(v)
            if i != -1:
                found.append((i, gloss))
                break   # 같은 색은 한 번만 잡는다

    for m in _ZONE_RE.finditer(text):
        found.append((m.start(), f"{m.group(1)}번구역"))

    for gloss, variants in _ACTION_PATTERNS.items():
        for v in variants:
            i = text.find(v)
            if i != -1:
                found.append((i, gloss))
                break

    if "좌우대칭" in text or "대칭" in text:
        idx = text.find("좌우대칭")
        if idx == -1:
            idx = text.find("대칭")
        found.append((idx, "좌우대칭"))
    elif "똑같이" in text:
        found.append((text.find("똑같이"), "똑같이"))

    found.sort(key=lambda t: t[0])
    return [g for _, g in found]


def command_ready_from_text(text, zones):
    """정규화 + sign_command.command_ready() 를 한 번에. (steps, how) 또는 None."""
    sys.path.insert(0, HERE)
    import sign_command as sc
    glosses = normalize(text)
    return sc.command_ready(glosses, list(zones))


# ── legacy voice_processing 이식 ─────────────────────────────────────

@dataclass
class MicConfig:
    chunk: int = 12000
    rate: int = 48000
    channels: int = 1
    fmt: int = None            # __post_init__ 에서 pyaudio.paInt16 로 채운다
    buffer_size: int = 24000

    def __post_init__(self):
        if self.fmt is None:
            import pyaudio
            self.fmt = pyaudio.paInt16


class MicController:
    """legacy/voice_processing/voice_processing/MicController.py 에서 확장.

    웨이크워드 감지 동안 스트림을 계속 읽는 용도다. 실제 명령 문장은
    STT가 sounddevice로 따로 녹음한다(legacy 구조 그대로 — 아래 STT 참고).

    PyAudio 호스트(self.audio)는 첫 open_stream() 에서 한 번만 만들고
    close_stream() 에서는 스트림만 닫는다 — 원래는 매 발화마다 호스트를
    새로 띄웠다(pyaudio.PyAudio() 는 드라이버 열거를 포함해 비용이 있다).
    _Engine 이 이 인스턴스를 세션 내내 재사용하므로 호스트는 프로세스가
    끝날 때까지 살아있어도 된다.
    """

    def __init__(self, config: MicConfig = None):
        self.config = config or MicConfig()
        self.frames = []
        self.audio = None
        self.stream = None
        self.sample_width = None

    def open_stream(self):
        import pyaudio
        if self.audio is None:
            self.audio = pyaudio.PyAudio()
            self.sample_width = self.audio.get_sample_size(self.config.fmt)
        self.stream = self.audio.open(
            format=self.config.fmt,
            channels=self.config.channels,
            rate=self.config.rate,
            input=True,
            frames_per_buffer=self.config.chunk,
        )

    def close_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None


class WakeupWord:
    """legacy/voice_processing/voice_processing/wakeup_word.py 에서 확장 —
    모델을 ROS 패키지 share 대신 WAKE_MODEL 로 찾고, **내장 모델도 받는다.**
    """

    def __init__(self, buffer_size):
        self.model = None
        # 결과 딕셔너리의 열쇠. **모델을 만든 뒤 모델에게 직접 물어본다**
        # (set_stream 참고). 예전에는 파일명에서 잘라 추측했는데 —
        #     hello_rokey_8332_32.tflite → "hello_rokey_8332_32"   맞음
        #     alexa_v0.1.tflite          → "alexa_v0"  ✗ 실제 키는 "alexa"
        # 내장 모델을 쓰려는 순간 KeyError 로 죽는다. 추측하지 않는다.
        self.model_name = None
        self.stream = None
        self.buffer_size = buffer_size

    def is_wakeup(self, on_level=None):
        """on_level 이 있으면 이번 청크의 음량(0~1)을 콜백으로 넘긴다.

        마이크가 실제로 뭘 듣고 있는지 눈으로 확인할 방법이 없다는 피드백을
        받고 추가했다 — 웨이크워드 대기 중(0.25초 청크마다)이 마이크 입력을
        가장 자주 샘플링하는 지점이라 여기서 뽑는다.
        """
        import numpy as np
        from scipy.signal import resample
        audio_chunk = np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False),
            dtype=np.int16)
        if on_level is not None:
            level = float(np.abs(audio_chunk).mean()) / 32768.0
            # 화면 표시용 증폭 — 평범하게 말하는 소리는 최댓값(32768) 근처까지
            # 거의 안 가서 그대로 쓰면 막대가 항상 조금만 채워진다. 실제 녹음엔
            # 안 쓰고(STT는 원본 audio_chunk 그대로 씀) 눈에 보이는 막대만
            # sqrt 로 압축해서 조용한 소리도 잘 보이게 한다.
            level = min(level * LEVEL_GAIN, 1.0) ** 0.5
            on_level(level)
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / 48000))
        outputs = self.model.predict(audio_chunk, threshold=0.1)
        confidence = outputs[self.model_name]
        return confidence > WAKE_THRESHOLD

    def set_stream(self, stream):
        if self.model is None:
            # .tflite 로딩 + 추론기 초기화는 한 번이면 된다 — 원래는 매
            # set_stream() 호출(즉 매 발화)마다 다시 만들어서 명령 하나에
            # 몇 초씩 지연이 낍니다. self 가 _Engine 에 물려 세션 내내
            # 살아있으므로 여기서 한 번 캐싱해두면 그다음부터는 재사용된다.
            from openwakeword.model import Model
            # **sys.exit 를 쓰지 않는다.** 이 함수는 run_voice() 의 배경
            # 스레드에서 불린다. SystemExit 은 Exception 이 아니라
            # BaseException 이라 거기 `except Exception` 에 안 걸리고,
            # 스레드만 조용히 사라진다 — 화면은 "불러주세요" 를 계속 띄우는데
            # 실제로는 죽어 있다(2026-08-10 에 이걸로 한참 헤맸다).
            # 예외로 올려야 부르는 쪽이 로그·대시보드에 남길 수 있다.
            if _is_model_file(WAKE_MODEL) and not os.path.exists(WAKE_MODEL):
                raise FileNotFoundError(
                    f"웨이크워드 모델이 없습니다: {WAKE_MODEL}\n"
                    f"  legacy/voice_processing/resource/hello_rokey_8332_32.tflite 를 "
                    f"block_sort/ 로 복사하거나, 내장 모델을 쓰세요 "
                    f"(VOICE_WAKE_MODEL=alexa).")
            self.model = Model(wakeword_models=[WAKE_MODEL])
            # **열쇠는 모델에게 물어본다.** 파일명에서 추측하면 내장 모델
            # (alexa_v0.1.tflite → 키는 "alexa")에서 어긋난다.
            keys = list(self.model.models.keys())
            if not keys:
                raise RuntimeError(f"웨이크워드 모델을 못 읽었습니다: {WAKE_MODEL}")
            self.model_name = keys[0]
        self.stream = stream


class STT:
    """legacy/voice_processing/voice_processing/stt.py 그대로.

    MicController와 별도로 sounddevice로 자기 녹음을 한다(legacy 구조).
    웨이크워드 판정용 스트림(MicController)은 이 녹음 전에 닫는다 —
    두 오디오 라이브러리가 같은 장치를 동시에 여는 걸 피하기 위해서다
    (legacy 원본은 안 닫고 넘어갔는데, 실기 없이 검증 못 해서 안전한 쪽으로).
    """

    def __init__(self, openai_api_key):
        from openai import OpenAI
        self.client = OpenAI(api_key=openai_api_key)
        self.duration = RECORD_SECONDS
        self.samplerate = 16000   # Whisper는 16kHz를 선호

    def speech2text(self):
        import sounddevice as sd
        import scipy.io.wavfile as wav
        print(f"음성 녹음을 시작합니다. {self.duration}초 동안 말해주세요...")
        audio = sd.rec(int(self.duration * self.samplerate),
                       samplerate=self.samplerate, channels=1, dtype='int16')
        sd.wait()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, self.samplerate, audio)
            with open(f.name, "rb") as fh:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1", file=fh)
        print(f"STT 결과: {transcript.text}")
        return transcript.text


# ── 인식 엔진 ─────────────────────────────────────────────────────────
# 무거운 것(openwakeword 모델, OpenAI 클라이언트)을 첫 호출 때만 만든다 —
# selftest/parse는 이것들을 아예 안 건드리므로 마이크·API 키가 없어도
# 카메라·로봇 없는 개발 환경에서 그대로 돌아간다.
_engine = None


class _Engine:
    def __init__(self):
        sys.path.insert(0, HERE)
        import sign_command as sc
        key = sc._api_key()
        if key is None:
            sys.exit("OPENAI_API_KEY 가 없습니다 — 음성 인식은 STT(Whisper)에 "
                      "키가 필요합니다. 저장소 최상위 .env 를 확인하세요.")
        self.mic_cfg = MicConfig()
        self.mic = MicController(self.mic_cfg)   # 세션 내내 재사용 (PyAudio 호스트 유지)
        self.wake = WakeupWord(self.mic_cfg.buffer_size)
        self.stt = STT(openai_api_key=key)

    def listen(self, should_stop=None, on_level=None, on_wake=None, require_wake=True):
        """웨이크워드 대기 → 녹음 → STT. 문장 텍스트를 돌려준다.

        should_stop 은 대시보드에서 강제로 모드를 바꿨을 때 이 대기를 빨리
        끝내기 위한 콜백이다 — 안 주면(단독 CLI 실행 등) 그냥 계속 기다린다.
        웨이크워드가 안 들리면 무한정 블로킹되므로, block_sort.py 의
        run_voice() 처럼 대시보드로도 빠져나와야 하는 자리에서는 꼭 넘긴다.

        on_level 은 WakeupWord.is_wakeup() 참고 — 실시간 마이크 입력 레벨을
        화면에 보여줄 때 쓴다.

        on_wake 는 웨이크워드가 감지된 바로 그 순간(녹음 시작 직전) 한 번
        불린다 — "hello rokey 라고 불러주세요" 안내문을 그 시점에 "듣는
        중..."으로 바꿔주는 용도. 이게 없으면 대기 중과 실제 녹음 중을
        화면만 보고 구분할 수 없었다.

        require_wake=False 면 웨이크워드 대기를 건너뛰고 바로 녹음한다 —
        명령을 이미 한 번 인식한 직후에 "실행"이라고 확인만 받는 자리
        (run_voice() 의 확인 단계)에서 쓴다. 이걸 안 주면 확인 단계도
        "hello rokey"부터 다시 불러야 하는데, 화면엔 그 안내가 없어서
        실사용에서 계속 타임아웃으로 취소되는 문제가 있었다.
        """
        if not require_wake:
            return self.stt.speech2text()

        try:
            self.mic.open_stream()
        except OSError as e:
            print(f"마이크를 열지 못했습니다: {e}")
            return None
        try:
            self.wake.set_stream(self.mic.stream)
            print(f"웨이크워드 대기 중 ('{WAKE_PHRASE}')...")
            while not self.wake.is_wakeup(on_level=on_level):
                if should_stop is not None and should_stop():
                    return None
            print("웨이크워드 감지!")
            if on_wake is not None:
                on_wake()
        finally:
            self.mic.close_stream()   # STT가 sounddevice로 다시 열기 전에 장치를 비워준다
                                       # (예외로 빠져나가도 항상 닫는다)
        return self.stt.speech2text()


def _get_engine():
    global _engine
    if _engine is None:
        _engine = _Engine()
    return _engine


def listen_once(should_stop=None, on_level=None, on_wake=None, require_wake=True):
    """웨이크워드 대기 → 녹음 → STT. 문장 텍스트만 돌려준다 (해석은 안 함).

    should_stop 은 _Engine.listen() 참고 — 대시보드 강제 전환에 빨리
    반응해야 하는 자리(block_sort.py 의 진입 스레드·run_voice())에서 넘긴다.
    on_level 도 _Engine.listen() 참고 — 실시간 마이크 레벨 표시용.
    on_wake 도 _Engine.listen() 참고 — 웨이크워드 감지 순간 화면 문구를
    바꾸는 용도.
    require_wake=False 도 _Engine.listen() 참고 — 명령 확인("실행") 단계처럼
    이미 한 번 웨이크워드를 거친 직후에는 다시 안 기다리고 바로 녹음한다.
    """
    return _get_engine().listen(should_stop=should_stop, on_level=on_level,
                                 on_wake=on_wake, require_wake=require_wake)


def watch(zones=(1, 2, 3, 4)):
    """로봇 없이 인식만 계속 보는 모드. sign_demo.py run 의 음성판.

    웨이크워드 → STT → 정규화 → 해석까지 반복해서 찍기만 한다.
    run_one()/copy_human() 을 절대 안 부른다 — 여기서는 확인("실행")을
    받아도 실행할 대상이 없다. Ctrl+C 로 종료한다.
    """
    print("음성 인식 감시 모드 — 로봇을 움직이지 않습니다. Ctrl+C 로 종료.\n")
    try:
        while True:
            text = listen_once()
            if not text:
                continue
            print(f'\n들은 문장: "{text}"')
            if is_enter_voice(text):
                print("  → 음성모드 진입 문구로 인식됨 (실제 전환은 안 함)")
                continue
            if is_exit_voice(text):
                print("  → 작업모드 복귀 문구로 인식됨 (실제 전환은 안 함)")
                continue
            if is_confirm(text):
                print("  → 확인 문구로 인식됨")
                continue
            glosses = normalize(text)
            r = command_ready_from_text(text, zones)
            print(f"  정규화: {glosses}")
            if r and r[0]:
                steps, how = r
                print(f"  해석[{how}]: " + ", ".join(f"{s}→{z}번" for s, z in steps))
            else:
                print("  해석: 실패 — 명령이 되지 않음")
    except KeyboardInterrupt:
        print("\n종료합니다.")


def draw_level_bar(frame, x, y, w, h, level):
    """실시간 마이크 입력 레벨을 녹색 막대로 그린다. level 은 0~1.

    block_sort.py 의 run_voice() 와 여기(watch_gui)가 같이 쓴다 — 마이크가
    실제로 반응하는지 눈으로 볼 방법이 없다는 피드백으로 추가했다.
    """
    import cv2
    cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1)      # 빈 테두리
    fill_w = max(0, min(int(w * level), w))
    if fill_w > 0:
        cv2.rectangle(frame, (x + 1, y + 1), (x + fill_w - 1, y + h - 1),
                      (60, 200, 60), -1)                                 # 채워진 만큼 녹색


def watch_gui(zones=(1, 2, 3, 4)):
    """block_sort.py 의 run_voice() 와 같은 화면을 로봇 없이 테스트한다.

    sign_processing 카메라 창을 그대로 띄우고, 왼쪽 아래에 음성 상태를
    그린다. 로직은 run_voice() 와 똑같이 짜되, self.run_one()/copy_human()
    호출 자리만 print 로 바꿨다 — **여기서는 로봇이 전혀 필요 없다.**
    카메라와 sign_control 모델(model.pt, hand/pose task)만 있으면 된다.
    창에서 Q 를 누르면 종료한다.
    """
    import threading
    import cv2

    sys.path.insert(0, HERE)
    import sign_command as sc
    sys.path.insert(0, sc.SIGN_PROCESSING_DIR)
    from sign_processing.gesture_recognizer import WIN

    print("GUI 테스트 모드 — 로봇을 전혀 부르지 않습니다. 창에서 Q 로 종료.\n")
    rec = sc.make_recognizer()

    stop = {"v": False}
    status = {"text": f'🎤 "{WAKE_PHRASE}" 라고 불러주세요', "sub": "", "level": 0.0}
    lock = threading.Lock()

    def set_status(text, sub=""):
        with lock:
            status["text"], status["sub"] = text, sub

    def set_level(lvl):
        with lock:
            status["level"] = lvl

    def _audio_loop():
        while not stop["v"]:
            set_status(f'🎤 "{WAKE_PHRASE}" 라고 불러주세요')
            text = listen_once(should_stop=lambda: stop["v"], on_level=set_level)
            if not text:
                continue
            set_status(f'들음: "{text}"', "해석 중...")
            if is_exit_voice(text) or is_enter_voice(text):
                set_status(f'들음: "{text}"', "모드 전환 문구로 인식됨 (여기선 전환 안 함)")
                continue
            r = command_ready_from_text(text, zones)
            if not r or not r[0]:
                set_status(f'들음: "{text}"', "해석 실패 — 다시 말해주세요")
                continue
            steps, how = r
            desc = ", ".join(f"{s}→{z}번" for s, z in steps)
            print(f'[{how}] "{text}" → {desc}  — {CONFIRM_TIMEOUT_SEC:.0f}초 안에 "실행"이라고 말하세요')
            set_status(f"[{how}] {desc}", f'{CONFIRM_TIMEOUT_SEC:.0f}초 안에 "실행"이라고 말하세요')

            deadline = time.time() + CONFIRM_TIMEOUT_SEC
            confirmed = False
            while time.time() < deadline and not stop["v"]:
                reply = listen_once(should_stop=lambda: stop["v"] or time.time() >= deadline,
                                    on_level=set_level, require_wake=False)
                if reply and is_confirm(reply):
                    confirmed = True
                break
            if confirmed:
                print(f"(로봇 없음) 실행했을 것: {desc}")
                set_status(f"실행(가상): {desc}")
            else:
                print("취소됨 — 확인 없음")
                set_status(f'들음: "{text}"', "취소됨 — 확인 없음")

    threading.Thread(target=_audio_loop, daemon=True).start()

    cap = rec._camera()
    try:
        while not stop["v"]:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            panel = min(150, h // 3)
            cv2.rectangle(frame, (0, h - panel), (w, h), (25, 25, 25), -1)
            with lock:
                text, sub, level = status["text"], status["sub"], status["level"]
            draw_level_bar(frame, 30, h - panel + 96, 260, 18, level)
            if rec.painter:
                items = [((30, h - panel + 12), [(text, (255, 255, 255))], 30, 0)]
                if sub:
                    items.append(((30, h - panel + 58), [(sub, (120, 255, 255))], 22, 0))
                frame = rec.painter.render(frame, items)
            else:
                cv2.putText(frame, text, (30, h - panel + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(WIN, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                stop["v"] = True
    finally:
        stop["v"] = True
        rec.close()
        print("종료합니다.")


# ── 자체 검증 ───────────────────────────────────────────────────────
# 마이크·로봇·네트워크 없이 돌아가는 것들만 넣는다 (정규화는 순수 함수라
# 그대로, command_ready() 도 API 키 없으면 규칙 해석으로 내려간다).
_CASES = [
    ("빨간색 블록을 3번 구역에 놓아줘",        ["빨강", "3번구역", "놓다"]),
    ("파란색은 1번 구역에 초록색은 2번 구역에 놔줘",
     ["파랑", "1번구역", "초록", "2번구역", "놓다"]),
    ("3번 구역에 있는 거 1번 구역으로 옮겨줘",  ["3번구역", "1번구역"]),
    ("똑같이 놔줘",                          ["똑같이", "놓다"]),
    ("좌우대칭으로 배치해줘",                  ["좌우대칭"]),
    ("음성모드로 바꿔줘",                      []),   # 색·구역 어휘가 아니므로 빈손
]


def selftest():
    ok = True
    print("정규화 테스트")
    for text, expected in _CASES:
        got = normalize(text)
        mark = "OK" if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"  [{mark}] \"{text}\" → {got}" + (f"  (기대 {expected})" if got != expected else ""))

    print("\n전환·확인 문구 테스트")
    checks = [
        (is_enter_voice("음성모드로 바꿔줘"), True, "음성모드 진입 감지"),
        (is_exit_voice("작업모드로 돌아가줘"), True, "작업모드 복귀 감지"),
        (is_confirm("네 실행해줘"), True, "확인 문구 감지"),
        (is_enter_voice("빨간색 3번 구역"), False, "일반 명령은 진입 문구 아님"),
    ]
    for got, expected, label in checks:
        mark = "OK" if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"  [{mark}] {label}")

    print("\nsign_command.command_ready() 연동 테스트 (규칙 해석 기준)")
    sys.path.insert(0, HERE)
    import sign_command as sc
    link_cases = [
        ("빨간색 블록을 3번 구역에 놓아줘", [("빨간색", 3)]),
        ("3번 구역에 있는 거 1번 구역으로 옮겨줘", [(3, 1)]),
    ]
    for text, expected in link_cases:
        glosses = normalize(text)
        r = sc.rule_parse(" ".join(glosses), [1, 2, 3, 4])
        mark = "OK" if r == expected else "FAIL"
        if r != expected:
            ok = False
        print(f"  [{mark}] \"{text}\" → {r}" + (f"  (기대 {expected})" if r != expected else ""))

    print(f"\n{'모두 통과' if ok else '실패 있음'}")
    return ok


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    m = sys.argv[1]
    if m == "selftest":
        sys.exit(0 if selftest() else 1)
    elif m == "parse":
        if len(sys.argv) < 3:
            sys.exit('사용법: parse "빨간색 3번 구역에 놔줘"')
        text = sys.argv[2]
        glosses = normalize(text)
        r = command_ready_from_text(text, (1, 2, 3, 4))
        print(f"정규화: {glosses}")
        print(f"해석: {r}")
    elif m == "listen":
        text = listen_once()
        print(f"\n들은 문장: {text}")
        if text:
            print(f"정규화: {normalize(text)}")
    elif m == "watch":
        watch()
    elif m == "watch-gui":
        watch_gui()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
