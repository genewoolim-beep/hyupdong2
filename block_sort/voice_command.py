#!/usr/bin/env python3
"""
음성 → 문장(STT) → 글로스 정규화 → (색, 구역) 명령   (로봇 의존성 없음)

  python3 voice_command.py selftest                    정규화·해석 자체 검증 (마이크 불필요)
  python3 voice_command.py parse "빨간색 3번 구역에 놔줘"   정규화+해석만 확인 (마이크 불필요)
  python3 voice_command.py listen                       웨이크워드 → 녹음 → STT 텍스트만 출력

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
import io
import os
import re
import sys
import tempfile
import time
import wave
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))

# ── 마이크 설정 ──────────────────────────────────────────────────────
RECORD_SECONDS = int(os.environ.get("VOICE_RECORD_SEC", 5))
WAKE_MODEL_PATH = os.environ.get(
    "VOICE_WAKE_MODEL", os.path.join(HERE, "hello_rokey_8332_32.tflite"))
WAKE_THRESHOLD = float(os.environ.get("VOICE_WAKE_THRESHOLD", 0.3))

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
    """legacy/voice_processing/voice_processing/MicController.py 그대로.

    웨이크워드 감지 동안 스트림을 계속 읽는 용도다. 실제 명령 문장은
    STT가 sounddevice로 따로 녹음한다(legacy 구조 그대로 — 아래 STT 참고).
    """

    def __init__(self, config: MicConfig = None):
        self.config = config or MicConfig()
        self.frames = []
        self.audio = None
        self.stream = None
        self.sample_width = None

    def open_stream(self):
        import pyaudio
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
        if self.audio:
            self.audio.terminate()
            self.audio = None


class WakeupWord:
    """legacy/voice_processing/voice_processing/wakeup_word.py 그대로 —
    모델 경로만 ROS 패키지 share 디렉터리 대신 WAKE_MODEL_PATH를 쓴다.
    """

    def __init__(self, buffer_size):
        self.model = None
        self.model_name = os.path.basename(WAKE_MODEL_PATH).split(".", 1)[0]
        self.stream = None
        self.buffer_size = buffer_size

    def is_wakeup(self):
        import numpy as np
        from scipy.signal import resample
        audio_chunk = np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False),
            dtype=np.int16)
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / 48000))
        outputs = self.model.predict(audio_chunk, threshold=0.1)
        confidence = outputs[self.model_name]
        return confidence > WAKE_THRESHOLD

    def set_stream(self, stream):
        from openwakeword.model import Model
        if not os.path.exists(WAKE_MODEL_PATH):
            sys.exit(
                f"웨이크워드 모델이 없습니다: {WAKE_MODEL_PATH}\n"
                f"  legacy/voice_processing/resource/hello_rokey_8332_32.tflite 를 "
                f"block_sort/ 로 복사하세요.")
        self.model = Model(wakeword_models=[WAKE_MODEL_PATH])
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
        self.wake = WakeupWord(self.mic_cfg.buffer_size)
        self.stt = STT(openai_api_key=key)

    def listen(self, should_stop=None):
        """웨이크워드 대기 → 녹음 → STT. 문장 텍스트를 돌려준다.

        should_stop 은 대시보드에서 강제로 모드를 바꿨을 때 이 대기를 빨리
        끝내기 위한 콜백이다 — 안 주면(단독 CLI 실행 등) 그냥 계속 기다린다.
        웨이크워드가 안 들리면 무한정 블로킹되므로, block_sort.py 의
        run_voice() 처럼 대시보드로도 빠져나와야 하는 자리에서는 꼭 넘긴다.
        """
        mic = MicController(self.mic_cfg)
        try:
            mic.open_stream()
        except OSError as e:
            print(f"마이크를 열지 못했습니다: {e}")
            return None
        self.wake.set_stream(mic.stream)
        print("웨이크워드 대기 중 ('hello rokey')...")
        while not self.wake.is_wakeup():
            if should_stop is not None and should_stop():
                mic.close_stream()
                return None
        print("웨이크워드 감지!")
        mic.close_stream()   # STT가 sounddevice로 다시 열기 전에 장치를 비워준다
        return self.stt.speech2text()


def _get_engine():
    global _engine
    if _engine is None:
        _engine = _Engine()
    return _engine


def listen_once(should_stop=None):
    """웨이크워드 대기 → 녹음 → STT. 문장 텍스트만 돌려준다 (해석은 안 함).

    should_stop 은 _Engine.listen() 참고 — 대시보드 강제 전환에 빨리
    반응해야 하는 자리(block_sort.py 의 진입 스레드·run_voice())에서 넘긴다.
    """
    return _get_engine().listen(should_stop=should_stop)


def collect_command(zones, should_stop=None):
    """listen_once() 로 받은 문장을 해석까지. sign_command.collect_command()와
    같은 모양 (text, steps, how) 을 돌려준다. 문장을 못 받으면 None.
    """
    text = listen_once(should_stop=should_stop)
    if not text:
        return None
    steps, how = command_ready_from_text(text, zones) or ([], "규칙")
    return text, steps, how


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
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
