# 협동 2 — Doosan M0609 프로젝트 모음

Doosan M0609 협동로봇을 대상으로 한 프로젝트들. 각 폴더가 독립적으로 빌드·실행된다.

| 폴더 | 프로젝트 | 입력 | 상태 |
|---|---|---|---|
| [`voice_pickandplace/`](voice_pickandplace/) | 음성 기반 Pick & Place | 음성 | 동작 확인 |
| `sign_control/` | 수어 기반 로봇 제어 | 수어 | 준비 중 |

---

## voice_pickandplace

음성 명령("hello rokey" → "사과 가져와")으로 로봇암이 대상을 찾아 집어 옮긴다.

```
음성 → 웨이크워드 → STT → LLM 키워드 추출 → YOLO 검출 → 좌표 변환 → 로봇 동작
```

`voice_processing` / `object_detection` / `robot_control` 세 개의 독립 ROS2 패키지로
구성되어 있어 각각 따로 실행하거나 컨테이너화할 수 있다.

자세한 내용은 [voice_pickandplace/README.md](voice_pickandplace/README.md) 참고.

---

## sign_control (준비 중)

음성 대신 **수어**로 같은 로봇을 제어한다. 후단(LLM → 태스크 → 로봇)은
`voice_pickandplace` 와 동일한 구조를 재사용하고, 입력단만 교체한다.

```
수어 → MediaPipe 랜드마크 → 시퀀스 분류 → 글로스 → LLM → 태스크 → 로봇
```

---

## 공통 요구사항

```
Ubuntu 22.04 / ROS2 Humble
Python 3.10, numpy 1.26.4      # 2.x 는 cv_bridge 와 충돌한다
Doosan M0609 + OnRobot RG2
Intel RealSense D435i
```

별도로 Doosan 드라이버 워크스페이스(`dsr_msgs2`, `dsr_bringup2` 등)가 필요하다.

## API 키

이 저장소는 **공개 저장소**다. `.env` 는 `.gitignore` 로 차단되어 있으며
어떤 경우에도 커밋해서는 안 된다. 각 패키지의 `.env.example` 을 복사해 사용한다.

```bash
cp <패키지>/resource/.env.example <패키지>/resource/.env
```
