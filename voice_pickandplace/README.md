# Doosan M0609 — 음성 기반 Pick & Place

음성 명령("hello rokey" → "사과 가져와")으로 Doosan M0609 로봇암이
RealSense로 대상을 찾아 집어 옮기는 ROS2 시스템.

과일(`fruit_best.pt`)과 공구(`yolov8n_tools_0122.pt`) 두 가지 모델을 지원한다.

```
음성 → 웨이크워드 → STT → LLM 키워드 추출 → YOLO 검출 → 좌표 변환 → 로봇 동작
```

---

## 구성

### 현재 사용 패키지

| 패키지 | 노드 | 역할 |
|---|---|---|
| `voice_processing` | `get_keyword` | 웨이크워드, Whisper STT, GPT 키워드 추출 |
| `object_detection` | `object_detection` | YOLO 검출 + RealSense 깊이 → 3D 좌표 |
| `robot_control` | `robot_control` | DSR API 로봇 제어, RG2 그리퍼 |
| `robot_control` | `person_tracker` | J1축만 회전시켜 사람 추종 (별개 기능) |
| `od_msg` | — | 서비스/메시지 정의 |

세 노드는 독립 패키지로 분리되어 있어 각각 따로 실행하거나 컨테이너화할 수 있다.

### 초기 버전 (참고용)

| 패키지 | 비고 |
|---|---|
| `pick_and_place_voice` | 위 세 기능을 한 패키지에 담았던 초기 통합판. `COLCON_IGNORE` 로 빌드에서 제외됨 |
| `pick_and_place_text` | 음성 대신 텍스트로 입력하는 변형 |
| `rokey` | 공용 유틸 |

---

## 요구 환경

```
Ubuntu 22.04 / ROS2 Humble
Python 3.10, numpy 1.26.4      # 2.x 는 cv_bridge 와 충돌해 세그폴트가 난다
ultralytics, openai, langchain-openai, openwakeword
Doosan M0609 + OnRobot RG2
Intel RealSense D435i
```

별도로 Doosan 드라이버 워크스페이스(`dsr_msgs2`, `dsr_bringup2` 등)가 필요하다.

---

## 설치

```bash
git clone https://github.com/genewoolim-beep/hyupdong2.git
cd hyupdong2
pip install -r requirements.txt        # numpy 1.26.4 고정이 중요하다

cd voice_pickandplace
colcon build --symlink-install
source install/setup.bash
```

> `colcon build` 는 반드시 `voice_pickandplace/` 안에서 실행한다.
> 저장소 최상위에서 돌리면 `build/` `install/` 이 루트에 생긴다.

Doosan 드라이버 워크스페이스를 **먼저** 소싱해야 `dsr_msgs2` 가 잡힌다.

```bash
source ~/<doosan_ws>/install/setup.bash
```

### 환경변수

| 변수 | 기본값 | 용도 |
|---|---|---|
| `YOLO_MODEL` | `yolov8n.pt` | `person_tracker` 가 쓸 가중치. 없으면 자동으로 내려받는다 |
| `ROS_DOMAIN_ID` | — | 노드끼리 맞춰야 한다 |
| `RMW_IMPLEMENTATION` | — | 전 노드가 동일해야 한다 |

### API 키 설정

`.env` 는 저장소에 포함되지 않는다. 예시 파일을 복사해 본인 키를 넣는다.

```bash
cp src/voice_processing/resource/.env.example src/voice_processing/resource/.env
# 편집기로 열어 OPENAI_API_KEY 값을 채운다
```

> **주의** — `.env` 는 `.gitignore` 에 등록되어 있다. 강제로 커밋하지 말 것.
> 공개 저장소에 올라간 API 키는 수분 내에 자동 수집된다.

---

## 실행

DDS 설정을 먼저 맞춘다. 도메인이 다르면 노드끼리 서로를 못 본다.

```bash
export ROS_DOMAIN_ID=65
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

터미널 4개로 나눠 실행한다.

```bash
# 1) 로봇 드라이버
ros2 launch m0609_rg2_bringup m0609_rg2.launch.py

# 2) RealSense
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

# 3) 검출 + 로봇 제어
ros2 run object_detection object_detection
ros2 run robot_control robot_control

# 4) 음성 입력
ros2 run voice_processing get_keyword
```

`hello rokey` 로 깨운 뒤 명령을 말한다.

---

## 캘리브레이션

`T_gripper2camera.npy` 가 있어야 카메라 좌표를 로봇 좌표로 옮길 수 있다.
체커보드로 핸드아이 캘리브레이션을 수행한다 (재투영 표준편차 2 mm 수준).

카메라를 재장착했다면 반드시 다시 구해야 한다.

---

## 알려진 함정

작업하며 실제로 겪은 것들이다.

| 증상 | 원인과 해결 |
|---|---|
| `_ARRAY_API not found` 후 세그폴트 | numpy 2.x. `pip install numpy==1.26.4` |
| `No module named pymodbus.client.sync` | pymodbus 3.x. 버전 분기 임포트로 처리됨 |
| 그리퍼 `No response after 3 retries` | 툴체인저가 유휴 TCP를 끊는다. 재접속 래퍼로 처리됨 |
| `The passed service type is invalid` | Doosan 워크스페이스 미소싱 |
| `SetSingularHandlingForce` NameError | `dsr_msgs2` 가 두 곳에 빌드됨. 순서를 지켜 소싱할 것 |
| `movel` 이 무시됨 | `trans()` 에 리스트를 넘기면 좌표가 깨진다. z를 직접 더할 것 |
| 노드끼리 안 보임 | `ROS_DOMAIN_ID` / `RMW_IMPLEMENTATION` 불일치 |
| DDS 가 docker0 까지 탐색 | `CYCLONEDDS_URI` 로 인터페이스 고정 |

---

## 속도 설정

`robot_control` 상단 상수로 조정한다. `movej` 는 관절 속도(deg/s),
`movel` 은 직선 속도(mm/s) 로 단위가 다르다.

```python
VELOCITY, ACC     = 81, 81      # 관절
L_VELOCITY, L_ACC = 150, 150    # 직선
```

물건을 집은 뒤 z 방향으로 300 mm 들어올린 다음 이송한다.
