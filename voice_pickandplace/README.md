# Doosan M0609 — 음성 기반 Pick & Place

음성 명령("hello rokey" → "사과 가져와")으로 Doosan M0609 로봇암이
RealSense로 대상을 찾아 집어 옮기는 ROS2 시스템.

검출 방식은 두 가지다. **색깔(HSV) 검출이 기본**이고, YOLO 는 옵션으로 남아 있다.

```
음성 → 웨이크워드 → STT → LLM 키워드 추출 → 검출(색깔 또는 YOLO) → 좌표 변환 → 로봇 동작
```

---

## 구성

### 현재 사용 패키지

| 패키지 | 노드 | 역할 |
|---|---|---|
| `voice_processing` | `get_keyword` | 웨이크워드, Whisper STT, GPT 키워드 추출 |
| `object_detection` | `object_detection` | 검출 + RealSense 깊이 → 3D 좌표. `get_3d_position` 서비스 제공 |
| `object_detection` | `color_view` | 색깔 검출 결과를 화면에 그려 발행 (확인용) |
| `object_detection` | `hsv_probe` | 화면의 hue 분포를 찍어보는 진단 도구 (1회 실행 후 종료) |
| `robot_control` | `robot_control` | DSR API 로봇 제어, RG2 그리퍼 |
| `robot_control` | `person_tracker` | J1축만 회전시켜 사람 추종 (별개 기능) |
| `od_msg` | — | 서비스/메시지 정의 |

노드는 독립 패키지로 분리되어 있어 각각 따로 실행하거나 컨테이너화할 수 있다.
`color_view` 와 `hsv_probe` 는 확인·진단 전용이라 켜지 않아도 검출 자체는 동작한다.

### 초기 버전 (참고용)

| 패키지 | 비고 |
|---|---|
| `pick_and_place_voice` | 위 세 기능을 한 패키지에 담았던 초기 통합판. `COLCON_IGNORE` 로 빌드에서 제외됨 |
| `pick_and_place_text` | 음성 대신 텍스트로 입력하는 변형 |
| `rokey` | 공용 유틸 |

---

## 검출 방식

`object_detection` 은 검출기를 두 개 갖고 있고, 하나를 골라서 쓴다.
`detection.py` 의 `ObjectDetectionNode(model_name=...)` 가 그 스위치다.

| `model_name` | 클래스 | 방식 | 대상 |
|---|---|---|---|
| `'color'` **(기본값)** | `ColorModel` | OpenCV HSV 임계값 | 색깔 블록 |
| `'yolo'` | `YoloModel` | Ultralytics YOLO (`yolov8n_tools_0122.pt`) | 학습된 공구 클래스 |

둘 다 `get_best_detection(img_node, target) -> (box, score)` 라는 같은 형태로 돌려주기 때문에,
그 뒤의 깊이 조회·좌표 변환 코드는 어느 쪽이 쓰였는지 몰라도 그대로 동작한다.
검출기를 새로 추가할 때도 이 형태만 맞추면 된다.

### 색깔 검출

학습이 필요 없다. `target` 으로 색 이름을 넘기면 (`파란색` / `blue` 둘 다 됨)
HSV 마스크 → 모폴로지 → 컨투어 순으로 그 색 덩어리를 찾는다.
지원하는 색은 빨강 · 주황 · 노랑 · 초록 · 파랑 · 보라 여섯 가지다.

```bash
ros2 service call /get_3d_position od_msg/srv/SrvDepthPosition "{target: '파란색'}"
```

한 장만 보면 그 순간의 그림자나 반사광에 그대로 속기 때문에,
**0.4초 동안 여러 프레임을 모아 매번 같은 자리에 나온 것만 인정한다.**
그래서 돌려주는 `score` 는 덩어리 크기가 아니라 **몇 프레임에서 보였는지의 비율**이다
(0.8 이면 10 장 중 8 장에서 같은 자리에 보였다는 뜻이라, 그대로 신뢰도로 읽으면 된다).

### 색 범위 조정

**HSV 범위는 조명과 물체에 따라 달라진다.** 값은 코드가 아니라 설정 파일에 있다.

```
src/object_detection/resource/color_ranges.json
```

색 범위뿐 아니라 덩어리 크기(`min_area`, `max_area_ratio`), 가로세로 비율(`max_aspect_ratio`),
프레임 합의 설정(`duration_sec`, `min_hit_ratio`) 도 전부 여기 있다.

현장에서 값을 빨리 맞춰볼 때는 **설치본을 직접 고치면 빌드 없이 노드 재시작만으로 반영된다.**

```
install/object_detection/share/object_detection/resource/color_ranges.json
```

> 맞춘 값은 원본(`src/...`)에도 옮겨 적어야 다음 빌드 때 되돌아가지 않는다.

값은 감으로 고치지 말고 `hsv_probe` 로 실제 분포를 재고 맞추는 편이 빠르다.
픽셀이 몰린 구간을 찾아 `color_ranges.json` 에 그대로 넣을 수 있는 형태로 출력해 준다.

```bash
ros2 run object_detection hsv_probe
```

파랑(hue≈100)과 보라(hue≈122)처럼 가까이 붙은 색은 범위가 겹치면
한쪽이 다른 쪽을 통째로 삼켜서 영영 안 잡히니 주의할 것.

### 결과를 눈으로 확인하기

```bash
ros2 run object_detection color_view
ros2 run rqt_image_view rqt_image_view      # /object_detection/color_debug_image 선택
```

여섯 색을 동시에 검출해 박스와 이름을 그려 발행한다. 로그로 두 가지를 알려준다.

- 안 잡히는 색이 있으면 그 이유 (HSV 범위 밖인지, 너무 작아서 걸러졌는지, 배경으로 판단됐는지)
- 서로 다른 색의 박스가 겹치면 경고 — 두 색의 HSV 범위가 겹쳤다는 신호다

---

## 요구 환경

```
Ubuntu 22.04 / ROS2 Humble
Python 3.10, numpy 1.26.4      # 2.x 는 cv_bridge 와 충돌해 세그폴트가 난다
opencv-python, pillow          # 색깔 검출
ultralytics, openai, langchain-openai, openwakeword
Doosan M0609 + OnRobot RG2
Intel RealSense D435i
```

별도로 Doosan 드라이버 워크스페이스(`dsr_msgs2`, `dsr_bringup2` 등)가 필요하다.

`color_view` 가 박스 위에 한글 이름을 그리려면 한글 폰트가 있어야 한다.
OpenCV 내장 폰트는 한글을 못 그려서 Pillow 로 그리기 때문이다. 없으면 노드가 시작하다 죽는다.

```bash
sudo apt install fonts-nanum
```

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
# 1) 로봇 드라이버 — 런치 파일 이름은 bringup.launch.py 다.
ros2 launch m0609_rg2_bringup bringup.launch.py mode:=real host:=192.168.1.100
# 실물 없이 확인만 할 때는 mode:=virtual (기본값)

# 2) RealSense — align_depth 가 켜져야 aligned_depth_to_color 토픽이 나온다.
#    이게 없으면 검출 노드가 깊이를 못 읽고 계속 재시도만 한다.
#    rs_align_depth_launch.py 는 이 버전의 realsense2_camera 에 없다. rs_launch.py 에
#    align_depth.enable:=true 를 주면 같은 토픽이 나온다 (2026-08-04 실측 확인).
ros2 launch realsense2_camera rs_launch.py \
    depth_module.depth_profile:=848x480x30 \
    rgb_camera.color_profile:=1280x720x30 \
    align_depth.enable:=true

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
| 특정 색만 계속 안 잡힘 | HSV 범위 밖. `hsv_probe` 로 실제 hue 를 재고 `COLOR_HSV_RANGES` 를 고칠 것 |
| 색 하나가 옆 색까지 먹음 | 두 색의 hue 범위가 겹쳐 있다. 파랑(≈100)/보라(≈122) 가 대표적 |
| 박스가 옆 블록까지 늘어남 | 붙어 있는 블록이 모폴로지로 합쳐진 것. `MAX_ASPECT_RATIO` 로 걸러진다 |
| 한글 라벨이 `???` 로 나옴 | `cv2.putText` 는 한글을 못 그린다. Pillow + `fonts-nanum` 필요 |
| `rqt_image_view` 가 멈추거나 회색만 나옴 | 뷰어 쪽 문제. 토픽을 바꿔가며 보다 생기니 창을 껐다 새로 띄울 것.<br>`ros2 topic hz` 로 데이터가 실제로 흐르는지 먼저 확인 |

---

## 속도 설정

`robot_control` 상단 상수로 조정한다. `movej` 는 관절 속도(deg/s),
`movel` 은 직선 속도(mm/s) 로 단위가 다르다.

```python
VELOCITY, ACC     = 81, 81      # 관절
L_VELOCITY, L_ACC = 150, 150    # 직선
```

물건을 집은 뒤 z 방향으로 300 mm 들어올린 다음 이송한다.
