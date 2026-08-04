# 협동 2 — Doosan M0609 프로젝트 모음

Doosan M0609 협동로봇을 대상으로 한 프로젝트들. 각 폴더가 독립적으로 빌드·실행된다.

| 폴더 | 프로젝트 | 입력 | 상태 |
|---|---|---|---|
| [`voice_pickandplace/`](voice_pickandplace/) | 음성 기반 Pick & Place | 음성 | 동작 확인 |
| `sign_control/` + `sign_processing` | 수어 기반 로봇 제어 | 수어 | 동작 확인 (물체 인식만, 어휘는 재정비 필요) |
| `block_sort/` | 색깔 블록 → 구역 배치 | 텍스트 | 코드 준비 완료, 실기 미검증 |

---

## voice_pickandplace

음성 명령("hello rokey" → "사과 가져와")으로 로봇암이 대상을 찾아 집어 옮긴다.

```
음성 → 웨이크워드 → STT → LLM 키워드 추출 → YOLO 검출 → 좌표 변환 → 로봇 동작
```

`voice_processing` / `object_detection` / `robot_control` 세 개의 독립 ROS2 패키지로
구성되어 있어 각각 따로 실행하거나 컨테이너화할 수 있다.

자세한 내용은 [voice_pickandplace/README.md](voice_pickandplace/README.md) 참고.

> `object_detection`은 두 검출기(`YoloModel`/`ColorModel`)를 다 갖고 있고
> `ObjectDetectionNode(model_name=...)`로 고른다. 현재 **기본값은 `color`**다
> (아래 `block_sort` 참고). `yolo`로 쓰려면 `model_name='yolo'`를 명시해야
> 하고, 그 안에서도 `yolo.py`의 `YOLO_MODEL_FILENAME`이 지금 도구 모델
> (`yolov8n_tools_0122.pt`)로 맞춰져 있어 위 예시("사과 가져와")는 바로
> 안 된다 — `fruit_best.pt`/`fruit_class_name.json`으로 되돌려야 한다.

---

## sign_control + sign_processing

음성 대신 **수어**로 같은 로봇을 제어한다. `sign_processing` 이
`voice_processing` 의 `/get_keyword` 서비스(`std_srvs/Trigger`)를 그대로
대체하므로 `robot_control` 은 코드 변경 없이 그대로 쓴다.

```
수어 → MediaPipe 랜드마크 → GRU 분류 → 글로스 → 물체명 매핑 → robot_control
```

LLM은 쓰지 않는다 — 분류기가 이미 이산 단어를 출력하므로 문장 파싱이 필요 없다.
목적지(구역) 인식은 아직 연동하지 않았다 — 이번 범위는 물체 인식만이다.

`sign_processing/get_keyword.py` 의 `GLOSS_TO_OBJECT` 매핑은 지금
`망치 -> hammer` 하나뿐이다. `object_detection` 도구 모델의 나머지 클래스
(drill/pliers/screwdriver/wrench)에 대한 수어를 `sign_control` 에서
녹화·재학습하면 매핑을 한 줄씩 늘리면 된다.

> 오늘 실제 목표는 도구가 아니라 **색깔 블록**(아래 `block_sort` 참고)이라,
> `GLOSS_TO_OBJECT`가 지금 가리키는 어휘와 실제 과제가 어긋나 있다.
> `sign_control`엔 이미 색깔 6종·1~6번구역·똑같이·좌우대칭 수어가
> 녹화되어 있으니, 다음 단계는 `GLOSS_TO_OBJECT`를 색깔/구역 쪽으로
> 확장해 `block_sort`와 잇는 것이다. 목적지(구역) 인식도 아직 안 붙어 있다.

`sign_control/` 자체(녹화/학습/단독 데모, [sign_control/README.md](sign_control/README.md))는
그대로 독립 실행되고, `voice_pickandplace/src/sign_processing/` 은 그 학습된
모델을 ROS2 서비스로 감싸는 얇은 레이어다. 모델(`model.pt`)과 mediapipe
`.task` 파일은 복사하지 않고 `sign_control/` 을 그대로 참조한다
(`SIGN_CONTROL_DIR` 환경변수로 경로 재정의 가능).

```bash
colcon build --symlink-install --packages-select od_msg object_detection sign_processing robot_control
source install/setup.bash
SIGN_CAM=0 ros2 run sign_processing get_keyword   # voice_processing 대신
```

카메라 인덱스는 `SIGN_CAM`, 인식 중 카메라 미리보기 창은 `SIGN_SHOW_WINDOW=0`
으로 끌 수 있다.

---

## block_sort

색깔 블록을 집어서 지정한 구역에 놓는다. `voice_pickandplace`의 로봇 제어
파이프라인을 그대로 쓰되, 놓는 자리를 `BUCKET_POS` 한 곳 대신 구역 1~4로
나누고 입력을 텍스트 인자로 받는 버전이다. ROS2 패키지가 아니라
[block_sort/block_sort.py](block_sort/block_sort.py) 스크립트 하나로 되어 있다.

```
관측 → (ColorModel로) 검출 → 파지 → zones.yaml 구역에 배치
```

```bash
# 터미널 1 — object_detection (기본값이 color 라 그대로 실행하면 됨)
ros2 run object_detection object_detection

# 터미널 2 — Doosan 드라이버 워크스페이스 source 먼저
cd block_sort
python3 block_sort.py observe 빨간색   # 검출만 먼저 확인 (로봇은 관측 자세로만 감)
python3 block_sort.py pick 빨간색 1    # 검출 → 파지 → 1번 구역 배치
python3 block_sort.py run              # "<색깔> <구역번호>" 반복 입력
```

색 이름은 `object_detection`의 `color_model.py`(`COLOR_HSV_RANGES`)에 등록된
것만 인식된다 — 빨강/주황/노랑/초록/파랑/보라(한글 또는 영문). 구역 좌표
(`zones.yaml`)와 파지 높이·그리퍼 중심 보정(`center_calib.yaml`)은 이미
실측값으로 채워져 있다.

> **상태**: 코드 리뷰로는 준비된 것으로 보인다 — 색 검출이 다중 프레임
> 합의로 안정화됐고 구역 티칭·파지 보정도 끝나 있다. 하지만 이 기기에는
> Doosan 드라이버 워크스페이스(`~/cobot_ws`)가 비어 있어(`import DR_init`
> 부터 실패) **실제 로봇으로 끝까지 검증하지 못했다.** 드라이버가 설치된
> PC에서 위 명령을 이어서 확인해봐야 한다.

---

## 공통 요구사항

```
Ubuntu 22.04 / ROS2 Humble
Python 3.10
Doosan M0609 + OnRobot RG2      # voice_pickandplace, block_sort 해당
Intel RealSense D435i           # voice_pickandplace, block_sort 해당
웹캠                             # sign_control
```

```bash
pip install -r requirements.txt
```

`numpy==1.26.4` 와 `mediapipe<1.0` 이 고정되어 있다. 이 둘을 놓치면
`cv_bridge` 와 ABI 가 충돌해 import 직후 세그폴트가 난다.

`voice_pickandplace` 와 `block_sort` 는 별도로 Doosan 드라이버 워크스페이스
(`dsr_msgs2`, `dsr_bringup2` 등, 보통 `~/cobot_ws`)와
`ros-humble-realsense2-camera` 가 필요하다. 이 드라이버 워크스페이스가
비어 있으면 `import DR_init` 단계에서 바로 실패한다.
`sign_control` 은 웹캠만 있으면 된다.

## API 키

이 저장소는 **공개 저장소**다. `.env` 는 `.gitignore` 로 차단되어 있으며
어떤 경우에도 커밋해서는 안 된다. 각 패키지의 `.env.example` 을 복사해 사용한다.

```bash
cp <패키지>/resource/.env.example <패키지>/resource/.env
```
