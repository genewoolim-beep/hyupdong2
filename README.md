# 협동 2 — Doosan M0609 프로젝트 모음

Doosan M0609 협동로봇을 대상으로 한 프로젝트들. 각 폴더가 독립적으로 빌드·실행된다.

| 폴더 | 프로젝트 | 입력 | 상태 |
|---|---|---|---|
| [`voice_pickandplace/`](voice_pickandplace/) | 음성 기반 Pick & Place | 음성 | 동작 확인 |
| `sign_control/` + `sign_processing` | 수어 기반 로봇 제어 | 수어 | 동작 확인 (물체 인식만) |
| [`block_sort/`](block_sort/) | 색 블록 구역 분류 + 배치 복제 | 수어 / 텍스트 | 실주행 확인 (복제 4/4) |

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

색 블록을 집어 판 위의 구역 1~4 에 배치한다. 입력은 **수어** 또는 텍스트다.

```
수어 → GRU 분류 → 글로스 나열 → LLM → (색, 구역) 목록 → 검출 → 파지 → 배치
```

`voice_pickandplace` 의 검출 파이프라인(`object_detection` 의 `/get_3d_position`)과
`sign_control` 의 학습된 분류기를 그대로 재사용한다. ROS2 패키지가 아니라
스크립트라 `colcon build` 없이 바로 돌아간다 — 드라이버와 검출 노드만 떠 있으면 된다.

```bash
ros2 run object_detection object_detection     # 색 모델이 기본이다
python3 block_sort/teach_zones.py teach        # 구역 좌표 티칭 (최초 1회)
python3 block_sort/block_sort.py sign          # 수어로 명령
python3 block_sort/block_sort.py pick 빨간색 3  # 빨간 블록을 3번으로
python3 block_sort/block_sort.py pick 3 1      # 3번에 있는 것을 1번으로 (색 무관)
```

색 이름은 `object_detection` 의 `color_model.py`(`COLOR_HSV_RANGES`)에 등록된 것만
인식된다 — 빨강/주황/노랑/초록/파랑/보라, 한글 또는 영문.

### 수어 명령

`"빨강 → 3번구역"` 처럼 서명한다. **끝맺음 동사는 필요 없다.** 글로스가 하나 늘
때마다 지금까지 모은 것으로 명령이 되는지 LLM 에게 물어보고, 말이 되는 순간
그대로 실행한다. `놓다` 를 붙여도 되고 안 붙여도 된다.

LLM 은 구조적으로 가능할 때만(색+구역, 또는 구역 둘) 부른다. 아직 아무것도
안 모인 상태까지 매번 물어보면 동작 하나마다 네트워크를 타서 굼떠진다.

그래서 **한 문장에 명령을 여러 개 넣을 수는 없다** — 첫 명령이 완성되는 즉시
실행된다. 카메라는 계속 열려 있으니 이어서 다음 명령을 하면 된다.

**색을 말하지 않아도 된다.** `"3번구역 → 들다 → 1번구역 → 놓다"` 처럼 구역을 둘
말하면 앞 구역에 있는 블록을 색과 무관하게 집어 뒤 구역으로 옮긴다. 검출 서비스는
색을 지정해야 답하므로, 아는 색을 전부 물어보고 **구역 중심에 가장 가까운 것**을
고른다 (`detect_in_zone`). 이 단계는 팔이 안 움직여 여섯 번 물어도 싸고, 색이
정해진 뒤로는 평소 경로(광축 정렬 → 여러 번 재기 → 기울기 보정)를 그대로 탄다.
구역 중심에서 `ZONE_RADIUS`(45mm) 밖이면 그 구역은 비었다고 본다.

### 구역 세 가지

| 이름 | 저장 위치 | 로봇이 하는 일 |
|---|---|---|
| 로봇구역 1~4 | `zones.yaml` 의 `zones` | 블록을 **놓는다**. 자세(rx,ry,rz)까지 쓴다 |
| 인간구역 1~4 | `zones.yaml` 의 `human_zones` | **보기만** 한다. x,y 만 쓴다 |
| 프리구역 | 저장하지 않음 | 여기 블록만 **집는다** |

프리구역은 티칭하지 않는다 — **8개 구역을 뺀 나머지 전부**로 정의되기 때문이다.
어느 구역 중심에서도 `ZONE_RADIUS` 밖이면 프리다.

`ZONE_RADIUS = 45mm` 의 근거 (2026-08-04 실측 기하):
```
블록 반대각선 절반   35 × 1.414 / 2 = 24.7mm   45도 돌아가 있어도 덮는다
놓기 오차            약 10mm                   calib-zones 보정 후
검출 오차            약 10mm                   광축 정렬 수렴 후 잔차
─────────────────────────────────
합계                 약 45mm
```
더 크면 구역 사이 통로를 잡아먹고(70mm 면 12.5mm 만 남는다), 더 작으면 구역에
제대로 놓인 블록을 프리로 오판해 도로 집어간다. 구역 간격이 150mm 라 45mm 면
통로가 62.5mm 열린다 — 35mm 블록이 넉넉히 지난다. `ZONE_RADIUS` 로 조정한다.

### 순회 관측

넓게 한 번에 보는 관측 자세는 쓰지 않는다. 멀리서 한 장에 다 담으면 블록이 작게
잡혀 최소 면적(`min_area 800`)에 걸리고, 비스듬히 보여 중심이 밀린다. 대신 관심
영역 위를 **경유점 4개를 따라 낮게 호버링하며 한 바퀴 돈다.**

경유점마다 멈춰서 본다 — 움직이는 중에 찍으면 영상과 팔 자세가 어긋나 좌표가
통째로 밀린다(`SETTLE_SEC`). 필요한 색을 프리구역에서 보는 순간 순회를 멈추고
그 자리에서 집는다.

```bash
python3 teach_zones.py scan            # 순회 경유점 4개
python3 block_sort.py scan             # 한 바퀴 돌며 무엇이 있는지만 읽기
```

경유점 자세는 기울어도 된다 — 검출·파지 때 `level_att()` 가 수직으로 편다.

### 인간구역 배치 복제

```bash
python3 teach_zones.py teach human     # 인간구역 1~4 (칸 한가운데, 블록 안 물려도 됨)
python3 teach_zones.py scan            # 순회 경유점 4개
python3 block_sort.py copy             # 그대로 복제
python3 block_sort.py copy-mirror      # 좌우대칭 복제
```

`copy` 는 먼저 한 바퀴를 다 돈다 — 인간구역 4칸을 모두 읽어야 계획을 세울 수
있기 때문이다. 그 뒤로는 순회에서 봐 둔 좌표로 곧장 가서 집는다.

수어로는 **`똑같이`** / **`좌우대칭`** 한 동작이면 된다 — `sign_control` 에 이미
녹화돼 있던 두 단어다. 색도 구역도 필요 없어 그 하나로 문장이 끝난다.

'명령이 될 때까지' 로 판단하므로 **잘못 끼어든 글로스는 저절로 무시된다.**
실측에서 `조립하다` 가 88% 로 첫 글로스에 잘못 잡혔는데(2026-08-04), 그것만으로는
명령이 안 되니 그냥 지나가고 뒤이은 `2번구역 들다 4번구역` 이 명령이 된다.
중간의 `들다` 도 같은 이유로 넘어간다.

동작 하나로 인정하는 확신도는 `CONF_TH` **60%** 다(`SIGN_CONF` 로 조정). 0.80 은
지나가는 손짓을 잘 걸렀지만 제대로 한 동작도 자주 흘렸다 — 실측 확신도가
80~91% 에 몰려 있어 여유가 없었다. 낮춘 대가로 오인식이 늘지만, 명령이 될
때까지 계속 모으는 방식과 화면의 글로스 패널이 그것을 받아낸다. 잘못 들어가면
`C` 로 지운다.

화면은 `sign_demo.py run` 과 같다. 하단 패널에 모은 글로스가 쌓이고,
**명사+동사로 짝이 맞은 자리는 파랗게** 칠해진다. 수집 중에는 중간 결과가
`~단어` 로 같이 보인다. `Q` 취소 / `C` 지우기.

**카메라는 세션 내내 열려 있다.** 동작마다도, 문장마다도 닫지 않는다 — 여는 데만
1초가 넘어 그 사이 동작이 통째로 사라지고 창이 깜빡인다. 로봇이 움직이고 돌아오면
큐에 남은 낡은 장면 몇 장만 버리고 이어서 받는다. 정리는 `close()` 에서 한 번만 한다.

**동사 가중치** — 직전 글로스가 명사면 동사 확률에 `+0.05` 를 더한다
(`VERB_BONUS`, `sign_demo.py` 와 같은 값·같은 이유). 곱셈+재정규화로 하면 이미
확률이 몰린 1등이 거의 안 움직여 정작 필요한 경계 구간에서 효과가 없다.
색(`빨강`…)은 **수식**이라 가중 대상이 아니고, 구역·물체가 명사다.

글로스 나열을 (색, 구역) 목록으로 바꾸는 것은 LLM(`gpt-4o`)이다. 어순이 흐트러지거나
`나사`·`좌우대칭` 같은 관계없는 글로스가 섞여도 걸러낸다. **키가 없거나 호출이
실패하면 규칙 해석으로 자동 대체**되므로 네트워크가 끊겨도 시연은 멈추지 않는다.
`zones.yaml` 에 없는 구역(5·6번)은 어느 쪽이든 거부된다.

해석 로직은 로봇·카메라 없이 따로 확인할 수 있다. 프롬프트를 고칠 때마다
팔을 켜지 않아도 되게 분리해 둔 것이다.

```bash
python3 block_sort/sign_command.py selftest                    # 해석·수집 자체 검증
python3 block_sort/sign_command.py cam                         # 카메라만
python3 block_sort/sign_command.py parse "빨강 3번구역 놓다"   # 해석만
python3 block_sort/sign_command.py listen                      # 인식만
python3 block_sort/sign_command.py run                         # 인식 + 해석
```

`selftest` 는 아무 장비 없이 돌아가고 실패하면 종료코드 1 을 낸다. 팔을 켜기 전에
여기부터 통과시킨다.

| 환경변수 | 기본값 | 뜻 |
|---|---|---|
| `SIGN_CAM` | `1` | 수어 카메라 인덱스. 이 PC 는 0번이 가상 루프백이고 3~8번이 RealSense 라 실제 웹캠은 1번이다 |
| `SIGN_SHOW_WINDOW` | `1` | 인식 중 미리보기 창 |
| `SIGN_CONF` | `0.60` | 동작 하나로 인정할 최소 확신도 |
| `SIGN_LLM_MODEL` | `gpt-4o` | 해석에 쓸 모델 |
| `OPENAI_API_KEY` | — | 없으면 규칙 해석으로 동작한다 |

### 워크스페이스 소싱

`DSR_ROBOT2.py` 와 `dsr_msgs2` 는 **같은 워크스페이스 것**을 써야 한다. 섞으면
`NameError: SetSingularHandlingForce` 로 import 단계에서 죽는다 — 한쪽의
`DSR_ROBOT2.py` 가 다른 쪽 `dsr_msgs2` 에 없는 서비스를 부르기 때문이다.

```bash
source /opt/ros/humble/setup.bash
source ~/ws_cobot_pjt/ws_dsr/install/setup.bash        # 드라이버 (dsr_msgs2 와 짝이 맞는 쪽)
source ~/doosan-voice-pickandplace/voice_pickandplace/install/setup.bash   # od_msg, object_detection
```

### 보정

파지 정확도를 좌우하는 값들은 코드 상단 주석에 실측 근거와 함께 적혀 있다.
판이나 조명이 바뀌면 `center` → `calib-zones` 순으로 다시 잰다.

### signbot_admin 연동

`--admin` 을 뒤에 붙이면 `signbot_admin` 대시보드(기본 `http://localhost:5000`,
`SIGN_ADMIN_URL` 로 변경 가능)로 상태를 쏜다. 안 붙이면 지금까지와 동일하게 동작한다.

```bash
python3 block_sort.py scan --admin          # 순회 결과를 로봇/인간 구역 현황에 반영
python3 block_sort.py copy-mirror --admin   # 순회 + 배치 결과 모두 반영
python3 block_sort.py pick 빨강 3 --admin    # 옮긴 구역 하나만 반영
```

- 시작할 때 `POST /api/robot/status` 로 `connected: true` 를 보내고, 끝나면(성공이든
  실패든) `connected: false` 를 보낸다. 1회성 스크립트라 "연결됨"은 지금 이 순간
  명령을 실행 중이라는 뜻이다.
- `scan`/`read-human`/`read-free`, `copy`/`copy-mirror`처럼 **전체 순회**를 하는 명령만
  구역 전체를 갱신한다. `pick` 처럼 부분 순회(`patrol(want=...)`)를 쓰는 경로는 옮긴
  구역 하나만 콕 집어 갱신한다 — 못 본 구역까지 비었다고 잘못 반영하지 않기 위해서다.
- 실행 중 예외가 나면 `POST /api/debug` 로 오류 메시지를 남긴다.

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
