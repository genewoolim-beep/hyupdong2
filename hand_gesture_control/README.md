# hand_gesture_control

웹캠으로 손을 인식해 로봇팔 xy/z/그리퍼를 조작한다.

**세 축 전부 같은 방식이다 — 방향만 보고 일정 속도.** 얼마나 벗어났는지는 속도를
안 바꾼다(속도가 변하지 않아야 예측 가능하다). 절대 위치로 매핑하지도 않는다 —
제어모드에 들어갈 때마다 손과 로봇의 시작 위치가 달라, 절대 매핑이면 진입 즉시
그 차이만큼 팔이 튄다. 방향만 보면 로봇은 늘 지금 있는 자리에서 출발하므로
영점 조절이 아예 필요 없다.

- xy(왼손, 검지 끝 기준): 화면 왼쪽 아래 **십자선**. 중앙 사각(데드존)이면 정지,
  어느 쪽으로 벗어나든 그 축으로 20mm/s.
  좌우는 **조종하는 사람이 어디 서 있는지**에 달렸다. 기본값은 `로봇을 마주보고`
  조종하는 것이라(`TELEOP_FACE_ROBOT=1`) 내 오른쪽 = 로봇의 왼쪽(base +y) 이다.
  로봇과 같은 방향을 보고 조종하면 `TELEOP_FACE_ROBOT=0`. 십자선에 찍히는 축
  글자(`R +Y` / `L -Y`)는 **실제로 나가는 부호에서 뽑으므로** 설정과 어긋날 수
  없다 — L/R 은 화면이 거울이라 늘 조작자 기준이다. 앞뒤(x)는 이 설정과 무관하다.
- z(오른손, 손바닥 기준): 화면 오른쪽 세로 게이지를 위/가운데/아래로 **3등분**.
  가운데(1/3)면 정지, 위/아래면 그 방향으로 20mm/s. 가운데를 xy 데드존(0.08)보다
  훨씬 넓게 잡은 이유는 높이를 **유지**하려는 조작인데 손떨림으로 살짝만 벗어나도
  움직여버리면 고정이 안 되기 때문이다. 이동평균+데드밴드로도 떨림을 더 억제한다.
- 그리퍼: Open_Palm(열기) / Closed_Fist(닫기)

## 로봇은 여기서 움직이지 않는다 (2026-08-06 이후)

**실제 조종은 `block_sort` 프로세스 안에서 돈다.** 수어로 `모드변경` 을 서명하면
`block_sort/teleop_mode.py` 가 이 폴더의 인식 로직(`HandController`)과 속도 지령
(`robot_teleop.py`)을 그대로 불러 쓴다.

왜 옮겼나 — 전에는 이 스크립트가 따로 DSR 에 붙었다. 한 로봇에 연결이 둘이 되어
TCP 조회가 0.0mm 로 풀리고, 모션이 전부 거부되고, `get_current_posx()` 가 응답
없이 멎었다(실측 2026-08-06). 연결·그리퍼·웹캠을 하나로 공유하면 셋이 함께 사라진다.

그래서 이 스크립트를 직접 띄우는 것은 **인식 확인용**이다. 작업모드와 함께 쓸 때
`--robot` 을 붙이지 말 것 — 붙이면 DSR 연결이 둘로 돌아간다.

## 실행

```bash
pip3 install --user mediapipe
python3 hand_gesture_control.py            # 인식만 (로봇 안 붙음)
```

`--robot` 은 작업모드를 안 쓰는 PC(로봇 + 웹캠만) 전용이다.

경로는 전부 스크립트 위치 기준이라 clone 후 바로 실행된다. 다른 곳에 두었다면
환경변수로 덮어쓴다.

| 변수 | 기본값 | 용도 |
|---|---|---|
| `GESTURE_CAM` | `1` | 카메라 인덱스 (`GESTURE_SOURCE=v4l2`일 때만). `/dev/video0`면 `0`으로. `ls /dev/video*` 로 확인 |
| `GESTURE_TASK` | 스크립트와 같은 폴더 | `gesture_recognizer.task` 위치 |
| `GESTURE_SOURCE` | `v4l2` | 프레임 출처. `v4l2`(장치 직접 열기) 또는 `ros`(토픽 구독) |
| `GESTURE_CAM_TOPIC` | `/webcam/image_raw` | `GESTURE_SOURCE=ros`일 때 구독할 토픽 |

```bash
GESTURE_CAM=0 python3 hand_gesture_control.py
```

### sign_demo.py와 동시에 쓰기 (카메라 공유)

V4L2 웹캠은 프로세스 하나만 열 수 있다. `sign_demo.py`가 이미 카메라를 잡고 있으면
이 스크립트는 `카메라 열기 실패`로 죽는다. 동시에 쓰려면 `webcam_publisher.py`가
장치를 혼자 열고 ROS 토픽으로 뿌리게 하고, 양쪽 다 그 토픽을 구독한다:

```bash
# 터미널 1 — 웹캠을 한 번만 열어 토픽으로 발행
ros2 run sign_processing webcam_publisher

# 터미널 2
cd hyupdong2/sign_control
SIGN_SOURCE=ros python3 sign_demo.py run --llm --admin

# 터미널 3
GESTURE_SOURCE=ros python3 hand_gesture_control.py --admin
```

토픽 이름(`/webcam/image_raw`)이 같아야 서로 다른 프로그램이 같은 화면을 본다 —
기본값을 그대로 두면 맞는다.

## 키

| 키 | 기능 | 설명 |
|---|---|---|
| Q | 종료 | 조종 세션을 끝낸다. `--admin`이면 작업모드로 신호를 보내고 대기 상태로 돌아간다(프로세스는 안 죽음). 아니면 프로그램 자체가 끝난다. |
| SPACE | 목표점 리셋 | `screen` 창의 누적 점을 화면 중앙으로 되돌린다. |
| [ / ] | 감도 감소/증가 | 그 누적 점이 움직이는 속도. 0.1~5.0. |
| S | 시간평균 on/off | 최근 5프레임 평균으로 신호를 부드럽게 할지 전환한다. |
| R | 전체 초기화 | 감도, 누적 점, z, 그리퍼 상태를 기본값으로 되돌린다. |
| F | 전체화면 | screen 창을 전체화면으로 전환/해제한다. |

> SPACE·`[`·`]`·S 는 이제 **`screen` 창의 누적 점에만** 영향을 준다. 로봇은
> 누적 점이 아니라 **지금 손이 십자선에서 벗어난 방향**으로 움직인다 — 누적값을
> 쓰면 손을 중앙으로 되돌려도 그 값이 남아 팔이 계속 기어갔다(실측 2026-08-06).
> `block_sort` 안의 제어모드 화면(`teleop_mode.py`)에는 `screen` 창이 없다.

프로그램 실행 중에는 `screen` 창 좌상단에 위 키 목록이 항상 표시된다.

## 필요 파일

용량이 커서 저장소에 포함하지 않았다(`.gitignore`의 `*.task`). **스크립트와 같은 폴더**에 받는다.

```bash
cd hand_gesture_control
wget https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
```

## 작업영역 교시 — teach_box.py

조종 경계를 로봇을 손으로 옮겨 잡는다. 로봇을 **수동 모드**로 두고:

```bash
python3 teach_box.py            # 두 모서리를 찍어 teleop_box.env 로 저장
python3 teach_box.py now        # 지금 위치만 보기
```

`robot_teleop.py` 가 그 파일을 직접 읽으므로, 저장하면 **다음 제어모드 진입부터
그대로 적용된다** — 환경변수로 다시 적어줄 필요가 없다(두 곳에 적으면 갈라지고,
갈라지면 팔이 판에 부딪히는 쪽이 된다). 그때만 다르게 쓰려면 `TELEOP_X_MIN` 등을
환경변수로 주면 파일보다 우선한다.

경계는 **밖으로 나가는 방향의 속도를 0** 으로 만드는 방식이다. 위치를 자르는 게
아니라 속도를 막는 것이라 실제 팔에 직접 걸린다. 정지에도 거리가 필요해 경계
30mm 안쪽(`TELEOP_MARGIN` 15 + `TELEOP_BRAKE` 15)에서 미리 끊는다.
**위치를 못 읽으면 아예 안 움직인다.**

## 작업모드 ↔ 제어모드 (signbot_admin 연동)

> 아래는 조종이 별 프로세스였을 때의 대기·인계 흐름이다. 지금은 `block_sort` 가
> 제어모드를 직접 돌기 때문에 **이 인계는 쓰이지 않는다.** 작업모드를 안 쓰는
> PC 에서 이 스크립트만으로 대시보드와 맞춰 보고 싶을 때 남겨둔 경로다.

`--admin`을 붙이면 로봇이 두 모드를 오가는 것을 signbot_admin 대시보드와 함께 확인할 수 있다.

```bash
GESTURE_SOURCE=ros python3 hand_gesture_control.py --admin
```

- 시작하면 카메라를 열지 않고 **대기 상태**로 들어간다. `signbot_admin`의 `/api/mode`를 0.5초마다
  확인하다가, `sign_demo.py` 쪽에서 "박수 - 모드변경 - 박수"로 제어모드가 되는 순간 자동으로
  카메라를 잡고 조종 창을 띄운다.
- 양손을 동시에 3초간 펼치면(`Open_Palm`) 작업모드로 신호를 보내고 창을 닫은 뒤 다시 대기 상태로
  돌아간다 — **프로세스는 종료되지 않는다.** `Q`를 눌러 수동으로 끝내도 마찬가지다.
- 완전히 종료하려면 `Ctrl+C`.
- `camera` 창(실제 카메라 + 조이스틱 원 + 손 스켈레톤 + z 게이지가 다 그려진 화면)을
  `POST /api/frame/control`로 전송해서 signbot_admin "제어 화면" 페이지의 작은 패널에
  그대로 띄운다 — 원격에서도 지금 손이 어디 있는지 보이게 하려고 `screen`(추상 표시) 대신
  이쪽을 보낸다. 모드가 바뀌면 대시보드 화면도 자동으로 작업 화면 ↔ 제어 화면을 오간다.
- 카메라를 못 열면(다른 프로그램이 이미 잡고 있는 등) 제어권을 못 가져간 것으로 보고 즉시
  작업모드로 되돌린 뒤 1초 후 재시도한다.

## 로봇 시점(RealSense) 같이 보기

`hand_gesture_control.py`의 화면은 **조종자가 손을 어떻게 움직이고 있는지**만 보여준다.
실제로 로봇을 조종하려면 **로봇이 보는 화면**(RealSense)도 봐야 한다. `realsense_bridge.py`가
로봇 쪽 RealSense 컬러 토픽을 구독해 signbot_admin으로 중계한다 — 로컬 창은 안 띄운다.

```bash
python3 realsense_bridge.py
```

| 변수 | 기본값 | 용도 |
|---|---|---|
| `REALSENSE_TOPIC` | `/camera/camera/color/image_raw` | 구독할 RealSense 컬러 토픽 |
| `REALSENSE_JPEG` | `70` | JPEG 품질 |
| `REALSENSE_FPS` | `12` | signbot_admin 전송 상한 fps (모니터링용이라 그 이상 불필요) |

signbot_admin "제어 화면" 페이지에서는 RealSense가 **큰 화면**(로봇 시점, 주 화면)으로,
조종 화면은 **작은 화면**(보조)으로 배치된다 — 로봇을 조종할 땐 결국 로봇이 보는
화면이 더 중요하기 때문이다. 작은 화면으로 가는 프레임은 이제
`block_sort`(제어모드)가 `POST /api/frame/control` 로 보낸다.
