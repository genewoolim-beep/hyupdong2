# hand_gesture_control

웹캠으로 손을 인식해 로봇팔 xy/z/그리퍼를 조작하는 데모.

- xy(왼손, 검지 끝 기준): 조이스틱 방식. 화면 왼쪽 아래 원 두 개 — 안쪽 데드존
  안에서는 움직임 없음, 바깥으로 나가면 그 방향으로 계속 이동(속도 제어).
- z(오른손, 손바닥 기준): 화면 오른쪽 세로 게이지. xy 십자선과 같은 방식이다 —
  게이지를 위/가운데/아래로 3등분해서 가운데(1/3)면 정지, 위/아래면 그 방향으로
  계속 이동한다(절대 높이로 이동하지 않는다 — 제어모드 진입 시점마다 손·로봇
  위치가 달라 영점을 맞출 필요가 없도록). 가운데를 넓게 잡아 손떨림에도 높이를
  안정적으로 고정할 수 있다. 이동평균+데드밴드로도 손떨림을 추가로 억제한다.
- 그리퍼: Open_Palm(열기) / Closed_Fist(닫기)

## 실행

```bash
pip3 install --user mediapipe
python3 hand_gesture_control.py
```

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
| SPACE | 목표점 리셋 | 조이스틱으로 누적된 목표점을 화면 중앙으로 되돌린다. 한쪽 구석까지 밀고 간 뒤 되돌아올 때 유용하다. |
| [ | 감도 감소 | 조이스틱을 밀었을 때 목표점이 움직이는 속도를 줄인다 (더 정밀한 조작). 최소 0.1까지 내려간다. |
| ] | 감도 증가 | 같은 편차에도 목표점이 더 빨리 움직이도록 감도를 높인다. 최대 5.0까지 올라간다. |
| S | 시간평균 on/off | 최근 5프레임 평균으로 조이스틱 신호를 부드럽게 할지(on), 매 프레임 원시값을 바로 쓸지(off) 전환한다. on일수록 떨림은 줄지만 반응이 살짝 늦어진다. |
| R | 전체 초기화 | 감도, 목표점, z, 그리퍼 상태를 모두 기본값으로 되돌린다. |
| F | 전체화면 | screen(로봇 작업공간) 창을 전체화면으로 전환/해제한다. |

프로그램 실행 중에는 `screen` 창 좌상단에 위 키 목록이 항상 표시된다.

## 필요 파일

용량이 커서 저장소에 포함하지 않았다(`.gitignore`의 `*.task`). **스크립트와 같은 폴더**에 받는다.

```bash
cd hand_gesture_control
wget https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
```

## 작업모드 ↔ 제어모드 (signbot_admin 연동)

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
`hand_gesture_control.py`의 조종 화면은 **작은 화면**(보조)으로 배치된다 — 로봇을 조종할
땐 결국 로봇이 보는 화면이 더 중요하기 때문이다.
