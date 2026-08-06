# hand_gesture_control

웹캠으로 손을 인식해 로봇팔 xy/z/그리퍼를 조작하는 데모.

- xy: 손목 위치
- z: Thumb_Up(상승) / Thumb_Down(하강)
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
| Q | 종료 | 프로그램을 완전히 종료한다. |
| SPACE | 중심 재설정 | 지금 손이 있는 위치를 화면 정중앙 기준으로 다시 잡는다. 카메라 정중앙에 서지 않아도, 편한 위치에서 SPACE로 캘리브레이션할 수 있다. |
| [ | 게인 감소 | 손을 움직였을 때 화면상 포인터가 움직이는 폭을 줄인다 (더 정밀한 조작). 최소 0.1까지 내려간다. |
| ] | 게인 증가 | 손을 조금만 움직여도 포인터가 크게 움직이도록 민감도를 높인다. 최대 5.0까지 올라간다. |
| S | 시간평균 on/off | 최근 5프레임 평균으로 좌표를 부드럽게 할지(on), 매 프레임 원시값을 바로 쓸지(off) 전환한다. on일수록 떨림은 줄지만 반응이 살짝 늦어진다. |
| R | 전체 초기화 | 게인, 중심 오프셋, z, 그리퍼 상태를 모두 기본값으로 되돌린다. |
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
- 조종 화면(`screen` 창, xy 점/z 게이지/그리퍼 상태)을 `POST /api/frame/control`로 전송해서
  signbot_admin의 "제어 화면" 페이지에 그대로 띄운다. 모드가 바뀌면 대시보드 화면도 자동으로
  작업 화면 ↔ 제어 화면을 오간다.
- 카메라를 못 열면(다른 프로그램이 이미 잡고 있는 등) 제어권을 못 가져간 것으로 보고 즉시
  작업모드로 되돌린 뒤 1초 후 재시도한다.
