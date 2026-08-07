# sign_pickandplace — ROS2 워크스페이스

정상 동작 플로우가 쓰는 ROS2 패키지만 있다. 이름이 `voice_pickandplace` 였던
이유는 이 저장소가 음성 pick & place 로 시작했기 때문이고, 지금 입력은
**수어와 손동작**이다 — 음성 인식은 쓰지 않는다(음성 코드는 `../legacy/`).

| 패키지 | 무엇 | 누가 쓰는가 |
|---|---|---|
| `object_detection` | 색으로 블록을 찾아 3차원 좌표를 주는 노드 (`/get_3d_position`) | `block_sort` 가 서비스로 부르고, 파지 AR(`grasp_overlay`)이 색 판정만 빌려 쓴다 |
| `od_msg` | 그 서비스 정의 (`SrvDepthPosition`) | `object_detection`, `block_sort` |
| `sign_processing` | 웹캠을 토픽으로 발행(`webcam_publisher`) + 수어 인식기(`gesture_recognizer`) | `block_sort`(수어), `hand_gesture_control`(손동작) 이 같은 토픽을 본다 |

`robot_control` 이 없는 것은 의도한 것이다. 로봇을 움직이는 코드는 ROS 패키지가
아니라 저장소의 `block_sort/` 스크립트다 — `colcon build` 없이 고쳐 돌릴 수 있는
편이 현장에서 빠르다.

## 빌드

```bash
source /opt/ros/humble/setup.bash
cd sign_pickandplace && colcon build --symlink-install
source install/setup.bash
```

`colcon build` 는 반드시 이 폴더 안에서 실행한다. 저장소 최상위에서 돌리면
최상위에 `build/ install/ log/` 가 또 생겨, 어느 것을 소싱했는지 헷갈린다.

> **폴더 이름을 바꾸면 반드시 다시 빌드해야 한다.** `install/` 에 절대경로가
> 박혀 있어(egg-link, setup.sh) 이름만 바꾸면 옛 경로를 가리킨 채 조용히
> 옛 코드를 쓴다. `build/ install/ log/` 를 지우고 새로 빌드한다.

## 자원 파일이 어디로 갔나

| 파일 | 지금 자리 | 왜 |
|---|---|---|
| `T_gripper2camera.npy` (핸드아이) | `../calib/` | `block_sort` 와 `hand_gesture_control` 이 **함께** 읽는다. 한 패키지 안에 두면 두 벌이 되고, 한쪽만 재보정하면 화면과 실제 파지가 갈라진다 |
| `.env` (LLM 키) | `../.env` | 음성이든 수어든 저장소 공용. `sign_command.py` 가 읽고, 환경변수도 먹는다 |
| `color_ranges.json` (색 범위) | `src/object_detection/resource/` | 검출 노드의 것이 유일한 원본이다. `block_sort` 도 이 파일을 읽는다 |
