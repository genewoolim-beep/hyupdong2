# 의존성 정리

`legacy/`는 뺐다 — README 대로 더 이상 쓰지 않는 옛 코드다.

## 1. 실행 환경

```
Ubuntu 22.04 / ROS2 Humble
Python 3.10
```

## 2. 하드웨어 (컨테이너에 넣을 수 없는 것)

| 하드웨어 | 누가 쓰는가 | 컨테이너 관점 |
|---|---|---|
| Doosan M0609 + OnRobot RG2 | `block_sort`, `hand_gesture_control` | TCP/모드버스로 네트워크 접근 필요 |
| Intel RealSense D435i | `object_detection`, `hand_gesture_control`(파지 AR) | USB 디바이스 패스스루 |
| 웹캠 | `sign_control`, `sign_processing`(웹캠 발행) | `/dev/video*` 패스스루, 한 프로세스만 열 수 있음 |
| 마이크 | `block_sort`(음성모드) | `/dev/snd` 패스스루 |

## 3. 외부 워크스페이스 — 이 저장소에 없음

`sign_pickandplace`와 `block_sort`는 별도 Doosan 드라이버 워크스페이스
(`dsr_msgs2`, `dsr_bringup2` 등 — 보통 `~/cobot_ws` 또는 `~/ws_cobot_pjt/ws_dsr`,
`run_all.sh`의 `DSR_WS`)가 있어야 `DR_init`/`DSR_ROBOT2` 임포트가 성공한다.
벤더가 배포하는 워크스페이스라 이 저장소에 포함하지 않는다 — 컨테이너에는
런타임에 볼륨으로 마운트한다.

## 4. apt / ROS 패키지

| 패키지 | 무엇에 쓰이는가 |
|---|---|
| `ros-humble-cv-bridge`, `ros-humble-vision-opencv` | ROS 이미지 ↔ OpenCV 변환 (`object_detection`, 웹캠 발행) |
| `ros-humble-sensor-msgs`, `ros-humble-std-srvs` | 토픽/서비스 메시지 타입 |
| `ros-humble-tf2-ros` | 좌표 변환 |
| `ros-humble-ament-index-python` | 패키지 리소스 경로 조회 |
| `ros-humble-rosidl-default-generators` | `od_msg` 커스텀 서비스 빌드 |
| `ros-humble-realsense2-camera` | RealSense 드라이버 노드 (`realsense2_camera_node`) |
| `portaudio19-dev`, `libasound2-dev` | `pyaudio`/`sounddevice` 빌드·런타임 (음성모드) |
| `python3-tk` | 교시 GUI (`teach_zones.py`, `teach_box.py`) |
| `libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender1` | OpenCV/MediaPipe 창 렌더링 |
| `v4l-utils`, `usbutils` | 웹캠/RealSense 장치 점검용 (`v4l2-ctl --list-devices` 등, README 트러블슈팅 참고) |

## 5. Python 패키지

`requirements.txt`(공통) + `signbot_admin/requirements.txt`(`Flask==3.0.3`)를
합친 것. 버전이 고정된 두 개는 이유가 있다 — `numpy==1.26.4`(2.x는 `cv_bridge`와
ABI 충돌로 임포트 직후 세그폴트), `mediapipe<1.0`(1.x가 numpy 2.x를 끌어옴).

| 패키지 | 용도 |
|---|---|
| `numpy==1.26.4` | 배열 연산 — 버전 고정 이유 위 참고 |
| `mediapipe<1.0` | 손동작 랜드마크 (`hand_gesture_control`) |
| `opencv-python` | 비전 공통 |
| `ultralytics` | YOLOv8 (도구 검출, `object_detection`) |
| `torch` | ultralytics·mediapipe 백엔드. CUDA 버전은 별도 설치 권장(레포 주석) |
| `pillow` | 한글 텍스트 렌더링 |
| `openwakeword` | 웨이크워드 검출 (`voice_command.py`) |
| `sounddevice`, `pyaudio` | 마이크 입출력 |
| `openai`, `langchain-openai`, `langchain-core` | 수어/음성 명령 해석 (LLM) |
| `python-dotenv` | `.env` 로드 |
| `pymodbus` | OnRobot RG2 그리퍼 제어 |
| `scipy` | 좌표/회전 변환, 오디오 파일 처리 |
| `Flask==3.0.3` | `signbot_admin` 대시보드 |

## 6. 컴포넌트별 요구사항

| 컴포넌트 | ROS 필요 | 로봇 드라이버 필요 | 하드웨어 |
|---|---|---|---|
| `sign_pickandplace`(`object_detection`/`sign_processing`) | O (colcon 빌드) | 아니오 | RealSense/웹캠 |
| `block_sort` | O(토픽/서비스 클라이언트) | O | 로봇, 그리퍼, RealSense, (음성모드는)마이크 |
| `hand_gesture_control` | O | `--robot` 옵션에서만 (보통 `block_sort` 안에서 돎) | 웹캠, RealSense |
| `signbot_admin` | 아니오 (Flask만) | 아니오 | 없음 |
| `sign_control` | 아니오 | 아니오 | 웹캠 |

## 7. API 키 / 비밀정보

`.env`는 `.gitignore`로 막혀 있다. `OPENAI_API_KEY`가 없거나 LLM 호출이
실패하면 규칙 기반 해석으로 자동 대체되므로(README) 필수는 아니지만 없으면
정확도가 떨어진다. 이미지에 굽지 않고 컨테이너 실행 시 `--env-file`로 넘긴다.

## 8. Docker

루트의 `Dockerfile`이 위 1·4·5·`sign_pickandplace` colcon 빌드까지 담당한다.
2·3·7은 컨테이너 밖에서 준비해 실행 시 마운트/환경변수로 넘긴다 — 자세한
빌드·실행 예시는 `Dockerfile` 상단 주석 참고.
