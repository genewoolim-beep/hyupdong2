# hyupdong2 개발/실행 컨테이너
#
# 이 이미지에 들어가는 것: ROS2 Humble + sign_pickandplace(od_msg·object_detection·
# sign_processing) colcon 빌드 + block_sort·hand_gesture_control·signbot_admin·
# sign_control 이 쓰는 파이썬 의존성.
#
# 이 이미지에 들어가지 않는 것 (README "공통 요구사항" 참고):
#   - Doosan 드라이버 워크스페이스(dsr_msgs2, dsr_bringup2 — DR_init/DSR_ROBOT2 의
#     출처). 벤더 워크스페이스라 이 저장소에 없다. 컨테이너 실행 시 볼륨으로
#     마운트한다 (아래 "실행" 참고). 없으면 block_sort.py 의 `import DR_init` 이
#     바로 실패한다 — 그 외 스크립트(scan/watch 등 로봇 없이 도는 것)는 영향 없다.
#   - 실제 하드웨어: Doosan M0609, OnRobot RG2, RealSense D435i, 웹캠, 마이크.
#     디바이스/네트워크를 컨테이너에 넘겨줘야 한다.
#   - API 키(.env, OPENAI_API_KEY 등). 이미지에 굽지 않는다 — 실행 시
#     --env-file 로 넘긴다.
#
# 빌드:
#   docker build -t hyupdong2 .
#   docker build -t hyupdong2 --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 .   # GPU torch
#
# 실행 예시 (block_sort 작업모드, 로봇·카메라·마이크 접근 포함):
#   docker run -it --rm \
#     --net=host --privileged \
#     -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -v ~/ws_cobot_pjt/ws_dsr:/dsr_ws:ro \
#     --env-file .env \
#     hyupdong2 bash

FROM ros:humble-ros-base

SHELL ["/bin/bash", "-c"]

# ── apt: ROS 패키지 + 시스템 패키지 ──────────────────────────────
# cv_bridge/tf2_ros/realsense2_camera 는 block_sort·hand_gesture_control·
# object_detection 이 직접 의존한다(requirements.txt 상단 주석 참고 — numpy/mediapipe
# 버전 고정도 cv_bridge 와의 ABI 충돌 때문). portaudio/asound 는 pyaudio·sounddevice
# (음성모드), python3-tk 는 block_sort/hand_gesture_control 의 교시 GUI(teach_zones.py,
# teach_box.py)가 쓴다.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl \
        python3-pip python3-colcon-common-extensions python3-rosdep \
        ros-humble-cv-bridge \
        ros-humble-vision-opencv \
        ros-humble-sensor-msgs \
        ros-humble-std-srvs \
        ros-humble-tf2-ros \
        ros-humble-ament-index-python \
        ros-humble-rosidl-default-generators \
        ros-humble-realsense2-camera \
        portaudio19-dev libasound2-dev \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        python3-tk v4l-utils usbutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── 파이썬 의존성 ────────────────────────────────────────────────
# torch 는 requirements.txt 에서 버전 미고정(주석: "CUDA 버전은 pytorch.org
# 안내대로 별도 설치 권장") — 기본은 CPU 휠, GPU 는 빌드 인자로 바꾼다.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements.txt
COPY signbot_admin/requirements.txt signbot_admin-requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir torch --index-url "$TORCH_INDEX_URL" \
    && python3 -m pip install --no-cache-dir -r requirements.txt -r signbot_admin-requirements.txt

# ── colcon 워크스페이스: od_msg · object_detection · sign_processing ─────
# README: "sign_pickandplace 는 정상 동작 플로우가 쓰는 ROS2 패키지만 들어 있다."
COPY sign_pickandplace/src sign_pickandplace/src
RUN source /opt/ros/humble/setup.bash \
    && cd sign_pickandplace \
    && colcon build --symlink-install --packages-select od_msg object_detection sign_processing

# ── 나머지 소스 (colcon 빌드 불필요 — 스크립트로 직접 실행) ──────────
COPY block_sort block_sort
COPY hand_gesture_control hand_gesture_control
COPY signbot_admin signbot_admin
COPY sign_control sign_control
COPY calib calib
COPY run_all.sh run_all.sh

# Doosan 드라이버 워크스페이스 마운트 지점 (이미지에는 없음 — 위 헤더 참고)
ENV DSR_WS=/dsr_ws

COPY docker/entrypoint.sh /entrypoint.sh
# 저장소에 .gitattributes 가 없어서, 이 이미지를 core.autocrlf=true 인 환경(주로
# Windows)에서 clone 한 소스로 빌드하면 .sh 가 CRLF 로 체크아웃될 수 있다 —
# 그러면 컨테이너 안에서 "#!/usr/bin/env bash\r" 로 셔뱅이 깨진다. 빌드하는
# 컴퓨터의 git 설정에 기대지 않도록 이미지 안에서 직접 LF 로 정규화한다.
RUN sed -i 's/\r$//' /entrypoint.sh run_all.sh && chmod +x /entrypoint.sh run_all.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
