#!/usr/bin/env bash
# 웹캠 발행(webcam) + 작업모드 수어 인식(sign) + 제어모드 손동작 조종(gesture) +
# 로봇 RealSense 시점 중계(realsense)를 한 터미널에서 한 번에 띄운다.
# 개별로도 켤 수 있다 — 인자로 골라서 넘긴다.
#
#   ./run_all.sh                 # 넷 다
#   ./run_all.sh webcam sign     # 웹캠 + 작업모드만
#   ./run_all.sh gesture         # 제어모드만 (webcam_publisher 가 이미 따로 떠 있다고 가정)
#   ./run_all.sh realsense       # RealSense 중계만
#
# Ctrl+C 한 번으로 이 스크립트가 띄운 것들을 전부(SIGINT로 곱게) 정리한다.
#
# 카메라 하나를 여러 프로세스가 같이 보려면 ROS 토픽 경유가 필요해서
# (SIGN_SOURCE=ros / GESTURE_SOURCE=ros), sign/gesture 를 webcam 없이 단독으로
# 켤 때는 webcam_publisher 가 어딘가에 이미 떠 있어야 한다.
# realsense 는 로봇 쪽에 실제 RealSense 가 있을 때만 의미가 있다 — 없으면 토픽을
# 구독한 채 조용히 아무 프레임도 못 받을 뿐, 에러로 죽지는 않는다.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIGN_CONTROL_DIR="$HERE/sign_control"
GESTURE_DIR="$HERE/hand_gesture_control"
WEBCAM_PUB="$HERE/voice_pickandplace/src/sign_processing/sign_processing/webcam_publisher.py"

if [ "$#" -eq 0 ]; then
    TARGETS=(webcam sign gesture realsense)
else
    TARGETS=("$@")
fi

want() {
    local t
    for t in "${TARGETS[@]}"; do
        [ "$t" == "$1" ] && return 0
    done
    return 1
}

PIDS=()
NAMES=()

cleanup() {
    echo
    echo "[run_all] 종료 중 — 띄운 프로세스에 SIGINT 전달..."
    local pid
    for pid in "${PIDS[@]}"; do
        kill -INT "$pid" 2>/dev/null
    done
    # 곱게 끝날 시간을 조금 준 뒤에도 살아있으면 강제 종료한다.
    for _ in 1 2 3 4 5; do
        local alive=0
        for pid in "${PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive=1
        done
        [ "$alive" -eq 0 ] && break
        sleep 0.5
    done
    for pid in "${PIDS[@]}"; do
        kill -KILL "$pid" 2>/dev/null
    done
}
trap cleanup EXIT INT TERM

launch() {
    local name="$1"; shift
    # 프로세스 치환(> >(...))으로 출력에 접두어를 붙이면서도, $! 은 원래 명령의
    # PID를 그대로 가리킨다 — 일반 파이프(|)로 연결하면 $! 이 파이프 끝(sed)의
    # PID가 되어 나중에 진짜 프로세스를 못 죽인다.
    "$@" > >(sed -u "s/^/[$name] /") 2>&1 &
    PIDS+=($!)
    NAMES+=("$name")
    echo "[run_all] $name 시작 (pid $!)"
}

if want webcam; then
    launch webcam env PYTHONUNBUFFERED=1 python3 -u "$WEBCAM_PUB"
fi

if want sign; then
    launch sign env PYTHONUNBUFFERED=1 SIGN_SOURCE=ros \
        python3 -u "$SIGN_CONTROL_DIR/sign_demo.py" run --llm --admin
fi

if want gesture; then
    launch gesture env PYTHONUNBUFFERED=1 GESTURE_SOURCE=ros \
        python3 -u "$GESTURE_DIR/hand_gesture_control.py" --admin
fi

if want realsense; then
    launch realsense env PYTHONUNBUFFERED=1 python3 -u "$GESTURE_DIR/realsense_bridge.py"
fi

if [ "${#PIDS[@]}" -eq 0 ]; then
    echo "알 수 없는 대상: ${TARGETS[*]}  (webcam / sign / gesture / realsense 중에서 고르세요)"
    exit 1
fi

echo "[run_all] 실행 중: ${NAMES[*]}  —  Ctrl+C 로 전부 종료"
wait
