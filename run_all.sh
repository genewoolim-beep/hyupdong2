#!/usr/bin/env bash
# 이 저장소의 노드들을 한 터미널에서 **순서대로** 띄운다.
#
#   ./run_all.sh                    # 한 벌 전부 (realsense detect webcam admin gui bridge sign)
#   ./run_all.sh realsense detect   # 골라서
#   ./run_all.sh --list             # 타깃 설명만 보고 끝
#
# Ctrl+C 한 번으로 **이 스크립트가 띄운 것만** 정리한다. 로봇 드라이버는 안 건드린다.
#
# ── 왜 순서와 대기가 필요한가 ─────────────────────────────────────
# detect 는 RealSense 프레임을 받아야 초기화가 끝나고, sign 은 webcam 토픽을
# 구독하며, sign/bridge 는 admin 에 상태를 올린다. 동시에 띄우면 뒤엣것이
# 앞엣것을 못 찾는다. 그래서 단계마다 준비될 때까지 실제로 기다린다
# (sleep 으로 때우면 느린 날 실패한다).
#
# ── 로봇 드라이버는 왜 여기 없나 ──────────────────────────────────
# 실기 연결이라 자동으로 띄우지 않는다. 이미 떠 있는데 또 띄우면 컨트롤러에
# 둘이 붙어 위험하다. 먼저 직접 띄워 두고 이 스크립트를 쓴다:
#   ros2 launch m0609_rg2_bringup bringup.launch.py mode:=real host:=192.168.1.100 model:=m0609
#
# ── 모드 전환 ─────────────────────────────────────────────────────
# 제어모드(손동작 조종)는 이제 **sign 프로세스 안에서** 돈다. 따로 띄울 타깃이
# 없다 — '모드변경' 을 서명하면 그 안에서 조종 창이 열린다.
# 한 로봇에 DSR 연결을 하나만 두기 위한 것이다(block_sort/teleop_mode.py).
# 제어모드에서 작업모드로 돌아오는 방법은 셋이다:
#   ① 양손 Open_Palm 3초   ② 조종창에서 Q   ③ 대시보드에서 작업모드로 전환
#
# hand_gesture_control.py 를 따로 띄우지 말 것. --robot 까지 붙이면 DSR 연결이
# 둘이 되어 예전 증상(TCP 0.0mm, 모션 거부)이 그대로 돌아온다.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLOCK_DIR="$HERE/block_sort"
GESTURE_DIR="$HERE/hand_gesture_control"
ADMIN_DIR="$HERE/signbot_admin"
WS="$HERE/voice_pickandplace"
DSR_WS="${DSR_WS:-$HOME/ws_cobot_pjt/ws_dsr}"

ADMIN_URL="${SIGN_ADMIN_URL:-http://localhost:5000}"

# 카메라 하나를 여러 프로세스가 같이 보려면 ROS 토픽을 거쳐야 한다.
# V4L2 장치는 한 프로세스만 열 수 있기 때문이다.
export SIGN_SOURCE="${SIGN_SOURCE:-ros}"
export GESTURE_SOURCE="${GESTURE_SOURCE:-ros}"
export DISPLAY="${DISPLAY:-:1}"

ALL_TARGETS=(realsense detect webcam admin gui bridge sign)
DEFAULT_TARGETS=("${ALL_TARGETS[@]}")

if [[ "${1:-}" == "--list" ]]; then
    printf '%s\n' \
        "  realsense  RealSense 카메라 (검출·로봇시점의 원천)" \
        "  detect     색 검출 노드              ← realsense 필요" \
        "  webcam     웹캠 → ROS 토픽 발행      (수어·손동작의 원천)" \
        "  admin      signbot_admin 대시보드    :5000" \
        "  gui        브라우저로 대시보드 열기   ← admin 필요" \
        "  sign       작업모드 + 제어모드 ('모드변경' 서명 시 조종) [로봇 움직임]" \
        "             ← webcam detect admin" \
        "  bridge     로봇 시점을 제어 화면으로 중계            ← realsense admin"
    exit 0
fi

if [[ $# -eq 0 || "${1:-}" == "all" ]]; then
    TARGETS=("${DEFAULT_TARGETS[@]}")
else
    TARGETS=("$@")
fi
# gesture 는 sign 안으로 들어갔다. 옛 명령줄을 그대로 쓰는 사람이 조용히
# 아무것도 못 얻는 것보다, 왜 없어졌는지 알려주는 편이 낫다.
for t in "${TARGETS[@]}"; do
    if [[ "$t" == "gesture" ]]; then
        echo "  ! gesture 타깃은 없어졌습니다 — 제어모드가 sign 안으로 들어갔습니다."
        echo "    sign 을 띄운 뒤 '모드변경' 을 서명하면 조종 창이 열립니다."
        exit 1
    fi
done

want() { local t; for t in "${TARGETS[@]}"; do [[ "$t" == "$1" ]] && return 0; done; return 1; }

# ── 소싱 ──
# ros2_ws 를 소싱하면 dsr_msgs2 버전이 안 맞아 SetSingularHandlingForce 로 죽는다.
# 반드시 DSR_WS 것을 쓴다.
# ROS 의 setup.bash 는 미정의 변수를 참조한다. set -u 가 켜진 채로 소싱하면
# "AMENT_TRACE_SETUP_FILES: 바인딩 해제한 변수" 로 죽는다. 이 구간만 끈다.
set +u
source /opt/ros/humble/setup.bash
[[ -f "$DSR_WS/install/setup.bash" ]] && source "$DSR_WS/install/setup.bash"
[[ -f "$WS/install/setup.bash" ]] && source "$WS/install/setup.bash"
set -u

PIDS=()
NAMES=()
cleanup() {
    trap - INT TERM EXIT
    echo
    echo "[run_all] 정리 중..."
    # 역순으로 내린다 — 의존하는 쪽을 먼저 끊어야 깔끔하다.
    local i
    for ((i=${#PIDS[@]}-1; i>=0; i--)); do
        kill -INT "${PIDS[$i]}" 2>/dev/null && echo "  ${NAMES[$i]} 중지"
    done
    sleep 3
    for ((i=${#PIDS[@]}-1; i>=0; i--)); do kill -TERM "${PIDS[$i]}" 2>/dev/null; done
    sleep 1
    for ((i=${#PIDS[@]}-1; i>=0; i--)); do kill -9 "${PIDS[$i]}" 2>/dev/null; done
    echo "[run_all] 완료. 로봇 드라이버는 그대로 둡니다."
}
trap cleanup INT TERM EXIT

launch() {   # launch <이름> <명령...>
    #
    # 파이프(`| sed`)로 접두어를 붙이면 $! 가 파이프라인의 **마지막**(sed) PID 를
    # 잡는다. 그러면 실제 프로세스가 죽어도 알 수 없고, 정리할 때도 sed 만
    # 죽여 노드가 살아남는다 — 다음 실행이 V4L2 장치를 못 열어 무너진다.
    # 프로세스 치환을 쓰면 $! 가 명령 자신의 PID 다.
    local name="$1"; shift
    echo "▶ $name"
    "$@" > >(sed "s/^/[$name] /") 2>&1 &
    local pid=$!
    sleep 0.5
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "  ! $name 이 곧바로 종료됐습니다"
        return 1
    fi
    PIDS+=("$pid")
    NAMES+=("$name")
}

wait_topic() {   # wait_topic <토픽> [초]
    local topic="$1" limit="${2:-30}" n=0
    while ! ros2 topic list 2>/dev/null | grep -qx "$topic"; do
        sleep 1; n=$((n+1))
        if (( n >= limit )); then echo "  ! $topic 이 ${limit}초 안에 안 올라왔습니다"; return 1; fi
    done
    echo "  ✓ $topic"
}

wait_http() {    # wait_http <url> [초]
    local url="$1" limit="${2:-30}" n=0
    while ! curl -s -o /dev/null --max-time 2 "$url"; do
        sleep 1; n=$((n+1))
        if (( n >= limit )); then echo "  ! $url 이 ${limit}초 안에 안 떴습니다"; return 1; fi
    done
    echo "  ✓ $url"
}

# ── 시작 전 점검 ──
# 이전 실행이 남아 있으면 새로 뜬 것이 죽는다. 특히 webcam_publisher 는 V4L2
# 장치를 독점하므로, 남아 있으면 새 것이 "프레임을 못 받았습니다" 로 종료되고
# 그 뒤 sign 이 프레임을 못 받아 줄줄이 무너진다.
# 실측 2026-08-06: 이것 때문에 제어모드가 켜지자마자 작업모드로 튕겼다.
STALE=$(pgrep -f "lib/sign_processing/webcam_publisher|lib/object_detection/object_detection|realsense2_camera_node|realsense_bridge.py|hand_gesture_control.py|block_sort.py sign|signbot_admin/app.py" 2>/dev/null | tr '\n' ' ')
if [[ -n "${STALE// /}" ]]; then
    echo "  ! 이전 실행이 남아 있습니다: $STALE"
    echo "    정리하고 다시 실행하세요:"
    echo "      pkill -INT -f 'block_sort.py sign'; sleep 3"
    echo "      pkill -TERM -f 'webcam_publisher|object_detection|realsense|hand_gesture|signbot_admin/app.py'"
    exit 1
fi

echo "타깃: ${TARGETS[*]}"
if ! pgrep -f ros2_control_node >/dev/null 2>&1; then
    echo "  ! 로봇 드라이버가 안 보입니다 — sign 이 로봇을 못 움직입니다."
    echo "    ros2 launch m0609_rg2_bringup bringup.launch.py mode:=real host:=192.168.1.100 model:=m0609"
fi
echo

if want realsense; then
    launch realsense ros2 launch realsense2_camera rs_launch.py \
        align_depth.enable:=true \
        depth_module.depth_profile:=848x480x30 \
        rgb_camera.color_profile:=1280x720x30
    wait_topic /camera/camera/color/image_raw 40
fi

if want webcam; then
    launch webcam ros2 run sign_processing webcam_publisher
    # sign 이 이 토픽에 전적으로 의존한다 — 수어도 손동작도 여기서 온다.
    # 없는데 진행하면 프레임을 못 받아 조용히 헛돈다. 여기서 멈춘다.
    if ! wait_topic /webcam/image_raw/compressed 20; then
        echo "  ! 웹캠 발행이 안 됩니다 — sign 이 동작할 수 없어 중단합니다."
        echo "    다른 프로세스가 장치를 잡고 있는지 확인하세요:"
        echo "      v4l2-ctl --list-devices"
        exit 1
    fi
fi

if want admin; then
    if ! launch admin env PYTHONUNBUFFERED=1 python3 -u "$ADMIN_DIR/app.py"; then
        echo "  ! 대시보드를 못 띄웠습니다 — :5000 을 다른 프로세스가 쓰고 있는지"
        echo "    확인하세요:  pkill -f 'signbot_admin/app.py'"
        exit 1
    fi
    wait_http "$ADMIN_URL/" 20
fi

# 대시보드가 응답한 뒤에 연다. 먼저 열면 빈 페이지가 뜬다.
if want gui; then
    if command -v google-chrome >/dev/null 2>&1; then
        echo "▶ gui"
        nohup google-chrome --new-window "$ADMIN_URL" >/dev/null 2>&1 &
        # 브라우저는 PIDS 에 안 넣는다 — Ctrl+C 로 창까지 닫히면 불편하다.
        echo "  ✓ 브라우저 창 (Ctrl+C 로 닫히지 않음)"
    else
        echo "  ! google-chrome 이 없습니다 — $ADMIN_URL 을 직접 여세요"
    fi
fi

# detect 는 RealSense 프레임을 받아야 초기화가 끝난다. 위 대기 뒤에 띄운다.
if want detect; then
    launch detect ros2 run object_detection object_detection
    sleep 5
fi

if want bridge; then
    launch bridge env PYTHONUNBUFFERED=1 python3 -u "$GESTURE_DIR/realsense_bridge.py"
fi

# 로봇을 움직이는 것들은 마지막에. 앞의 것이 다 준비된 뒤에 붙어야 한다.
if want sign; then
    launch sign env PYTHONUNBUFFERED=1 SIGN_SHOW_WINDOW=1 \
        python3 -u "$BLOCK_DIR/block_sort.py" sign --admin
fi

echo
echo "전부 띄웠습니다. 대시보드: $ADMIN_URL"
echo "Ctrl+C 로 이 스크립트가 띄운 것들을 정리합니다."
wait
