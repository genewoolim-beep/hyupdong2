#!/usr/bin/env bash
# 컨테이너 진입점. ROS2, (있으면) Doosan 드라이버 워크스페이스, 이 저장소의
# colcon 워크스페이스를 순서대로 source 한 뒤 CMD/명령을 그대로 실행한다.
# run_all.sh 의 소싱 순서·set +u 처리와 동일한 이유를 따른다 — ROS 의
# setup.bash 가 미정의 변수를 참조해서 set -u 상태로는 죽는다.
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
[[ -f "${DSR_WS:-/dsr_ws}/install/setup.bash" ]] && source "${DSR_WS:-/dsr_ws}/install/setup.bash"
[[ -f /workspace/sign_pickandplace/install/setup.bash ]] && source /workspace/sign_pickandplace/install/setup.bash
set -u

exec "$@"
