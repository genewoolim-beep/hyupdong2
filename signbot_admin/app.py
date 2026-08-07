"""
SignBot Control - 수어 인식 로봇 제어 관리자 대시보드
Flask 기반 관리자 UI 백엔드
"""
from datetime import datetime
import random
import threading
import time

from flask import Flask, render_template, jsonify, request, Response

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 목(mock) 데이터 - 실제 연동 전 UI 확인용. 아래 함수들을 ROS2 노드/DB 연동으로 교체하세요.
# ---------------------------------------------------------------------------

SIGN_LABELS = ["앞으로 이동", "정지", "왼쪽으로 회전", "오른쪽으로 회전", "알 수 없음"]
COMMAND_MAP = {
    "앞으로 이동": "MOVE_FORWARD",
    "정지": "STOP",
    "왼쪽으로 회전": "TURN_LEFT",
    "오른쪽으로 회전": "TURN_RIGHT",
    "알 수 없음": "-",
}

# 실제 인식 이벤트로 채워진다 (sign_demo.py --admin 이 POST /api/recognize 로 기록).
# 서버 재시작 시 초기화되고 별도 저장소가 없으므로, "오늘"은 사실상 "이 서버 프로세스가
# 떠 있는 동안"을 뜻한다.
recognition_logs = []

# block_sort.py --admin 이 POST /api/robot/status 로 실제 값을 보고하기 전까지의 초기 상태.
# 지금은 로봇 제어 프로세스가 아무것도 안 떠 있으므로 connected: False 가 사실과 맞다.
robot_state = {
    "connected": False,
    "model": "Doosan M0609",
    "mode": "-",
    "last_command": "-",
    "power": "-",
    "e_stop": False,
}

# block_sort.py --admin 이 POST /api/hardware 로 5초마다 갱신한다.
# tcp.length_mm 는 설정값이 아니라 로봇이 지금 쓰고 있는 실측이다 — 같은 순간의
# TCP 좌표와 플랜지 좌표를 읽어 그 차이를 잰다. 공구를 바꾸면 바로 반영된다.
# cameras 는 각 영상 토픽에 발행자가 있는지다. USB 가 빠지면 False 로 떨어진다.
# 지금 제어모드를 잡고 있는 프로세스가 살아 있다는 신호.
# 2026-08-06 이후 제어모드는 block_sort 안에서 도므로 block_sort 가 보낸다.
# 뜻이 뒤집혔다 — 전에는 '넘길 상대가 있는가' 였고, 지금은 block_sort 가
# 제어모드에 들어가기 전에 '**남이** 이미 잡고 있는가' 를 보는 데 쓴다
# (둘이 speedl 을 쏘면 한 로봇에 지령이 둘이 된다).
# hand_gesture_control.py --admin 을 단독으로 쓸 때도 같은 신호를 보낸다.
control_alive = {"at": None}

hardware_state = {
    # tcp 는 block_sort 가 보고한다. ok/expect_mm 은 **보고하는 쪽이 판정한 결과**다 —
    # 여기서 다시 판단하지 않는다(문턱이 두 곳에 적히면 한쪽만 고쳐진다).
    "tcp": {"name": None, "length_mm": None, "offset_mm": None,
            "expect": None, "expect_mm": None, "tol_mm": None, "ok": None},
    "cameras": {"realsense": False, "webcam": False, "detection": False},
    "updated_at": None,
}

COLOR_HEX = {
    "빨강": "#DE3D3D",
    "주황": "#ED6B33",
    "노랑": "#F2B705",
    "초록": "#21C478",
    "파랑": "#336BF2",
    "보라": "#8B5CF6",
}

# 구역별 현재 컬러 블록 상태. color가 None이면 빈 구역.
# 물리적 제약으로 5·6번구역은 운용하지 않기로 함 — 1~4번구역만 2x2로 표시.
# 공간이 둘로 나뉜다: 로봇 존(로봇이 물건을 놓는 공간) / 휴먼 존(사람이 참고용 물건을 놓는 공간).
# "똑같이" · "좌우대칭" 명령은 휴먼 존 상태를 참고해서 로봇 존에 명령을 내리는 데 쓰인다.
# 실제 연동 시 오브젝트 디텍션 노드가 구역을 인식할 때마다 POST /api/zones/<robot|human> 로 갱신하세요.
robot_zone_state = {
    "1번구역": {"color": None, "updated_at": "-"},
    "2번구역": {"color": None, "updated_at": "-"},
    "3번구역": {"color": None, "updated_at": "-"},
    "4번구역": {"color": None, "updated_at": "-"},
}
human_zone_state = {
    "1번구역": {"color": None, "updated_at": "-"},
    "2번구역": {"color": None, "updated_at": "-"},
    "3번구역": {"color": None, "updated_at": "-"},
    "4번구역": {"color": None, "updated_at": "-"},
}
ZONE_SPACES = {"robot": robot_zone_state, "human": human_zone_state}

# 수어 인식 중 실시간으로 쌓이는 단어(글로스) 시퀀스.
# status: idle(대기) / building(구성 중) / translated(LLM 통역 완료)
# live_word: 아직 확정 전, 지금 손을 들고 있는 동안의 실시간 추정 단어(있으면 즉시 사용자에게 피드백용)
# 실제 연동 시 sign_control 노드가 단어를 인식할 때마다 POST /api/sentence 로 append 하세요.
current_sentence = {
    "words": [],
    "live_word": "",
    "status": "idle",
    "translated": None,
    "updated_at": "14:32:15",
}

# 디버깅용 오류/이벤트 로그. 로봇 상태 · 제어 패널 하단에 표시된다.
# 실제 연동 시 로봇 제어 노드, LLM 호출, ROS2 브리지 등에서 문제가 생기면
# POST /api/debug 로 기록하세요. level: info / warn / error
debug_logs = []

# 지금 어느 인터페이스가 활성인지. work(작업모드 — 수어로 로봇에 명령) /
# control(제어모드 — 사람이 손동작으로 로봇을 직접 조종).
# block_sort.py sign --admin 이 "모드변경" 글로스를 확정하면 control 로,
# 제어모드에서 나올 때(양손 3초 / Q / 여기서 전환) work 로 보고한다.
# 여기서 work 로 돌리면 제어모드가 그것을 보고 스스로 빠져나온다 —
# 대시보드가 조종을 끊는 유일한 길이다.
MODE_LABEL = {"work": "작업모드", "control": "제어모드"}
system_mode = {"mode": "work", "updated_at": "-"}


def get_zone_list(state):
    return [
        {
            "zone": zone,
            "color": info["color"],
            "color_hex": COLOR_HEX.get(info["color"]),
            "updated_at": info["updated_at"],
        }
        for zone, info in state.items()
    ]


def get_stats():
    total = len(recognition_logs)
    success = len([l for l in recognition_logs if l["result"] == "성공"])
    # 알람 = 실제 로봇 상태 이상 신호 — 비상정지가 걸려 있거나 로봇 연결이 끊겼을 때.
    alerts = (1 if robot_state["e_stop"] else 0) + (0 if robot_state["connected"] else 1)
    return {
        "connected_robots": 1 if robot_state["connected"] else 0,
        "today_count": total,
        "success_rate": round(success / total * 100, 1) if total else 0,
        "active_alerts": alerts,
    }


# ---------------------------------------------------------------------------
# 페이지 라우트
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html",
        active="work",
        stats=get_stats(),
        robot=robot_state,
        latest=recognition_logs[0] if recognition_logs else None,
        logs=recognition_logs[:5],
        robot_zones=get_zone_list(robot_zone_state),
        human_zones=get_zone_list(human_zone_state),
        sentence=current_sentence,
        debug_logs=debug_logs[:20],
        mode=system_mode,
        mode_label=MODE_LABEL,
    )


@app.route("/control")
def control():
    return render_template(
        "control.html",
        active="control",
        robot=robot_state,
        mode=system_mode,
        mode_label=MODE_LABEL,
    )


# ---------------------------------------------------------------------------
# API 라우트 (프런트에서 fetch로 폴링/제어) - 실제 로봇/비전 파이프라인과 연결 지점
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    """대시보드 상단 상태/통계 폴링용"""
    return jsonify({"robot": robot_state, "stats": get_stats()})


@app.route("/api/hardware")
def api_hardware():
    """TCP · 카메라 연결 상태 폴링용"""
    return jsonify(hardware_state)


@app.route("/api/hardware", methods=["POST"])
def api_hardware_update():
    """block_sort.py --admin 이 5초마다 보고한다.

    예: {"tcp": {"name": "rg2", "length_mm": 195.4, "offset_mm": [0,0,195.4]},
         "cameras": {"realsense": true, "webcam": true, "detection": true}}
    보낸 항목만 갱신한다 — 일부만 아는 쪽에서 보내도 나머지가 지워지지 않게.
    """
    data = request.get_json(force=True, silent=True) or {}
    if isinstance(data.get("tcp"), dict):
        hardware_state["tcp"].update(data["tcp"])
    if isinstance(data.get("cameras"), dict):
        hardware_state["cameras"].update(data["cameras"])
    hardware_state["updated_at"] = datetime.now().strftime("%H:%M:%S")
    return jsonify(hardware_state)


@app.route("/api/logs")
def api_logs():
    return jsonify(recognition_logs)


@app.route("/api/zones/<space>")
def api_zones(space):
    """대시보드 '로봇/인간 구역 현황' 패널 폴링용. space: robot | human"""
    state = ZONE_SPACES.get(space)
    if state is None:
        return jsonify({"error": f"unknown zone space: {space}"}), 404
    return jsonify(get_zone_list(state))


@app.route("/api/zones/<space>", methods=["POST"])
def api_zones_update(space):
    """
    오브젝트 디텍션 노드가 구역별 블록 색상을 인식할 때마다 이 엔드포인트로 POST하세요.
    space는 robot(로봇이 물건을 놓는 구역) 또는 human(사람이 참고용 물건을 놓는 구역)입니다.
    예: requests.post(".../api/zones/robot", json={"zone": "3번구역", "color": "파랑"})
        requests.post(".../api/zones/human", json={"zone": "2번구역", "color": "노랑"})
    color를 생략하거나 null로 보내면 해당 구역을 빈 구역으로 표시합니다.
    """
    state = ZONE_SPACES.get(space)
    if state is None:
        return jsonify({"error": f"unknown zone space: {space}"}), 404
    data = request.get_json(force=True, silent=True) or {}
    zone = data.get("zone")
    if zone not in state:
        return jsonify({"error": f"unknown zone: {zone}"}), 400
    color = data.get("color") or None
    state[zone] = {"color": color, "updated_at": datetime.now().strftime("%H:%M:%S")}
    return jsonify(get_zone_list(state))


@app.route("/api/sentence")
def api_sentence():
    """카메라 모니터링 화면의 '인식 중인 문장' 패널 폴링용"""
    return jsonify(current_sentence)


@app.route("/api/sentence", methods=["POST"])
def api_sentence_update():
    """
    sign_control 노드가 단어를 인식/삭제/통역할 때마다 이 엔드포인트로 POST하세요.
    예: requests.post(".../api/sentence", json={"action": "start"})
        requests.post(".../api/sentence", json={"action": "peek", "word": "파랑"})
        requests.post(".../api/sentence", json={"action": "append", "word": "파랑"})
        requests.post(".../api/sentence", json={"action": "clear"})
        requests.post(".../api/sentence", json={"action": "translate", "text": "파란 블록을 가져와줘"})

    peek은 아직 확정되지 않은, 지금 손을 들고 있는 동안의 실시간 추정값이다.
    다른 액션은 모두 확정된 상태 변화이므로 부수적으로 live_word를 비운다.
    """
    data = request.get_json(force=True, silent=True) or {}
    action = data.get("action", "append")
    if action == "start":
        # 문장 캡처를 막 시작함(예: 여는 박수) — 첫 단어가 오기 전에도 "구성 중" 상태를 바로 보여준다.
        current_sentence["words"] = []
        current_sentence["live_word"] = ""
        current_sentence["status"] = "building"
        current_sentence["translated"] = None
    elif action == "peek":
        current_sentence["live_word"] = data.get("word", "")
    elif action == "append":
        word = data.get("word")
        if word:
            current_sentence["words"].append(word)
            current_sentence["live_word"] = ""
            current_sentence["status"] = "building"
            current_sentence["translated"] = None
    elif action == "clear":
        current_sentence["words"] = []
        current_sentence["live_word"] = ""
        current_sentence["status"] = "idle"
        current_sentence["translated"] = None
    elif action == "translate":
        current_sentence["translated"] = data.get("text", "")
        current_sentence["live_word"] = ""
        current_sentence["status"] = "translated"
    current_sentence["updated_at"] = datetime.now().strftime("%H:%M:%S")
    return jsonify(current_sentence)


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """
    비전/수어 인식 노드가 인식 결과를 이 엔드포인트로 POST하도록 연동하세요.
    예: requests.post(".../api/recognize", json={"sign": "파랑", "confidence": 0.91,
                                                  "command": "문장에 추가", "result": "성공"})
    command/result을 생략하면 (예전 방향 명령 데모 호환용) COMMAND_MAP으로 추정합니다.
    """
    data = request.get_json(force=True, silent=True) or {}
    sign = data.get("sign", random.choice(SIGN_LABELS))
    confidence = float(data.get("confidence", round(random.uniform(0.4, 0.99), 2)))
    command = data.get("command")
    if command is None:
        command = COMMAND_MAP.get(sign, "-")
    result = data.get("result")
    if result is None:
        result = "성공" if confidence >= 0.6 and command != "-" else "실패"

    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "sign": sign,
        "confidence": confidence,
        "command": command,
        "result": result,
    }
    recognition_logs.insert(0, entry)
    if result == "성공":
        robot_state["last_command"] = command
    return jsonify(entry), 201


frame_lock = threading.Lock()
latest_frame = {"jpg": None}


@app.route("/api/frame", methods=["POST"])
def api_frame():
    """
    sign_control(sign_demo.py --admin)이 매 프레임(JPEG 바이트, Content-Type: image/jpeg)을
    이 엔드포인트로 POST하면 /video_feed 가 그 화면을 그대로 중계합니다.
    """
    data = request.get_data()
    if data:
        with frame_lock:
            latest_frame["jpg"] = data
    return "", 204


control_frame_lock = threading.Lock()
latest_control_frame = {"jpg": None}


@app.route("/api/frame/control", methods=["POST"])
def api_frame_control():
    """
    제어모드(block_sort/teleop_mode.py)가 매 프레임(실카메라 + 십자선 +
    손 스켈레톤 + z 게이지, JPEG 바이트)을 이 엔드포인트로 POST하면
    /control_video_feed 가 그 화면을 그대로 중계합니다.
    /api/frame(작업모드 카메라)과 별도 버퍼다 — 같은 걸 쓰면 두 화면이 서로 덮어쓴다.
    """
    data = request.get_data()
    if data:
        with control_frame_lock:
            latest_control_frame["jpg"] = data
    return "", 204


realsense_frame_lock = threading.Lock()
latest_realsense_frame = {"jpg": None}


@app.route("/api/frame/realsense", methods=["POST"])
def api_frame_realsense():
    """
    로봇의 RealSense 컬러 영상을 중계하는 브리지(realsense_bridge.py 등)가 매
    프레임(JPEG 바이트)을 이 엔드포인트로 POST하면 /realsense_video_feed 가
    그대로 중계합니다. 제어 화면에서 "로봇이 실제로 보는 시점"을 보여주는 용도라,
    조이스틱 조작용 카메라(/api/frame/control)와는 완전히 다른 화면·별도 버퍼다.
    """
    data = request.get_data()
    if data:
        with realsense_frame_lock:
            latest_realsense_frame["jpg"] = data
    return "", 204


@app.route("/api/debug")
def api_debug():
    """로봇 상태 · 제어 패널 하단 디버그 로그 폴링용"""
    return jsonify(debug_logs[:20])


@app.route("/api/debug", methods=["POST"])
def api_debug_add():
    """
    오류/디버그 이벤트를 기록하세요. level: info | warn | error
    예: requests.post(".../api/debug",
        json={"level": "error", "source": "robot_control", "message": "DSR_ROBOT2 연결 끊김"})
    """
    data = request.get_json(force=True, silent=True) or {}
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "level": data.get("level", "info"),
        "source": data.get("source", "-"),
        "message": data.get("message", ""),
    }
    debug_logs.insert(0, entry)
    del debug_logs[20:]
    return jsonify(entry), 201


@app.route("/api/mode")
def api_mode():
    """상단 배지 · 제어 화면 폴링용. 지금 활성 인터페이스(work/control)를 반환한다."""
    return jsonify(system_mode)


@app.route("/api/control/alive")
def api_control_alive():
    """제어 프로세스가 최근에 신호를 보냈는가. block_sort 가 전환 전에 확인한다."""
    at = control_alive["at"]
    fresh = at is not None and (datetime.now() - at).total_seconds() < 10
    return jsonify({"alive": fresh,
                    "at": at.strftime("%H:%M:%S") if at else None})


@app.route("/api/control/alive", methods=["POST"])
def api_control_alive_update():
    """hand_gesture_control.py 가 주기적으로 두드린다 (대기 중에도)."""
    control_alive["at"] = datetime.now()
    return jsonify({"ok": True})


@app.route("/api/mode", methods=["POST"])
def api_mode_update():
    """
    sign_demo.py --admin("모드변경" 확정 시 control) / hand_gesture_control.py --admin
    (양손 3초 유지 시 work) 이 모드가 바뀔 때마다 이 엔드포인트로 보고한다.
    예: requests.post(".../api/mode", json={"mode": "control"})
    """
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get("mode")
    if mode not in MODE_LABEL:
        return jsonify({"error": f"unknown mode: {mode}"}), 400
    system_mode["mode"] = mode
    system_mode["updated_at"] = datetime.now().strftime("%H:%M:%S")
    return jsonify(system_mode)


@app.route("/api/robot/status", methods=["POST"])
def api_robot_status():
    """
    실제 로봇 제어 프로세스(block_sort.py --admin 등)가 자기 상태를 보고하는 용도.
    /api/robot/control 과 방향이 반대다 (로봇 → 관리자). 넘어온 키만 갱신한다.
    예: requests.post(".../api/robot/status",
        json={"connected": true, "model": "Doosan M0609", "last_command": "pick 빨강 3"})
    """
    data = request.get_json(force=True, silent=True) or {}
    for key in ("connected", "mode", "last_command", "model", "power", "e_stop"):
        if key in data:
            robot_state[key] = data[key]
    return jsonify(robot_state)


@app.route("/api/robot/control", methods=["POST"])
def api_robot_control():
    """일시정지 / 비상정지 등 관리자 수동 제어. 실제 로봇 제어 노드로 명령을 전달하도록 구현하세요."""
    action = (request.get_json(force=True, silent=True) or {}).get("action")
    if action == "pause":
        robot_state["mode"] = "PAUSED"
    elif action == "resume":
        robot_state["mode"] = "AUTO"
    elif action == "e_stop":
        robot_state["mode"] = "E-STOP"
        robot_state["e_stop"] = True
    return jsonify(robot_state)


def gen_video_frames():
    boundary = b"--frame\r\n"
    while True:
        with frame_lock:
            frame = latest_frame["jpg"]
        if frame:
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.05)  # /api/frame으로 들어오는 속도 이상으로 돌 필요 없음


@app.route("/video_feed")
def video_feed():
    """
    sign_control(sign_demo.py --admin)이 POST /api/frame 으로 밀어넣는 최신 프레임을
    MJPEG로 중계합니다. 아직 프레임이 한 번도 안 왔으면 스트림은 열리되 데이터는 안 나갑니다.
    """
    return Response(
        gen_video_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def gen_control_video_frames():
    boundary = b"--frame\r\n"
    while True:
        with control_frame_lock:
            frame = latest_control_frame["jpg"]
        if frame:
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.05)


@app.route("/control_video_feed")
def control_video_feed():
    """
    hand_gesture_control.py --admin 이 POST /api/frame/control 로 밀어넣는 최신
    camera 화면(실카메라+조이스틱 원+손 스켈레톤+z 게이지)을 MJPEG로 중계합니다.
    제어 화면 페이지의 "조종 화면" 패널(작게 표시)용.
    """
    return Response(
        gen_control_video_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def gen_realsense_video_frames():
    boundary = b"--frame\r\n"
    while True:
        with realsense_frame_lock:
            frame = latest_realsense_frame["jpg"]
        if frame:
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.05)


@app.route("/realsense_video_feed")
def realsense_video_feed():
    """
    로봇의 RealSense 중계 브리지가 POST /api/frame/realsense 로 밀어넣는 최신
    화면을 MJPEG로 중계합니다. 제어 화면 페이지의 "로봇 시점" 패널(크게 표시)용 —
    실제로 로봇을 조종하려면 로봇이 보는 화면을 봐야 하므로 여기가 주 화면이다.
    """
    return Response(
        gen_realsense_video_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    # 개발 서버. 배포 시 gunicorn/uwsgi 등으로 교체하세요.
    # threaded=True 필수 — /video_feed 스트리밍 연결이 다른 API 폴링을 막지 않도록.
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
