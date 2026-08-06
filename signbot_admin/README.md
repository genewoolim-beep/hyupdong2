# SignBot Control - 수어 로봇 제어 관리자 대시보드

Figma 디자인: https://www.figma.com/design/GQEjFB8XUtMj0pvXY5J68o

## 실행 방법

```bash
cd signbot_admin
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

브라우저에서 http://localhost:5000 접속.

## 구조

```
signbot_admin/
├── app.py                 # Flask 라우트 + 목(mock) API
├── templates/
│   ├── base.html          # 상단 네비바 + 레이아웃 (단일 페이지라 사이드바 없음)
│   └── dashboard.html     # 유일한 화면 — 통계/카메라/문장/로봇/구역/로그를 한 화면에
└── static/
    ├── css/style.css      # Figma 디자인 토큰 반영 (색상 변수는 :root 참고)
    └── js/dashboard.js    # 폴링으로 상태/로그/구역/문장 갱신, 제어 버튼 이벤트
```

## 실제 시스템과 연동하기

`hyupdong2/sign_control/sign_demo.py run --llm --admin` 과 `hyupdong2/block_sort/block_sort.py
<mode> --admin` 이 아래 1·2·3·5·6번을 전부 이미 구현해서 쏘고 있습니다 — 새로 붙이는 다른
인식/로봇 노드가 있다면 그 코드를 참고하세요.

1. **수어 인식 결과/로그 전송**: 단어가 확정될 때마다 `POST /api/recognize` 에
   `{"sign": "파랑", "confidence": 0.91, "command": "문장에 추가", "result": "성공"}` 형태로 전송하세요.
   `command`/`result`를 직접 주면 그대로 로그에 기록되고(대시보드 "오늘 인식 횟수"/"인식 성공률"의 근거),
   생략하면 예전 방향 명령 데모용 `COMMAND_MAP` 으로 추정합니다.

2. **로봇 상태 보고**: `block_sort.py --admin` 은 실행 시작 시 `POST /api/robot/status` 에
   `{"connected": true, "model": "Doosan M0609", "last_command": "pick 빨강 3"}` 를, 끝나면
   `{"connected": false}` 를 보냅니다. `/api/robot/control`(관리자 → 로봇, pause/resume/e_stop)과
   방향이 반대인 엔드포인트입니다 — 넘어온 키만 `robot_state`에 반영되고, `connected`/`e_stop`은
   대시보드의 "연결된 로봇"/"활성 알람" 통계에도 바로 반영됩니다. block_sort.py 는 명령 하나 실행하고
   끝나는 1회성 스크립트라 "연결됨"은 사실상 "지금 이 순간 명령을 실행 중"이라는 뜻입니다 — 상시
   연결 상태를 보려면 로봇 제어를 상주 노드로 바꿔야 합니다(다음 단계 후보).

3. **카메라 스트리밍**: `/video_feed` 는 `POST /api/frame` (JPEG 바이트, `Content-Type: image/jpeg`)으로
   들어온 최신 프레임을 그대로 MJPEG로 중계합니다. `sign_demo.py --admin` 이 매 프레임을 인코딩해
   백그라운드 스레드로 12fps 상한으로 전송합니다 — 웹캠 장치는 한 프로세스만 열 수 있으므로, Flask가
   카메라를 직접 여는 대신 이미 캡처 중인 인식 프로세스에서 프레임을 받아오는 구조입니다.

4. **실시간성 개선**: 폴링(문장/구역은 0.3~5초) 대신 `Flask-SocketIO` 로 서버 → 브라우저 push를
   구현하면 지연을 더 줄일 수 있습니다.

5. **구역별 블록 현황**: 구역은 로봇 존(로봇이 물건을 놓는 1~4번구역)과 휴먼 존(사람이 참고용
   물건을 놓는 1~4번구역), 둘로 나뉩니다 — "똑같이"/"좌우대칭" 명령이 휴먼 존을 참고해 로봇 존에
   지시를 내리는 구조라서요. 물리적 제약으로 5·6번구역은 운용하지 않습니다 (2x2로 표시).
   구역 상태를 인식할 때마다 `POST /api/zones/robot` 또는 `POST /api/zones/human` 에
   `{"zone": "3번구역", "color": "파랑"}` 형태로 전송하세요. `color`를 생략/`null`로 보내면 해당
   구역이 빈 구역으로 표시됩니다. 대시보드의 "로봇/인간 구역 현황" 패널이 5초마다 각각
   `GET /api/zones/<robot|human>` 를 폴링해 갱신합니다.

   `block_sort.py --admin` 이 실제 연동 예시입니다: `scan`/`read-human`/`read-free` 는 전체 순회
   후 두 구역 다 갱신하고, `copy`/`copy-mirror`(`copy_human()`)는 순회로 읽은 초기 상태 +
   배치 완료 후의 로봇 존을 갱신하고, `pick <색|구역> <구역>`(`run_one()`)은 옮긴 구역 하나만
   갱신합니다. 부분 순회(`patrol(want=...)`)의 결과는 밀지 않습니다 — 못 본 구역까지 비었다고
   잘못 표시할 수 있어서입니다.

6. **인식 중인 문장 표시**: `POST /api/sentence` 로 문장 상태를 보냅니다.
   - `{"action": "start"}` — 문장 캡처 시작(예: 여는 박수). 첫 단어 전에도 "구성 중" 배지가 뜬다.
   - `{"action": "peek", "word": "파랑"}` — 아직 확정 안 된, 지금 손을 든 동안의 실시간 추정 단어.
     점선 칩으로 표시되어 사용자가 자기가 제대로 신호를 보내고 있는지 바로 확인할 수 있다.
   - `{"action": "append", "word": "파랑"}` — 확정된 단어를 문장에 추가.
   - `{"action": "clear"}` — 문장 비우기.
   - `{"action": "translate", "text": "..."}` — LLM 통역 결과(또는 글로스를 이어붙인 대체 텍스트) 표시.
   카메라 모니터링 패널이 0.3초마다 `GET /api/sentence` 를 폴링해 반영합니다.

7. **디버그 로그**: 로봇 제어, LLM 호출, ROS2 브리지 등에서 오류/이상 상황이 생기면
   `POST /api/debug` 에 `{"level": "error", "source": "robot_control", "message": "..."}` 형태로
   전송하세요 (`level`: info/warn/error). "로봇 상태 · 제어" 패널 하단에 실시간으로 쌓입니다.
   아직 어디서도 이 엔드포인트를 호출하지 않으므로 지금은 빈 상태입니다 — 향후 로봇 제어 연동 시
   예외 처리부에서 이 엔드포인트를 호출하도록 붙이세요.

8. **작업모드 ↔ 제어모드**: `GET/POST /api/mode` 로 지금 활성 인터페이스(`work`/`control`)를
   보고합니다. `sign_demo.py --admin` 은 "모드변경" 글로스가 확정되면 `control` 을,
   `hand_gesture_control.py --admin` 은 양손 3초 유지로 `work` 를 보냅니다. 모드가 실제로 바뀌면
   브라우저가 작업 화면(`/`) ↔ 제어 화면(`/control`)으로 자동 이동합니다.
   `POST /api/frame/control` 은 `hand_gesture_control.py` 의 `camera` 창(조이스틱 원 + 손
   스켈레톤 + z 게이지가 그려진 화면)을 받아 `/control_video_feed` 로 중계합니다 — 제어 화면의
   작은 보조 패널용입니다.

9. **로봇 시점(RealSense) 중계**: `hand_gesture_control/realsense_bridge.py` 가 로봇의 RealSense
   컬러 토픽을 구독해 `POST /api/frame/realsense` 로 올리면 `/realsense_video_feed` 로 중계합니다.
   제어 화면의 **큰 주 패널**입니다 — 손동작으로 로봇을 조종하려면 조종자 웹캠(위 8번)이 아니라
   로봇이 실제로 보는 화면을 봐야 하기 때문입니다.

## Docker 공유 시 참고

YOLO 모델(`Fruits/train_20260731_103755/weights/best.pt` 등)이나 카메라/GPU 장치를 이 관리자 UI와
같은 컨테이너에서 쓰려면 `docker run --device=/dev/video0 --gpus all ...` 형태로 장치를 마운트하고,
`requirements.txt`에 `ultralytics`, `opencv-python`, `pyrealsense2` 등을 추가하세요.
