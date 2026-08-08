/**
 * SignBot Control - 대시보드 프런트엔드 로직
 * /api/status, /api/logs 를 폴링하여 화면을 갱신하고,
 * 로봇 제어 버튼 클릭 시 /api/robot/control 로 명령을 전송합니다.
 *
 * 실제 실시간성이 중요하다면 폴링 대신 Flask-SocketIO 등으로 교체하는 것을 권장합니다.
 */
(function () {
  const POLL_INTERVAL_MS = 5000;
  const SENTENCE_POLL_INTERVAL_MS = 300; // 진행중 단어(live_word)까지 보여주려면 훨씬 자주 폴링해야 함
  const MODE_POLL_INTERVAL_MS = 1000;    // 작업모드/제어모드 전환을 빠르게 체감하도록
  const MODE_LABEL = { work: "작업모드", control: "제어모드", voice: "음성모드" };

  function fmtNumber(n) {
    return Number(n).toLocaleString("ko-KR");
  }

  async function fetchStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) return;
      const data = await res.json();
      updateStats(data.stats);
      updateRobot(data.robot);
    } catch (e) {
      console.error("status polling failed", e);
    }
  }

  // TCP 길이와 카메라 연결 상태. block_sort.py --admin 이 5초마다 보고한다.
  // 보고하는 프로세스가 없으면 '미보고' 로 남긴다 — 거짓으로 '정상' 이라고
  // 쓰는 것보다 낫다. 실제로 오늘 RealSense USB 가 빠진 채로 돌아간 적이 있다.
  async function fetchHardware() {
    try {
      const res = await fetch("/api/hardware");
      if (!res.ok) return;
      const hw = await res.json();

      const tcpEl = document.getElementById("statTcp");
      if (tcpEl) {
        const t = hw.tcp || {};
        tcpEl.textContent = (t.length_mm === null || t.length_mm === undefined)
          ? "미보고"
          : Number(t.length_mm).toFixed(1) + " mm" + (t.name ? " · " + t.name : "");
      }

      syncTcpAlarm(hw.tcp);

      const camEl = document.getElementById("statCams");
      if (camEl) {
        const c = hw.cameras || {};
        const label = { realsense: "RealSense", webcam: "웹캠", detection: "검출" };
        const parts = Object.keys(label).map(function (k) {
          return (c[k] ? "🟢 " : "🔴 ") + label[k];
        });
        camEl.textContent = hw.updated_at ? parts.join("  ") : "미보고";
      }
    } catch (e) {
      console.error("hardware polling failed", e);
    }
  }

  function updateStats(stats) {
    const map = {
      statConnectedRobots: stats.connected_robots + "대",
      statTodayCount: fmtNumber(stats.today_count) + "회",
      statSuccessRate: stats.success_rate + "%",
      statAlerts: stats.active_alerts + "건",
    };
    Object.entries(map).forEach(([id, text]) => {
      const el = document.getElementById(id);
      if (el) el.textContent = text;
    });
  }

  function updateRobot(robot) {
    const pill = document.getElementById("robotStatusPill");
    const pillText = document.getElementById("robotStatusText");
    if (pill && pillText) {
      if (robot.connected) {
        pill.classList.remove("offline");
        pillText.textContent = "로봇 연결됨";
      } else {
        pill.classList.add("offline");
        pillText.textContent = "로봇 연결 끊김";
      }
    }
    const statConn = document.getElementById("statConn");
    if (statConn) statConn.textContent = robot.connected ? "정상 · " + robot.model : "연결 끊김";
    const statMode = document.getElementById("statMode");
    if (statMode) statMode.textContent = robot.mode;
    const statLastCmd = document.getElementById("statLastCmd");
    if (statLastCmd) statLastCmd.textContent = robot.last_command;
  }

  let lastKnownMode = null; // 실제로 "바뀐 순간"에만 자동 이동하기 위한 기준값

  function renderMode(data) {
    const label = MODE_LABEL[data.mode] || data.mode;

    const pill = document.getElementById("modePill");
    if (pill) pill.className = "mode-pill mode-" + data.mode;
    const text = document.getElementById("modeText");
    if (text) text.textContent = label;

    const banner = document.getElementById("modeBanner");
    if (banner) banner.className = "panel mode-banner mode-banner-" + data.mode;
    // 제어 화면 제목의 "제어 화면: 작업모드" 부분. 색으로도 모드를 알린다 —
    // 큰 배너를 없앴으니(카메라가 한 화면에 들어와야 한다) 이 한 줄이 그 몫이다.
    const bannerValue = document.getElementById("modeBannerValue");
    if (bannerValue) {
      bannerValue.textContent = label;
      bannerValue.className = "page-title-mode mode-" + data.mode;
    }
    const hint = document.getElementById("modeBannerHint");
    if (hint) {
      const HINTS = {
        control: "사람이 손동작(hand_gesture_control.py)으로 로봇을 직접 조종 중입니다.",
        voice: '음성으로 명령하는 중입니다. 명령 후 "실행"이라고 말해야 움직입니다. "작업모드"라고 말하거나 아래 버튼으로 돌아갈 수 있습니다.',
      };
      hint.textContent = HINTS[data.mode]
        || '로봇은 수어 명령을 기다리는 중입니다. "박수 - 모드변경 - 박수"로 제어권을 넘길 수 있습니다.';
    }
    const updatedAt = document.getElementById("modeUpdatedAt");
    if (updatedAt) updatedAt.textContent = data.updated_at;

    // 음성모드 버튼은 지금 모드에 맞는 쪽만 보인다 — work 면 "음성모드 진입",
    // voice 면 "작업모드로 복귀". 둘 다 있으면 헷갈리므로 하나만 남긴다.
    const enterBtn = document.getElementById("btnVoiceModeEnter");
    const exitBtn = document.getElementById("btnVoiceModeExit");
    if (enterBtn) enterBtn.style.display = data.mode === "work" ? "" : "none";
    if (exitBtn) exitBtn.style.display = data.mode === "voice" ? "" : "none";

    // 모드가 실제로 전환된 순간에만, 그 모드에 맞는 화면으로 자동 이동한다.
    // (매 폴링마다 같은 값이 와도 리다이렉트하면 안 되므로 이전 값과 비교한다.)
    if (lastKnownMode !== null && lastKnownMode !== data.mode) {
      const path = window.location.pathname;
      if (data.mode === "control" && path === "/") {
        window.location.href = "/control";
      } else if (data.mode === "work" && path === "/control") {
        window.location.href = "/";
      }
    }
    lastKnownMode = data.mode;
  }

  async function fetchMode() {
    try {
      const res = await fetch("/api/mode");
      if (!res.ok) return;
      renderMode(await res.json());
    } catch (e) {
      console.error("mode polling failed", e);
    }
  }

  async function fetchLogs() {
    const tbody = document.getElementById("logTableBody");
    if (!tbody) return;
    try {
      const res = await fetch("/api/logs");
      if (!res.ok) return;
      const logs = await res.json();
      tbody.innerHTML = logs
        .slice(0, 5)
        .map(
          (log) => `
        <tr>
          <td>${log.time}</td>
          <td>${log.sign}</td>
          <td>${log.confidence}</td>
          <td>${log.command}</td>
          <td class="${log.result === "성공" ? "result-success" : "result-fail"}">${log.result}</td>
        </tr>`
        )
        .join("");
      if (logs[0]) {
        const recogValue = document.getElementById("recogValue");
        if (recogValue) recogValue.textContent = `"${logs[0].sign}" (신뢰도 ${logs[0].confidence})`;
      }
    } catch (e) {
      console.error("log polling failed", e);
    }
  }

  function renderZoneGrid(gridId, zones) {
    const grid = document.getElementById(gridId);
    if (!grid) return;
    grid.innerHTML = zones
      .map((z) =>
        z.color
          ? `<div class="zone-card">
               <div class="zone-name">${z.zone}</div>
               <div class="zone-swatch" style="background:${z.color_hex}"></div>
               <div class="zone-color-label">${z.color}</div>
             </div>`
          : `<div class="zone-card">
               <div class="zone-name">${z.zone}</div>
               <div class="zone-swatch zone-empty"></div>
               <div class="zone-color-label zone-empty-label">비어있음</div>
             </div>`
      )
      .join("");
  }

  async function fetchZoneSpace(space, gridId) {
    if (!document.getElementById(gridId)) return;
    try {
      const res = await fetch(`/api/zones/${space}`);
      if (!res.ok) return;
      renderZoneGrid(gridId, await res.json());
    } catch (e) {
      console.error(`zone(${space}) polling failed`, e);
    }
  }

  async function fetchZones() {
    await Promise.all([
      fetchZoneSpace("robot", "zoneGridRobot"),
      fetchZoneSpace("human", "zoneGridHuman"),
    ]);
  }

  function renderDebugLog(entries) {
    const box = document.getElementById("debugLog");
    if (!box) return;
    box.innerHTML = entries.length
      ? entries
          .map(
            (d) => `
        <div class="debug-entry level-${d.level}">
          <span class="debug-time">${d.time}</span>
          <span class="debug-source">[${d.source}]</span>
          <span class="debug-message">${d.message}</span>
        </div>`
          )
          .join("")
      : '<span class="sentence-placeholder">오류/디버그 메시지가 여기 표시됩니다</span>';
  }

  async function fetchDebugLog() {
    if (!document.getElementById("debugLog")) return;
    try {
      const res = await fetch("/api/debug");
      if (!res.ok) return;
      renderDebugLog(await res.json());
    } catch (e) {
      console.error("debug log polling failed", e);
    }
  }

  const SENTENCE_STATUS_LABEL = { idle: "대기 중", building: "문장 구성 중…", translated: "통역 완료" };

  function renderSentence(data) {
    const wordsEl = document.getElementById("sentenceWords");
    if (!wordsEl) return;
    const chips = data.words.map((w) => `<span class="word-chip">${w}</span>`).join("");
    const liveChip = data.live_word
      ? `<span class="word-chip word-chip-pending">${data.live_word}</span>`
      : "";
    wordsEl.innerHTML =
      chips + liveChip || '<span class="sentence-placeholder">아직 인식된 단어가 없습니다</span>';

    const statusEl = document.getElementById("sentenceStatus");
    if (statusEl) {
      statusEl.textContent = SENTENCE_STATUS_LABEL[data.status] || data.status;
      statusEl.className = "sentence-status status-" + data.status;
    }

    const translatedEl = document.getElementById("sentenceTranslated");
    if (translatedEl) {
      if (data.translated) {
        translatedEl.style.display = "block";
        translatedEl.textContent = `"${data.translated}"`;
      } else {
        translatedEl.style.display = "none";
      }
    }
  }

  async function fetchSentence() {
    if (!document.getElementById("sentenceWords")) return;
    try {
      const res = await fetch("/api/sentence");
      if (!res.ok) return;
      renderSentence(await res.json());
    } catch (e) {
      console.error("sentence polling failed", e);
    }
  }

  async function sendControl(action) {
    try {
      const res = await fetch("/api/robot/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      const robot = await res.json();
      updateRobot(robot);
    } catch (e) {
      console.error("control command failed", e);
    }
  }

  function bindControlButtons() {
    const pause = document.getElementById("btnPause");
    const resume = document.getElementById("btnResume");
    const estop = document.getElementById("btnEStop");
    if (pause) pause.addEventListener("click", () => sendControl("pause"));
    if (resume) resume.addEventListener("click", () => sendControl("resume"));
    if (estop)
      estop.addEventListener("click", () => {
        if (confirm("비상정지를 실행하시겠습니까?")) sendControl("e_stop");
      });
  }

  // 음성모드 진입·복귀 — 대시보드가 /api/mode 에 직접 쓰는 유일한 자리다.
  // 나머지 전환(work↔control, work↔voice 음성 트리거)은 block_sort.py 쪽이
  // 스스로 판단해서 쓴다. 여기서는 사람이 버튼을 눌렀을 때만 쏜다.
  async function sendMode(mode) {
    try {
      await fetch("/api/mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
      renderMode(await (await fetch("/api/mode")).json());
    } catch (e) {
      console.error("mode switch failed", e);
    }
  }

  function bindModeButtons() {
    const enterBtn = document.getElementById("btnVoiceModeEnter");
    const exitBtn = document.getElementById("btnVoiceModeExit");
    if (enterBtn) enterBtn.addEventListener("click", () => sendMode("voice"));
    if (exitBtn) exitBtn.addEventListener("click", () => sendMode("work"));
  }

  // 엔드이펙터(TCP) 가 틀리면 화면을 덮는 큰 경고를 띄운다.
  //
  // **판정은 여기서 하지 않는다.** block_sort 가 보내는 tcp.ok 를 그대로 쓴다
  // (tcp_info 가 이름과 길이를 같이 본다). 여기서 "250mm 인가" 를 다시 적으면
  // 문턱이 두 곳에 생기고, 한쪽만 고쳤을 때 화면이 조용히 거짓말을 한다.
  //
  // 보고가 아직 없으면(length_mm 없음) 경고하지 않는다 — 그건 '틀렸다' 가 아니라
  // '로봇 프로그램이 아직 안 떴다' 는 뜻이다.
  let tcpAlarmDismissed = null;      // 사용자가 확인한 값(같은 값이면 다시 안 띄운다)

  function syncTcpAlarm(t) {
    const box = document.getElementById("tcpAlarm");
    if (!box) return;
    const reported = t && t.length_mm !== null && t.length_mm !== undefined;
    const bad = reported && t.ok === false;
    if (!bad) {
      box.style.display = "none";
      tcpAlarmDismissed = null;      // 정상으로 돌아왔다 — 다음에 틀리면 다시 띄운다
      return;
    }
    const key = String(t.name) + "/" + String(t.length_mm);
    if (tcpAlarmDismissed === key) return;

    const val = document.getElementById("tcpAlarmValue");
    if (val) {
      val.textContent = Number(t.length_mm).toFixed(1) + " mm"
        + (t.name ? "  ·  " + t.name : "  ·  이름 없음");
    }
    const desc = document.getElementById("tcpAlarmDesc");
    if (desc && t.expect_mm) {
      desc.innerHTML = "티치펜던트에서 공구(TCP)를 <b>"
        + (t.expect || "") + " / " + Number(t.expect_mm).toFixed(0)
        + "mm</b> 로 설정하세요.";
    }
    box.style.display = "flex";
    // 지금 값을 요소에 적어 둔다. 닫기 버튼은 한 번만 묶으므로, key 를 클로저로
    // 잡으면 **처음 값에 굳어** 다음 오류를 닫았을 때 엉뚱한 값을 기억한다.
    box.dataset.key = key;
    const btn = document.getElementById("tcpAlarmClose");
    if (btn && !btn.dataset.bound) {
      btn.dataset.bound = "1";
      btn.addEventListener("click", () => {
        tcpAlarmDismissed = box.dataset.key;
        box.style.display = "none";
      });
    }
  }

  // 카메라 안내문("아직 연결되지 않았습니다")은 **영상이 없을 때만** 보여준다.
  //
  // 전에는 기본이 '보임' 이라 영상이 잘 와도 그 글자가 <img> 옆에 나란히 놓여
  // 여백을 만들었다. onerror 로만 켜도록 두면 안 되는 이유: MJPEG 연결은
  // 열려 있고 프레임만 안 오는 상태에서는 onerror 가 아예 안 난다
  // (app.py 의 gen_*_video_frames 는 프레임이 없으면 아무것도 내보내지 않는다).
  // 그래서 naturalWidth 로 **실제로 그려진 게 있는지**를 본다 — 첫 프레임이
  // 도착하면 0 보다 커진다.
  const FEEDS = [
    ["realsenseVideoFeed", "realsensePlaceholder"],
    ["controlVideoFeed", "controlVideoPlaceholder"],
    ["videoFeed", "videoPlaceholderText"],
  ];

  function syncFeedPlaceholders() {
    FEEDS.forEach(([imgId, phId]) => {
      const img = document.getElementById(imgId);
      const ph = document.getElementById(phId);
      if (!img || !ph) return;
      const live = img.naturalWidth > 0;
      ph.style.display = live ? "none" : "block";
      // 영상이 다시 오기 시작하면 되살린다 — onerror 가 껐을 수 있다.
      if (live) img.style.display = "";
    });
  }

  function startPolling() {
    syncFeedPlaceholders();
    setInterval(syncFeedPlaceholders, 1000);
    fetchStatus();
    fetchLogs();
    fetchZones();
    fetchSentence();
    fetchDebugLog();
    fetchMode();
    fetchHardware();
    setInterval(fetchHardware, POLL_INTERVAL_MS);
    setInterval(fetchStatus, POLL_INTERVAL_MS);
    setInterval(fetchLogs, POLL_INTERVAL_MS);
    setInterval(fetchZones, POLL_INTERVAL_MS);
    setInterval(fetchSentence, SENTENCE_POLL_INTERVAL_MS);
    setInterval(fetchDebugLog, POLL_INTERVAL_MS);
    setInterval(fetchMode, MODE_POLL_INTERVAL_MS);
  }

  document.addEventListener("DOMContentLoaded", bindControlButtons);
  document.addEventListener("DOMContentLoaded", bindModeButtons);

  window.SignBotDashboard = { startPolling, fetchStatus, fetchLogs };
})();
