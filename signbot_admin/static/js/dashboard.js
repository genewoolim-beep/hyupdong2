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

  function startPolling() {
    fetchStatus();
    fetchLogs();
    fetchZones();
    fetchSentence();
    fetchDebugLog();
    setInterval(fetchStatus, POLL_INTERVAL_MS);
    setInterval(fetchLogs, POLL_INTERVAL_MS);
    setInterval(fetchZones, POLL_INTERVAL_MS);
    setInterval(fetchSentence, SENTENCE_POLL_INTERVAL_MS);
    setInterval(fetchDebugLog, POLL_INTERVAL_MS);
  }

  document.addEventListener("DOMContentLoaded", bindControlButtons);

  window.SignBotDashboard = { startPolling, fetchStatus, fetchLogs };
})();
