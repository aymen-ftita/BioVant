/**
 * Results Component Logic
 * Handles rendering of hypnograms, AASM metrics, and stage breakdowns.
 */
let _stages = null, _stagesInt = null, _classNames = null;

function renderResults(data) {
  const dyn = document.getElementById("dynamic-results");
  if (!dyn) return;
  dyn.innerHTML = "";

  if (!data.results || data.results.length === 0) return;

  const primary = data.results[0];
  window._stages = primary.stages;
  window._stagesInt = primary.stages_int;
  window._classNames = primary.stats.class_names;

  const resultsSec = document.getElementById("results");
  if (resultsSec) resultsSec.classList.add("visible");
  
  const osaLbl = document.getElementById("step2-lbl");
  if (osaLbl) osaLbl.style.display = "flex";
  
  const osaPanel = document.getElementById("osa-panel");
  if (osaPanel) osaPanel.style.display = "block";
  
  const osaReport = document.getElementById("osa-report");
  if (osaReport) osaReport.style.display = "none";
  
  // Auto-extract and display hypnogram features in the OSA panel
  if (typeof fetchAndDisplayFeatures === 'function') {
    fetchAndDisplayFeatures();
  }
  
  if (typeof goToStep === 'function') goToStep(4);

  data.results.forEach((res, i) => {
    const { model_info, stages, stats } = res;
    const wrapper = document.createElement("div");
    wrapper.style.marginBottom = "40px";
    wrapper.innerHTML = `
      <div style="font-family:var(--serif); font-size:20px; font-weight:700; color:var(--red); margin-bottom:16px; border-bottom:1px solid var(--border); padding-bottom:8px;">Comparaison: Modèle ${model_info.type}</div>
      <div class="sec-lbl">Hypnogram</div>
      <div class="hypno-wrap">
        <div class="hypno-hdr">
          <div class="hypno-title">Architecture du Sommeil</div>
          <div class="legend" id="leg-${i}"></div>
        </div>
        <canvas id="hypno-canvas-${i}" style="width:100%; height:180px; display:block; cursor:crosshair;"></canvas>
        <div class="time-axis" id="time-axis-${i}"></div>
      </div>
      <div class="sec-lbl">AASM Metrics</div>
      <div class="stats-grid" id="stats-grid-${i}"></div>
      <div class="sec-lbl">Stage Breakdown</div>
      <div class="breakdown-grid" id="breakdown-grid-${i}"></div>
      <div class="sec-lbl">Time Distribution</div>
      <div class="bar-chart" id="bar-chart-${i}"></div>
      <div class="alerts-section" id="alerts-section-${i}"></div>
    `;
    dyn.appendChild(wrapper);

    renderStats(stats, `stats-grid-${i}`);
    renderBreakdown(stats, `breakdown-grid-${i}`);
    renderBarChart(stats, `bar-chart-${i}`);
    renderAlerts(stats, `alerts-section-${i}`);
    requestAnimationFrame(() => requestAnimationFrame(() => renderHypnogram(stages, stats.class_names, `hypno-canvas-${i}`, `leg-${i}`, `time-axis-${i}`)));
  });
}

function renderHypnogram(stages, class_names, canvasId, legId, axisId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const wrap = canvas.parentElement;
  const tip = document.getElementById("tooltip");
  const SC = window.SC || { Wake: '#c0392b', NREM: '#1d4ed8', REM: '#047857' };

  const order = class_names || ["Wake", "NREM", "REM"];
  const leg = document.getElementById(legId);
  if (leg) {
    leg.innerHTML = order.map(s => `<div class="leg-item"><div class="leg-dot" style="background:${SC[s] || SC['NREM']}"></div>${s}</div>`).join('');
  }

  const W = wrap.clientWidth - 48, H = 180;
  canvas.style.width = W + "px"; canvas.style.height = H + "px";
  canvas.width = Math.round(W * devicePixelRatio); canvas.height = Math.round(H * devicePixelRatio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.scale(devicePixelRatio, devicePixelRatio);
  const PAD = { top: 18, right: 10, bottom: 8, left: 46 }, CW = W - PAD.left - PAD.right, CH = H - PAD.top - PAD.bottom;
  const rowH = CH / order.length;
  
  ctx.strokeStyle = "rgba(38,28,16,.07)"; ctx.lineWidth = 1;
  for (let i = 0; i <= order.length; i++) { const y = PAD.top + i * rowH; ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(W - PAD.right, y); ctx.stroke(); }
  
  order.forEach((s, i) => { ctx.fillStyle = SC[s] || SC['NREM']; ctx.font = "500 10px 'DM Mono',monospace"; ctx.textAlign = "right"; ctx.fillText(s, PAD.left - 8, PAD.top + i * rowH + rowH / 2 + 4); });
  
  const segW = CW / stages.length;
  stages.forEach((st, i) => { const yi = order.indexOf(st); ctx.globalAlpha = 0.65; ctx.fillStyle = SC[st] || SC['NREM']; ctx.fillRect(PAD.left + i * segW, PAD.top + yi * rowH + 1, segW + 0.5, rowH - 2); });
  
  ctx.globalAlpha = 1;
  ctx.beginPath(); ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.strokeStyle = "rgba(38,28,16,.5)";
  stages.forEach((st, i) => { const x = PAD.left + i * segW, yi = order.indexOf(st), y = PAD.top + yi * rowH + rowH / 2; i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
  ctx.stroke();

  const tot = stages.length * 0.5, ax = document.getElementById(axisId); if (ax) ax.innerHTML = "";
  for (let t = 0; t <= 7; t++) { const m = Math.round(t * tot / 7), h = Math.floor(m / 60), mm = m % 60; if (ax) ax.innerHTML += `<span>${h}h${String(mm).padStart(2, "0")}</span>`; }

  canvas.addEventListener("mousemove", e => {
    const r = canvas.getBoundingClientRect(), usable = r.width - PAD.left - PAD.right;
    const idxHover = Math.floor((e.clientX - r.left - PAD.left) / usable * stages.length);
    if (idxHover >= 0 && idxHover < stages.length) {
      const mins = (idxHover * 30) / 60, h = Math.floor(mins / 60), mm = Math.floor(mins % 60);
      if (tip) {
        tip.textContent = `${h}h${String(mm).padStart(2, "0")} · Époque ${idxHover + 1} · ${stages[idxHover]}`;
        tip.classList.add("visible"); tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY - 10) + "px";
      }
    } else if (tip) tip.classList.remove("visible");
  });
}

function renderStats(s, containerId) {
  const g = document.getElementById(containerId); if (!g) return;
  g.innerHTML = "";
  const fmt = (n) => n != null ? n : "–";
  [{ lbl: "Sleep Efficiency", val: s.se, unit: "%", note: "Normal ≥85%", cls: s.se >= 85 ? "good" : s.se >= 75 ? "warn" : "danger" },
  { lbl: "Total Sleep Time", val: fmt(s.tst), unit: "min", note: (s.tst / 60).toFixed(1) + "h", cls: "" },
  { lbl: "Time in Bed", val: fmt(s.tib), unit: "min", note: (s.tib / 60).toFixed(1) + "h", cls: "" },
  { lbl: "Sleep Latency", val: s.sol, unit: "min", note: "Normal 10–20 min", cls: s.sol > 20 ? "warn" : s.sol < 5 ? "danger" : "good" },
  { lbl: "REM Latency", val: s.rem_latency != null ? s.rem_latency : "N/A", unit: "min", note: "Normal 90–120 min", cls: s.rem_latency != null && s.rem_latency < 60 ? "danger" : "" },
  { lbl: "WASO", val: s.waso, unit: "min", note: "Wake After Sleep Onset", cls: s.waso > 30 ? "warn" : "" },
  ].forEach(c => {
    const d = document.createElement("div"); d.className = "stat-card " + c.cls;
    d.innerHTML = `<div class="stat-lbl">${c.lbl}</div><div class="stat-val">${c.val}<span class="stat-unit">${c.unit}</span></div><div class="stat-note">${c.note}</div>`;
    g.appendChild(d);
  });
}

function renderBreakdown(s, containerId) {
  const g = document.getElementById(containerId); if (!g) return;
  g.innerHTML = "";
  const fmt = (n) => n != null ? n : "–";
  const SC = window.SC || { Wake: '#c0392b', NREM: '#1d4ed8', REM: '#047857' };
  const order = s.class_names || ["Wake", "NREM", "REM"];
  order.forEach(st => {
    const d = document.createElement("div"); d.className = "stage-card";
    const color = SC[st] || SC['NREM'];
    const abbr = st === "Wake" ? "W" : st.replace("NREM", "NR").replace("REM", "R");
    d.innerHTML = `<div class="stage-sw" style="background:${color}18;color:${color};border:1.5px solid ${color}33">${abbr}</div><div><div class="stage-name">${st}</div><div class="stage-min" style="color:${color}">${fmt(s.stage_minutes[st])}<span style="font-size:11px;font-weight:400;color:var(--text3);font-family:var(--mono)"> min</span></div><div class="stage-pct">${s.stage_pct[st]}% ${st !== "Wake" ? "of TST" : "of TIB"}</div></div>`;
    g.appendChild(d);
  });
}

function renderBarChart(s, containerId) {
  const c = document.getElementById(containerId); if (!c) return;
  c.innerHTML = "";
  const fmt = (n) => n != null ? n : "–";
  const SC = window.SC || { Wake: '#c0392b', NREM: '#1d4ed8', REM: '#047857' };
  const order = s.class_names || ["Wake", "NREM", "REM"];
  const max = Math.max(...order.map(k => s.stage_minutes[k]));
  order.forEach(k => {
    const pct = Math.round(s.stage_minutes[k] / max * 100);
    const d = document.createElement("div"); d.className = "bar-row";
    const color = SC[k] || SC['NREM'];
    d.innerHTML = `<div class="bar-lbl">${k.toUpperCase()}</div><div class="bar-track"><div class="bar-fill" id="bf-${k}" style="background:${color};width:0%">${fmt(s.stage_minutes[k])}m</div></div>`;
    c.appendChild(d);
    requestAnimationFrame(() => setTimeout(() => { const el = document.getElementById("bf-" + k); if (el) el.style.width = pct + "%"; }, 120));
  });
}

function renderAlerts(s, containerId) {
  const sec = document.getElementById(containerId); if (!sec) return;
  sec.innerHTML = "";
  if (!s.alerts?.length) return;
  const lbl = document.createElement("div"); lbl.className = "sec-lbl"; lbl.textContent = "Alertes Cliniques"; lbl.style.marginBottom = "16px";
  sec.appendChild(lbl);
  s.alerts.forEach(a => {
    const d = document.createElement("div"); d.className = "alert-item";
    d.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg>${a}`;
    sec.appendChild(d);
  });
}
