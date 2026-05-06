/**
 * OSAAnalysis Component Logic
 * Handles the OSA severity prediction and clinical report generation.
 */

/**
 * Fetch extracted hypnogram features from the server and display them.
 * Called automatically when the OSA panel becomes visible (after hypnogram generation).
 */
async function fetchAndDisplayFeatures() {
  const wrapper = document.getElementById("extracted-features-wrapper");
  const loading = document.getElementById("feat-loading");
  const content = document.getElementById("feat-content");
  const badge = document.getElementById("feat-count-badge");
  
  if (!wrapper || !window._stagesInt || !window._classNames) return;
  
  // Show wrapper with loading state
  wrapper.style.display = "block";
  loading.style.display = "flex";
  content.style.display = "none";
  badge.textContent = "…";

  try {
    const res = await fetch("http://localhost:5000/extract_features", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        stages_int: window._stagesInt,
        class_names: window._classNames,
      })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    // Render each group
    const groups = ["timing", "stages", "latencies", "fragmentation", "rem_distribution"];
    let totalFeats = 0;
    let cardIdx = 0;

    groups.forEach(groupKey => {
      const grid = document.getElementById("feat-grid-" + groupKey);
      if (!grid || !data[groupKey]) return;
      grid.innerHTML = "";

      data[groupKey].forEach(feat => {
        totalFeats++;
        const card = document.createElement("div");
        card.className = "feat-card";
        card.style.animationDelay = (cardIdx * 40) + "ms";
        
        let valueDisplay = feat.value;
        if (feat.value === -1) valueDisplay = "N/A";
        
        let noteHtml = "";
        if (feat.note) {
          noteHtml = `<div class="feat-card-note">${feat.note}</div>`;
        }

        card.innerHTML = `
          <div class="feat-card-name">${feat.name}</div>
          <div class="feat-card-value">
            ${valueDisplay}<span class="feat-card-unit">${feat.unit}</span>
          </div>
          ${noteHtml}
        `;
        grid.appendChild(card);
        cardIdx++;
      });
    });

    // Metadata line
    const meta = data.metadata;
    let existingMeta = content.querySelector(".feat-meta-info");
    if (existingMeta) existingMeta.remove();
    
    const metaDiv = document.createElement("div");
    metaDiv.className = "feat-meta-info";
    metaDiv.innerHTML = `
      <span><div class="feat-meta-dot"></div>${meta.n_epochs} époques</span>
      <span><div class="feat-meta-dot"></div>${meta.is_3class ? "3 classes" : "5 classes"} (${meta.class_names.join(" / ")})</span>
      <span><div class="feat-meta-dot"></div>${totalFeats} features extraites</span>
    `;
    content.appendChild(metaDiv);

    // Update badge and reveal
    badge.textContent = totalFeats + " features";
    loading.style.display = "none";
    content.style.display = "block";

  } catch (e) {
    loading.innerHTML = `<span style="color:var(--red);">⚠ Erreur: ${e.message}</span>`;
    console.error("Feature extraction error:", e);
  }
}

async function predictOSA() {
  const btn = document.getElementById("btn-predict-osa");
  if (!btn) return;
  btn.classList.add("running"); btn.innerHTML = "Analyse en cours...";
  
  try {
    const payload = {
      stages_int: window._stagesInt,
      class_names: window._classNames,
      clinical_data: {
        age: document.getElementById("osa-age").value,
        gender: document.getElementById("osa-gender").value,
        bmi: document.getElementById("osa-bmi").value,
        avgsat: document.getElementById("osa-avg-sat").value,
        minsat: document.getElementById("osa-min-sat").value,
        pctsa90h: document.getElementById("osa-pctsa90").value || null,
        pctsa85h: document.getElementById("osa-pctsa85").value || null,
        pctsa95h: document.getElementById("osa-pctsa95").value || null,
        ai_all: document.getElementById("osa-ai-all").value || null,
        ai_nrem: document.getElementById("osa-ai-nrem").value || null,
        ai_rem: document.getElementById("osa-ai-rem").value || null,
      }
    };
    
    const res = await fetch("http://localhost:5000/predict_osa", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    renderOSAReport(data);

  } catch (e) {
    if (typeof showErr === 'function') showErr("Step 2 Error: " + e.message);
    else alert("Error: " + e.message);
  } finally {
    btn.classList.remove("running");
    btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/><path d="M12 8v4l3 3"/></svg> Générer le Rapport Clinique';
  }
}

function renderOSAReport(data) {
    const report = document.getElementById("osa-report");
    if (!report) return;
    report.style.display = "block";

    // 1. Severity Badge
    const badge = document.getElementById("osa-sev-badge");
    if (badge) {
        badge.textContent = data.severity;
        const s = data.severity.toLowerCase();
        badge.className = "osa-severity-badge " + (s.includes("severe") ? "sev-severe" : s.includes("moderate") ? "sev-moderate" : s.includes("mild") ? "sev-mild" : "sev-normal");
    }

    // 2. Probability Distribution
    if (data.probabilities) {
      const probaContainer = document.getElementById("osa-proba-bars");
      if (probaContainer) {
          probaContainer.innerHTML = "";
          const classOrder = ["Normal", "Mild", "Moderate", "Severe"];
          const classColors = { Normal: "#059669", Mild: "#d97706", Moderate: "#ea580c", Severe: "#dc2626" };
          
          classOrder.forEach(cls => {
            const pct = data.probabilities[cls] || 0;
            const row = document.createElement("div");
            row.className = "osa-proba-row";
            row.innerHTML = `
              <div class="osa-proba-label" style="color:${classColors[cls]}">${cls}</div>
              <div class="osa-proba-track">
                <div class="osa-proba-fill proba-${cls.toLowerCase()}" style="width:0%" data-w="${(pct * 100)}%">${(pct * 100).toFixed(1)}%</div>
              </div>
            `;
            probaContainer.appendChild(row);
            requestAnimationFrame(() => setTimeout(() => {
              const fill = row.querySelector(".osa-proba-fill");
              if (fill) fill.style.width = fill.getAttribute("data-w") + "%";
            }, 100));
          });
      }
    }

    // 3. SHAP
    const shaps = document.getElementById("osa-shaps");
    if (shaps && data.shap_explanations) {
        shaps.innerHTML = "";
        const maxImp = Math.max(...data.shap_explanations.map(x => Math.abs(x.impact)), 0.1);
        data.shap_explanations.forEach((sh, idx) => {
            const isPos = sh.impact > 0;
            const pct = (Math.abs(sh.impact) / maxImp * 45).toFixed(1);
            const row = document.createElement("div");
            row.className = "shap-item";
            row.innerHTML = `
                <div class="shap-lbl">${sh.feature} <span>(${sh.value})</span></div>
                <div class="shap-bar-wrap">
                    <div class="shap-bar ${isPos ? 'shap-pos' : 'shap-neg'}" style="width:0%" data-w="${pct}%"></div>
                </div>
                <div class="shap-val">${sh.impact.toFixed(3)}</div>
            `;
            shaps.appendChild(row);
            setTimeout(() => {
                const b = row.querySelector(".shap-bar");
                if (b) b.style.width = b.getAttribute("data-w");
            }, 100 + idx * 30);
        });
    }
    
    setTimeout(() => report.scrollIntoView({ behavior: "smooth", block: "nearest" }), 100);
}
