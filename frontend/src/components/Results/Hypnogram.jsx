import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { PlusCircle, X, MessageSquare, Clipboard, CheckCircle2 } from 'lucide-react';

const SC = {
  Wake: '#c0392b',
  NREM: '#1d4ed8',
  REM: '#047857',
  N1: '#d97706',
  N2: '#1d4ed8',
  N3: '#6d28d9'
};

const Hypnogram = ({ stages, classNames, spo2, apneaTimeline, activePsgId, onExport, onExportAnnotated }) => {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const [tooltip, setTooltip] = useState({ visible: false, text: '', x: 0, y: 0 });
  const [annotations, setAnnotations] = useState([]);
  const [modal, setModal] = useState({ visible: false, epochIndex: 0, note: '' });
  
  const hasExportedInitial = useRef(false);

  // Reset the initial export flag when the active PSG changes
  useEffect(() => {
    hasExportedInitial.current = false;
  }, [activePsgId]);

  // Load annotations on mount or when activePsgId changes
  useEffect(() => {
    if (!activePsgId) return;
    const token = localStorage.getItem('token');
    axios.get(`http://localhost:8000/psgs/${activePsgId}/annotations`, {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(res => {
        setAnnotations(res.data);
      })
      .catch(err => {
        console.warn('[Hypnogram Annotations] Failed to load:', err);
      });
  }, [activePsgId]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap || !stages || stages.length === 0) return;

    const order = classNames || ["Wake", "NREM", "REM"];
    
    // Determine overall height based on available data panels
    const hasApnea = apneaTimeline && apneaTimeline.length > 0;
    const hasSpo2 = spo2 && spo2.length > 0;
    
    const panelStagingH = 160;
    const panelApneaH = hasApnea ? 80 : 0;
    const panelSpo2H = hasSpo2 ? 100 : 0;
    
    const W = wrap.clientWidth;
    const H = panelStagingH + panelApneaH + panelSpo2H + 40; // total height
    
    canvas.style.width = W + "px"; 
    canvas.style.height = H + "px";
    
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(W * dpr); 
    canvas.height = Math.round(H * dpr);
    
    const ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0); 
    ctx.scale(dpr, dpr);
    
    const PAD = { top: 20, right: 15, bottom: 25, left: 55 };
    const CW = W - PAD.left - PAD.right;
    
    ctx.clearRect(0, 0, W, H);
    
    // Draw layout backgrounds & grid lines
    ctx.strokeStyle = "var(--border)"; 
    ctx.lineWidth = 1;
    
    // ─────────────────────────────────────────────
    // PANEL 1: Sleep Staging (Hypnogram)
    // ─────────────────────────────────────────────
    const stagingTop = PAD.top;
    const stagingH = panelStagingH - 20;
    const rowH = stagingH / order.length;
    
    // Draw staging grid rows
    for (let i = 0; i <= order.length; i++) { 
      const y = stagingTop + i * rowH; 
      ctx.beginPath(); 
      ctx.moveTo(PAD.left, y); 
      ctx.lineTo(W - PAD.right, y); 
      ctx.stroke(); 
    }
    
    // Draw staging Y-axis labels
    ctx.font = "600 10px var(--mono)"; 
    ctx.textAlign = "right";
    order.forEach((s, i) => { 
      ctx.fillStyle = SC[s] || SC['NREM']; 
      ctx.fillText(s, PAD.left - 10, stagingTop + i * rowH + rowH / 2 + 3); 
    });
    
    const segW = CW / stages.length;
    
    // Draw staging background blocks
    stages.forEach((st, i) => { 
      const yi = order.indexOf(st); 
      if (yi === -1) return;
      ctx.globalAlpha = 0.12; 
      ctx.fillStyle = SC[st] || SC['NREM']; 
      ctx.fillRect(PAD.left + i * segW, stagingTop + yi * rowH + 1, segW + 0.3, rowH - 2); 
    });
    
    // Draw hypnogram connection line
    ctx.globalAlpha = 1;
    ctx.beginPath(); 
    ctx.lineWidth = 2; 
    ctx.lineJoin = "round"; 
    ctx.strokeStyle = "rgba(192, 57, 43, 0.75)"; // deep red curve
    stages.forEach((st, i) => { 
      const x = PAD.left + i * segW + segW / 2;
      const yi = order.indexOf(st);
      const y = stagingTop + (yi !== -1 ? yi * rowH : 0) + rowH / 2; 
      if (i === 0) ctx.moveTo(x, y); 
      else ctx.lineTo(x, y); 
    });
    ctx.stroke();

    // ─────────────────────────────────────────────
    // PANEL 2: Apnea Events timeline
    // ─────────────────────────────────────────────
    let apneaTop = stagingTop + stagingH + 30;
    if (hasApnea) {
      // Draw grid
      ctx.strokeStyle = "var(--border)";
      ctx.beginPath();
      ctx.moveTo(PAD.left, apneaTop);
      ctx.lineTo(W - PAD.right, apneaTop);
      ctx.moveTo(PAD.left, apneaTop + 50);
      ctx.lineTo(W - PAD.right, apneaTop + 50);
      ctx.stroke();
      
      // Label
      ctx.fillStyle = "var(--text)";
      ctx.font = "bold 9px var(--sans)";
      ctx.textAlign = "right";
      ctx.fillText("APNÉES", PAD.left - 10, apneaTop + 20);
      ctx.font = "9px var(--mono)";
      ctx.fillStyle = "var(--text3)";
      ctx.fillText("(AHI/30m)", PAD.left - 10, apneaTop + 34);
      
      // Plot Apnea density blocks
      const apnW = CW / apneaTimeline.length;
      apneaTimeline.forEach((val, i) => {
        if (val > 0) {
          ctx.fillStyle = val > 15 ? 'rgba(220, 38, 38, 0.85)' : val > 5 ? 'rgba(234, 88, 12, 0.75)' : 'rgba(217, 119, 6, 0.65)';
          const barH = Math.min(val * 2.5, 48); // scale bar
          ctx.fillRect(PAD.left + i * apnW + 2, apneaTop + 50 - barH, apnW - 4, barH);
          
          // Render count above bar
          ctx.fillStyle = "var(--text2)";
          ctx.font = "700 8px var(--mono)";
          ctx.textAlign = "center";
          ctx.fillText(String(val), PAD.left + i * apnW + apnW / 2, apneaTop + 47 - barH);
        }
      });
    }

    // ─────────────────────────────────────────────
    // PANEL 3: SpO2 continuous signal
    // ─────────────────────────────────────────────
    let spo2Top = apneaTop + (hasApnea ? 70 : 0);
    if (hasSpo2) {
      ctx.strokeStyle = "var(--border)";
      ctx.beginPath();
      ctx.moveTo(PAD.left, spo2Top);
      ctx.lineTo(W - PAD.right, spo2Top);
      ctx.moveTo(PAD.left, spo2Top + 80);
      ctx.lineTo(W - PAD.right, spo2Top + 80);
      ctx.stroke();
      
      // SpO2 Y axis helper guides (100%, 90%, 80%)
      ctx.fillStyle = "var(--text3)";
      ctx.font = "9px var(--mono)";
      ctx.textAlign = "right";
      ctx.fillText("100%", PAD.left - 10, spo2Top + 8);
      ctx.fillText("90%", PAD.left - 10, spo2Top + 40);
      ctx.fillText("80%", PAD.left - 10, spo2Top + 72);
      
      ctx.strokeStyle = "rgba(255,255,255,0.03)";
      ctx.beginPath();
      ctx.moveTo(PAD.left, spo2Top + 40);
      ctx.lineTo(W - PAD.right, spo2Top + 40);
      ctx.stroke();
      
      // Draw SpO2 oxygenation line
      const ptW = CW / spo2.length;
      ctx.beginPath();
      ctx.lineWidth = 1.8;
      ctx.strokeStyle = '#059669'; // Green oxygen signal
      
      spo2.forEach((val, i) => {
        const x = PAD.left + i * ptW;
        // Map SpO2 values (100% is at top, 60% at bottom)
        const pct = (100 - val) / 40; // 0 to 1
        const y = spo2Top + pct * 80;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      
      // Highlight critical desaturation dips below 90% in red
      spo2.forEach((val, i) => {
        if (val < 90) {
          const x = PAD.left + i * ptW;
          const pct = (100 - val) / 40;
          const y = spo2Top + pct * 80;
          ctx.fillStyle = '#ef4444';
          ctx.beginPath();
          ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
          ctx.fill();
        }
      });
    }

    // ─────────────────────────────────────────────
    // COMMON: Draw vertical epoch separators
    // ─────────────────────────────────────────────
    ctx.strokeStyle = "var(--border)";
    ctx.lineWidth = 0.5;
    const gridCols = 8;
    for (let i = 0; i <= gridCols; i++) { 
      const x = PAD.left + CW * i / gridCols; 
      ctx.beginPath(); 
      ctx.moveTo(x, PAD.top); 
      ctx.lineTo(x, H - PAD.bottom); 
      ctx.stroke(); 
    }

    // ─────────────────────────────────────────────
    // PANEL 4: Clinical Annotations Markers
    // ─────────────────────────────────────────────
    annotations.forEach((anno) => {
      const idx = anno.epoch_index;
      if (idx < 0 || idx >= stages.length) return;
      const x = PAD.left + idx * segW + segW / 2;
      
      // Draw a yellow flag/marker on the top line of hypnogram
      ctx.fillStyle = '#fbbf24';
      ctx.beginPath();
      ctx.moveTo(x, stagingTop - 6);
      ctx.lineTo(x - 5, stagingTop - 12);
      ctx.lineTo(x + 5, stagingTop - 12);
      ctx.fill();
      
      ctx.strokeStyle = '#d97706';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, stagingTop);
      ctx.lineTo(x, H - PAD.bottom);
      ctx.stroke();
    });

    // ─────────────────────────────────────────────
    // EVENTS AND INTERACTIVE LISTENERS
    // ─────────────────────────────────────────────
    const handleMouseMove = (e) => {
      const r = canvas.getBoundingClientRect();
      const usable = r.width - PAD.left - PAD.right;
      const mouseX = e.clientX - r.left - PAD.left;
      const idxHover = Math.floor(mouseX / usable * stages.length);
      
      if (idxHover >= 0 && idxHover < stages.length) {
        const mins = (idxHover * 30) / 60;
        const h = Math.floor(mins / 60);
        const mm = Math.floor(mins % 60);
        const st = stages[idxHover];
        
        let txt = `${h}h${String(mm).padStart(2, "0")} · Époque ${idxHover + 1} · Stade ${st}`;
        
        // Append annotation note if present
        const matchAnno = annotations.find(a => a.epoch_index === idxHover);
        if (matchAnno) {
          txt += ` (Note: "${matchAnno.note}")`;
        }

        // Add SpO2 value if hover
        if (hasSpo2) {
          // 3 points per epoch
          const spIdx = idxHover * 3;
          if (spIdx < spo2.length) {
            txt += ` · SpO₂: ${spo2[spIdx]}%`;
          }
        }
        
        setTooltip({
          visible: true,
          text: txt,
          x: e.clientX + 14,
          y: e.clientY - 10
        });
      } else {
        setTooltip(prev => ({ ...prev, visible: false }));
      }
    };

    const handleMouseLeave = () => {
      setTooltip(prev => ({ ...prev, visible: false }));
    };

    const handleCanvasClick = (e) => {
      const r = canvas.getBoundingClientRect();
      const usable = r.width - PAD.left - PAD.right;
      const mouseX = e.clientX - r.left - PAD.left;
      const idxClicked = Math.floor(mouseX / usable * stages.length);
      
      if (idxClicked >= 0 && idxClicked < stages.length) {
        // Open Annotation dialog modal
        const matchAnno = annotations.find(a => a.epoch_index === idxClicked);
        setModal({
          visible: true,
          epochIndex: idxClicked,
          note: matchAnno ? matchAnno.note : ''
        });
      }
    };

    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseleave', handleMouseLeave);
    canvas.addEventListener('click', handleCanvasClick);

    if (onExport && !hasExportedInitial.current) {
      hasExportedInitial.current = true;
      setTimeout(() => {
        canvas.toBlob((blob) => {
          if (blob) onExport(blob);
        }, 'image/png');
      }, 1000); // Wait 1000ms to guarantee canvas has finished initial drawing
    }

    return () => {
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseleave', handleMouseLeave);
      canvas.removeEventListener('click', handleCanvasClick);
    };

  }, [stages, classNames, spo2, apneaTimeline, annotations, onExport]);

  const triggerAnnotatedExport = () => {
    const canvas = canvasRef.current;
    if (!canvas || !onExportAnnotated || !activePsgId) return;
    
    const btn = document.getElementById('btn-sync-annotated');
    let originalHTML = "";
    if (btn) {
      originalHTML = btn.innerHTML;
      btn.innerText = "Synchronisation...";
      btn.disabled = true;
    }
    
    canvas.toBlob((blob) => {
      if (blob) {
        onExportAnnotated(activePsgId, blob);
        setTimeout(() => {
          if (btn) {
            btn.innerHTML = `✓ Version Annotée Synchronisée`;
            btn.disabled = false;
            setTimeout(() => {
              btn.innerHTML = originalHTML;
            }, 3000);
          }
        }, 1200);
      }
    }, 'image/png');
  };

  const saveAnnotation = () => {
    if (!activePsgId) return;
    const token = localStorage.getItem('token');
    
    axios.post(`http://localhost:8000/psgs/${activePsgId}/annotations`, {
      epoch_index: modal.epochIndex,
      note: modal.note
    }, {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(res => {
        // Update local state
        setAnnotations(prev => {
          const filtered = prev.filter(a => a.epoch_index !== modal.epochIndex);
          return [...filtered, res.data];
        });
        setModal({ visible: false, epochIndex: 0, note: '' });
      })
      .catch(err => {
        alert("Erreur lors de l'enregistrement de la note.");
      });
  };

  const order = classNames || ["Wake", "NREM", "REM"];
  const tot = stages ? stages.length * 0.5 : 0;
  const timeLabels = [];
  for (let t = 0; t <= 8; t++) { 
    const m = Math.round(t * tot / 8);
    const h = Math.floor(m / 60);
    const mm = m % 60; 
    timeLabels.push(`${h}h${String(mm).padStart(2, "0")}`); 
  }

  return (
    <div className="hypno-wrap" ref={wrapRef} style={{ position: 'relative' }}>
      <div className="hypno-hdr" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <div className="hypno-title" style={{ fontWeight: 'bold', fontSize: '14px', color: 'var(--red)' }}>Graphiques Intégrés Polysomnographie</div>
          <span style={{ fontSize: '9px', background: 'rgba(251,191,36,0.1)', color: '#d97706', padding: '1px 6px', borderRadius: '4px', fontWeight: 600 }}>Cliquer pour annoter</span>
          {onExportAnnotated && (
            <button 
              id="btn-sync-annotated"
              className="btn-next" 
              onClick={triggerAnnotatedExport} 
              style={{ 
                height: '24px', 
                fontSize: '10px', 
                padding: '0 10px', 
                background: '#059669', 
                borderColor: '#047857',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                marginLeft: '8px',
                fontWeight: 600
              }}
            >
              <CheckCircle2 size={12} /> Synchroniser le Hypnogramme Annoté
            </button>
          )}
        </div>
        <div className="legend" style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          {order.map(s => (
            <div key={s} className="leg-item" style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '11px' }}>
              <div className="leg-dot" style={{ width: '8px', height: '8px', borderRadius: '50%', background: SC[s] || SC['NREM'] }}></div>
              {s}
            </div>
          ))}
          {spo2 && spo2.length > 0 && (
            <div className="leg-item" style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '11px' }}>
              <div className="leg-dot" style={{ width: '10px', height: '2px', background: '#059669' }}></div>
              SpO₂ (%)
            </div>
          )}
        </div>
      </div>
      
      <canvas ref={canvasRef} style={{ display: 'block', background: 'rgba(255,255,255,0.01)', border: '1px solid var(--border)', borderRadius: '8px', cursor: 'crosshair' }}></canvas>
      
      <div className="time-axis" style={{ display: 'flex', justifyContent: 'space-between', paddingLeft: '55px', paddingRight: '15px', fontSize: '10px', color: 'var(--text3)', marginTop: '5px', fontWeight: 600 }}>
        {timeLabels.map((lbl, i) => <span key={i}>{lbl}</span>)}
      </div>

      {/* Interactive Tooltip */}
      {tooltip.visible && (
        <div style={{
          position: 'fixed',
          top: tooltip.y,
          left: tooltip.x,
          background: 'rgba(15,23,42,0.95)',
          color: 'var(--text)',
          border: '1px solid var(--border)',
          boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
          padding: '6px 12px',
          borderRadius: '6px',
          fontSize: '11px',
          fontFamily: 'var(--serif)',
          pointerEvents: 'none',
          zIndex: 1000
        }}>
          {tooltip.text}
        </div>
      )}

      {/* Annotation Dialog Modal */}
      {modal.visible && (
        <div style={{
          position: 'absolute',
          top: '30px',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '280px',
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: '8px',
          padding: '14px',
          boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
          zIndex: 500,
          animation: 'fadeIn 0.2s ease-out'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <h4 style={{ fontSize: '12px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '6px' }}>
              <MessageSquare size={14} color="#fbbf24" /> Annoter Époque {modal.epochIndex + 1}
            </h4>
            <button 
              onClick={() => setModal({ visible: false, epochIndex: 0, note: '' })}
              style={{ background: 'none', border: 'none', color: 'var(--text3)', cursor: 'pointer' }}
            >
              <X size={14} />
            </button>
          </div>
          
          <textarea
            value={modal.note}
            onChange={(e) => setModal({ ...modal, note: e.target.value })}
            placeholder="Saisissez une note clinique (ex: Désaturation critique, mouvement...)"
            rows={3}
            style={{
              width: '100%',
              fontSize: '11px',
              padding: '6px',
              background: 'var(--bg)',
              color: 'var(--text)',
              border: '1px solid var(--border)',
              borderRadius: '4px',
              outline: 'none',
              resize: 'none',
              fontFamily: 'inherit',
              marginBottom: '10px'
            }}
          />
          
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
            <button
              onClick={() => setModal({ visible: false, epochIndex: 0, note: '' })}
              style={{ padding: '4px 10px', fontSize: '10px', background: 'rgba(255,255,255,0.05)', border: 'none', borderRadius: '4px', color: 'var(--text2)', cursor: 'pointer' }}
            >
              Annuler
            </button>
            <button
              onClick={saveAnnotation}
              disabled={!activePsgId}
              style={{ padding: '4px 10px', fontSize: '10px', background: '#d97706', border: 'none', borderRadius: '4px', color: 'white', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
            >
              <Clipboard size={10} /> Enregistrer
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Hypnogram;
