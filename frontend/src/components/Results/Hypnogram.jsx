import React, { useEffect, useRef, useState } from 'react';

const SC = {
  Wake: '#c0392b',
  NREM: '#1d4ed8',
  REM: '#047857',
  N1: '#d97706',
  N2: '#1d4ed8',
  N3: '#6d28d9'
};

const Hypnogram = ({ stages, classNames }) => {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const [tooltip, setTooltip] = useState({ visible: false, text: '', x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap || !stages || stages.length === 0) return;

    const order = classNames || ["Wake", "NREM", "REM"];
    
    // Set actual width
    const W = wrap.clientWidth;
    const H = 180;
    
    canvas.style.width = W + "px"; 
    canvas.style.height = H + "px";
    
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(W * dpr); 
    canvas.height = Math.round(H * dpr);
    
    const ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0); 
    ctx.scale(dpr, dpr);
    
    const PAD = { top: 18, right: 10, bottom: 8, left: 46 };
    const CW = W - PAD.left - PAD.right;
    const CH = H - PAD.top - PAD.bottom;
    const rowH = CH / order.length;
    
    ctx.clearRect(0, 0, W, H);
    
    // Draw background grid lines
    ctx.strokeStyle = "rgba(38,28,16,.07)"; 
    ctx.lineWidth = 1;
    for (let i = 0; i <= order.length; i++) { 
      const y = PAD.top + i * rowH; 
      ctx.beginPath(); 
      ctx.moveTo(PAD.left, y); 
      ctx.lineTo(W - PAD.right, y); 
      ctx.stroke(); 
    }
    for (let i = 0; i <= 8; i++) { 
      const x = PAD.left + CW * i / 8; 
      ctx.beginPath(); 
      ctx.moveTo(x, PAD.top); 
      ctx.lineTo(x, H - PAD.bottom); 
      ctx.stroke(); 
    }
    
    // Draw Y-axis labels
    ctx.font = "500 10px 'DM Mono',monospace"; 
    ctx.textAlign = "right";
    order.forEach((s, i) => { 
      ctx.fillStyle = SC[s] || SC['NREM']; 
      ctx.fillText(s, PAD.left - 8, PAD.top + i * rowH + rowH / 2 + 4); 
    });
    
    const segW = CW / stages.length;
    
    // Draw shading blocks
    stages.forEach((st, i) => { 
      const yi = order.indexOf(st); 
      ctx.globalAlpha = 0.65; 
      ctx.fillStyle = SC[st] || SC['NREM']; 
      ctx.fillRect(PAD.left + i * segW, PAD.top + yi * rowH + 1, segW + 0.5, rowH - 2); 
    });
    
    // Draw hypnogram connecting line
    ctx.globalAlpha = 1;
    ctx.beginPath(); 
    ctx.lineWidth = 2; 
    ctx.lineJoin = "round"; 
    ctx.strokeStyle = "rgba(38,28,16,.5)";
    stages.forEach((st, i) => { 
      const x = PAD.left + i * segW;
      const yi = order.indexOf(st);
      const y = PAD.top + yi * rowH + rowH / 2; 
      if (i === 0) ctx.moveTo(x, y); 
      else ctx.lineTo(x, y); 
    });
    ctx.stroke();

    const handleMouseMove = (e) => {
      const r = canvas.getBoundingClientRect();
      const usable = r.width - PAD.left - PAD.right;
      const mouseX = e.clientX - r.left - PAD.left;
      const idxHover = Math.floor(mouseX / usable * stages.length);
      
      if (idxHover >= 0 && idxHover < stages.length) {
        const mins = (idxHover * 30) / 60;
        const h = Math.floor(mins / 60);
        const mm = Math.floor(mins % 60);
        setTooltip({
          visible: true,
          text: `${h}h${String(mm).padStart(2, "0")} · Époque ${idxHover + 1} · ${stages[idxHover]}`,
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

    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseleave', handleMouseLeave);
    };

  }, [stages, classNames]);

  const order = classNames || ["Wake", "NREM", "REM"];
  const tot = stages ? stages.length * 0.5 : 0;
  const timeLabels = [];
  for (let t = 0; t <= 7; t++) { 
    const m = Math.round(t * tot / 7);
    const h = Math.floor(m / 60);
    const mm = m % 60; 
    timeLabels.push(`${h}h${String(mm).padStart(2, "0")}`); 
  }

  return (
    <div className="hypno-wrap" ref={wrapRef} style={{ position: 'relative' }}>
      <div className="hypno-hdr" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
        <div className="hypno-title" style={{ fontWeight: 'bold' }}>Architecture du Sommeil</div>
        <div className="legend" style={{ display: 'flex', gap: '15px' }}>
          {order.map(s => (
            <div key={s} className="leg-item" style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '11px' }}>
              <div className="leg-dot" style={{ width: '8px', height: '8px', borderRadius: '50%', background: SC[s] || SC['NREM'] }}></div>
              {s}
            </div>
          ))}
        </div>
      </div>
      <canvas ref={canvasRef} style={{ display: 'block', cursor: 'crosshair' }}></canvas>
      <div className="time-axis" style={{ display: 'flex', justifyContent: 'space-between', paddingLeft: '46px', paddingRight: '10px', fontSize: '10px', color: 'var(--text3)', marginTop: '5px' }}>
        {timeLabels.map((lbl, i) => <span key={i}>{lbl}</span>)}
      </div>

      {/* Tooltip */}
      {tooltip.visible && (
        <div style={{
          position: 'fixed',
          top: tooltip.y,
          left: tooltip.x,
          background: 'rgba(0,0,0,0.8)',
          color: 'white',
          padding: '4px 8px',
          borderRadius: '4px',
          fontSize: '11px',
          pointerEvents: 'none',
          zIndex: 1000
        }}>
          {tooltip.text}
        </div>
      )}
    </div>
  );
};

export default Hypnogram;
