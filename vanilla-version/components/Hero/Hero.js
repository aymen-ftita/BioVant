/**
 * Hero Component Logic
 * Handles the animated live PSG signals on the hero section.
 */
function initHeroSignals() {
  const cfgs = [
    { id: 'sig-eeg', color: 'rgba(192,57,43,.75)', lw: 1.6, fn(t) { return Math.sin(t * .022) * 16 + Math.sin(t * .11) * 4 + Math.sin(t * .31) * 2 + (Math.random() - .5) * .5 }, speed: .9 },
    { id: 'sig-emg', color: 'rgba(29,78,216,.65)', lw: 1.2, fn(t) { return Math.sin(t * .08) * 3 + (Math.random() - .5) * 7 }, speed: 1.2 },
    { id: 'sig-eog', color: 'rgba(109,40,217,.6)', lw: 1.5, fn(t) { return Math.sin(t * .015) * 18 + Math.sin(t * .047) * 6 + (Math.random() - .5) * 1.5 }, speed: .6 },
  ];
  
  cfgs.forEach(cfg => {
    const canvas = document.getElementById(cfg.id);
    if (!canvas) return;
    const wrap = canvas.parentElement;
    let W, H, ctx, phase = 0;
    
    function resize() { 
      W = wrap.offsetWidth; 
      H = wrap.offsetHeight || 62; 
      canvas.width = W * devicePixelRatio; 
      canvas.height = H * devicePixelRatio; 
      canvas.style.width = W + "px"; 
      canvas.style.height = H + "px"; 
      ctx = canvas.getContext("2d"); 
      ctx.scale(devicePixelRatio, devicePixelRatio); 
    }
    
    resize(); 
    window.addEventListener("resize", resize);
    
    const NL = 2000, noise = new Float32Array(NL); 
    for (let i = 0; i < NL; i++) noise[i] = (Math.random() - .5) * 2;
    
    function sn(t) { 
      const i = Math.floor(t) % NL, j = (i + 1) % NL, f = t - Math.floor(t); 
      return noise[i] * (1 - f) + noise[j] * f; 
    }
    
    function draw() {
      if (!ctx) { requestAnimationFrame(draw); return; }
      ctx.clearRect(0, 0, W, H); 
      ctx.beginPath(); 
      ctx.lineWidth = cfg.lw; 
      ctx.strokeStyle = cfg.color; 
      ctx.lineJoin = "round";
      
      const mid = H / 2;
      for (let x = 0; x <= W; x += 1.5) { 
        const t = x * .8 + phase * cfg.speed * 60; 
        const y = mid + cfg.fn(t) + sn(t * .03) * 2; 
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); 
      }
      ctx.stroke(); 
      phase += .016; 
      requestAnimationFrame(draw);
    }
    draw();
  });
}

// Automatically start signals if the elements exist
if (document.readyState === 'complete' || document.readyState === 'interactive') {
    initHeroSignals();
} else {
    document.addEventListener('DOMContentLoaded', initHeroSignals);
}
