const $ = (id) => document.getElementById(id);
// Recursos fijados a la misma versión para evitar incompatibilidades entre JS y WASM.
// Dependencias cargadas desde CDN para despliegue estático (sin node_modules).
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const MP_VERSION = '0.10.22';
const MP_CDNS = [
  {
    name: 'jsDelivr',
    module: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`,
    wasm: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`
  },
  {
    name: 'UNPKG',
    module: `https://unpkg.com/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`,
    wasm: `https://unpkg.com/@mediapipe/tasks-vision@${MP_VERSION}/wasm`
  }
];
const APP_VERSION = '1.0.4';
let poseTask = null, poseModule = null, poseLoading = null;
let fileUrl = null, stream = null, liveRaf = 0, liveStartedAt = 0, lastVideoTime = -1;
let processing = false, stopRequested = false, processedFrames = 0, validFrames = 0;
let lastDetectionTimestamp = -1, detectionTimestampOffset = 0;
let samples = [], currentMetrics = null, lastRawAngle = null, lastLandmarks = null;
let neutralRawAngle = 90, calibrationDescription = 'Referencia geométrica inicial: 90°';
let sourceLabel = '', sourceType = '', sessionHistory = [], chartRefs = {};
let toastTimeout = null;

function toast(message, type='') {
  const el = $('toast'); el.textContent = message; el.className = `toast show ${type}`;
  clearTimeout(toastTimeout); toastTimeout = setTimeout(() => el.className = 'toast', 3600);
}
function setStatus(message, active=false) {
  $('caseStatus').innerHTML = `<b>${message}</b>`;
  $('caseDot').style.background = active ? '#24b27d' : '#c4cbd5';
  $('caseDot').style.boxShadow = active ? '0 0 0 4px #24b27d18' : 'none';
}
function setProgress(value) {
  const pct = Math.max(0, Math.min(100, value));
  $('progressFill').style.width = `${pct}%`; $('progressLabel').textContent = `${Math.round(pct)}%`;
}
function nextDetectionTimestamp(preferred) {
  // MediaPipe exige timestamps crecientes incluso al cambiar de archivo a webcam.
  const raw = Number.isFinite(preferred) ? preferred : performance.now();
  if (raw + detectionTimestampOffset <= lastDetectionTimestamp) {
    detectionTimestampOffset = lastDetectionTimestamp + 1 - raw;
  }
  lastDetectionTimestamp = raw + detectionTimestampOffset;
  return lastDetectionTimestamp;
}
function showVideoStage() {
  $('emptyState').classList.add('hidden'); $('mediaVideo').classList.remove('hidden'); $('poseCanvas').classList.remove('hidden');
}
function setStageRatio(w,h) {
  if (w && h) $('mediaStage').style.aspectRatio = `${w} / ${h}`;
}
function fmt(v, digits=1) { return Number.isFinite(v) ? v.toFixed(digits).replace('.', ',') : '—'; }
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob); const a = document.createElement('a');
  a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}
function safeName(value) { return (value || 'registro').normalize('NFD').replace(/[\u0300-\u036f]/g,'').replace(/[^a-zA-Z0-9_-]+/g,'_').replace(/^_+|_+$/g,'').slice(0,40) || 'registro'; }

class MotionLabCanvasChart {
  constructor(canvas, config) {
    this.canvas = canvas;
    this.type = config.type || 'line';
    this.data = config.data || { labels: [], datasets: [] };
    this.options = config.options || {};
    this.update();
  }

  update() {
    const canvas = this.canvas;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(280, Math.round(rect.width || canvas.parentElement?.clientWidth || 520));
    const height = Math.max(150, Math.round(rect.height || canvas.parentElement?.clientHeight || 220));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const labels = this.data.labels || [];
    const datasets = this.data.datasets || [];
    const values = datasets.flatMap(ds => (ds.data || []).map(Number).filter(Number.isFinite));
    const left = 48, right = 14, top = 14, bottom = 38;
    const plotW = width - left - right;
    const plotH = height - top - bottom;
    if (plotW <= 0 || plotH <= 0) return;

    const yOpts = this.options.scales?.y || {};
    let min = Number.isFinite(yOpts.min) ? yOpts.min : (this.type === 'bar' || yOpts.beginAtZero ? 0 : Math.min(0, ...values));
    let max = Number.isFinite(yOpts.max) ? yOpts.max : Math.max(1, ...values);
    if (!Number.isFinite(min)) min = 0;
    if (!Number.isFinite(max)) max = 1;
    if (max <= min) max = min + 1;
    if (!Number.isFinite(yOpts.max) && this.type !== 'bar' && !yOpts.beginAtZero) {
      const pad = (max - min) * 0.08 || 1;
      min -= pad; max += pad;
    } else if (!Number.isFinite(yOpts.max)) {
      max = max * 1.12 || 1;
    }

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);
    ctx.font = '10px Arial, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.strokeStyle = '#edf1f5';
    ctx.fillStyle = '#8996a7';
    for (let i = 0; i <= 4; i++) {
      const y = top + (plotH * i / 4);
      const value = max - ((max - min) * i / 4);
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke();
      ctx.fillText(Number(value).toFixed(0), left - 8, y);
    }

    const tickCount = Math.min(labels.length, 6);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i < tickCount; i++) {
      const index = tickCount === 1 ? 0 : Math.round(i * (labels.length - 1) / (tickCount - 1));
      const x = left + (labels.length <= 1 ? plotW / 2 : plotW * index / (labels.length - 1));
      ctx.fillText(String(labels[index] ?? ''), x, top + plotH + 8);
    }

    if (this.type === 'bar') {
      const data = datasets[0]?.data || [];
      const slot = plotW / Math.max(1, data.length);
      const barW = Math.min(42, slot * 0.56);
      data.forEach((raw, i) => {
        const value = Number(raw);
        if (!Number.isFinite(value)) return;
        const x = left + slot * i + (slot - barW) / 2;
        const y = top + ((max - value) / (max - min)) * plotH;
        const zeroY = top + ((max - 0) / (max - min)) * plotH;
        ctx.fillStyle = Array.isArray(datasets[0].backgroundColor)
          ? datasets[0].backgroundColor[i % datasets[0].backgroundColor.length]
          : (datasets[0].backgroundColor || '#17a6a1');
        ctx.fillRect(x, y, barW, Math.max(1, zeroY - y));
      });
    } else {
      datasets.forEach(ds => {
        const data = ds.data || [];
        if (!data.length) return;
        ctx.beginPath();
        let started = false;
        data.forEach((raw, i) => {
          const value = Number(raw);
          if (!Number.isFinite(value)) { started = false; return; }
          const x = left + (data.length <= 1 ? plotW / 2 : plotW * i / (data.length - 1));
          const y = top + ((max - value) / (max - min)) * plotH;
          if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = ds.borderColor || '#17a6a1';
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.stroke();
        if (ds.fill && data.length > 1) {
          const first = Number(data[0]);
          const last = Number(data[data.length - 1]);
          const firstY = top + ((max - first) / (max - min)) * plotH;
          const lastY = top + ((max - last) / (max - min)) * plotH;
          ctx.lineTo(left + plotW, top + plotH);
          ctx.lineTo(left, top + plotH);
          ctx.closePath();
          ctx.globalAlpha = 0.10;
          ctx.fillStyle = ds.borderColor || '#17a6a1';
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      });
    }

    const yTitle = yOpts.title?.text;
    const xTitle = this.options.scales?.x?.title?.text;
    ctx.fillStyle = '#8996a7';
    ctx.font = '10px Arial, sans-serif';
    if (yTitle) {
      ctx.save(); ctx.translate(11, top + plotH / 2); ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center'; ctx.fillText(yTitle, 0, 0); ctx.restore();
    }
    if (xTitle) {
      ctx.textAlign = 'center'; ctx.fillText(xTitle, left + plotW / 2, height - 11);
    }
    if (!values.length) {
      ctx.fillStyle = '#9aa7b8'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('Esperando datos del análisis…', left + plotW / 2, top + plotH / 2);
    }
  }

  destroy() { /* No se requiere liberar recursos para el renderizador Canvas */ }
}

function initCharts() {
  const ChartEngine = window.Chart || MotionLabCanvasChart;
  const common = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { intersect: false, mode: 'index' },
    plugins: { legend: { labels: { usePointStyle: true, boxWidth: 7, font: { family: 'DM Sans', size: 10 }, color: '#7c8ba0' } } },
    scales: {
      x: { ticks: { color: '#96a2b1', maxTicksLimit: 8, font: { size: 9 } }, grid: { display: false }, title: { display: true, text: 'Tiempo (s)', color: '#8996a7', font: { size: 9 } } },
      y: { ticks: { color: '#96a2b1', font: { size: 9 } }, grid: { color: '#edf1f5' }, title: { display: true, text: 'Ángulo (°)', color: '#8996a7', font: { size: 9 } } }
    }
  };
  const make = (id, type, data, options) => {
    const canvas = $(id);
    if (!canvas) { console.warn(`No se encontró el canvas #${id}`); return null; }
    return new ChartEngine(canvas, { type, data, options });
  };

  chartRefs.angle = make('angleChart', 'line', {
    labels: [], datasets: [{ label: 'Ángulo tobillo', data: [], borderColor: '#17a6a1', backgroundColor: '#17a6a11b', fill: true, tension: .25, pointRadius: 0, borderWidth: 2 }]
  }, typeof structuredClone === 'function' ? structuredClone(common) : common);

  const rangeOptions = {
    ...common,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { display: false }, ticks: { color: '#73839a', font: { size: 9 } } },
      y: { beginAtZero: true, grid: { color: '#edf1f5' }, ticks: { color: '#96a2b1', font: { size: 9 } }, title: { display: true, text: 'Grados (°)', color: '#8996a7', font: { size: 9 } } }
    }
  };
  chartRefs.range = make('rangeChart', 'bar', {
    labels: ['Pico dorsi', 'Pico planti', 'ROM total'],
    datasets: [{ label: 'Grados (°)', data: [0, 0, 0], backgroundColor: ['#17a6a1', '#e6a15b', '#557cda'], borderRadius: 7, maxBarThickness: 35 }]
  }, rangeOptions);

  chartRefs.velocity = make('velocityChart', 'line', {
    labels: [], datasets: [{ label: 'Velocidad angular', data: [], borderColor: '#557cda', backgroundColor: '#557cda18', fill: true, tension: .25, pointRadius: 0, borderWidth: 1.8 }]
  }, typeof structuredClone === 'function' ? structuredClone(common) : common);

  const confOptions = typeof structuredClone === 'function' ? structuredClone(common) : JSON.parse(JSON.stringify(common));
  confOptions.scales.y.min = 0;
  confOptions.scales.y.max = 100;
  confOptions.scales.y.title.text = 'Confianza (%)';
  confOptions.scales.x.title.text = 'Tiempo (s)';
  chartRefs.confidence = make('confidenceChart', 'line', {
    labels: [], datasets: [{ label: 'Confianza', data: [], borderColor: '#d99b53', backgroundColor: '#d99b5318', fill: true, tension: .2, pointRadius: 0, borderWidth: 1.8 }]
  }, confOptions);

  if (!window.Chart) console.info('MotionLab usa su motor Canvas integrado para dibujar los gráficos.');
}

async function initPose() {
  if (poseTask) return poseTask;
  if (poseLoading) return poseLoading;

  poseLoading = (async () => {
    $('modelState').textContent = 'Conectando con MediaPipe…';
    const errors = [];

    try {
      for (const cdn of MP_CDNS) {
        try {
          $('modelState').textContent = `Cargando MediaPipe desde ${cdn.name}…`;
          console.info(`MotionLab: intentando MediaPipe desde ${cdn.name}`);
          const module = await import(cdn.module);
          const vision = await module.FilesetResolver.forVisionTasks(cdn.wasm);
          let lastError = null;

          for (const delegate of ['GPU', 'CPU']) {
            try {
              const task = await module.PoseLandmarker.createFromOptions(vision, {
                baseOptions: { modelAssetPath: MODEL_URL, delegate },
                runningMode: 'VIDEO', numPoses: 1,
                minPoseDetectionConfidence: 0.5,
                minPosePresenceConfidence: 0.5,
                minTrackingConfidence: 0.5
              });
              poseModule = module;
              poseTask = task;
              $('modelState').textContent = `Motor listo · ${delegate} · ${cdn.name}`;
              return poseTask;
            } catch (error) {
              lastError = error;
              console.warn(`MediaPipe ${delegate} falló en ${cdn.name}:`, error);
            }
          }
          throw lastError || new Error('No se pudo crear PoseLandmarker.');
        } catch (error) {
          errors.push(`${cdn.name}: ${error.message || error}`);
          console.warn(`No se pudo cargar MediaPipe desde ${cdn.name}:`, error);
        }
      }
      throw new Error('No se pudo cargar MediaPipe. Comprueba la conexión a internet. ' + errors.join(' | '));
    } catch (error) {
      console.error('Error inicializando MediaPipe:', error);
      $('modelState').textContent = 'Motor no disponible';
      throw error;
    } finally {
      poseLoading = null;
    }
  })();
  return poseLoading;
}

function angleBetweenAtAnkle(knee, ankle, toe, width, height) {
  // MediaPipe entrega x e y normalizados por separado. Para calcular geometría
  // real en una imagen no cuadrada, primero convertimos ambas coordenadas a píxeles.
  const toPixels = p => ({x:p.x * width, y:p.y * height});
  const kneePx = toPixels(knee), anklePx = toPixels(ankle), toePx = toPixels(toe);
  const a = {x:kneePx.x-anklePx.x,y:kneePx.y-anklePx.y};
  const b = {x:toePx.x-anklePx.x,y:toePx.y-anklePx.y};
  const la = Math.hypot(a.x,a.y), lb = Math.hypot(b.x,b.y);
  if (la < 3 || lb < 3) return null;
  const dot = a.x*b.x + a.y*b.y;
  const cosine = Math.max(-1,Math.min(1,dot/(la*lb)));
  return Math.acos(cosine) * 180 / Math.PI;
}
function extractAnkle(landmarks) {
  const side = $('sideSelect').value;
  const ids = side === 'left' ? {knee:25,ankle:27,heel:29,toe:31} : {knee:26,ankle:28,heel:30,toe:32};
  const points = Object.fromEntries(Object.entries(ids).map(([key,index])=>[key,landmarks[index]]));
  if (Object.values(points).some(p=>!p)) return null;
  const threshold = Number($('visibilitySelect').value);
  // Rodilla, tobillo y punta definen el ángulo. El talón se dibuja como referencia,
  // pero no invalida la medición cuando queda parcialmente oculto.
  const required = [points.knee,points.ankle,points.toe];
  const requiredVisibility = required.map(p=>typeof p.visibility === 'number' ? p.visibility : 0);
  const confidence = requiredVisibility.reduce((a,b)=>a+b,0)/requiredVisibility.length;
  if (Math.min(...requiredVisibility) < threshold) return {valid:false,confidence,points};
  const video = $('mediaVideo');
  const width = video.videoWidth || 1, height = video.videoHeight || 1;
  const raw = angleBetweenAtAnkle(points.knee,points.ankle,points.toe,width,height);
  if (!Number.isFinite(raw)) return {valid:false,confidence,points};
  return {valid:true,raw,angle:neutralRawAngle-raw,confidence,points};
}
function drawLandmarks(points) {
  const video = $('mediaVideo'), canvas = $('poseCanvas');
  if (!video.videoWidth || !video.videoHeight) return;
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) { canvas.width=video.videoWidth; canvas.height=video.videoHeight; }
  const ctx = canvas.getContext('2d'); ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!points) return;
  const keys=['knee','ankle','heel','toe'];
  ctx.lineWidth=Math.max(3,canvas.width*.003);ctx.strokeStyle='#61e0d0';ctx.fillStyle='#f5fbff';ctx.lineCap='round';ctx.lineJoin='round';
  const xy=keys.map(k=>({x:points[k].x*canvas.width,y:points[k].y*canvas.height}));
  ctx.beginPath();ctx.moveTo(xy[0].x,xy[0].y);ctx.lineTo(xy[1].x,xy[1].y);ctx.lineTo(xy[3].x,xy[3].y);ctx.stroke();
  ctx.beginPath();ctx.moveTo(xy[1].x,xy[1].y);ctx.lineTo(xy[2].x,xy[2].y);ctx.strokeStyle='#e6a45f';ctx.stroke();
  xy.forEach((p,i)=>{ctx.beginPath();ctx.arc(p.x,p.y,Math.max(5,canvas.width*.006),0,Math.PI*2);ctx.fillStyle=i===1?'#52d2c7':'#f4f8fc';ctx.fill();ctx.lineWidth=2;ctx.strokeStyle='#17334e';ctx.stroke();});
}
function processPoseResult(result, timestamp, draw=true) {
  processedFrames++;
  const pose = result?.landmarks?.[0];
  if (!pose) { if(draw) drawLandmarks(null); return false; }
  const found=extractAnkle(pose); lastLandmarks=pose;
  if (found?.raw !== undefined) lastRawAngle=found.raw;
  if (draw) drawLandmarks(found?.points || null);
  if (!found || !found.valid) return false;
  validFrames++;
  samples.push({t:Number(timestamp),rawAngle:found.raw,angle:found.angle,confidence:found.confidence,valid:true});
  if (samples.length > 5000) samples.shift();
  return true;
}
function computeMetrics() {
  if (!samples.length) return null;
  const ordered=[...samples].sort((a,b)=>a.t-b.t);
  const angles=ordered.map(s=>s.angle);
  const max=Math.max(...angles), min=Math.min(...angles);
  const velocities=[];
  for(let i=1;i<ordered.length;i++){
    const dt=ordered[i].t-ordered[i-1].t;
    if(dt>0.0001) velocities.push({t:ordered[i].t,value:(ordered[i].angle-ordered[i-1].angle)/dt});
  }
  const peakVelocity=velocities.length?Math.max(...velocities.map(v=>Math.abs(v.value))):0;
  const meanConfidence=ordered.reduce((sum,s)=>sum+s.confidence,0)/ordered.length;
  let cycles=0,lastCycleTime=-Infinity;
  for(let i=1;i<angles.length-1;i++){
    if(angles[i]>angles[i-1] && angles[i]>=angles[i+1] && angles[i]>3 && ordered[i].t-lastCycleTime>.55){cycles++;lastCycleTime=ordered[i].t;}
  }
  const span=ordered.length>1?ordered[ordered.length-1].t-ordered[0].t:0;
  return {peakDorsi:Math.max(0,max),peakPlantar:Math.max(0,-min),rom:max-min,peakVelocity,meanConfidence,coverage:processedFrames?validFrames/processedFrames*100:0,validFrames,processedFrames,cycles,duration:span,minimum:min,maximum:max,velocities,ordered};
}
function chartDownsample(items, maxPoints=500) {
  const step=Math.max(1,Math.ceil(items.length/maxPoints)); return items.filter((_,i)=>i%step===0 || i===items.length-1);
}
function updateCharts(metrics) {
  if (!chartRefs.angle) return;
  const points=chartDownsample(metrics.ordered);
  const labels=points.map(s=>fmt(s.t,2));
  chartRefs.angle.data.labels=labels; chartRefs.angle.data.datasets[0].data=points.map(s=>Number(s.angle.toFixed(2))); chartRefs.angle.update('none');
  chartRefs.range.data.datasets[0].data=[metrics.peakDorsi,metrics.peakPlantar,metrics.rom];chartRefs.range.update('none');
  const vpoints=chartDownsample(metrics.velocities);
  chartRefs.velocity.data.labels=vpoints.map(s=>fmt(s.t,2));chartRefs.velocity.data.datasets[0].data=vpoints.map(s=>Number(s.value.toFixed(2)));chartRefs.velocity.update('none');
  chartRefs.confidence.data.labels=labels;chartRefs.confidence.data.datasets[0].data=points.map(s=>Number((s.confidence*100).toFixed(1)));chartRefs.confidence.update('none');
}
function updateNarrative(metrics) {
  if (!metrics) {
    $('qualityScore').textContent='—';$('qualityTitle').textContent='Esperando captura';$('qualitySubtitle').textContent='La calidad de detección aparecerá al procesar un video o iniciar la cámara.';
    $('patternFinding').textContent='Aún no hay datos suficientes para describir el movimiento.';$('kpiFinding').textContent='Se mostrarán pico de dorsiflexión, plantiflexión, ROM y velocidad angular.';$('qualityFinding').textContent='Comprueba que el pie completo se mantenga dentro del encuadre.';
    $('qualityRing').style.background='conic-gradient(#53cec4 0deg 0deg,#ffffff1b 0deg 360deg)';return;
  }
  const q=Math.round(metrics.coverage*.65+metrics.meanConfidence*100*.35);
  $('qualityScore').textContent=`${q}`;$('qualityRing').style.background=`conic-gradient(#53cec4 0deg ${q*3.6}deg,#ffffff1b ${q*3.6}deg 360deg)`;
  $('qualityTitle').textContent=q>=85?'Captura consistente':q>=65?'Captura utilizable':'Captura con limitaciones';
  $('qualitySubtitle').textContent=`${metrics.validFrames} muestras válidas de ${metrics.processedFrames} fotogramas procesados.`;
  $('patternFinding').textContent=`Dorsiflexión máxima ${fmt(metrics.peakDorsi)}°, plantiflexión máxima ${fmt(metrics.peakPlantar)}° y ROM observado ${fmt(metrics.rom)}°.`;
  $('kpiFinding').textContent=`Velocidad angular pico ${fmt(metrics.peakVelocity)}°/s; ${metrics.cycles} pico(s) de dorsiflexión detectado(s) como aproximación de ciclos.`;
  $('qualityFinding').textContent=metrics.coverage<65?'Cobertura baja: revisa visibilidad, iluminación, ángulo de cámara y lado seleccionado.':metrics.coverage<85?'Algunas muestras se descartaron por visibilidad insuficiente; interpreta con cautela.':'Buena continuidad de detección en este registro; verifica visualmente los landmarks.';
}
function renderMetrics() {
  const m=currentMetrics;
  $('metricDorsi').innerHTML=m?`${fmt(m.peakDorsi)} <small>°</small>`:'— <small>°</small>';
  $('metricPlantar').innerHTML=m?`${fmt(m.peakPlantar)} <small>°</small>`:'— <small>°</small>';
  $('metricRom').innerHTML=m?`${fmt(m.rom)} <small>°</small>`:'— <small>°</small>';
  $('metricVelocity').innerHTML=m?`${fmt(m.peakVelocity)} <small>°/s</small>`:'— <small>°/s</small>';
  $('overviewSamples').textContent=m?m.validFrames.toLocaleString('es-CL'):'—';
  $('overviewCoverage').textContent=m?`${fmt(m.coverage,0)}%`:'—';
  updateNarrative(m);
  if(m) updateCharts(m); else if(chartRefs.angle){Object.values(chartRefs).forEach(c=>{c.data.labels=[];c.data.datasets.forEach(d=>d.data=[]);c.update('none');});}
}
function finalizeAnalysis() {
  currentMetrics=computeMetrics(); renderMetrics();
  $('overviewRecords').textContent=String(sessionHistory.length+1);
  if(!currentMetrics){setStatus('Sin landmarks suficientes',false);toast('No se detectó el tobillo con calidad suficiente. Revisa el encuadre y vuelve a intentar.','error');return;}
  sessionHistory.push({name:sourceLabel||'Captura webcam',source:sourceType,createdAt:new Date().toISOString(),metrics:{peakDorsi:currentMetrics.peakDorsi,peakPlantar:currentMetrics.peakPlantar,rom:currentMetrics.rom,peakVelocity:currentMetrics.peakVelocity,coverage:currentMetrics.coverage}});
  setStatus('Análisis completado',true);setProgress(100);
  $('overviewRecords').textContent=String(sessionHistory.length);
  toast(`Análisis completado: ${currentMetrics.validFrames} muestras válidas.`, 'success');
}

async function loadVideoFile(file) {
  if (!file) return;
  if (!(file.type.startsWith('video/') || /\.(mp4|mov|m4v|webm|ogv)$/i.test(file.name))) {toast('Archivo no compatible. Selecciona un video MP4, MOV o WebM.','error');return;}
  if (file.size>500*1024*1024) {toast('El archivo supera 500 MB. Usa un video más liviano o recórtalo.','error');return;}
  stopWebcam(false);clearCurrentData(true);
  if(fileUrl) URL.revokeObjectURL(fileUrl);
  fileUrl=URL.createObjectURL(file);const video=$('mediaVideo');video.srcObject=null;video.src=fileUrl;video.controls=true;video.muted=false;video.load();
  sourceLabel=file.name;sourceType='video';$('mediaFilename').innerHTML=`<strong>Fuente:</strong> ${escapeHtml(file.name)}`;$('mediaDuration').textContent='Duración: cargando…';
  video.onloadedmetadata=()=>{
    if(!Number.isFinite(video.duration)||video.duration===Infinity){toast('No fue posible leer la duración del video.','error');return;}
    setStageRatio(video.videoWidth,video.videoHeight);showVideoStage();$('mediaDuration').textContent=`Duración: ${fmt(video.duration,1)} s`;
    $('mediaFps').textContent=`Muestreo: ${$('fpsSelect').value} FPS`;$('analyzeBtn').disabled=false;$('cameraStart').disabled=false;$('calibrateBtn').disabled=false;
    setStatus('Video cargado',true);toast('Video cargado. Pausa en posición neutra y calibra antes de analizar.','success');
  };
  video.onerror=()=>{toast('El navegador no puede decodificar este video. Prueba MP4 H.264 o WebM.','error');$('mediaDuration').textContent='Error de lectura';};
}
function escapeHtml(str){return String(str).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function clearCurrentData(resetCalibration=true) {
  stopRequested=true;samples=[];currentMetrics=null;processedFrames=0;validFrames=0;lastRawAngle=null;lastLandmarks=null;
  if(resetCalibration){neutralRawAngle=90;calibrationDescription='Referencia geométrica inicial: 90°';$('calibrationState').textContent=calibrationDescription;}
  setProgress(0);renderMetrics();$('overviewSamples').textContent='—';$('overviewCoverage').textContent='—';$('overviewRecords').textContent=String(sessionHistory.length);
}
function resetApp() {
  if(processing){stopRequested=true;processing=false;}
  stopWebcam(false);clearCurrentData();
  const video=$('mediaVideo');video.pause();video.removeAttribute('src');video.srcObject=null;video.load();video.classList.add('hidden');$('poseCanvas').classList.add('hidden');$('emptyState').classList.remove('hidden');
  if(fileUrl){URL.revokeObjectURL(fileUrl);fileUrl=null;}
  sourceLabel='';sourceType='';$('mediaFilename').innerHTML='<strong>Fuente:</strong> no seleccionada';$('mediaDuration').textContent='Duración: —';$('mediaFps').textContent='Muestreo: —';$('analyzeBtn').disabled=true;$('calibrateBtn').disabled=true;$('stopBtn').disabled=true;
  setStatus('Sin registro activo',false);toast('Registro reiniciado.');
}
function waitForVideoData(video, timeoutMs=10000) {
  if(video.readyState>=HTMLMediaElement.HAVE_CURRENT_DATA) return Promise.resolve();
  return new Promise((resolve,reject)=>{
    let timer;
    const cleanup=()=>{clearTimeout(timer);video.removeEventListener('loadeddata',onReady);video.removeEventListener('error',onError);};
    const onReady=()=>{cleanup();resolve();};
    const onError=()=>{cleanup();reject(new Error('El navegador no pudo decodificar un fotograma del video. Prueba MP4 H.264 o WebM.'));};
    video.addEventListener('loadeddata',onReady,{once:true});
    video.addEventListener('error',onError,{once:true});
    timer=setTimeout(()=>{cleanup();reject(new Error('El video tardó demasiado en preparar sus fotogramas.'));},timeoutMs);
  });
}
async function seekTo(video,time) {
  if(video.readyState<HTMLMediaElement.HAVE_CURRENT_DATA) await waitForVideoData(video);
  if(Math.abs(video.currentTime-time)<0.015 && video.readyState>=HTMLMediaElement.HAVE_CURRENT_DATA) return;
  await new Promise((resolve,reject)=>{
    let timer;
    const cleanup=()=>{clearTimeout(timer);video.removeEventListener('seeked',onSeeked);video.removeEventListener('error',onError);};
    const onSeeked=()=>{cleanup();resolve();};
    const onError=()=>{cleanup();reject(new Error('No fue posible posicionarse en el fotograma solicitado.'));};
    video.addEventListener('seeked',onSeeked,{once:true});
    video.addEventListener('error',onError,{once:true});
    timer=setTimeout(()=>{cleanup();reject(new Error(`El video no respondió al salto en ${fmt(time,2)} s. Intenta convertirlo a MP4 H.264.`));},10000);
    try{video.currentTime=time;}catch(error){cleanup();reject(error);}
  });
  if(video.readyState<HTMLMediaElement.HAVE_CURRENT_DATA) await waitForVideoData(video);
  // Da tiempo al navegador para presentar el fotograma recién decodificado antes
  // de entregarlo al modelo de visión. requestVideoFrameCallback no existe en todos.
  if(typeof video.requestVideoFrameCallback==='function') {
    await Promise.race([
      new Promise(resolve=>video.requestVideoFrameCallback(()=>resolve())),
      new Promise(resolve=>setTimeout(resolve,350))
    ]);
  } else {
    await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
  }
}
async function analyzeVideo() {
  const video=$('mediaVideo');
  if(!video.src||!Number.isFinite(video.duration)){toast('Primero carga un video válido.','error');return;}
  if(stream){toast('Detén la webcam antes de analizar un archivo.','error');return;}
  processing=true;stopRequested=false;processedFrames=0;validFrames=0;samples=[];currentMetrics=null;renderMetrics();
  $('analyzeBtn').disabled=true;$('cameraStart').disabled=true;$('stopBtn').disabled=false;setStatus('Procesando video…',true);setProgress(0);
  try {
    await initPose();
    const fps=Number($('fpsSelect').value);$('mediaFps').textContent=`Muestreo: ${fps} FPS`;
    const duration=video.duration, step=1/fps;let i=0;
    video.pause();
    await waitForVideoData(video);
    for(let t=0;t<duration;t+=step){
      if(stopRequested) break;
      const sampleTime=Math.min(t,Math.max(0,duration-.001));
      await seekTo(video,sampleTime);
      if(stopRequested) break;
      const result=poseTask.detectForVideo(video,nextDetectionTimestamp(sampleTime*1000));
      processPoseResult(result,sampleTime,true);i++;
      if(i%4===0){setProgress(sampleTime/duration*100);await new Promise(requestAnimationFrame);}
    }
    if(!stopRequested)setProgress(100);
    if(processedFrames>0) finalizeAnalysis();
    if(stopRequested&&samples.length){finalizeAnalysis();toast('Procesamiento detenido. Se conservaron las muestras analizadas.');}
    if(!samples.length) {setStatus('Sin detección válida',false);toast('No se detectaron suficientes puntos del tobillo. Revisa vista lateral, encuadre y confianza mínima.','error');}
  } catch(error) {console.error(error);toast(error.message||'Ocurrió un error durante el análisis.','error');setStatus('Error de procesamiento',false);}
  finally {processing=false;stopRequested=false;$('analyzeBtn').disabled=false;$('cameraStart').disabled=false;$('stopBtn').disabled=true;}
}

async function startWebcam() {
  if(processing){toast('Detén el procesamiento actual antes de iniciar webcam.','error');return;}
  if(stream){toast('La webcam ya está activa. Detén la captura antes de volver a iniciarla.');return;}
  const startButton=$('cameraStart');
  startButton.disabled=true;
  let candidateStream=null;
  try {
    if(!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      throw new Error('El navegador bloquea la webcam en esta dirección. Abre la app desde localhost o desde un sitio HTTPS; no uses file:// ni una dirección HTTP pública.');
    }
    // `ideal` permite utilizar webcams integradas que no declaran cámara trasera.
    candidateStream=await navigator.mediaDevices.getUserMedia({
      audio:false,
      video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30,max:30}}
    });
    stream=candidateStream;
    clearCurrentData(true);stopRequested=false;
    sourceLabel='Captura webcam';sourceType='webcam';
    const video=$('mediaVideo');
    video.pause();video.removeAttribute('src');video.srcObject=stream;video.muted=true;video.playsInline=true;video.controls=false;
    await video.play();
    await waitForVideoData(video);
    showVideoStage();setStageRatio(video.videoWidth||16,video.videoHeight||9);
    $('mediaFilename').innerHTML='<strong>Fuente:</strong> cámara en directo';
    $('mediaDuration').textContent='Duración: captura en vivo';
    $('mediaFps').textContent='Muestreo: adaptativo';
    // La vista previa ya está disponible mientras se inicializa el modelo.
    await initPose();
    $('analyzeBtn').disabled=true;$('stopBtn').disabled=false;$('calibrateBtn').disabled=false;
    liveStartedAt=performance.now();lastVideoTime=-1;setStatus('Webcam en directo',true);
    toast('Webcam activa. Mantén rodilla, tobillo, talón y punta visibles; calibra la posición neutra antes de evaluar.','success');
    const loop=()=>{
      if(!stream)return;
      if(video.readyState>=HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime!==lastVideoTime){
        lastVideoTime=video.currentTime;
        try{
          const now=performance.now();
          const result=poseTask.detectForVideo(video,nextDetectionTimestamp(now));
          processPoseResult(result,(now-liveStartedAt)/1000,true);
          if(processedFrames%4===0){currentMetrics=computeMetrics();renderMetrics();}
        }catch(error){console.warn('Error procesando fotograma de webcam:',error);}
      }
      liveRaf=requestAnimationFrame(loop);
    };
    loop();
  } catch(error) {
    if(candidateStream) candidateStream.getTracks().forEach(track=>track.stop());
    if(stream===candidateStream) stream=null;
    const video=$('mediaVideo');
    if(video.srcObject){video.pause();video.srcObject=null;}
    $('stopBtn').disabled=true;$('calibrateBtn').disabled=true;$('analyzeBtn').disabled=!video.src;
    if(sourceType==='webcam') {
      sourceType='';sourceLabel='';
      $('mediaFilename').innerHTML='<strong>Fuente:</strong> no seleccionada';
      $('mediaDuration').textContent='Duración: —';$('mediaFps').textContent='Muestreo: —';
      video.classList.add('hidden');$('poseCanvas').classList.add('hidden');$('emptyState').classList.remove('hidden');
    }
    startButton.disabled=false;
    const msg=error.name==='NotAllowedError'
      ?'Permiso de cámara denegado. Autoriza el acceso desde el candado de la barra del navegador y vuelve a intentarlo.'
      :error.name==='NotFoundError'
        ?'No se encontró una cámara disponible. Conecta una webcam o utiliza un video grabado.'
        :error.name==='NotReadableError'
          ?'La cámara está ocupada por otra aplicación. Ciérrala y vuelve a intentar.'
          :error.message||'No fue posible iniciar la cámara.';
    console.error('Error iniciando webcam:',error);
    toast(msg,'error');setStatus('Cámara no disponible',false);
  }
}
function stopWebcam(showMessage=true) {
  if(liveRaf)cancelAnimationFrame(liveRaf);liveRaf=0;
  const wasWebcam=sourceType==='webcam' || !!stream;
  if(stream){stream.getTracks().forEach(track=>track.stop());stream=null;}
  const video=$('mediaVideo');
  if(video.srcObject){video.pause();video.srcObject=null;}
  if(wasWebcam){
    $('stopBtn').disabled=true;$('cameraStart').disabled=false;$('analyzeBtn').disabled=true;$('calibrateBtn').disabled=true;
    if(samples.length){finalizeAnalysis();}
    else if(showMessage){setStatus('Webcam detenida',false);toast('Webcam detenida. No se detectaron muestras válidas.','error');}
  }
}
async function calibrateNeutral() {
  const liveCapture=!!stream;
  try {
    await initPose();
    const video=$('mediaVideo');
    if(!liveCapture) {
      if(!video.src) throw new Error('Primero carga un video o inicia la webcam para calibrar.');
      video.pause();
      await waitForVideoData(video);
      // Evita reutilizar un ángulo antiguo si el fotograma actual no detecta el pie.
      lastRawAngle=null;
      const result=poseTask.detectForVideo(video,nextDetectionTimestamp(performance.now()));
      const pose=result?.landmarks?.[0];
      const detected=pose?extractAnkle(pose):null;
      if(detected?.valid) lastRawAngle=detected.raw;
    }
  } catch(error) {
    toast(error.message||'No fue posible inicializar el modelo para calibrar.','error');
    return;
  }
  if(!Number.isFinite(lastRawAngle)) {
    toast('No hay una detección válida del tobillo para calibrar. Coloca rodilla, tobillo y punta del pie visibles y vuelve a intentarlo.','error');
    return;
  }
  neutralRawAngle=lastRawAngle;
  calibrationDescription=`Calibrado · ${fmt(neutralRawAngle)}° geométricos`;
  $('calibrationState').textContent=calibrationDescription;
  if(liveCapture) {
    // La nueva referencia debe aplicarse a toda la serie: inicia una medición limpia.
    samples=[];currentMetrics=null;processedFrames=0;validFrames=0;
    liveStartedAt=performance.now();lastVideoTime=-1;stopRequested=false;
    setProgress(0);renderMetrics();
    toast('Posición neutra calibrada. Comienza una nueva medición en vivo.','success');
  } else {
    toast('Posición neutra calibrada. Pulsa “Analizar video” para aplicar esta referencia al registro.','success');
  }
}
function loadDemo() {
  stopWebcam(false);clearCurrentData(false);sourceLabel='Caso demostrativo sintético';sourceType='demo';
  processedFrames=81;validFrames=81;
  for(let i=0;i<=80;i++){
    const t=i*.1;const angle=18*Math.sin(2*Math.PI*.5*t);const raw=neutralRawAngle-angle;
    const confidence=.94+.035*Math.sin(i*.28);
    samples.push({t,rawAngle:raw,angle,confidence,valid:true});
  }
  finalizeAnalysis();setStatus('Caso de demostración',true);$('mediaFilename').innerHTML='<strong>Fuente:</strong> caso sintético de demostración';$('mediaDuration').textContent='Duración: 8,0 s';$('mediaFps').textContent='Muestreo: 10 FPS';
  toast('Caso sintético cargado. No representa a un paciente ni datos clínicos reales.','success');
}

const validation = [];
for(let record=1;record<=2;record++) for(const metric of [{key:'peakDorsi',label:'Pico dorsiflexión'},{key:'peakPlantar',label:'Pico plantiflexión'},{key:'rom',label:'ROM total'}]) validation.push({record,key:metric.key,label:metric.label,result:null,reference:'',tolerance:'5'});
function renderValidation() {
  $('validationBody').innerHTML=validation.map((v,i)=>{
    const err=v.result!==null && v.reference!=='' && Number.isFinite(Number(v.reference))?Math.abs(v.result-Number(v.reference)):null;
    const status=err===null?'<span class="status-pill status-pending">Pendiente</span>':err<=Number(v.tolerance)?'<span class="status-pill status-pass">Dentro de tolerancia</span>':'<span class="status-pill status-fail">Fuera de tolerancia</span>';
    return `<tr><td>Registro ${v.record}</td><td>${v.label}</td><td>${v.result===null?'—':fmt(v.result)} ${v.result===null?'':'°'}</td><td><input type="number" step="0.1" min="0" data-index="${i}" data-field="reference" value="${escapeHtml(v.reference)}" aria-label="Referencia externa ${v.label} registro ${v.record}" placeholder="Valor °"></td><td>${err===null?'—':`${fmt(err)}°`}</td><td><input type="number" step="0.1" min="0" data-index="${i}" data-field="tolerance" value="${escapeHtml(v.tolerance)}" aria-label="Tolerancia ${v.label} registro ${v.record}"></td><td>${status}</td></tr>`;
  }).join('');
  $('validationBody').querySelectorAll('input').forEach(input=>input.addEventListener('change',()=>{const i=Number(input.dataset.index);validation[i][input.dataset.field]=input.value;renderValidation();}));
}
function linkCurrentToValidation() {
  if(!currentMetrics){toast('Primero completa un análisis para vincular sus resultados.','error');return;}
  const record=Number($('validationRecord').value);
  validation.filter(v=>v.record===record).forEach(v=>v.result=currentMetrics[v.key]);renderValidation();
  toast(`Resultados actuales vinculados al Registro ${record}. Introduce las referencias externas.`, 'success');
}
function exportValidationCSV() {
  const rows=[['Registro','Indicador','Resultado MotionLab (°)','Referencia externa (°)','Error absoluto (°)','Tolerancia (°)','Estado']];
  validation.forEach(v=>{const err=v.result!==null&&v.reference!==''?Math.abs(v.result-Number(v.reference)):null;const status=err===null?'Pendiente':err<=Number(v.tolerance)?'Dentro de tolerancia':'Fuera de tolerancia';rows.push([v.record,v.label,v.result??'',v.reference,err??'',v.tolerance,status]);});
  const csv='\ufeff'+rows.map(r=>r.map(x=>`"${String(x).replace(/"/g,'""')}"`).join(';')).join('\n');downloadBlob(new Blob([csv],{type:'text/csv;charset=utf-8'}),'motionlab_validacion.csv');
}
function exportCSV() {
  if(!samples.length){toast('No hay muestras para exportar. Analiza un video o carga el caso demo.','error');return;}
  const m=currentMetrics||computeMetrics();const rows=[['tiempo_s','angulo_relativo_deg','angulo_geometrico_deg','confianza_landmarks','valido']];
  samples.forEach(s=>rows.push([s.t,s.angle,s.rawAngle,s.confidence,s.valid?1:0]));
  const csv='\ufeff'+rows.map(r=>r.join(';')).join('\n');downloadBlob(new Blob([csv],{type:'text/csv;charset=utf-8'}),`${safeName(sourceLabel)}_muestras.csv`);
  toast('CSV de muestras exportado.','success');
}
function exportJSON() {
  if(!samples.length){toast('No hay sesión para exportar. Analiza un video o carga el caso demo.','error');return;}
  const payload={application:'MotionLab',version:APP_VERSION,createdAt:new Date().toISOString(),source:sourceLabel,sourceType,side:$('sideSelect').value,samplingFps:sourceType==='video'?Number($('fpsSelect').value):'adaptativo',minimumVisibility:Number($('visibilitySelect').value),neutralRawAngle,calibration:calibrationDescription,units:{angle:'degrees',time:'seconds',angularVelocity:'degrees/second',confidence:'percent'},method:'Ángulo 2D entre vector tobillo-rodilla y tobillo-punta del pie; ángulo relativo = neutro calibrado menos ángulo geométrico.',metrics:currentMetrics?{peakDorsi:currentMetrics.peakDorsi,peakPlantar:currentMetrics.peakPlantar,rom:currentMetrics.rom,peakVelocity:currentMetrics.peakVelocity,meanConfidence:currentMetrics.meanConfidence,coverage:currentMetrics.coverage,cycles:currentMetrics.cycles}:null,samples};
  downloadBlob(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}),`${safeName(sourceLabel)}_sesion.json`);toast('Sesión JSON exportada con parámetros y trazabilidad.','success');
}
function preparePrintReport() {
  if(!currentMetrics){toast('No hay resultados para generar el informe.','error');return;}
  const m=currentMetrics;
  $('printReport').innerHTML=`<h1>MotionLab · Informe de análisis de tobillo</h1><p><b>Fecha:</b> ${new Date().toLocaleString('es-CL')} &nbsp; <b>Fuente:</b> ${escapeHtml(sourceLabel)} &nbsp; <b>Lado:</b> ${$('sideSelect').value==='left'?'Izquierdo':'Derecho'}</p><p><b>Calibración:</b> ${escapeHtml(calibrationDescription)} &nbsp; <b>Modelo:</b> MediaPipe Pose Landmarker &nbsp; <b>Versión aplicación:</b> ${APP_VERSION}</p><h2>Indicadores principales</h2><table><thead><tr><th>Indicador</th><th>Resultado</th><th>Unidad</th><th>Definición</th></tr></thead><tbody><tr><td>Pico dorsiflexión</td><td>${fmt(m.peakDorsi)}</td><td>°</td><td>Máximo ángulo positivo relativo</td></tr><tr><td>Pico plantiflexión</td><td>${fmt(m.peakPlantar)}</td><td>°</td><td>Magnitud del mínimo ángulo relativo</td></tr><tr><td>ROM</td><td>${fmt(m.rom)}</td><td>°</td><td>Máximo menos mínimo del registro</td></tr><tr><td>Velocidad angular pico</td><td>${fmt(m.peakVelocity)}</td><td>°/s</td><td>Máximo absoluto de Δángulo / Δtiempo</td></tr><tr><td>Cobertura</td><td>${fmt(m.coverage)}</td><td>%</td><td>Muestras válidas / procesadas</td></tr></tbody></table><h2>Interpretación y limitaciones</h2><p>Este análisis estima movimiento angular 2D a partir de landmarks de visión artificial. La validez depende del plano de cámara, oclusiones, visibilidad y calibración. Debe contrastarse con una referencia independiente y evaluación profesional. No constituye diagnóstico médico ni determina por sí solo una patología.</p><p><b>Parámetros:</b> Frecuencia de muestreo ${$('fpsSelect').value} FPS (video); confianza mínima ${$('visibilitySelect').value}; ${m.validFrames} muestras válidas / ${m.processedFrames} fotogramas procesados.</p>`;
  window.print();
}

function navTo(target) {
  document.querySelectorAll('.nav button').forEach(button => {
    button.classList.toggle('active', button.dataset.target === target);
  });
  $(target)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function safeBind(id, eventName, handler) {
  const element = $(id);
  if (!element) {
    console.warn(`MotionLab: no se encontró #${id}; se omite su evento.`);
    return;
  }
  element.addEventListener(eventName, event => {
    try {
      const result = handler(event);
      if (result && typeof result.catch === 'function') {
        result.catch(error => {
          console.error(`Error en ${id}:`, error);
          toast(error.message || 'Ocurrió un error al ejecutar esta acción.', 'error');
        });
      }
    } catch (error) {
      console.error(`Error en ${id}:`, error);
      toast(error.message || 'Ocurrió un error al ejecutar esta acción.', 'error');
    }
  });
}

function initializeMotionLab() {
  try {
    document.querySelectorAll('.nav button').forEach(button => {
      button.addEventListener('click', () => navTo(button.dataset.target));
    });

    safeBind('chooseVideo', 'click', () => $('videoFile')?.click());
    safeBind('videoFile', 'change', event => {
      loadVideoFile(event.target.files?.[0]);
      event.target.value = '';
    });
    safeBind('mediaStage', 'dragover', event => {
      event.preventDefault();
      $('mediaStage')?.classList.add('dragover');
    });
    safeBind('mediaStage', 'dragleave', () => $('mediaStage')?.classList.remove('dragover'));
    safeBind('mediaStage', 'drop', event => {
      event.preventDefault();
      $('mediaStage')?.classList.remove('dragover');
      loadVideoFile(event.dataTransfer?.files?.[0]);
    });

    safeBind('analyzeBtn', 'click', analyzeVideo);
    safeBind('cameraStart', 'click', startWebcam);
    safeBind('stopBtn', 'click', () => {
      if (stream) stopWebcam(true);
      else if (processing) {
        stopRequested = true;
        $('stopBtn').disabled = true;
        setStatus('Deteniendo…', true);
      }
    });
    safeBind('resetBtn', 'click', resetApp);
    safeBind('calibrateBtn', 'click', calibrateNeutral);
    safeBind('fpsSelect', 'change', () => {
      $('mediaFps').textContent = `Muestreo: ${$('fpsSelect').value} FPS`;
    });
    safeBind('modeVideo', 'click', () => {
      $('modeVideo')?.classList.add('active');
      $('modeCamera')?.classList.remove('active');
      $('videoFile')?.click();
    });
    safeBind('modeCamera', 'click', () => {
      $('modeCamera')?.classList.add('active');
      $('modeVideo')?.classList.remove('active');
      startWebcam();
    });

    safeBind('linkValidation', 'click', linkCurrentToValidation);
    safeBind('exportValidation', 'click', exportValidationCSV);
    safeBind('exportCsv', 'click', exportCSV);
    safeBind('exportJsonTop', 'click', exportJSON);
    safeBind('printPdfBtn', 'click', preparePrintReport);

    // La tabla y los gráficos se inicializan de manera independiente.
    try { renderValidation(); } catch (error) { console.error('No se pudo iniciar validación:', error); }
    try { initCharts(); } catch (error) { console.error('No se pudieron iniciar gráficos:', error); toast('No se pudieron iniciar los gráficos. Revisa la consola.', 'error'); }

    // Botón de demostración útil para probar gráficos sin depender de cámara ni MediaPipe.
    const controls = document.querySelector('.media-controls');
    if (controls && !$('demoButton')) {
      const demoButton = document.createElement('button');
      demoButton.type = 'button';
      demoButton.className = 'btn btn-ghost';
      demoButton.id = 'demoButton';
      demoButton.textContent = 'Ver caso demo';
      demoButton.style.marginLeft = 'auto';
      demoButton.addEventListener('click', () => {
        try { loadDemo(); } catch (error) { console.error(error); toast(error.message || 'No se pudo cargar la demo.', 'error'); }
      });
      controls.prepend(demoButton);
    }

    window.addEventListener('beforeunload', () => {
      stopWebcam(false);
      if (fileUrl) URL.revokeObjectURL(fileUrl);
      if (poseTask) poseTask.close();
    });

    console.info(`MotionLab ${APP_VERSION}: interfaz inicializada.`);
  } catch (error) {
    console.error('Error al iniciar MotionLab:', error);
    toast('La interfaz encontró un error al iniciar. Abre F12 → Console para ver el detalle.', 'error');
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeMotionLab, { once: true });
} else {
  initializeMotionLab();
}
