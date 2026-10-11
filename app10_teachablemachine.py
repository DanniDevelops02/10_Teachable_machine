import json

import streamlit as st
import streamlit.components.v1 as components

# ---------- Configuración de página ----------
st.set_page_config(
    page_title="YMCA · Detector de poses",
    page_icon="🕺",
    layout="centered",
)

# ---------- Modelo y audios ----------
# El modelo de poses de Teachable Machine solo se exporta para TensorFlow.js,
# por eso la detección corre en el navegador (en tiempo real) y no con Keras.
MODEL_URL = "https://teachablemachine.withgoogle.com/models/4fkkxYXuS/"
SECUENCIA = ["Y", "M", "C", "A"]  # "Neutral" no suena

# Audios en la carpeta audio/ del repositorio (Y.mp3, M.mp3, C.mp3, A.mp3).
# Cambia USUARIO/REPO por los de tu GitHub. Si un audio falta, suena un tono.
AUDIO_BASE_URL = "https://github.com/DanniDevelops02/10_Teachable_machine/tree/main/Audios"

# ---------- Estilo (misma paleta que app.py) ----------
PRIMARIO = "#3f71a6"
OSCURO = "#1b2f45"
CLARO = "#eaf2f9"
SUAVE = "#a9c7e3"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', sans-serif;
        font-size: 1.08rem;
    }}
    .stApp {{
        background: linear-gradient(180deg, #f6fafd 0%, {CLARO} 100%);
    }}
    .block-container {{
        max-width: 860px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }}
    .hero {{
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }}
    .hero .saludo {{
        color: {PRIMARIO};
        letter-spacing: .25em;
        text-transform: uppercase;
        font-size: .85rem;
        font-weight: 600;
    }}
    .hero h1 {{
        font-family: 'Playfair Display', serif;
        font-size: 3.2rem;
        color: {OSCURO};
        margin: .3rem 0 .6rem 0;
        line-height: 1.1;
    }}
    .hero p {{
        color: #4d6479;
        font-size: 1.2rem;
        max-width: 640px;
        margin: 0 auto;
    }}
    .linea {{
        width: 70px;
        height: 4px;
        border-radius: 4px;
        background: {PRIMARIO};
        margin: 1.4rem auto;
    }}
    h2, h3 {{
        font-family: 'Playfair Display', serif !important;
        color: {OSCURO} !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: white;
        border: 1px solid {SUAVE}66 !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 24px rgba(63, 113, 166, .08);
        padding: .6rem .8rem;
    }}
    [data-testid="stSidebar"] {{
        background: {OSCURO};
    }}
    [data-testid="stSidebar"] * {{
        color: #e6f0f9 !important;
    }}
    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Ajustes de detección")
    umbral = st.slider(
        "Confianza mínima", 0.50, 0.99, 0.75, 0.01,
        help="Probabilidad que debe superar una pose para contar.",
    )
    frames = st.slider(
        "Frames sostenidos", 1, 10, 3,
        help="Frames seguidos que debe mantenerse la pose. Menos = más rápido, más = más estable.",
    )
    espera = st.slider(
        "Pausa entre sonidos (s)", 0.0, 1.0, 0.15, 0.05,
        help="En la canción cada letra cae aprox. cada 0,47 s, así que conviene dejarla baja.",
    )
    cortar = st.checkbox(
        "Cortar el audio anterior al sonar el siguiente", value=True,
        help="Evita que los audios se encimen cuando vas rápido.",
    )
    st.caption("Si suena sin querer, sube la confianza o los frames. Si se siente lenta, bájalos.")

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero">
        <div class="saludo">Teachable Machine · Poses</div>
        <h1>Detector YMCA</h1>
        <p>Párate frente a la cámara con el cuerpo visible de la cintura para arriba
        y haz las letras Y · M · C · A con los brazos. Cada pose dispara su sonido.</p>
        <div class="linea"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

config = {
    "modelUrl": MODEL_URL,
    "secuencia": SECUENCIA,
    "audio": {p: f"{AUDIO_BASE_URL}{p}.mp3" for p in SECUENCIA},
    "umbral": umbral,
    "frames": frames,
    "esperaMs": int(espera * 1000),
    "cortar": cortar,
    "ventanaMs": 4000,  # tiempo máximo entre letras para contar la secuencia
}

# ---------- Componente en el navegador (TensorFlow.js + Teachable Machine Pose) ----------
HTML = """
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@teachablemachine/pose@0.8.6/dist/teachablemachine-pose.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;600&display=swap" rel="stylesheet">

<style>
  :root { --primario: __PRIMARIO__; --oscuro: __OSCURO__; --claro: __CLARO__; --suave: __SUAVE__; }
  body { margin: 0; font-family: 'Inter', sans-serif; color: var(--oscuro); }
  .wrap { display: flex; flex-direction: column; align-items: center; gap: 14px; padding: 6px; }
  button { background: var(--primario); color: #fff; border: 0; border-radius: 999px;
           padding: .7rem 2.2rem; font-size: 1.05rem; font-weight: 600; cursor: pointer;
           font-family: inherit; transition: all .2s ease; }
  button:hover:not(:disabled) { background: var(--oscuro); transform: translateY(-2px); }
  button:disabled { opacity: .6; cursor: default; }
  #estado { font-size: .95rem; color: #4d6479; min-height: 1.2em; text-align: center; }
  .escena { position: relative; }
  canvas { display: block; border-radius: 20px; background: var(--claro); max-width: 100%;
           box-shadow: 0 12px 32px rgba(27, 47, 69, .18); }
  #letra { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
           font-family: 'Playfair Display', serif; font-size: 170px; font-weight: 700; color: #fff;
           text-shadow: 0 6px 24px rgba(27, 47, 69, .6); opacity: 0; transform: scale(.7);
           transition: opacity .08s, transform .08s; pointer-events: none; }
  #letra.on { opacity: 1; transform: scale(1); }
  #fps { position: absolute; top: 10px; right: 12px; font-size: 12px; color: #fff;
         background: rgba(27, 47, 69, .55); padding: 2px 8px; border-radius: 999px; }
  .fichas { display: flex; gap: 10px; }
  .ficha { width: 58px; height: 58px; border-radius: 14px; display: flex; align-items: center;
           justify-content: center; font-family: 'Playfair Display', serif; font-size: 30px; font-weight: 700;
           background: #fff; border: 2px solid var(--suave); color: var(--suave); transition: all .12s; }
  .ficha.hecha { background: var(--primario); border-color: var(--primario); color: #fff; }
  .ficha.pulso { transform: scale(1.15); }
  #mensaje { font-family: 'Playfair Display', serif; font-size: 1.4rem; min-height: 1.6em; color: var(--primario); }
  .barras { width: 380px; max-width: 100%; display: grid; gap: 6px; }
  .fila { display: grid; grid-template-columns: 70px 1fr 46px; align-items: center; gap: 8px; font-size: 14px; }
  .pista { height: 12px; background: var(--claro); border-radius: 6px; overflow: hidden; }
  .lleno { height: 100%; width: 0; background: var(--suave); }
  .fila.activa { font-weight: 600; }
  .fila.activa .lleno { background: var(--primario); }
</style>

<div class="wrap">
  <button id="btn">▶ Iniciar cámara</button>
  <div id="estado">Al presionar el botón el navegador te pedirá permiso para usar la cámara.</div>
  <div class="escena">
    <canvas id="canvas" width="320" height="320"></canvas>
    <div id="letra"></div>
    <div id="fps"></div>
  </div>
  <div class="fichas" id="fichas"></div>
  <div id="mensaje"></div>
  <div class="barras" id="barras"></div>
</div>

<script>
const CFG = __CONFIG__;
const TAM = 320;
const $ = (id) => document.getElementById(id);

let model, webcam, ctx, labels = [];
let audioCtx = null, fuenteActual = null;
const buffers = {};
const fallosAudio = {};  // letra -> motivo (p. ej. 404), para diagnosticar

// Estado de la detección
let candidata = null, conteo = 0, ultimaSonada = null, ultimoSonidoMs = 0;
// Estado de la secuencia Y-M-C-A
let paso = 0, ultimoPasoMs = 0;
// Medidor de velocidad
let framesFps = 0, inicioFps = performance.now();

// ---------- Audio: Web Audio API para latencia casi cero ----------
async function cargarAudios() {
  await Promise.all(CFG.secuencia.map(async (p) => {
    try {
      const r = await fetch(CFG.audio[p]);
      if (!r.ok) throw new Error(r.status);
      buffers[p] = await audioCtx.decodeAudioData(await r.arrayBuffer());
    } catch (e) {
      buffers[p] = null;  // se usará el tono de respaldo
      fallosAudio[p] = e.message === "404" ? "no encontrado (404)" : (e.message || "no se pudo leer");
      console.warn("Audio", p, CFG.audio[p], fallosAudio[p]);
    }
  }));
}

const NOTAS = { Y: 523.25, M: 659.25, C: 783.99, A: 1046.5 };

function sonar(p) {
  if (CFG.cortar && fuenteActual) { try { fuenteActual.stop(); } catch (e) {} }
  let fuente;
  if (buffers[p]) {
    fuente = audioCtx.createBufferSource();
    fuente.buffer = buffers[p];
    fuente.connect(audioCtx.destination);
    fuente.start();
  } else {
    // Tono de respaldo: una nota distinta por letra
    fuente = audioCtx.createOscillator();
    const g = audioCtx.createGain(), t = audioCtx.currentTime;
    fuente.type = "triangle";
    fuente.frequency.value = NOTAS[p] || 440;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.4, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.35);
    fuente.connect(g).connect(audioCtx.destination);
    fuente.start(t);
    fuente.stop(t + 0.4);
  }
  fuenteActual = fuente;

  const l = $("letra");
  l.textContent = p;
  l.classList.add("on");
  clearTimeout(l._t);
  l._t = setTimeout(() => l.classList.remove("on"), 350);

  avanzarSecuencia(p);
}

// ---------- Secuencia Y-M-C-A ----------
function pintarFichas() {
  CFG.secuencia.forEach((p, i) => $("ficha-" + p).classList.toggle("hecha", i < paso));
}

function avanzarSecuencia(p) {
  const ahora = performance.now();
  if (paso > 0 && ahora - ultimoPasoMs > CFG.ventanaMs) paso = 0;

  if (p === CFG.secuencia[paso]) paso++;
  else paso = (p === CFG.secuencia[0]) ? 1 : 0;
  ultimoPasoMs = ahora;

  const f = $("ficha-" + p);
  f.classList.add("pulso");
  setTimeout(() => f.classList.remove("pulso"), 150);

  pintarFichas();
  if (paso === CFG.secuencia.length) {
    $("mensaje").textContent = "¡Y·M·C·A completa! 🎉";
    setTimeout(() => { paso = 0; pintarFichas(); $("mensaje").textContent = ""; }, 1500);
  } else {
    $("mensaje").textContent = "";
  }
}

// ---------- Interfaz ----------
function armarInterfaz() {
  $("fichas").innerHTML = CFG.secuencia
    .map((p) => `<div class="ficha" id="ficha-${p}">${p}</div>`).join("");
  $("barras").innerHTML = labels.map((n) => `
    <div class="fila" id="f-${n}">
      <span>${n}</span>
      <div class="pista"><div class="lleno" id="b-${n}"></div></div>
      <span id="v-${n}">0%</span>
    </div>`).join("");
}

async function iniciar() {
  const btn = $("btn");
  btn.disabled = true;
  $("estado").textContent = "Cargando modelo y audios…";
  try {
    // El clic desbloquea el audio en el navegador
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    // Modelo y audios se cargan en paralelo
    const [m] = await Promise.all([
      tmPose.load(CFG.modelUrl + "model.json", CFG.modelUrl + "metadata.json"),
      cargarAudios(),
    ]);
    model = m;
    labels = model.getClassLabels();
    armarInterfaz();

    $("estado").textContent = "Pidiendo acceso a la cámara…";
    webcam = new tmPose.Webcam(TAM, TAM, true);  // true = efecto espejo
    await webcam.setup();
    await webcam.play();
    ctx = $("canvas").getContext("2d");

    // Calentamiento: la primera inferencia compila los shaders de WebGL y es lenta
    webcam.update();
    const { posenetOutput } = await model.estimatePose(webcam.canvas);
    await model.predict(posenetOutput);

    const sinAudio = CFG.secuencia.filter((p) => !buffers[p]);
    $("estado").textContent = sinAudio.length
      ? `¡Listo! Sin audio para ${sinAudio.map((p) => `${p} (${fallosAudio[p]})`).join(", ")}: sonará un tono. ` +
        `Revisa ${CFG.audio[sinAudio[0]]}`
      : "¡Listo! Haz las poses Y · M · C · A";
    btn.textContent = "Cámara activa";
    window.requestAnimationFrame(loop);
  } catch (e) {
    console.error(e);
    $("estado").textContent = "No se pudo iniciar: " + String(e.message || e).replace(/\.+$/, "") +
      ". Revisa el permiso de cámara del navegador.";
    btn.disabled = false;
  }
}

async function loop() {
  webcam.update();
  await predecir();
  window.requestAnimationFrame(loop);
}

async function predecir() {
  const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
  const pred = await model.predict(posenetOutput);

  let mejor = pred[0];
  for (const p of pred) {
    if (p.probability > mejor.probability) mejor = p;
    const pct = Math.round(p.probability * 100);
    $("b-" + p.className).style.width = pct + "%";
    $("v-" + p.className).textContent = pct + "%";
  }
  for (const n of labels) $("f-" + n).classList.toggle("activa", n === mejor.className);

  decidir(mejor);
  dibujar(pose);
  medirFps();
}

// Suena cuando una letra supera el umbral durante N frames seguidos.
// La misma letra no se repite hasta pasar por otra pose o por Neutral.
function decidir(mejor) {
  const nombre = mejor.probability >= CFG.umbral ? mejor.className : null;

  if (nombre !== candidata) { candidata = nombre; conteo = 0; }
  conteo++;
  if (candidata === null || conteo < CFG.frames) return;

  if (!CFG.secuencia.includes(candidata)) { ultimaSonada = null; return; }

  const ahora = performance.now();
  if (candidata === ultimaSonada || ahora - ultimoSonidoMs < CFG.esperaMs) return;

  sonar(candidata);
  ultimaSonada = candidata;
  ultimoSonidoMs = ahora;
}

function dibujar(pose) {
  ctx.drawImage(webcam.canvas, 0, 0);
  if (pose) {
    tmPose.drawKeypoints(pose.keypoints, 0.5, ctx);
    tmPose.drawSkeleton(pose.keypoints, 0.5, ctx);
  }
}

function medirFps() {
  framesFps++;
  const ahora = performance.now();
  if (ahora - inicioFps >= 1000) {
    $("fps").textContent = Math.round(framesFps * 1000 / (ahora - inicioFps)) + " fps";
    framesFps = 0;
    inicioFps = ahora;
  }
}

$("btn").addEventListener("click", iniciar);
</script>
"""

html = (
    HTML.replace("__CONFIG__", json.dumps(config))
    .replace("__PRIMARIO__", PRIMARIO)
    .replace("__OSCURO__", OSCURO)
    .replace("__CLARO__", CLARO)
    .replace("__SUAVE__", SUAVE)
)

with st.container(border=True):
    components.html(html, height=800, scrolling=False)

st.write("")

with st.container(border=True):
    st.subheader("¿Cómo funciona?")
    st.markdown(
        """
- **Modelo:** Teachable Machine (poses) con las clases Y, M, C, A y Neutral.
- **Dónde corre:** en tu navegador con TensorFlow.js, en tiempo real. La cámara no sale de tu equipo.
- **Cuándo suena:** cuando una letra supera la confianza mínima durante los frames indicados.
- **Secuencia:** las fichas se encienden si haces Y → M → C → A en orden, como en la canción.
- **Audios:** carpeta `audio/` del repositorio. Si alguno falta, suena un tono de respaldo.
"""
    )
