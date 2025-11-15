// Anda harus mengganti URL ini dengan lokasi model.json Anda yang sebenarnya
const MODEL_URL = "https://xmrz7019w0uk3zke.public.blob.vercel-storage.com/model.json";

const statusEl = document.getElementById('status');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let model = null;
const INPUT_SIZE = 640;
const SCORE_THRESHOLD = 0.1;
const NMS_IOU = 0.45;
const MAX_OUTPUT = 50;
const NUM_CLASSES = 16;

const CLASS_NAMES = [
  "Alpukat", "Anggur", "Apel", "Apel Hijau",
  "Jeruk", "Lemon", "Mangga", "Melon",
  "Nanas", "Pepaya", "Pir", "Pisang",
  "Rambutan", "Salak", "Semangka", "Stroberi"
];

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a,b)=>a+b,0);
  return exps.map(e => e / sum);
}

function xywh_to_xyxy(x, y, w, h) {
  return [x - w/2, y - h/2, x + w/2, y + h/2];
}


// =======================================================================
// FIX MIRROR — Video mirror, bounding box + text normal
// =======================================================================
function drawBoxes(dets) {

  // ==== 1. Gambar VIDEO (MIRROR) ====
  ctx.save();
  ctx.scale(-1, 1);
  ctx.translate(-canvas.width, 0);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.restore();
  // ---- video selesai digambar, transform dibersihkan ----

  // ==== 2. Gambar BOX + TEKS (TANPA MIRROR) ====
  dets.forEach(b => {

    // scaling dari 640 → ukuran canvas
    const scaleX = canvas.width / INPUT_SIZE;
    const scaleY = canvas.height / INPUT_SIZE;

    let x1 = b.x1 * scaleX;
    let y1 = b.y1 * scaleY;
    let x2 = b.x2 * scaleX;
    let y2 = b.y2 * scaleY;

    // konversi mirror → balik posisi horizontal
    const mirroredX1 = canvas.width - x2;
    const mirroredX2 = canvas.width - x1;

    const w = mirroredX2 - mirroredX1;
    const h = y2 - y1;

    // BOX
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 3;
    ctx.strokeRect(mirroredX1, y1, w, h);

    // LABEL
    const className = CLASS_NAMES[b.classId] || ("Class " + b.classId);
    const label = `${className} ${(b.score*100).toFixed(1)}%`;

    ctx.font = "18px Arial";
    const textW = ctx.measureText(label).width;

    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(mirroredX1, y1 - 22, textW + 10, 22);

    ctx.fillStyle = "lime";
    ctx.fillText(label, mirroredX1 + 5, y1 - 6);
  });
}


// =======================================================================
// POST PROCESSING
// =======================================================================
async function postprocess(outputTensor) {

  const transposed = tf.tidy(() => outputTensor.squeeze().transpose());
  const data = await transposed.array();
  transposed.dispose();

  const boxes = [];
  const scores = [];
  const classIds = [];

  for (let i = 0; i < data.length; i++) {
    const row = data[i];

    const x = row[0];
    const y = row[1];
    const w = row[2];
    const h = row[3];

    const probs = softmax(row.slice(4));
    const maxProb = Math.max(...probs);
    const classId = probs.indexOf(maxProb);

    if (maxProb < SCORE_THRESHOLD) continue;

    const [x1, y1, x2, y2] = xywh_to_xyxy(x, y, w, h);

    // Format untuk TF NMS: [y1, x1, y2, x2]
    boxes.push([y1 * INPUT_SIZE, x1 * INPUT_SIZE, y2 * INPUT_SIZE, x2 * INPUT_SIZE]);
    scores.push(maxProb);
    classIds.push(classId);
  }

  if (boxes.length === 0) return [];

  const bT = tf.tensor2d(boxes);
  const sT = tf.tensor1d(scores);

  const idx = await tf.image.nonMaxSuppressionAsync(bT, sT, MAX_OUTPUT, NMS_IOU, SCORE_THRESHOLD);
  const selected = await idx.array();

  const result = [];
  for (let i of selected) {
    const [y1,x1,y2,x2] = boxes[i];
    result.push({
      x1, y1, x2, y2,
      score: scores[i],
      classId: classIds[i]
    });
  }

  bT.dispose();
  sT.dispose();
  idx.dispose();

  return result;
}


// =======================================================================
// MODEL & CAMERA
// =======================================================================
async function loadModel() {
  statusEl.textContent = "Loading model...";
  model = await tf.loadGraphModel(MODEL_URL);
  statusEl.textContent = "Model loaded.";
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" },
    audio: false
  });
  video.srcObject = stream;
  await new Promise(res => video.onloadedmetadata = res);
  video.play();

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}


// =======================================================================
// MAIN LOOP
// =======================================================================
async function detectLoop() {
  if (!model) return;

  tf.engine().startScope();

  const input = tf.tidy(() => {
    return tf.browser.fromPixels(video)
      .resizeBilinear([INPUT_SIZE, INPUT_SIZE])
      .div(255)
      .expandDims(0);
  });

  let output;
  try {
    const res = await model.executeAsync(input);
    output = Array.isArray(res) ? res[0] : res;
  } catch (err) {
    console.error("Inference error:", err);
    tf.engine().endScope();
    requestAnimationFrame(detectLoop);
    return;
  }

  const dets = await postprocess(output);
  drawBoxes(dets);

  tf.dispose([input, output]);
  tf.engine().endScope();

  requestAnimationFrame(detectLoop);
}


// =======================================================================
// START
// =======================================================================
(async () => {
  await loadModel();
  await setupCamera();
  statusEl.textContent = "Model ready — running detection.";
  detectLoop();
})();
