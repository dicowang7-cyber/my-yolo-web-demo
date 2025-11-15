// Ganti model URL Anda
const MODEL_URL = "https://xmrz7019w0uk3zke.public.blob.vercel-storage.com/model.json";

const statusEl = document.getElementById('status');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Offscreen canvas untuk menggambar video mirrored
const offscreen = document.createElement('canvas');
const offctx = offscreen.getContext('2d');

let model = null;
const INPUT_SIZE = 640;
const SCORE_THRESHOLD = 0.1;
const NMS_IOU = 0.45;
const MAX_OUTPUT = 50;

const CLASS_NAMES = [
  "Alpukat","Anggur","Apel","Apel Hijau",
  "Jeruk","Lemon","Mangga","Melon",
  "Nanas","Pepaya","Pir","Pisang",
  "Rambutan","Salak","Semangka","Stroberi"
];

function softmax(arr){
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a,b)=>a+b,0);
  return exps.map(e => e / sum);
}

function xywh_to_xyxy(x,y,w,h){
  return [x - w/2, y - h/2, x + w/2, y + h/2];
}

// -----------------------------
// Draw pipeline: use offscreen for mirrored video
// -----------------------------
function drawBoxes(dets){
  // pastikan canvas utama bersih
  ctx.clearRect(0,0,canvas.width,canvas.height);

  // 1) Gambar video yang sudah di-MIRROR melalui offscreen canvas
  offscreen.width = canvas.width;
  offscreen.height = canvas.height;

  // gambar video ke offscreen dengan transform mirror
  offctx.save();
  offctx.clearRect(0,0,offscreen.width, offscreen.height);
  offctx.scale(-1, 1);
  offctx.translate(-offscreen.width, 0);
  offctx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
  offctx.restore();

  // salin offscreen ke canvas utama tanpa merubah transform canvas utama
  ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);

  // 2) Gambar kotak & label pada canvas utama (identitas transform)
  // Pastikan transform identitas (menghindari sisa transform)
  ctx.setTransform(1,0,0,1,0,0);
  ctx.textBaseline = "top";

  const scaleX = canvas.width / INPUT_SIZE;
  const scaleY = canvas.height / INPUT_SIZE;

  dets.forEach(b => {
    // b.x1..b.y2 dalam koordinat input-space (0..INPUT_SIZE),
    // convert ke pixel canvas
    let x1 = b.x1 * scaleX;
    let y1 = b.y1 * scaleY;
    let x2 = b.x2 * scaleX;
    let y2 = b.y2 * scaleY;

    // Karena video sudah di-mirror, kita harus membalik koordinat X
    // Mirror transform: mirroredX = canvas.width - originalX
    const mirroredX1 = canvas.width - x2;
    const mirroredX2 = canvas.width - x1;
    const boxW = mirroredX2 - mirroredX1;
    const boxH = y2 - y1;

    // draw box
    ctx.lineWidth = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 200));
    ctx.strokeStyle = "lime";
    ctx.strokeRect(mirroredX1, y1, boxW, boxH);

    // label background + text
    const className = CLASS_NAMES[b.classId] || ("Class " + b.classId);
    const label = `${className} ${(b.score*100).toFixed(1)}%`;
    ctx.font = "16px Arial";
    const pad = 6;
    const textW = ctx.measureText(label).width;

    const labelX = mirroredX1;
    const labelY = Math.max(0, y1 - 20);

    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(labelX, labelY, textW + pad, 18);

    ctx.fillStyle = "lime";
    ctx.fillText(label, labelX + 4, labelY + 1);
  });
}

// -----------------------------
// Postprocess (NMS)
// Model expected output -> transpose etc.
// -----------------------------
async function postprocess(outputTensor){
  // output tensor: [1, C, N] / or [1, N, C] tergantung model.
  // Kode ini mengikuti asumsi output squeeze()->transpose() => [N, 20]
  const transposed = tf.tidy(()=> outputTensor.squeeze().transpose());
  const data = await transposed.array();
  transposed.dispose();

  const boxes = [];
  const scores = [];
  const classIds = [];

  for (let i = 0; i < data.length; i++){
    const row = data[i];
    const x = row[0];
    const y = row[1];
    const w = row[2];
    const h = row[3];
    const logits = row.slice(4);
    const probs = softmax(logits);
    const maxProb = Math.max(...probs);
    const classId = probs.indexOf(maxProb);

    if (maxProb < SCORE_THRESHOLD) continue;

    const [x1,y1,x2,y2] = xywh_to_xyxy(x,y,w,h);
    // Simpan dalam satuan INPUT_SIZE (0..INPUT_SIZE)
    // NOTE: NonMaxSuppression tf expects: [y1, x1, y2, x2]
    boxes.push([y1 * INPUT_SIZE, x1 * INPUT_SIZE, y2 * INPUT_SIZE, x2 * INPUT_SIZE]);
    scores.push(maxProb);
    classIds.push(classId);
  }

  if (boxes.length === 0) return [];

  const boxesTensor = tf.tensor2d(boxes);
  const scoresTensor = tf.tensor1d(scores);
  const selectedIdx = await tf.image.nonMaxSuppressionAsync(
    boxesTensor, scoresTensor, MAX_OUTPUT, NMS_IOU, SCORE_THRESHOLD
  );
  const selected = await selectedIdx.array();

  const final = [];
  for (let idx of selected){
    const [y1,x1,y2,x2] = boxes[idx];
    // Kembalikan ke format x1,y1,x2,y2 (dalam satuan INPUT_SIZE)
    final.push({
      x1: x1 / 1.0, // masih dalam rentang 0..INPUT_SIZE
      y1: y1 / 1.0,
      x2: x2 / 1.0,
      y2: y2 / 1.0,
      score: scores[idx],
      classId: classIds[idx]
    });
  }

  boxesTensor.dispose();
  scoresTensor.dispose();
  selectedIdx.dispose();

  return final;
}

// -----------------------------
// Model & camera setup
// -----------------------------
async function loadModel(){
  try {
    statusEl.textContent = "Loading model...";
    model = await tf.loadGraphModel(MODEL_URL);
    statusEl.textContent = "Model loaded.";
    console.log("Model loaded:", model);
  } catch (err) {
    console.error("Failed to load model:", err);
    statusEl.textContent = "Failed to load model. Check console.";
  }
}

async function setupCamera(){
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false
    });
    video.srcObject = stream;
    await new Promise(res => video.onloadedmetadata = res);
    video.play();

    // atur ukuran canvas berdasarkan video actual
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    offscreen.width = canvas.width;
    offscreen.height = canvas.height;
  } catch (err) {
    console.error("Camera error:", err);
    statusEl.textContent = "Camera error. Allow camera and reload.";
  }
}

// -----------------------------
// Detection loop
// -----------------------------
async function detectLoop(){
  if (!model) return;

  tf.engine().startScope();

  const input = tf.tidy(()=>{
    return tf.browser.fromPixels(video)
      .resizeBilinear([INPUT_SIZE, INPUT_SIZE])
      .div(255.0)
      .expandDims(0);
  });

  let output = null;
  try {
    const res = await model.executeAsync(input);
    output = Array.isArray(res) ? res[0] : res;
    // dispose remainder if array
    if (Array.isArray(res)) {
      for (let i = 1; i < res.length; i++) if (res[i] && res[i].dispose) res[i].dispose();
    }
  } catch (err) {
    console.error("Inference error:", err);
    statusEl.textContent = "Inference error (see console).";
    tf.dispose(input);
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

// -----------------------------
// Start
// -----------------------------
(async ()=>{
  await loadModel();
  await setupCamera();
  statusEl.textContent = "Model ready — running detection.";
  detectLoop();
})();
