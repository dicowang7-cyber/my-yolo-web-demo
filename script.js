// Anda harus mengganti URL ini dengan lokasi model.json Anda yang sebenarnya
const MODEL_URL = "https://xmrz7019w0uk3zke.public.blob.vercel-storage.com/model.json";

const statusEl = document.getElementById('status');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let model = null;
const INPUT_SIZE = 640;    // Sesuai export YOLO (640x640)
const SCORE_THRESHOLD = 0.1; // <--- DIATUR SANGAT RENDAH (0.1) UNTUK PENGUJIAN
const NMS_IOU = 0.45;
const MAX_OUTPUT = 50;    // Max boxes returned by NMS
const NUM_CLASSES = 16;   // Dihitung dari output model 20 - 4 koordinat = 16

// --- Utility Functions ---

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

// Fungsi Softmax untuk mengkonversi logit kelas menjadi probabilitas
function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a,b)=>a+b,0);
  return exps.map(e => e / sum);
}

// Konversi format box dari [center_x, center_y, width, height] ke [x1, y1, x2, y2]
function xywh_to_xyxy(x, y, w, h) {
  const x1 = x - w/2;
  const y1 = y - h/2;
  const x2 = x + w/2;
  const y2 = y + h/2;
  return [x1, y1, x2, y2];
}

function drawBoxes(boxes) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  boxes.forEach(b => {
    // Scaling koordinat kotak dari 640x640 ke ukuran canvas/video
    const x = b.x1 * (canvas.width / INPUT_SIZE);
    const y = b.y1 * (canvas.height / INPUT_SIZE);
    const w = (b.x2 - b.x1) * (canvas.width / INPUT_SIZE);
    const h = (b.y2 - b.y1) * (canvas.height / INPUT_SIZE);

    // Box
    ctx.strokeStyle = "lime";
    ctx.lineWidth = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 200));
    ctx.strokeRect(x, y, w, h);

    // Label bg
    const label = `Class ${b.classId} ${(b.score*100).toFixed(1)}%`;
    ctx.font = "18px Arial";
    const textW = ctx.measureText(label).width;
    const pad = 6;
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(x, y - 22, textW + pad, 22);

    // Text
    ctx.fillStyle = "lime";
    ctx.fillText(label, x + 4, y - 6);
  });
}

// --- Post-Processing dengan Koreksi YOLOv8 dan Debugging ---

async function postprocess(outputTensor) {
  // Model output shape: [1, 20, 8400] -> transpose -> [8400, 20]
  // 20 = 4 (koordinat) + 16 (kelas)
  let t = outputTensor;

  // Transpose dan konversi ke array JS
  const transposed = tf.tidy(() => t.squeeze().transpose()); // [8400, 20]
  const data = await transposed.array();
  transposed.dispose();

  const boxes = [];
  const scores = [];
  const classIds = [];
    
    let maxOverallScore = 0; // Untuk debugging

  for (let i = 0; i < data.length; i++) {
    const row = data[i];
    // row: [x, y, w, h, cls0, cls1, cls2, ..., cls15]
    
    const x = row[0];
    const y = row[1];
    const w = row[2];
    const h = row[3];
    
    // Logit kelas dimulai dari index 4
    const classLogits = row.slice(4); 
    
    const probs = softmax(classLogits);
    const maxProb = Math.max(...probs);
    const classId = probs.indexOf(maxProb); 

    // Skor akhir adalah probabilitas kelas maksimum (koreksi YOLOv8)
    const finalScore = maxProb; 
    
    if (finalScore > maxOverallScore) {
        maxOverallScore = finalScore;
    }

    if (finalScore < SCORE_THRESHOLD) continue; 

    const [x1, y1, x2, y2] = xywh_to_xyxy(x, y, w, h);

    // Simpan kotak dalam format [y1, x1, y2, x2] untuk NMS TFJS
    boxes.push([y1 * INPUT_SIZE, x1 * INPUT_SIZE, y2 * INPUT_SIZE, x2 * INPUT_SIZE]); 
    scores.push(finalScore);
    classIds.push(classId); // Menyimpan ID kelas
  }

    // DEBUGGING: Cek skor tertinggi yang ditemukan
    console.log(`Max class score found: ${maxOverallScore.toFixed(3)}`);
    console.log(`Detections passing initial threshold (${SCORE_THRESHOLD}): ${boxes.length}`);

  if (boxes.length === 0) {
    return [];
  }
  
  // Run NMS (Non-Max Suppression)
  const boxesTensor = tf.tensor2d(boxes);
  const scoresTensor = tf.tensor1d(scores);
  const selectedIdx = await tf.image.nonMaxSuppressionAsync(
    boxesTensor, scoresTensor, MAX_OUTPUT, NMS_IOU, SCORE_THRESHOLD
  );
  const selected = await selectedIdx.array();

  const final = [];
  for (let idx of selected) {
    // Ambil data dari array asli menggunakan indeks yang dipilih NMS
    const [y1,x1,y2,x2] = boxes[idx];
    final.push({
      x1: x1, y1: y1, x2: x2, y2: y2,
      score: scores[idx],
      classId: classIds[idx] 
    });
  }

  boxesTensor.dispose();
  scoresTensor.dispose();
  selectedIdx.dispose();

  return final;
}

// --- Inisialisasi dan Setup Kamera/Model ---

async function loadModel() {
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

async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false
    });
    video.srcObject = stream;
    await new Promise(resolve => video.onloadedmetadata = resolve);
    video.play();
    // set canvas size to video display size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  } catch (err) {
    console.error("Camera error:", err);
    statusEl.textContent = "Camera error. Allow camera and reload.";
  }
}

// --- Detection Loop ---

async function detectLoop() {
  if (!model) return;

  tf.engine().startScope();

  // Preprocess: capture current frame, resize to INPUT_SIZE
  const input = tf.tidy(() => {
    const img = tf.browser.fromPixels(video);
    return img.resizeBilinear([INPUT_SIZE, INPUT_SIZE]).div(255.0).expandDims(0);
  });

  // Model Inference
  let output = null;
  try {
    const res = await model.executeAsync(input);
    if (Array.isArray(res)) {
      output = res[0];
      for (let i = 1; i < res.length; i++) if (res[i] && res[i].dispose) res[i].dispose();
    } else {
      output = res;
    }
  } catch (err) {
    console.error("Model inference failed:", err);
    statusEl.textContent = "Inference error (see console).";
    tf.dispose(input);
    tf.engine().endScope();
    requestAnimationFrame(detectLoop);
    return;
  }

  const detections = await postprocess(output);

  // Gambar video frame sebagai latar belakang
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Gambar bounding boxes
  drawBoxes(detections);

  tf.dispose([input, output]);
  tf.engine().endScope();

  requestAnimationFrame(detectLoop);
}

// --- Main Execution ---

(async () => {
  await loadModel();
  await setupCamera();
  // pastikan canvas ukurannya sesuai dengan video
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  statusEl.textContent = "Model ready — running detection.";
  detectLoop();
})();
