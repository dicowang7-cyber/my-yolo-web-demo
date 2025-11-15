const MODEL_URL = "https://xmrz7019w0uk3zke.public.blob.vercel-storage.com/model.json";

const statusEl = document.getElementById('status');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let model = null;
const INPUT_SIZE = 640;    // Sesuai export YOLO (640x640)
const SCORE_THRESHOLD = 0.1; // Tetap 0.1 untuk debugging
const NMS_IOU = 0.45;
const MAX_OUTPUT = 50;    // Max boxes returned by NMS
const NUM_CLASSES = 16;   

// --- DEFINISI NAMA KELAS AKTUAL (16 Kelas) ---
const CLASS_NAMES = [
    "Alpukat",
    "Anggur",
    "Apel",
    "Apel Hijau",
    "Jeruk",
    "Lemon",
    "Mangga",
    "Melon",
    "Nanas",
    "Pepaya",
    "Pir",
    "Pisang",
    "Rambutan",
    "Salak",
    "Semangka", 
    "Stroberi" 
];
// ------------------------------------------

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

// --- FUNGSI UTAMA PERBAIKAN: Mirroring Video dan Deteksi ---
function drawBoxes(boxes) {
  
  // 1. Hapus frame sebelumnya (penting untuk real-time)
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 2. Gambar video frame dengan MIRRORING
  // Konteks digeser dan dibalik (sumbu X)
  ctx.save();
  ctx.scale(-1, 1);
  ctx.translate(-canvas.width, 0); 
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.restore(); // Kembalikan konteks ke normal setelah menggambar video
  
  // 3. Gambar deteksi dan teks di konteks NORMAL
  boxes.forEach(b => {
    // Ambil koordinat yang sudah diskalakan (berdasarkan INPUT_SIZE model)
    const scaledX1 = b.x1 * (canvas.width / INPUT_SIZE);
    const scaledX2 = b.x2 * (canvas.width / INPUT_SIZE);
    const scaledY1 = b.y1 * (canvas.height / INPUT_SIZE);
    const scaledY2 = b.y2 * (canvas.height / INPUT_SIZE);
    
    // Terapkan formula MIRRORING ke koordinat X untuk menampilkan di posisi yang benar
    // Posisi X awal deteksi harus dibalik: canvas.width - X2_scaled (untuk x1)
    const x = canvas.width - scaledX2; 
    const y = scaledY1;
    const w = scaledX2 - scaledX1;
    const h = scaledY2 - scaledY1;
    
    // Box (Digambar dalam koordinat normal canvas)
    ctx.strokeStyle = "lime";
    ctx.lineWidth = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 200));
    ctx.strokeRect(x, y, w, h);

    // Label: Gambar Teks di konteks NORMAL (Teks TIDAK terbalik)
    
    // Mengambil nama kelas dan membentuk label
    const className = CLASS_NAMES[b.classId] || `Kelas Tidak Dikenal ${b.classId}`; 
    const label = `${className} (${(b.score*100).toFixed(1)}%)`; 
    
    ctx.font = "18px Arial";
    const textW = ctx.measureText(label).width;
    const pad = 6;
    
    // Label bg (koordinat normal)
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(x, y - 22, textW + pad, 22);

    // Text (koordinat normal)
    ctx.fillStyle = "lime";
    // **Teks digambar dari kiri ke kanan (normal) karena konteks sudah di-restore**
    ctx.fillText(label, x + 4, y - 6); 
  });
}

// --- Post-Processing dengan Koreksi YOLOv8 ---

async function postprocess(outputTensor) {
  // Model output shape: [1, 20, 8400] -> transpose -> [8400, 20]
  let t = outputTensor;

  const transposed = tf.tidy(() => t.squeeze().transpose()); // [8400, 20]
  const data = await transposed.array();
  transposed.dispose();

  const boxes = [];
  const scores = [];
  const classIds = [];
    
    let maxOverallScore = 0; 

  for (let i = 0; i < data.length; i++) {
    const row = data[i];
    
    const x = row[0]; // Center X
    const y = row[1]; // Center Y
    const w = row[2]; // Width
    const h = row[3]; // Height
    
    const classLogits = row.slice(4); 
    const probs = softmax(classLogits);
    const maxProb = Math.max(...probs);
    const classId = probs.indexOf(maxProb); 

    const finalScore = maxProb; 
    
    if (finalScore > maxOverallScore) {
        maxOverallScore = finalScore;
    }

    if (finalScore < SCORE_THRESHOLD) continue; 

    const [x1, y1, x2, y2] = xywh_to_xyxy(x, y, w, h);

    // [y1, x1, y2, x2] diperlukan oleh tf.image.nonMaxSuppressionAsync
    boxes.push([y1 * INPUT_SIZE, x1 * INPUT_SIZE, y2 * INPUT_SIZE, x2 * INPUT_SIZE]); 
    scores.push(finalScore);
    classIds.push(classId); 
  }

    console.log(`Max class score found: ${maxOverallScore.toFixed(3)}`);
    console.log(`Detections passing initial threshold (${SCORE_THRESHOLD}): ${boxes.length}`);

  if (boxes.length === 0) {
    return [];
  }
  
  const boxesTensor = tf.tensor2d(boxes);
  const scoresTensor = tf.tensor1d(scores);
  const selectedIdx = await tf.image.nonMaxSuppressionAsync(
    boxesTensor, scoresTensor, MAX_OUTPUT, NMS_IOU, SCORE_THRESHOLD
  );
  const selected = await selectedIdx.array();

  const final = [];
  for (let idx of selected) {
    // Kembali ke format normalisasi 0-INPUT_SIZE
    const [y1,x1,y2,x2] = boxes[idx];
    final.push({
      x1: x1 / INPUT_SIZE, y1: y1 / INPUT_SIZE, x2: x2 / INPUT_SIZE, y2: y2 / INPUT_SIZE,
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
        // Modifikasi untuk menampilkan persentase loading
        statusEl.textContent = "Memuat model: 0%";
        model = await tf.loadGraphModel(MODEL_URL, {
            onProgress: (fraction) => {
                const percent = Math.min(100, Math.round(fraction * 100)); 
                statusEl.textContent = `Memuat model: ${percent}%`;
            }
        });
        statusEl.textContent = "✅ Model berhasil dimuat. Memulai kamera...";
        console.log("Model loaded:", model);
    } catch (err) {
        console.error("Failed to load model:", err);
        statusEl.textContent = "❌ Gagal memuat model. Periksa konsol.";
    }
}

async function setupCamera() {
  try {
    // Minta kamera depan
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" }, 
      audio: false
    });
    video.srcObject = stream;
    await new Promise(resolve => video.onloadedmetadata = resolve);
    video.play();
    
    // Set ukuran canvas sesuai ukuran video sebenarnya
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    video.style.visibility = 'hidden'; // Sembunyikan video asli
    
    // Set ukuran kontainer agar sesuai
    const container = document.getElementById('container');
    container.style.width = `${video.videoWidth}px`;
    container.style.height = `${video.videoHeight}px`;

  } catch (err) {
    console.error("Camera error:", err);
    statusEl.textContent = "❌ Error kamera. Izinkan akses kamera dan coba lagi.";
  }
}

// --- Detection Loop ---

async function detectLoop() {
  if (!model || video.paused || video.ended) return;

  tf.engine().startScope();

  const input = tf.tidy(() => {
    // Konversi frame video
    const img = tf.browser.fromPixels(video);
    // Resize dan normalisasi
    return img.resizeBilinear([INPUT_SIZE, INPUT_SIZE]).div(255.0).expandDims(0);
  });

  let output = null;
  try {
    const res = await model.executeAsync(input);
    if (Array.isArray(res)) {
      output = res[0]; // Ambil output pertama (asumsi YOLOv8)
      // Buang tensor output lainnya (jika ada)
      for (let i = 1; i < res.length; i++) if (res[i] && res[i].dispose) res[i].dispose();
    } else {
      output = res;
    }
  } catch (err) {
    console.error("Model inference failed:", err);
    statusEl.textContent = "Inference error (check console).";
    tf.dispose(input);
    tf.engine().endScope();
    requestAnimationFrame(detectLoop);
    return;
  }

  const detections = await postprocess(output);

  drawBoxes(detections);

  tf.dispose([input, output]);
  tf.engine().endScope();

  requestAnimationFrame(detectLoop);
}

// --- Main Execution ---

(async () => {
  await loadModel();
  await setupCamera();
  statusEl.textContent = "Model siap — menjalankan deteksi.";
  detectLoop();
})();
