// Anda harus memastikan file model.json dan bin-nya ada di direktori yang sama
const MODEL_URL = 'https://xmrz7019w0uk3zke.public.blob.vercel-storage.com/model.json';
const INPUT_SIZE = 640;
const CLASSES = ["Alpukat", "Anggur", "Apel", "Apel Hijau", "Jeruk", "Lemon", "Mangga", "Melon", "Nanas", "Pepaya", "Pir", "Pisang", "Rambutan", "Salak", "Semangka", "Stroberi"];
const CONF_THRESHOLD = 0.25; // Ambang batas kepercayaan deteksi minimal
const IOU_THRESHOLD = 0.45; // Ambang batas IoU untuk Non-Maximum Suppression (NMS)

const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusDiv = document.getElementById('status');
const fpsSpan = document.getElementById('fps');
const classCountSpan = document.getElementById('class-count');

let model;
let lastTime = 0;
let frameCount = 0;

/**
 * Memulai akses kamera dan setup canvas.
 */
async function setupWebcam() {
    return new Promise((resolve, reject) => {
        const videoConstraints = {
            video: {
                width: { exact: INPUT_SIZE },
                height: { exact: INPUT_SIZE }
            }
        };

        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia(videoConstraints)
                .then(stream => {
                    video.srcObject = stream;
                    video.addEventListener('loadeddata', () => resolve(video), { once: true });
                })
                .catch(error => reject(new Error(`Akses kamera ditolak atau gagal: ${error.name}`)));
        } else if (navigator.getUserMedia) {
            // Fallback untuk browser lama
            navigator.getUserMedia(videoConstraints, stream => {
                video.srcObject = stream;
                video.addEventListener('loadeddata', () => resolve(video), { once: true });
            }, error => reject(error));
        } else {
            reject(new Error("Webcam tidak didukung di browser ini."));
        }
    });
}

/**
 * Memuat model TensorFlow.js.
 */
async function loadModel() {
    try {
        statusDiv.innerHTML = "Memuat model...";
        model = await tf.loadGraphModel(MODEL_URL);

        // Warmup model
        const dummyInput = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
        await model.executeAsync(dummyInput);
        dummyInput.dispose();

        statusDiv.innerHTML = "Model Siap!";
        statusDiv.classList.remove('loading');
        statusDiv.classList.add('ready');
    } catch (e) {
        statusDiv.innerHTML = `Gagal memuat model: ${e.message}`;
        console.error("Gagal memuat model", e);
    }
}

/**
 * Menggambar bounding box dan label pada canvas.
 */
function drawDetections(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    detections.forEach(d => {
        const [x1, y1, x2, y2, score, classId] = d;
        const className = CLASSES[classId];

        // Koordinat untuk canvas (diskalakan dari 640x640)
        const x = x1;
        const y = y1;
        const width = x2 - x1;
        const height = y2 - y1;

        // Gambar Bounding Box
        ctx.strokeStyle = '#FF5722'; // Warna kotak
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        // Gambar Background Label
        const text = `${className} ${(score * 100).toFixed(1)}%`;
        ctx.font = 'bold 14px Arial';
        const textMetrics = ctx.measureText(text);
        const textWidth = textMetrics.width;
        const textHeight = 16; 
        
        ctx.fillStyle = '#4CAF50'; // Warna background label
        ctx.fillRect(x, y - textHeight - 4, textWidth + 8, textHeight + 4);

        // Gambar Teks Label
        ctx.fillStyle = '#FFFFFF'; // Warna teks
        ctx.fillText(text, x + 4, y - 4);
    });
}

/**
 * Memproses output tensor dari model YOLOv8 (Non-Maximum Suppression dan konversi koordinat).
 * Output model: [1, 8400, 20] -> [batch, boxes, (4+16)] (4 box coords, 1 confidence, 15 classes)
 */
async function processOutput(predictions) {
    if (!predictions || predictions.length === 0) return [];
    
    // Asumsi predictions[0] adalah output tensor utama [1, 8400, 20]
    const outputTensor = predictions[0];
    const data = outputTensor.squeeze().arraySync(); // Bentuk [8400, 20]

    let boxes = [];
    let scores = [];
    let classIds = [];
    
    // Perhatikan: Output YOLOv8 TFJS GraphModel seringkali memiliki format: 
    // [Bounding Box (4), Confidence Skor (1), Class Scores (N)] 
    // ATAU [Bounding Box (4), Class Scores (N), Confidence Skor (1)]

    // Berdasarkan file metadata Anda yang menyarankan output [1, 8400, 20] (4 box + 16 classes)
    // Format yang umum untuk YOLOv8 di TFJS adalah: [box_x, box_y, box_w, box_h, class_scores...]
    
    data.forEach(row => {
        const [cx, cy, w, h] = row.slice(0, 4); // Koordinat Center, Width, Height
        const classScores = row.slice(4); // Skor untuk 16 kelas

        // Temukan kelas dengan skor tertinggi
        const maxScore = Math.max(...classScores);
        const classId = classScores.indexOf(maxScore);
        
        // Skor deteksi adalah MAX(Class Score)
        const score = maxScore; 

        if (score >= CONF_THRESHOLD) {
            // Konversi dari format center-width-height (cx, cy, w, h) ke format sudut (x1, y1, x2, y2)
            const x1 = cx - w / 2;
            const y1 = cy - h / 2;
            const x2 = cx + w / 2;
            const y2 = cy + h / 2;
            
            // Konversi ke koordinat tampilan 640x640 (tidak perlu scaling jika input=output size)
            // Jika model sudah menghasilkan koordinat 640x640, tidak perlu scaling lagi.
            
            boxes.push([x1, y1, x2, y2]);
            scores.push(score);
            classIds.push(classId);
        }
    });
    
    // Non-Maximum Suppression (NMS) untuk menghilangkan kotak yang tumpang tindih
    const nms = await tf.image.nonMaxSuppressionAsync(
        tf.tensor2d(boxes), 
        tf.tensor1d(scores), 
        100, // maxBoxes
        IOU_THRESHOLD, 
        CONF_THRESHOLD
    );
    
    const selectedIndices = nms.dataSync();
    
    const finalDetections = selectedIndices.map(i => {
        return [...boxes[i], scores[i], classIds[i]];
    });
    
    // Pembersihan memori tensor
    nms.dispose();
    
    return finalDetections;
}

/**
 * Loop deteksi utama menggunakan requestAnimationFrame untuk FPS yang mulus.
 */
function detectFrame(timestamp) {
    if (!model) {
        requestAnimationFrame(detectFrame);
        return;
    }

    // Hitung FPS
    frameCount++;
    const elapsed = timestamp - lastTime;
    if (elapsed > 1000) {
        const fps = (frameCount * 1000 / elapsed).toFixed(1);
        fpsSpan.textContent = `FPS: ${fps}`;
        lastTime = timestamp;
        frameCount = 0;
    }

    // 1. Dapatkan frame video
    tf.tidy(() => {
        const tensor = tf.browser.fromPixels(video)
            .resizeNearestNeighbor([INPUT_SIZE, INPUT_SIZE])
            .div(255.0)
            .expandDims(0); 

        // 2. Lakukan Prediksi (asumsi asynchronous untuk model graph/frozen)
        model.executeAsync(tensor).then(predictions => {
            // 3. Proses Prediksi, Terapkan NMS, dan Gambar Bounding Box
            processOutput(predictions).then(detections => {
                drawDetections(detections);
                classCountSpan.textContent = `Total Deteksi: ${detections.length}`;
            });
            
            // Pembersihan memori tensor prediksi
            predictions.forEach(t => t.dispose());
            
            // Lanjut ke frame berikutnya
            requestAnimationFrame(detectFrame);
        });
    });
}

/**
 * Fungsi inisialisasi utama.
 */
async function init() {
    try {
        canvas.width = INPUT_SIZE;
        canvas.height = INPUT_SIZE;
        await loadModel();
        await setupWebcam();
        
        // Mulai looping deteksi setelah video siap
        video.play();
        video.onloadedmetadata = () => {
            requestAnimationFrame(detectFrame);
        };

    } catch (error) {
        console.error('Inisialisasi Gagal:', error);
        statusDiv.innerHTML = `ERROR: ${error.message}`;
        statusDiv.classList.remove('loading');
    }
}

init();
