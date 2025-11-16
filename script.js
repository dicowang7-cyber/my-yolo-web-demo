// Konfigurasi Konstanta
const MODEL_URL = 'https://xmrz7019w0uk3zke.public.blob.vercel-storage.com/model.json';
const INPUT_SIZE = 640;
const CLASSES = ["Alpukat", "Anggur", "Apel", "Apel Hijau", "Jeruk", "Lemon", "Mangga", "Melon", "Nanas", "Pepaya", "Pir", "Pisang", "Rambutan", "Salak", "Semangka", "Stroberi"];
const CLASS_COUNT = CLASSES.length;
const CONF_THRESHOLD = 0.25; // Ambang batas kepercayaan deteksi minimal
const IOU_THRESHOLD = 0.45; // Ambang batas IoU untuk Non-Maximum Suppression (NMS)

// Elemen DOM
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
 * Memulai akses kamera dan setup canvas. (Fungsi ini sudah benar)
 */
async function setupWebcam() {
    // Implementasi setupWebcam yang sama...
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
 * Memuat model TensorFlow.js. (Fungsi ini sudah benar)
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
 * Menggambar bounding box dan label pada canvas. (Fungsi ini sudah benar)
 */
function drawDetections(detections) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    detections.forEach(d => {
        // d = [x1, y1, x2, y2, score, classId]
        const [x1, y1, x2, y2, score, classId] = d;
        const className = CLASSES[classId];

        const x = x1;
        const y = y1;
        const width = x2 - x1;
        const height = y2 - y1;

        // Gambar Bounding Box
        ctx.strokeStyle = '#FF5722'; 
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        // Gambar Background Label
        const text = `${className} ${(score * 100).toFixed(1)}%`;
        ctx.font = 'bold 14px Arial';
        const textMetrics = ctx.measureText(text);
        const textWidth = textMetrics.width;
        const textHeight = 16; 
        
        ctx.fillStyle = '#4CAF50'; 
        // Menggambar label di atas kotak, memastikan tidak keluar batas atas canvas
        const labelY = y < textHeight ? y + 2 : y - textHeight - 4;

        ctx.fillRect(x, labelY, textWidth + 8, textHeight + 4);

        // Gambar Teks Label
        ctx.fillStyle = '#FFFFFF'; 
        ctx.fillText(text, x + 4, labelY + textHeight);
    });
}

/**
 * Memproses output tensor dari model YOLOv8.
 * Mengimplementasikan konversi koordinat dan NMS.
 */
async function processOutput(predictions) {
    if (!predictions || predictions.length === 0) return [];
    
    // Asumsi: predictions[0] adalah tensor output utama [1, 8400, 20]
    const outputTensor = predictions[0];
    
    // PERBAIKAN: Transpose tensor ke format [N, D] di mana N=8400
    // Model YOLOv8 TFJS seringkali memiliki output dengan shape yang terbalik
    // dari [1, 8400, 20] ke [1, 20, 8400]. Kita perlu transpose kembali.
    const transposed = outputTensor.transpose([0, 2, 1]); // [1, 20, 8400] -> [1, 8400, 20]

    // Squeeze the batch dimension
    const data = transposed.squeeze().dataSync(); // Bentuk [8400 * 20]
    
    const numDetections = transposed.shape[1]; // 8400
    const stride = transposed.shape[2]; // 20 (4 box + 16 classes)

    let boxes = [];
    let scores = [];
    let classIds = [];
    
    for (let i = 0; i < numDetections; i++) {
        const offset = i * stride;
        
        // Data Bounding Box (cx, cy, w, h)
        const cx = data[offset + 0];
        const cy = data[offset + 1];
        const w = data[offset + 2];
        const h = data[offset + 3];

        // Data Class Scores (Kolom 4 hingga 19)
        let maxScore = 0;
        let classId = -1;
        
        for (let j = 0; j < CLASS_COUNT; j++) {
            const score = data[offset + 4 + j];
            if (score > maxScore) {
                maxScore = score;
                classId = j;
            }
        }
        
        if (maxScore >= CONF_THRESHOLD) {
            // Konversi dari format center-width-height (cx, cy, w, h) ke format sudut (x1, y1, x2, y2)
            // Koordinat relatif terhadap 640x640
            const x1 = cx - w / 2;
            const y1 = cy - h / 2;
            const x2 = cx + w / 2;
            const y2 = cy + h / 2;
            
            boxes.push([x1, y1, x2, y2]);
            scores.push(maxScore);
            classIds.push(classId);
        }
    }
    
    // Pembersihan memori tensor yang baru dibuat
    transposed.dispose();

    if (boxes.length === 0) return [];

    // Lakukan Non-Maximum Suppression (NMS)
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
    
    // Pembersihan memori NMS
    nms.dispose();
    
    return finalDetections;
}

/**
 * Loop deteksi utama.
 */
function detectFrame(timestamp) {
    if (!model) {
        requestAnimationFrame(detectFrame);
        return;
    }

    // Hitung FPS (Logika FPS sudah benar dan lancar)
    frameCount++;
    const elapsed = timestamp - lastTime;
    if (elapsed > 1000) {
        const fps = (frameCount * 1000 / elapsed).toFixed(1);
        fpsSpan.textContent = `FPS: ${fps}`;
        lastTime = timestamp;
        frameCount = 0;
    }

    // 1. Dapatkan frame video & pre-processing
    tf.tidy(() => {
        const tensor = tf.browser.fromPixels(video)
            .resizeNearestNeighbor([INPUT_SIZE, INPUT_SIZE])
            .div(255.0)
            .expandDims(0); 

        // 2. Lakukan Prediksi
        model.executeAsync(tensor).then(predictions => {
            // 3. Proses, NMS, dan Gambar
            processOutput(predictions).then(detections => {
                drawDetections(detections);
                classCountSpan.textContent = `Total Deteksi: ${detections.length}`;
            }).catch(e => console.error("Error during post-processing:", e));
            
            // Pembersihan memori tensor prediksi
            predictions.forEach(t => t.dispose());
            
            // Lanjut ke frame berikutnya
            requestAnimationFrame(detectFrame);
        }).catch(e => console.error("Error during model execution:", e));

    }); // tf.tidy end
}

/**
 * Fungsi inisialisasi utama. (Fungsi ini sudah benar)
 */
async function init() {
    try {
        canvas.width = INPUT_SIZE;
        canvas.height = INPUT_SIZE;
        await loadModel();
        await setupWebcam();
        
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
