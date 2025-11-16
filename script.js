// Anda harus memastikan file model.json dan bin-nya ada di direktori yang sama
const MODEL_URL = 'https://xmrz7019w0uk3zke.public.blob.vercel-storage.com/model.json';
const INPUT_SIZE = 640;
const CLASSES = ["Alpukat", "Anggur", "Apel", "Apel Hijau", "Jeruk", "Lemon", "Mangga", "Melon", "Nanas", "Pepaya", "Pir", "Pisang", "Rambutan", "Salak", "Semangka", "Stroberi"];

const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusDiv = document.getElementById('status');
const fpsSpan = document.getElementById('fps');

let model;
let lastTime = 0;
let frameCount = 0;

/**
 * Memulai akses kamera dan setup canvas.
 */
async function setupWebcam() {
    return new Promise((resolve, reject) => {
        const navigatorAny = navigator;
        navigator.getUserMedia = navigator.getUserMedia ||
            navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
            navigatorAny.msGetUserMedia;

        if (navigator.getUserMedia) {
            navigator.getUserMedia(
                { video: { width: INPUT_SIZE, height: INPUT_SIZE } },
                stream => {
                    video.srcObject = stream;
                    video.addEventListener('loadeddata', () => resolve(video), { once: true });
                },
                error => reject(error)
            );
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

        // Warmup model untuk inisialisasi GPU dan alokasi memori
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
    // tf.tidy memastikan memori tensor yang tidak perlu dibersihkan
    tf.tidy(() => {
        const tensor = tf.browser.fromPixels(video)
            .resizeNearestNeighbor([INPUT_SIZE, INPUT_SIZE]) // Ubah ukuran ke input model
            .div(255.0) // Normalisasi
            .expandDims(0); // Tambahkan dimensi batch

        // 2. Lakukan Prediksi (asumsi asynchronous untuk model graph/frozen)
        model.executeAsync(tensor).then(predictions => {
            // 3. Proses Prediksi dan Gambar Bounding Box
            // *** Bagian ini sangat bergantung pada struktur output model YOLOv8 Anda! ***
            // Anda perlu logic untuk mem-parsing output, menerapkan NMS, dan memetakan ke koordinat.
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Contoh Sederhana (Ganti dengan logika NMS dan Bounding Box yang benar)
            // Asumsi predictions[0] adalah tensor bounding box dan scores
            if (predictions && predictions.length > 0) {
                const outputData = predictions[0].dataSync(); // Ambil data
                
                // Placeholder untuk hasil deteksi
                const detections = [];
                // Logika placeholder: Ambil hasil prediksi (ganti dengan NMS yang benar)
                // for (let i = 0; i < outputData.length; i += 6) { 
                //    const [boxX, boxY, boxW, boxH, conf, classId] = outputData.slice(i, i+6);
                //    if (conf > 0.5) detections.push({boxX, boxY, boxW, boxH, classId});
                // }
                
                // Deteksi placeholder (ganti dengan logika model Anda)
                // Untuk contoh visualisasi:
                ctx.font = 'bold 16px Arial';
                ctx.fillStyle = '#FF5722';
                ctx.fillText(`Deteksi aktif! (Logic NMS perlu diimplementasi)`, 10, 30);
                
                // Update jumlah deteksi (Contoh: selalu 1 jika aktif)
                document.getElementById('class-count').textContent = `Total Deteksi: 1`;
            }
            
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
