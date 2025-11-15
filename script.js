/**
 * Konfigurasi
 */
const MODEL_URL = 'https://xmrz7019w0uk3zke.public.blob.vercel-storage.com/model.json';
const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;

/**
 * Elemen DOM
 */
const loadingPercentEl = document.getElementById('loading-percent');
const loadingContainerEl = document.getElementById('loading-container');
const videoContainerEl = document.getElementById('video-container');
const videoEl = document.getElementById('webcam');

/**
 * Variabel Global
 */
let model = null;
let rafId = null; // RequestAnimationFrame ID

/**
 * 1. Fungsi Memuat Model Kustom
 */
async function loadCustomModel() {
    console.log("Memulai pemuatan model dari:", MODEL_URL);
    try {
        model = await tf.loadGraphModel(MODEL_URL, {
            onProgress: (fraction) => {
                const percent = Math.min(100, Math.round(fraction * 100)); // Pastikan tidak lebih dari 100
                loadingPercentEl.textContent = `${percent}%`;
                console.log(`Pemuatan progres: ${percent}%`);
            }
        });
        
        console.log("Model berhasil dimuat.");
        loadingContainerEl.textContent = "✅ Model berhasil dimuat! Memulai kamera...";
        
        // Setelah model dimuat, inisialisasi kamera
        await setupWebcam();

    } catch (error) {
        console.error("Gagal memuat model:", error);
        loadingContainerEl.innerHTML = `❌ Gagal memuat model. Periksa konsol untuk detail.`;
    }
}

/**
 * 2. Fungsi Mengaktifkan Kamera
 */
async function setupWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            'audio': false,
            'video': {
                facingMode: 'user', // Menggunakan kamera depan
                width: VIDEO_WIDTH,
                height: VIDEO_HEIGHT
            }
        });
        
        videoEl.srcObject = stream;
        videoEl.onloadedmetadata = () => {
            videoEl.width = VIDEO_WIDTH;
            videoEl.height = VIDEO_HEIGHT;
            videoContainerEl.style.width = `${VIDEO_WIDTH}px`;
            videoContainerEl.style.height = `${VIDEO_HEIGHT}px`;
            videoContainerEl.style.display = 'block';
            videoEl.play();
            
            // Mulai loop deteksi setelah video siap
            detectFrame(); 
            
            loadingContainerEl.style.display = 'none'; // Sembunyikan pesan loading
            console.log("Kamera siap. Mulai deteksi.");
        };

    } catch (error) {
        console.error("Gagal mengakses kamera:", error);
        loadingContainerEl.textContent = `❌ Gagal mengakses kamera: ${error.name}.`;
    }
}

/**
 * 3. Fungsi Deteksi dan Tampilan
 */

// Fungsi untuk membersihkan semua bounding box sebelumnya
function clearBBoxes() {
    const boxes = videoContainerEl.querySelectorAll('.detection-box');
    boxes.forEach(box => box.remove());
}

// Fungsi untuk menggambar bounding box
function drawBBox(bbox, label, score) {
    const [x, y, w, h] = bbox;

    // Pastikan koordinat dan dimensi valid
    if (w <= 0 || h <= 0) return; 

    // Buat elemen DIV untuk kotak deteksi
    const boxEl = document.createElement('div');
    boxEl.className = 'detection-box';
    boxEl.style.left = `${x}px`;
    boxEl.style.top = `${y}px`;
    boxEl.style.width = `${w}px`;
    boxEl.style.height = `${h}px`;

    // Buat elemen DIV untuk label
    const labelEl = document.createElement('div');
    labelEl.className = 'detection-label';
    
    // **PENTING: Pastikan penulisan label dari kiri ke kanan**
    // Karena kita menggunakan tag <div> normal, teks akan ditulis dari kiri ke kanan (default CSS).
    labelEl.textContent = `${label} (${Math.round(score * 100)}%)`;
    
    boxEl.appendChild(labelEl);
    videoContainerEl.appendChild(boxEl);
}


// Fungsi utama untuk menjalankan deteksi dalam loop
function detectFrame() {
    if (!model || videoEl.paused || videoEl.ended) {
        // Hentikan jika model belum siap atau video berhenti
        return; 
    }

    // Konversi frame video menjadi tensor
    const inputTensor = tf.browser.fromVideo(videoEl)
        .resizeBilinear([VIDEO_HEIGHT, VIDEO_WIDTH]) // Resize ke ukuran yang diharapkan model
        .expandDims(0) // Tambahkan dimensi batch (1)
        .toFloat(); // Ubah ke float

    // **Asumsi:** Model Anda adalah GraphModel yang menerima input berupa tensor (1, H, W, 3) dan memiliki banyak output (anak bin/output node)
    
    model.executeAsync(inputTensor).then(outputs => {
        // Bersihkan bounding box sebelumnya
        clearBBoxes();
        
        // **CATATAN PENTING UNTUK MODEL DENGAN BANYAK ANAK BIN/OUTPUT NODE:**
        // Array 'outputs' akan berisi satu tensor untuk setiap output node yang didefinisikan dalam model Anda.
        // Anda harus tahu urutan dan arti dari setiap tensor output (misalnya: index 0 = bounding box, index 1 = class scores, index 2 = jumlah deteksi).
        // Karena saya tidak tahu struktur output *tepat* dari model kustom Anda, saya akan menggunakan struktur umum:
        
        // GANTI BAGIAN INI dengan logika pengambilan data deteksi yang sesuai
        // Contoh Logika Umum (Anda harus menyesuaikannya berdasarkan model kustom Anda!):
        
        // const [boxes, scores, classes, numDetections] = outputs;
        // const num = numDetections.dataSync()[0];
        // const boxesData = boxes.dataSync();
        // const scoresData = scores.dataSync();
        // const classesData = classes.dataSync();

        // **Jika output Anda sudah diformat seperti COCO-SSD (boxes, classes, scores, num_detections):**
        // Lakukan NMS (Non-Maximum Suppression) jika model Anda belum melakukannya.
        
        // --- LOGIKA DETEKSI SIMPLIFIED ---
        // Contoh: Ambil 5 deteksi teratas (GANTI DENGAN LOGIKA MODEL ANDA)
        // Kita asumsikan output[0] berisi data Bounding Box dan Class Info
        if (Array.isArray(outputs) && outputs.length > 0) {
            
            // Contoh sederhana: Simulasi hasil deteksi (GANTI INI)
            // Asumsi: outputs[0].shape = [1, MAX_DETECTIONS, 6] -> [y_min, x_min, y_max, x_max, score, class_id]
            const outputTensor = outputs[0];
            const [numDetections, maxBoxes] = outputTensor.shape.slice(1);
            const data = outputTensor.dataSync();

            // Hanya proses deteksi dengan confidence > 0.5
            const threshold = 0.5; 
            
            for (let i = 0; i < maxBoxes; i++) {
                const offset = i * 6; // Sesuaikan dengan jumlah elemen per deteksi
                const [y_min, x_min, y_max, x_max, score, class_id] = data.slice(offset, offset + 6);
                
                if (score > threshold) {
                    // Konversi koordinat normalisasi (0-1) ke piksel
                    const x = x_min * VIDEO_WIDTH;
                    const y = y_min * VIDEO_HEIGHT;
                    const w = (x_max - x_min) * VIDEO_WIDTH;
                    const h = (y_max - y_min) * VIDEO_HEIGHT;

                    // Ganti angka class_id dengan nama label yang sesuai
                    // Anda harus memiliki array nama kelas (misalnya: ['orang', 'mobil', 'kucing', ...])
                    // Contoh dummy label:
                    const className = `Objek-${Math.floor(class_id)}`; 
                    
                    // Gambar bounding box
                    drawBBox([x, y, w, h], className, score);
                }
            }
            
        } else {
            // Jika model hanya mengembalikan 1 tensor atau memiliki struktur yang berbeda
            console.warn("Struktur output model kustom tidak dikenali. Harap sesuaikan `detectFrame`.");
        }
        
        // Bersihkan tensor yang tidak lagi digunakan untuk mencegah kebocoran memori
        tf.dispose(outputs); 
        tf.dispose(inputTensor);

        // Ulangi loop
        rafId = requestAnimationFrame(detectFrame);
    });
}

/**
 * 4. Fungsi Inisialisasi
 */
function init() {
    loadCustomModel();
}

// Mulai aplikasi
init();
