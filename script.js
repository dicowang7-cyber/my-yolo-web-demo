// Impor TensorFlow.js di dalam Web Worker
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.2.0/dist/tf.min.js');

let model = null;
let lastInferenceEnd = 0;

// --- KONSTANTA & KONFIGURASI ---
const INPUT_SIZE = 640;
const SCORE_THRESHOLD = 0.1;
const NMS_IOU = 0.45;
const MAX_OUTPUT = 50;
const NUM_CLASSES = 16; 

// DEFINISI NAMA KELAS AKTUAL (Harus sama dengan di index.html)
const CLASS_NAMES = [
    "Alpukat", "Anggur", "Apel", "Apel Hijau", "Jeruk", "Lemon", "Mangga", "Melon", 
    "Nanas", "Pepaya", "Pir", "Pisang", "Rambutan", "Salak", "Semangka", "Stroberi" 
];


// --- Utility Functions (Dibuat di Worker) ---

function softmax(arr) {
    const max = arr.reduce((a, b) => Math.max(a, b), -Infinity);
    const exps = arr.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
}

function xywh_to_xyxy(x, y, w, h) {
    const x1 = x - w/2;
    const y1 = y - h/2;
    const x2 = x + w/2;
    const y2 = y + h/2;
    return [x1, y1, x2, y2];
}

async function postprocess(outputTensor) {
    // Logic Postprocessing sama persis dengan yang ada di kode asli Anda.
    let t = outputTensor;

    const transposed = tf.tidy(() => t.squeeze().transpose());
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
        
        // Logika YOLOv8: confidence score sudah termasuk dalam kelas
        const classLogits = row.slice(4);
        const probs = softmax(classLogits);
        const maxProb = Math.max(...probs);
        const classId = probs.indexOf(maxProb);

        const finalScore = maxProb;
        
        if (finalScore < SCORE_THRESHOLD) continue;

        const [x1, y1, x2, y2] = xywh_to_xyxy(x, y, w, h);

        // tf.image.nonMaxSuppressionAsync membutuhkan koordinat dalam format [y1, x1, y2, x2]
        boxes.push([y1 * INPUT_SIZE, x1 * INPUT_SIZE, y2 * INPUT_SIZE, x2 * INPUT_SIZE]);
        scores.push(finalScore);
        classIds.push(classId);
    }

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
        const [y1,x1,y2,x2] = boxes[idx];
        final.push({
            x1: x1 / INPUT_SIZE, // Normalisasi kembali ke 0-1 untuk worker
            y1: y1 / INPUT_SIZE,
            x2: x2 / INPUT_SIZE,
            y2: y2 / INPUT_SIZE,
            score: scores[idx],
            classId: classIds[idx],
            className: CLASS_NAMES[classIds[idx]] // Tambahkan nama kelas
        });
    }

    tf.dispose([boxesTensor, scoresTensor, selectedIdx]);

    return final;
}


// --- Handler Inferensi di Worker ---
async function runInference(data) {
    if (!model) {
        console.warn("Model belum dimuat di worker.");
        return;
    }
    
    const startTime = performance.now();
    
    // 1. Buat tf.Tensor dari imageData (yang sudah dikirim via Transferable)
    // Ingat: imageData.data adalah Uint8ClampedArray (r,g,b,a)
    const pixels = new Uint8ClampedArray(data.imageData);
    const imageTensor = tf.tensor4d(pixels, [data.height, data.width, 4]).slice([0, 0, 0], [-1, -1, 3]); // Hapus alpha channel (A)

    tf.engine().startScope();
    
    const input = tf.tidy(() => {
        // Pre-processing: resize, normalisasi, dan expandDims
        return imageTensor.resizeBilinear([INPUT_SIZE, INPUT_SIZE])
                          .div(255.0)
                          .expandDims(0);
    });

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
        console.error("Model inference failed in worker:", err);
        postMessage({ type: 'ERROR', message: 'Inference failed in worker.' });
        tf.dispose(input);
        tf.engine().endScope();
        return;
    }
    
    const detections = await postprocess(output);
    
    tf.dispose([input, output, imageTensor]); // Membersihkan tensor
    tf.engine().endScope();
    
    const endTime = performance.now();
    const inferenceTime = endTime - startTime;
    const fps = 1000 / inferenceTime;

    // Kirim hasil deteksi kembali ke thread utama
    postMessage({
        type: 'RESULT',
        boxes: detections,
        fps: fps,
        // Kirim kembali buffer imageData untuk digunakan lagi di thread utama (jika menggunakan transfer)
        // Saat ini tidak perlu karena buffer sudah dibuang oleh Worker, tapi ini praktik yang baik
        // imageData: data.imageData 
    });
}


// --- Main Message Listener ---
self.onmessage = async (event) => {
    const data = event.data;

    if (data.type === 'INIT') {
        // Inisialisasi model
        tf.setBackend('webgl').then(async () => {
            console.log("Worker: TFJS WebGL backend set.");
            try {
                model = await tf.loadGraphModel(data.modelUrl, {
                    onProgress: (fraction) => {
                        postMessage({ type: 'LOADING', progress: (fraction * 100) });
                    }
                });
                // Lakukan satu inferensi dummy untuk "pemanasan" (warm-up)
                model.predict(tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3])).dispose();
                postMessage({ type: 'LOADED' });
            } catch (err) {
                postMessage({ type: 'ERROR', message: 'Failed to load model in worker.' });
            }
        });
    } else if (data.type === 'INFER') {
        // Jalankan inferensi
        runInference(data);
    }
};
