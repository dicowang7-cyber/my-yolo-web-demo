// Impor TensorFlow.js di dalam Web Worker
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.2.0/dist/tf.min.js');

let model = null;
let MODEL_URL;
let INPUT_SIZE;

// --- KONSTANTA & KONFIGURASI ---
const SCORE_THRESHOLD = 0.1; 
const NMS_IOU = 0.45;
const MAX_OUTPUT = 50;
const NUM_CLASSES = 16; 

// DEFINISI NAMA KELAS AKTUAL
const CLASS_NAMES = [
    "Alpukat", "Anggur", "Apel", "Apel Hijau", "Jeruk", "Lemon", "Mangga", "Melon", 
    "Nanas", "Pepaya", "Pir", "Pisang", "Rambutan", "Salak", "Semangka", "Stroberi" 
];


// --- POST-PROCESSING KRITIS (Koreksi Bounding Box) ---

async function postprocess(outputTensor) {
    let results = [];
    
    const [finalBoxes, finalScores, finalClassIds] = tf.tidy(() => {
        
        // Output dari model: [1, 20, 8400] -> Transpose menjadi: [8400, 20]
        const transposed = outputTensor.squeeze([0]).transpose([1, 0]);

        // 1. Pisahkan Bounding Box (4) dan Class Logits (16)
        const boxes = transposed.slice([0, 0], [-1, 4]); // [8400, 4] (x, y, w, h)
        const classScores = transposed.slice([0, 4], [-1, NUM_CLASSES]); // [8400, 16] (Logits Kelas)

        // 2. Cari Max Score dan Max Class ID (Confidence Score adalah Logit Tertinggi)
        const maxScores = classScores.max(1); // [8400] 
        const classIds = classScores.argMax(1); // [8400] 
        
        // --- Filtering Awal menggunakan Threshold ---
        // Buat boolean mask [8400]
        const validScoresMask = maxScores.greater(SCORE_THRESHOLD); 
        // Ambil indeks yang valid
        const validIndices = validScoresMask.where(validScoresMask, tf.zerosLike(validScoresMask)).arraySync().map((isValid, index) => isValid === 1 ? index : -1).filter(i => i !== -1);

        if (validIndices.length === 0) {
            return [tf.tensor2d([]), tf.tensor1d([]), tf.tensor1d([])];
        }

        // Filter Boxes, Scores, dan Class IDs
        const filteredBoxes = boxes.gather(validIndices);
        const filteredScores = maxScores.gather(validIndices);
        const filteredClassIds = classIds.gather(validIndices);
        
        // 3. Ubah format dari XYWH ke XYXY (Relatif terhadap ukuran input model: 640)
        // TFJS NMS membutuhkan format [y1, x1, y2, x2]
        
        const [x, y, w, h] = tf.split(filteredBoxes, 4, 1);
        
        const x1 = tf.sub(x, tf.div(w, 2));
        const y1 = tf.sub(y, tf.div(h, 2));
        const x2 = tf.add(x, tf.div(w, 2));
        const y2 = tf.add(y, tf.div(h, 2));
        
        // Normalisasi koordinat menjadi 0-1 (sudah dilakukan karena x,y,w,h sudah 0-640)
        const nmsBoxes = tf.stack([tf.div(y1, INPUT_SIZE), tf.div(x1, INPUT_SIZE), tf.div(y2, INPUT_SIZE), tf.div(x2, INPUT_SIZE)], 1); // [N, 4] format 0-1
        
        return [nmsBoxes, filteredScores, filteredClassIds];
    });

    // 4. Non-Max Suppression (NMS)
    const selectedIdx = await tf.image.nonMaxSuppressionAsync(
        finalBoxes, // Format 0-1
        finalScores, 
        MAX_OUTPUT, 
        NMS_IOU, 
        SCORE_THRESHOLD
    );
    
    // 5. Ambil hasil akhir
    const finalBoxesArray = await finalBoxes.gather(selectedIdx).array();
    const finalScoresArray = await finalScores.gather(selectedIdx).array();
    const finalClassIdsArray = await finalClassIds.gather(selectedIdx).array();
    
    // 6. Format hasil
    for (let i = 0; i < finalScoresArray.length; i++) {
        const [y1, x1, y2, x2] = finalBoxesArray[i];
        
        results.push({
            // Koordinat dikirim kembali sebagai nilai 0-1
            x1: x1, y1: y1, x2: x2, y2: y2, 
            score: finalScoresArray[i],
            classId: finalClassIdsArray[i],
            className: CLASS_NAMES[finalClassIdsArray[i]]
        });
    }

    // Pembersihan tensor
    tf.dispose([finalBoxes, finalScores, finalClassIds, selectedIdx]);

    return results;
}


// --- Handler Inferensi di Worker ---
async function runInference(data) {
    if (!model) return;
    
    const startTime = performance.now();
    
    // 1. Buat tf.Tensor dari imageData (sudah dikirim via Transferable)
    const pixels = new Uint8ClampedArray(data.imageData);
    const imageTensor = tf.tensor4d(pixels, [data.height, data.width, 4]).slice([0, 0, 0], [-1, -1, 3]); // Hapus alpha channel (A)

    tf.engine().startScope();
    
    const input = tf.tidy(() => {
        // Pre-processing: resize, normalisasi, dan expandDims
        return imageTensor.resizeBilinear([INPUT_SIZE, INPUT_SIZE])
                          .div(255.0) // Normalisasi ke 0-1
                          .expandDims(0); // Tambah dimensi Batch
    });

    let output = null;
    try {
        const res = await model.executeAsync(input);
        output = Array.isArray(res) ? res[0] : res; // Ambil output pertama jika array
    } catch (err) {
        console.error("Model inference failed in worker:", err);
        postMessage({ type: 'ERROR', message: 'Inference failed in worker.' });
        tf.dispose([input, imageTensor]);
        tf.engine().endScope();
        return;
    }
    
    const detections = await postprocess(output);
    
    tf.dispose([input, output, imageTensor]); 
    tf.engine().endScope();
    
    const endTime = performance.now();
    const inferenceTime = endTime - startTime;
    const fps = 1000 / inferenceTime;

    // Kirim hasil deteksi kembali ke thread utama
    postMessage({
        type: 'RESULT',
        boxes: detections,
        fps: fps,
    });
}


// --- Main Message Listener ---
self.onmessage = async (event) => {
    const data = event.data;

    if (data.type === 'INIT') {
        MODEL_URL = data.modelUrl;
        INPUT_SIZE = data.inputSize;

        tf.setBackend('webgl').then(async () => {
            console.log("Worker: TFJS WebGL backend set.");
            try {
                model = await tf.loadGraphModel(MODEL_URL, {
                    onProgress: (fraction) => {
                        postMessage({ type: 'LOADING', progress: (fraction * 100) });
                    }
                });
                // Warm-up inferensi
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
