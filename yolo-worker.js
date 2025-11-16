// Impor TensorFlow.js di dalam Web Worker
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.2.0/dist/tf.min.js');

let model = null;
let INPUT_SIZE;
let SCORE_THRESHOLD;
let NMS_IOU;
let MAX_OUTPUT;
let NUM_CLASSES;
let CLASS_NAMES;

// --- POST-PROCESSING KRITIS (YOLOv8 Fixed) ---

async function postprocess(outputTensor) {
    let results = [];
    
    const [finalBoxes, finalScores, finalClassIds] = tf.tidy(() => {
        
        // Output: [1, 20, 8400] -> Transpose menjadi: [8400, 20]
        const transposed = outputTensor.squeeze([0]).transpose([1, 0]);

        // 1. Pisahkan Bounding Box (4) dan Class Logits (16)
        // YOLOv8 standar: 4 Box + N Kelas
        const boxes = transposed.slice([0, 0], [-1, 4]); 
        const classScores = transposed.slice([0, 4], [-1, NUM_CLASSES]); 

        // 2. Cari Max Score dan Max Class ID
        // Skor tertinggi di antara logits kelas adalah skor objek/kelas
        const maxScores = classScores.max(1); 
        const classIds = classScores.argMax(1); 
        
        // --- Filtering Awal menggunakan Threshold ---
        const validScoresMask = maxScores.greater(SCORE_THRESHOLD); 
        const validIndices = validScoresMask.where(validScoresMask, tf.zerosLike(validScoresMask)).arraySync().map((isValid, index) => isValid === 1 ? index : -1).filter(i => i !== -1);

        if (validIndices.length === 0) {
            return [tf.tensor2d([]), tf.tensor1d([]), tf.tensor1d([])];
        }

        const filteredBoxes = boxes.gather(validIndices);
        const filteredScores = maxScores.gather(validIndices);
        const filteredClassIds = classIds.gather(validIndices);
        
        // 3. Ubah format dari XYWH ke XYXY (Normalisasi ke 0-1 untuk NMS)
        // TFJS NMS membutuhkan format [y1, x1, y2, x2]
        
        const [x, y, w, h] = tf.split(filteredBoxes, 4, 1);
        
        const x1 = tf.sub(x, tf.div(w, 2));
        const y1 = tf.sub(y, tf.div(h, 2));
        const x2 = tf.add(x, tf.div(w, 2));
        const y2 = tf.add(y, tf.div(h, 2));
        
        // Gabungkan dalam format NMS [y1, x1, y2, x2] (Normalisasi koordinat 0-1)
        const nmsBoxes = tf.stack([tf.div(y1, INPUT_SIZE), tf.div(x1, INPUT_SIZE), tf.div(y2, INPUT_SIZE), tf.div(x2, INPUT_SIZE)], 1);
        
        return [nmsBoxes, filteredScores, filteredClassIds];
    });

    // 4. Non-Max Suppression (NMS)
    const selectedIdx = await tf.image.nonMaxSuppressionAsync(
        finalBoxes, finalScores, MAX_OUTPUT, NMS_IOU, SCORE_THRESHOLD
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

    tf.dispose([finalBoxes, finalScores, finalClassIds, selectedIdx]);
    return results;
}


// --- Handler Inferensi di Worker ---
async function runInference(data) {
    if (!model) return;
    
    const startTime = performance.now();
    const pixels = new Uint8ClampedArray(data.imageData);
    const imageTensor = tf.tensor4d(pixels, [data.height, data.width, 4]).slice([0, 0, 0], [-1, -1, 3]);

    tf.engine().startScope();
    
    const input = tf.tidy(() => {
        return imageTensor.resizeBilinear([INPUT_SIZE, INPUT_SIZE])
                          .div(255.0) 
                          .expandDims(0); 
    });

    let output = null;
    try {
        const res = await model.executeAsync(input);
        output = Array.isArray(res) ? res[0] : res;
    } catch (err) {
        console.error("Inference failed in worker:", err);
        tf.dispose([input, imageTensor]);
        tf.engine().endScope();
        return;
    }
    
    const detections = await postprocess(output);
    
    tf.dispose([input, output, imageTensor]); 
    tf.engine().endScope();
    
    const inferenceTime = performance.now() - startTime;

    postMessage({
        type: 'RESULT',
        boxes: detections,
        fps: 1000 / inferenceTime,
    });
}


// --- Main Message Listener ---
self.onmessage = async (event) => {
    const data = event.data;

    if (data.type === 'INIT') {
        // Terima konfigurasi dari main thread
        INPUT_SIZE = data.inputSize;
        SCORE_THRESHOLD = data.scoreThreshold;
        NMS_IOU = data.nmsIou;
        MAX_OUTPUT = data.maxOutput;
        NUM_CLASSES = data.numClasses;
        CLASS_NAMES = data.classNames;

        tf.setBackend('webgl').then(async () => {
            try {
                model = await tf.loadGraphModel(data.modelUrl, {
                    onProgress: (fraction) => {
                        postMessage({ type: 'LOADING', progress: (fraction * 100) });
                    }
                });
                // Warm-up
                model.predict(tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3])).dispose();
                postMessage({ type: 'LOADED' });
            } catch (err) {
                postMessage({ type: 'ERROR', message: 'Failed to load model in worker.' });
            }
        });
    } else if (data.type === 'INFER') {
        runInference(data);
    }
};
