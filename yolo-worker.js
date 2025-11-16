// Impor TensorFlow.js di dalam Web Worker
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.2.0/dist/tf.min.js');

let model = null;
let INPUT_SIZE;
let SCORE_THRESHOLD; 
let NMS_IOU;
let MAX_OUTPUT;
let NUM_CLASSES;
let CLASS_NAMES;

// --- UTILITY FUNCTIONS ASLI ANDA ---

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

// --- FUNGSI POST-PROCESSING SESUAI LOGIKA ANDA ---

async function postprocess(outputTensor) {
    // Proses ini lambat karena menggunakan Array JS/CPU loop, tetapi ini adalah logika yang Anda minta.

    // 1. Dapatkan data array mentah dari output tensor
    const transposed = tf.tidy(() => outputTensor.squeeze().transpose()); 
    const data = await transposed.array();
    transposed.dispose();

    const boxes = [];
    const scores = [];
    const classIds = [];
    
    let maxOverallScore = 0;
        
    for (let i = 0; i < data.length; i++) {
        const row = data[i];
        
        const x = row[0];
        const y = row[1];
        const w = row[2];
        const h = row[3];
        
        // Logika Asli Anda:
        const classLogits = row.slice(4); 
        const probs = softmax(classLogits);
        const maxProb = Math.max(...probs);
        const classId = probs.indexOf(maxProb); 

        const finalScore = maxProb; 
        
        if (finalScore > maxOverallScore) {
            maxOverallScore = finalScore;
        }
        
        // Gunakan threshold sangat rendah
        if (finalScore < SCORE_THRESHOLD) continue; 

        const [x1, y1, x2, y2] = xywh_to_xyxy(x, y, w, h);

        // Koordinat disamakan dengan format output Anda: [y1, x1, y2, x2] dalam PIXEL (0-640)
        boxes.push([y1 * INPUT_SIZE, x1 * INPUT_SIZE, y2 * INPUT_SIZE, x2 * INPUT_SIZE]); 
        scores.push(finalScore);
        classIds.push(classId); 
    }

    console.log(`Worker Max Score: ${maxOverallScore.toFixed(6)}. Detections passed: ${boxes.length}`);

    if (boxes.length === 0) {
        return [];
    }
    
    // Non-Max Suppression (NMS) - Perlu Tensor
    const boxesTensor = tf.tensor2d(boxes);
    const scoresTensor = tf.tensor1d(scores);
    const selectedIdx = await tf.image.nonMaxSuppressionAsync(
        boxesTensor, scoresTensor, MAX_OUTPUT, NMS_IOU, SCORE_THRESHOLD
    );
    const selected = await selectedIdx.array();

    const final = [];
    for (let idx of selected) {
        const [y1, x1, y2, x2] = boxes[idx];
        final.push({
            // Koordinat dikembalikan sebagai PIXEL (0-640)
            x1: x1, y1: y1, x2: x2, y2: y2, 
            score: scores[idx],
            classId: classIds[idx],
            className: CLASS_NAMES[classIds[idx]]
        });
    }

    tf.dispose([boxesTensor, scoresTensor, selectedIdx]);

    return final;
}


// --- Handler Inferensi di Worker (Sama) ---
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
