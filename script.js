const MODEL_URL = "https://xmrz7019w0uk3zke.public.blob.vercel-storage.com/model.json";

let model;
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

async function loadModel() {
    console.log("Loading model...");
    model = await tf.loadGraphModel(MODEL_URL);
    console.log("Model loaded.");
}

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false
    });
    video.srcObject = stream;

    return new Promise(resolve => {
        video.onloadedmetadata = () => resolve(video);
    });
}

// Sigmoid
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// Convert xywh → xyxy
function xywh2xyxy(x, y, w, h) {
    return [
        x - w / 2, // x1
        y - h / 2, // y1
        x + w / 2, // x2
        y + h / 2  // y2
    ];
}

// Decode YOLO output
function decodeYOLO(tensor) {
    const arr = tensor.squeeze().transpose().arraySync(); // [8400,20]

    const results = [];

    for (let i = 0; i < arr.length; i++) {
        const row = arr[i];

        const x = row[0];
        const y = row[1];
        const w = row[2];
        const h = row[3];

        const objConf = sigmoid(row[4]);
        if (objConf < 0.4) continue;

        const classScores = row.slice(5);
        const classId = classScores.indexOf(Math.max(...classScores));

        const [x1, y1, x2, y2] = xywh2xyxy(x, y, w, h);

        results.push({
            x1, y1, x2, y2,
            score: objConf,
            classId
        });
    }

    return results;
}

function drawBoxes(boxes) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    boxes.forEach(b => {
        ctx.strokeStyle = "lime";
        ctx.lineWidth = 3;
        ctx.strokeRect(
            b.x1,
            b.y1,
            b.x2 - b.x1,
            b.y2 - b.y1
        );

        ctx.fillStyle = "lime";
        ctx.font = "18px Arial";
        ctx.fillText((b.score * 100).toFixed(1) + "%", b.x1, b.y1 - 5);
    });
}

async function detectLoop() {
    tf.engine().startScope();

    const input = tf.tidy(() =>
        tf.browser.fromPixels(video)
            .resizeBilinear([640, 640])
            .div(255)
            .expandDims(0)
    );

    const output = await model.executeAsync(input);

    const detections = decodeYOLO(output[0]);
    drawBoxes(detections);

    tf.engine().endScope();

    requestAnimationFrame(detectLoop);
}

(async () => {
    await loadModel();
    await setupCamera();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    detectLoop();
})();
