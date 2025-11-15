const MODEL_URL = "https://xmrz7019w0uk3zke.public.blob.vercel-storage.com/model.json";

let model;
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

async function loadModel() {
    console.log("Loading model...");
    model = await tf.loadGraphModel(MODEL_URL);
    console.log("Model loaded");
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

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function decodeYOLO(output) {
    // output shape: [1, 20, 8400]
    const data = output.squeeze().transpose().arraySync(); // [8400, 20]

    const boxes = [];

    data.forEach(d => {
        const [x, y, w, h] = d.slice(0, 4);
        const obj = sigmoid(d[4]);

        if (obj < 0.4) return; // threshold

        const classScores = d.slice(5);
        const classId = classScores.indexOf(Math.max(...classScores));

        const left = x - w / 2;
        const top = y - h / 2;

        boxes.push({
            x: left,
            y: top,
            width: w,
            height: h,
            score: obj,
            classId: classId
        });
    });

    return boxes;
}

function drawBoxes(boxes) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 3;
    ctx.font = "18px Arial";

    boxes.forEach(b => {
        ctx.strokeRect(b.x, b.y, b.width, b.height);
        ctx.fillText((b.score * 100).toFixed(1) + "%", b.x, b.y - 5);
    });
}

async function detectFrame() {
    tf.engine().startScope();

    const input = tf.browser.fromPixels(video)
        .resizeNearestNeighbor([640, 640])
        .expandDims(0)
        .toFloat();

    const result = await model.executeAsync(input);
    const boxes = decodeYOLO(result[0]);

    drawBoxes(boxes);

    tf.engine().endScope();
    requestAnimationFrame(detectFrame);
}

(async () => {
    await loadModel();
    await setupCamera();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    detectFrame();
})();
