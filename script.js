const MODEL_URL = "https://xmrz7019w0uk3zke.public.blob.vercel-storage.com/model.json";

let model;
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

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

async function loadModel() {
    console.log("Loading model...");
    model = await tf.loadGraphModel(MODEL_URL);
    console.log("Model loaded.");
}

function drawBoxes(predictions) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    predictions.forEach(box => {
        const [x1, y1, x2, y2, score, classId] = box;

        ctx.strokeStyle = "lime";
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.fillStyle = "lime";
        ctx.font = "18px Arial";
        ctx.fillText(`${(score * 100).toFixed(1)}%`, x1, y1 - 5);
    });
}

async function detectFrame() {
    tf.engine().startScope();

    const input = tf.browser.fromPixels(video).expandDims(0).toFloat();
    const preds = await model.executeAsync(input);

    const boxes = preds[0].arraySync(); // YOLO output shape (N, 6)
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
