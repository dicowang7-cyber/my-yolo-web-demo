const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const statusText = document.getElementById("status");

let model;

// -----------------------------
// 1. LOAD MODEL
// -----------------------------
async function loadModel() {
  statusText.innerText = "Loading model...";
  model = await tf.loadGraphModel("model_web/model.json");
  statusText.innerText = "Model loaded!";
}

// -----------------------------
// 2. START CAMERA
// -----------------------------
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" } // back camera, change to "user" if needed
  });

  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => resolve();
  });
}

// -----------------------------
// 3. RUN YOLO INFERENCE
// -----------------------------
async function detect() {
  // Set canvas size = video size
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Mirror video, BUT NOT canvas
  // So we need to flip frame before sending to model
  const input = tf.tidy(() => {
    const frame = tf.browser.fromPixels(video);

    // UN-MIRROR frame so model gets correct orientation
    const unMirrored = frame.reverse(1);

    const resized = tf.image.resizeBilinear(unMirrored, [640, 640]);
    const casted = resized.expandDims(0).div(255);
    return casted;
  });

  const outputs = await model.executeAsync(input);

  const boxes = outputs[0].arraySync()[0];
  const scores = outputs[1].arraySync()[0];
  const classes = outputs[2].arraySync()[0];

  tf.dispose([input, outputs]);

  drawBoxes(boxes, scores, classes);

  requestAnimationFrame(detect);
}

// -----------------------------
// 4. DRAW BOXES
// -----------------------------
function drawBoxes(boxes, scores, classes) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < scores.length; i++) {
    if (scores[i] < 0.5) continue;

    const [y1, x1, y2, x2] = boxes[i];

    const left = x1 * canvas.width;
    const top = y1 * canvas.height;
    const width = (x2 - x1) * canvas.width;
    const height = (y2 - y1) * canvas.height;

    ctx.strokeStyle = "lime";
    ctx.lineWidth = 3;
    ctx.strokeRect(left, top, width, height);

    ctx.fillStyle = "lime";
    ctx.font = "18px Arial";
    ctx.fillText(
      `ID:${classes[i]} (${(scores[i] * 100).toFixed(0)}%)`,
      left,
      top - 4
    );
  }
}

// -----------------------------
// MAIN
// -----------------------------
(async () => {
  await loadModel();
  await startCamera();
  detect();
})();
