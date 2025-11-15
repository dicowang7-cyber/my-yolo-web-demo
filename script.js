// Ganti kalau perlu, tapi pakai URL model.json yang sudah ada di Vercel
const MODEL_URL = "https://xmrz7019w0uk3zke.public.blob.vercel-storage.com/model.json";

const statusEl = document.getElementById('status');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let model = null;
const INPUT_SIZE = 640;   // sesuai export YOLO -> model menerima 640x640
const SCORE_THRESHOLD = 0.4;
const NMS_IOU = 0.45;
const MAX_OUTPUT = 50;    // max boxes returned by NMS

async function loadModel() {
  try {
    statusEl.textContent = "Loading model...";
    model = await tf.loadGraphModel(MODEL_URL);
    statusEl.textContent = "Model loaded.";
    console.log("Model loaded:", model);
  } catch (err) {
    console.error("Failed to load model:", err);
    statusEl.textContent = "Failed to load model. Check console.";
  }
}

async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false
    });
    video.srcObject = stream;
    await new Promise(resolve => video.onloadedmetadata = resolve);
    video.play();
    // set canvas size to video display size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  } catch (err) {
    console.error("Camera error:", err);
    statusEl.textContent = "Camera error. Allow camera and reload.";
  }
}

// utility
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a,b)=>a+b,0);
  return exps.map(e => e / sum);
}

function xywh_to_xyxy(x, y, w, h) {
  const x1 = x - w/2;
  const y1 = y - h/2;
  const x2 = x + w/2;
  const y2 = y + h/2;
  return [x1, y1, x2, y2];
}

function drawBoxes(boxes) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  boxes.forEach(b => {
    const x = b.x1 * (canvas.width / INPUT_SIZE);
    const y = b.y1 * (canvas.height / INPUT_SIZE);
    const w = (b.x2 - b.x1) * (canvas.width / INPUT_SIZE);
    const h = (b.y2 - b.y1) * (canvas.height / INPUT_SIZE);

    // Box
    ctx.strokeStyle = "lime";
    ctx.lineWidth = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 200));
    ctx.strokeRect(x, y, w, h);

    // Label bg
    const label = `${b.classId} ${(b.score*100).toFixed(1)}%`;
    ctx.font = "18px Arial";
    const textW = ctx.measureText(label).width;
    const pad = 6;
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(x, y - 22, textW + pad, 22);

    // Text
    ctx.fillStyle = "lime";
    ctx.fillText(label, x + 4, y - 6);
  });
}

async function postprocess(outputTensor) {
  // model output shape from your model: [1, 20, 8400]  -> transpose -> [8400, 20]
  let t = outputTensor;
  // if model.executeAsync returns array, caller should pass correct tensor; handled in detect loop
  // transpose and convert to JS array
  const transposed = tf.tidy(() => t.squeeze().transpose()); // [8400,20]
  const data = await transposed.array(); // array of length 8400 each length 20
  transposed.dispose();

  const boxes = [];
  const scores = [];

  for (let i = 0; i < data.length; i++) {
    const row = data[i];
    // row: [x, y, w, h, obj_conf, cls0, cls1, ...]
    // apply sigmoid for obj_conf; class logits -> softmax
    const x = row[0];
    const y = row[1];
    const w = row[2];
    const h = row[3];
    const objLogit = row[4];
    const objConf = sigmoid(objLogit);
    if (objConf < SCORE_THRESHOLD) continue;

    const classLogits = row.slice(5);
    const probs = softmax(classLogits);
    const maxProb = Math.max(...probs);
    const classId = probs.indexOf(maxProb);

    const finalScore = objConf * maxProb;
    if (finalScore < SCORE_THRESHOLD) continue;

    const [x1, y1, x2, y2] = xywh_to_xyxy(x, y, w, h);

    // push in model coordinate space (relative to INPUT_SIZE)
    boxes.push([y1 * INPUT_SIZE, x1 * INPUT_SIZE, y2 * INPUT_SIZE, x2 * INPUT_SIZE]); // note order [y1,x1,y2,x2] for TFJS NMS
    scores.push(finalScore);
  }

  if (boxes.length === 0) {
    return [];
  }

  // Run NMS to filter boxes
  const boxesTensor = tf.tensor2d(boxes); // shape [num,4]
  const scoresTensor = tf.tensor1d(scores);
  const selectedIdx = await tf.image.nonMaxSuppressionAsync(
    boxesTensor, scoresTensor, MAX_OUTPUT, NMS_IOU, SCORE_THRESHOLD
  );
  const selected = await selectedIdx.array();

  const final = [];
  for (let idx of selected) {
    // original boxes array order: [y1,x1,y2,x2]
    const [y1,x1,y2,x2] = boxes[idx];
    final.push({
      x1: x1, y1: y1, x2: x2, y2: y2,
      score: scores[idx],
      classId: 0 // we didn't keep classId per box in same arrays; could extend to keep class too
    });
  }

  boxesTensor.dispose();
  scoresTensor.dispose();
  selectedIdx.dispose();

  return final;
}

async function detectLoop() {
  if (!model) return;

  tf.engine().startScope();

  // preprocess: capture current frame, resize to INPUT_SIZE
  const input = tf.tidy(() => {
    const img = tf.browser.fromPixels(video);
    return img.resizeBilinear([INPUT_SIZE, INPUT_SIZE]).div(255.0).expandDims(0);
  });

  // model may return tensor or array; handle both
  let output = null;
  try {
    const res = await model.executeAsync(input);
    // If res is array, pick first tensor that matches expected shape, else res
    if (Array.isArray(res)) {
      output = res[0];
      // dispose rest
      for (let i = 1; i < res.length; i++) if (res[i] && res[i].dispose) res[i].dispose();
    } else {
      output = res;
    }
  } catch (err) {
    console.error("Model inference failed:", err);
    statusEl.textContent = "Inference error (see console).";
    tf.dispose(input);
    tf.engine().endScope();
    requestAnimationFrame(detectLoop);
    return;
  }

  const detections = await postprocess(output);

  // draw video frame as background (optional)
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  drawBoxes(detections);

  tf.dispose([input, output]);
  tf.engine().endScope();

  requestAnimationFrame(detectLoop);
}

(async () => {
  await loadModel();
  await setupCamera();
  // ensure canvas matches the displayed video size
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  statusEl.textContent = "Model ready — running detection.";
  detectLoop();
})();
