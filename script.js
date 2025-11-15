const MODEL_URL =
  "https://xmrz7019w0uk3zke.public.blob.vercel-storage.com/model.json";

let model;

async function loadModel() {
  document.getElementById("status").textContent = "Loading model...";
  try {
    model = await tf.loadGraphModel(MODEL_URL);
    document.getElementById("status").textContent = "Model loaded successfully!";
  } catch (err) {
    console.error(err);
    document.getElementById("status").textContent =
      "Failed to load model. Check console.";
  }
}

loadModel();

// ---------------------------------------
// INPUT GAMBAR
// ---------------------------------------
document.getElementById("imgInput").addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (!file) return;

  const img = document.getElementById("preview");
  img.src = URL.createObjectURL(file);

  img.onload = () => {
    runInference(img);
  };
});

// ---------------------------------------
// INFERENSI MODEL
// ---------------------------------------
async function runInference(img) {
  if (!model) {
    alert("Model belum selesai loading.");
    return;
  }

  document.getElementById("status").textContent = "Running inference...";

  const input = tf.browser.fromPixels(img)
    .resizeNearestNeighbor([224, 224])  // sesuaikan jika model kamu beda
    .toFloat()
    .div(255.0)
    .expandDims(0);

  const prediction = await model.predict(input);
  const result = prediction.dataSync();

  document.getElementById("status").textContent = "Done!";
  document.getElementById("result").textContent = JSON.stringify(result, null, 2);

  console.log("Prediction:", result);
}
