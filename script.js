async function startCamera() {
  const video = document.getElementById("video");
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  return new Promise(resolve => video.onloadeddata = resolve);
}

async function loadModel() {
  console.log("Loading model...");
  const model = await tf.loadGraphModel("model/model.json");
  console.log("Model loaded.");
  return model;
}

async function detect(model) {
  const video = document.getElementById("video");

  setInterval(async () => {
    const input = tf.browser.fromPixels(video).expandDims(0);
    const output = await model.executeAsync(input);

    console.log(output); // Untuk testing

    tf.dispose([input, output]);
  }, 200);
}

(async () => {
  await startCamera();
  const model = await loadModel();
  detect(model);
})();
