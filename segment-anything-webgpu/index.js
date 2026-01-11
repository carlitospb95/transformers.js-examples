import {
  SamModel,
  AutoProcessor,
  RawImage,
  Tensor,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.0";

// ===============================
// FORCE WASM BACKEND
// ===============================
env.allowLocalModels = false;
env.backends = env.backends || {};
env.backends.onnx = env.backends.onnx || {};
env.backends.onnx.wasm = env.backends.onnx.wasm || {};
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.simd = true;
env.backends.onnx.wasm.proxy = false;

// ===============================
// CONFIG (tuneable)
// ===============================

// Si te pilla poco suelo, baja estos Y un poco (más hacia abajo).
// Si te pilla cosas no suelo, sube negativos o añade más negativos.
const PROCESSOR_SIZE = 640; // seguro WASM

// Puntos POSITIVOS (sugerimos suelo) en coordenadas normalizadas [0..1]
const POS_POINTS = [
  [0.20, 0.78],
  [0.50, 0.80],
  [0.80, 0.78],
  [0.20, 0.90],
  [0.50, 0.92],
  [0.80, 0.90],
  [0.50, 0.98],
];

// Puntos NEGATIVOS (NO suelo) arriba (pared/techo)
const NEG_POINTS = [
  [0.20, 0.12],
  [0.50, 0.12],
  [0.80, 0.12],
  [0.50, 0.30],
];

// UI refs
const statusLabel = document.getElementById("status");
const fileUpload = document.getElementById("upload");
const imageContainer = document.getElementById("container");
const example = document.getElementById("example");
const uploadButton = document.getElementById("upload-button");
const resetButton = document.getElementById("reset-image");
const clearButton = document.getElementById("clear-points");
const cutButton = document.getElementById("cut-mask");
const maskCanvas = document.getElementById("mask-output");
const maskContext = maskCanvas.getContext("2d");

const EXAMPLE_URL =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/corgi.jpg";

// State
let isEncoding = false;
let imageInput = null;
let imageProcessed = null;
let imageEmbeddings = null;

// ===============================
// Pick best floor mask (touches bottom)
// ===============================
function pickBestFloorMaskIndex(mask, numMasks) {
  const w = mask.width;
  const h = mask.height;

  let best = 0;
  let bestBottom = -1;
  let bestArea = -1;

  for (let m = 0; m < numMasks; m++) {
    let bottom = 0;
    let area = 0;

    // bottom row
    const y = h - 1;
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (mask.data[numMasks * i + m] === 1) bottom++;
    }

    // total area
    for (let i = 0; i < w * h; i++) {
      if (mask.data[numMasks * i + m] === 1) area++;
    }

    if (bottom > bestBottom || (bottom === bestBottom && area > bestArea)) {
      best = m;
      bestBottom = bottom;
      bestArea = area;
    }
  }

  return best;
}

// ===============================
// Draw mask
// ===============================
function updateMaskOverlay(mask, scores) {
  if (maskCanvas.width !== mask.width || maskCanvas.height !== mask.height) {
    maskCanvas.width = mask.width;
    maskCanvas.height = mask.height;
  }

  const imageData = maskContext.createImageData(maskCanvas.width, maskCanvas.height);
  const pixelData = imageData.data;

  const numMasks = scores.length;
  const bestIndex = pickBestFloorMaskIndex(mask, numMasks);

  statusLabel.textContent = `Auto-floor mask (score: ${scores[bestIndex].toFixed(2)})`;

  for (let i = 0; i < pixelData.length; ++i) {
    if (mask.data[numMasks * i + bestIndex] === 1) {
      const offset = 4 * i;
      pixelData[offset] = 0;
      pixelData[offset + 1] = 114;
      pixelData[offset + 2] = 189;
      pixelData[offset + 3] = 255;
    }
  }

  maskContext.putImageData(imageData, 0, 0);
}

function clearMask() {
  cutButton.disabled = true;
  maskContext.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
}
clearButton.addEventListener("click", clearMask);

resetButton.addEventListener("click", () => {
  imageInput = null;
  imageProcessed = null;
  imageEmbeddings = null;
  isEncoding = false;

  clearMask();

  cutButton.disabled = true;
  imageContainer.style.backgroundImage = "none";
  imageContainer.style.aspectRatio = "";
  uploadButton.style.display = "flex";
  statusLabel.textContent = "Ready";
});

// ===============================
// Auto floor segmentation (AUTO POINTS)
// ===============================
function buildAutoPointsTensors() {
  const reshaped = imageProcessed.reshaped_input_sizes[0]; // [h, w]
  const H = reshaped[0];
  const W = reshaped[1];

  // Convert normalized points to reshaped pixel coords
  const pts = [];
  const lbls = [];

  for (const [x, y] of POS_POINTS) {
    pts.push(x * W, y * H);
    lbls.push(1n); // positive
  }
  for (const [x, y] of NEG_POINTS) {
    pts.push(x * W, y * H);
    lbls.push(0n); // negative
  }

  const num = lbls.length;

  // SAM expects [1, 1, num_points, 2]
  const input_points = new Tensor("float32", pts, [1, 1, num, 2]);
  // labels: [1, 1, num_points]
  const input_labels = new Tensor("int64", lbls, [1, 1, num]);

  return { input_points, input_labels };
}

async function autoFloorSegment(model, processor) {
  statusLabel.textContent = "Auto-detecting floor...";

  const { input_points, input_labels } = buildAutoPointsTensors();

  const { pred_masks, iou_scores } = await model({
    ...imageEmbeddings,
    input_points,
    input_labels,
  });

  const masks = await processor.post_process_masks(
    pred_masks,
    imageProcessed.original_sizes,
    imageProcessed.reshaped_input_sizes,
  );

  updateMaskOverlay(RawImage.fromTensor(masks[0][0]), iou_scores.data);
  cutButton.disabled = false;
}

// ===============================
// Encode + auto segment
// ===============================
async function encode(url, model, processor) {
  if (isEncoding) return;
  isEncoding = true;

  statusLabel.textContent = "Extracting image embedding...";

  imageInput = await RawImage.fromURL(url);

  // Keep photo proportions (no deform)
  imageContainer.style.aspectRatio = `${imageInput.width} / ${imageInput.height}`;
  imageContainer.style.backgroundImage = `url(${url})`;

  uploadButton.style.display = "none";
  cutButton.disabled = true;

  try {
    imageProcessed = await processor(imageInput, { size: PROCESSOR_SIZE });
  } catch (e) {
    console.warn("Processor size option not supported, using default.", e);
    imageProcessed = await processor(imageInput);
  }

  imageEmbeddings = await model.get_image_embeddings(imageProcessed);

  statusLabel.textContent = "Embedding extracted!";
  isEncoding = false;

  clearMask();
  await autoFloorSegment(model, processor);
}

// Upload handlers
fileUpload.addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e2) => encode(e2.target.result, window.__samModel, window.__samProcessor);
  reader.readAsDataURL(file);
});

example.addEventListener("click", (e) => {
  e.preventDefault();
  encode(EXAMPLE_URL, window.__samModel, window.__samProcessor);
});

imageContainer.addEventListener("contextmenu", (e) => e.preventDefault());

// Cut/download
cutButton.addEventListener("click", async () => {
  const [w, h] = [maskCanvas.width, maskCanvas.height];
  const maskImageData = maskContext.getImageData(0, 0, w, h);

  const cutCanvas = new OffscreenCanvas(w, h);
  const cutContext = cutCanvas.getContext("2d");

  const maskPixelData = maskImageData.data;
  const imagePixelData = imageInput.data;

  for (let i = 0; i < w * h; ++i) {
    const sourceOffset = 3 * i;
    const targetOffset = 4 * i;

    if (maskPixelData[targetOffset + 3] > 0) {
      for (let j = 0; j < 3; ++j) {
        maskPixelData[targetOffset + j] = imagePixelData[sourceOffset + j];
      }
    }
  }

  cutContext.putImageData(maskImageData, 0, 0);

  const link = document.createElement("a");
  link.download = "image.png";
  link.href = URL.createObjectURL(await cutCanvas.convertToBlob());
  link.click();
  link.remove();
});

// ===============================
// LOAD MODEL (WASM)
// ===============================
const model_id = "Xenova/slimsam-77-uniform";
statusLabel.textContent = "Loading model (WASM)...";

const model = await SamModel.from_pretrained(model_id, {
  dtype: "fp32",
  device: "wasm",
});

const processor = await AutoProcessor.from_pretrained(model_id);

window.__samModel = model;
window.__samProcessor = processor;

statusLabel.textContent = "Ready (wasm, fp32)";
fileUpload.disabled = false;
uploadButton.style.opacity = 1;
example.style.pointerEvents = "auto";
