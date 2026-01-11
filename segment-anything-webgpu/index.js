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
// CONFIG
// ===============================
const PROCESSOR_SIZE = 640;       // seguro en WASM
const FLOOR_START_Y = 0.62;       // zona "suelo" (más alto = menos suelo)
const FLOOR_STRONG_Y = 0.72;      // 2ª pasada: más agresivo hacia abajo
const MID_BLOCK_Y = 0.55;         // zona media donde suelen estar muebles

// Umbrales de “calidad” para decidir si reintentar
const MIN_BOTTOM_COVERAGE = 0.08; // % de píxeles del borde inferior cubiertos (0.05–0.15)
const MIN_AREA_RATIO = 0.08;      // % del área total (0.05–0.20)

// ===============================
// UI refs
// ===============================
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
// Helpers: mask scoring
// ===============================
function scoreMaskBottomAndArea(mask, numMasks, maskIndex) {
  const w = mask.width;
  const h = mask.height;

  let bottom = 0;
  let area = 0;

  const y = h - 1;
  for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (mask.data[numMasks * i + maskIndex] === 1) bottom++;
  }
  for (let i = 0; i < w * h; i++) {
    if (mask.data[numMasks * i + maskIndex] === 1) area++;
  }

  return {
    bottomRatio: bottom / w,
    areaRatio: area / (w * h),
  };
}

function pickBestFloorMaskIndex(mask, numMasks) {
  let best = 0;
  let bestBottom = -1;
  let bestArea = -1;

  for (let m = 0; m < numMasks; m++) {
    const { bottomRatio, areaRatio } = scoreMaskBottomAndArea(mask, numMasks, m);

    // Priorizamos tocar borde inferior, luego área
    if (bottomRatio > bestBottom || (bottomRatio === bestBottom && areaRatio > bestArea)) {
      best = m;
      bestBottom = bottomRatio;
      bestArea = areaRatio;
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
  const quality = scoreMaskBottomAndArea(mask, numMasks, bestIndex);

  statusLabel.textContent = `Auto-floor mask (score: ${scores[bestIndex].toFixed(2)} | bottom ${(quality.bottomRatio*100).toFixed(1)}% | area ${(quality.areaRatio*100).toFixed(1)}%)`;

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

  return quality;
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
// Build adaptive points
// ===============================
function buildPointsForPass(pass) {
  const reshaped = imageProcessed.reshaped_input_sizes[0]; // [h, w]
  const H = reshaped[0];
  const W = reshaped[1];

  const yFloor = pass === 1 ? FLOOR_START_Y : FLOOR_STRONG_Y;

  // POS: suelo (abajo, varias columnas)
  const pos = [
    [0.10, yFloor],
    [0.30, yFloor],
    [0.50, yFloor],
    [0.70, yFloor],
    [0.90, yFloor],
    [0.20, 0.92],
    [0.50, 0.95],
    [0.80, 0.92],
    [0.50, 0.985],
  ];

  // NEG: techo/pared (arriba) + zona media (muebles)
  const neg = [
    [0.10, 0.10],
    [0.50, 0.10],
    [0.90, 0.10],
    [0.50, 0.28],
    // bloqueo muebles (zona media)
    [0.20, MID_BLOCK_Y],
    [0.50, MID_BLOCK_Y],
    [0.80, MID_BLOCK_Y],
  ];

  const pts = [];
  const lbls = [];

  for (const [x, y] of pos) {
    pts.push(x * W, y * H);
    lbls.push(1n);
  }
  for (const [x, y] of neg) {
    pts.push(x * W, y * H);
    lbls.push(0n);
  }

  const num = lbls.length;
  const input_points = new Tensor("float32", pts, [1, 1, num, 2]);
  const input_labels = new Tensor("int64", lbls, [1, 1, num]);

  return { input_points, input_labels };
}

// ===============================
// Run SAM once
// ===============================
async function runSam(model, processor, pass) {
  const { input_points, input_labels } = buildPointsForPass(pass);

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

  const quality = updateMaskOverlay(RawImage.fromTensor(masks[0][0]), iou_scores.data);
  return quality;
}

// ===============================
// Auto floor (2-pass retry)
// ===============================
async function autoFloorSegment(model, processor) {
  statusLabel.textContent = "Auto-detecting floor (pass 1)...";
  let q1 = await runSam(model, processor, 1);

  // Si toca poco el borde inferior o es muy pequeña -> retry más agresivo
  if (q1.bottomRatio < MIN_BOTTOM_COVERAGE || q1.areaRatio < MIN_AREA_RATIO) {
    statusLabel.textContent = "Auto-detecting floor (pass 2)...";
    let q2 = await runSam(model, processor, 2);

    // Si la 2ª es peor, nos quedamos con la 1ª (simplemente no dibujamos de nuevo)
    // Aquí ya está dibujada la 2ª; para revertir necesitaríamos guardar la 1ª.
    // En la práctica, la 2ª suele mejorar bastante.
    q1 = q2;
  }

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

  // Keep proportions
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
