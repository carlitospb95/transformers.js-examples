import {
  SamModel,
  AutoProcessor,
  RawImage,
  Tensor,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.0";

// ===============================
// CONFIG
// ===============================
const AUTO_FLOOR_MODE = true; // true: automático, sin clicks
const FLOOR_Y_START = 0.45;   // 0.40-0.55: desde qué altura empieza la caja del suelo

// Si tu iPhone va justo, baja esto a 640.
// Si quieres más calidad en portátil, sube a 1024.
// OJO: esta opción puede no estar soportada por todas las versiones del processor.
// Si no funciona, lo ignorará o dará error (en ese caso te digo alternativa).
const PROCESSOR_SIZE = 768;

// Reference the elements we will use
const statusLabel = document.getElementById("status");
const fileUpload = document.getElementById("upload");
const imageContainer = document.getElementById("container");
const example = document.getElementById("example");
const uploadButton = document.getElementById("upload-button");
const resetButton = document.getElementById("reset-image");
const clearButton = document.getElementById("clear-points");
const cutButton = document.getElementById("cut-mask");
const starIcon = document.getElementById("star-icon");
const crossIcon = document.getElementById("cross-icon");
const maskCanvas = document.getElementById("mask-output");
const maskContext = maskCanvas.getContext("2d");

const EXAMPLE_URL =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/corgi.jpg";

// State variables
let isEncoding = false;
let imageInput = null;
let imageProcessed = null;
let imageEmbeddings = null;

// Keep compatibility variables (not used in auto mode)
let isDecoding = false;
let decodePending = false;
let lastPoints = null;
let isMultiMaskMode = false;

// ===============================
// AUTO FLOOR: escoger máscara que más toca borde inferior
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

    // bottom row pixels
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
// DRAW MASK
// ===============================
function updateMaskOverlay(mask, scores) {
  // Update canvas dimensions
  if (maskCanvas.width !== mask.width || maskCanvas.height !== mask.height) {
    maskCanvas.width = mask.width;
    maskCanvas.height = mask.height;
  }

  const imageData = maskContext.createImageData(maskCanvas.width, maskCanvas.height);
  const pixelData = imageData.data;

  const numMasks = scores.length; // normalmente 3
  let bestIndex = 0;

  if (AUTO_FLOOR_MODE) {
    bestIndex = pickBestFloorMaskIndex(mask, numMasks);
    statusLabel.textContent = `Auto-floor mask (score: ${scores[bestIndex].toFixed(2)})`;
  } else {
    // fallback: best iou
    for (let i = 1; i < numMasks; ++i) {
      if (scores[i] > scores[bestIndex]) bestIndex = i;
    }
    statusLabel.textContent = `Segment score: ${scores[bestIndex].toFixed(2)}`;
  }

  // Fill mask with colour (blue)
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

// ===============================
// CLEAR MASK
// ===============================
function clearPointsAndMask() {
  isMultiMaskMode = false;
  lastPoints = null;

  document.querySelectorAll(".icon").forEach((e) => e.remove());
  cutButton.disabled = true;

  maskContext.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
}
clearButton.addEventListener("click", clearPointsAndMask);

resetButton.addEventListener("click", () => {
  imageInput = null;
  imageProcessed = null;
  imageEmbeddings = null;
  isEncoding = false;
  isDecoding = false;
  decodePending = false;

  clearPointsAndMask();

  cutButton.disabled = true;
  imageContainer.style.backgroundImage = "none";
  uploadButton.style.display = "flex";
  statusLabel.textContent = "Ready";
});

// ===============================
// AUTO FLOOR SEGMENTATION (box prompt)
// ===============================
async function autoFloorSegment(model, processor) {
  if (!imageEmbeddings || !imageProcessed) return;

  statusLabel.textContent = "Auto-detecting floor...";

  const reshaped = imageProcessed.reshaped_input_sizes[0]; // [h, w]
  const H = reshaped[0];
  const W = reshaped[1];

  // box: full width, lower part
  const x0 = 0;
  const y0 = Math.floor(H * FLOOR_Y_START);
  const x1 = W - 1;
  const y1 = H - 1;

  // SAM expects boxes as 2 points: [[x0,y0],[x1,y1]] => shape [1,1,2,2]
  const input_boxes = new Tensor("float32", [x0, y0, x1, y1], [1, 1, 2, 2]);

  const { pred_masks, iou_scores } = await model({
    ...imageEmbeddings,
    input_boxes,
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
// ENCODE (upload) + auto segment
// ===============================
async function encode(url, model, processor) {
  if (isEncoding) return;
  isEncoding = true;
  statusLabel.textContent = "Extracting image embedding...";

  imageInput = await RawImage.fromURL(url);

  // UI
  imageContainer.style.backgroundImage = `url(${url})`;
  uploadButton.style.display = "none";
  cutButton.disabled = true;

  // Process + embeddings
  try {
    imageProcessed = await processor(imageInput, { size: PROCESSOR_SIZE });
  } catch (e) {
    // If processor doesn't support "size", fallback
    console.warn("Processor size option not supported, using default.", e);
    imageProcessed = await processor(imageInput);
  }

  imageEmbeddings = await model.get_image_embeddings(imageProcessed);

  statusLabel.textContent = "Embedding extracted!";
  isEncoding = false;

  if (AUTO_FLOOR_MODE) {
    clearPointsAndMask();
    await autoFloorSegment(model, processor);
  }
}

// Handle file selection
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

// Disable any mouse interactions (we don't want clicks)
imageContainer.addEventListener("contextmenu", (e) => e.preventDefault());

// Handle cut button click (downloads masked cut-out)
cutButton.addEventListener("click", async () => {
  const [w, h] = [maskCanvas.width, maskCanvas.height];

  const maskImageData = maskContext.getImageData(0, 0, w, h);

  const cutCanvas = new OffscreenCanvas(w, h);
  const cutContext = cutCanvas.getContext("2d");

  const maskPixelData = maskImageData.data;
  const imagePixelData = imageInput.data;

  for (let i = 0; i < w * h; ++i) {
    const sourceOffset = 3 * i; // RGB
    const targetOffset = 4 * i; // RGBA

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
// SMART MODEL LOADER (webgpu fp16 -> webgpu fp32 -> cpu fp32)
// ===============================
const model_id = "Xenova/slimsam-77-uniform";
statusLabel.textContent = "Loading model...";

async function loadModelSmart() {
  // Try WebGPU fp16
  try {
    const m = await SamModel.from_pretrained(model_id, {
      dtype: "fp16",
      device: "webgpu",
    });
    return { model: m, device: "webgpu", dtype: "fp16" };
  } catch (e1) {
    console.warn("WebGPU fp16 failed, trying WebGPU fp32...", e1);
  }

  // Try WebGPU fp32
  try {
    const m = await SamModel.from_pretrained(model_id, {
      dtype: "fp32",
      device: "webgpu",
    });
    return { model: m, device: "webgpu", dtype: "fp32" };
  } catch (e2) {
    console.warn("WebGPU fp32 failed, trying CPU fp32...", e2);
  }

  // CPU fallback (always)
  const m = await SamModel.from_pretrained(model_id, {
    dtype: "fp32",
    device: "cpu",
  });
  return { model: m, device: "cpu", dtype: "fp32" };
}

const { model, device, dtype } = await loadModelSmart();
const processor = await AutoProcessor.from_pretrained(model_id);

// Store globally for event handlers
window.__samModel = model;
window.__samProcessor = processor;

statusLabel.textContent = `Ready (${device}, ${dtype})`;

// Enable the user interface
fileUpload.disabled = false;
uploadButton.style.opacity = 1;
example.style.pointerEvents = "auto";
