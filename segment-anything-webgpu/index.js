import {
  SamModel,
  AutoProcessor,
  RawImage,
  Tensor,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.0";

// ===============================
// CONFIG (AJUSTA AQUÍ)
// ===============================
const AUTO_FLOOR_MODE = true; // Mantener true: sin clicks
const FLOOR_Y_START = 0.45;   // 0.40-0.55: desde qué altura empieza la caja (más alto = más “solo suelo”)

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
let isDecoding = false;
let decodePending = false;
let lastPoints = null;
let isMultiMaskMode = false;
let imageInput = null;
let imageProcessed = null;
let imageEmbeddings = null;

// ===============================
// AUTO FLOOR: escoger la máscara que más “toca” el borde inferior
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

    // Cuenta píxeles activos en la última fila (tocar borde inferior)
    const y = h - 1;
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (mask.data[numMasks * i + m] === 1) bottom++;
    }

    // Área total (para desempate)
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
// DIBUJAR MÁSCARA
// ===============================
function updateMaskOverlay(mask, scores) {
  // Update canvas dimensions (if different)
  if (maskCanvas.width !== mask.width || maskCanvas.height !== mask.height) {
    maskCanvas.width = mask.width;
    maskCanvas.height = mask.height;
  }

  // Allocate buffer for pixel data
  const imageData = maskContext.createImageData(
    maskCanvas.width,
    maskCanvas.height,
  );

  // Select best mask
  const numMasks = scores.length; // normalmente 3
  let bestIndex = 0;

  if (AUTO_FLOOR_MODE) {
    bestIndex = pickBestFloorMaskIndex(mask, numMasks);
    statusLabel.textContent = `Auto-floor mask (score: ${scores[bestIndex].toFixed(2)})`;
  } else {
    for (let i = 1; i < numMasks; ++i) {
      if (scores[i] > scores[bestIndex]) {
        bestIndex = i;
      }
    }
    statusLabel.textContent = `Segment score: ${scores[bestIndex].toFixed(2)}`;
  }

  // Fill mask with colour
  const pixelData = imageData.data;
  for (let i = 0; i < pixelData.length; ++i) {
    if (mask.data[numMasks * i + bestIndex] === 1) {
      const offset = 4 * i;
      pixelData[offset] = 0; // red
      pixelData[offset + 1] = 114; // green
      pixelData[offset + 2] = 189; // blue
      pixelData[offset + 3] = 255; // alpha
    }
  }

  // Draw image data to context
  maskContext.putImageData(imageData, 0, 0);
}

// ===============================
// LIMPIAR
// ===============================
function clearPointsAndMask() {
  // Reset state
  isMultiMaskMode = false;
  lastPoints = null;

  // Remove points from previous mask (if any)
  document.querySelectorAll(".icon").forEach((e) => e.remove());

  // Disable cut button
  cutButton.disabled = true;

  // Reset mask canvas
  maskContext.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
}
clearButton.addEventListener("click", clearPointsAndMask);

resetButton.addEventListener("click", () => {
  // Reset the state
  imageInput = null;
  imageProcessed = null;
  imageEmbeddings = null;
  isEncoding = false;
  isDecoding = false;
  decodePending = false;

  // Clear points and mask (if present)
  clearPointsAndMask();

  // Update UI
  cutButton.disabled = true;
  imageContainer.style.backgroundImage = "none";
  uploadButton.style.display = "flex";
  statusLabel.textContent = "Ready";
});

// ===============================
// (Opcional) Decode por puntos: lo dejamos por compatibilidad, pero NO se usa en AUTO_FLOOR_MODE
// ===============================
async function decode() {
  if (AUTO_FLOOR_MODE) return;

  // Only proceed if we are not already decoding
  if (isDecoding) {
    decodePending = true;
    return;
  }
  isDecoding = true;

  // Prepare inputs for decoding
  const reshaped = imageProcessed.reshaped_input_sizes[0];
  const points = lastPoints
    .map((x) => [x.position[0] * reshaped[1], x.position[1] * reshaped[0]])
    .flat(Infinity);
  const labels = lastPoints.map((x) => BigInt(x.label)).flat(Infinity);

  const num_points = lastPoints.length;
  const input_points = new Tensor("float32", points, [1, 1, num_points, 2]);
  const input_labels = new Tensor("int64", labels, [1, 1, num_points]);

  // Generate the mask
  const { pred_masks, iou_scores } = await model({
    ...imageEmbeddings,
    input_points,
    input_labels,
  });

  // Post-process the mask
  const masks = await processor.post_process_masks(
    pred_masks,
    imageProcessed.original_sizes,
    imageProcessed.reshaped_input_sizes,
  );

  isDecoding = false;

  updateMaskOverlay(RawImage.fromTensor(masks[0][0]), iou_scores.data);

  // Check if another decode is pending
  if (decodePending) {
    decodePending = false;
    decode();
  }
}

// ===============================
// AUTO FLOOR: segmentar con caja inferior
// ===============================
async function autoFloorSegment() {
  if (!imageEmbeddings || !imageProcessed) return;

  statusLabel.textContent = "Auto-detecting floor...";

  // Tamaño reshaped (donde se pasan prompts a SAM)
  const reshaped = imageProcessed.reshaped_input_sizes[0]; // [h, w]
  const H = reshaped[0];
  const W = reshaped[1];

  // Caja inferior
  const x0 = 0;
  const y0 = Math.floor(H * FLOOR_Y_START);
  const x1 = W - 1;
  const y1 = H - 1;

  // SAM usa cajas como 2 puntos: [[x0,y0],[x1,y1]]
  // Shape: [1, 1, 2, 2]
  const input_boxes = new Tensor(
    "float32",
    [x0, y0, x1, y1],
    [1, 1, 2, 2],
  );

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
// ENCODE (subir imagen) + auto floor
// ===============================
async function encode(url) {
  if (isEncoding) return;
  isEncoding = true;
  statusLabel.textContent = "Extracting image embedding...";

  imageInput = await RawImage.fromURL(url);

  // Update UI
  imageContainer.style.backgroundImage = `url(${url})`;
  uploadButton.style.display = "none";
  cutButton.disabled = true;

  // Recompute image embeddings
  imageProcessed = await processor(imageInput);
  imageEmbeddings = await model.get_image_embeddings(imageProcessed);

  statusLabel.textContent = "Embedding extracted!";
  isEncoding = false;

  // AUTO FLOOR
  if (AUTO_FLOOR_MODE) {
    clearPointsAndMask();
    await autoFloorSegment();
  }
}

// Handle file selection
fileUpload.addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();

  // Set up a callback when the file is loaded
  reader.onload = (e2) => encode(e2.target.result);

  reader.readAsDataURL(file);
});

example.addEventListener("click", (e) => {
  e.preventDefault();
  encode(EXAMPLE_URL);
});

// ===============================
// EVENTOS DE RATÓN: deshabilitados en AUTO_FLOOR_MODE
// ===============================

// Clamp a value inside a range [min, max]
function clamp(x, min = 0, max = 1) {
  return Math.max(Math.min(x, max), min);
}

function getPoint(e) {
  // Get bounding box
  const bb = imageContainer.getBoundingClientRect();

  // Get the mouse coordinates relative to the container
  const mouseX = clamp((e.clientX - bb.left) / bb.width);
  const mouseY = clamp((e.clientY - bb.top) / bb.height);

  return {
    position: [mouseX, mouseY],
    label:
      e.button === 2 // right click
        ? 0 // negative prompt
        : 1, // positive prompt
  };
}

imageContainer.addEventListener("mousedown", (e) => {
  if (AUTO_FLOOR_MODE) return;

  if (e.button !== 0 && e.button !== 2) return;
  if (!imageEmbeddings) return;

  if (!isMultiMaskMode) {
    lastPoints = [];
    isMultiMaskMode = true;
    cutButton.disabled = false;
  }

  const point = getPoint(e);
  lastPoints.push(point);

  // add icon
  const icon = (point.label === 1 ? starIcon : crossIcon).cloneNode();
  icon.style.left = `${point.position[0] * 100}%`;
  icon.style.top = `${point.position[1] * 100}%`;
  imageContainer.appendChild(icon);

  decode();
});

imageContainer.addEventListener("contextmenu", (e) => e.preventDefault());

imageContainer.addEventListener("mousemove", (e) => {
  if (AUTO_FLOOR_MODE) return;

  if (!imageEmbeddings || isMultiMaskMode) return;
  lastPoints = [getPoint(e)];
  decode();
});

// Handle cut button click
cutButton.addEventListener("click", async () => {
  const [w, h] = [maskCanvas.width, maskCanvas.height];

  // Get the mask pixel data (and use this as a buffer)
  const maskImageData = maskContext.getImageData(0, 0, w, h);

  // Create a new canvas to hold the cut-out
  const cutCanvas = new OffscreenCanvas(w, h);
  const cutContext = cutCanvas.getContext("2d");

  // Copy the image pixel data to the cut canvas
  const maskPixelData = maskImageData.data;
  const imagePixelData = imageInput.data;
  for (let i = 0; i < w * h; ++i) {
    const sourceOffset = 3 * i; // RGB
    const targetOffset = 4 * i; // RGBA

    if (maskPixelData[targetOffset + 3] > 0) {
      // Only copy opaque pixels
      for (let j = 0; j < 3; ++j) {
        maskPixelData[targetOffset + j] = imagePixelData[sourceOffset + j];
      }
    }
  }
  cutContext.putImageData(maskImageData, 0, 0);

  // Download image
  const link = document.createElement("a");
  link.download = "image.png";
  link.href = URL.createObjectURL(await cutCanvas.convertToBlob());
  link.click();
  link.remove();
});

// ===============================
// LOAD MODEL
// ===============================
const model_id = "Xenova/slimsam-77-uniform";
statusLabel.textContent = "Loading model...";
const model = await SamModel.from_pretrained(model_id, {
  dtype: "fp16", // or "fp32"
  device: "webgpu",
});
const processor = await AutoProcessor.from_pretrained(model_id);
statusLabel.textContent = "Ready";

// Enable the user interface
fileUpload.disabled = false;
uploadButton.style.opacity = 1;
example.style.pointerEvents = "auto";
