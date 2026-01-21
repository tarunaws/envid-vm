/* eslint-disable no-console */

const express = require('express');
let tf = null;
let tfNode = null;
try {
  // Prefer the TensorFlow C++ backend when available.
  // This can fail on some platforms (e.g. linux/arm64) if prebuilt binaries aren't present.
  // It can also fail on very new Node versions.
  // In those cases we fall back to pure JS tfjs.
  // eslint-disable-next-line global-require
  tfNode = require('@tensorflow/tfjs-node');
  tf = tfNode;
} catch (err) {
  // eslint-disable-next-line global-require
  tf = require('@tensorflow/tfjs');
  // eslint-disable-next-line global-require
  require('@tensorflow/tfjs-backend-cpu');
}

const jpeg = require('jpeg-js');
const { PNG } = require('pngjs');
const nsfwjs = require('nsfwjs');

const app = express();
app.use(express.json({ limit: process.env.JSON_LIMIT || '50mb' }));

const PORT = parseInt(process.env.PORT || '5082', 10);
const INCLUDE_MODEL_DETAILS = String(process.env.INCLUDE_MODEL_DETAILS || '').trim() === '1';

let modelPromise = null;
let modelLoaded = false;
let modelLoadError = null;
async function getModel() {
  if (!modelPromise) {
    if (typeof tf.ready === 'function') {
      await tf.ready();
    }

    if (!tfNode && typeof tf.setBackend === 'function') {
      await tf.setBackend('cpu');
    }

    modelPromise = nsfwjs
      .load()
      .then((m) => {
        modelLoaded = true;
        return m;
      })
      .catch((err) => {
        modelLoadError = err;
        throw err;
      });
  }
  return modelPromise;
}

function likelihoodFromScore(score) {
  const s = Number.isFinite(score) ? score : 0.0;
  if (s >= 0.9) return 'VERY_LIKELY';
  if (s >= 0.7) return 'LIKELY';
  if (s >= 0.4) return 'POSSIBLE';
  if (s >= 0.2) return 'UNLIKELY';
  return 'VERY_UNLIKELY';
}

function severityFromScore(score) {
  // Neutral, model-agnostic scale (no provider-specific buckets).
  const s = Number.isFinite(score) ? score : 0.0;
  if (s >= 0.9) return 'critical';
  if (s >= 0.7) return 'high';
  if (s >= 0.4) return 'medium';
  if (s >= 0.2) return 'low';
  return 'minimal';
}

function computeUnsafeProbability(classifications) {
  // nsfwjs returns array like: [{className: 'Porn', probability: 0.01}, ...]
  const unsafeClasses = new Set(['Porn', 'Hentai', 'Sexy']);
  let unsafe = 0.0;
  for (const c of classifications || []) {
    if (unsafeClasses.has(c.className)) unsafe += Number(c.probability) || 0.0;
  }
  if (!Number.isFinite(unsafe)) unsafe = 0.0;
  if (unsafe < 0) unsafe = 0.0;
  if (unsafe > 1) unsafe = 1.0;
  return unsafe;
}

function decodeImageToTensor(buffer, mimeType) {
  if (tfNode && tfNode.node && typeof tfNode.node.decodeImage === 'function') {
    return tfNode.node.decodeImage(buffer, 3);
  }

  const mime = (mimeType || '').toLowerCase();
  if (mime.includes('png')) {
    const png = PNG.sync.read(buffer);
    const { width, height, data } = png; // RGBA

    const rgb = new Uint8Array(width * height * 3);
    for (let i = 0, j = 0; i < data.length; i += 4) {
      rgb[j++] = data[i];
      rgb[j++] = data[i + 1];
      rgb[j++] = data[i + 2];
    }
    return tf.tensor3d(rgb, [height, width, 3], 'int32');
  }

  // Default to JPEG.
  const decoded = jpeg.decode(buffer, { useTArray: true });
  const { width, height, data } = decoded; // RGBA

  const rgb = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < data.length; i += 4) {
    rgb[j++] = data[i];
    rgb[j++] = data[i + 1];
    rgb[j++] = data[i + 2];
  }

  return tf.tensor3d(rgb, [height, width, 3], 'int32');
}

app.get('/health', async (req, res) => {
  res.json({
    status: modelLoadError ? 'degraded' : 'ok',
    service: 'local-moderation-nsfwjs',
    has_nsfwjs: true,
    tf_backend: tf.getBackend(),
    model_loaded: modelLoaded,
    model_loading: !!modelPromise && !modelLoaded && !modelLoadError,
    model_error: modelLoadError ? String(modelLoadError && modelLoadError.message ? modelLoadError.message : modelLoadError) : null,
  });
});

app.post('/moderate/frames', async (req, res) => {
  try {
    const body = req.body || {};
    const frames = Array.isArray(body.frames) ? body.frames : [];
    const model = await getModel();

    const explicit_frames = [];

    for (const f of frames) {
      const t = Number(f && f.time);
      const image_b64 = (f && f.image_b64) || '';
      const image_mime = (f && f.image_mime) || 'image/jpeg';
      if (!image_b64) continue;

      const buffer = Buffer.from(image_b64, 'base64');

      // Returns int32 [h,w,3] in RGB
      const imageTensor = decodeImageToTensor(buffer, image_mime);
      try {
        // model.classify expects a tf.Tensor3D
        const classifications = await model.classify(imageTensor);
        const unsafe = computeUnsafeProbability(classifications);
        const safe = Math.max(0.0, Math.min(1.0, 1.0 - unsafe));

        explicit_frames.push({
          time: Number.isFinite(t) ? t : 0.0,
          severity: severityFromScore(unsafe),
          unsafe,
          safe,
          ...(INCLUDE_MODEL_DETAILS ? { model_details: { categories: classifications } } : {}),
        });
      } finally {
        imageTensor.dispose();
      }
    }

    res.json({ explicit_frames });
  } catch (err) {
    res.status(500).json({ error: String(err && err.message ? err.message : err) });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`local-moderation-nsfwjs listening on ${PORT}`);

  // Warm-load the model in the background so first requests are fast.
  getModel()
    .then(() => console.log('nsfwjs model loaded'))
    .catch((err) => console.warn('nsfwjs model load failed:', err && err.message ? err.message : err));
});
