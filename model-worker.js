/**
 * webml-kit Browser Worker (model-worker.js)
 *
 * Implements the webml-kit WorkerCommand / WorkerResponse protocol with:
 * - Strict per-pipeline mutex lock (withLock) so two ONNX InferenceSession.run()
 *   calls NEVER overlap ("Error: Session already started" is impossible).
 * - InterruptableStoppingCriteria + cancelEpoch so interrupting a stream or
 *   queued TTS synthesis drains the queue cleanly before starting new work.
 * - Multi-file download progress aggregation.
 * - Support for text-generation (Qwen / SmolLM2 / Llama) and text-to-speech (Kokoro-82M).
 */

const _origWarn = console.warn.bind(console);
const _origError = console.error.bind(console);
function isBenignOrtNotice(args) {
  const first = typeof args[0] === 'string' ? args[0] : '';
  return (
    first.includes('Unable to determine content-length') ||
    first.includes('VerifyEachNodeIsAssignedToAnEp') ||
    first.includes('Some nodes were not assigned') ||
    first.includes('Rerunning with verbose output') ||
    first.includes('Unable to add response to browser cache')
  );
}
console.warn = (...args) => {
  if (isBenignOrtNotice(args)) return;
  _origWarn(...args);
};
console.error = (...args) => {
  if (isBenignOrtNotice(args)) return;
  _origError(...args);
};

import {
  pipeline as hfPipeline,
  TextStreamer,
  InterruptableStoppingCriteria,
  env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0';

env.allowLocalModels = false;
env.useBrowserCache = true;
if (env.backends?.onnx) {
  env.backends.onnx.logLevel = 'error';
  if (env.backends.onnx.wasm && !self.crossOriginIsolated) {
    env.backends.onnx.wasm.numThreads = 1;
  }
}

const instances = new Map();
const executionLocks = new Map();
const kvCaches = new Map();
const stoppingCriteria = new InterruptableStoppingCriteria();
let cancelEpoch = 0;

function modelKey(task, modelId) {
  return `${task}::${modelId}`;
}

function send(msg, transfer) {
  if (transfer && transfer.length > 0) {
    self.postMessage(msg, transfer);
  } else {
    self.postMessage(msg);
  }
}

/**
 * Serializes all async operations on a pipeline so two ONNX session.run()
 * calls can never execute concurrently on the same session/backend.
 */
function withLock(key, fn) {
  const prev = executionLocks.get(key) ?? Promise.resolve();
  const next = prev.catch(() => {}).then(fn);
  executionLocks.set(key, next);
  return next;
}

async function checkDevice() {
  let backend = 'cpu';
  let gpu = null;

  try {
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });
      if (adapter) {
        backend = 'webgpu';
        const info = adapter.info;
        const vram = Number(adapter.limits?.maxBufferSize ?? 0);
        const fmtVram =
          vram >= 1024 ** 3
            ? `${(vram / 1024 ** 3).toFixed(1)} GB`
            : `${(vram / 1024 ** 2).toFixed(1)} MB`;
        gpu = {
          vendor: info?.vendor ?? 'unknown',
          architecture: info?.architecture ?? 'unknown',
          description: info?.description ?? 'unknown',
          vram,
          vramFormatted: fmtVram,
        };
      }
    }
  } catch {}

  if (backend === 'cpu' && typeof WebAssembly !== 'undefined') {
    backend = 'wasm';
  }

  const vram = gpu?.vram ?? 0;
  const recommendedDtype = vram >= 8e9 ? 'fp16' : vram >= 4e9 ? 'q8' : 'q4';

  send({ type: 'device-info', data: { backend, gpu, recommendedDtype } });
}

function createProgressAggregator() {
  const files = new Map();
  return (info) => {
    if (!info) return;
    if ((info.status === 'progress' || info.status === 'download') && info.file) {
      files.set(info.file, {
        loaded: Number(info.loaded ?? 0),
        total: Number(info.total ?? 0),
        progress: Number(info.progress ?? 0),
      });
      let loaded = 0;
      let total = 0;
      for (const f of files.values()) {
        loaded += f.loaded;
        total += f.total;
      }
      const percent =
        total > 0
          ? Math.min(99, Math.round((loaded / total) * 100))
          : Math.round(info.progress ?? 0);
      send({
        type: 'progress',
        data: {
          status: 'downloading',
          file: String(info.file),
          loaded,
          total,
          percent,
        },
      });
    }
  };
}

async function loadPipeline(config) {
  const key = modelKey(config.task, config.modelId);

  if (kvCaches.has(key)) {
    kvCaches.get(key)?.dispose?.();
    kvCaches.delete(key);
  }

  if (!instances.has(key)) {
    const onProgress = createProgressAggregator();

    const loadFn = () =>
      withLock(key, async () => {
        send({
          type: 'progress',
          data: { status: 'downloading', loaded: 0, total: 0, percent: 0 },
        });

        // Kokoro-82M neural TTS
        if (config.modelId.toLowerCase().includes('kokoro')) {
          const { KokoroTTS } = await import(
            'https://cdn.jsdelivr.net/npm/kokoro-js@1.2.0/+esm'
          );
          const tts = await KokoroTTS.from_pretrained(config.modelId, {
            dtype: config.dtype ?? 'q8',
            device: config.device ?? 'wasm',
            progress_callback: onProgress,
          });
          return tts;
        }

        const preferredDevice = config.device ?? 'webgpu';
        const preferredDtype = config.dtype ?? 'q4';

        let pipe;
        try {
          pipe = await hfPipeline(config.task, config.modelId, {
            device: preferredDevice,
            dtype: preferredDtype,
            revision: config.revision,
            session_options: { logSeverityLevel: 3 },
            progress_callback: onProgress,
          });
        } catch (_) {
          pipe = await hfPipeline(config.task, config.modelId, {
            device: 'wasm',
            dtype: 'q4',
            revision: config.revision,
            session_options: { logSeverityLevel: 3 },
            progress_callback: onProgress,
          });
        }

        // Warmup inside loadFn lock so no runInference call can ever overlap warmup
        if (config.task === 'text-generation' && pipe && 'tokenizer' in pipe) {
          send({
            type: 'progress',
            data: { status: 'compiling', loaded: 0, total: 0, percent: 100 },
          });
          try {
            const inputs = pipe.tokenizer('a');
            await pipe.model.generate({
              ...inputs,
              max_new_tokens: 1,
            });
          } catch {}
        }

        return pipe;
      });

    instances.set(key, loadFn());
  }

  try {
    await instances.get(key);
    send({ type: 'ready', modelKey: key });
  } catch (e) {
    instances.delete(key);
    send({
      type: 'error',
      id: 'load',
      data: e instanceof Error ? e.message : String(e),
    });
  }
}

async function runInference(id, task, input, options) {
  let pipelinePromise = null;
  let matchedKey = '';

  for (const [key, promise] of instances) {
    if (key.startsWith(`${task}::`)) {
      pipelinePromise = promise;
      matchedKey = key;
      break;
    }
  }

  if (!pipelinePromise) {
    send({ type: 'error', id, data: `No pipeline loaded for task: ${task}` });
    return;
  }

  const requestEpoch = cancelEpoch;

  // If a new text-generation request arrives while an old one is running,
  // interrupt the active generation so the lock releases after the current token.
  if (task === 'text-generation') {
    stoppingCriteria.interrupt();
  }

  await withLock(matchedKey, async () => {
    // Skip stale queued requests that were cancelled while waiting for the lock
    if (requestEpoch !== cancelEpoch) {
      send({
        type: 'result',
        id,
        data: { text: '', aborted: true, numTokens: 0, tps: 0, timeToFirstToken: 0, totalTime: 0 },
      });
      return;
    }

    try {
      const pipelineInstance = await pipelinePromise;

      if (requestEpoch !== cancelEpoch) {
        send({
          type: 'result',
          id,
          data: { text: '', aborted: true, numTokens: 0, tps: 0, timeToFirstToken: 0, totalTime: 0 },
        });
        return;
      }

      if (task === 'text-generation') {
        await runTextGeneration(id, pipelineInstance, input, options, requestEpoch);
      } else if (
        task === 'text-to-speech' &&
        pipelineInstance &&
        typeof pipelineInstance.generate === 'function'
      ) {
        const voice = options?.voice || 'af_heart';
        const speed = options?.speed || 1.0;
        const raw = await pipelineInstance.generate(String(input), { voice, speed });
        if (requestEpoch !== cancelEpoch) {
          send({ type: 'result', id, data: { aborted: true } });
          return;
        }
        const samples = new Float32Array(raw.audio);
        send(
          {
            type: 'result',
            id,
            data: {
              audio: samples,
              sampling_rate: raw.sampling_rate || 24000,
            },
          },
          [samples.buffer]
        );
      } else {
        const result = await pipelineInstance(input, options);
        send({ type: 'result', id, data: result });
      }
    } catch (e) {
      send({
        type: 'error',
        id,
        data: e instanceof Error ? e.message : String(e),
      });
    }
  });
}

async function runTextGeneration(id, generator, messages, options, requestEpoch) {
  let startTime = null;
  let numTokens = 0;
  let tps = 0;
  let firstTokenTime = 0;

  stoppingCriteria.reset();

  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (output) => {
      if (requestEpoch !== cancelEpoch) return;
      send({
        type: 'token',
        id,
        data: {
          token: output,
          tps,
          numTokens,
          timeToFirstToken: firstTokenTime,
        },
      });
    },
    token_callback_function: () => {
      const now = performance.now();
      if (startTime === null) {
        startTime = now;
        firstTokenTime = now;
      }
      numTokens++;
      if (numTokens > 1 && startTime !== null) {
        tps = (numTokens / (now - startTime)) * 1000;
      }
    },
  });

  const doSample =
    options?.doSample ?? options?.do_sample ?? false;

  const genOptions = {
    max_new_tokens: options?.maxNewTokens ?? options?.max_new_tokens ?? 512,
    do_sample: doSample,
    ...(doSample
      ? {
          temperature: options?.temperature,
          top_p: options?.topP ?? options?.top_p,
          top_k: options?.topK ?? options?.top_k,
          repetition_penalty: options?.repetitionPenalty ?? options?.repetition_penalty,
        }
      : {}),
    streamer,
    stopping_criteria: stoppingCriteria,
  };

  for (const k of Object.keys(genOptions)) {
    if (genOptions[k] === undefined) delete genOptions[k];
  }

  const output = await generator(messages, genOptions);

  let finalText = '';
  if (Array.isArray(output) && output[0]?.generated_text) {
    const genText = output[0].generated_text;
    finalText = Array.isArray(genText)
      ? genText[genText.length - 1]?.content ?? ''
      : genText;
  }

  send({
    type: 'result',
    id,
    data: {
      text: finalText,
      numTokens,
      tps,
      timeToFirstToken: firstTokenTime,
      totalTime: startTime ? performance.now() - startTime : 0,
    },
  });
}

function resetKVCache() {
  cancelEpoch++;
  stoppingCriteria.interrupt();
  for (const cache of kvCaches.values()) {
    cache?.dispose?.();
  }
  kvCaches.clear();
}

function disposeModel(targetKey) {
  cancelEpoch++;
  stoppingCriteria.interrupt();
  if (targetKey) {
    const p = instances.get(targetKey);
    p?.then((inst) => inst?.dispose?.());
    instances.delete(targetKey);
    if (kvCaches.has(targetKey)) {
      kvCaches.get(targetKey)?.dispose?.();
      kvCaches.delete(targetKey);
    }
  } else {
    for (const p of instances.values()) {
      p.then((inst) => inst?.dispose?.());
    }
    instances.clear();
    for (const cache of kvCaches.values()) {
      cache?.dispose?.();
    }
    kvCaches.clear();
  }
}

self.addEventListener('message', async (e) => {
  const cmd = e.data;
  if (!cmd || !cmd.type) return;

  switch (cmd.type) {
    case 'check':
      await checkDevice();
      break;
    case 'load':
      await loadPipeline(cmd.config);
      break;
    case 'run':
      await runInference(cmd.id, cmd.task, cmd.input, cmd.options);
      break;
    case 'interrupt':
      cancelEpoch++;
      stoppingCriteria.interrupt();
      break;
    case 'reset':
      resetKVCache();
      break;
    case 'dispose':
      disposeModel(cmd.modelKey);
      break;
  }
});
