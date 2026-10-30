/**
 * webml-kit Browser Worker (model-worker.js)
 *
 * Implements the webml-kit WorkerCommand / WorkerResponse protocol with:
 * - Single Worker-wide ONNX mutex (withWorkerLock) so NO two ONNX Runtime
 *   session creations, warmups, disposes, or InferenceSession.run() calls can
 *   ever overlap inside the Worker's shared WASM/JSEP runtime ("Error: Session
 *   already started" and "Error: Session mismatch" are structurally impossible).
 * - One active pipeline per task with clean lock-serialized disposal on model switch.
 * - InterruptableStoppingCriteria + cancelEpoch so interrupting a stream or
 *   queued TTS synthesis drains cleanly before starting new work.
 * - Multi-file download progress aggregation.
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
  if (env.backends.onnx.wasm) {
    env.backends.onnx.wasm.proxy = false;
    if (!self.crossOriginIsolated) {
      env.backends.onnx.wasm.numThreads = 1;
    }
  }
}

// Active loaded pipeline per task: Map<task, { key: string, instance: any }>
const activeByTask = new Map();
const kvCaches = new Map();
const stoppingCriteria = new InterruptableStoppingCriteria();
let workerLock = Promise.resolve();
let cancelEpoch = 0;

function makeModelKey(task, modelId) {
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
 * Serializes ALL ONNX Runtime operations across the entire Worker so two
 * session.run() / create / dispose calls can never overlap on the shared
 * ONNX WASM/JSEP runtime state.
 */
function withWorkerLock(fn) {
  const next = workerLock.catch(() => {}).then(fn);
  workerLock = next;
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

async function safelyDisposeInstance(inst) {
  if (!inst) return;
  try {
    await inst.dispose?.();
  } catch {}
  try {
    await inst.model?.dispose?.();
  } catch {}
}

async function loadPipeline(config, reqId = 'load') {
  const key = makeModelKey(config.task, config.modelId);

  if (config.task === 'text-generation') {
    stoppingCriteria.interrupt();
  }

  await withWorkerLock(async () => {
    const existing = activeByTask.get(config.task);
    if (existing && existing.key === key && existing.instance) {
      send({ type: 'ready', id: reqId, modelKey: key });
      return;
    }

    // Cleanly dispose any prior model on this task inside the global lock
    if (existing) {
      activeByTask.delete(config.task);
      if (kvCaches.has(existing.key)) {
        try {
          kvCaches.get(existing.key)?.dispose?.();
        } catch {}
        kvCaches.delete(existing.key);
      }
      await safelyDisposeInstance(existing.instance);
    }

    const onProgress = createProgressAggregator();

    try {
      send({
        type: 'progress',
        data: { status: 'downloading', loaded: 0, total: 0, percent: 0 },
      });

      let instance;
      if (config.modelId.toLowerCase().includes('kokoro')) {
        const { KokoroTTS } = await import(
          'https://cdn.jsdelivr.net/npm/kokoro-js@1.2.0/+esm'
        );
        const wantWebGPU = config.device === 'webgpu';
        let loadedOnWebGPU = false;
        try {
          instance = await KokoroTTS.from_pretrained(config.modelId, {
            dtype: wantWebGPU ? (config.dtype ?? 'fp32') : (config.dtype ?? 'q8'),
            device: wantWebGPU ? 'webgpu' : 'wasm',
            progress_callback: onProgress,
          });
          loadedOnWebGPU = wantWebGPU;
        } catch (_) {
          instance = await KokoroTTS.from_pretrained(config.modelId, {
            dtype: 'q8',
            device: 'wasm',
            progress_callback: onProgress,
          });
        }
        // Pre-warm voice embedding + WGSL compute shaders during background load so first speech is <350ms
        if (loadedOnWebGPU && instance && typeof instance.generate === 'function') {
          try {
            await instance.generate('Steady.', { voice: 'af_heart', speed: 1.0 });
          } catch {}
        }
      } else {
        const preferredDevice = config.device ?? 'wasm';
        // Transformers.js WebGPU requires q4f16/fp16 rather than CPU-oriented q4 MatMulNBits
        const preferredDtype =
          preferredDevice === 'webgpu' && (!config.dtype || config.dtype === 'q4')
            ? 'q4f16'
            : config.dtype ?? 'q4';

        try {
          instance = await hfPipeline(config.task, config.modelId, {
            device: preferredDevice,
            dtype: preferredDtype,
            revision: config.revision,
            session_options: { logSeverityLevel: 3 },
            progress_callback: onProgress,
          });
        } catch (_) {
          instance = await hfPipeline(config.task, config.modelId, {
            device: 'wasm',
            dtype: 'q4',
            revision: config.revision,
            session_options: { logSeverityLevel: 3 },
            progress_callback: onProgress,
          });
        }

        // Warmup inside global worker lock so no runInference call can ever overlap warmup
        if (config.task === 'text-generation' && instance && 'tokenizer' in instance) {
          send({
            type: 'progress',
            data: { status: 'compiling', loaded: 0, total: 0, percent: 100 },
          });
          try {
            const inputs = instance.tokenizer('a');
            await instance.model.generate({
              ...inputs,
              max_new_tokens: 1,
            });
          } catch {}
        }
      }

      activeByTask.set(config.task, { key, instance });
      send({ type: 'ready', id: reqId, modelKey: key });
    } catch (e) {
      activeByTask.delete(config.task);
      send({
        type: 'error',
        id: reqId,
        data: e instanceof Error ? e.message : String(e),
      });
    }
  });
}

async function runInference(id, task, input, options) {
  const requestEpoch = cancelEpoch;

  // If a new text-generation request arrives while an old one is running,
  // interrupt the active generation so the global lock releases after the current token.
  if (task === 'text-generation') {
    stoppingCriteria.interrupt();
  }

  await withWorkerLock(async () => {
    if (requestEpoch !== cancelEpoch) {
      send({
        type: 'result',
        id,
        data: { text: '', aborted: true, numTokens: 0, tps: 0, timeToFirstToken: 0, totalTime: 0 },
      });
      return;
    }

    const entry = activeByTask.get(task);
    if (!entry || !entry.instance) {
      send({ type: 'error', id, data: `No pipeline loaded for task: ${task}` });
      return;
    }

    const pipelineInstance = entry.instance;

    try {
      if (task === 'text-generation') {
        await runTextGeneration(id, pipelineInstance, input, options, requestEpoch);
      } else if (
        task === 'text-to-speech' &&
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

  const doSample = options?.doSample ?? options?.do_sample ?? false;

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
  withWorkerLock(async () => {
    for (const cache of kvCaches.values()) {
      try {
        cache?.dispose?.();
      } catch {}
    }
    kvCaches.clear();
  });
}

function disposeModel(targetKey) {
  cancelEpoch++;
  stoppingCriteria.interrupt();
  withWorkerLock(async () => {
    if (targetKey) {
      for (const [task, entry] of activeByTask.entries()) {
        if (entry.key === targetKey) {
          activeByTask.delete(task);
          await safelyDisposeInstance(entry.instance);
        }
      }
      if (kvCaches.has(targetKey)) {
        try {
          kvCaches.get(targetKey)?.dispose?.();
        } catch {}
        kvCaches.delete(targetKey);
      }
    } else {
      const entries = Array.from(activeByTask.values());
      activeByTask.clear();
      for (const entry of entries) {
        await safelyDisposeInstance(entry.instance);
      }
      for (const cache of kvCaches.values()) {
        try {
          cache?.dispose?.();
        } catch {}
      }
      kvCaches.clear();
    }
  });
}

self.addEventListener('message', (e) => {
  const cmd = e.data;
  if (!cmd || !cmd.type) return;

  switch (cmd.type) {
    case 'check':
      checkDevice();
      break;
    case 'load':
      loadPipeline(cmd.config, cmd.id);
      break;
    case 'run':
      runInference(cmd.id, cmd.task, cmd.input, cmd.options);
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
