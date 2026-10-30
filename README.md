# affirmations

Science-backed affirmations with unified `LFM3-Omni-2B` WebGPU text + 24kHz speech streaming and `@ondevice/localfirst` mobile fallback (`SmolLM2-135M` + `Kokoro-82M`).

![affirmations demo](./affirmations-social.gif)

```bash
npx serve .
```

## Quick start

```js
import { Lfm3OmniForConditionalGeneration } from './Lfm3OmniForConditionalGeneration.js';

const avail = await Lfm3OmniForConditionalGeneration.checkAvailability('LiquidAI/LFM3-Omni-2B-GGUF');
const model = await Lfm3OmniForConditionalGeneration.from_pretrained('LiquidAI/LFM3-Omni-2B-GGUF');
const session = model.create_session();

const stream = session.generate(
  [
    { role: 'system', content: 'Write one grounded first-person affirmation.\nRespond with interleaved text and audio.' },
    { role: 'user', content: 'Situation: Shipping a complex release' }
  ],
  { generation_mode: 'interleaved', max_new_tokens: 240 }
);

for await (const ev of stream) {
  if (ev.type === 'text') renderCard(ev.text);
  if (ev.type === 'audio') schedule24kHzPCM(ev.audio, model.audio_sampling_rate);
}
```

Runs Liquid AI `LFM3-Omni-2B` (`Q4_0`, 1.26GB) natively in WebGPU to stream affirmation tokens and 24kHz `Float32Array` speech frames from a single unified session, reusing cached system-prompt KV state across cards.

## Hardware-aware fallback

```js
import localfirst, { detect } from '@ondevice/localfirst';
import fit from 'browser-llm-fit';
import { ModelClient } from 'webml-kit';

const telemetry = await detect();
const check = await fit('smollm2-135m-instruct', {
  ram: telemetry.memory.totalGb,
  gpu: telemetry.gpu.webgpu ? 'webgpu-f16' : 'wasm-cpu'
});

const llm = new ModelClient('./model-worker.js');
const tts = new ModelClient('./model-worker.js');
await llm.load({ task: 'text-generation', modelId: 'HuggingFaceTB/SmolLM2-135M-Instruct', device: 'wasm', dtype: 'q4' });
await tts.load({ task: 'text-to-speech', modelId: 'onnx-community/Kokoro-82M-v1.0-ONNX', device: 'wasm', dtype: 'q8' });
```

`detect()` probes WebGPU limits, RAM, and Wasm SIMD in under 2ms. Desktop 8GB+ WebGPU selects `LFM3-Omni-2B` (`interleaved` + `audio` modes), stepping down automatically to `Qwen2.5-0.5B` (350MB), `SmolLM2-360M` (240MB), or `SmolLM2-135M` (92MB) + `Kokoro-82M` on mobile and WASM devices. Click the header badge at any time to cycle tiers.

## Science lenses

- `Attentional Control · dlPFC Focus` — selective attention and working-memory gating
- `Arousal Reappraisal · Crum et al.` — reframing physiological arousal as cognitive readiness
- `Self-Affirmation · Steele & Cascio vmPFC` — anchoring worth in core values over single outcomes
- `Cognitive Defusion · ACT` — separating transient urgency from self-identity
- `Autonomic Regulation · Vagal Brake` — slow exhalation and parasympathetic pacing
- `Implementation Intention · Gollwitzer` — shrinking friction to the next verifiable action
- `Neuroplastic Error Learning · Dweck` — treating bugs and obstacles as diagnostic signal
- `Working-Memory Protection · Cognitive Load` — protecting unfragmented focus
- `Self-Distancing · Kross & Neff` — objective, compassionate internal dialogue
- `Zeigarnik Closure · Bounded Progress` — honoring bounded daily effort

## Demo

- Live: [https://h3manth.com/ai/affirmations/](https://h3manth.com/ai/affirmations/)

## Related

- [LiquidAI/LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B)
- [@ondevice/localfirst](https://github.com/hemanth/localfirst)
- [browser-llm-fit](https://github.com/hemanth/browser-llm-fit)
- [webml-kit](https://github.com/hemanth/webml-kit)

## License

MIT © [Hemanth.HM](https://h3manth.com)
