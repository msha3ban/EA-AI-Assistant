# Local model accuracy results

Evaluation date: 2026-10-02. The test Recording was approximately five minutes, recorded from a Teams Meeting joined by phone. The hand-corrected reference Transcript contains 13 critical facts (F1–F13) and 22 Vocabulary terms. Test hardware was a GTX 1060 6 GB (Pascal; `int8` runs as `int8_float32`), i7-7700HQ, and 16 GB RAM. Software: faster-whisper 1.2.1, ctranslate2 4.8.2, Ollama 0.34.2. See [normalisation rules](accuracy-normalisation.md) for raw and normalised scoring.

## Speech-to-text runs

WER and CER are raw / normalised. Vocabulary recognition is the recognised share of reference term occurrences. Resource figures are minutes per audio hour and peak VRAM. Rows marked `STT only` produced speech-to-text metrics but the later LLM stage failed; runtime and VRAM were not available in the report. Report A's fact and Flagged Passage scores used the old scorer and are intentionally omitted here.

| Run | WER raw / normalised | CER raw / normalised | Vocabulary recognised | min/audio hour | Peak VRAM MiB |
|---|---:|---:|---:|---:|---:|
| large-v3, int8_float32, ar, VAD, Vocabulary off (A) | 71.7% / 60.1% | 45.4% / 41.1% | 4.3% | 96.5 | 5044 |
| large-v3, int8_float32, ar, VAD, hotwords (A) | 54.0% / 38.7% | 23.5% / 19.0% | 89.1% | 116.7 | 5710 |
| large-v3, int8_float32, auto→ar (0.99), VAD, Vocabulary off (A) | 71.7% / 60.1% | 45.4% / 41.1% | 4.3% | 98.8 | 5044 |
| large-v3-turbo, int8_float32, ar, VAD, Vocabulary off (A) | 69.5% / 55.6% | 39.1% / 34.4% | 32.6% | 106.6 | 5044 |
| large-v3, int8_float32, ar, VAD, initial prompt (B; STT only) | 59.3% / 49.1% | 24.6% / 20.3% | 37.0% | n/a | n/a |
| large-v3-turbo, int8_float32, ar, VAD, initial prompt (B; STT only) | 64.0% / 47.9% | 29.3% / 24.1% | 73.9% | n/a | n/a |
| large-v3-turbo, int8_float32, ar, VAD, hotwords (B; STT only) | 72.5% / 57.9% | 40.1% / 34.9% | 56.5% | n/a | n/a |
| large-v3, int8_float32, ar, VAD, hotwords (C; selected end-to-end run) | 54.3% / 38.7% | 23.2% / 18.7% | 89.1% | 156.9 | 5774 |

## LLM and MoM runs

The direct runs below build the MoM from the corrected reference Transcript and measure the LLM in isolation. The two end-to-end runs use real speech-to-text output and two-step MoM generation (English Transcript, then extract). “Found” includes facts found in any section; “expected” gives those found in their expected section. Invented entries are candidates for human review, not automatic judgements. Report A's direct-run fact counts used the old section-strict scorer and are excluded.

| Run | Input | Facts found / expected section / wrong / missing | Invented candidates | min/audio hour |
|---|---|---:|---:|---:|
| qwen3:8b, thinking off (B) | Corrected reference Transcript; direct | 13 / 6 / 0 / 0 | 5 | 69.2 |
| qwen3:8b, thinking on (B) | Corrected reference Transcript; direct | 10 / 3 / 0 / 3 | 2 | 46.9 |
| qwen2.5:7b (B) | Corrected reference Transcript; direct | 10 / 4 / 0 / 3 | 2 | 27.2 |
| qwen3:8b, thinking off (C) | Real speech-to-text; two-step | 13 / 5 / 0 / 0 | 18 | 156.9 |
| qwen2.5:7b (C) | Real speech-to-text; two-step | 7 / 3 / 0 / 6 | 8 | 48.0 |

## Selected application defaults

- Speech-to-text: faster-whisper `large-v3`, compute type `int8` (effective `int8_float32` on the GTX 1060), language `ar`, Vocabulary prompt mode `hotwords`, and VAD on with the existing default parameters.
- LLM: `qwen3:8b`, thinking off for translation, extraction, and Summary.
- MoM generation: two-step (English Transcript, then extraction). It is the only pipeline measured end to end here; direct generation from real speech-to-text output was not measured.

The choice is led by MoM critical-fact accuracy and Flagged Passage recall, not WER alone. On real speech-to-text output, qwen3:8b found all 13/13 facts (five in the expected section), versus 7/13 for qwen2.5:7b (three in the expected section). The qwen3:8b run had zero wrong and zero missing facts; qwen2.5:7b had six missing. Current Flagged Passage detectors flagged 0 of the 4 real-error segments (0% recall) in the selected run, so Flagged Passage recall remains an open problem. The speech-to-text tooling investigation is tracked in [wayfinder ticket #35](https://github.com/msha3ban/EA-AI-Assistant/issues/35) and [its research note](https://github.com/msha3ban/EA-AI-Assistant/blob/research/stt-tooling/docs/research/stt-tooling-and-flags.md).

## Limitations and follow-up

- This selection is based on one approximately five-minute Recording.
- The 18 invented candidates from the qwen3:8b end-to-end run still need a human read.
- Only 5/13 facts landed in their expected MoM section for that run; most were found in Summary or Purpose rather than Decisions or Technical Details.
- The 8192 Summary context default is a stopgap for the observed token-budget failures. Issue [#41](https://github.com/msha3ban/EA-AI-Assistant/issues/41) remains open for chunking.
- Scorer fixes landed between report A and report B: report A used section-strict fact scoring and lacked alias-aware Flagged Passage scoring. Its MoM-fact and Flagged Passage numbers are not comparable with B/C; its speech-to-text metrics, speed, and VRAM remain useful.
- Future model candidates are listed in the [wayfinder map, ticket #31](https://github.com/msha3ban/EA-AI-Assistant/issues/31).
