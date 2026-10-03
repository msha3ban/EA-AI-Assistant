# Local model accuracy results

The 2026-10-02 selection below is retained as history and superseded by the 2026-10-03 LLM decision. The test Recording was approximately five minutes, recorded from a Teams Meeting joined by phone. The hand-corrected reference Transcript contains 13 critical facts (F1–F13) and 22 Vocabulary terms. Test hardware was a GTX 1060 6 GB (Pascal; `int8` runs as `int8_float32`), i7-7700HQ, and 16 GB RAM. Software: faster-whisper 1.2.1, ctranslate2 4.8.2, Ollama 0.34.2. See [normalisation rules](accuracy-normalisation.md) for raw and normalised scoring.

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

## 2026-10-02 selection (superseded)

- Speech-to-text: faster-whisper `large-v3`, compute type `int8` (effective `int8_float32` on the GTX 1060), language `ar`, Vocabulary prompt mode `hotwords`, and VAD on with the existing default parameters.
- LLM at the time: `qwen3:8b`, thinking off for English Transcript, extract and Summary. This is no longer the default or a recommended option.
- MoM generation: two-step (English Transcript, then extraction). It is the only pipeline measured end to end here; direct generation from real speech-to-text output was not measured.

That historical choice was led by MoM critical-fact accuracy and Flagged Passage recall, not WER alone. In the selected run on real speech-to-text output, qwen3:8b found all 13/13 facts (five in the expected section), versus 7/13 for qwen2.5:7b (three in the expected section). The qwen3:8b run had zero wrong and zero missing facts; qwen2.5:7b had six missing. Flagged Passage detectors flagged 0 of the 4 real-error segments (0% recall) in that run, so Flagged Passage recall remained an open problem. The speech-to-text tooling investigation is tracked in [wayfinder ticket #35](https://github.com/msha3ban/EA-AI-Assistant/issues/35) and [its research note](https://github.com/msha3ban/EA-AI-Assistant/blob/research/stt-tooling/docs/research/stt-tooling-and-flags.md).

## 2026-10-02 limitations and follow-up (historical)

- This selection is based on one approximately five-minute Recording.
- The historical qwen3:8b end-to-end run had 18 invented candidates; that count was not a human judgement of the MoM.
- Only 5/13 facts landed in their expected MoM section for that run; most were found in Summary or Purpose rather than Decisions or Technical Details.
- At the time, the 8192 Summary context default was a stopgap for observed token-budget failures, with chunking tracked in [#41](https://github.com/msha3ban/EA-AI-Assistant/issues/41).
- Scorer fixes landed between report A and report B: report A used section-strict fact scoring and lacked alias-aware Flagged Passage scoring. Its MoM-fact and Flagged Passage numbers are not comparable with B/C; its speech-to-text metrics, speed, and VRAM remain useful.
- Model candidates are tracked in the [wayfinder map, ticket #31](https://github.com/msha3ban/EA-AI-Assistant/issues/31).

## 2026-10-03: LLM candidates

The same five-minute Recording was evaluated on current main with faster-whisper `large-v3`, `int8` (effective `int8_float32`), Vocabulary hotwords, language `ar`, VAD, two-step MoM generation and chunked Summary. Ollama was 0.35.1. The speech-to-text candidate runs used `qwen3:8b` as the LLM.

### Speech-to-text candidates

| Run | WER normalised | CER normalised | Vocabulary recognised | MoM facts found | min/audio hour |
|---|---:|---:|---:|---:|---:|
| large-v3 + hotwords (baseline) | 39% | 19% | 89% | 13/13 | ~157 |
| whisper-medium-arabic-codeswitched-ct2 + hotwords | 41.5% | 24.0% | 63% | 10/13 | 98 |
| whisper-medium-arabic-codeswitched-ct2, Vocabulary off | 39.7% | 19.9% | 61% | 10/13 | 75 |

`large-v3` remains the speech-to-text default.

### LLM direct on the corrected reference Transcript

| LLM | Facts found | In expected section | Missing | Invented candidates | min/audio hour | Peak VRAM MiB |
|---|---:|---:|---:|---:|---:|---:|
| qwen3:8b, thinking off | 13 | 3 | 0 | 8 | 46 | 5048 |
| qwen2.5:7b | 10 | 4 | 3 | 2 | 27 | 4928 |
| qwen3.5:9b, thinking off | 13 | 5 | 0 | 4 | 51 | 5388 |
| gemma4:12b | 13 | 2 | 0 | 1 | 54 | 5702 |
| gemma4:e4b | 11 | 3 | 2 | 0 | 12 | 4844 |

### End-to-end runs with real speech-to-text output

| LLM | Run | Facts found | In expected section | Missing | Invented candidates | min/audio hour |
|---|---:|---:|---:|---:|---:|---:|
| qwen3:8b, thinking off | 1 | — | — | — | — | 263 before failure |
| qwen3.5:9b, thinking off | 1 | 11 | 2 | 2 | 3 | 98 |
| gemma4:12b | 1 | 12 | 4 | 1 | 0 | 121 |
| gemma4:12b | 2 | 11 | 2 | 2 | 1 | 110 |
| gemma4:12b | 3 | 12 | 3 | 1 | 1 | 155 |
| gemma4:e4b | 1 | 12 | 2 | 1 | 0 | 39 |
| gemma4:e4b | 2 | 11 | 2 | 2 | 0 | 45 |
| gemma4:e4b | 3 | 11 | 0 | 2 | 1 | 46 |

The qwen3:8b run failed at extract with “Ollama returned an incomplete or empty response” after retries were exhausted. Neither gemma4 model failed in its three runs. gemma4:12b reached approximately 5.9 GB peak VRAM and 4.8 GB peak RAM. Flagged Passage recall was 0% in every run.

### Decision and limitations

`gemma4:12b` is the default for English Transcript, extract and Summary, with thinking off. It had the best median facts found (12/13) and facts in the expected section (3/13) in the end-to-end runs, no failures, and remained within the approximately 4x speed budget. `gemma4:e4b` is the faster TOML option (`[llm] model = "gemma4:e4b"`), at about 3x the end-to-end speed with slightly lower MoM accuracy. `qwen3:8b` failed end to end intermittently and is no longer recommended. The [wayfinder map](https://github.com/msha3ban/EA-AI-Assistant/issues/31), [decision](https://github.com/msha3ban/EA-AI-Assistant/issues/42) and [candidate runs](https://github.com/msha3ban/EA-AI-Assistant/issues/38) retain the related work.

These results use one five-minute Recording. Differences of one fact are within run-to-run noise: speech-to-text output varied slightly, with normalised WER from 38.2% to 41.3%. Invented-candidate counts still need a human read of the MoM.

## 2026-10-03: Cohere Transcribe and word-level Flagged Passages

Cohere Transcribe Arabic 07-2026 ran as GGUF Q8_0 through transcribe.cpp 0.3.0 on CPU; this machine had no CUDA toolkit. The end-to-end runs used two-step MoM generation with gemma4:12b. Facts found are out of 13; the parenthesised number is the count in the expected section.

| Run | WER norm | CER norm | Vocabulary recognised | Facts found (in expected section) | min/audio hour |
|---|---:|---:|---:|---:|---:|
| large-v3 + hotwords (reference, this date) | 38.7% | 19.0% | 89% | 12/13 (4) | 126 |
| Cohere, Vocabulary given to the LLM stages | 44.3% | 27.2% | 52% | 11/13 (3) | 89 |
| Cohere, no Vocabulary | 44.3% | 27.2% | 52% | 10/13 (3) | 140 |

Cohere takes no Vocabulary and returns no token probabilities, so confidence-based Flagged Passages are unavailable. It transcribed 30 seconds of audio in approximately 12 seconds on CPU. `large-v3` stays the default; the transcribe.cpp engine remains available as an option. See [#45](https://github.com/msha3ban/EA-AI-Assistant/issues/45).

The word-level Flagged Passage measurement used `large-v3` with hotwords and word probabilities from forced alignment after decoding. Of 19 segments, 4 had real errors.

| Word probability | Minimum low-confidence words | Segments flagged | Recall | Precision |
|---:|---:|---:|---:|---:|
| 0.5 | 1 | 19/19 | 100% | 21% |
| 0.2 | 1 | 15/19 | 100% | 27% |
| 0.1 | 1 | 13/19 | 75% | 23% |
| 0.1 | 2 | 5/19 | 25% | 20% |

Precision never rose meaningfully above the 21% base rate, so the detector ships off by default. This is based on one Recording with only 4 error segments; segments averaged approximately 15 seconds after VAD. On this Recording, faster-whisper's `word_timestamps=True` changed decoding and raised normalised WER from 39% to 61%, which is why word probabilities come from alignment after decoding. The report's Flag recall column now measures the application's flags; see [the scoring definition](accuracy-normalisation.md). See [#46](https://github.com/msha3ban/EA-AI-Assistant/issues/46).
