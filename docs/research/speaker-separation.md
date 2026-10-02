# Speaker separation (diarization) on a GTX 1060 6 GB / i7-7700HQ

Research for #36 (map #31, feature #28). Researched 2026-10-02 from model cards, official repos, release notes and package metadata. Nothing was downloaded or run; all speed and fit figures for *this* hardware are inferences until measured.

**Question.** Which local diarization tools work on mixed Egyptian Arabic/English Teams Recordings (often captured on a phone), fit a GTX 1060 6 GB (Pascal, cc 6.1) when run one after another with faster-whisper (GPU freed between stages), or run on CPU (i7-7700HQ, 16 GB), within ~4x real time per audio hour in total?

## Verdict

- **Use `pyannote/speaker-diarization-community-1` (pyannote.audio 4.0.x).** Run it after faster-whisper has released the GPU, as a separate process with telemetry turned off, loaded from a local clone. Assign speakers from its *exclusive* diarization output to faster-whisper words.
- **Challenger to measure:** `nvidia/Nemotron-3-Diarization`, released 2026-09-23. It has an open licence, needs no token, handles up to 8 speakers, and scores better on English benchmarks. But Pascal is not on its supported-hardware list, it was trained without Arabic, and it is nine days old.
- **CPU fallback:** the same pyannote pipeline on CPU. If torch itself becomes the problem, use sherpa-onnx (ONNX, CPU, no torch or token).
- **Rejected:** DiariZen (non-commercial weights), Sortformer v1 (non-commercial, ~12-minute limit), streaming Sortformer v2/v2.1 (4-speaker cap, superseded by Nemotron 3), and NeMo MSDD (older cascade, heavy NeMo dependency, no advantage shown).
- **Main integration risk.** It is not VRAM. It is the **torch/CUDA wheel stack on Pascal**: torch cu12.8+ wheels dropped Pascal, cu12.6 wheels pin a different cuDNN from this project's faster-whisper pins, and PyTorch 2.14 is the last release with Pascal wheels.

## Shortlist

DER figures are each vendor's own numbers. Scoring rules differ (collar, overlap handling), so **compare within a row, not across vendors**.

| Candidate | Licence / gating | Max speakers | Training languages | DER evidence (lower is better) | Fits GTX 1060 / CPU? | Verdict |
|---|---|---|---|---|---|---|
| **pyannote community-1** (pyannote.audio 4.0.7) | Weights CC-BY-4.0, **gated**: accept contact-sharing terms and use an HF token once to clone. Code MIT. **Telemetry on by default.** | Unbounded (clustering, VBx) | Multi-domain (AISHELL, AliMeeting, AMI, DIHARD, MSDWild, VoxConverse...). Arabic not listed. | AMI-SDM 19.9, AliMeeting 20.3, DIHARD3 20.2, MSDWild 22.8, VoxConverse 11.2 (3.1: 22.7 / 24.5 / 21.4 / 25.4 / 11.2). Vendor-run Arabic: 15.57 (12 files) | GPU: yes if torch cu126 wheels are used (small models; inference). CPU: yes, about 1x real time (inference) | **Recommend** |
| pyannote 3.1 (pyannote.audio 3.x) | MIT, gated | Unbounded (AHC) | as above | see 3.1 column above | yes | Legacy; superseded by community-1 |
| **Nemotron-3-Diarization** (NVIDIA, 100M) | OpenMDW-1.1, **not gated** | 8 | ~10k h real speech (En, Zh, Hi, Kn, Te, Bn) plus simulated speech in 21 languages. **No Arabic listed.** "Performance degrades with non-English." | DIHARD3 12.73, CALLHOME-p2 9.10, AMI-MHM 9.25, AMI-SDM 11.14-12.95, AliMeeting far 10.47-11.60 (offline, 30.4 s). Vendor-run Arabic: 16.54 (MLX port) | Pascal **not in the supported list** (Ampere+). Plain PyTorch, 100M params, so fp32 on Pascal or CPU is plausible but unverified | **Measure as challenger** |
| Streaming Sortformer 4spk-v2 / v2.1 (NeMo) | v2 CC-BY-4.0; v2.1 NVIDIA Open Model Licence; not gated | 4 ("degrades with 5+") | Mainly English | v2.1: DIHARD3 full 20.21, CALLHOME 2/3/4-spk 6.65/11.25/13.35, AMI-IHM 16.67, DIHARD3 5+spk 41.42 | Heavy NeMo dependency; Pascal untested | Reject: 4-speaker cap, superseded |
| Sortformer 4spk-v1 | **CC-BY-NC-4.0** | 4 | English | DIHARD3 14.76 | "~12 min max on a 48 GB A6000" | Reject |
| NeMo clustering diarizer / MSDD (MarbleNet + TitaNet + MSDD) | NeMo Apache-2.0. MSDD checkpoint card requires login (not verified) | Unbounded | Telephonic/English | No current comparison found | Heavy NeMo dependency | Reject (no advantage shown) |
| DiariZen (BUT, WavLM + pyannote) | Code MIT, **weights CC-BY-NC-4.0** | Unbounded | Multi-domain | Large-s80-v2: AMI-SDM 13.9, AliMeeting far 10.8, DIHARD3 14.5, MSDWild 15.8, VoxConverse 9.1 (pyannote 3.1 in the same table: 22.4/24.4/21.7/25.3/11.3) | WavLM-Large plausibly fits 6 GB (inference) | Reject: NC licence, since EA work is professional use. Best open-weight accuracy, though |
| sherpa-onnx offline diarization (pyannote segmentation-3.0 ONNX + 3D-Speaker/NeMo embeddings) | Apache-2.0 code. Models from GitHub releases, no HF token | Unbounded (clustering, threshold or `num_clusters`) | n/a | None published | CPU, no torch | CPU fallback |
| `diarize` (FoxNoseTech; Silero VAD + WeSpeaker ResNet34 ONNX) | Apache-2.0, no HF account | Unbounded | n/a | Self-reported, VoxConverse only, and inconsistent (README ~4.8% vs HN post ~10.8%) | CPU RTF 0.12 vs community-1 0.86 (MacBook, self-reported) | Not trusted; possible speed fallback |

## Facts (with sources)

### pyannote
- pyannote.audio **4.0.0** (2025-09-29) introduced `speaker-diarization-community-1`. It switches clustering from AHC to **VBx** and returns an **exclusive** diarization (one speaker at a time) "that simplifies the reconciliation between fine-grained speaker diarization timestamps and (sometimes not so precise) transcription timestamps". Source: [release 4.0.0](https://github.com/pyannote/pyannote-audio/releases/tag/4.0.0).
- **Offline use:** accept the agreement, `git clone https://hf.co/pyannote/speaker-diarization-community-1`, then `Pipeline.from_pretrained('/local/dir')` "works without internet connection". The token is needed only for the clone. Source: same release notes.
- **Licence and gating:** the community-1 card says CC-BY-4.0 and gated ("you have to accept the conditions"). Segmentation-3.0 and 3.1 are MIT and gated. Sources: [community-1 card](https://huggingface.co/pyannote/speaker-diarization-community-1), [3.1 card](https://huggingface.co/pyannote/speaker-diarization-3.1), [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
- **DER table** (community-1 vs 3.1; precision-2 is the paid cloud model and out of scope under ADR-0001): AISHELL-4 11.7/12.2, AliMeeting ch1 20.3/24.5, AMI-IHM 17.0/18.8, AMI-SDM 19.9/22.7, AVA-AVD 44.6/49.7, CALLHOME p2 26.7/28.5, DIHARD3 20.2/21.4, Ego4D 46.8/51.2, MSDWild 22.8/25.4, RAMC 20.8/22.2, REPERE 8.9/7.9, VoxConverse 11.2/11.2. Source: community-1 card.
- **Speed:** "31 s per hour of audio" (AMI) and "37 s per hour" (DIHARD 3), self-hosted on an **H100 80 GB**. Source: [pyannote-audio README](https://github.com/pyannote/pyannote-audio).
- **Package requirements** (PyPI 4.0.7, 2026-06-30): `torch>=2.8.0`, `torchaudio>=2.8.0`, `torchcodec>=0.7.0`, `pyannoteai-sdk`, `opentelemetry-*`. Source: [PyPI JSON](https://pypi.org/pypi/pyannote.audio/json).
- **Telemetry is enabled by default.** `telemetry/config.yaml` at tag 4.0.7 has `metrics_enabled: true` and `otlp_endpoint: https://otel.pyannote.ai/v1/traces`. It sends the model origin, file duration and `num/min/max_speakers`, not audio. Turn it off with `PYANNOTE_METRICS_ENABLED=0`. Sources: [config.yaml@4.0.7](https://github.com/pyannote/pyannote-audio/blob/4.0.7/src/pyannote/audio/telemetry/config.yaml), [README "Telemetry"](https://github.com/pyannote/pyannote-audio#telemetry).
- Input is mono 16 kHz; it downmixes and resamples automatically. It accepts an in-memory `{"waveform", "sample_rate"}`, and supports `num_speakers`, `min_speakers` and `max_speakers`. Source: community-1 card.

### PyTorch on Pascal (affects every torch-based candidate)
- PyTorch 2.8+ **CUDA 12.8/12.9 wheels dropped sm_50–sm_70 (Maxwell and Pascal)**. Use the **CUDA 12.6** wheels for Pascal. Source: [PyTorch dev-discuss](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128).
- **CUDA 12.6 wheels stop at PyTorch 2.15**, so **2.14 is the last prebuilt release for Pascal**. Source: [notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432), [RFC #190385](https://github.com/pytorch/pytorch/issues/190385).
- `torch 2.14.1+cu126` and `2.8.0+cu126` both pin `nvidia-cudnn-cu12==9.10.2.21`, and 2.8.0+cu126 pins `nvidia-cublas-cu12==12.6.4.1`. This project pins `nvidia-cudnn-cu12==9.27.0.42` and `nvidia-cublas-cu12==12.9.2.10` in the `stt` extra (`pyproject.toml`). Source: wheel metadata at `download.pytorch.org/whl/cu126/torch-2.14.1%2Bcu126-...whl.metadata`, checked 2026-10-02.

### NVIDIA
- **Nemotron-3-Diarization:** 100M parameters, up to 8 speakers, OpenMDW-1.1, streaming plus offline, NeMo v3.0 or Transformers (`AutoModelForAudioFrameClassification`). Supported hardware: Ampere, Ada, Hopper, Blackwell. Tested on RTX PRO 5000. "Performance degrades with non-English languages." Real speech used in training: English, Mandarin, Hindi, Kannada, Telugu, Bengali. Source: [model card](https://huggingface.co/nvidia/Nemotron-3-Diarization), [NVIDIA blog](https://huggingface.co/blog/nvidia/nemotron-diarization). The blog's ASR example assigns "each word with the speaker active at its midpoint".
- **Streaming Sortformer v2.1:** 117M, 4 speakers, mainly English, NVIDIA Open Model Licence. Outputs `begin, end, speaker_index` segments or a T x 4 frame-probability matrix (80 ms frames). RTF 0.093 at 1.04 s latency on an RTX 6000 Ada. Source: [card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1). v2 is CC-BY-4.0 ([card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2)).
- **Sortformer v1:** CC-BY-NC-4.0; the maximum length depends on VRAM, "around 12 minutes" on a 48 GB A6000. Source: [card](https://huggingface.co/nvidia/diar_sortformer_4spk-v1).
- NeMo 3.0.0 (2026-08-07) requires `torch>=2.6.0`. The NeMo docs still list the cascaded MarbleNet + TitaNet + MSDD diarizer alongside Sortformer. Sources: [PyPI](https://pypi.org/pypi/nemo-toolkit/json), [NeMo diarization models](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/models.html).

### Others
- **DiariZen:** code MIT, all weights CC-BY-NC-4.0; built on pyannote 3.1. DER figures are in the table above. Source: [repo](https://github.com/BUTSpeechFIT/DiariZen).
- **sherpa-onnx:** `OfflineSpeakerDiarization` uses pyannote segmentation-3.0 (or reverb-diarization-v1) ONNX plus 3D-Speaker or NeMo embedding ONNX. Models come from GitHub releases. Clustering is set by `num_clusters` or `threshold`. Output is `start -- end speaker_NN`. Sources: [docs](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html), [example](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-speaker-diarization.py).
- **WhisperX** (BSD-2, 3.8.6) defaults to community-1 and pins `torch~=2.8.0`. Its `assign_word_speakers` picks, for each word or segment, the speaker with the **largest time intersection**, with optional `fill_nearest` (nearest segment midpoint) for words with no overlap. Source: [whisperx/diarize.py](https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py).
- **faster-whisper** gives per-word `start` and `end` with `transcribe(..., word_timestamps=True)`. Source: [README](https://github.com/SYSTRAN/faster-whisper#word-level-timestamps). The project's adapter (`src/ea_assistant/real_adapters.py`) does **not** set `word_timestamps` today, so it has segment-level times only.

### Arabic and code-switching evidence
- **No primary, independent diarization benchmark for Egyptian Arabic or Arabic-English code-switching was found.** The only Arabic numbers come from a **vendor** benchmark (Soniqo, which sells speech products): 12 Arabic files, dialect and source unspecified, 0.25 s collar. Results: pyannote community-1 **15.57%**, Nemotron 3 (MLX port) 16.54%, Nemotron 3 (CoreML) 18.16%, Sortformer streaming 22.32%. Source: [soniqo.audio/benchmarks/diarization](https://soniqo.audio/benchmarks/diarization).
- An academic multi-system benchmark (196.6 h; English, Mandarin, German, Japanese, Spanish) found missed speech the biggest error source, then speaker confusion. Arabic is not included. Source: [arXiv 2509.26177](https://arxiv.org/abs/2509.26177).

## Inferences (not verified on this hardware)

1. **Language matters less than for STT.** Diarization works on acoustic speaker identity, not words, so Arabic/English code-switching is not expected to hurt much. The vendor Arabic figures (15–17%) are in the same range as the English meeting benchmarks, which supports this. What will matter more for these Recordings is the **acoustic condition**: a phone recording a laptop speaker is far-field, reverberant, compressed, and Teams has already mixed it to mono. The closest benchmarks are AMI-SDM, AliMeeting-far, DIHARD3 and MSDWild, where community-1 scores **~20–23% DER**. Expect that range or worse.
2. **VRAM is not the constraint.** pyannote's segmentation and embedding models are small (a few million parameters each; the card does not give sizes). Nemotron is 100M parameters, about 0.4 GB in fp32. Either should fit well within 6 GB once faster-whisper has been unloaded, in a separate process so the CUDA context is torn down cleanly.
3. **Speed on the GTX 1060:** H100 = 31–37 s per audio hour. Allowing roughly 10–30x lower throughput for a fp32-only Pascal card suggests **~5–20 min per audio hour** on GPU. On CPU, the only figure is a third party's RTF 0.86 on an unspecified MacBook. A 4-core 2017 i7-7700HQ will likely be ~0.8–1.5x real time, so **~50–90 min per audio hour**. GPU fits the 240 min/h budget easily. CPU fits only if STT plus LLM stages leave about an hour spare.
4. **Dependency isolation is required.** The torch cu126 cuDNN pin (9.10.2.21) conflicts with the project's faster-whisper pin (9.27.0.42) in one `uv` environment. Practical options:
   - (a) Run diarization in a **separate venv or subprocess** with its own torch cu126. This also matches the one-model-at-a-time rule and guarantees VRAM is released.
   - (b) Use **CPU-only torch** for diarization (no NVIDIA wheels).
   - (c) Re-pin the `stt` extra to torch's cuDNN and re-run `ea preflight`.

   Option (a) is the safest. Pin torch at or below 2.14 for as long as the GTX 1060 is in use.
5. **The speaker cap matters.** Teams meetings often have more than 4 voices, so Sortformer's 4-speaker cap rules it out. Nemotron (8) is usually enough. pyannote has no cap.
6. **Nemotron on Pascal:** it is plain PyTorch through Transformers, so fp32 on cc 6.1 should run, but NVIDIA lists only Ampere and newer. Its English DER lead is large (DIHARD3 12.7 vs 20.2). The vendor Arabic run shows no lead (16.5 vs 15.6), from a tiny sample. Only the accuracy suite can decide between them.

## Integration notes (for the later ready-for-agent issue)

- **Order:** faster-whisper (GPU), unload, diarize (GPU subprocess or CPU), merge on CPU. Feed the diarizer the same normalised 16 kHz mono WAV the STT stage uses, so the timelines match.
- **Pinning:** `pyannote.audio==4.0.7`, torch/torchaudio from `https://download.pytorch.org/whl/cu126` (at most 2.14.x), and torchcodec matching that torch, or pass the waveform in memory to avoid FFmpeg decoding.
- **Privacy (ADR-0001):** set `PYANNOTE_METRICS_ENABLED=0` (and persist it), set `HF_HUB_OFFLINE=1`, and load from a local clone path. `ea setup-models` can do the one-time gated clone. `ea preflight` should fail if telemetry is enabled. Telemetry sends no audio, but it is a network call the local-only rule should not allow.
- **Alignment:**
  1. Turn on `word_timestamps=True` in faster-whisper. This costs extra time; measure it.
  2. Take pyannote's `exclusive_speaker_diarization` (not the overlapping one).
  3. Give each word the speaker with the largest overlap. Fall back to the nearest turn by midpoint, as WhisperX does.
  4. **Split a faster-whisper segment wherever the word-level speaker changes**, because faster-whisper segments are not speaker-aware. Re-number the `Segment` ids afterwards; the `transcript_segments.speaker` column already exists.
  5. Without word timestamps, the fallback is segment-level majority overlap, which is coarser.
- **Speaker count:** expose an optional `min_speakers`/`max_speakers` in config. Meeting attendee counts are often known, and bounding them is the cheapest accuracy gain.
- **Consistency across Recordings (#28):** either diarize the concatenated Meeting audio in one pass, or keep the per-speaker embeddings (pyannote returns `speaker_embeddings`) and match speakers across Recordings by cosine similarity.
- **Failure path:** on any diarization error, keep the single `Speaker 1` (already the default in `models.py`).

## Open questions

1. What DER does community-1 actually get on the user's phone-captured Teams Recordings? The accuracy suite has no DER today. A reference needs speaker-labelled turns, scored with `pyannote.metrics` (collar 0.25 s, overlap scored). Also: does speaker attribution change MoM critical-fact accuracy (Action Item owners)?
2. Does Nemotron-3-Diarization run on cc 6.1 in fp32, and with which `transformers` version? The card says to install from source. How does its DER compare on the same Recording?
3. Real GTX 1060 and i7-7700HQ timings for both, and the extra time `word_timestamps=True` adds to faster-whisper large-v3.
4. Can ctranslate2 4.8.2 run against cuDNN 9.10.2.21, so one environment would be possible? Or should the project use the subprocess approach?
5. Is the CC-BY-NC licence of DiariZen acceptable for this user's use? It is professional EA work, so this is assumed **no**.

## Could not verify
- Parameter counts and VRAM use of community-1's internal models (gated config).
- The real-time-factor statement often quoted for 3.1 (~2.5% on V100). It does not appear on the current card text.
- The MSDD checkpoint card (HF returned 401).
- The `diarize` project's benchmark hardware, and its two conflicting DER figures.
- Whether Nemotron 3 is in a released `transformers` version, or still needs a source install.
