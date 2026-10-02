# Speech-to-text candidates that could beat large-v3 on Egyptian Arabic/English

Research for #32 (part of the wayfinder map #31). Researched 2026-10-02. Nothing was downloaded or run; every number below is published by someone else, so treat it as a reason to measure, not as a result. The decision is made by `ea accuracy` on the user's own Recording (see [accuracy.md](../accuracy.md)).

**Question.** Which local speech-to-text models could beat faster-whisper `large-v3` (`int8_float32`, hotwords, `language="ar"`) on Egyptian Arabic with English technical terms, on a GTX 1060 6 GB (Pascal, compute 6.1; CTranslate2 types `int8` / `int8_float32` / `float32` only), i7-7700HQ, 16 GB RAM, within about 4x real time?

**Baseline on the user's Recording (#1):** CER norm 19%, WER norm 39%, Vocabulary 89%.

## Short answer

1. **Cohere Transcribe Arabic (`CohereLabs/cohere-transcribe-arabic-07-2026`)** is the strongest candidate. It is Apache-2.0, has 2B parameters, ranks #2 on the Open Universal Arabic ASR Leaderboard, and has the best *independent* Egyptian Arabic–English result: 13.4% WER on Perle, against 20.8% for large-v3 with a style hint. It does not run in faster-whisper. On Pascal the way to run it is the GGUF build through transcribe.cpp (ggml: CUDA or CPU).
2. **`Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched-ct2`** is the cheapest test, because it is already a CTranslate2 model. It is a drop-in `[stt] model` change with no code changes. It scored 18.5% WER on Perle and kept 90% of English terms in Latin script, the best of any model measured.
3. **Audar-ASR-V1-Turbo (GGUF Q8_0 through llama.cpp)** ranks #1 on the leaderboard (avg WER 23.17). The only independent Egyptian test ran it on whole clips at Q4 and got poor results, so it needs a fair re-test. Its licence needs checking.
4. **Meta omniASR_LLM_1B_v2 / Unlimited_1B_v2** comes 4th (leaderboard avg 29.96, Apache-2.0, `arz` listed). It probably needs the CPU on Pascal, and it uses a different stack (fairseq2).

Not worth suite time: SeamlessM4T v2, MMS-1b-all, Qwen3-ASR, NVIDIA Canary-1b-v2 / Parakeet-TDT-v3 (no Arabic), NVIDIA Arabic FastConformer, and the public Whisper-large Egyptian LoRA. Reasons are given below.

## Shortlist

"Published WER" is the model's own claim unless marked *indep.* (independent). The Open Universal Arabic ASR Leaderboard (OUAAL) is independent (Elm Research Center). It has **no Egyptian-only or code-switched test set**: its six sets are SADA (Saudi), Common Voice 18, MASC clean/noisy, MGB-2 (broadcast, mostly MSA) and Casablanca (8 dialects including Egyptian). Perle and ArzEn numbers come from one third-party GitHub benchmark ("Sedjem", see [Sources](#sources)): 40 clips each, run on a laptop CPU.

| # | Model | Source | Published Arabic / Egyptian WER | Licence | VRAM / RAM at usable precision on Pascal | CTranslate2 / faster-whisper? | Fits 6 GB? |
|---|---|---|---|---|---|---|---|
| 1 | Cohere Transcribe Arabic 07-2026 (2B, FastConformer enc + Transformer dec) | [model card](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026), [blog](https://huggingface.co/blog/CohereLabs/cohere-transcribe-arabic-07-2026-release) | OUAAL avg **25.87** (#2/37); MGB-2 15.54; Casablanca 49.71; SADA 37.47. Own: Egyptian 19.16, AR-EN code-switched 27.84 (Cohere internal sets). *Indep.* Perle **13.4%** (CER 8.2%, English kept 77%); ArzEn 6.7% (likely in its training data) | Apache-2.0 (gated, auto-approve) | GGUF files: Q8_0 2.41 GB, Q6_K 1.97 GB, Q4_K_M 1.56 GB, F16 4.11 GB. *Inferred:* Q8_0 needs about 3 GB of VRAM including activations. BF16 safetensors are 4.13 GB, fp32 would be about 8 GB. | **No.** Runs in transformers ≥5.4, vLLM (needs compute ≥7.5, so not Pascal) or [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) GGUF (CUDA/Vulkan/CPU) | **Yes** with GGUF Q8_0/Q6_K (inferred). No as fp32 PyTorch. |
| 2 | whisper-medium-arabic-codeswitched (769M, Whisper medium fine-tune) | [model card](https://huggingface.co/Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched), [ct2 build](https://huggingface.co/Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched-ct2) | Own: test 17.9% (1% held-out slice of EJUST + MohamedRashad code-switching data; stock Whisper 52.4% on the same slice). *Indep.* Perle **18.5%** (CER 7.1%, English kept **90%**); ArzEn 10.8% | Apache-2.0 (HF model); ct2 repo tagged MIT | ct2 `model.bin` is 0.77 GB (int8). *Inferred:* about 1.5–2 GB of VRAM at `int8_float32` | **Yes, already converted.** The ct2 repo has no `tokenizer.json`; copy it from the HF model | **Yes** |
| 3 | Audar-ASR-V1-Turbo (2.35B, Whisper-style encoder + Qwen3 decoder) | [model card](https://huggingface.co/audarai/Audar-ASR-V1-Turbo), [repo](https://github.com/AudarAI/Audar-ASR-V1) | OUAAL avg **23.17** (#1/37); SADA 28.92; MGB-2 11.08; Casablanca 47.02. No Egyptian-only number. *Indep.* Perle 43.5% / ArzEn 18.9%, but both on **whole clips** (no VAD chunking) at **Q4**, where large-v3 also got 39.7% | AudarAI **Community** License v1.0 ("research and limited commercial use for qualifying Community Entities") | GGUF: Q8_0 2.17 GB + mmproj 0.64 GB (mmproj must stay BF16). The card says ≥12 GB for full precision. *Inferred:* about 3.5 GB at Q8_0 | **No.** transformers (remote code), vLLM (not Pascal), llama.cpp `llama-mtmd-cli` / `llama-server` (Qwen3-ASR audio merged upstream April 2026) | **Yes** with GGUF Q8_0 (inferred) |
| 4 | omniASR_LLM_1B_v2 / LLM_Unlimited_1B_v2 (2.28B) | [repo](https://github.com/facebookresearch/omnilingual-asr) | OUAAL avg 29.96 for `omniASR_LLM_1B` (v1); MGB-2 15.34; Casablanca 60.68. No Egyptian-only number | Apache-2.0 | Repo: "~6 GiB" inference VRAM in BF16 on A100; 8.5 GiB download (fp32). Pascal has no BF16 tensor path, so *inferred:* fp32 is about 9 GB+, which means CPU or offload | **No** (fairseq2) | **Probably not on GPU; CPU only** (inferred). Non-Unlimited variants are limited to 40 s of audio |
| – | *Baseline:* openai/whisper-large-v3 | [OUAAL](https://huggingface.co/spaces/elmresearchcenter/open_universal_arabic_asr_leaderboard) | OUAAL avg 36.86; SADA 55.96; MGB-2 16.26; Casablanca 71.81. *Indep.* Perle 20.8% with style hint (English kept 88%), 39.7% on whole clips without a hint | MIT | Already in use: `int8_float32` on the 1060 | Yes | Yes |

### Considered and rejected

| Model | Evidence | Why not |
|---|---|---|
| facebook/seamless-m4t-v2-large (2.3B) | OUAAL avg 38.16 (worse than large-v3's 36.86). `arz` speech input supported ([card](https://huggingface.co/facebook/seamless-m4t-v2-large)) | Not better than the baseline on the leaderboard; **CC-BY-NC-4.0** |
| facebook/mms-1b-all | OUAAL avg 54.54 | Far worse; CC-BY-NC-4.0 ([card](https://huggingface.co/facebook/mms-1b-all)) |
| Qwen/Qwen3-ASR-1.7B | OUAAL avg 33.36. *Indep.* Perle 50.6% (whole clip, Q8), and "English almost always" lost | Better than large-v3 on the leaderboard, but worst-in-class on code-switched Egyptian; Audar Turbo (Qwen3 decoder; Audar Flash ships `modeling_qwen3_asr.py`, so it is likely a Qwen3-ASR derivative) beats it on every OUAAL set |
| NVIDIA canary-1b-v2 / parakeet-tdt-0.6b-v3 | 25 European languages, no Arabic ([card](https://huggingface.co/nvidia/canary-1b-v2), [paper](https://arxiv.org/abs/2509.14128)) | No Arabic |
| NVIDIA Parakeet-CTC-1.1B Arabic (universal/concat), conformer-ctc-large-arabic | OUAAL avg 51.96 / 46.54 / 32.91 (with LM) | Worse or no better. The Arabic FastConformer card says it "can have poor performance in dialectal Arabic speech" ([card](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0)) |
| AbdelrahmanHassan/whisper-large-v3-egyptian-arabic | LoRA r=8, 100 steps, WER 47.39% on MGB-3 ([card](https://huggingface.co/AbdelrahmanHassan/whisper-large-v3-egyptian-arabic)) | Weak result. It would need a LoRA merge plus ct2 conversion |
| oddadmix/whisper-large-v3-turbo-arabic-dialectal | 34.4% WER on its own 932-clip held-out set, no per-dialect split ([card](https://huggingface.co/oddadmix/whisper-large-v3-turbo-arabic-dialectal)) | No Egyptian or code-switching evidence. Convertible to ct2 if wanted later |
| IbrahimAmin/code-switched-egyptian-arabic-whisper-small, MAdel121/whisper-*-egy | FLEURS ar_eg 24.36%, MGB-3 44–49% (small); medium-egy 18.03% on own data, but a **SpeechBrain checkpoint**, not HF format | Small capacity or unconvertible format; #2 covers the same niche better |
| Omni CTC models, Voxtral, Gemma-4 audio, VibeVoice-ASR | OUAAL avg ≥32.98 (Gemma-4-E4B) up to 52.99 | No better than #1–#4; larger or not ASR-specialised |

## Recommended order to try in `ea accuracy`

1. **whisper-medium-arabic-codeswitched-ct2.** Zero code: point `[stt] model` at the ct2 directory and keep hotwords, `language="ar"` and the Flagged Passage detectors (which rely on Whisper's `avg_logprob` / `compression_ratio` / `no_speech_prob`). This gives a quick read on whether a code-switching fine-tune beats large-v3 on *this* Recording. Run with and without the Vocabulary `initial_prompt`/hotwords.
2. **Cohere Transcribe Arabic, GGUF Q8_0 via transcribe.cpp (CUDA on the 1060; CPU as fallback).** This is the most likely accuracy win. It needs a new STT adapter (out of scope for #31; becomes a ready-for-agent issue). Before relying on it for Flagged Passages and speaker attribution, note: no hotword/prompt input, no timestamps (needs VAD chunking plus forced alignment), and weaker English-term retention (77%). A *pre-adapter* spot check is possible with the transcribe.cpp CLI on the 5-minute reference excerpt, scored by hand-feeding its text into the suite's scorer if the suite allows it (open question).
3. **Audar-ASR-V1-Turbo, GGUF Q8_0 + BF16 mmproj via `llama-server`.** Use VAD chunking ≤30 s and pass the Vocabulary as the system/context turn. Only do this after the licence check. It would reuse the same "external runtime, text only" adapter shape as #2.
4. **omniASR_LLM_Unlimited_1B_v2 on CPU,** only if 1–3 all disappoint. It is a heavy new dependency (fairseq2).

Decide on MoM critical-fact accuracy, Flagged Passage recall and Vocabulary recognition, as #31 says, not on WER alone. Cohere's lower English-term retention could cost Vocabulary score even if its WER is lower.

## Observed facts vs inferences

**Observed (from the cited sources):**
- OUAAL averages, per-set WERs and the 37-model ranking come from the leaderboard Space's `app.py` (last modified 2026-08-20, which added Audar Turbo/Flash).
- Audar's own GitHub README still says "24.78% avg, #1 of 36". The Space and model card now show 23.17 (#1 of 37). The Space is the authority used here.
- Cohere's "Egyptian 19.16 / code-switched 27.84" are labelled "Cohere internal evaluations". The test sets are not public.
- The Perle/ArzEn numbers are from Sedjem (commit `7665f96`, 2026-09-27): 40 clips, about 650 reference words. Its own paired bootstrap says Cohere−whisper-medium = −5.1 points (95% CI −10.9 to +0.8) and whisper-medium−large-v3 = −2.3 (−7.3 to +2.2). **Neither gap is statistically proven.** It also found that both leading models were "almost certainly" trained on ArzEn.
- Sedjem's Cohere on CPU (i5-1245U): 0.36x real time. large-v3 with hint on CPU: 2.15x.
- vLLM requires compute capability ≥7.5 (docs). That rules it out on the 1060.
- llama.cpp lists `ggml-org/Qwen3-ASR-{0.6B,1.7B}-GGUF` as supported audio models. transcribe.cpp lists Cohere Transcribe Arabic, Qwen3-ASR and Whisper with CUDA/Vulkan/CPU backends.
- faster-whisper converts any Transformers Whisper checkpoint with `ct2-transformers-converter --model <hf id or dir> --output_dir <dir> --copy_files tokenizer.json preprocessor_config.json --quantization <type>`. Use `int8` or `float32` here, since `float16` gives no speedup on Pascal.

**Inferred (not verified, must be measured):**
- VRAM figures for GGUF models are file size plus about 0.5–1 GB of activations/KV cache. All of #1–#3 should fit in 6 GB at Q8_0.
- ggml's CUDA backend runs on compute 6.1 (llama.cpp is commonly used on Pascal cards). Not checked for transcribe.cpp's prebuilt wheels, which may need a source build with `CMAKE_CUDA_ARCHITECTURES=61`.
- On the i7-7700HQ (older and AVX2-only, but 4 big cores) CPU speed should be similar to or slightly slower than Sedjem's i5-1245U. Cohere on CPU should still be well inside the 4x budget.
- A PyTorch (transformers) run of Cohere or Audar on the 1060 would need fp16 (Pascal has no BF16, and fp32 for 2B+ parameters is about 8–9 GB). fp16 numerics and speed on GP106 are uncertain, so GGUF is the safer path.
- Perle (Egyptian tech/work talk) is the public set closest to the user's meetings, but Sedjem's real-meeting check showed 44–49% disagreement with ElevenLabs for both local models. Expect the user's Recording to look more like the 39% baseline than like Perle's 13–20%.

## Open questions

- **Audar Turbo licence:** does a single individual using it on-device count as a "qualifying Community Entity"? The full licence text was not read. (Audar Flash uses the more permissive AudarAI *Open* License, but scores worse: OUAAL avg 32.04.)
- **transcribe.cpp on Pascal:** do the `transcribe-cpp` PyPI wheels include `sm_61` CUDA kernels, or is a source build needed? Is the Cohere GGUF output bit-identical across Q8_0/Q6_K/Q4_K_M? (Sedjem only used Q4_K_M.)
- **Flagged Passages without Whisper confidences:** Cohere/Audar return text only. Which detector signals survive (e.g. transcribe.cpp token probabilities, if exposed)? This affects #31's Flagged Passage recall criterion.
- **Vocabulary biasing:** Cohere documents no prompt or hotword input. Does a post-correction LLM pass (an open item on #31) recover what hotwords give Whisper today?
- **Can the accuracy suite score an externally produced Transcript** (text file) so candidates 2–3 can be measured before an adapter is built?
- **Audar re-test:** does Audar Turbo at Q8_0 with ≤30 s VAD chunks close the gap Sedjem saw on whole clips? There is no independent fair Egyptian measurement yet.
- No Whisper-*large* fine-tune with a credible Egyptian code-switching benchmark was found. Fine-tuning large-v3 or turbo on ArzEn + code-switching data would be the user's own work and is outside this ticket.

## Sources

- Open Universal Arabic ASR Leaderboard: Space https://huggingface.co/spaces/elmresearchcenter/open_universal_arabic_asr_leaderboard (data in `app.py`); repo https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard; paper https://arxiv.org/abs/2412.13788
- Cohere Transcribe Arabic: https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026 · https://huggingface.co/blog/CohereLabs/cohere-transcribe-arabic-07-2026-release · GGUF https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf
- transcribe.cpp: https://github.com/handy-computer/transcribe.cpp
- Audar-ASR-V1: https://huggingface.co/audarai/Audar-ASR-V1-Turbo · https://huggingface.co/audarai/Audar-ASR-V1-Flash · https://github.com/AudarAI/Audar-ASR-V1
- llama.cpp multimodal (Qwen3-ASR audio): https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- Whisper code-switching fine-tune: https://huggingface.co/Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched · https://huggingface.co/Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched-ct2
- Other Whisper fine-tunes: https://huggingface.co/IbrahimAmin/code-switched-egyptian-arabic-whisper-small · https://huggingface.co/MAdel121/whisper-medium-egy · https://huggingface.co/AbdelrahmanHassan/whisper-large-v3-egyptian-arabic · https://huggingface.co/oddadmix/whisper-large-v3-turbo-arabic-dialectal
- Omnilingual ASR: https://github.com/facebookresearch/omnilingual-asr
- Qwen3-ASR: https://huggingface.co/Qwen/Qwen3-ASR-1.7B · https://github.com/QwenLM/Qwen3-ASR
- SeamlessM4T v2: https://huggingface.co/facebook/seamless-m4t-v2-large · MMS: https://huggingface.co/facebook/mms-1b-all
- NVIDIA: https://huggingface.co/nvidia/canary-1b-v2 · https://arxiv.org/abs/2509.14128 · https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0
- vLLM GPU requirements: https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
- faster-whisper conversion: https://github.com/SYSTRAN/faster-whisper
- Independent Egyptian benchmark (third party, small sample): https://github.com/MohammedEl-sayedAhmed/arabic-stt-mvp (`docs/03-results.md`, `docs/results-tables.md`); Perle set https://huggingface.co/datasets/Perle-ai/ASR_Code_Switch, paper https://arxiv.org/abs/2605.19069
