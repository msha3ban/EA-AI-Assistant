# Local LLM candidates for the English Transcript and MoM extraction

Research for issue #33 (map #31). Researched 2026-10-02. Nothing was downloaded or run; every throughput figure below is an **inference** from memory bandwidth and file size, and must be measured with `ea accuracy`.

**Question.** Which local LLMs should replace or join `qwen3:8b` / `qwen2.5:7b` for the English Transcript (Egyptian Arabic/English → English) and MoM extraction (strict JSON schema via Ollama `format`)? Hardware: GTX 1060 6 GB (Pascal, CC 6.1), i7-7700HQ 4C/8T, 16 GB RAM, CPU offload allowed, budget up to ~4× real time per audio hour.

## Short answer

Try these in the accuracy suite, in this order. Details and evidence follow.

1. **`qwen3.5:9b`** (Qwen3.5-9B, Q4_K_M, non-thinking) for both stages. It is the strongest multilingual model in the size class that has an official Ollama tag. Its hybrid attention makes a whole-Meeting context cheap.
2. **`gemma4:12b`** (Gemma 4 12B, Q4_K_M or Google's QAT Q4_0), mainly for the English Transcript. Its predecessor Gemma-3-12B was the best small open model on Egyptian→English in an independent 2026 benchmark, and Qwen3-8B ranked near the bottom.
3. **`gemma4:e4b`** (Gemma 4 E4B, Q4): fits fully on the GPU and is fast. This is the fallback if 1 and 2 are too slow.
4. **Nile-Chat-12B or Nile-Chat-4B** (Egyptian-specific, Gemma 3 base, GGUF via `hf.co/...`), for the **English Transcript stage only**, with small chunks. It has the best Egyptian→English numbers published, but they come from the authors' own benchmark. It was trained at 2,048 tokens and has no JSON evidence.
5. Keep **`qwen2.5:7b`** as the baseline. Drop `qwen3:8b`.
6. Stretch goal, only if 1–4 miss critical facts: a **3-bit MoE**, either Qwen3.5-35B-A3B (UD-IQ3_XXS, 13.1 GB) or Gemma 4 26B-A4B (UD-IQ3_XXS, 11.4 GB). These are RAM-marginal on 16 GB.

There is also a likely cause of the intermittent invalid/truncated JSON that is independent of the model choice. The app's default `temperature = 0.0` is greedy decoding, and the Qwen3 model card says "DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions". A repetition loop runs until `num_predict` and leaves truncated JSON. See [Open questions](#open-questions).

## Context: what the pipeline asks of the model (facts from this repo)

- Ollama `/api/chat` with `format` = JSON schema; one retry on JSON/schema failure (`src/ea_assistant/ollama.py`, `application.py`).
- Defaults (`src/ea_assistant/config.py`): model `qwen3:8b`; translate and extract `num_ctx 8192`, `num_predict 4096`, `think false`, `temperature 0.0`; summary `num_ctx 4096`, `num_predict 512`. Input is chunked to `(num_ctx − num_predict) × 2.5` characters.
- Prompts are short instructions (`src/ea_assistant/prompts.py`). The extract prompt does not include the schema text. Ollama recommends also passing the schema in the prompt ([Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)).

## Hardware facts used for estimates

| Item | Value | Source |
|---|---|---|
| CPU memory bandwidth | 37.5 GB/s max, DDR4-2400, 2 channels, AVX2 (no AVX-512) | [Intel ARK i7-7700HQ](https://www.intel.com/content/www/us/en/products/sku/97185/intel-core-i77700hq-processor-6m-cache-up-to-3-80-ghz/specifications.html) |
| GPU memory bandwidth | ~192 GB/s (192-bit, 8 Gbps GDDR5); not re-verified against an NVIDIA page | widely published spec, unverified |
| Ollama on Pascal | CC 5.0+ supported; CC 5.0–6.2 needs driver ≥ 570 (repo verified driver 580) | [Ollama GPU docs](https://docs.ollama.com/gpu) |
| KV-cache quantisation | `OLLAMA_KV_CACHE_TYPE=q8_0` ≈ ½ the memory of f16, needs flash attention; Ollama default context 4096 | [Ollama FAQ](https://docs.ollama.com/faq) |

**Rule of thumb (inference).** Decoding is memory-bandwidth-bound: tokens/s ≈ effective bandwidth ÷ bytes read per token. The estimates assume about 125 GB/s effective on the GPU and about 22 GB/s on the CPU. A dense model that spills 1 GB to RAM loses about 45 ms per token. A MoE model reads only its active experts.

**Workload size (inference, to be measured).** One hour of meeting speech is roughly 7–9k words. That is about 12–20k input tokens of Arabic-script text, depending on the tokenizer, and roughly 9–12k tokens of English output. The English Transcript dominates generation time. At 8 tok/s that is about 25 min of decoding per audio hour; at 15 tok/s, about 12 min. Both are well inside the ~240 min budget alongside speech-to-text. Throughput is therefore not the binding constraint, and slower but more accurate models are acceptable. Measure the real counts from Ollama's `prompt_eval_count` and `eval_count` on the suite's 5-minute excerpt.

## Shortlist

The VRAM/RAM column is the quantised file size plus the KV cache at the stated context (f16 KV). The tok/s column is an **inference** for this machine, unmeasured. "Ollama?" means there is an official library tag; `hf.co/...` means a community or official GGUF pulled through Ollama's Hugging Face integration.

| # | Model (quant) | Arabic / translation evidence (facts) | Licence | Memory at quant | Est. decode tok/s (inference) | Ollama? | Fits? |
|---|---|---|---|---|---|---|---|
| 1 | **Qwen3.5-9B** (Q4_K_M 5.68 GB; Q3_K_M 4.67 GB) | Model card: WMT24++ **72.6** (Qwen3-30B-A3B 69.3), MMMLU 81.2, INCLUDE 75.6, IFEval 91.5; 201 languages. No Egyptian-specific result found. | Apache-2.0 | Q4_K_M ≈ 5.7 GB + KV 32 KiB/token (only 8 of 32 layers use full attention) → 16k ctx ≈ +0.5 GiB, 32k ≈ +1 GiB. About 0.5–1 GB spills to RAM on a 6 GB card; Q3_K_M fits fully. | Q4_K_M partial offload 10–18; Q3_K_M on GPU 18–25 | `qwen3.5:9b` (6.6 GB incl. vision projector) | **Yes** (small spill at Q4) |
| 2 | **Gemma 4 12B** (Google QAT Q4_0 6.98 GB; Ollama Q4_K_M ~7.7 GB) | Model card: MMMLU **83.4** (Gemma 3 27B: 70.7); 140+ languages pretrain, 35+ supported. Predecessor Gemma-3-12B was **best small open model** on Egyptian→English in Alexandria (spBLEU 40.6, chrF++ 60.2, vs Qwen3-8B 33.1 / 53.1). | Apache-2.0 | ~7 GB weights; global KV ~8–16 KiB/token (8 global layers, 1 KV head) + sliding-window cache (40 layers × 1,024 tokens) ≈ 0.3 GiB → ~1.5–2.5 GB in RAM | 5–9 | `gemma4:12b` | **Yes, with offload** |
| 3 | **Gemma 4 E4B** (QAT Q4_0 5.15 GB incl. per-layer embeddings; Ollama Q4_K_M 6.6 GB incl. vision/audio) | MMMLU **76.6** (model card). Successor of Gemma-3-4B (Alexandria EG→EN spBLEU 36.4, above Qwen3-8B's 33.1). | Apache-2.0 | ~5 GB; effective 4.5B parameters per token; small KV (7 full-attention layers, KV sharing) | 15–30 | `gemma4:e4b` | **Yes** |
| 4 | **Nile-Chat-12B** (Q4_K_M 7.3 GB; Q3_K_M 6.0 GB) / **Nile-Chat-4B** (Q4_K_M 2.49 GB) | Egyptian-specific (Gemma 3 base; Arabic script + Arabizi). Authors' EgyptianBench long translation BLEU/chrF: 12B **40.53/60.61**, 4B 37.49/58.40 vs Qwen2.5-7B 19.89/44.80, Gemma-3-12B 22.90/45.97, ALLaM-7B 26.57/52.59. Directions mix EGY↔EN and EGY↔MSA. | Gemma terms | 12B: KV 64 KiB/token (8 global layers) + 0.3 GiB sliding; 4B: ~24 KiB/token | 12B 5–9 (Q4), 4B 30–45 | No official tag; `hf.co/mradermacher/Nile-Chat-12B-GGUF` | **Yes** (12B with offload); translate stage only |
| 5 | **Qwen2.5-7B** (Q4_K_M ~4.7 GB) — baseline | Found the most facts in the user's direct test (observed). Nile-Chat table: long translation 19.89 BLEU. Jais-2 card: Arabic IFEval prompt 46.04. | Apache-2.0 | 56 KiB/token → 16k ≈ +0.9 GiB; fits on GPU at 8k | 22–30 | `qwen2.5:7b` | **Yes** |
| 6 | **Qwen3.5-35B-A3B** (UD-IQ3_XXS 13.1 GB) or **Qwen3.6-35B-A3B** (13.2 GB) | Same family as #1 at larger scale (256 experts, 8 active, ~3B active). Its own scores were not fetched. | Apache-2.0 | ~13 GB: ~4–5 GB on GPU, ~8–9 GB in RAM; KV 20 KiB/token | 6–12 | Official tags are Q4_K_M 24 GB (**does not fit**); 3-bit only via `hf.co/unsloth/...` | **Marginal** (RAM) |
| 7 | **Gemma 4 26B-A4B** (UD-IQ3_XXS 11.4 GB; UD-Q3_K_M 12.7 GB) | MMMLU **86.3**; 3.8B active | Apache-2.0 | ~11–13 GB split GPU/RAM | 5–10 | Official `gemma4:26b` is 16–19 GB (**does not fit**); 3-bit via `hf.co/unsloth/...` | **Marginal** (RAM) |
| – | qwen3:8b (current, Q4_K_M 5.2 GB) | Alexandria EG→EN spBLEU **33.1** (only ALLaM-7B and Qwen3-4B lower among models listed); Jais-2 card AraGen-12-24 **36.52** (last of 14) but Arabic IFEval 58.66 (2nd). Intermittent invalid JSON observed. | Apache-2.0 | KV **144 KiB/token** → 16k ≈ +2.25 GiB, which forces offload at full-Meeting contexts | 15–25 at 8k | `qwen3:8b` | Yes, but **replace** |

### Considered and not shortlisted

| Model | Why not (facts) |
|---|---|
| Command R7B Arabic (`command-r7b-arabic:7b`, 5.1 GB, 16k ctx in Ollama) | Good Arabic instruction following (IFEval Arabic 69.0 on own card; 62.38/70.57 on Jais-2 card). **CC-BY-NC** licence, and EA work for an employer is likely commercial use. Alexandria EG→EN spBLEU 36.1, below Gemma-3-4B. Gated repo. |
| Aya Expanse 8B (`aya:8b`) | **CC-BY-NC**, 8K context. Alexandria EG→EN 37.8 spBLEU, below Gemma-3-12B. |
| ALLaM-7B-Instruct-preview | Apache-2.0, but **4,096-token** context and full multi-head attention (32 KV heads → **512 KiB/token**). Strong in Nile-Chat's table (26.57 BLEU) but lower tier in Alexandria (EG→EN 33.0). No official Ollama tag. |
| Fanar-1-9B-Instruct | Apache-2.0, Gemma 2 base, **4,096-token** context. Alexandria EG→EN 40.7 spBLEU, level with Gemma-3-12B. A possible translate-only alternative to #4 if Nile-Chat disappoints. Community GGUFs only. |
| SILMA-9B-Instruct v1.0 | Gemma 2 base (2024); OALL MMLU-ar 52.55. Superseded by the Gemma 4 / Qwen3.5 generation. Gemma licence. |
| Jais-2-8B-Chat | Apache-2.0, 8,192-token context, claims Arabic–English code-switching, AraGen-12-24 **58.64** (best in its card's table). Gated, custom `Jais2` architecture; llama.cpp/Ollama support not verified; no Egyptian→English number found. **Worth revisiting** once a GGUF is confirmed to run in Ollama. |
| Falcon-H1-Arabic-7B | OALL 71.7 and 256k context per TII blog; covers Egyptian dialect. Hybrid Mamba. The HF repo returned 401 to me and no official Ollama tag exists, so licence and GGUF were **not verified**. Worth revisiting. |
| Jais family 6.7B / jais-adapted-7b | Alexandria and Nile-Chat show them well below Qwen2.5/Gemma 3 (e.g. long translation 12.71 BLEU). |
| qwen3:14b (9.3 GB), Qwen3-30B-A3B-2507 (Q3_K_M 14.7 GB) | Superseded by Qwen3.5-9B on its card (WMT24++ 72.6 vs 69.3 for 30B-A3B), with a larger KV cache. |
| Qwen3.8-27B, Qwen3.8-Flash-Next (125B total / 6B active) | Too large for 22 GB total memory. |

## Evidence notes (facts, with caveats)

- **Alexandria (Jan 2026)** is a multi-domain dialectal Arabic MT benchmark covering 13 dialects with context-level conversational turns ([arXiv 2601.13099](https://arxiv.org/abs/2601.13099)). The Dialect→English numbers for the **EG** column were read from the PDF text of Figure 5 (spBLEU) and Figure C.2 (chrF++):

  | Model | EG→EN spBLEU | EG→EN chrF++ |
  |---|---|---|
  | gemma-3-27b-it | 43.0 | – |
  | Fanar-1-9B-Instruct | 40.7 | 59.4 |
  | gemma-3-12b-it | 40.6 | 60.2 |
  | Qwen3-Next-80B-A3B | 40.6 | – |
  | aya-expanse-8b | 37.8 | 57.1 |
  | Qwen3-32B | 37.0 | – |
  | gemma-3-4b-it | 36.4 | 56.4 |
  | c4ai-command-r7b-arabic | 36.1 | 55.5 |
  | Qwen3-8B | 33.1 | 53.1 |
  | ALLaM-7B-Instruct-preview | 33.0 | 51.2 |
  | Qwen3-4B | 28.6 | 49.4 |

  Caveats. Decoding was greedy (temperature 0), which Qwen3's card advises against, so Qwen3 may be under-rated. The benchmark is dialect-only, not code-switched. It does not include Qwen2.5, Qwen3.5, Gemma 4 or Nile-Chat. The paper concludes "overall model strength is a strong predictor" and that small open models such as ALLaM-7B and Fanar-1-9B sit in a lower tier on its other views.
- **Nile-Chat** ([arXiv 2507.04569](https://arxiv.org/abs/2507.04569), [model card](https://huggingface.co/MBZUAI-Paris/Nile-Chat-12B)). Its translation set combines EGY↔EN and EGY↔MSA, with "long" documents drawn from Egyptian Wikipedia; this is the authors' own benchmark. The paper states "maximum input context length was configured to 2,048 tokens" in training, and the config allows 131k (Gemma 3). MoE variants Nile-Chat-2x4B-A6B and 3x4B-A6B score 41.98 and 42.43 long-translation BLEU, but they have no GGUF verified here.
- **Qwen3.5** ([Qwen3.5-9B card](https://huggingface.co/Qwen/Qwen3.5-9B), [config](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)). Released Feb 2026, Apache-2.0, 262k context. The 9B is dense with Gated DeltaNet linear attention on 24 layers and full attention on 8 (4 KV heads × 256 dim), which gives the 32 KiB/token KV figure. Thinking is on by default and disabled with `enable_thinking: False`. Recommended non-thinking sampling is temperature 0.7, top_p 0.8, top_k 20 and presence_penalty 1.5. Ollama tags: [qwen3.5](https://ollama.com/library/qwen3.5/tags). GGUF sizes: [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF).
- **Gemma 4** ([gemma-4-12B-it card](https://huggingface.co/google/gemma-4-12B-it), [E4B card](https://huggingface.co/google/gemma-4-E4B-it)). Apache-2.0. E2B/E4B have 128k context; 12B, 26B-A4B and 31B have 256k. Thinking is off unless `<|think|>` is placed in the system prompt. Config (12B): 48 layers, of which 40 are sliding (window 1,024) and 8 are full attention with 1 global KV head × 512 dim and `attention_k_eq_v`. Official QAT GGUF: [google/gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf). Ollama tags: [gemma4](https://ollama.com/library/gemma4/tags). The card publishes MMMLU only; no WMT24++ number for Gemma 4 was found.
- **Jais-2-8B** ([card](https://huggingface.co/inceptionai/Jais-2-8B-Chat)). Its comparison tables are the only primary source found that places Qwen2.5-7B, Qwen3-8B, Command R7B Arabic, ALLaM, Fanar, Aya Expanse and Falcon-H1-7B on the same Arabic IFEval and AraGen runs. Selected rows (Arabic IFEval prompt / AraGen 3C3H): Command R7B Arabic 62.38 / 49.18; Qwen3-8B 58.66 / 36.52; Fanar 48.27 / 53.16; Qwen2.5-7B 46.04 / 47.46; ALLaM v1 45.54 / 53.16; Jais-2-8B 58.17 / 58.64.
- **Command R7B Arabic** ([card](https://huggingface.co/CohereLabs/c4ai-command-r7b-arabic-02-2025)): CC-BY-NC, 128k context (16k in the Ollama tag), IFEval Arabic 69.0 vs Qwen2.5-7B 62.4.
- **Aya Expanse 8B** ([card](https://huggingface.co/CohereLabs/aya-expanse-8b)): CC-BY-NC, 8K context, 23 languages.
- **Falcon-H1-Arabic** ([HF blog](https://huggingface.co/blog/tiiuae/falcon-h1-arabic), [TII page](https://falconllm.tii.ae/falcon-h1-arabic.html)): 3B/7B/34B, released Jan 2026, OALL 7B 71.7.
- **Qwen3-8B sampling** ([card](https://huggingface.co/Qwen/Qwen3-8B)): non-thinking temperature 0.7, top_p 0.8, top_k 20; "DO NOT use greedy decoding"; presence_penalty 0–2 reduces endless repetitions.
- **KV-cache sizes** were computed from each model's `config.json` as (full-attention layers × KV heads × head_dim × 2 × 2 bytes). The sliding-window caches are capped at the window size. Whether Ollama's engine caps sliding-window caches for Gemma 4 and Nile-Chat is **an inference**, not verified.

## Inferences (labelled)

- The current JSON failures are probably **sampling-induced repetition loops** (greedy decoding) hitting `num_predict 4096`, not a fundamental limit of Qwen3. The same risk applies to Qwen3.5. Any run with a Qwen model should use the card's non-thinking sampling plus `presence_penalty`, or at least `repeat_penalty`, before it is judged.
- **Qwen3.5-9B's hybrid attention changes the context-length trade-off.** A whole 1-hour English Transcript (~12k tokens) plus schema and output fits in about 0.6 GiB of KV, against about 2.3 GiB for qwen3:8b. This makes single-pass direct extraction (no chunk merging) feasible on 6 GB.
- Gemma 4 12B should beat Gemma-3-12B on Egyptian→English, because the same family rose from 70.7 (Gemma 3 27B) to 83.4 MMMLU. **Not measured anywhere I found.**
- The best pairing may be a **split model per stage**: an Arabic-strong translator (Gemma 4 12B or Nile-Chat) for the English Transcript, and the most schema-reliable model (Qwen3.5-9B or Qwen2.5-7B) for extraction. Each stage unloads the previous model, so stage-specific models cost load time only.
- Throughput figures assume the GPU is otherwise idle and Whisper is unloaded (the app unloads between stages).

## Open questions

1. **Ollama on Pascal with new architectures.** Do `qwen3.5` (Gated DeltaNet) and `gemma4` run on CUDA CC 6.1 in the installed Ollama version, and is flash attention (needed for a q8_0 KV cache) used on Pascal? Check `ollama ps` and the server log on first load.
2. **Does Ollama load the vision projector** for `qwen3.5:9b` (0.9 GB) and `gemma4:*` text-only requests? If it does, an `hf.co/unsloth/...` text-only GGUF or a Modelfile without the projector may avoid the spill. Does `hf.co/` import support the `qwen35` / `gemma4` architectures in Ollama's runner?
3. **Real token counts** for a 1-hour Meeting (Arabic input and English output) from `prompt_eval_count` / `eval_count` on the 5-minute excerpt × 12.
4. **Sampling and prompt.** Does switching the suite from `temperature 0` to card-recommended sampling (plus `presence_penalty`) remove the qwen3:8b JSON failures? Does adding the schema text to the extract prompt, as Ollama recommends, change fact recall? These are configuration and prompt experiments, so a task ticket should own them.
5. **Nile-Chat at longer chunks.** It was trained at 2,048 tokens. Does quality hold at translate chunks of ~2k input + ~1.5k output? This needs `[llm.stages.translate] num_ctx 4096` and smaller chunks.
6. **Not verified:** Falcon-H1-Arabic licence, HF availability and Ollama compatibility (HF returned 401); Jais-2-8B GGUF runnability in Ollama; GTX 1060 bandwidth from an NVIDIA primary page; Qwen3.5-35B-A3B's own multilingual scores.
7. **Code-switched evidence gap.** No benchmark found measures *code-switched* Egyptian Arabic/English → English on 2026 models. The user's own suite is the only reliable judge.
