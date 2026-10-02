# What a GPU upgrade would buy this pipeline

Research for #37 (part of the wayfinder map #31). Researched 2026-10-02. This note is for the record only: no purchase is planned. Nothing was run on the GPU. Prices are left out because no dated primary source was checked.

**Question.** Which GPU memory tiers (12 / 16 / 24 GB, either as a Thunderbolt eGPU on the current laptop or in a desktop) would unlock which speech-to-text and LLM candidates from #32 and #33? What accuracy and speed gain should we expect? What does a newer architecture bring apart from memory?

**Current machine.** Lenovo `80VR` (DMI product name; Legion Y720-15IKB family), i7-7700HQ, 16 GB RAM, GTX 1060 Mobile 6 GB (GP106M, Pascal, compute capability 6.1, about 192 GB/s).

Labels: **[F]** is a fact from a cited source. **[O]** is observed on this machine. **[I]** is an inference that has not been verified.

## Short answer

| Tier (example cards, CC) | Unlocks: speech-to-text | Unlocks: LLM (Ollama tag sizes) | Expected gain | Caveats |
|---|---|---|---|---|
| **Today: 6 GB Pascal** (GTX 1060, 6.1) | large-v3 / whisper-medium-codeswitched at `int8_float32`; Cohere Transcribe and Audar only as GGUF Q8_0 (#32) | qwen3.5:9b Q4 with a small spill to RAM; gemma4:12b with about 2 GB offloaded; MoE models only at 3-bit and RAM-marginal (#33) | Baseline | No fp16 or bf16 in CTranslate2. No vLLM. PyTorch 2.14 is the last release with Pascal wheels. Driver branch 580 is the last. |
| **12 GB** (RTX 3060 12 GB, 8.6) | Cohere Transcribe in native **BF16** through transformers or vLLM (4.1 GB of weights). Audar Turbo at full precision (its card says ≥12 GB). omniASR_LLM_1B on GPU (about 6 GiB BF16). large-v3 at `int8_float16` / `float16` with `BatchedInferencePipeline` | Everything from #33's shortlist runs **fully on the GPU**: gemma4:12b QAT 7.2 GB or Q4_K_M 8.0 GB, qwen3.5:9b Q4 6.6 GB with a whole-Meeting context, Nile-Chat-12B Q4 7.3 GB. qwen3.5:9b Q8_0 (11 GB) only with a short context | **Speed, mostly.** [I] LLM decoding is about 2–4x faster (no offload, about 1.9x the bandwidth). [I] STT goes from tens of minutes to a few minutes per audio hour. **Accuracy only at the margin.** [I] Removes Q8/Q4 quantisation loss on the STT candidates and makes several-pass decoding (decode disagreement from #35) cheap. | Same model *families* as today, so no new accuracy class. The 3-bit MoEs (11–13 GB) still spill. |
| **16 GB** (RTX 4060 Ti 16 GB, 8.9; RTX 5060 Ti 16 GB, 12.0) | Same as 12 GB, with room to keep STT, a speaker-separation model and Silero VAD loaded together | gemma4:12b **Q8_0 (13 GB)**; qwen3.5:9b Q8_0 (11 GB) with a long context. The 3-bit MoEs from #33's stretch item (Qwen3.5-35B-A3B UD-IQ3_XXS 13.1 GB, Gemma 4 26B-A4B 11.4 GB) fit fully on the GPU instead of being RAM-marginal | [I] Small accuracy gain from Q8 over Q4 on the 9–12B models. The bigger gain is that the MoE stretch candidates become practical: about 3–4B active parameters, so they are fast. | Both example cards use a **128-bit** bus (NVIDIA spec pages), so their bandwidth is close to a 12 GB RTX 3060's. Official 4-bit tags of the 26–35B models (16–24 GB) still do not fit with a KV cache. |
| **24 GB** (RTX 3090, 8.6; RTX 4090, 8.9) | Every STT candidate at full precision, with headroom | **A new accuracy class:** qwen3.5:27b Q4_K_M (17 GB); gemma4:31b QAT (19 GB) or Q4_K_M (20 GB), which is tight with a KV cache; gemma4:26b-a4b Q4_K_M (18 GB). qwen3.5:35b-a3b Q4_K_M (24 GB) still does not fit | **[F] Model-card deltas:** Qwen3.5-27B WMT24++ 77.6 vs 72.6 for 9B, IFEval 95.0 vs 91.5. Gemma 4 31B MMMLU 88.4 vs 83.4 for 12B. [I] Expect better English Transcripts and fewer missed MoM facts. Only `ea accuracy` can confirm this on Egyptian code-switched speech. [I] A dense 27B model at about 900+ GB/s should decode at roughly 30–45 tok/s, which is well within budget. | Only this tier changes *which* LLM is used, not just how fast it runs. |

**Conclusion.**

- The 4x real-time budget is not the bottleneck today (#33 estimates 12–25 min of LLM time per audio hour). Every candidate on the shortlists already fits in 6 GB with GGUF or offload. A 12 or 16 GB card therefore mostly buys **speed and headroom**, not accuracy.
- **24 GB is the first tier that unlocks a better model class**: 27–31B dense LLMs at 4-bit. That is where a real gain in Summary/MoM quality and English Transcript quality could come from.
- On the speech-to-text side, the most likely gains do not depend on memory. They depend on **architecture** (≥ 7.5 for vLLM, ≥ 8.0 for BF16), because these let Cohere Transcribe and Audar run in their native stacks instead of community GGUF ports.
- The strongest reason to upgrade eventually is **Pascal's end of life**, not VRAM. Pascal is stuck on PyTorch ≤ 2.14, CUDA 12.x, and driver branch 580 (security fixes until October 2028). Any new card should be **Ampere (8.x) or newer**: Turing (7.5) gets vLLM but has no BF16 and no FlashAttention-2.
- If buying, a **24 GB desktop** beats a 24 GB eGPU. A desktop also lifts the 16 GB RAM limit (MoE offload) and the 4-core CPU, and avoids the Thunderbolt link described below.

## Non-memory gains of a newer architecture

| Feature | Pascal 6.1 (today) | Needs | Source |
|---|---|---|---|
| CTranslate2 `float16` | No | CC ≥ 7.0 | [F] [CTranslate2 quantization docs](https://opennmt.net/CTranslate2/quantization.html) |
| CTranslate2 `bfloat16` / `int8_bfloat16` | No (falls back) | CC ≥ 8.0 | [F] same |
| CTranslate2 `int8` | Yes (6.1 is explicitly listed) | CC ≥ 7.0 or 6.1 | [F] same |
| CTranslate2 AWQ 4-bit | No | CC ≥ 7.5 | [F] same |
| CTranslate2 flash attention | n/a | Removed from the Python wheel in 4.4.0; C++ build flag only | [F] [CTranslate2 CHANGELOG](https://github.com/OpenNMT/CTranslate2/blob/master/CHANGELOG.md) |
| vLLM (the recommended serving path in the Cohere Transcribe card; used by Audar) | No | CC ≥ 7.5 | [F] [vLLM GPU install](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/), [Cohere card](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) |
| FlashAttention-2 (transformers / PyTorch stacks) | No | Ampere, Ada, Hopper (Turing via a separate fork) | [F] [flash-attention README](https://github.com/Dao-AILab/flash-attention) |
| llama.cpp / Ollama flash attention (needed for a q8_0 KV cache) | Yes, but on the slow "vec" fallback kernels. There is an unfixed crash report at >24k context (closed as not planned) | Tensor cores (Turing+) for the MMA kernels | [F] [llama.cpp #22032](https://github.com/ggml-org/llama.cpp/issues/22032) |
| Prompt processing (prefill) on 12–20k-token Meeting inputs | CUDA cores only; GP106 has no tensor cores | Tensor cores | [I] Prefill is compute-bound, so this is where Turing+ gains most beyond bandwidth |
| PyTorch prebuilt wheels | ≤ 2.14 (cu126). cu128+ wheels dropped sm_50–sm_70 | Any newer architecture | [F] [PyTorch notice, 2026-09-04](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432) |
| CUDA toolkit | 12.x only. 13.0 removed offline compilation and library support for Maxwell, Pascal and Volta | Turing+ | [F] [CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-toolkit-release-notes/index.html) |
| NVIDIA driver | R580 is the last branch. Security-only updates from October 2025 through October 2028 | n/a | [F] [NVIDIA support plan](https://nvidia.custhelp.com/app/answers/detail/a_id/5676/~/support-plan-for-maxwell%2C-pascal%2C-and-volta-series-geforce-gpus.) |
| Speaker separation: Nemotron-3-Diarization | Not on the supported list | Ampere+ | [F] per `docs/research/speaker-separation.md` (#36), [card](https://huggingface.co/nvidia/Nemotron-3-Diarization) |
| Ollama | Supported (CC ≥ 5.0; 5.0–6.2 need driver ≥ 570) | n/a | [F] [Ollama GPU docs](https://docs.ollama.com/gpu) |

**Speed reference [F].** faster-whisper's README benchmark (large-v2, 13 min of audio, RTX 3070 Ti 8 GB, CUDA 12.4) reports these times: fp16 1m03s (4.5 GB), int8 59s (2.9 GB), batched (batch_size 8) fp16 17s and int8 16s ([faster-whisper](https://github.com/SYSTRAN/faster-whisper)). [I] Scaled up, an Ampere card should transcribe an audio hour in about 1–5 min, with the same accuracy as today at the same model and settings. The faster-whisper speed-up alone does **not** change Transcript accuracy.

## Example cards (facts from NVIDIA pages, bandwidth derived)

| Card | VRAM / bus (NVIDIA spec page) | CC ([CUDA GPUs](https://developer.nvidia.com/cuda-gpus)) | Bandwidth [I]: bus width × commonly published memory speed; not on NVIDIA's pages |
|---|---|---|---|
| GTX 1060 Mobile (today) | 6 GB GDDR5, 192-bit | 6.1 | ~192 GB/s |
| [RTX 3060](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/) | 12 GB GDDR6, 192-bit (an 8 GB 128-bit variant also exists) | 8.6 | ~360 GB/s |
| RTX 4060 Ti 16 GB | 16 GB, 128-bit | 8.9 | ~288 GB/s |
| [RTX 5060 Ti 16 GB](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/) | 16 GB GDDR7, 128-bit (an 8 GB variant also exists) | 12.0 | ~448 GB/s |
| [RTX 3090](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/) | 24 GB GDDR6X, 384-bit | 8.6 | ~936 GB/s |
| [RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) | 24 GB GDDR6X, 384-bit | 8.9 | ~1,000 GB/s |

LLM model sizes are taken from Ollama's tag pages: [gemma4](https://ollama.com/library/gemma4/tags) and [qwen3.5](https://ollama.com/library/qwen3.5/tags). The model-card scores are from [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) and [gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it). STT memory figures are from #32 (`docs/research/stt-candidates.md`).

[I] **KV cache for Qwen3.5-27B.** Its card lists 16 full-attention layers × 4 KV heads × 256 dim. That works out to about 64 KiB per token, so a 32k context needs about 2 GiB. With the 17 GB Q4_K_M weights, it fits in 24 GB.

## Laptop eGPU vs desktop

**Does this laptop have Thunderbolt 3?** It depends on the laptop model, not the CPU: the i7-7700HQ has no integrated Thunderbolt, so it needs a separate Intel controller. **On this machine [O]**, `lspci -nn` shows an `Intel JHL6340 Thunderbolt 3 Bridge (C step) [Alpine Ridge 2C 2016] [8086:15da]` and a `JHL6340 Thunderbolt 3 NHI`. The `thunderbolt` kernel module is loaded. The controller sits behind **chipset root port 00:1c.4**, not the CPU's x16 lanes. [F] The JHL6340 is a single-port controller with a 4 × PCIe Gen 3 host interface ([Intel 6000-series brief](https://www.thunderbolttechnology.net/sites/default/files/18-241_ThunderboltController_Brief_HI.pdf), [ARK](https://www.intel.com/content/www/us/en/products/sku/94032/intel-jhl6340-thunderbolt-3-controller/specifications.html)). Thunderbolt 3 carries "4 lanes of PCI Express Gen 3" inside a 40 Gbps link ([Intel TB3 brief](https://www.intel.com/content/dam/www/public/us/en/documents/product-briefs/thunderbolt-overview-brief.pdf)).

Still to verify, by the user with root:

```sh
lspci -nn | grep -iE 'thunderbolt|JHL'             # controller present? (yes here)
sudo lspci -vv -s 00:1c.4 | grep -E 'LnkCap|LnkSta' # OEM wiring: x4 or x2, 8GT/s?
boltctl domains                                     # one domain per controller, shows the security level
boltctl list -a                                     # devices; authorize an enclosure with `boltctl enroll <uuid>`
cat /sys/bus/thunderbolt/devices/domain0/security   # none / user / secure / dponly
```

In this agent's sandbox, `boltctl domains` printed nothing and `/sys/bus/thunderbolt` was absent, even though the controller and module are present. This is probably a sandbox or BIOS power-state effect ([boltctl(1)](https://manpages.debian.org/testing/bolt/boltctl.1.en.html)). Also check that the BIOS does not set Thunderbolt to `dponly` (display only), and that the USB-C port carries the lightning-bolt mark.

**eGPU caveats [I].**

- The link is PCIe 3.0 x4 at best: about 32 Gbps raw, about 2.5–3 GB/s usable. That is fine once a model sits fully in VRAM, since decode speed is then set by the card's own bandwidth. It hurts model loading (about 7 s per 17 GB at best) and any CPU↔GPU split, which is exactly what MoE offload needs.
- The link also passes through the chipset's DMI link, which is shared with the NVMe SSD (the SSD is on root port 00:1d.0).
- An eGPU keeps the 16 GB RAM and the 4-core CPU, so the "RAM-marginal" MoE caveat from #33 remains for anything that does not fit in VRAM.
- **Driver lock-in.** Running the GTX 1060 and a new card on one driver pins the system to R580, the last branch for Pascal. A Blackwell card (CC 12.0) needs a newer CUDA (12.8+) and current PyTorch wheels, while the 1060 stays on ≤ 2.14 / CUDA 12.x. In practice, the 1060 would be ignored for compute.
- A desktop removes all of this and can also add RAM (for example 64 GB). That RAM makes official 4-bit MoE tags such as qwen3.5:35b-a3b (24 GB) practical with partial offload.

## Open questions / not verified

- Actual PCIe link width and speed of the Thunderbolt root port (`LnkCap`/`LnkSta` need root). Whether `boltd` sees the domain outside the sandbox, and the BIOS Thunderbolt security mode.
- Card memory bandwidths: NVIDIA's spec pages give only bus width, so the bandwidth column is derived.
- Whether CTranslate2 4.x PyPI wheels include `sm_120` (Blackwell) kernels or rely on PTX JIT. Not checked.
- The accuracy gain from 27–31B LLMs on **Egyptian code-switched** input. Only general multilingual model-card scores exist, so `ea accuracy` must decide.
- The STT accuracy difference between GGUF Q8_0 (6 GB path) and native BF16 for Cohere Transcribe and Audar. No published comparison was found.
- Real per-hour timings on the 1060 for each stage. These come from the measurement ticket, not from this note.
