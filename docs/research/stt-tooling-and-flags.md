# Speech-to-text tooling and Flagged Passage recall

Research for #35 (part of map #31). Question: which local tools or signals would improve Transcripts and Flagged Passage recall on a GTX 1060 6 GB / i7-7700HQ / 16 GB RAM within about 4x real time, and why do the current detectors flag about 0% of segments with real errors?

Method: read the repository code, then the installed faster-whisper 1.2.1 and CTranslate2 4.8.2 source and docstrings (`.venv/lib/python3.12/site-packages/faster_whisper/transcribe.py`, `vad.py`, and `ctranslate2._ext` help text), then papers and official repositories. No models were run and the GPU was not used. Line numbers refer to faster-whisper 1.2.1 as installed.

Labels used below: **Fact** means read in code or in a cited primary source. **Inference** means my reasoning from those facts; it has not been measured on the user's Recording.

## Summary

- The current detectors cannot work as built. "Likely text over silence" asks for a condition that faster-whisper has already dropped from its output. "Repetition loop" asks for something that temperature fallback almost always removes first. All three signals are per 30-second decode window, not per segment, and the speech is already cut down by voice activity detection (VAD) before decoding. Part of the 0% is also a labelling problem in the scoring code.
- The most promising signals, ranked:
  1. Word probabilities from `word_timestamps=True`, combined per segment by the minimum or by a count of low-probability words.
  2. Disagreement between two decodes, either two models or the same model with different settings.
  3. Language-ID and script checks per VAD chunk, for code-switched English.
  4. An independent CTC forced-alignment score.
  5. A per-segment VAD speech fraction, to replace `no_speech_prob`.
- Keep Silero VAD. Look at pyannote only if speaker separation (#36) brings in pyannote anyway.
- Treat hotwords as a source of errors that needs its own check, not as a way to find errors.

## Part 1: why the current thresholds almost never fire

### Facts from the code

1. **The flags are set in two places with different rules.**
   - In the app, `Application._mark_segment_flags` (`src/ea_assistant/application.py`) adds "repetition loop" when `compression_ratio > 2.4`. It adds "likely text over silence" only when `no_speech_prob > 0.7` **and** `avg_logprob < -1.0` (`DetectionConfig` in `src/ea_assistant/config.py`).
   - The accuracy suite (`src/ea_assistant/accuracy/flags.py`) adds `low_logprob = avg_logprob < -0.8`. Its "combined" signal is `marker or low_logprob`.
2. **faster-whisper drops the exact case that "text over silence" looks for.**
   - With the defaults the adapter uses (`no_speech_threshold=0.6`, `log_prob_threshold=-1.0`; `transcribe.py` 767–769), `generate_segments` skips a window entirely when `no_speech_prob > 0.6` and `avg_logprob <= -1.0` (`should_skip`, lines 1214–1233).
   - The app's flag needs `no_speech_prob > 0.7` **and** `avg_logprob < -1.0`, which is a strict subset of the skipped case.
   - So no segment can ever reach the app with that flag. It is dead by construction.
3. **Temperature fallback hides repetition loops and low average log-probability.**
   - `generate_with_fallback` (lines 1402–1530) decodes again at higher temperatures whenever `compression_ratio > 2.4` or `avg_logprob < -1.0`.
   - It keeps the first decode that passes both tests.
   - If every temperature fails, it picks the decode with the highest `avg_logprob`, choosing first among decodes that are under the compression threshold.
   - Result: a surviving `compression_ratio > 2.4` means all six temperatures (0.0–1.0) looped, and a surviving `avg_logprob < -1.0` means none passed. The suite's −0.8 cut only catches the narrow band from −1.0 to −0.8.
4. **The signals belong to the window, not the segment.**
   - `avg_logprob`, `compression_ratio`, `no_speech_prob` and `temperature` are computed once per decode window of up to 30 seconds (`generate_with_fallback`).
   - Every segment yielded from that window gets the same values (the `yield Segment(...)` block, lines 1354–1370).
   - `avg_logprob` is `cum_logprob / (seq_len + 1)` over the whole window's tokens (line 1466), so one or two wrong words among about 100 tokens barely move it.
   - Every segment in a window is therefore flagged or not flagged together, which cannot isolate the 40–66 bad segments.
5. **VAD removes the silence before Whisper sees it.**
   - With `vad_filter=True`, `WhisperModel.transcribe` calls `get_speech_timestamps`, then `collect_chunks`, then `np.concatenate` (lines 885–892).
   - Whisper therefore decodes only the joined speech, so `no_speech_prob` stays low by design.
   - The project sets `min_silence_duration_ms=500` and `speech_pad_ms=200` (`DEFAULT_VAD_PARAMETERS`). That is more aggressive than faster-whisper's defaults (2000 ms and 400 ms in `VadOptions`, `vad.py` 37–42; the README says the default "only removes silence longer than 2 seconds").
6. **The suite counts spelling and alias differences as real errors.**
   - `flag_recall` calls `normalize(reference)` and `normalize(segment.text)` with **no Vocabulary aliases** (`flags.py`).
   - `score_text` does pass aliases (`metrics.py`).
   - So a segment where the reference writes an English term in Latin script, and Whisper writes the same word in Arabic script or as an alias, counts as a "real error" at CER > 0.25. No acoustic confidence signal can catch that, because the model heard the word correctly.

### Inferences

- The baseline transcript scores CER 19% normalised and WER 39% normalised (#31), and a segment counts as a real error at CER > 0.25. So a large share of segments are "positives", and recall can only come from signals that vary **within** a window.
- Many of the real errors are probably confident ones: English terms forced into Arabic script because the language is set to `ar`, Egyptian dialect spellings, and names. Whisper is known to be overconfident on errors. [Huo et al., 2025](https://arxiv.org/abs/2509.07195) report that in noise, 10–20% of tokens are wrong while having confidence above 0.7. Probability-only detectors will miss some of these errors.
- Independent evidence that the built-in signals are weak: on human-annotated Whisper large-v3 hallucinations (HALAS, Earnings-22), a classifier using only avg log-prob, compression ratio and no-speech probability reached **23.6% F1**. Internal decoder-state probing reached 62.1%, and late fusion reached 68.3% ([Jasiński et al., 2026](https://arxiv.org/abs/2606.23060)). That paper is about hallucinations, not every kind of error, so it supports rather than proves the point here.

## Part 2: ranked techniques

Cost assumptions:
- The STT run is the expensive part.
- The README's benchmark (RTX 3070 Ti) gives faster-whisper large-v2 at int8 with beam 5 2926 MB of VRAM, and 4500 MB with `batch_size=8` ([faster-whisper README](https://github.com/SYSTRAN/faster-whisper)).
- large-v3 has the same size and architecture, so I expect it to fit in 6 GB the same way (inference).
- The GTX 1060's actual minutes per audio hour come from the #1 suite run and were not available to me, so the cost figures below are relative.

### 0. Fix the measurement and the dead rules first (no runtime cost)

- **What:**
  - Score `flag_recall` with the same alias normalisation as `score_text`.
  - Report how many real errors are script or transliteration mismatches.
  - Delete or rewrite the "text over silence" rule (Part 1, item 2).
  - Sweep the thresholds on the reference instead of using fixed values (−0.8, 2.4), and report precision and recall at each.
- **Evidence:** Part 1 facts.
- **Plugs in:** `accuracy/flags.py`, `DetectionConfig`, `Application._mark_segment_flags`.
- **Why first:** without this, every detector below is measured against labels that include errors no detector could see.

### 1. Word-level probabilities via `word_timestamps=True` (highest value for the cost)

- **Facts:**
  - With `word_timestamps=True`, faster-whisper calls `find_alignment`. That runs CTranslate2's `Whisper.align(encoder_output, sot_sequence, text_tokens, num_frames)` on the window's encoder output, which is already computed (lines 1567+ and 1698+).
  - `WhisperAlignmentResult` has `alignments` and `text_token_probs` ("Probabilities of text tokens"; `ctranslate2._ext` docstring; [CTranslate2 Whisper API](https://opennmt.net/CTranslate2/python/ctranslate2.models.Whisper.html)).
  - The word probability is the **mean** of its token probabilities (`word_probabilities = [np.mean(text_token_probs[i:j]) ...]`).
  - The alignment pass is conditioned on `sot_sequence` only, not on the decode prompt (previous text, `initial_prompt` or hotwords). So these probabilities are not the ones beam search used.
  - faster-whisper already contains a word-level anomaly heuristic: `word_anomaly_score` adds 1.0 for probability < 0.15 and penalises words shorter than 0.133 s or longer than 2.0 s. `is_segment_anomaly` scores the first 8 words, with a threshold of ≥ 3 or close to the word count. It is used only when `hallucination_silence_threshold` is set (lines 1242–1260).
  - Prior work takes word confidence as the **minimum** token probability ([Aggarwal et al., ICASSP 2025, arXiv 2502.13446](https://arxiv.org/abs/2502.13446)).
- **Plug-in:**
  - Request `word_timestamps=True` in `FasterWhisper.transcribe`, and keep `words` (word, start, end, probability) on `Segment`.
  - Add detectors "low-confidence words" and "anomalous word timing", fed by the minimum word probability, the number of words below p, and `word_anomaly_score`.
  - Set the per-segment thresholds by a sweep in the suite.
  - Compute the minimum over tokens directly as well; this needs a small fork of `find_alignment` or a recomputation from `text_token_probs`.
- **Cost (inference):** one teacher-forced decoder pass per window plus DTW, which is much cheaper than beam-5 decoding with fallback. I expect a few to about 15% extra time and no meaningful extra VRAM, but it needs measuring. Turbo's decoder has 4 layers versus 32 in large-v3 ([model card](https://huggingface.co/openai/whisper-large-v3-turbo)), so the pass is cheaper there too.
- **Limits:** the model is overconfident on dialect and code-switched errors (see Part 1). Expect recall gains on garbled or mumbled speech and names, and fewer gains on confident transliterations.

### 2. Disagreement between decodes (best for confident errors)

- **Facts:**
  - Combining several recognisers' outputs by alignment and voting (ROVER) gives lower WER than any single system, and the alignment shows where the systems disagree ([Fiscus 1997, IEEE ASRU, doi:10.1109/ASRU.1997.659110](https://www.nist.gov/publications/post-processing-system-yield-reduced-word-error-rates-recognizer-output-voting-error)).
  - CTranslate2 `generate` supports `num_hypotheses` and `return_scores`, and `return_logits_vocab` (per-step log-probs), which gives N-best lists and top-2 margins without a second encoder pass (4.8.2 docstring).
- **Plug-in:**
  - Run a second decode and character-align it to the primary decode. `accuracy/flags.py` already has a numpy character `_alignment` that could move into the app.
  - Flag segments whose disagreement rate is above a threshold, with the reason "models disagree".
  - Useful second decodes, best first (inference):
    - (a) large-v3 with Vocabulary hotwords versus without them. This exposes hotword insertions, see item 6.
    - (b) large-v3 versus large-v3-turbo.
    - (c) `language="ar"` versus `language="en"` or `None`. This exposes code-switched spans.
    - (d) N-best or temperature samples from the same encoder output.
- **Cost (inference):**
  - (a) to (c) roughly double STT time. They run one after the other in VRAM (one model loaded at a time, as the adapter already does), so peak VRAM is unchanged.
  - Doubling fits the 4x budget only if one large-v3 pass is ≤ about 1.5–2x real time on this GPU. Check that against the #1 timing.
  - `BatchedInferencePipeline` speeds up the second pass ("drop-in replacement"; README large-v2 int8 59 s → 16 s at batch 8). It sets `condition_on_previous_text=False` (line 547) and peaks near 4.5 GB at batch 8.
  - (d) costs an extra decoder pass only.
- **Why it helps where probabilities fail:** two systems rarely make the same confident mistake on a dialect word or a transliterated English term.

### 3. Language-ID and script checks for code-switching

- **Facts:**
  - `multilingual=True` makes faster-whisper run `detect_language` on **each 30 s window's** encoder output and switch the tokenizer language (lines 1192–1198). That is one decision per window, not per segment or word.
  - CTranslate2 `detect_language` returns (language, probability) pairs per input.
  - Changing Whisper's language-token prompt improved zero-shot code-switched ASR ([Peng et al., Interspeech 2023, arXiv 2305.11095](https://arxiv.org/abs/2305.11095); the reported gains across their zero-shot tasks were 10–45%).
  - faster-whisper's public API exposes one `language`, so that kind of prompt needs a custom `generate` call.
- **Plug-in:**
  - Run `detect_language` on each VAD chunk; the chunk boundaries are already known.
  - Flag "possible English spoken" when P(en) is non-trivial, for example ≥ 0.2 (to tune), while the text is all Arabic script.
  - Also flag Arabic-script tokens that match a Vocabulary alias.
  - Optionally run item 2(c) only on those chunks.
- **Cost (inference):** language-ID needs an encoder pass per chunk if the chunks are not the decode windows. On large-v3 the encoder is the bulk of the compute, so restrict it to short chunks or reuse the window encoder output (free when `multilingual=True`).
- **Caveat:** most switches inside a sentence are shorter than a window, so window-level language-ID will dilute them. That is an inference; per-chunk scores are the thing to measure.

### 4. Independent CTC forced alignment as an acoustic check

- **Facts:**
  - WhisperX adds VAD cut and merge plus forced phoneme alignment with a wav2vec2-style model, which improves word timing and reduces "drifting, hallucination & repetition" ([Bain et al., Interspeech 2023, arXiv 2303.00747](https://arxiv.org/abs/2303.00747)).
  - [`ctc-forced-aligner`](https://github.com/MahmoudAshraf97/ctc-forced-aligner) uses [MMS-300m forced-aligner](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner): 0.3B parameters, converted from torchaudio MMS, **CC-BY-NC-4.0**, and it romanises text (uroman) because the CTC vocabulary is Latin. Arabic must be romanised first ([PyTorch MMS forced-alignment tutorial](https://docs.pytorch.org/audio/2.8/tutorials/ctc_forced_alignment_api_tutorial.html)).
- **Plug-in:**
  - After STT, align each segment's text to its audio.
  - Flag words with a very low CTC alignment score or implausible durations, with the reason "text does not match audio".
  - Like item 2, this is a model independent of Whisper, so it can catch confident Whisper errors, especially inserted text (hallucinations or hotwords) and dropped speech.
- **Cost (inference):** a 300M wav2vec2-type model is practical on CPU for a few minutes of audio, or on the GPU after Whisper is unloaded. It needs PyTorch, which the project does not install today (faster-whisper uses CTranslate2 and onnxruntime only). Check PyTorch wheel support for Pascal (sm_61).
- **Caveat:** the CC-BY-NC licence is fine for personal use and should be recorded in an ADR before any wider use. Romanised alignment quality for Egyptian dialect has not been verified.

### 5. VAD choice and settings

- **Facts:**
  - faster-whisper 1.2.1 bundles Silero VAD v6 as ONNX (`faster_whisper/assets/silero_vad_v6.onnx`).
  - Silero is about 2 MB, takes under 1 ms per 30+ ms chunk on one CPU thread, is MIT-licensed, and was trained on corpora covering 6000+ languages ([silero-vad](https://github.com/snakers4/silero-vad)).
  - pyannote `segmentation-3.0` is a 10 s-window powerset model (non-speech, up to 3 speakers, overlaps). It is used as VAD through `VoiceActivityDetection` with `min_duration_on` / `min_duration_off`, is MIT-licensed but gated on Hugging Face, and needs `pyannote.audio` and PyTorch ([model card](https://huggingface.co/pyannote/segmentation-3.0)).
  - I found no head-to-head VAD benchmark from a primary source on meeting audio that I could verify.
- **Plug-in:**
  - Keep Silero.
  - Treat `speech_pad_ms` (200 → 400) and `min_silence_duration_ms` (500 → 1000–2000) as suite parameters. Shorter padding can clip word onsets, and that becomes "real errors" (inference).
  - Use VAD differently for flags: keep the per-frame Silero speech probabilities and flag segments whose word timestamps (item 1) fall mostly on low-speech-probability frames. That replaces the `no_speech_prob` rule, which cannot fire under VAD concatenation.
  - Consider pyannote only if #36 adopts pyannote for speaker separation. Then one segmentation pass can serve both, and overlapped speech, which is a likely error source, becomes a flag reason.
- **Cost:** Silero is negligible. pyannote segmentation is small, but brings PyTorch and a gated download.

### 6. Hotwords: behaviour, limits, and a detector for their side effects

- **Facts** (`get_prompt`, lines 1532–1565):
  - Hotwords are encoded as `" " + hotwords.strip()` and placed after `<|startofprev|>` in **every** window's prompt, before the previous-text tokens.
  - They are silently truncated to `max_length // 2 - 1` = **223 tokens** (`max_length = 448`).
  - They are ignored whenever `prefix` is set.
  - The previous text is also capped at 223 tokens.
  - `generate_with_fallback` raises an error if `len(prompt) + max_new_tokens > 448`. Without `max_new_tokens`, it passes `max_length = 448` (lines 1415–1430). A user hit "Maximum decoding length" errors with prompt plus hotwords at about 401 tokens ([faster-whisper #948](https://github.com/SYSTRAN/faster-whisper/issues/948), closed as not planned).
  - Hotwords are output on **silent** audio (repeated hotword text; [faster-whisper #1356](https://github.com/SYSTRAN/faster-whisper/issues/1356), open).
  - Hotwords are a prompt, not a logit bias. A third-party page claims they "must be a list"; the source shows a `str` that gets tokenised, so that claim is wrong for 1.2.1.
  - The project sends `", ".join(canonical terms)` with no length check in HOTWORDS mode (`PromptSettings.stt_prompt_values`), and the adapter keeps the default `condition_on_previous_text=True`.
- **Inferences:**
  - With a long Vocabulary, later terms are silently dropped.
  - Hotwords plus up to 223 tokens of previous text can leave little or no room for decoding (#948).
  - Terms with Arabic aliases tokenise to many byte-level tokens, so 223 tokens is fewer terms than it looks.
  - Inserted hotwords are likely **confident**, so item 1 will not catch them. Item 2(a) (decode without hotwords) or item 4 (alignment) will.
- **Plug-in:**
  - Count hotword tokens with the model's tokenizer and log what was dropped.
  - Add a "Vocabulary term inserted?" check: a Vocabulary term in the output whose span has low VAD speech probability, or no support from the hotword-free decode.

### 7. Cheap hallucination guards

- **Facts:**
  - faster-whisper's `hallucination_silence_threshold` (needs `word_timestamps=True`) skips silence around segments that `is_segment_anomaly` marks (lines 1294–1340).
  - Whisper hallucinations correlate with non-vocal durations ([Koenecke et al., FAccT 2024, arXiv 2402.08021](https://arxiv.org/abs/2402.08021); about 1% of transcriptions contained whole invented phrases).
  - A "bag of hallucinations" of frequent outputs on non-speech supports a text-based filter ([Barański et al., ICASSP 2025, arXiv 2501.11378](https://arxiv.org/abs/2501.11378)).
- **Plug-in:**
  - Record `is_segment_anomaly` as a flag reason instead of only skipping.
  - Keep a small user-editable list of known Arabic and English hallucination strings, such as subtitle credits, as a "known hallucination phrase" flag.
- **Cost:** negligible once item 1 is in place.

### Not recommended now

- **Trained confidence estimators:** fine-tuned Whisper confidence ([2502.13446](https://arxiv.org/abs/2502.13446)), beam-search feature CEMs ([Jia & Van hamme, 2026, arXiv 2607.29299](https://arxiv.org/abs/2607.29299); not evaluated on Whisper), and internal-state probes ([2606.23060](https://arxiv.org/abs/2606.23060)). All need labelled training data from the user's domain, which one 5-minute reference cannot supply.
- **LLM-based error detection** ([Demystifying Hallucination in Speech Foundation Models, ACL Findings 2025](https://aclanthology.org/2025.findings-acl.1190.pdf)): possible later with the local LLM, but it overlaps the open "LLM post-correction pass" question on #31 and costs LLM time.
- **Different STT models:** covered by #32. Here a second model matters only as a disagreement signal (item 2b).

## Part 3: segment-level versus word-level thresholds

- **Fact:** the current detectors use per-window values copied to each segment (Part 1, item 4).
- **Recommendation (inference):**
  - Decide at word level (minimum probability, anomaly score, CTC score, disagreement span), then raise the flag at segment level with the worst word's reason, so the user still reviews whole segments.
  - Choose thresholds from a precision/recall sweep in `ea accuracy`, with a cap on flags per audio hour so review stays bearable.
  - Use the "combined" OR of detectors only after each one is calibrated.

## Suggested order to measure (input to #38)

1. Item 0, the scoring fix.
2. `word_timestamps=True` with minimum and mean word probability and `word_anomaly_score`. No new dependency.
3. Hotwords on versus off disagreement, plus large-v3 versus turbo disagreement.
4. Per-chunk language-ID flag.
5. VAD padding and silence sweep, plus the VAD speech-fraction flag.
6. MMS CTC alignment score, which adds PyTorch.

## Open questions

- How many of the 40–66 "real error" segments are alias or script mismatches once the scoring is alias-normalised? This decides how much recall any detector can reach.
- What is large-v3 int8_float32 speed on the GTX 1060, in minutes per audio hour, from #1? It decides whether a second full decode fits the 4x budget.
- Is the mean word probability from `align()` (conditioned without the prompt) good enough, or is the beam-search token log-prob (`return_logits_vocab`) better?
- Do current PyTorch wheels still support Pascal sm_61 on CUDA? If not, items 4 and 5 (pyannote) are CPU-only. Not verified.
- Is the CC-BY-NC licence of the MMS aligner acceptable? An ADR is needed if the project goes beyond personal use.
- Is there a primary-source VAD benchmark on conversational or meeting audio comparing Silero v6 and pyannote segmentation-3.0? None verified.
- Can a two-language-token prompt (Peng et al.) be done through faster-whisper without forking it, or does it need a direct CTranslate2 `generate` call?
