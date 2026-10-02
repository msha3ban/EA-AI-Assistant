# EA-AI-Assistant

Local-only processing for Egyptian Arabic and English meeting Recordings. It creates a Transcript, English Transcript, Summary and Draft MoM as Markdown in one Meeting folder, with state in SQLite. AI calls use local faster-whisper and Ollama; remote Ollama endpoints are rejected unless explicitly enabled in config.

Install with `uv sync --extra stt`, then run `ea setup-models` to explicitly download the configured speech-to-text model. `ea preflight` reports runtime readiness. Process a Recording with `ea process recording.m4a --title "Title" --date 2026-10-01`.

Optional TOML config supports `data_dir`, `[stt]` (`model`, `device`, `compute_type`, `language`, `beam_size`, `vad_filter`, `vad_parameters`, `download_root`), `[llm]` (`endpoint`, `model`, `timeout`, `unload_timeout`, `unload_poll_interval`, `[llm.stages.translate|extract|summary]` with `num_ctx`, `num_predict`, `think`, `temperature`), `[thermal]` (`hard_stop_c`, `require_ac`), `[detection]` (`compression_ratio`, `no_speech_prob`, `avg_logprob`), and `allow_remote_endpoint`.

Verified Pascal hardware profile: GTX 1060 (compute capability 6.1), NVIDIA driver 580.178.04, ctranslate2 4.8.2, faster-whisper 1.2.1, cuBLAS 12.9.2.10 and cuDNN 9.27.0.42 (pip wheels in the `stt` extra, preloaded automatically). `int8` runs as `int8_float32` on this card. Supported CUDA compute types on this device are `int8`, `int8_float32`, and `float32`. `ea preflight` runs a one-second local speech-to-text inference and reports the effective compute type, driver, CUDA, and cuDNN details where available.

Normal processing defaults to faster-whisper `large-v3`, `int8`, Arabic, VAD enabled, and Vocabulary hotwords; Ollama defaults to `qwen3:8b` with thinking off. The Summary stage uses `num_ctx = 8192`, `num_predict = 512`, and a 300-second timeout. These settings can be overridden in TOML. See [the measured local model results](docs/accuracy-results.md).
