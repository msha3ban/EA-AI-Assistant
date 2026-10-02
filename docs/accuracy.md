# Local model accuracy suite

The opt-in `ea accuracy suite.toml` command compares combinations of local STT and LLM settings on a user's hand-corrected reference Recording. Pick a representative excerpt of about five minutes, hand-correct its Transcript verbatim in the Egyptian Arabic and English actually spoken, write down the critical facts that must survive into the MoM, and run the suite. Put the suite, Recording, reference Transcript, Vocabulary, and facts outside this repository. Example files in `docs/accuracy-example/` contain invented content only.

The report includes raw and normalised word error rate (WER) and character error rate (CER), Vocabulary term recognition/canonical spelling, number precision and recall, resource use (minutes per audio hour, VRAM and RAM), Flagged Passage detector recall/precision and flags per audio hour, and MoM critical-fact found/wrong/missing counts plus invented candidates for human review. “Invented” is a candidate list for human review, not an automatic judgement.

Use `ea accuracy /absolute/path/to/suite.toml --config /absolute/path/to/app-config.toml --out /absolute/path/to/results` to choose an external results directory. Omit `--config` to use the normal application config; omit `--out` to use the suite's `output_dir` (default `results` beside the suite). Repeat configurations with multiple `[[run]]` entries, or pass `--only name-a name-b` to run a subset. A Vocabulary `initial_prompt` is conservatively kept at no more than 224 UTF-8 bytes, a safe upper bound for Whisper's 224-token prompt limit; the report records whether terms were truncated.

Ticket #1 should compare MoM critical-fact accuracy and Flagged Passage recall alongside WER, rather than selecting a model on WER alone. A low WER can still omit or reverse an important Decision, while a higher WER may preserve all critical facts. Review the artifact files and invented candidates before choosing settings. `direct` runs extract and render from the corrected mixed-language reference Transcript; `two-step` also runs speech-to-text and English Transcript generation.

Reference files stay outside the repository. Results are written beneath the configured output directory; the CLI refuses output paths inside this repository. For the checked-in invented example only, pass `--allow-suite-in-repo` and set `--out` to a directory outside the checkout.

See [the published scoring normalisation rules](accuracy-normalisation.md) for the exact Arabic, punctuation, alias, and number handling.
