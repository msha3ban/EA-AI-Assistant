# All AI processing runs on infrastructure the user controls

Meetings can contain sensitive enterprise information, so Recordings, Transcripts and every generated document are processed only by models running locally (speech-to-text and LLM). No cloud AI APIs are used, even where they would be more accurate. The model runtime is reached through a configurable endpoint so that it can move from the user's laptop (GTX 1060, 6 GB VRAM) to an on-prem company server later without design changes.
