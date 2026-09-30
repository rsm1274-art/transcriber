# Air-gapped complaint audio intake

faster-whisper (transcribe) → pyannote (speaker diarization) → Ollama (10th-grade narrative + allegations).
Server binds to 127.0.0.1 only; models load offline (`HF_HUB_OFFLINE=1`). The only connection it makes is to local Ollama.

## One-time setup (internet-connected machine)
1. `pip install -r requirements.txt` (for GPU: install CUDA torch from pytorch.org, plus `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` for faster-whisper).
2. Accept the terms (free HF account) at huggingface.co/pyannote/speaker-diarization-3.1 and /segmentation-3.0, create a read token.
3. `python setup_models.py --hf-token hf_xxx --whisper distil-large-v3 medium`
4. `ollama pull llama3.2:3b` (fits 4 GB VRAM; `qwen2.5:3b` is an alternative; 7-8B models spill to CPU/RAM and are slow).
5. Copy the project (incl. `models/`) and the Ollama model store (`~/.ollama/models`) to the air-gapped machine; install the same pip packages there (e.g. from a wheel cache: `pip download -r requirements.txt -d wheels`).

## Run
```
python server.py      # then open http://127.0.0.1:8000  (not the HTML file directly)
```
Processing is sequential so each model is freed from VRAM before the next loads (Whisper → pyannote → Ollama). On GPU failure/OOM Whisper retries on CPU.

## Use
Attach audio → **Transcribe & Diarize**. Check the **Speaker Roles** panel in the Transcript tab (first speaker defaults to the intake officer) and click **Regenerate Summary** if you change roles or names.

## Notes / known limits
- `distil-large-v3` is English-only; use `medium` or `large-v3` for other languages.
- If pyannote fails with a `weights_only` / `torch.load` error, use `torch==2.5.1`.
- Without pyannote models, everything is labelled one speaker (a warning is shown); no guessing from punctuation.
- Small local LLMs can miss details or return bad JSON (one automatic retry, then an explicit error; the transcript is still shown). Review all output against the audio.
