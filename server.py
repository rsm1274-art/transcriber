"""Air-gapped complaint-audio intake server.

Pipeline: faster-whisper (transcribe) -> pyannote (diarize) -> Ollama (summarize).
Serves the UI from ./static on 127.0.0.1 only. Makes no outbound connections
except to the local Ollama daemon.
"""
import gc
import json
import os
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

# Hard-disable any network/telemetry behaviour of the ML libraries.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
MODELS = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MAX_SPEAKER_GAP_S = 5.0

app = FastAPI(title="IA Audio Intake")
_job_lock = threading.Lock()  # one GPU job at a time (4 GB VRAM)


# ---------------------------------------------------------------- helpers

def cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def whisper_models():
    d = MODELS / "whisper"
    return sorted(p.name for p in d.glob("*") if (p / "model.bin").exists()) if d.exists() else []


def diarizer_ready() -> bool:
    p = MODELS / "pyannote"
    return all((p / f).exists() for f in ("config.yaml", "segmentation/pytorch_model.bin", "embedding/pytorch_model.bin"))


def ollama_json(path, body=None, timeout=600):
    req = urllib.request.Request(
        OLLAMA_URL + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ollama_models():
    try:
        return [m["name"] for m in ollama_json("/api/tags", timeout=3).get("models", [])]
    except Exception:
        return None


def ev(obj) -> bytes:
    return (json.dumps(obj) + "\n").encode()


# ---------------------------------------------------------------- transcription

def transcribe(audio, model_name, language):
    """Generator: yields progress events, returns (segments, device)."""
    from faster_whisper import WhisperModel

    path = str(MODELS / "whisper" / model_name)
    attempts = [("cuda", "int8_float16"), ("cpu", "int8")] if cuda_available() else [("cpu", "int8")]
    model, device, err = None, None, None
    for device, ctype in attempts:
        try:
            model = WhisperModel(path, device=device, compute_type=ctype)
            segs, info = model.transcribe(
                audio,
                language=language or None,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            out = []
            for s in segs:  # lazy generator: real work happens here
                out.append({"start": s.start, "end": s.end, "text": s.text.strip()})
                yield ev({"type": "progress", "message": "Transcribing (Whisper)...",
                          "pct": 10 + int(50 * min(s.end / max(info.duration, 1e-6), 1.0))})
            break
        except Exception as e:  # e.g. CUDA OOM / missing cuDNN -> retry on CPU
            err = e
            out = None
            model = None
            free_gpu()
    if out is None:
        raise RuntimeError(f"Whisper failed: {err}")
    del model
    free_gpu()
    return [s for s in out if s["text"]], device


# ---------------------------------------------------------------- diarization

def diarize(audio, num_speakers):
    import torch
    import yaml
    from pyannote.audio import Pipeline

    base = MODELS / "pyannote"
    cfg = yaml.safe_load((base / "config.yaml").read_text())
    params = cfg["pipeline"]["params"]
    params["segmentation"] = str(base / "segmentation" / "pytorch_model.bin")
    params["embedding"] = str(base / "embedding" / "pytorch_model.bin")
    runtime_cfg = base / "_runtime.yaml"
    runtime_cfg.write_text(yaml.safe_dump(cfg))

    pipe = Pipeline.from_pretrained(str(runtime_cfg))
    if torch.cuda.is_available():
        try:
            pipe.to(torch.device("cuda"))
        except Exception:
            pass
    wave = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000}
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    result = pipe(wave, **kwargs)
    annotation = getattr(result, "speaker_diarization", result)
    tracks = [(t.start, t.end, label) for t, _, label in annotation.itertracks(yield_label=True)]
    del pipe
    free_gpu()
    return tracks


def assign_speakers(segments, tracks):
    """Give each whisper segment the diarization speaker with the most overlap
    (nearest speaker turn if none overlaps). Returns segments with 'speaker'."""
    for seg in segments:
        overlap = {}
        for s, e, label in tracks:
            o = min(seg["end"], e) - max(seg["start"], s)
            if o > 0:
                overlap[label] = overlap.get(label, 0.0) + o
        if overlap:
            seg["speaker"] = max(overlap, key=overlap.get)
        elif tracks:
            mid = (seg["start"] + seg["end"]) / 2
            seg["speaker"] = min(tracks, key=lambda t: 0 if t[0] <= mid <= t[1] else min(abs(mid - t[0]), abs(mid - t[1])))[2]
        else:
            seg["speaker"] = "SPEAKER_00"
    return segments


def build_turns(segments):
    """Merge consecutive same-speaker segments; relabel speakers 'Speaker N' by first appearance."""
    names = {}
    turns = []
    for seg in segments:
        label = names.setdefault(seg["speaker"], f"Speaker {len(names) + 1}")
        if turns and turns[-1]["speaker"] == label and seg["start"] - turns[-1]["_end"] <= MAX_SPEAKER_GAP_S:
            turns[-1]["text"] += " " + seg["text"]
            turns[-1]["_end"] = seg["end"]
        else:
            turns.append({"time": int(seg["start"]), "speaker": label, "text": seg["text"], "_end": seg["end"]})
    for t in turns:
        del t["_end"]
    return turns, list(names.values())


# ---------------------------------------------------------------- endpoints

@app.get("/api/health")
def health():
    return {
        "cuda": cuda_available(),
        "whisperModels": whisper_models(),
        "diarizer": diarizer_ready(),
        "ollamaModels": ollama_models(),  # None => daemon unreachable
    }


@app.post("/api/process")
async def process(
    file: UploadFile = File(...),
    whisper_model: str = Form(...),
    language: str = Form(""),
    num_speakers: int = Form(0),
):
    suffix = Path(file.filename or "audio").suffix or ".bin"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)

    def gen():
        if not _job_lock.acquire(blocking=False):
            os.unlink(tmp)
            yield ev({"type": "error", "message": "Another file is already being processed."})
            return
        try:
            if whisper_model not in whisper_models():
                raise RuntimeError(f"Whisper model '{whisper_model}' is not installed in {MODELS / 'whisper'}.")
            warnings = []

            yield ev({"type": "progress", "message": "Decoding audio...", "pct": 5})
            from faster_whisper.audio import decode_audio
            audio = decode_audio(tmp, sampling_rate=16000)
            duration = len(audio) / 16000

            yield ev({"type": "progress", "message": "Transcribing (Whisper)...", "pct": 10})
            segments, device = yield from transcribe(audio, whisper_model, language)
            if not segments:
                raise RuntimeError("No speech detected in the recording.")

            if diarizer_ready():
                yield ev({"type": "progress", "message": "Identifying speakers (pyannote)...", "pct": 65})
                try:
                    tracks = diarize(audio, num_speakers or None)
                    segments = assign_speakers(segments, tracks)
                except Exception as e:
                    warnings.append(f"Speaker diarization failed ({e}); all speech is labelled as one speaker.")
                    segments = assign_speakers(segments, [])
            else:
                warnings.append("Diarization models not installed (models/pyannote); all speech is labelled as one speaker.")
                segments = assign_speakers(segments, [])

            turns, speakers = build_turns(segments)
            yield ev({"type": "result", "turns": turns, "speakers": speakers, "duration": duration,
                      "device": device, "warnings": warnings})
        except Exception as e:
            yield ev({"type": "error", "message": str(e)})
        finally:
            _job_lock.release()
            try:
                os.unlink(tmp)
            except OSError:
                pass
            free_gpu()

    return StreamingResponse(gen(), media_type="application/x-ndjson")


SUMMARY_PROMPT = """You are an Internal Affairs intake investigator. Below is a diarized transcript of a complaint intake call.
Use ONLY facts stated in the transcript. If something is not stated, write "Not stated". Return a single JSON object with exactly these keys:
- "narrativeSummary": chronological multi-paragraph summary written at a 10th-grade reading level (clear prose, no legal jargon, all material facts: what led to the encounter, officer conduct, scope of any search, items seized, threats made, intake outcome)
- "caseNumber": complaint/tracking number given in the call
- "complainantName": caller's full name
- "incidentTime": date/day/time of the incident
- "investigator": name/title of the person taking the complaint
- "address": address the caller gave
- "phone": phone number
- "allegations": array of {{"title": string, "category": string, "description": string}}
- "physicalEvidence": array of strings
- "followUpActions": array of strings

Transcript:
{transcript}"""

STR_KEYS = ["narrativeSummary", "caseNumber", "complainantName", "incidentTime", "investigator", "address", "phone"]


def fmt(sec):
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


def normalize_summary(d):
    if not isinstance(d, dict):
        raise ValueError("model did not return a JSON object")
    out = {k: str(d.get(k) or "Not stated") for k in STR_KEYS}
    out["allegations"] = [
        {"title": str(a.get("title", "")), "category": str(a.get("category", "")), "description": str(a.get("description", ""))}
        for a in (d.get("allegations") if isinstance(d.get("allegations"), list) else []) if isinstance(a, dict)
    ]
    for k in ("physicalEvidence", "followUpActions"):
        v = d.get(k) or []
        out[k] = [str(x) for x in (v if isinstance(v, list) else [v])]
    return out


class Turn(BaseModel):
    time: float
    speaker: str
    text: str


class SummarizeReq(BaseModel):
    turns: list[Turn]
    model: str


@app.post("/api/summarize")
def summarize(req: SummarizeReq):
    transcript = "\n".join(f"[{fmt(t.time)}] {t.speaker}: {t.text}" for t in req.turns)
    est_tokens = len(transcript) // 3
    num_ctx = max(4096, min(16384, est_tokens + 2500))
    warning = "Transcript is very long; the local model may have ignored its start." if est_tokens > 13000 else None
    body = {
        "model": req.model,
        "prompt": SUMMARY_PROMPT.format(transcript=transcript),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2, "num_ctx": num_ctx},
    }
    last = None
    for _ in range(2):
        try:
            data = ollama_json("/api/generate", body)
            return {"summary": normalize_summary(json.loads(data["response"])), "warning": warning}
        except urllib.error.URLError as e:
            return JSONResponse({"error": f"Ollama unreachable at {OLLAMA_URL}: {e.reason}"}, status_code=502)
        except (ValueError, KeyError) as e:  # bad JSON from the model: retry once
            last = e
        except Exception as e:
            return JSONResponse({"error": f"Ollama error: {e}"}, status_code=502)
    return JSONResponse({"error": f"Model returned invalid JSON twice ({last}). Try a larger model."}, status_code=502)


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
