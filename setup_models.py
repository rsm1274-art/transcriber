"""Run ONCE on a machine with internet, then copy the whole `models/` folder to the air-gapped machine.

    python setup_models.py --hf-token hf_xxx [--whisper distil-large-v3 medium]

Before running, accept the terms (free, logged in) on these Hugging Face pages:
    https://huggingface.co/pyannote/speaker-diarization-3.1
    https://huggingface.co/pyannote/segmentation-3.0
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).parent
MODELS = ROOT / "models"

WHISPER_REPOS = {
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "large-v3": "Systran/faster-whisper-large-v3",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "small": "Systran/faster-whisper-small",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-token", required=True)
    ap.add_argument("--whisper", nargs="+", default=["distil-large-v3"], choices=list(WHISPER_REPOS))
    ap.add_argument("--skip-pyannote", action="store_true")
    args = ap.parse_args()

    for name in args.whisper:
        print(f"Downloading whisper model {name} ...")
        snapshot_download(WHISPER_REPOS[name], local_dir=MODELS / "whisper" / name, token=args.hf_token)

    if not args.skip_pyannote:
        pya = MODELS / "pyannote"
        print("Downloading pyannote diarization models ...")
        cfg_dir = snapshot_download("pyannote/speaker-diarization-3.1", local_dir=pya / "_pipeline", token=args.hf_token)
        shutil.copy(Path(cfg_dir) / "config.yaml", pya / "config.yaml")
        snapshot_download("pyannote/segmentation-3.0", local_dir=pya / "segmentation", token=args.hf_token)
        snapshot_download("pyannote/wespeaker-voxceleb-resnet34-LM", local_dir=pya / "embedding", token=args.hf_token)

    print("\nDone. Also run on the online machine:  ollama pull llama3.2:3b")
    print("Then copy this project (including models/) and the Ollama model store to the air-gapped machine.")


if __name__ == "__main__":
    main()
