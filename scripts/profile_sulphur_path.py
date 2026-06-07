"""Instrument Sulphur inference sub-phases through the real API path.

Usage:
  .\\venv\\Scripts\\python.exe scripts\\profile_sulphur_path.py
  .\\venv\\Scripts\\python.exe scripts\\profile_sulphur_path.py --warmup
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")

SETTINGS_PATH = ROOT / "scripts" / "profile_perf_settings.json"
REPORT_PATH = ROOT / "outputs" / "profile_sulphur_path_report.json"

_timings: list[dict] = []
_stack: list[tuple[str, float]] = []


@contextmanager
def _span(name: str):
    t0 = time.perf_counter()
    _stack.append((name, t0))
    try:
        yield
    finally:
        start_name, start_t = _stack.pop()
        assert start_name == name
        ms = round((time.perf_counter() - start_t) * 1000, 1)
        _timings.append({"phase": name, "ms": ms})
        print(f"[profile] {name}: {ms:.1f} ms")


def _install_patches() -> None:
    import shared.utils.audio_video as av
    import shared.utils.hot_models as hot_models
    from models.ltx2.ltx_core.model import audio_vae as ltx_audio_vae
    from models.ltx2.ltx_core.model import video_vae as ltx_video_vae

    from models.ltx2.ltx_core.text_encoders.gemma.encoders import base_encoder as gemma_encoder

    orig_decode_video = ltx_video_vae.decode_video_to_tensor
    orig_decode_audio = ltx_audio_vae.decode_audio
    orig_encode_text = gemma_encoder.encode_text
    orig_save_video = av.save_video
    orig_combine = av.combine_and_concatenate_video_with_audio_tracks
    orig_unload = hot_models.unload_all_unless_hot

    def encode_text(*args, **kwargs):
        with _span("text_encode_gemma"):
            return orig_encode_text(*args, **kwargs)

    def decode_video_to_tensor(*args, **kwargs):
        with _span("vae_decode_video"):
            return orig_decode_video(*args, **kwargs)

    def decode_audio(*args, **kwargs):
        with _span("vae_decode_audio"):
            return orig_decode_audio(*args, **kwargs)

    def save_video(*args, **kwargs):
        with _span("save_video_encode"):
            return orig_save_video(*args, **kwargs)

    def combine_and_concatenate_video_with_audio_tracks(*args, **kwargs):
        with _span("ffmpeg_mux"):
            return orig_combine(*args, **kwargs)

    def unload_all_unless_hot(manager):
        with _span("mmgp_unload_all"):
            return orig_unload(manager)

    gemma_encoder.encode_text = encode_text
    ltx_video_vae.decode_video_to_tensor = decode_video_to_tensor
    ltx_audio_vae.decode_audio = decode_audio
    av.save_video = save_video
    av.combine_and_concatenate_video_with_audio_tracks = combine_and_concatenate_video_with_audio_tracks
    hot_models.unload_all_unless_hot = unload_all_unless_hot


def _run_once(*, label: str) -> dict:
    from shared.api import init

    global _timings
    _timings = []
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    settings["client_id"] = f"profile_{label}_{int(time.time())}"

    _install_patches()
    with _span(f"total_{label}"):
        session = init(root=ROOT, console_output=False)
        with _span("api_submit_to_result"):
            job = session.submit_task(settings)
            result = job.result()

    report = {
        "label": label,
        "success": result.success,
        "generated_files": result.generated_files,
        "timings": _timings,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", action="store_true", help="Run twice; second run is warm (models resident)")
    args = parser.parse_args()

    runs = []
    runs.append(_run_once(label="cold"))
    if args.warmup:
        runs.append(_run_once(label="warm"))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
    print(f"[profile] Report: {REPORT_PATH}")
    for run in runs:
        total = next((t["ms"] for t in run["timings"] if t["phase"] == f"total_{run['label']}"), 0)
        print(f"[profile] {run['label']}: {total:.1f} ms success={run['success']}")
        for entry in run["timings"]:
            if entry["phase"].startswith("total_"):
                continue
            print(f"  - {entry['phase']}: {entry['ms']:.1f} ms")
    return 0 if all(r["success"] for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
