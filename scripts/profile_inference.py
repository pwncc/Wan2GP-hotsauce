"""Run one headless WanGP inference with torch.profiler + phase timers.

Usage (from Wan2GP root):
  .\\venv\\Scripts\\python.exe scripts\\profile_inference.py
"""
from __future__ import annotations

import functools
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

SETTINGS_PATH = ROOT / "scripts" / "profile_perf_settings.json"
TRACE_PATH = ROOT / "outputs" / "profile_trace.json"
REPORT_PATH = ROOT / "outputs" / "profile_report.json"

os.environ["WAN2GP_LTX2_BENCH_TRANSFORMER"] = "1"
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
# StabilityMatrix base Python ships distutils outside the venv; setuptools must use stdlib.
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")

sys.argv = [
    "wgp.py",
    "--process",
    str(SETTINGS_PATH),
    "--profile",
    "4",
    "--preload",
    "0",
    "--disable-step-preview",
    "--verbose",
    "2",
]

import torch
from torch.profiler import ProfilerActivity, profile, record_function

PHASE_MS: dict[str, float] = {}
PHASE_ORDER: list[str] = []


def _mark_phase(name: str, start: float) -> None:
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    PHASE_MS[name] = elapsed_ms
    if name not in PHASE_ORDER:
        PHASE_ORDER.append(name)
    print(f"[profile] {name}: {elapsed_ms:.1f} ms")


def _load_wgp():
    spec = importlib.util.spec_from_file_location("wgp", ROOT / "wgp.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load wgp.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["wgp"] = module
    spec.loader.exec_module(module)
    return module


def _patch_mmgp_for_profiling() -> None:
    """mmgp profile() always calls model.to(cpu); mmap/quanto weights break swap_tensors."""
    from mmgp import offload as mmgp_offload

    if getattr(mmgp_offload, "_wgp_profile_patched", False):
        return

    original_load_model_data = mmgp_offload.load_model_data
    original_fast_load = mmgp_offload.fast_load_transformers_model
    original_profile = mmgp_offload.profile

    @functools.wraps(original_load_model_data)
    def load_model_data_for_profiling(model, file_path, *args, **kwargs):
        kwargs["writable_tensors"] = True
        return original_load_model_data(model, file_path, *args, **kwargs)

    @functools.wraps(original_fast_load)
    def fast_load_for_profiling(model_path, *args, **kwargs):
        kwargs["writable_tensors"] = True
        return original_fast_load(model_path, *args, **kwargs)

    @functools.wraps(original_profile)
    def profile_for_profiling(pipe_or_dict, *args, **kwargs):
        from shared.qtypes.gguf import materialize_module_source_tensors

        pipe = pipe_or_dict
        if isinstance(pipe, dict) and "pipe" in pipe:
            pipe = pipe["pipe"]
        if isinstance(pipe, dict):
            for model_id, model in pipe.items():
                if model is None:
                    continue
                converted = materialize_module_source_tensors(model)
                if converted:
                    print(f"[profile] materialized {converted} GGUF tensors in {model_id}")
        return original_profile(pipe_or_dict, *args, **kwargs)

    mmgp_offload.load_model_data = load_model_data_for_profiling
    mmgp_offload.fast_load_transformers_model = fast_load_for_profiling
    mmgp_offload.profile = profile_for_profiling
    mmgp_offload._wgp_profile_patched = True


def _patch_wgp(wgp):
    _patch_mmgp_for_profiling()

    # Respect offload profile 4: do not inherit 23GB preload from wgp_config.json.
    wgp.server_config["preload_in_VRAM"] = 0
    profile_no = float(wgp.args.profile)
    wgp.server_config["profile"] = profile_no
    wgp.server_config["video_profile"] = profile_no
    wgp.force_profile_no = profile_no
    # Quanto int8 text encoders fail model.to(cpu) via QLinearQuantoRouter weakref swap.
    wgp.text_encoder_quantization = ""
    wgp.server_config["text_encoder_quantization"] = ""

    original_load_models = wgp.load_models
    original_generate_video = wgp.generate_video

    profiler_holder: dict[str, object] = {"prof": None}

    @functools.wraps(original_load_models)
    def timed_load_models(*args, **kwargs):
        t0 = time.perf_counter()
        result = original_load_models(*args, **kwargs)
        _mark_phase("model_load", t0)
        return result

    @functools.wraps(original_generate_video)
    def timed_generate_video(*args, **kwargs):
        t0 = time.perf_counter()
        # Do not wrap load_models in torch.profiler: it leaves weakrefs on mmap/quanto
        # tensors and breaks mmgp offload.profile()'s model.to(cpu) swap_tensors path.
        result = original_generate_video(*args, **kwargs)
        _mark_phase("generate_video_total", t0)
        return result

    wgp.load_models = timed_load_models
    wgp.generate_video = timed_generate_video
    return profiler_holder


def _summarize_profiler(prof) -> dict:
    if prof is None:
        return {}

    key_averages = prof.key_averages(group_by_stack_n=5)
    cuda_rows = []
    cpu_rows = []

    for evt in key_averages:
        row = {
            "name": evt.key,
            "cuda_time_us": round(getattr(evt, "cuda_time_total", 0) or 0, 1),
            "cpu_time_us": round(getattr(evt, "cpu_time_total", 0) or 0, 1),
            "count": evt.count,
        }
        if row["cuda_time_us"] > 0:
            cuda_rows.append(row)
        elif row["cpu_time_us"] > 0:
            cpu_rows.append(row)

    cuda_rows.sort(key=lambda r: r["cuda_time_us"], reverse=True)
    cpu_rows.sort(key=lambda r: r["cpu_time_us"], reverse=True)

    return {
        "top_cuda_ops": cuda_rows[:25],
        "top_cpu_ops": cpu_rows[:25],
    }


def _run_cli(wgp) -> int:
    args = wgp.args
    state = {
        "gen": {
            "queue": [],
            "in_progress": False,
            "file_list": [],
            "file_settings_list": [],
            "audio_file_list": [],
            "audio_file_settings_list": [],
            "selected": 0,
            "audio_selected": 0,
            "prompt_no": 0,
            "prompts_max": 0,
            "repeat_no": 0,
            "total_generation": 1,
            "window_no": 0,
            "total_windows": 0,
            "progress_status": "",
            "process_status": "process:main",
        },
        "loras": [],
    }

    queue, error = wgp._parse_settings_json(str(SETTINGS_PATH), state)
    if error:
        print(f"[ERROR] {error}")
        return 1
    if not queue:
        print("[ERROR] Empty queue")
        return 1

    validated, validation_error = wgp.validate_task(queue[0], state)
    if validated is None:
        print(f"[ERROR] Validation failed: {validation_error}")
        return 1

    state["gen"]["queue"] = queue
    t0 = time.perf_counter()
    ok = wgp.process_tasks_cli(queue, state)
    _mark_phase("queue_total", t0)
    return 0 if ok else 1


def main() -> int:
    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    from shared.ffmpeg_setup import download_ffmpeg

    download_ffmpeg()
    _patch_mmgp_for_profiling()

    print("[profile] Loading WanGP (this imports models and reads config)...")
    t_import = time.perf_counter()
    wgp = _load_wgp()
    _mark_phase("wgp_import", t_import)

    profiler_holder = _patch_wgp(wgp)

    print(f"[profile] Starting headless inference (profile {wgp.force_profile_no}, no preload, no keep-models-hot)")
    print(f"[profile] Settings: {SETTINGS_PATH}")
    try:
        rc = _run_cli(wgp)
    except Exception:
        traceback.print_exc()
        return 1

    prof = profiler_holder.get("prof")
    if prof is not None:
        try:
            prof.export_chrome_trace(str(TRACE_PATH))
            print(f"[profile] Chrome trace: {TRACE_PATH}")
        except Exception as exc:
            print(f"[profile] Failed to export chrome trace: {exc}")

    report = {
        "phases_ms": {name: round(PHASE_MS[name], 2) for name in PHASE_ORDER},
        "profiler": _summarize_profiler(prof),
        "settings": json.loads(SETTINGS_PATH.read_text(encoding="utf-8")),
        "cli": {
            "profile": float(wgp.force_profile_no),
            "preload_mb": 0,
            "keep_models_hot": bool(wgp.args.keep_models_hot),
            "disable_step_preview": True,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[profile] Report: {REPORT_PATH}")

    if PHASE_MS:
        print("\n[profile] Phase breakdown:")
        total = PHASE_MS.get("queue_total") or sum(PHASE_MS.values())
        for name in PHASE_ORDER:
            ms = PHASE_MS[name]
            pct = (ms / total * 100.0) if total else 0.0
            print(f"  {name:24s} {ms:10.1f} ms  ({pct:5.1f}%)")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
