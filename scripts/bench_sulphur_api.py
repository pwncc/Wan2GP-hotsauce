"""Benchmark Sulphur Q4_K_M through WanGP's real inference paths.

Runs both:
  - api:    shared.api.init() -> generate_video (headless API worker)
  - gradio: live wgp.py server -> Load Queue -> process_tasks chain

Usage (from Wan2GP root):
  .\\venv\\Scripts\\python.exe scripts\\bench_sulphur_api.py
  .\\venv\\Scripts\\python.exe scripts\\bench_sulphur_api.py --mode gradio
  .\\venv\\Scripts\\python.exe scripts\\bench_sulphur_api.py --mode api
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")

SETTINGS_PATH = ROOT / "scripts" / "profile_perf_settings.json"
REPORT_PATH = ROOT / "outputs" / "bench_sulphur_api_report.json"
DEFAULT_GRADIO_PORT = int(os.environ.get("WAN2GP_BENCH_PORT", "7861"))
GRADIO_LOG_PATH = ROOT / "outputs" / "bench_gradio_server.log"


class BenchCallbacks:
    def __init__(self, prefix: str = "bench") -> None:
        self.prefix = prefix
        self.t0 = time.perf_counter()
        self.phases: list[dict] = []
        self._last_phase = ""
        self._last_phase_t = self.t0

    def _mark(self, label: str) -> None:
        now = time.perf_counter()
        self.phases.append({"phase": label, "ms": round((now - self._last_phase_t) * 1000, 1)})
        self._last_phase = label
        self._last_phase_t = now
        print(f"[{self.prefix}] {label}: {self.phases[-1]['ms']:.1f} ms")

    def on_status(self, status) -> None:
        status = str(status or "").strip()
        if status:
            print(f"[{self.prefix}|status] {status}")

    def on_progress(self, update) -> None:
        phase = str(getattr(update, "phase", "") or getattr(update, "status", "") or "").strip()
        if phase and phase != self._last_phase:
            self._mark(phase)
        step = getattr(update, "current_step", None)
        total = getattr(update, "total_steps", None)
        progress = getattr(update, "progress", None)
        if step is not None and total is not None:
            print(f"[{self.prefix}|progress] {phase or self._last_phase} step {step}/{total} ({progress}%)")


def _load_settings() -> dict:
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    settings.setdefault("client_id", f"bench_sulphur_{int(time.time())}")
    return settings


def _output_dirs() -> list[Path]:
    dirs = [ROOT / "outputs"]
    try:
        cfg = json.loads((ROOT / "wgp_config.json").read_text(encoding="utf-8"))
        for key in ("save_path", "image_save_path", "audio_save_path"):
            value = str(cfg.get(key, "") or "").strip()
            if value:
                path = Path(value)
                if not path.is_absolute():
                    path = (ROOT / path).resolve()
                dirs.append(path)
    except Exception:
        pass
    return list(dict.fromkeys(dirs))


def _snapshot_outputs() -> set[Path]:
    files: set[Path] = set()
    for directory in _output_dirs():
        if directory.is_dir():
            files.update(directory.glob("*.mp4"))
    return files


def _wait_for_new_output(before: set[Path], timeout_s: float = 900.0) -> Path | None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        now = set()
        for directory in _output_dirs():
            if directory.is_dir():
                now.update(directory.glob("*.mp4"))
        added = now - before
        if added:
            return max(added, key=lambda path: path.stat().st_mtime)
        time.sleep(2.0)
    return None


def _tail_log(path: Path, offset: int, last_printed: list[float]) -> int:
    if not path.is_file():
        return offset
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        chunk = handle.read()
        if chunk:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            offset = handle.tell()
            last_printed[0] = time.perf_counter()
    return offset


def _parse_log_phases(log_text: str) -> list[dict]:
    phases: list[dict] = []
    markers = [
        ("model_load", re.compile(r"Loading model sulfur2Base|Hooked to model 'transformer'", re.I)),
        ("encoding_text", re.compile(r"Encoding Prompt", re.I)),
        ("inference", re.compile(r"^\s*\d+%\|", re.M)),
        ("decoding", re.compile(r"VAE Decoding", re.I)),
        ("saved", re.compile(r"New video saved to Path:", re.I)),
    ]
    lines = log_text.splitlines()
    last_t = None
    for line in lines:
        if "load_settings" in line or "Loading Model '" in line:
            if not any(p["phase"] == "model_load_start" for p in phases):
                phases.append({"phase": "model_load_start", "line": line.strip()})
        for name, pattern in markers:
            if pattern.search(line):
                phases.append({"phase": name, "line": line.strip()})
    step_times = re.findall(r"(\d+\.\d+)s/steps", log_text)
    if step_times:
        phases.append({"phase": "denoise_sec_per_step", "values": [float(x) for x in step_times[-3:]]})
    return phases


def _run_standalone_api(settings: dict):
    from shared.api import init

    print("[api] In-process WanGP API (wgp_config.json, generate_video worker)")
    session = init(root=ROOT, console_output=True)
    t0 = time.perf_counter()
    callbacks = BenchCallbacks("api")
    job = session.submit_task(settings, callbacks=callbacks)
    result = job.result()
    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "mode": "api",
        "success": result.success,
        "total_ms": total_ms,
        "phases": callbacks.phases,
        "generated_files": result.generated_files,
        "errors": [str(e) for e in result.errors],
    }


def _wait_for_server(url: str, timeout_s: float = 180.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2.0)
    raise RuntimeError(f"Gradio server did not become ready at {url}")


def _start_gradio_server(port: int) -> subprocess.Popen:
    GRADIO_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_handle = GRADIO_LOG_PATH.open("w", encoding="utf-8")
    cmd = [
        str(ROOT / "venv" / "Scripts" / "python.exe"),
        str(ROOT / "wgp.py"),
        "--server-port",
        str(port),
    ]
    env = os.environ.copy()
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    env["PYTHONUNBUFFERED"] = "1"
    print(f"[gradio] Starting wgp.py on port {port}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )
    _wait_for_server(f"http://127.0.0.1:{port}")
    print(f"[gradio] Server ready (pid={proc.pid}, log={GRADIO_LOG_PATH})")
    return proc


def _stop_gradio_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    print(f"[gradio] Stopping server pid={proc.pid}")
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def _run_gradio_api(settings: dict, port: int, *, manage_server: bool) -> dict:
    from gradio_client import Client, handle_file

    proc = None
    url = f"http://127.0.0.1:{port}"
    if manage_server:
        proc = _start_gradio_server(port)
    else:
        _wait_for_server(url)

    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    before = _snapshot_outputs()
    log_start = GRADIO_LOG_PATH.stat().st_size if GRADIO_LOG_PATH.is_file() else 0
    log_offset = log_start
    t0 = time.perf_counter()

    try:
        client = Client(url)
        print("[gradio] 1/2 load queue JSON via /load_queue_action")
        queue_html = client.predict(handle_file(str(SETTINGS_PATH)), api_name="/load_queue_action")
        print(f"[gradio] queue html: {str(queue_html)[:160]}")

        print("[gradio] 2/2 start generation via /process_tasks_1 (same Gradio session)")
        print(f"[gradio] streaming server log: {GRADIO_LOG_PATH}")
        job = client.submit(api_name="/process_tasks_1")
        last_printed = [time.perf_counter()]
        output_path = None
        deadline = time.perf_counter() + 900.0
        while time.perf_counter() < deadline:
            log_offset = _tail_log(GRADIO_LOG_PATH, log_offset, last_printed)
            if job.done():
                try:
                    job.result()
                except Exception as exc:
                    print(f"[gradio] process_tasks finished with: {exc}")
                break
            added = _snapshot_outputs() - before
            if added:
                output_path = max(added, key=lambda path: path.stat().st_mtime)
                break
            if time.perf_counter() - last_printed[0] > 10:
                print(f"[gradio] still running... {int(time.perf_counter() - t0)}s elapsed")
                last_printed[0] = time.perf_counter()
            time.sleep(1.0)

        if output_path is None:
            output_path = _wait_for_new_output(before, timeout_s=30.0)
        log_offset = _tail_log(GRADIO_LOG_PATH, log_offset, last_printed)
        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        log_tail = ""
        if GRADIO_LOG_PATH.is_file():
            with GRADIO_LOG_PATH.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(log_start)
                log_tail = handle.read()
        return {
            "mode": "gradio",
            "success": output_path is not None,
            "total_ms": total_ms,
            "url": url,
            "generated_files": [str(output_path)] if output_path else [],
            "log_phases": _parse_log_phases(log_tail),
            "errors": [] if output_path else ["Timed out waiting for Gradio queue output"],
        }
    finally:
        if manage_server:
            _stop_gradio_server(proc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Sulphur through WanGP API and Gradio queue")
    parser.add_argument("--mode", choices=("all", "api", "gradio"), default="all")
    parser.add_argument("--port", type=int, default=DEFAULT_GRADIO_PORT)
    parser.add_argument("--keep-server", action="store_true", help="Leave Gradio running after gradio mode")
    args = parser.parse_args()

    settings = _load_settings()
    print(
        f"[bench] model={settings.get('model_type')} resolution={settings.get('resolution')} "
        f"frames={settings.get('video_length')} steps={settings.get('num_inference_steps')}"
    )

    report: dict = {"settings": settings, "runs": []}
    exit_code = 0

    if args.mode in ("all", "api"):
        api_result = _run_standalone_api(settings)
        report["runs"].append(api_result)
        if not api_result.get("success"):
            exit_code = 1

    if args.mode in ("all", "gradio"):
        gradio_result = _run_gradio_api(
            settings,
            args.port,
            manage_server=not args.keep_server,
        )
        report["runs"].append(gradio_result)
        if not gradio_result.get("success"):
            exit_code = 1

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[bench] Report: {REPORT_PATH}")
    for run in report["runs"]:
        print(
            f"[bench] {run['mode']}: {run.get('total_ms', 0):.1f} ms "
            f"success={run.get('success')} files={run.get('generated_files', [])}"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
