from __future__ import annotations

import atexit
import json
import os
import shlex
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import modal


APP_NAME = "wan2gp"
REPO_ROOT = Path(__file__).resolve().parent
REMOTE_APP_ROOT = "/root/wan2gp"

MODELS_MOUNT = "/vol/models"
STATE_MOUNT = "/vol/state"
OUTPUTS_MOUNT = "/vol/outputs"

MODELS_VOLUME_NAME = "wan2gp-models"
STATE_VOLUME_NAME = "wan2gp-state"
OUTPUTS_VOLUME_NAME = "wan2gp-outputs"

WAN_PORT = 7860
GPU_ENV = "H200"
CPU_COUNT = 12.0
IDLE_SECONDS = 900
FUNCTION_TIMEOUT = 60 * 60 * 24
STARTUP_TIMEOUT = 3600
UI_READY_TIMEOUT = STARTUP_TIMEOUT
UI_RESTART_DELAY = 5
UI_CONNECT_TIMEOUT = 2
CUDA_FLAVOR = "auto"
INSTALL_SAGE_DEFAULT = True
INSTALL_FLASH_ATTN_DEFAULT = True
INSTALL_GGUF_CUDA_KERNELS_DEFAULT = True
BUILD_SAGE_FROM_SOURCE = True
BUILD_JOBS = "16"
NVCC_THREADS = "4"
FLASH_ATTN_WHEEL_CU130 = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3%2Bcu130torch2.10-cp311-cp311-linux_x86_64.whl"
FLASH_ATTN_WHEEL_CU128 = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3%2Bcu128torch2.10-cp311-cp311-linux_x86_64.whl"
GGUF_CUDA_WHEEL_CU130_TORCH210 = "https://github.com/deepbeepmeep/kernels/releases/download/GGUF_Kernels/llamacpp_gguf_cuda-1.0.2%2Btorch210cu13py311-cp311-cp311-linux_x86_64.whl"
SAGEATTENTION_SOURCE = "https://github.com/thu-ml/SageAttention.git"
SAGEATTENTION_WHEEL_URL_OVERRIDE = ""
DEFAULT_WGP_ARGS = [
    "--profile",
    "3",
    "--vram-safety-coefficient",
    "0.99",
    "--perc-reserved-mem-max",
    "0",
    "--keep-models-hot",
    "--disable-step-preview",
]
UI_EXTRA_ARGS = ""
BUILD_ENV_PREFIX = (
    f"MAX_JOBS={BUILD_JOBS} "
    f"NVCC_THREADS={NVCC_THREADS} "
    f"CMAKE_BUILD_PARALLEL_LEVEL={BUILD_JOBS} "
    f"EXT_PARALLEL={BUILD_JOBS} "
    f"PARALLEL_LEVEL={BUILD_JOBS} "
    f"NINJA_NUM_JOBS={BUILD_JOBS} "
    f"MAKEFLAGS=-j{BUILD_JOBS} "
    f"NINJAFLAGS=-j{BUILD_JOBS} "
    "NINJA_STATUS='[%f/%t %es] ' "
    "FORCE_CUDA=1 "
)

UPLOAD_IGNORE = [
    ".git",
    ".git/**",
    ".venv",
    ".venv/**",
    "venv",
    "venv/**",
    "ckpts",
    "ckpts/**",
    "loras",
    "loras/**",
    "outputs",
    "outputs/**",
    "settings",
    "settings/**",
    "finetunes",
    "finetunes/**",
    "__pycache__",
    "__pycache__/**",
    "*.pyc",
]


def _parse_gpu_spec(raw: str):
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    return values if len(values) > 1 else (values[0] if values else "H100")


def _normalize_cuda_flavor(raw: str) -> str:
    return str(raw or "auto").strip().lower()


def _gpu_spec_text(gpu_spec) -> str:
    return ",".join(gpu_spec) if isinstance(gpu_spec, list) else str(gpu_spec)


def _use_cuda13(gpu_spec) -> bool:
    cuda_flavor = _normalize_cuda_flavor(CUDA_FLAVOR)
    flat = _gpu_spec_text(gpu_spec)
    if cuda_flavor in {"13", "cu130", "cuda13"}:
        return True
    if cuda_flavor in {"12.8", "128", "cu128", "cuda12", "cuda12.8"}:
        return False
    if cuda_flavor not in {"", "auto"}:
        raise ValueError(f"Unsupported CUDA_FLAVOR: {CUDA_FLAVOR!r}")
    flat_upper = flat.upper()
    cuda13_gpus = ("H100", "H200", "B200", "B300", "GB200", "GB300")
    return any(gpu in flat_upper for gpu in cuda13_gpus)


def _cuda_wheel_tag(use_cuda13: bool) -> str:
    return "cu130" if use_cuda13 else "cu128"


def _torch_install_command(use_cuda13: bool) -> str:
    if use_cuda13:
        return (
            "python -m pip install torch==2.10.0+cu130 torchvision==0.25.0+cu130 "
            "torchaudio==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130"
        )
    return (
        "python -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 "
        "torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128"
    )


def _cuda_image(use_cuda13: bool) -> str:
    return "nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04" if use_cuda13 else "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04"


def _flash_attn_wheel_url(use_cuda13: bool) -> str:
    return FLASH_ATTN_WHEEL_CU130 if use_cuda13 else FLASH_ATTN_WHEEL_CU128


def _default_cuda_arch_list(gpu_spec) -> str:
    flat = ",".join(gpu_spec) if isinstance(gpu_spec, list) else str(gpu_spec)
    flat_upper = flat.upper()
    if "A100" in flat_upper or "A40" in flat_upper:
        return "8.0"
    if "A10" in flat_upper or "A6000" in flat_upper or "A5000" in flat_upper or "A4000" in flat_upper:
        return "8.6"
    if "L4" in flat_upper or "L40" in flat_upper or "L40S" in flat_upper:
        return "8.9"
    if "H100" in flat_upper or "H200" in flat_upper:
        return "9.0"
    if "B300" in flat_upper or "GB300" in flat_upper:
        return "10.3"
    if "B200" in flat_upper or "GB200" in flat_upper:
        return "10.0"
    return "8.0;8.6;8.9;9.0;10.0"


GPU_TYPE = _parse_gpu_spec(GPU_ENV)
USE_CUDA13 = _use_cuda13(GPU_TYPE)
CUDA_WHEEL_TAG = _cuda_wheel_tag(USE_CUDA13)
INSTALL_SAGE = INSTALL_SAGE_DEFAULT
INSTALL_FLASH_ATTN = (not USE_CUDA13) if INSTALL_FLASH_ATTN_DEFAULT is None else INSTALL_FLASH_ATTN_DEFAULT
INSTALL_GGUF_CUDA_KERNELS = USE_CUDA13 and INSTALL_GGUF_CUDA_KERNELS_DEFAULT
RUNTIME_THREADS = str(int(CPU_COUNT))


DEFAULT_CUDA_ARCH_LIST = _default_cuda_arch_list(GPU_TYPE)
CUDA_ARCH_LIST = DEFAULT_CUDA_ARCH_LIST


def _default_sageattention_wheel_url(cuda_arch_list: str) -> str:
    arch_values = {
        value.strip()
        for value in cuda_arch_list.replace(";", ",").replace(" ", ",").split(",")
        if value.strip()
    }
    if arch_values & {"10", "10.0", "10.3", "sm100", "sm103"}:
        return "https://huggingface.co/UmeAiRT/ComfyUI-Auto-Installer-Assets/resolve/main/whl/sm100/sageattention-2.2.0-cp311-cp311-linux_x86_64.whl"
    return "https://huggingface.co/UmeAiRT/ComfyUI-Auto-Installer-Assets/resolve/main/whl/sm90/sageattention-2.2.0-cp311-cp311-linux_x86_64.whl"


SAGEATTENTION_WHEEL_URL = SAGEATTENTION_WHEEL_URL_OVERRIDE or _default_sageattention_wheel_url(CUDA_ARCH_LIST)

CUDA_IMAGE = _cuda_image(USE_CUDA13)
TORCH_INSTALL = _torch_install_command(USE_CUDA13)

models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)
state_volume = modal.Volume.from_name(STATE_VOLUME_NAME, create_if_missing=True)
outputs_volume = modal.Volume.from_name(OUTPUTS_VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME)

image_builder = (
    modal.Image.from_registry(
        CUDA_IMAGE,
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "wget",
        "curl",
        "build-essential",
        "clang",
        "cmake",
        "ninja-build",
        "pkg-config",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PIP_PROGRESS_BAR": "off",
            "TORCH_CUDA_ARCH_LIST": CUDA_ARCH_LIST,
            "TORCH_ALLOW_TF32_CUBLAS": "1",
            "TORCH_ALLOW_TF32_CUDNN": "1",
            "WGP_GGUF_LLAMACPP_CUDA": "1",
            "HF_HOME": f"{MODELS_MOUNT}/huggingface",
            "TORCH_HOME": f"{MODELS_MOUNT}/torch",
            "XDG_CACHE_HOME": f"{MODELS_MOUNT}/cache",
            "GRADIO_SERVER_NAME": "0.0.0.0",
            "GRADIO_SERVER_PORT": str(WAN_PORT),
        }
    )
    .workdir(REMOTE_APP_ROOT)
    .run_commands("mkdir -p /tmp/wangp-build")
    .add_local_file(REPO_ROOT / "requirements.txt", "/tmp/wangp-build/requirements.txt", copy=True)
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        TORCH_INSTALL,
        "python -m pip install -r /tmp/wangp-build/requirements.txt",
        TORCH_INSTALL,
        "python -m pip install -U triton",
        "python -c \"import torch, torchaudio, torchvision; print('torch stack', torch.__version__, torchaudio.__version__, torchvision.__version__)\"",
    )
)

if INSTALL_FLASH_ATTN:
    image_builder = image_builder.run_commands(
        f'python -m pip install "{_flash_attn_wheel_url(USE_CUDA13)}"',
    )

if INSTALL_GGUF_CUDA_KERNELS:
    image_builder = image_builder.run_commands(
        f'python -m pip install "{GGUF_CUDA_WHEEL_CU130_TORCH210}"',
        "python -c \"import llamacpp_gguf_cuda; print('gguf cuda kernels', llamacpp_gguf_cuda.__file__)\"",
    )

if INSTALL_SAGE:
    if BUILD_SAGE_FROM_SOURCE:
        image_builder = image_builder.run_commands(
            "python -m pip install \"setuptools<=75.8.2\"",
            "rm -rf /tmp/sageattention",
            f'git clone --depth 1 "{SAGEATTENTION_SOURCE}" /tmp/sageattention',
            f"{BUILD_ENV_PREFIX} python -m pip install --no-build-isolation --verbose /tmp/sageattention",
        )
    else:
        image_builder = image_builder.run_commands(
            f'python -m pip install --no-deps "{SAGEATTENTION_WHEEL_URL}"',
        )

image_builder = image_builder.add_local_dir(REPO_ROOT, remote_path=REMOTE_APP_ROOT, ignore=UPLOAD_IGNORE)

image = image_builder


def _copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _ensure_runtime_layout() -> Path:
    models_root = Path(MODELS_MOUNT)
    state_root = Path(STATE_MOUNT)
    outputs_root = Path(OUTPUTS_MOUNT)
    repo_root = Path(REMOTE_APP_ROOT)

    for path in [
        models_root / "ckpts",
        models_root / "loras",
        models_root / "huggingface",
        models_root / "torch",
        models_root / "cache",
        state_root / "torchinductor",
        state_root / "triton",
        state_root / "config",
        state_root / "settings",
        state_root / "finetunes",
        outputs_root,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    repo_settings = repo_root / "settings"
    persisted_settings = state_root / "settings"
    if repo_settings.exists() and not any(persisted_settings.iterdir()):
        _copy_tree_contents(repo_settings, persisted_settings)

    repo_finetunes = repo_root / "finetunes"
    persisted_finetunes = state_root / "finetunes"
    if repo_finetunes.exists() and not any(persisted_finetunes.iterdir()):
        _copy_tree_contents(repo_finetunes, persisted_finetunes)

    if repo_finetunes.exists() and not repo_finetunes.is_symlink():
        shutil.rmtree(repo_finetunes)
    if not repo_finetunes.exists():
        repo_finetunes.symlink_to(persisted_finetunes, target_is_directory=True)

    config_path = state_root / "config" / "wgp_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as infile:
            config = json.load(infile)
    else:
        repo_config_path = repo_root / "wgp_config.json"
        if repo_config_path.exists():
            with repo_config_path.open("r", encoding="utf-8") as infile:
                config = json.load(infile)
        else:
            config = {}

    config["checkpoints_paths"] = [str(models_root / "ckpts"), REMOTE_APP_ROOT]
    config["loras_root"] = str(models_root / "loras")
    config["save_path"] = str(outputs_root)
    config["image_save_path"] = str(outputs_root)
    config["audio_save_path"] = str(outputs_root)

    with config_path.open("w", encoding="utf-8") as outfile:
        json.dump(config, outfile, indent=4)

    state_volume.commit()
    outputs_volume.commit()
    models_volume.commit()
    return config_path


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=CPU_COUNT,
    timeout=FUNCTION_TIMEOUT,
    startup_timeout=STARTUP_TIMEOUT,
    volumes={
        MODELS_MOUNT: models_volume,
        STATE_MOUNT: state_volume,
        OUTPUTS_MOUNT: outputs_volume,
    },
    max_containers=1,
    min_containers=1,
)
@modal.concurrent(max_inputs=1000)
@modal.web_server(WAN_PORT, startup_timeout=STARTUP_TIMEOUT)
def run_ui():
    _run_ui_process(UI_EXTRA_ARGS, wait=False)


def _run_ui_process(extra_args: str = "", wait: bool = True):
    config_path = _ensure_runtime_layout()
    config_dir = config_path.parent
    settings_dir = Path(STATE_MOUNT) / "settings"

    env = os.environ.copy()
    env["HOME"] = "/root"
    _set_runtime_thread_env(env)

    command = [
        "python",
        "-u",
        "wgp.py",
        "--advanced",
        "--listen",
        "--server-name",
        "0.0.0.0",
        "--server-port",
        str(WAN_PORT),
        "--config",
        str(config_dir),
        "--settings",
        str(settings_dir),
        "--loras",
        f"{MODELS_MOUNT}/loras",
    ]

    command.extend(DEFAULT_WGP_ARGS)
    if extra_args:
        command.extend(shlex.split(extra_args))

    if wait:
        _run_wgp_process(command, env)
    else:
        _start_wgp_server_process(command, env)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=CPU_COUNT,
    timeout=FUNCTION_TIMEOUT,
    startup_timeout=STARTUP_TIMEOUT,
    volumes={
        MODELS_MOUNT: models_volume,
        STATE_MOUNT: state_volume,
        OUTPUTS_MOUNT: outputs_volume,
    },
)
def run_cli(
    process_path: str,
    dry_run: bool = False,
    extra_args: str = "",
):
    config_path = _ensure_runtime_layout()
    config_dir = config_path.parent
    settings_dir = Path(STATE_MOUNT) / "settings"

    env = os.environ.copy()
    env["HOME"] = "/root"
    _set_runtime_thread_env(env)

    command = [
        "python",
        "-u",
        "wgp.py",
        "--advanced",
        "--process",
        process_path,
        "--output-dir",
        OUTPUTS_MOUNT,
        "--config",
        str(config_dir),
        "--settings",
        str(settings_dir),
        "--loras",
        f"{MODELS_MOUNT}/loras",
    ]

    command.extend(DEFAULT_WGP_ARGS)
    if dry_run:
        command.append("--dry-run")
    if extra_args:
        command.extend(shlex.split(extra_args))

    _run_wgp_process(command, env)


ui_process = None
ui_process_lock = threading.Lock()
ui_monitor_thread = None
ui_cleanup_registered = False
ui_stop_requested = False


def _print_runtime_command(command: list[str]) -> None:
    print(
        f"Runtime CPU request: {CPU_COUNT:g} cores; "
        f"thread env: OMP/MKL/OPENBLAS/NUMEXPR={RUNTIME_THREADS}",
        flush=True,
    )
    print("Running:", shlex.join(command), flush=True)


def _start_wgp_server_process(command: list[str], env: dict[str, str]) -> None:
    global ui_cleanup_registered, ui_monitor_thread

    _print_runtime_command(command)
    with ui_process_lock:
        if ui_process is None or ui_process.poll() is not None:
            _spawn_wgp_server_process_locked(command, env)
        else:
            print("Wan2GP UI process is already running.", flush=True)

        if ui_monitor_thread is None or not ui_monitor_thread.is_alive():
            ui_monitor_thread = threading.Thread(
                target=_monitor_wgp_server_process,
                args=(command, env.copy()),
                name="wan2gp-ui-supervisor",
                daemon=True,
            )
            ui_monitor_thread.start()

        if not ui_cleanup_registered:
            atexit.register(_cleanup_wgp_server_process)
            ui_cleanup_registered = True

    _wait_for_wgp_server_ready(WAN_PORT, UI_READY_TIMEOUT)


def _spawn_wgp_server_process_locked(command: list[str], env: dict[str, str]) -> subprocess.Popen:
    global ui_process, ui_stop_requested

    ui_stop_requested = False
    ui_process = subprocess.Popen(
        command,
        cwd=REMOTE_APP_ROOT,
        env=env,
    )
    print(f"Wan2GP UI process started with pid {ui_process.pid}.", flush=True)
    return ui_process


def _monitor_wgp_server_process(command: list[str], env: dict[str, str]) -> None:
    while True:
        with ui_process_lock:
            process = ui_process
            should_stop = ui_stop_requested

        if should_stop:
            return
        if process is None:
            with ui_process_lock:
                if not ui_stop_requested:
                    _spawn_wgp_server_process_locked(command, env)
            continue

        return_code = process.wait()
        with ui_process_lock:
            should_stop = ui_stop_requested or process is not ui_process
        if should_stop:
            return

        print(
            f"Wan2GP UI process exited unexpectedly with code {return_code}; "
            f"restarting in {UI_RESTART_DELAY}s.",
            flush=True,
        )
        time.sleep(UI_RESTART_DELAY)
        with ui_process_lock:
            if not ui_stop_requested and process is ui_process:
                _spawn_wgp_server_process_locked(command, env)


def _wait_for_wgp_server_ready(port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    last_pid = None
    while time.monotonic() < deadline:
        with ui_process_lock:
            process = ui_process
            should_stop = ui_stop_requested

        if should_stop:
            raise RuntimeError("Wan2GP UI startup was cancelled.")
        if process is not None and process.pid != last_pid:
            last_pid = process.pid
            print(f"Waiting for Wan2GP UI process {last_pid} to accept connections...", flush=True)

        try:
            with socket.create_connection(("127.0.0.1", port), timeout=UI_CONNECT_TIMEOUT):
                print(f"Wan2GP UI process is accepting connections on port {port}.", flush=True)
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1)

    raise TimeoutError(
        f"Wan2GP UI process did not accept connections on port {port} "
        f"within {timeout}s. Last error: {last_error}"
    )


def _cleanup_wgp_server_process() -> None:
    global ui_stop_requested

    ui_stop_requested = True
    with ui_process_lock:
        process = ui_process
    _cleanup_wgp_process(process)


def _cleanup_wgp_process(process) -> None:
    try:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    finally:
        outputs_volume.commit()
        state_volume.commit()


def _run_wgp_process(command: list[str], env: dict[str, str]) -> None:
    _print_runtime_command(command)
    process = None
    try:
        process = subprocess.Popen(
            command,
            cwd=REMOTE_APP_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    finally:
        _cleanup_wgp_process(process)


def _set_runtime_thread_env(env: dict[str, str]) -> None:
    env["OMP_NUM_THREADS"] = RUNTIME_THREADS
    env["MKL_NUM_THREADS"] = RUNTIME_THREADS
    env["OPENBLAS_NUM_THREADS"] = RUNTIME_THREADS
    env["NUMEXPR_NUM_THREADS"] = RUNTIME_THREADS
    env["TORCHINDUCTOR_CACHE_DIR"] = f"{STATE_MOUNT}/torchinductor"
    env["TRITON_CACHE_DIR"] = f"{STATE_MOUNT}/triton"


def _batch_upload_directory(volume: modal.Volume, local_dir: Path, remote_dir: str) -> None:
    if not local_dir.exists():
        print(f"Skipping missing path: {local_dir}")
        return

    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(local_dir), remote_dir)
    print(f"Uploaded {local_dir} -> {volume.name}:{remote_dir}")


@app.local_entrypoint()
def main(
    process: str = "",
    dry_run: bool = False,
    extra_args: str = "",
    seed_models: bool = False,
    seed_settings: bool = False,
):
    if seed_models:
        _batch_upload_directory(models_volume, REPO_ROOT / "ckpts", "/ckpts")
        _batch_upload_directory(models_volume, REPO_ROOT / "loras", "/loras")

    if seed_settings:
        _batch_upload_directory(state_volume, REPO_ROOT / "settings", "/settings")
        local_config = REPO_ROOT / "wgp_config.json"
        if local_config.exists():
            with state_volume.batch_upload(force=True) as batch:
                batch.put_file(str(local_config), "/config/wgp_config.json")
            print(f"Uploaded {local_config} -> {state_volume.name}:/config/wgp_config.json")

    if not process:
        if extra_args:
            print("Ignoring --extra-args for Modal web UI. Set UI_EXTRA_ARGS in launchModal.py instead.")
        print("Start the Wan2GP web UI with:")
        print("  modal serve launchModal.py")
        print("Modal will print the public web_server URL for run_ui.")
        print("Persisted settings/config live in the Modal state volume.")
        print("Local settings are only uploaded when you pass --seed-settings.")
        return

    process_path = Path(process)
    if process_path.exists():
        remote_process_path = f"{STATE_MOUNT}/queue/{process_path.name}"
        with state_volume.batch_upload(force=True) as batch:
            batch.put_file(str(process_path), f"/queue/{process_path.name}")
        print(f"Uploaded {process_path} -> {state_volume.name}:/queue/{process_path.name}")
    else:
        remote_process_path = process

    run_cli.remote(
        remote_process_path,
        dry_run=dry_run,
        extra_args=extra_args,
    )
