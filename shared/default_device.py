from __future__ import annotations

import os
import sys

from collections.abc import MutableSequence
from pathlib import Path

import json


def _arg_name_to_option(arg_name: str) -> str:
    arg_name = str(arg_name or "").strip()
    if not arg_name:
        return ""
    return arg_name if arg_name.startswith("--") else f"--{arg_name}"


def _cuda_visible_device(device: str) -> str:
    device = str(device or "").strip().lower()
    if device.startswith("cuda:"):
        device = device.split(":", 1)[1]
    return device if device.isdigit() else ""


def _preserve_cuda_visibility_for_multi_gpu(argv: MutableSequence[str]) -> bool:
    if os.environ.get("WAN2GP_MULTI_GPU_OFFLOAD", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    if any(arg == "--multi-gpu-offload" or str(arg).startswith("--multi-gpu-offload=") for arg in argv[1:]):
        return True
    return _config_requests_multi_gpu_offload(argv)


def _config_requests_multi_gpu_offload(argv: MutableSequence[str]) -> bool:
    config_dir = ""
    for index, arg in enumerate(argv[1:], start=1):
        if arg == "--config" and index + 1 < len(argv):
            config_dir = str(argv[index + 1]).strip()
            break
        if str(arg).startswith("--config="):
            config_dir = str(arg).split("=", 1)[1].strip()
            break
    candidates = []
    if config_dir:
        candidates.append(Path(config_dir) / "wgp_config.json")
    candidates.append(Path("wgp_config.json"))
    for candidate in candidates:
        try:
            if not candidate.is_file():
                continue
            with candidate.open("r", encoding="utf-8") as handle:
                value = json.load(handle).get("multi_gpu_offload", 0)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value != 0
            return str(value).strip().lower() in ("1", "true", "yes", "on")
        except Exception:
            continue
    return False


def _rewrite_arg_value(argv: MutableSequence[str], option: str, value: str) -> None:
    for index, arg in enumerate(argv):
        if arg == option and index + 1 < len(argv):
            argv[index + 1] = value
            return
        if str(arg).startswith(f"{option}="):
            argv[index] = f"{option}={value}"
            return


def set_default_cuda_device_from_arg(arg_name: str, default_device: str = "cuda:0") -> bool:
    option = _arg_name_to_option(arg_name)
    if not option:
        return False
    argv = sys.argv
    for index, arg in enumerate(argv[1:], start=1):
        if arg == option and index + 1 < len(argv):
            visible_device = _cuda_visible_device(argv[index + 1])
            break
        if str(arg).startswith(f"{option}="):
            visible_device = _cuda_visible_device(str(arg).split("=", 1)[1])
            break
    else:
        return False

    if not visible_device:
        return False
    if _preserve_cuda_visibility_for_multi_gpu(argv):
        _rewrite_arg_value(argv, option, f"cuda:{visible_device}")
        return False
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device
    _rewrite_arg_value(argv, option, default_device)
    return True
