from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import torch


ONE_MB = 1024 * 1024
DEFAULT_RAM_CACHE_RATIO = 0.70


@dataclass
class CacheStats:
    gpu_bytes: int = 0
    ram_bytes: int = 0
    skipped_bytes: int = 0
    tensors_gpu: int = 0
    tensors_ram: int = 0
    tensors_skipped: int = 0


def _parse_cuda_index(device: str | int | None) -> int:
    if device is None or str(device).strip() == "":
        return torch.cuda.current_device() if torch.cuda.is_available() else 0
    if isinstance(device, int):
        return device
    text = str(device).strip().lower()
    if text == "cuda":
        return torch.cuda.current_device() if torch.cuda.is_available() else 0
    if text.startswith("cuda:"):
        text = text.split(":", 1)[1]
    return int(text) if text.isdigit() else 0


def _parse_secondary_devices(devices: str | list[str] | tuple[str, ...] | None, primary_index: int) -> list[str]:
    if not torch.cuda.is_available():
        return []
    count = torch.cuda.device_count()
    if devices is None or devices == "" or str(devices).strip().lower() == "auto":
        return [f"cuda:{idx}" for idx in range(count) if idx != primary_index]
    if isinstance(devices, str):
        raw_devices = [part.strip() for part in devices.split(",")]
    else:
        raw_devices = [str(part).strip() for part in devices]
    parsed: list[str] = []
    for raw in raw_devices:
        if not raw:
            continue
        if raw.lower().startswith("cuda:"):
            idx_text = raw.split(":", 1)[1]
        else:
            idx_text = raw
        if not idx_text.isdigit():
            continue
        idx = int(idx_text)
        if idx == primary_index or idx < 0 or idx >= count:
            continue
        parsed.append(f"cuda:{idx}")
    return list(dict.fromkeys(parsed))


def _available_ram_budget_bytes(ratio: float = DEFAULT_RAM_CACHE_RATIO) -> int:
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
    except Exception:
        return 0
    ratio = min(max(float(ratio), 0.0), 0.95)
    return int(available * ratio)


def _available_vram_budget_bytes(device: str, ratio: float) -> int:
    ratio = min(max(float(ratio), 0.0), 0.95)
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(torch.device(device))
    except Exception:
        idx = _parse_cuda_index(device)
        props = torch.cuda.get_device_properties(idx)
        with torch.cuda.device(idx):
            free_bytes = props.total_memory - torch.cuda.memory_reserved(idx)
    return max(0, int(free_bytes * ratio))


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:
        try:
            return int(tensor.numel() * tensor.element_size())
        except Exception:
            return 0


def _clone_to_cache(tensor: torch.Tensor, device: str) -> torch.Tensor:
    with torch.no_grad():
        cached = tensor.detach().to(device)
        if device == "cpu":
            try:
                cached = cached.clone()
            except Exception:
                pass
        return cached


def _make_cached_holder(tensor: torch.Tensor, is_buffer: bool) -> torch.Tensor:
    if is_buffer:
        return torch.nn.Buffer(tensor)
    return torch.nn.Parameter(tensor, requires_grad=False)


def _iter_block_entries(manager: Any):
    blocks_of_modules = getattr(manager, "blocks_of_modules", {})
    for entry_name, blocks_params in list(blocks_of_modules.items()):
        for index, param in enumerate(list(blocks_params)):
            yield entry_name, blocks_params, index, param


def _tied_source_keys(manager: Any) -> set[tuple[int, str]]:
    keys: set[tuple[int, str]] = set()
    for _entry_name, _blocks_params, _index, param in _iter_block_entries(manager):
        _parent_module, _name, _tensor, _is_buffer, tied_param = param
        if tied_param is not None:
            tied_parent, tied_name = tied_param
            keys.add((id(tied_parent), tied_name))
    return keys


def enable_dynamic_offload_cache(
    manager: Any,
    *,
    enabled: bool,
    primary_device: str | int | None = None,
    secondary_devices: str | list[str] | tuple[str, ...] | None = "auto",
    secondary_vram_ratio: float = 0.80,
    ram_cache: bool = True,
    ram_cache_ratio: float = DEFAULT_RAM_CACHE_RATIO,
    verbose: int = 1,
) -> CacheStats:
    stats = CacheStats()
    if manager is None or not torch.cuda.is_available():
        return stats
    if not enabled and not ram_cache:
        return stats

    primary_index = _parse_cuda_index(primary_device)
    gpu_devices = _parse_secondary_devices(secondary_devices, primary_index) if enabled else []
    gpu_budgets = {device: _available_vram_budget_bytes(device, secondary_vram_ratio) for device in gpu_devices}
    ram_budget = _available_ram_budget_bytes(ram_cache_ratio) if ram_cache else 0
    tied_sources = _tied_source_keys(manager)
    cache_by_ref: dict[int, torch.Tensor] = {}

    if verbose >= 1 and (gpu_devices or ram_cache):
        gpu_bits = ", ".join(f"{dev}~{budget / ONE_MB:.0f}MB" for dev, budget in gpu_budgets.items()) or "none"
        ram_bits = f"{ram_budget / ONE_MB:.0f}MB" if ram_cache else "disabled"
        print(f"[multi-gpu-offload] Cache targets: secondary VRAM={gpu_bits}, RAM={ram_bits}")

    for entry_name, blocks_params, index, param in _iter_block_entries(manager):
        parent_module, name, tensor, is_buffer, tied_param = param
        if not torch.is_tensor(tensor):
            continue
        size = _tensor_nbytes(tensor)
        if size <= 0:
            continue

        cache_key = id(tensor)
        cached = cache_by_ref.get(cache_key)
        if cached is None:
            target = None
            can_use_secondary = tied_param is None and (id(parent_module), name) not in tied_sources
            if can_use_secondary:
                for device in gpu_devices:
                    if gpu_budgets[device] >= size:
                        target = device
                        gpu_budgets[device] -= size
                        stats.gpu_bytes += size
                        stats.tensors_gpu += 1
                        break
            if target is None and ram_cache and ram_budget >= size:
                target = "cpu"
                ram_budget -= size
                stats.ram_bytes += size
                stats.tensors_ram += 1
            if target is None:
                stats.skipped_bytes += size
                stats.tensors_skipped += 1
                continue
            cached = _make_cached_holder(_clone_to_cache(tensor, target), is_buffer)
            cache_by_ref[cache_key] = cached

        blocks_params[index] = (parent_module, name, cached, is_buffer, tied_param)
        try:
            setattr(parent_module, name, cached)
        except Exception:
            if verbose >= 2:
                print(f"[multi-gpu-offload] Unable to replace cached tensor for {entry_name}:{name}")

    for device in gpu_devices:
        try:
            torch.cuda.synchronize(torch.device(device))
        except Exception:
            pass
    gc.collect()
    setattr(manager, "_wgp_dynamic_offload_cache", stats)

    if verbose >= 1 and (stats.gpu_bytes or stats.ram_bytes or stats.skipped_bytes):
        print(
            "[multi-gpu-offload] Cached "
            f"{stats.gpu_bytes / ONE_MB:.0f}MB in secondary VRAM, "
            f"{stats.ram_bytes / ONE_MB:.0f}MB in RAM; "
            f"{stats.skipped_bytes / ONE_MB:.0f}MB left disk-backed."
        )
    return stats
