from __future__ import annotations

import types
from typing import Any

import gc


HOT_MODE_ATTR = "_wgp_keep_models_hot"
HOT_LOADED_ATTR = "_wgp_hot_model_ids_loaded"
CPU_BACKUPS_DROPPED_ATTR = "_wgp_cpu_model_backups_dropped"
_global_keep_models_hot = False


def set_global_keep_models_hot(enabled: bool) -> None:
    global _global_keep_models_hot
    _global_keep_models_hot = bool(enabled)


def global_keep_models_hot_enabled() -> bool:
    return _global_keep_models_hot


def keep_models_hot_enabled(manager: Any) -> bool:
    return bool(getattr(manager, HOT_MODE_ATTR, False))


def enable_keep_models_hot(manager: Any) -> Any:
    if manager is None:
        return manager
    setattr(manager, HOT_MODE_ATTR, True)
    if not hasattr(manager, "_wgp_original_can_model_be_cotenant"):
        manager._wgp_original_can_model_be_cotenant = manager.can_model_be_cotenant

        def _can_model_be_cotenant(self, model_id):
            return True

        manager.can_model_be_cotenant = types.MethodType(_can_model_be_cotenant, manager)
    return manager


def load_all_models_to_vram(manager: Any, force: bool = False) -> None:
    if manager is None or not keep_models_hot_enabled(manager):
        return
    model_ids = tuple(getattr(manager, "models", {}).keys())
    active_models_ids = getattr(manager, "active_models_ids", [])
    active_model_ids_set = set(active_models_ids)
    already_loaded = (
        getattr(manager, HOT_LOADED_ATTR, None) == model_ids
        and active_model_ids_set.issuperset(model_ids)
    )
    cpu_backups_dropped = bool(getattr(manager, CPU_BACKUPS_DROPPED_ATTR, False))
    if already_loaded and cpu_backups_dropped and not force:
        return
    for model_id in model_ids:
        if force or model_id not in active_model_ids_set:
            manager.gpu_load(model_id)
    if force or not cpu_backups_dropped:
        drop_cpu_model_backups(manager)
        setattr(manager, CPU_BACKUPS_DROPPED_ATTR, True)
    setattr(manager, HOT_LOADED_ATTR, model_ids)


def drop_cpu_model_backups(manager: Any) -> None:
    blocks_of_modules = getattr(manager, "blocks_of_modules", None)
    if not isinstance(blocks_of_modules, dict):
        return
    for entry_name, blocks_params in list(blocks_of_modules.items()):
        new_blocks_params = []
        for parent_module, name, parameter, is_buffer, tied_param in blocks_params:
            current_value = getattr(parent_module, name, parameter)
            new_blocks_params.append((parent_module, name, current_value, is_buffer, tied_param))
        blocks_of_modules[entry_name] = new_blocks_params
    gc.collect()


def unload_all_unless_hot(manager: Any) -> None:
    if manager is None or keep_models_hot_enabled(manager):
        return
    manager.unload_all()
