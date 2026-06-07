from __future__ import annotations

import types
from typing import Any

import gc


HOT_MODE_ATTR = "_wgp_keep_models_hot"
HOT_LOADED_ATTR = "_wgp_hot_model_ids_loaded"
CPU_BACKUPS_DROPPED_ATTR = "_wgp_cpu_model_backups_dropped"
OFFLOAD_MANAGER_STATE_KEY = "_wgp_offload_manager"
TEXT_ENCODER_MODEL_IDS = (
    "text_encoder",
    "text_encoder_2",
    "text_embedding_projection",
    "text_embeddings_connector",
)
_global_keep_models_hot = False


def set_global_keep_models_hot(enabled: bool) -> None:
    global _global_keep_models_hot
    _global_keep_models_hot = bool(enabled)


def global_keep_models_hot_enabled() -> bool:
    return _global_keep_models_hot


def keep_models_hot_enabled(manager: Any) -> bool:
    return bool(getattr(manager, HOT_MODE_ATTR, False))


def register_offload_manager(manager: Any) -> None:
    try:
        from mmgp import offload as mmgp_offload
    except ImportError:
        return
    if manager is None:
        mmgp_offload.shared_state.pop(OFFLOAD_MANAGER_STATE_KEY, None)
        return
    mmgp_offload.shared_state[OFFLOAD_MANAGER_STATE_KEY] = manager


def get_registered_offload_manager() -> Any:
    try:
        from mmgp import offload as mmgp_offload
    except ImportError:
        return None
    return mmgp_offload.shared_state.get(OFFLOAD_MANAGER_STATE_KEY)


def _preload_model_blocks_on_gpu(manager: Any, model_id: str) -> None:
    manager.gpu_load_blocks(model_id, None, True)
    for block_name in getattr(manager, "preloaded_blocks_per_model", {}).get(model_id, ()):
        manager.gpu_load_blocks(model_id, block_name, True)


def ensure_text_encoder_models_on_gpu(manager: Any | None = None, force: bool = False) -> None:
    if manager is None:
        manager = get_registered_offload_manager()
    if manager is None:
        return
    if not (keep_models_hot_enabled(manager) or global_keep_models_hot_enabled()):
        return
    models = getattr(manager, "models", {})
    active_model_ids = set(getattr(manager, "active_models_ids", []))
    for model_id in TEXT_ENCODER_MODEL_IDS:
        if model_id not in models:
            continue
        if force or model_id not in active_model_ids:
            manager.gpu_load(model_id)
            active_model_ids.add(model_id)
        else:
            _preload_model_blocks_on_gpu(manager, model_id)


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
    ensure_text_encoder_models_on_gpu(manager, force=True)
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
