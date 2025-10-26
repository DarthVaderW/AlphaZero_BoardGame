# -*- coding: utf-8 -*-
"""
Simple task registry and YAML loader, inspired by Isaac Gym style configs.
"""
from typing import Optional, Dict, Any
import os
import yaml

_TASKS: Dict[str, str] = {
    "gobang_6x6": os.path.join(os.path.dirname(__file__), "tasks", "gobang_6x6.yaml"),
    "test_quick": os.path.join(os.path.dirname(__file__), "tasks", "test_quick.yaml"),
    "train_default": os.path.join(os.path.dirname(__file__), "tasks", "train_default.yaml"),
}


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def register_task(name: str, yaml_path: str) -> None:
    _TASKS[name] = yaml_path


def load_task(name: str, base_path: Optional[str] = None) -> Dict[str, Any]:
    if name not in _TASKS:
        raise KeyError(f"Unknown task: {name}")
    cfg = _load_yaml(_TASKS[name])
    if base_path:
        base_cfg = _load_yaml(base_path)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def load_config(path: str, base_path: Optional[str] = None) -> Dict[str, Any]:
    cfg = _load_yaml(path)
    if base_path:
        base_cfg = _load_yaml(base_path)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg