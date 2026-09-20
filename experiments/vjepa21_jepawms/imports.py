"""Collision-safe import of the exact JEPA-WMs decoder implementation.

Both FactorJEPA and JEPA-WMs expose top-level ``src`` packages. Importing the
decoder normally can therefore resolve JEPA-WMs imports against FactorJEPA's
``src`` package. This helper temporarily isolates the dependency and returns
the upstream decoder class after all global modules have been restored.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from experiments.vjepa21_jepawms.common import PROJECT_ROOT


_DECODER_CLASS = None


def get_official_decoder_class(dependency_root: str | Path | None = None):
    global _DECODER_CLASS
    if _DECODER_CLASS is not None:
        return _DECODER_CLASS

    root = Path(dependency_root or PROJECT_ROOT / "deps" / "jepa-wms").resolve()
    decoder_file = root / "app" / "plan_common" / "models" / "decoder.py"
    if not decoder_file.is_file():
        raise FileNotFoundError(
            f"JEPA-WMs dependency not found at {root}. Run the assets stage first."
        )

    saved_cwd = os.getcwd()
    saved_path = sys.path[:]
    saved_modules = {
        key: module
        for key, module in list(sys.modules.items())
        if key == "src" or key.startswith("src.") or key == "app" or key.startswith("app.")
    }
    for key in saved_modules:
        sys.modules.pop(key, None)

    try:
        os.chdir("/tmp")
        project_root = PROJECT_ROOT.resolve()
        filtered = []
        for entry in saved_path:
            if not entry:
                filtered.append(entry)
                continue
            try:
                if Path(entry).resolve() == project_root:
                    continue
            except (OSError, RuntimeError):
                pass
            filtered.append(entry)
        sys.path = [str(root)] + filtered
        importlib.invalidate_caches()
        module = importlib.import_module("app.plan_common.models.decoder")
        _DECODER_CLASS = module.VisionTransformerDecoder
    finally:
        for key in list(sys.modules):
            if key == "src" or key.startswith("src.") or key == "app" or key.startswith("app."):
                sys.modules.pop(key, None)
        sys.modules.update(saved_modules)
        sys.path = saved_path
        os.chdir(saved_cwd)
        importlib.invalidate_caches()

    return _DECODER_CLASS
