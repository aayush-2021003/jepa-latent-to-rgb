"""Import vjepa2 modules from deps/vjepa2/ without src.utils namespace collision.

V-JEPA 2.0 modules: deps/vjepa2/src/models/ (base ViT, predictor, masks)
V-JEPA 2.1 modules: deps/vjepa2/app/vjepa_2_1/models/ (hierarchical output, deep supervision, modality embedding)

get_vit_giant_xformers()     → 2.0 ViT-g (1B, 1408-dim, depth=40) — NO deep supervision
get_vit_gigantic_xformers()  → 2.1 ViT-G (2B, 1664-dim, depth=48) — WITH deep supervision
get_vit_predictor_2_1()      → 2.1 predictor (return_all_tokens, predictor_proj_context)
"""
import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACTORJEPA_SRC = PROJECT_ROOT / "src"
VJEPA2_ROOT = PROJECT_ROOT / "deps" / "vjepa2"

_loaded_base = False
_loaded_2_1 = False


def _without_factorjepa_paths(paths):
    """Remove this checkout's import roots without assuming its directory name."""
    excluded = {PROJECT_ROOT.resolve(), FACTORJEPA_SRC.resolve()}
    result = []
    for entry in paths:
        if not entry:
            # CWD is changed to /tmp while importing, so an empty entry is safe.
            result.append(entry)
            continue
        try:
            resolved = Path(entry).resolve()
        except (OSError, RuntimeError):
            result.append(entry)
            continue
        if resolved not in excluded:
            result.append(entry)
    return result


def _ensure_loaded_base():
    """One-time import of V-JEPA 2.0 (base) modules into sys.modules."""
    global _loaded_base
    if _loaded_base:
        return
    _loaded_base = True

    vjepa2_root = str(VJEPA2_ROOT)

    saved_cwd = os.getcwd()
    saved_path = sys.path[:]
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == "src" or key.startswith("src."):
            saved_modules[key] = sys.modules.pop(key)

    os.chdir("/tmp")
    sys.path = [vjepa2_root] + _without_factorjepa_paths(saved_path)
    importlib.invalidate_caches()

    try:
        importlib.import_module("src.utils.tensors")
        importlib.import_module("src.models.utils.modules")
        importlib.import_module("src.models.utils.patch_embed")
        importlib.import_module("src.models.utils.pos_embs")
        importlib.import_module("src.models.vision_transformer")
        importlib.import_module("src.models.predictor")
        importlib.import_module("src.models.attentive_pooler")   # probe_action — AttentivePooler + AttentiveClassifier
        importlib.import_module("src.masks.utils")
        importlib.import_module("src.masks.multiseq_multiblock3d")
    finally:
        os.chdir(saved_cwd)
        sys.path = saved_path
        for key in ["src", "src.utils"]:
            if key in saved_modules:
                sys.modules[key] = saved_modules[key]
        importlib.invalidate_caches()


def _ensure_loaded_2_1():
    """One-time import of V-JEPA 2.1 modules (app/vjepa_2_1/) into sys.modules.

    2.1 modules depend on base modules (src.utils.tensors, src.masks.utils)
    so we load base first, then add 2.1 on top.
    """
    global _loaded_2_1
    if _loaded_2_1:
        return
    _ensure_loaded_base()
    _loaded_2_1 = True

    vjepa2_root = str(VJEPA2_ROOT)

    saved_cwd = os.getcwd()
    saved_path = sys.path[:]
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == "src" or key.startswith("src."):
            saved_modules[key] = sys.modules.pop(key)
        if key == "app" or key.startswith("app."):
            saved_modules[key] = sys.modules.pop(key)

    os.chdir("/tmp")
    sys.path = [vjepa2_root] + _without_factorjepa_paths(saved_path)
    importlib.invalidate_caches()

    try:
        # Re-import base modules (needed by 2.1 as dependencies)
        importlib.import_module("src.utils.tensors")
        importlib.import_module("src.masks.utils")
        # Import 2.1 modules
        importlib.import_module("app.vjepa_2_1.models.utils.modules")
        importlib.import_module("app.vjepa_2_1.models.utils.patch_embed")
        importlib.import_module("app.vjepa_2_1.models.vision_transformer")
        importlib.import_module("app.vjepa_2_1.models.predictor")
    finally:
        os.chdir(saved_cwd)
        sys.path = saved_path
        # Restore ALL saved src.* / app.* modules — previously only restored
        # "src" and "src.utils", which dropped src.models.predictor et al.
        # This left subsequent calls to get_vit_predictor() / get_mask_generator()
        # with KeyError on sys.modules lookup (#50 post-split regression).
        # Only skip modules that are NEEDED by 2.1 AND already re-imported above.
        _freshly_imported = {
            "src.utils.tensors",
            "src.masks.utils",
        }
        for key, mod in saved_modules.items():
            if key in _freshly_imported:
                continue  # keep the freshly-imported one (may have 2.1-aware version)
            if key not in sys.modules:
                sys.modules[key] = mod
        importlib.invalidate_caches()


# ── V-JEPA 2.0 (base) ───────────────────────────────────────────────

def get_vit_giant_xformers():
    """V-JEPA 2.0 ViT-g (1B, embed_dim=1408, depth=40, 22 heads). NO deep supervision."""
    _ensure_loaded_base()
    return sys.modules["src.models.vision_transformer"].vit_giant_xformers


def get_vit_large():
    """ViT-L (300M, embed_dim=1024, depth=24). Base module — FROZEN baselines only
    (V-JEPA 1 ViT-L, V-JEPA 2.0/2 ViT-L). iter17: cross-arch frozen eval (§B1)."""
    _ensure_loaded_base()
    return sys.modules["src.models.vision_transformer"].vit_large


def get_vit_huge():
    """ViT-H (embed_dim=1280, depth=32). Base module — FROZEN baselines only
    (V-JEPA 1 ViT-H). iter17: cross-arch frozen eval (§B1)."""
    _ensure_loaded_base()
    return sys.modules["src.models.vision_transformer"].vit_huge


# ── V-JEPA 2.1 (with deep supervision + dense loss) ─────────────────

def get_vit_gigantic_xformers():
    """V-JEPA 2.1 ViT-G (2B, embed_dim=1664, depth=48, 26 heads).
    WITH deep supervision (hierarchical_layers, norms_block, return_hierarchical).
    """
    _ensure_loaded_2_1()
    return sys.modules["app.vjepa_2_1.models.vision_transformer"].vit_gigantic_xformers


def get_vit_giant_xformers_2_1():
    """V-JEPA 2.1 ViT-g (1B, embed_dim=1408, depth=40, 22 heads) WITH deep supervision
    (2.1 app module — norms_block/hierarchical). iter17: cross-arch 2.1 TRAINING (the base-module
    vit_giant_xformers drops the deep-sup norms; frozen eval tolerates it, training needs them)."""
    _ensure_loaded_2_1()
    return sys.modules["app.vjepa_2_1.models.vision_transformer"].vit_giant_xformers


def get_vit_predictor_2_1():
    """V-JEPA 2.1 predictor (return_all_tokens, predictor_proj_context for dense loss)."""
    _ensure_loaded_2_1()
    return sys.modules["app.vjepa_2_1.models.predictor"].vit_predictor


def get_vit_by_arch(arch: str):
    """Return ViT constructor by arch name from model config YAML."""
    dispatch = {
        "vit_giant_xformers": get_vit_giant_xformers,
        "vit_gigantic_xformers": get_vit_gigantic_xformers,
        "vit_large": get_vit_large,     # iter17: frozen V-JEPA 1 ViT-L / 2.0 / 2 ViT-L
        "vit_huge": get_vit_huge,       # iter17: frozen V-JEPA 1 ViT-H
        "vit_giant_xformers_2_1": get_vit_giant_xformers_2_1,  # iter17: 2.1 ViT-g deep-sup (TRAIN)
    }
    if arch not in dispatch:
        print(f"FATAL: Unknown arch '{arch}'. Supported: {list(dispatch.keys())}")
        sys.exit(1)
    return dispatch[arch]()


# ── Shared (works with both 2.0 and 2.1) ─────────────────────────────

def get_vit_predictor():
    """V-JEPA 2.0 base predictor (no return_all_tokens). Use get_vit_predictor_2_1() for 2.1."""
    _ensure_loaded_base()
    return sys.modules["src.models.predictor"].vit_predictor


def get_mask_generator():
    _ensure_loaded_base()
    return sys.modules["src.masks.multiseq_multiblock3d"]._MaskGenerator


def get_apply_masks():
    _ensure_loaded_base()
    return sys.modules["src.masks.utils"].apply_masks


# ── Attentive probe head (used by Meta's evals + probe_action) ──

def get_attentive_classifier():
    """Meta's AttentiveClassifier from deps/vjepa2/src/models/attentive_pooler.py.

    Wraps an N-layer AttentivePooler + Linear head. Constructor kwargs:
        embed_dim, num_classes, depth, num_heads, mlp_ratio,
        complete_block, use_activation_checkpointing.
    EXACT module Meta uses for SSv2/K400/Diving-48/EK100 attentive probes.
    """
    _ensure_loaded_base()
    return sys.modules["src.models.attentive_pooler"].AttentiveClassifier


def get_attentive_pooler():
    """Bare AttentivePooler (no Linear head). Use when only pooled features are needed."""
    _ensure_loaded_base()
    return sys.modules["src.models.attentive_pooler"].AttentivePooler
