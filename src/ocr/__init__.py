"""CCL-based OCR engine — Phases 0-2 (benchmark, CCL, component filtering)."""

from .config import DEFAULT_CONFIG, PipelineConfig
from .engine import CCLEngine, Engine, load_image
from .types import BBox, Component, ComponentKind, Line, PageResult, Word

__all__ = [
    "BBox",
    "CCLEngine",
    "Component",
    "ComponentKind",
    "DEFAULT_CONFIG",
    "Engine",
    "Line",
    "PageResult",
    "PipelineConfig",
    "Word",
    "load_image",
]
