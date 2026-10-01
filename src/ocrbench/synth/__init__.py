"""Synthetic document generation with exact ground truth."""

from .degrade import PROFILES, DegradeProfile, degrade
from .generate import DEFAULT_STRATA, Stratum, generate, generate_sample, plan
from .render import DocumentSpec, Shape, TextLine, render
from .templates import TEMPLATES

__all__ = [
    "DEFAULT_STRATA",
    "DocumentSpec",
    "DegradeProfile",
    "PROFILES",
    "Shape",
    "Stratum",
    "TEMPLATES",
    "TextLine",
    "degrade",
    "generate",
    "generate_sample",
    "plan",
    "render",
]
