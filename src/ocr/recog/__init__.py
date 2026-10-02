"""Phase 3 recogniser: glyph CNN, character LM and constrained decoding (PyTorch).

``charset`` and ``formats`` are pure Python; the model modules import torch lazily so
the rest of :mod:`ocr` never requires it.
"""
