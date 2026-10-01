"""Per-dataset adapters into the unified ground-truth format.

Verification status matters here and is recorded honestly, because an adapter written
against a documented format is not the same thing as an adapter known to work:

===========  ==========  ====================================================
adapter      status      note
===========  ==========  ====================================================
``funsd``    verified    downloaded and converted end to end
``sroie``    verified    downloaded and converted end to end
``cheque``   spec-only   needs the HF datasets; layout per the dataset card
``midv``     spec-only   124 GB behind an access form; quadrangle annotations
``ddi100``   spec-only   large download; pickled per-page annotations
===========  ==========  ====================================================

A spec-only adapter raises :class:`~ocrbench.adapters.base.AdapterError` with the
mismatch when the real layout differs, rather than silently producing empty samples.
"""

from . import base

__all__ = ["base"]
