# Explicit named re-exports (no star) — star imports here leak the submodule
# names (`prejepa`, `module`) into the package namespace, which then get
# re-leaked by wm/__init__.py's `from .prejepa import *` and shadow the
# parent package binding. Explicit list keeps `swm.wm.prejepa` resolving
# to the package, not the homonymous submodule.
from .prejepa import PreJEPA
from .module import (
    Attention,
    CausalPredictor,
    Embedder,
    FeedForward,
    Transformer,
)

__all__ = [
    'PreJEPA',
    'CausalPredictor',
    'Embedder',
    'Attention',
    'FeedForward',
    'Transformer',
]
