"""
Builds a torchvision-transforms-v2-compatible CLAHE op, backed by the same
albumentations.CLAHE used in src/data/augmentation.py, for DEIMv2/D-FINE's
custom transform registry (each repo's Compose resolves op `type` strings
via its own @register()-populated GLOBAL_CONFIG — see
third_party/DEIMv2/engine/data/transforms/_transforms.py and
third_party/D-FINE/src/data/transforms/_transforms.py for the pattern this
mirrors, e.g. ConvertPILImage).

Reusing albumentations.CLAHE (not reimplementing with cv2 directly) matters
for two reasons: it's already proven color-safe (converts to LAB, enhances
only the L channel, leaves hue/saturation — i.e. the dual-energy X-ray
material-color signal — untouched), and it guarantees byte-identical CLAHE
behavior across all 3 models, which is the whole point of having one
augmentation source of truth (CLAUDE.md hard rule #4).

This is a factory, not a class, because DEIMv2 and D-FINE each have their
own separate `register()`/GLOBAL_CONFIG — the same class object can't be
registered into two independent registries, so each wrapper entrypoint
calls this once with its own repo's `register` decorator.
"""
import sys

import albumentations as A
import numpy as np
from PIL import Image as PILImage


def make_clahe_transform_class(register, transform_base_cls, transformed_types):
    """
    Args:
        register: the target repo's engine.core.register (or src.core.register)
        transform_base_cls: that repo's torchvision.transforms.v2.Transform
            (same underlying torchvision class in practice, imported from
            whichever `T` module the caller already has)
        transformed_types: tuple of types this op should apply to, e.g.
            (PIL.Image.Image,) — matches ConvertPILImage's convention
    """

    @register()
    class CLAHE(transform_base_cls):
        _transformed_types = transformed_types

        def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
            super().__init__()
            self._clahe = A.CLAHE(
                clip_limit=clip_limit,
                tile_grid_size=(tile_grid_size, tile_grid_size),
                p=1.0,
            )

        def transform(self, inpt, params):
            # Newer torchvision (>=0.24-ish, e.g. D-FINE's env pulled 0.27.1
            # via its unpinned requirements.txt) renamed the abstract hook
            # from `_transform` to `transform`. DEIMv2's env has an older
            # torchvision (0.20.1) where only `_transform` exists and is
            # called directly, never `transform`. Defining both — `transform`
            # delegating to `_transform` — matches the shim D-FINE's own
            # ConvertPILImage/ConvertBoxes use, and works on both versions.
            return self._transform(inpt, params)

        def _transform(self, inpt, params):
            arr = np.array(inpt.convert("RGB"))
            out = self._clahe(image=arr)["image"]
            return PILImage.fromarray(out)

    # register()'s introspection later resolves the class via
    # getattr(importlib.import_module(cls.__module__), cls.__name__) — that
    # requires CLAHE to be a genuine top-level attribute of this module, but
    # a class defined inside this factory function is only a local variable.
    # Export it explicitly so the lookup succeeds.
    setattr(sys.modules[CLAHE.__module__], CLAHE.__name__, CLAHE)

    return CLAHE
