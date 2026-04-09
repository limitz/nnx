# NNX Coding Style Reference

## Signature Pattern: Underscore-prefixed imports

Every external import is aliased with a leading underscore — no exceptions.
This keeps the public namespace clean and makes library calls visually distinct from local code.

```python
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import numpy as _np
import cv2 as _cv2
import math as _math
import matplotlib.pyplot as _plt
```

Internal/relative imports follow the same rule:

```python
from . import functional as _Fx
from .. import functional as _Fx
```

---

## Naming Conventions

| Element | Convention | Example |
|---|---|---|
| External imports | `_underscore` prefix | `import math as _math` |
| Functions | `snake_case` | `pad_to()`, `crop_view()`, `bit_encode()` |
| Classes | `PascalCase` | `TensorFunction`, `StaticConv2d`, `EMA` |
| Local variables | lowercase | `x`, `idx`, `pad_value` |
| Constants | `UPPERCASE` | `TRANSPARENT`, `DEFAULT_PALETTE`, `NUMERIC_5X3` |

---

## Design Patterns

### Functional-first
Modules expose functions over classes where possible. Classes exist to wrap functionality into `nn.Module`-compatible interfaces, not to encapsulate state.

### Wrapper pattern
`TensorFunction`, `ModuleFunction`, and `FxFunction` wrap operations as `nn.Module` subclasses without deep inheritance chains. Subclasses are typically created with `...` (ellipsis) as the body:

```python
class Select(TensorFunction): ...
class Permute(TensorFunction): ...
class Reshape(TensorFunction): ...
```

### Composition over inheritance
Classes wrap rather than extend. Prefer passing a module or function as a parameter over subclassing.

---

## Error Handling

Assertions over exceptions:

```python
assert split in {"train", "eval"}
assert x.dtype in {_torch.uint8, _torch.long}
```

---

## Documentation Style

- Minimal comments — code is self-documenting through descriptive parameter names
- No docstrings
- Sparse inline comments only for non-obvious logic
- `# todo` markers (lowercase) for deferred work

---

## Notable Specifics

- **Custom terminal styling** (`console.py`): ANSI color system with `<color>text</>` syntax, `TrueColor` registry, and a custom `print` override
- **Context manager for determinism**: `with determinism(seed):` blocks for reproducible execution
- **`Lambda` module**: evaluates lambda strings with opt-in `unsafe=True` flag
- **Bit manipulation**: `bit_encode()` / `bit_decode()` for LSB steganography operations
- **Complex number support**: uses `_torch.cfloat` / `_torch.cdouble` dtypes
- **`guess_device()`**: recursively infers device from inputs rather than requiring explicit device args

---

## File / Module Structure

```
nnx/
├── __init__.py
├── modules.py          - PyTorch nn.Module wrappers
├── functional.py       - Functional utilities and helpers
├── console.py          - Terminal styling and UI components
├── determinism.py      - Reproducibility utilities
├── distributed.py      - Distributed training (autograd.Function wrappers)
├── projection.py       - Geometric projection functions
├── stegano.py          - Steganography utilities
├── data/
│   ├── widerface.py    - WIDERface dataset loader
│   └── random.py
├── graphics/
│   ├── functional.py   - Image processing utilities
│   ├── color.py        - Color space conversion and generation
│   ├── image.py
│   ├── text.py         - Text rendering (FreeType)
│   ├── audio.py
│   ├── video.py
│   ├── pattern.py
│   ├── polygon.py
│   ├── maze.py
│   └── sudoku.py
└── metrics/
    ├── psnr.py
    └── tests.py
```

---

*Generated from codebase analysis — update if conventions change.*
