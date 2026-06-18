from typing import TYPE_CHECKING

try:
    from transformers.utils import _LazyModule
    from transformers.utils.import_utils import define_import_structure
except Exception:  # pragma: no cover
    _LazyModule = None  # type: ignore
    define_import_structure = None  # type: ignore


if TYPE_CHECKING:
    from .data_loading import *
    from .preprocessing import *
else:
    import sys

    if _LazyModule is not None and define_import_structure is not None:
        _file = globals()["__file__"]
        sys.modules[__name__] = _LazyModule(
            __name__, _file, define_import_structure(_file), module_spec=__spec__
        )
