from typing import TYPE_CHECKING

from transformers.utils import _LazyModule
from transformers.utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .callbacks import *
    from .trainer import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
