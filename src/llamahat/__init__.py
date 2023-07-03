try:
    from ._version import (
        version as __version__,
        version_tuple,
    )
except ImportError:
    __version__ = "unknown (no version information available)"
    version_tuple = (0, 0, "unknown", "noinfo")

import torch as _torch  # noqa: F401
from rich.console import Console

console = Console(highlight=True)
err_console = Console(stderr=True)
