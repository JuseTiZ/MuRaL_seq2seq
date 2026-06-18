"""Compatibility package for importing the repository as ``mural_s2s``.

The source modules live at the repository root.  Adding that directory to this
package's search path lets entry-point scripts use stable ``mural_s2s.*``
imports regardless of the checkout directory name.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in __path__:
    __path__.append(str(_ROOT))
