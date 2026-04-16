"""Config loading utilities — env-var expansion + path normalisation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` and ``$VAR`` in strings inside a config tree.

    Parameters
    ----------
    value : Any
        A scalar, list, or dict from a YAML config.

    Returns
    -------
    Any
        Same structure with strings expanded. Unset variables are left in
        place (no error) so configs can declare optional placeholders.
    """
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            name = m.group(1) or m.group(2)
            return os.environ.get(name, m.group(0))
        return _ENV_VAR_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def load_config(path: str | Path) -> dict:
    """Load a YAML config and expand ``$HOME``-style references in strings."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return expand_env(cfg)
