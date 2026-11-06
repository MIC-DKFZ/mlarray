"""Curated unit names and their UCUM codes (MLArray spec section 4.5).

Writers should use a curated name and set its ``unit_code``; free text is
allowed when no listed unit fits.
"""

import json
from functools import lru_cache
from importlib import resources
from typing import Dict, Optional


@lru_cache(maxsize=1)
def curated_units() -> Dict[str, str]:
    """Map of curated unit name to UCUM code."""
    data = json.loads(resources.files("mlarray").joinpath("schemas/units-1.0.json").read_text())
    return {unit["name"]: unit["unit_code"] for unit in data["units"]}


def unit_code(unit: Optional[str]) -> Optional[str]:
    """UCUM code of a curated unit name, or None for free-text or missing units."""
    if unit is None:
        return None
    return curated_units().get(unit)
