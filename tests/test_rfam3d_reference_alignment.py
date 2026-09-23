from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "03_token_alignment"
    / "rfam3d_reference_alignment.py"
)
SPEC = importlib.util.spec_from_file_location("rfam3d_reference_alignment", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_contiguous_family_region_retains_unmapped_internal_positions() -> None:
    region = MODULE.contiguous_family_region({10: 2, 11: 3, 14: 6}, 9)

    np.testing.assert_array_equal(region, np.asarray([2, 3, 4, 5, 6]))


def test_contiguous_family_region_is_defined_per_chain() -> None:
    first = MODULE.contiguous_family_region({10: 1, 14: 5}, 8)
    second = MODULE.contiguous_family_region({10: 3, 14: 9}, 12)

    np.testing.assert_array_equal(first, np.arange(1, 6))
    np.testing.assert_array_equal(second, np.arange(3, 10))


def test_contiguous_family_region_rejects_invalid_mapping() -> None:
    with pytest.raises(ValueError, match="outside sequence"):
        MODULE.contiguous_family_region({10: 2, 14: 8}, 8)
