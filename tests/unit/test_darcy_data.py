"""
Unit tests for the Darcy experiment data pipeline.

Regression: subsampling a 241-grid to resolution 64 used stride
241//64 = 3 then truncated 81 → 64 points, silently cropping the
domain to [0, 0.7875] per axis. The model then had to predict u
from a permeability field it only partially saw — an irreducible
error floor (~3-4% vs the 1.08% paper target).

subsample_grid must cover the FULL domain (first and last grid
point kept) or raise.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "darcy_fno2d",
    Path(__file__).resolve().parents[2] / "experiments" / "darcy_fno2d.py",
)
_darcy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_darcy)

subsample_grid = _darcy.subsample_grid


class TestSubsampleGrid:
    @pytest.mark.parametrize("resolution", [61, 81, 121, 241])
    def test_full_domain_coverage(self, resolution):
        """First and last grid points must survive subsampling."""
        n = 241
        field = np.zeros((2, n, n), dtype=np.float32)
        field[:, 0, 0] = 1.0
        field[:, -1, -1] = 2.0

        sub = subsample_grid(field, resolution)

        assert sub.shape == (2, resolution, resolution)
        assert sub[0, 0, 0] == 1.0, "First grid point lost"
        assert sub[0, -1, -1] == 2.0, "Last grid point lost — domain was cropped"

    def test_rejects_cropping_resolution(self):
        """resolution=64 on a 241-grid cannot cover the full domain."""
        field = np.zeros((1, 241, 241), dtype=np.float32)
        with pytest.raises(ValueError, match="does not evenly subsample"):
            subsample_grid(field, 64)

    def test_error_message_lists_valid_resolutions(self):
        field = np.zeros((1, 241, 241), dtype=np.float32)
        with pytest.raises(ValueError, match="61"):
            subsample_grid(field, 64)
