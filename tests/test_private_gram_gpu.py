# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
"""Native HEonGPU exactness check; requires a built adapter and an NVIDIA GPU."""
import os
from pathlib import Path

import numpy as np
import pytest

from src.virtualset.private_gram_gpu import (
    compute_private_gram_heongpu,
    default_heongpu_executable,
)


def test_native_heongpu_recovers_exact_gram():
    executable = Path(os.environ.get(
        'HFLM_HEONGPU_GRAM_BIN', str(default_heongpu_executable())
    )).expanduser()
    if not executable.is_file():
        pytest.skip('Build the HEonGPU adapter to run the native GPU test')
    histograms = np.array([[7, 2, 1], [1, 7, 2], [2, 1, 7]], dtype=np.int64)
    gram, metadata = compute_private_gram_heongpu(histograms, executable=executable)
    np.testing.assert_array_equal(gram, histograms @ histograms.T)
    assert metadata['encrypted_gram_entries'] == 6
