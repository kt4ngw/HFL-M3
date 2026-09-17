# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
"""Python interface for the optional HEonGPU BFV Gram backend."""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np


_INPUT_HEADER = struct.Struct("<8sIII")
_OUTPUT_HEADER = struct.Struct("<8sII")
_INPUT_MAGIC = b"HFLMGRM1"
_OUTPUT_MAGIC = b"HFLMGMT1"
_FORMAT_VERSION = 1
_METADATA_PREFIX = "HFLM_GPU_JSON "


def default_heongpu_executable():
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "build" / "heongpu" / "private_gram_gpu"


def compute_private_gram_heongpu(
    histograms,
    *,
    executable=None,
    poly_modulus_degree=8192,
    plain_modulus=33832961,
    cuda_streams=16,
    batch_size=128,
):
    """Encrypt client histograms and recover only their Gram matrix on GPU.

    The native executable emulates the client, evaluator, and selector roles
    in one process while clearing its plaintext histogram buffer before the
    evaluator stage.  It is intended for reproducible protocol benchmarking;
    a deployment can use the same HEonGPU serialization format across separate
    role processes.
    """
    histograms = np.asarray(histograms)
    if histograms.ndim != 2 or min(histograms.shape) <= 0:
        raise ValueError("histograms must be a non-empty two-dimensional matrix")
    if not np.issubdtype(histograms.dtype, np.integer):
        if not np.all(np.equal(histograms, np.floor(histograms))):
            raise ValueError("histograms must contain exact integer counts")
    if np.any(histograms < 0):
        raise ValueError("histograms cannot contain negative values")
    histograms = np.ascontiguousarray(histograms, dtype="<u8")
    if (int(plain_modulus) - 1) % (2 * int(poly_modulus_degree)) != 0:
        raise ValueError(
            "plain_modulus must be congruent to 1 modulo "
            "2*poly_modulus_degree for BFV batching"
        )

    if executable is None:
        executable = os.environ.get(
            "HFLM_HEONGPU_GRAM_BIN",
            str(default_heongpu_executable()),
        )
    executable = Path(executable).expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(
            "HEonGPU Gram executable not found at {}. Run "
            "scripts/tools/build_heongpu_backend.sh first.".format(executable)
        )

    num_clients, num_classes = histograms.shape
    with tempfile.TemporaryDirectory(prefix="hflm_heongpu_") as temp_dir:
        input_path = Path(temp_dir) / "histograms.bin"
        output_path = Path(temp_dir) / "gram.bin"
        with input_path.open("wb") as handle:
            handle.write(
                _INPUT_HEADER.pack(
                    _INPUT_MAGIC,
                    _FORMAT_VERSION,
                    int(num_clients),
                    int(num_classes),
                )
            )
            histograms.tofile(handle)

        command = [
            str(executable),
            str(input_path),
            str(output_path),
            str(int(poly_modulus_degree)),
            str(int(plain_modulus)),
            str(int(cuda_streams)),
            str(int(batch_size)),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "HEonGPU private-Gram backend failed (exit {}):\n{}".format(
                    completed.returncode,
                    (completed.stderr or completed.stdout).strip(),
                )
            )

        metadata_line = next(
            (
                line[len(_METADATA_PREFIX):]
                for line in reversed(completed.stdout.splitlines())
                if line.startswith(_METADATA_PREFIX)
            ),
            None,
        )
        if metadata_line is None:
            raise RuntimeError("HEonGPU backend did not emit runtime metadata")
        metadata = json.loads(metadata_line)

        with output_path.open("rb") as handle:
            header = handle.read(_OUTPUT_HEADER.size)
            if len(header) != _OUTPUT_HEADER.size:
                raise RuntimeError("HEonGPU Gram output is truncated")
            magic, version, output_clients = _OUTPUT_HEADER.unpack(header)
            if magic != _OUTPUT_MAGIC or version != _FORMAT_VERSION:
                raise RuntimeError("HEonGPU Gram output has an invalid header")
            if output_clients != num_clients:
                raise RuntimeError("HEonGPU Gram output client count is incorrect")
            gram = np.fromfile(
                handle,
                dtype="<u8",
                count=num_clients * num_clients,
            )
            if gram.size != num_clients * num_clients:
                raise RuntimeError("HEonGPU Gram output matrix is truncated")
            if handle.read(1):
                raise RuntimeError("HEonGPU Gram output contains trailing data")

    if gram.size and gram.max() > np.iinfo(np.int64).max:
        raise OverflowError("decrypted Gram value does not fit int64")
    return gram.reshape(num_clients, num_clients).astype(np.int64), metadata
