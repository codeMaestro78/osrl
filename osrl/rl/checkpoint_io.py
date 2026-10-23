"""Portable SB3 checkpoint writing without changing third-party loaders."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path


def save_model(model, path: Path) -> None:
    """Atomically save an SB3 archive using supported legacy torch payloads.

    PyTorch 2.14.1's ZIP reader fails on unbuffered ZipExtFile streams in
    SB3.load, while the same payload loads from BytesIO/BufferedReader.
    Legacy tensor serialization avoids nested ZIP streams and remains
    supported by torch.load(weights_only=True). No global monkeypatches or
    changes to the user's existing checkpoints are needed.
    """
    import torch

    buffer = io.BytesIO()
    model.save(buffer)
    temporary = path.with_suffix(".tmp.zip")
    try:
        with zipfile.ZipFile(buffer) as source, zipfile.ZipFile(temporary, "w") as target:
            for name in source.namelist():
                raw = source.read(name)
                if name.endswith(".pth"):
                    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                    tensor_buffer = io.BytesIO()
                    torch.save(value, tensor_buffer, _use_new_zipfile_serialization=False)
                    raw = tensor_buffer.getvalue()
                target.writestr(name, raw)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
