from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load


_EXT: Optional[object] = None


def load_ext(verbose: bool = False):
    global _EXT
    if _EXT is not None:
        return _EXT

    exp_dir = Path(__file__).resolve().parents[1]
    src = exp_dir / "cpp_ext" / "src" / "ahc061_ext.cpp"
    include_dir = exp_dir / "cpp_core" / "include"
    build_dir = exp_dir / "artifacts" / "torch_ext_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    extra_cflags = [
        "-O3",
        "-march=native",
        "-std=c++20",
        "-DNDEBUG",
    ]
    extra_ldflags: list[str] = []

    _EXT = load(
        name="ahc061_exp001_cpp_v3",
        sources=[str(src)],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=[str(include_dir)],
        build_directory=str(build_dir),
        verbose=verbose,
        with_cuda=False,
    )
    return _EXT
