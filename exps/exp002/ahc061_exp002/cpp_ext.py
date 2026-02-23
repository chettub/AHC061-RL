from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from torch.utils.cpp_extension import load


_EXT: Optional[object] = None


def load_ext(*, verbose: bool = False):
    global _EXT
    if _EXT is not None:
        return _EXT

    pf_particles = int(os.environ.get("AHC061_PF_PARTICLES", "16"))
    if pf_particles <= 0:
        raise ValueError(f"AHC061_PF_PARTICLES must be >= 1: {pf_particles}")

    exp_dir = Path(__file__).resolve().parents[1]
    src = exp_dir / "cpp_ext" / "src" / "ahc061_ext.cpp"
    include_dir = exp_dir / "cpp_core" / "include"
    build_dir = exp_dir / "artifacts" / f"torch_ext_build_pf{pf_particles}"
    build_dir.mkdir(parents=True, exist_ok=True)

    extra_cflags = [
        "-O3",
        "-march=native",
        "-std=c++20",
        "-DNDEBUG",
        f"-DAHC061_PF_PARTICLES={pf_particles}",
    ]
    extra_ldflags: list[str] = []

    _EXT = load(
        name=f"ahc061_exp002_cpp_pf{pf_particles}_v1",
        sources=[str(src)],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=[str(include_dir)],
        build_directory=str(build_dir),
        verbose=verbose,
        with_cuda=False,
    )
    return _EXT
