from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from torch.utils.cpp_extension import load


_EXT_BY_PF_PARTICLES: dict[int, Any] = {}


def _find_eigen_include_dir() -> str | None:
    env = os.environ.get("EIGEN3_INCLUDE_DIR")
    if env:
        p = Path(env).expanduser()
        if (p / "Eigen" / "Dense").exists():
            return str(p.resolve())

    candidates = [
        Path("/usr/include/eigen3"),
        Path("/usr/local/include/eigen3"),
        Path("/opt/homebrew/include/eigen3"),
    ]
    for p in candidates:
        if (p / "Eigen" / "Dense").exists():
            return str(p)

    # Some distros may install directly under /usr/include/Eigen.
    if (Path("/usr/include") / "Eigen" / "Dense").exists():
        return "/usr/include"
    if (Path("/usr/local/include") / "Eigen" / "Dense").exists():
        return "/usr/local/include"

    return None


def load_ext(*, pf_particles: int, verbose: bool = False):
    if pf_particles in _EXT_BY_PF_PARTICLES:
        return _EXT_BY_PF_PARTICLES[pf_particles]

    if pf_particles <= 0:
        raise ValueError(f"pf_particles must be >= 1: {pf_particles}")

    exp_dir = Path(__file__).resolve().parents[1]
    src = exp_dir / "cpp_ext" / "src" / "ahc061_ext.cpp"
    include_dir = exp_dir.parent / "exp002" / "cpp_core" / "include"
    build_dir = exp_dir / "artifacts" / f"torch_ext_build_pf{pf_particles}"
    build_dir.mkdir(parents=True, exist_ok=True)

    extra_cflags = [
        "-O3",
        "-march=native",
        "-std=c++20",
        "-DNDEBUG",
        f"-DAHC061_PF_PARTICLES={pf_particles}",
        "-DEIGEN_NO_DEBUG",
    ]
    extra_ldflags: list[str] = []

    extra_include_paths = [str(include_dir)]
    eigen_inc = _find_eigen_include_dir()
    if eigen_inc is not None:
        extra_include_paths.append(str(eigen_inc))

    ext = load(
        name=f"ahc061_exp003_cpp_pf{pf_particles}_v4",
        sources=[str(src)],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=str(build_dir),
        verbose=verbose,
        with_cuda=False,
    )
    _EXT_BY_PF_PARTICLES[pf_particles] = ext
    return ext
