from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import torch

from ..env import BatchEnv, tools_input_paths


def run_official_tester(tools_input: str) -> int:
    repo_root = Path(__file__).resolve().parents[4]
    tester = repo_root / "tools" / "target" / "release" / "tester"
    solver = repo_root / "exps" / "exp001" / "submit" / "always_stay.py"
    venv_python = repo_root / ".venv" / "bin" / "python"
    python_cmd = str(venv_python) if venv_python.exists() else "python"
    cmd = [str(tester), python_cmd, str(solver)]
    with open(tools_input, "rb") as f:
        proc = subprocess.run(cmd, stdin=f, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    m = re.search(rb"^\[DATA\]\s+score\s*=\s*(\d+)\s*$", proc.stderr, re.MULTILINE)
    if not m:
        raise RuntimeError(f"failed to parse score from tester stderr:\n{proc.stderr.decode('utf-8', errors='replace')}")
    return int(m.group(1))


def run_ours(tools_input: str) -> int:
    env = BatchEnv(batch_size=1, pf_enabled=False, verbose_build=False)
    env.reset_from_tools([tools_input])
    for _ in range(env.spec.t_max):
        a = env.pos0()
        env.step(a)
    return int(env.official_score()[0].item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-begin", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=2)
    args = parser.parse_args()

    paths = tools_input_paths(args.seed_begin, args.seed_end)
    for p in paths:
        off = run_official_tester(p)
        ours = run_ours(p)
        print(f"{Path(p).name}: official={off} ours={ours}")
        if off != ours:
            raise SystemExit(1)

    print("OK")


if __name__ == "__main__":
    main()
