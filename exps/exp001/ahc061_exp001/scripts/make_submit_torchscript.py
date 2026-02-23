from __future__ import annotations

import argparse
import base64
import io
import re
from pathlib import Path

import torch
import torch.nn as nn

from ..models import build_policy_value_model, normalize_state_dict_keys


class PolicyOnly(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        # Export only the policy trunk + head so the submission does not embed value-head weights.
        self.stem = model.stem
        self.blocks = model.blocks
        self.policy_head = model.policy_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        logits = self.policy_head(x).flatten(1)
        return logits


def _write_base64_inc(out_path: Path, *, ts_bytes: bytes, meta: dict) -> None:
    b64 = base64.b64encode(ts_bytes).decode("ascii")
    lines: list[str] = []
    lines.append("// GENERATED FILE. DO NOT EDIT.\n")
    lines.append("#pragma once\n")
    lines.append("#include <cstddef>\n")
    lines.append("\n")
    for k, v in meta.items():
        if isinstance(v, str):
            lines.append(f'static constexpr const char* kModelMeta_{k} = "{v}";\n')
        else:
            lines.append(f"static constexpr std::size_t kModelMeta_{k} = {int(v)};\n")
    lines.append("\n")
    lines.append("static const char kModelTsBase64[] =\n")
    chunk = 16384
    for i in range(0, len(b64), chunk):
        lines.append(f"    \"{b64[i:i+chunk]}\"\n")
    lines.append("    ;\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def _bundle_cpp(entry: Path, *, include_dirs: list[Path]) -> str:
    include_re = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')
    seen: set[Path] = set()

    def strip_trailing_line_comment(line: str) -> str:
        # Remove trailing // comment not in string/char literals.
        # Keep the original newline (if any).
        keep_nl = line.endswith("\n")
        s = line[:-1] if keep_nl else line
        in_str = False
        in_chr = False
        esc = False
        for i in range(len(s) - 1):
            ch = s[i]
            if esc:
                esc = False
                continue
            if in_str:
                if ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if in_chr:
                if ch == "\\":
                    esc = True
                elif ch == "'":
                    in_chr = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "'":
                in_chr = True
                continue
            if ch == "/" and s[i + 1] == "/":
                s2 = s[:i].rstrip()
                return (s2 + "\n") if keep_nl else s2
        return line

    def should_drop_line(line: str, *, in_block_comment: bool) -> tuple[bool, bool]:
        s = line.lstrip()
        if in_block_comment:
            if "*/" in s:
                return True, False
            return True, True
        if not s.strip():
            return True, False
        if s.startswith("//"):
            return True, False
        if s.startswith("/*"):
            if "*/" in s:
                return True, False
            return True, True
        return False, False

    def resolve_include(cur: Path, name: str) -> Path | None:
        cand = (cur.parent / name).resolve()
        if cand.is_file():
            return cand
        for d in include_dirs:
            cand = (d / name).resolve()
            if cand.is_file():
                return cand
        return None

    def rec(path: Path) -> list[str]:
        out: list[str] = []
        in_block_comment = False
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
            if line.strip() == "#pragma once":
                continue
            drop, in_block_comment = should_drop_line(line, in_block_comment=in_block_comment)
            if drop:
                continue
            line = strip_trailing_line_comment(line)
            line = line.lstrip(" \t")
            m = include_re.match(line)
            if not m:
                out.append(line)
                continue
            name = m.group(1)
            inc = resolve_include(path, name)
            if inc is None:
                out.append(line)
                continue
            if inc in seen:
                continue
            seen.add(inc)
            out.extend(rec(inc))
        return out

    return "".join(rec(entry.resolve()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="exps/exp001/artifacts/checkpoints/ckpt_0400.pt")
    parser.add_argument("--out-dir", type=str, default="exps/exp001/submit")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    ckpt_path = (repo_root / args.ckpt).resolve() if not Path(args.ckpt).is_absolute() else Path(args.ckpt).resolve()
    out_dir = (repo_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    in_channels = int(ckpt["in_channels"])
    hidden = int(ckpt["hidden"])
    blocks = int(ckpt["blocks"])
    arch_name = str(ckpt.get("arch_name", "resnet_v1"))

    model = build_policy_value_model(
        arch_name,
        in_channels=in_channels,
        hidden_channels=hidden,
        blocks=blocks,
    ).to("cpu")
    model.load_state_dict(normalize_state_dict_keys(ckpt["model"]))
    model.eval()

    wrapper = PolicyOnly(model)
    wrapper.eval()
    example = torch.zeros((1, in_channels, 10, 10), dtype=torch.float32, device="cpu")
    with torch.inference_mode():
        ts = torch.jit.trace(wrapper, example)
        ts = torch.jit.freeze(ts)
        buf = io.BytesIO()
        torch.jit.save(ts, buf)
        ts_bytes = buf.getvalue()

    meta = {
        "arch_name": arch_name,
        "in_channels": in_channels,
        "hidden": hidden,
        "blocks": blocks,
        "ckpt_upd": int(ckpt.get("upd", -1)),
        "torch_version": torch.__version__,
    }

    inc_path = out_dir / "model_ts_base64.inc"
    _write_base64_inc(inc_path, ts_bytes=ts_bytes, meta=meta)

    template_dir = repo_root / "exps" / "exp001" / "submit"
    entry = out_dir / "solver_base.cpp"
    if not entry.is_file():
        entry.write_text((template_dir / "solver_base.cpp").read_text(encoding="utf-8"), encoding="utf-8")
    include_dirs = [
        out_dir,
        repo_root / "exps" / "exp001" / "cpp_core" / "include",
    ]
    bundled = _bundle_cpp(entry, include_dirs=include_dirs)
    main_cpp = out_dir / "Main.cpp"
    main_cpp.write_text(bundled, encoding="utf-8")

    print(f"[OK] wrote: {inc_path}")
    print(f"[OK] wrote: {main_cpp}")


if __name__ == "__main__":
    main()
