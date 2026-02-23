from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import torch
import torch.nn as nn

from ..ckpt import model_spec_from_ckpt, normalize_state_dict_keys, torch_load_maybe_weights_only
from ..models import build_policy_value_model

# Safe for C++ string literals and avoids trigraph warnings: exclude '"', '\\', and '?'.
_BASE91_ALPHABET = "".join(chr(c) for c in range(33, 127) if c not in (34, 63, 92))
_BASE91_ENC = _BASE91_ALPHABET
_BASE91_DEC = {ch: i for i, ch in enumerate(_BASE91_ALPHABET)}
if len(_BASE91_ALPHABET) != 91:
    raise RuntimeError("internal error: base91 alphabet size must be 91")


class PolicyOnly(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        # Export only the policy trunk + head so the submission does not embed value-head weights.
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_policy"):
            return self.model.forward_policy(x)
        x = self.model.stem(x)
        x = self.model.blocks(x)
        return self.model.policy_head(x).flatten(1)


def _encode_base91(data: bytes) -> str:
    b = 0
    n = 0
    out: list[str] = []
    for x in data:
        b |= x << n
        n += 8
        if n > 13:
            v = b & 8191
            if v > 88:
                b >>= 13
                n -= 13
            else:
                v = b & 16383
                b >>= 14
                n -= 14
            out.append(_BASE91_ENC[v % 91])
            out.append(_BASE91_ENC[v // 91])
    if n:
        out.append(_BASE91_ENC[b % 91])
        if n > 7 or b > 90:
            out.append(_BASE91_ENC[b // 91])
    return "".join(out)


def _decode_base91(text: str) -> bytes:
    v = -1
    b = 0
    n = 0
    out = bytearray()
    for ch in text:
        d = _BASE91_DEC.get(ch, -1)
        if d < 0:
            continue
        if v < 0:
            v = d
            continue
        v += d * 91
        b |= v << n
        n += 13 if (v & 8191) > 88 else 14
        while n > 7:
            out.append(b & 0xFF)
            b >>= 8
            n -= 8
        v = -1
    if v >= 0:
        out.append((b | (v << n)) & 0xFF)
    return bytes(out)


def _encode_model_payload_base91(ts_bytes: bytes) -> str:
    payload = _encode_base91(ts_bytes)
    if _decode_base91(payload) != ts_bytes:
        raise RuntimeError("[ENC] base91 roundtrip failed")
    return payload


def _write_encoded_inc(out_path: Path, *, encoded_payload: str, meta: dict) -> None:
    if ('"' in encoded_payload) or ("\\" in encoded_payload):
        raise RuntimeError('[ENC] payload contains C++-unsafe characters: " or \\')
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
    lines.append("static const char kModelTsEncoded[] =\n")
    chunk = 16384
    for i in range(0, len(encoded_payload), chunk):
        lines.append(f"    \"{encoded_payload[i:i+chunk]}\"\n")
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
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="exps/exp002/submit")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    ckpt_path = (repo_root / args.ckpt).resolve() if not Path(args.ckpt).is_absolute() else Path(args.ckpt).resolve()
    out_dir = (repo_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch_load_maybe_weights_only(ckpt_path)
    ms = model_spec_from_ckpt(ckpt)
    in_channels = int(ms.in_channels)
    hidden = int(ms.hidden)
    blocks = int(ms.blocks)
    arch_name = str(ms.arch_name)
    feature_id = str(ms.feature_id)

    def feature_cpp_header(fid: str) -> str:
        if fid == "submit_v1":
            return "ahc061/core/features.hpp"
        return f"ahc061/core/features_{fid}.hpp"

    def feature_channels_expr(fid: str) -> str:
        if fid == "submit_v1":
            return "FEATURE_C"
        return "FEATURE_" + fid.upper() + "_C"

    def feature_writer_name(fid: str) -> str:
        if fid == "submit_v1":
            return "write_features_submit_v1_from_common"
        return f"write_features_{fid}_from_common"

    model = build_policy_value_model(
        arch_name,
        in_channels=in_channels,
        hidden_channels=hidden,
        blocks=blocks,
        feature_id=feature_id,
        arch_kwargs=dict(ms.arch_kwargs),
    ).to("cpu")
    missing, unexpected = model.load_state_dict(normalize_state_dict_keys(ckpt["model"]), strict=False)
    if unexpected:
        raise RuntimeError(f"[CKPT] unexpected keys in state_dict: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    allowed_missing_prefixes = ("opp_move_head.", "opp_param_head.")
    bad_missing = [k for k in missing if not any(k.startswith(p) for p in allowed_missing_prefixes)]
    if bad_missing:
        raise RuntimeError(f"[CKPT] missing keys in state_dict: {bad_missing[:5]}{' ...' if len(bad_missing) > 5 else ''}")
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
    encoded_payload = _encode_model_payload_base91(ts_bytes)

    meta = {
        "arch_name": arch_name,
        "feature_id": feature_id,
        "in_channels": in_channels,
        "hidden": hidden,
        "blocks": blocks,
        "ckpt_upd": int(ckpt.get("upd", -1)),
        "torch_version": torch.__version__,
    }

    inc_path = out_dir / "model_ts_encoded.inc"
    _write_encoded_inc(inc_path, encoded_payload=encoded_payload, meta=meta)
    legacy_inc = out_dir / "model_ts_base64.inc"
    if legacy_inc.exists():
        legacy_inc.unlink()

    # Feature-specific config (header + channel count + writer function).
    feature_header = feature_cpp_header(feature_id)
    channels_expr = feature_channels_expr(feature_id)
    writer_name = feature_writer_name(feature_id)
    feature_inc = out_dir / "feature_config.inc"
    feature_inc.write_text(
        "".join(
            [
                "// GENERATED FILE. DO NOT EDIT.\n",
                "#pragma once\n",
                "\n",
                f'#include "{feature_header}"\n',
                "\n",
                "namespace ahc061::exp002 {\n",
                f"static constexpr int FEATURE_INPUT_C = {channels_expr};\n",
                f"inline void write_features_from_common(const FeatureCommon& common, float* out_board) {{\n",
                f"    {writer_name}(common, out_board);\n",
                "}\n",
                "}  // namespace ahc061::exp002\n",
            ]
        ),
        encoding="utf-8",
    )

    template_dir = repo_root / "exps" / "exp002" / "submit"
    entry = out_dir / "solver_base.cpp"
    entry.write_text((template_dir / "solver_base.cpp").read_text(encoding="utf-8"), encoding="utf-8")
    include_dirs = [
        out_dir,
        repo_root / "exps" / "exp002" / "cpp_core" / "include",
    ]
    bundled = _bundle_cpp(entry, include_dirs=include_dirs)
    main_cpp = out_dir / "Main.cpp"
    main_cpp.write_text(bundled, encoding="utf-8")

    print(f"[ENC] raw={len(ts_bytes)} b91={len(encoded_payload)} selected=base91")
    print(f"[OK] wrote: {feature_inc}")
    print(f"[OK] wrote: {inc_path}")
    print(f"[OK] wrote: {main_cpp}")


if __name__ == "__main__":
    main()
