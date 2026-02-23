from __future__ import annotations

import argparse
import heapq
import io
import re
import textwrap
from string import Template
from pathlib import Path

import numpy as np
import torch

from ..ckpt import model_spec_from_ckpt, normalize_state_dict_keys, torch_load_maybe_weights_only
from ..models import build_policy_value_model

# Compact alphabet for huff91/base91 payloads.
# We use almost all ASCII control+printable bytes except '\n' and '\r'
# (line separators in codegen/bundle pipeline).
_BASE91_EXTRA_CODES = tuple(c for c in range(1, 128) if c not in (10, 13))
_BASE91_ALPHABET = "".join(chr(c) for c in _BASE91_EXTRA_CODES)
_BASE91_ENC = _BASE91_ALPHABET
_BASE91_DEC = {ch: i for i, ch in enumerate(_BASE91_ALPHABET)}
_BASE91_RADIX = len(_BASE91_ALPHABET)
_BASE91_EXTRA = _BASE91_RADIX * _BASE91_RADIX - 8192  # number of 14-bit slots
if len(_BASE91_ALPHABET) != 125:
    raise RuntimeError("internal error: base91 alphabet size must be 125")
if not (0 <= _BASE91_EXTRA < 8192):
    raise RuntimeError("internal error: invalid base91 extra range")

# Base122 illegal set (+ '?' to avoid trigraph conversion in C++ source).
_BASE122_ILLEGALS = (0, 10, 13, 34, 38, 63, 92)
_BASE122_ILLEGAL_TO_INDEX = {v: i for i, v in enumerate(_BASE122_ILLEGALS)}
_BASE122_SHORTENED = 0b111

_PAYLOAD_CODEC_BASE91 = 91
_PAYLOAD_CODEC_BASE122 = 122
_PAYLOAD_CODEC_HUFF122 = 1122
_PAYLOAD_CODEC_HUFF91 = 1091


def _encode_base91(data: bytes) -> str:
    b = 0
    n = 0
    out: list[str] = []
    for x in data:
        b |= x << n
        n += 8
        if n > 13:
            v = b & 8191
            if v < _BASE91_EXTRA:
                v = b & 16383
                b >>= 14
                n -= 14
            else:
                b >>= 13
                n -= 13
            out.append(_BASE91_ENC[v % _BASE91_RADIX])
            out.append(_BASE91_ENC[v // _BASE91_RADIX])
    if n:
        out.append(_BASE91_ENC[b % _BASE91_RADIX])
        if n > 7 or b >= _BASE91_RADIX:
            out.append(_BASE91_ENC[b // _BASE91_RADIX])
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
        v += d * _BASE91_RADIX
        b |= v << n
        n += 14 if (v & 8191) < _BASE91_EXTRA else 13
        while n > 7:
            out.append(b & 0xFF)
            b >>= 8
            n -= 8
        v = -1
    if v >= 0:
        out.append((b | (v << n)) & 0xFF)
    return bytes(out)


def _encode_base122(data: bytes) -> str:
    cur_index = 0
    cur_bit = 0
    n = len(data)
    out = bytearray()

    def get7() -> int | None:
        nonlocal cur_index, cur_bit
        if cur_index >= n:
            return None
        first_byte = data[cur_index]
        first_part = ((0b11111110 >> cur_bit) & first_byte) << cur_bit
        first_part >>= 1
        cur_bit += 7
        if cur_bit < 8:
            return int(first_part)

        cur_bit -= 8
        cur_index += 1
        if cur_index >= n:
            return int(first_part)

        second_byte = data[cur_index]
        second_part = ((0xFF00 >> cur_bit) & second_byte) & 0xFF
        second_part >>= 8 - cur_bit
        return int(first_part | second_part)

    while True:
        bits = get7()
        if bits is None:
            break
        illegal_idx = _BASE122_ILLEGAL_TO_INDEX.get(bits, -1)
        if illegal_idx != -1:
            next_bits = get7()
            b1 = 0b11000010
            b2 = 0b10000000
            if next_bits is None:
                b1 |= (_BASE122_SHORTENED & 0b111) << 2
                next_bits = bits
            else:
                b1 |= (illegal_idx & 0b111) << 2
            b1 |= 1 if (next_bits & 0b01000000) else 0
            b2 |= next_bits & 0b00111111
            out.append(b1)
            out.append(b2)
        else:
            out.append(bits)
    return out.decode("utf-8")


def _decode_base122(text: str) -> bytes:
    illegals = _BASE122_ILLEGALS
    shortened = _BASE122_SHORTENED
    encoded = text.encode("utf-8")
    out = bytearray()
    cur_byte = 0
    bit_of_byte = 0

    def push7(seven: int) -> None:
        nonlocal cur_byte, bit_of_byte
        byte = (seven & 0x7F) << 1
        cur_byte |= (byte >> bit_of_byte) & 0xFF
        bit_of_byte += 7
        if bit_of_byte >= 8:
            out.append(cur_byte)
            bit_of_byte -= 8
            cur_byte = (byte << (7 - bit_of_byte)) & 0xFF

    i = 0
    n = len(encoded)
    while i < n:
        c0 = encoded[i]
        i += 1
        c = int(c0)
        if c > 127:
            if i >= n:
                break
            c1 = encoded[i]
            i += 1
            if (c1 & 0xC0) != 0x80:
                continue
            c = int(((c0 & 0x1F) << 6) | (c1 & 0x3F))
        if c > 127:
            illegal_index = (c >> 8) & 0b111
            if illegal_index != shortened:
                if illegal_index >= len(illegals):
                    continue
                push7(illegals[illegal_index])
            push7(c & 0x7F)
        else:
            push7(c)
    return bytes(out)


def _huff_code_lengths(data: bytes) -> list[int]:
    lengths = [0] * 256
    if not data:
        return lengths

    freq = [0] * 256
    for b in data:
        freq[b] += 1

    heap: list[tuple[int, int, int]] = []
    order = 0
    for sym, w in enumerate(freq):
        if w > 0:
            heapq.heappush(heap, (w, order, sym))
            order += 1

    if len(heap) == 1:
        lengths[heap[0][2]] = 1
        return lengths

    nodes: list[tuple[int, int]] = []
    while len(heap) > 1:
        w1, _, n1 = heapq.heappop(heap)
        w2, _, n2 = heapq.heappop(heap)
        node_id = 256 + len(nodes)
        nodes.append((n1, n2))
        heapq.heappush(heap, (w1 + w2, order, node_id))
        order += 1

    root = heap[0][2]
    stack: list[tuple[int, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if node < 256:
            lengths[node] = max(1, depth)
            continue
        left, right = nodes[node - 256]
        stack.append((left, depth + 1))
        stack.append((right, depth + 1))
    return lengths


def _huff_canonical_codes(lengths: list[int]) -> list[tuple[int, int]]:
    syms = sorted((ln, sym) for sym, ln in enumerate(lengths) if ln > 0)
    codes: list[tuple[int, int]] = [(0, 0)] * 256
    code = 0
    prev_len = 0
    for ln, sym in syms:
        if ln > 63:
            raise RuntimeError("[ENC] huff122 code length exceeds 63 bits")
        code <<= ln - prev_len
        codes[sym] = (code, ln)
        code += 1
        prev_len = ln
    return codes


def _huff122_compress(data: bytes) -> bytes:
    lengths = _huff_code_lengths(data)
    if max(lengths, default=0) > 63:
        raise RuntimeError("[ENC] huff122 max code length exceeded 63 bits")
    codes = _huff_canonical_codes(lengths)
    out = bytearray(lengths)
    cur = 0
    nbits = 0
    for b in data:
        code, ln = codes[b]
        for k in range(ln - 1, -1, -1):
            cur = (cur << 1) | ((code >> k) & 1)
            nbits += 1
            if nbits >= 8:
                out.append(cur & 0xFF)
                cur = 0
                nbits = 0
    if nbits > 0:
        out.append((cur << (8 - nbits)) & 0xFF)
    return bytes(out)


def _huff122_decompress(blob: bytes, *, expected_size: int) -> bytes:
    if expected_size < 0:
        raise RuntimeError("[ENC] huff122 expected_size must be non-negative")
    if expected_size == 0:
        return b""
    if len(blob) < 256:
        raise RuntimeError("[ENC] huff122 payload too short")

    lengths = list(blob[:256])
    syms = sorted((ln, sym) for sym, ln in enumerate(lengths) if ln > 0)
    if not syms:
        raise RuntimeError("[ENC] huff122 has no symbols")

    left = [-1]
    right = [-1]
    value = [-1]
    code = 0
    prev_len = 0
    for ln, sym in syms:
        code <<= ln - prev_len
        node = 0
        for k in range(ln - 1, -1, -1):
            bit = (code >> k) & 1
            nxt = right[node] if bit else left[node]
            if nxt < 0:
                left.append(-1)
                right.append(-1)
                value.append(-1)
                nxt = len(value) - 1
                if bit:
                    right[node] = nxt
                else:
                    left[node] = nxt
            node = nxt
        value[node] = sym
        code += 1
        prev_len = ln

    out = bytearray()
    node = 0
    for by in blob[256:]:
        for k in range(7, -1, -1):
            bit = (by >> k) & 1
            node = right[node] if bit else left[node]
            if node < 0:
                raise RuntimeError("[ENC] huff122 broken bitstream")
            sym = value[node]
            if sym >= 0:
                out.append(sym)
                if len(out) == expected_size:
                    return bytes(out)
                node = 0

    raise RuntimeError("[ENC] huff122 unexpected eof")


def _huff15_compress(data: bytes) -> bytes:
    lengths = _huff_code_lengths(data)
    if max(lengths, default=0) > 15:
        raise RuntimeError("[ENC] huff15 max code length exceeded 15 bits")
    codes = _huff_canonical_codes(lengths)
    head = bytearray(128)
    for i in range(0, 256, 2):
        head[i >> 1] = (lengths[i] & 0x0F) | ((lengths[i + 1] & 0x0F) << 4)
    out = bytearray(head)
    cur = 0
    nbits = 0
    for b in data:
        code, ln = codes[b]
        for k in range(ln - 1, -1, -1):
            cur = (cur << 1) | ((code >> k) & 1)
            nbits += 1
            if nbits >= 8:
                out.append(cur & 0xFF)
                cur = 0
                nbits = 0
    if nbits > 0:
        out.append((cur << (8 - nbits)) & 0xFF)
    return bytes(out)


def _huff15_decompress(blob: bytes, *, expected_size: int) -> bytes:
    if expected_size < 0:
        raise RuntimeError("[ENC] huff15 expected_size must be non-negative")
    if expected_size == 0:
        return b""
    if len(blob) < 128:
        raise RuntimeError("[ENC] huff15 payload too short")

    lengths = [0] * 256
    for i, by in enumerate(blob[:128]):
        lengths[i * 2] = by & 0x0F
        lengths[i * 2 + 1] = (by >> 4) & 0x0F
    syms = sorted((ln, sym) for sym, ln in enumerate(lengths) if ln > 0)
    if not syms:
        raise RuntimeError("[ENC] huff15 has no symbols")

    left = [-1]
    right = [-1]
    value = [-1]
    code = 0
    prev_len = 0
    for ln, sym in syms:
        code <<= ln - prev_len
        node = 0
        for k in range(ln - 1, -1, -1):
            bit = (code >> k) & 1
            nxt = right[node] if bit else left[node]
            if nxt < 0:
                left.append(-1)
                right.append(-1)
                value.append(-1)
                nxt = len(value) - 1
                if bit:
                    right[node] = nxt
                else:
                    left[node] = nxt
            node = nxt
        value[node] = sym
        code += 1
        prev_len = ln

    out = bytearray()
    node = 0
    for by in blob[128:]:
        for k in range(7, -1, -1):
            bit = (by >> k) & 1
            node = right[node] if bit else left[node]
            if node < 0:
                raise RuntimeError("[ENC] huff15 broken bitstream")
            sym = value[node]
            if sym >= 0:
                out.append(sym)
                if len(out) == expected_size:
                    return bytes(out)
                node = 0

    raise RuntimeError("[ENC] huff15 unexpected eof")


def _encode_model_payload(blob: bytes, *, encoding: str, use_huff15_for_huff91: bool = False) -> str:
    if encoding == "base91":
        payload = _encode_base91(blob)
        if _decode_base91(payload) != blob:
            raise RuntimeError("[ENC] base91 roundtrip failed")
    elif encoding == "base122":
        payload = _encode_base122(blob)
        if _decode_base122(payload) != blob:
            raise RuntimeError("[ENC] base122 roundtrip failed")
    elif encoding == "huff122":
        packed = _huff122_compress(blob)
        payload = _encode_base122(packed)
        decoded = _decode_base122(payload)
        if _huff122_decompress(decoded, expected_size=len(blob)) != blob:
            raise RuntimeError("[ENC] huff122 roundtrip failed")
    elif encoding == "huff91":
        if use_huff15_for_huff91:
            packed = _huff15_compress(blob)
        else:
            packed = _huff122_compress(blob)
        payload = _encode_base91(packed)
        decoded = _decode_base91(payload)
        if use_huff15_for_huff91:
            restored = _huff15_decompress(decoded, expected_size=len(blob))
        else:
            restored = _huff122_decompress(decoded, expected_size=len(blob))
        if restored != blob:
            raise RuntimeError("[ENC] huff91 roundtrip failed")
    else:
        raise RuntimeError(f"[ENC] unknown payload encoding: {encoding}")
    return payload


def _quantize_per_output_channel(weight: torch.Tensor, *, qmax: int = 127) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim < 1:
        raise ValueError(f"weight must have at least 1 dim, got shape={tuple(weight.shape)}")
    if qmax <= 0:
        raise ValueError(f"qmax must be positive, got {qmax}")
    w = weight.detach().to(torch.float32).contiguous()
    flat = w.view(w.shape[0], -1)
    max_abs = flat.abs().amax(dim=1)
    scale = torch.where(max_abs > 0.0, max_abs / float(qmax), torch.ones_like(max_abs))
    q = torch.round(flat / scale.unsqueeze(1)).clamp(-qmax, qmax).to(torch.int16)
    return q.view_as(w), scale


def _tensor_to_i8_bytes(t: torch.Tensor) -> bytes:
    a = t.detach().to(torch.int8).contiguous().view(-1).cpu().numpy().astype(np.int8, copy=False)
    return a.tobytes()


def _tensor_to_i4_packed_bytes(t: torch.Tensor) -> bytes:
    a = t.detach().to(torch.int16).contiguous().view(-1).cpu().numpy().astype(np.int16, copy=False)
    if np.any(a < -8) or np.any(a > 7):
        raise RuntimeError("[ENC] int4 tensor out of range [-8, 7]")
    u = (a & 0x0F).astype(np.uint8, copy=False)
    if (u.size & 1) != 0:
        u = np.concatenate((u, np.zeros(1, dtype=np.uint8)))
    packed = (u[0::2] | (u[1::2] << 4)).astype(np.uint8, copy=False)
    return packed.tobytes()


def _tensor_to_f16_bytes(t: torch.Tensor) -> bytes:
    a = t.detach().to(torch.float16).contiguous().view(-1).cpu().numpy().astype("<f2", copy=False)
    return a.tobytes()


def _tensor_to_fp8e4m3_bytes(t: torch.Tensor) -> bytes:
    a = (
        t.detach()
        .to(torch.float32)
        .to(torch.float8_e4m3fn)
        .contiguous()
        .view(torch.uint8)
        .cpu()
        .numpy()
        .astype(np.uint8, copy=False)
    )
    return a.tobytes()


def _tensor9_mxfp8e4m3_per_row_bytes(t: torch.Tensor) -> tuple[bytes, bytes]:
    if t.ndim != 2 or t.shape[1] != 9:
        raise ValueError(f"expected tensor shape [rows, 9], got {tuple(t.shape)}")
    w = t.detach().to(torch.float32).contiguous()
    max_abs = w.abs().amax(dim=1)
    scale = torch.where(max_abs > 0.0, max_abs, torch.ones_like(max_abs))
    wn = w / scale.unsqueeze(1)
    q_bytes = _tensor_to_fp8e4m3_bytes(wn.view(-1))
    s_bytes = _tensor_to_f16_bytes(scale.view(-1))
    return q_bytes, s_bytes


def _write_compact_inc(
    out_path: Path,
    *,
    encoded_q: str,
    encoded_h: str,
    meta: dict[str, int | str],
    short_names: bool = False,
) -> None:
    def cpp_escaped_lit(s: str, *, prefix: str) -> str:
        esc = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'{prefix}"{esc}"'

    def cpp_raw_lit(s: str, *, prefix: str) -> str:
        # Prefer short delimiters to reduce source size overhead.
        for delim in ("", "_", "x", "X", "q", "Q", "r", "R", "z", "Z"):
            marker = ")" + delim + '"'
            if marker not in s:
                return f'{prefix}R"{delim}({s}){delim}"'
        # Fallback with generated delimiters.
        for k in range(4096):
            delim = f"Q{k}X"
            marker = ")" + delim + '"'
            if marker not in s:
                return f'{prefix}R"{delim}({s}){delim}"'
        raise RuntimeError("[ENC] failed to choose C++ raw string delimiter")

    lines: list[str] = []
    lines.append("// GENERATED FILE. DO NOT EDIT.\n")
    lines.append("#pragma once\n")
    lines.append("\n")
    if short_names:
        qv = int(meta["q_bytes"])
        hv = int(meta["h_bytes"])
        lines.append(f"constexpr int QB={qv};\n")
        lines.append(f"constexpr int HB={hv};\n")
        qname = "Q"
        hname = "H"
        char_type = "char"
        lit_prefix = ""
    else:
        lines.append("#include <cstddef>\n")
        lines.append("\n")
        for k, v in meta.items():
            if isinstance(v, str):
                lines.append(f'static constexpr const char* kModelMeta_{k} = "{v}";\n')
            else:
                lines.append(f"static constexpr std::size_t kModelMeta_{k} = {int(v)};\n")
        lines.append("\n")
        qname = "kModelQEncoded"
        hname = "kModelHEncoded"
        char_type = "char8_t"
        lit_prefix = "u8"

    chunk = 1_000_000 if short_names else 16384
    if short_names:
        lines.append(f"{char_type} {qname}[]=")
        if encoded_q:
            for i in range(0, len(encoded_q), chunk):
                seg = encoded_q[i : i + chunk]
                lines.append(cpp_raw_lit(seg, prefix=lit_prefix))
        else:
            lines.append(f'{lit_prefix}""')
        lines.append(";\n")

        lines.append(f"{char_type} {hname}[]=")
        if encoded_h:
            for i in range(0, len(encoded_h), chunk):
                seg = encoded_h[i : i + chunk]
                lines.append(cpp_raw_lit(seg, prefix=lit_prefix))
        else:
            lines.append(f'{lit_prefix}""')
        lines.append(";\n")
    else:
        lines.append(f"static const {char_type} {qname}[]=\n")
        if encoded_q:
            for i in range(0, len(encoded_q), chunk):
                seg = encoded_q[i : i + chunk]
                lines.append(cpp_escaped_lit(seg, prefix=lit_prefix) + "\n")
        else:
            lines.append(f'{lit_prefix}""\n')
        lines.append(";\n")

        lines.append(f"static const {char_type} {hname}[]=\n")
        if encoded_h:
            for i in range(0, len(encoded_h), chunk):
                seg = encoded_h[i : i + chunk]
                lines.append(cpp_escaped_lit(seg, prefix=lit_prefix) + "\n")
        else:
            lines.append(f'{lit_prefix}""\n')
        lines.append(";\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def _bundle_cpp(entry: Path, *, include_dirs: list[Path]) -> str:
    include_re = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')
    include_sys_re = re.compile(r"^\s*#\s*include\s+<([^>]+)>\s*$")
    seen: set[Path] = set()
    seen_sys: set[str] = set()

    def strip_trailing_line_comment(line: str) -> str:
        # Raw string literal lines (R"delim(... )delim") may contain `//` payload.
        # Keep them untouched to avoid truncating encoded model blobs.
        if 'R"' in line:
            return line
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
        for d in include_dirs:
            cand = (d / name).resolve()
            if cand.is_file():
                return cand
        cand = (cur.parent / name).resolve()
        if cand.is_file():
            return cand
        return None

    def rec(path: Path) -> list[str]:
        out: list[str] = []
        in_block_comment = False
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")
        for li, seg in enumerate(lines):
            line = seg + ("\n" if li + 1 < len(lines) else "")
            if line.strip() == "#pragma once":
                continue
            drop, in_block_comment = should_drop_line(line, in_block_comment=in_block_comment)
            if drop:
                continue
            line = strip_trailing_line_comment(line)
            line = line.lstrip(" \t")
            ms = include_sys_re.match(line)
            if ms:
                hdr = ms.group(1)
                if hdr in seen_sys:
                    continue
                seen_sys.add(hdr)
                out.append(f"#include <{hdr}>\n")
                continue
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


def _compact_cpp_layout(src: str) -> str:
    multi_punct = [
        "<<=",
        ">>=",
        "<=>",
        "...",
        "->*",
        "->",
        "::",
        ".*",
        "++",
        "--",
        "<<",
        ">>",
        "<=",
        ">=",
        "==",
        "!=",
        "&&",
        "||",
        "*=",
        "/=",
        "%=",
        "+=",
        "-=",
        "&=",
        "^=",
        "|=",
        "##",
        "<:",
        ":>",
        "<%",
        "%>",
        "%:",
    ]
    multi_set = set(multi_punct)
    inc_sys_re = re.compile(r"^\s*#\s*include\s+<([^>]+)>\s*$")
    rename_tok = {
        "model": "z0",
        "cell_index": "z1",
        "read_aux_vec": "z2",
        "read_q_i8_block": "z3",
        "OpponentParam": "z4",
        "MoveSummary": "z5",
        "AdfBetaEstimator": "z6",
        "compute_move_dist_ai_like_from_moves": "z7",
        "summarize_ai_observation_from_moves": "z8",
        "categorize_move_for_ai": "z9",
        "compute_component_mask": "za",
        "enumerate_legal_moves": "zb",
        "expected_size": "zc",
        "kSiluLutSize": "zd",
        "kSiluLutRange": "ze",
        "kSiluLutMin": "zf",
        "kSiluLutMax": "zg",
        "delta_idx_from_cat": "zh",
        "delta_clip": "zi",
        "out_dist": "zj",
        "action_cell": "zk",
        "action_cat": "zl",
        "action_value": "zn",
        "action_in_b": "zo",
        "greedy_possible": "zp",
        "occupied_by_other": "zq",
        "out_comp_mask": "zr",
        "out_reach_mask": "zs",
        "read_dw_kernel": "zt",
        "PLAYER_BRANCH_N": "zu",
        "PLAYER_COMMON_INPUT_C": "zv",
        "PLAYER_ENEMY_FEAT_INPUT_C": "zw",
        "FEATURE_INPUT_C": "zx",
        "PLAYER_HIDDEN_C": "zy",
        "HIDDEN_C": "zz",
        "IN_CHANNELS": "zA",
        "BLOCKS_N": "zB",
        "FRONT_BLOCKS_N": "zC",
        "BACK_BLOCKS_N": "zD",
        "R4_GLOBAL_C": "zE",
        "R4_PLAYER_PER_C": "zF",
        "R4_PLAYER_BLOCK_C": "zG",
        "R4_TOTAL_C": "zH",
        "old_p": "zI",
        "channels": "zJ",
        "moves": "zK",
        "alpha": "zL",
        "beta_a": "zM",
        "beta_b": "zN",
        "eps_min": "zO",
        "eps_max": "zP",
        "u_max": "zQ",
        "t_max": "zR",
        "owner": "zT",
        "level": "zU",
        "board": "zV",
        "adf_beta": "zW",
        "CELL_MAX": "zX",
        "M_MAX": "zY",
        "State": "zZ",
        "reach": "z_",
        "sigma": "q0",
        "c_star": "q1",
        "param": "q2",
        "sigma_out": "q3",
        "inv_n": "q4",
        "b_off_h": "q5",
        "enemy": "q6",
        "st_start": "q7",
        "SymLen": "q8",
        "bo_pw": "q9",
        "mask": "u1",
        "next": "u2",
        "sign": "u3",
        "code": "u4",
        "scale": "u5",
        "acc0": "u6",
        "acc1": "u7",
        "acc2": "u8",
        "acc3": "u9",
        "acc4": "uA",
        "acc5": "uB",
        "acc6": "uC",
        "acc7": "uD",
        "e0_2": "uE",
        "e1_2": "uF",
        "CONC": "uG",
        "silu": "uH",
        "reset": "uI",
        "update": "uJ",
        "cnt": "r0",
        "dst": "r1",
        "sum": "r2",
        "node": "r3",
        "frac": "r4",
        "dist": "r5",
        "clip": "r6",
        "beta": "r7",
        "st0": "r8",
        "tmp": "r9",
        "est": "rA",
        "vmax": "rB",
        "lam": "rC",
        "kVec": "rD",
        "sumw": "rF",
        "mu_g": "rG",
        "cons": "rH",
        "syms": "rI",
        "mean": "rJ",
        "cidx": "rK",
        "turn": "rL",
        "mu_in": "rM",
        "left": "rN",
        "right": "rO",
        "value": "rP",
        "MAIN_GG": "v0",
        "PLAYER_GG": "v1",
        "read_h": "v2",
        "out_channels": "v3",
        "c_per_group": "v4",
        "legal_n": "v5",
        "ph_plane": "v6",
        "sigma_new": "v7",
        "sigma_in": "v8",
        "sigma_a": "v9",
        "sigma_g": "vA",
        "mu_new": "vB",
        "mu_out": "vC",
        "opp_old": "vD",
        "visited": "vE",
        "has_source": "vF",
        "mean_param": "vG",
        "trunc_halfspace_into": "vH",
        "beta_update_linear": "vI",
        "make_a_vec": "vJ",
        "INV_LEVEL_VALUE_SUM_SCALE": "vK",
        "INV_LEVEL_SUM_SCALE": "vL",
        "INV_SCORE_SCALE": "vM",
        "INV_MAX_CENTER": "vN",
        "INV_MAX_DIST": "vO",
        "INV_SQRT2PI": "vP",
        "out_moves": "vQ",
        "out_plane": "vR",
        "prior_std": "vS",
        "all_zero": "vT",
        "eps_mean": "vU",
        "next_u64": "vV",
        "prev_len": "vW",
        "sum_tail": "vX",
        "tmp_dist": "vY",
        "dst_cat": "vZ",
        "eps_bar": "v_",
        "h_bytes": "w9",
        "lut_ptr": "wA",
        "sq_tail": "wB",
        "best_v": "wC",
        "groups": "wD",
        "sum_lv": "wE",
        "v_star": "wF",
        "inv_w": "wG",
        "out_i": "wH",
        "score": "wI",
        "sum_l": "wJ",
        "vmean": "wK",
        "out_c": "wL",
        "in_c": "wM",
        "cons_n": "wN",
        "h_off": "wO",
        "inv_g": "wP",
        "log_m": "wQ",
        "old_o": "wR",
        "opp_n": "wS",
        "out_p": "wT",
        "sum_v": "wU",
        "sq_v": "wV",
        "owner_lv": "wW",
        "owner_l": "wX",
        "comp_lv": "wY",
        "comp_l": "wZ",
        "inv_u_fixed": "w_",
        "block_idx": "x1",
        "enemy_ch0": "x2",
        "apply_silu": "x3",
        "log_v_star": "x4",
        "b_off_dw": "x5",
        "b_off_pw": "x6",
        "dw_gn_b": "x7",
        "dw_gn_w": "x8",
        "pw_gn_b": "x9",
        "pw_gn_w": "xA",
        "id_bias": "xB",
        "q_bytes": "xC",
        "v_scale": "xD",
        "var_max": "xE",
        "symmetrize3": "xF",
        "NEIGH_RDLU": "xG",
        "XorShift64": "xH",
        "mfdw": "y0",
        "mfdgw": "y1",
        "mfdgb": "y2",
        "mfpw": "y3",
        "mfpgw": "y4",
        "mfpgb": "y5",
        "mbdw": "y6",
        "mbdgw": "y7",
        "mbdgb": "y8",
        "mbpw": "y9",
        "mbpgw": "yA",
        "mbpgb": "yB",
        "pscw": "yC",
        "psew": "yD",
        "peid": "yE",
        "psgw": "yF",
        "psgb": "yG",
        "pfdw": "yH",
        "pfdgw": "yI",
        "pfdgb": "yJ",
        "pfpw": "yK",
        "pfpgw": "yL",
        "pfpgb": "yM",
        "msgw": "yN",
        "msgb": "yO",
        "legal": "yP",
        "inv_std": "yQ",
        "blob": "yR",
        "packed": "yS",
        "q_off": "yT",
        "clvs": "yU",
        "olvs": "yV",
        "dot3": "yW",
        "sig2": "yX",
        "vmin": "yY",
        "vstd": "yZ",
        "PRIOR_STD": "y_",
        "is_source": "t0",
        "pos_floor": "t1",
        "mat_vec3": "t2",
        "INV_U64": "t3",
        "next_double01": "t4",
        "next_int": "t5",
        "in_bounds": "t6",
        "alpha1": "t7",
        "jitter": "t8",
        "offset": "t9",
        "cat_i": "tA",
        "cat_j": "tB",
        "kappa": "tC",
        "b91d": "tD",
        "coef": "tE",
        "csrc": "tF",
        "dw_w": "tG",
        "gn_b": "tH",
        "gn_w": "tI",
        "pw_w": "tJ",
        "kDec": "tK",
        "kEps": "tL",
        "prob": "tM",
        "ydst": "tN",
        "best": "tO",
        "bias": "tP",
        "bits": "tQ",
        "seed": "tR",
        "pcat": "tS",
        "vscale": "tT",
        "EPS0": "tU",
        "base": "tV",
        "eps0": "tW",
        "vbias": "tX",
        "c_other": "tY",
        "bo_dw": "tZ",
        "u0_raw": "t_",
    }
    type_alias_tok = {
        "size_t": "L",
        "uint8_t": "J",
        "uint16_t": "K",
        "int16_t": "U",
        "int8_t": "O",
        "int64_t": "P",
        "int": "I",
        "float": "G",
        "double": "M",
        "bool": "r",
    }
    # Additional token-level macro aliases for compact source size.
    # `_` is intentionally used for `const` because it has high frequency.
    macro_alias_tok = {
        "const": "_",
        "constexpr": "_a",
        "return": "_b",
        "static": "_c",
        "idx": "_d",
        "__m256": "_e",
        "for": "_f",
        "data": "_g",
        "inline": "_h",
        "continue": "_i",
        "resize": "_j",
        "__attribute__": "_k",
        "fill": "_l",
        "src": "_n",
        "comp": "_o",
        "push_back": "_p",
        "out": "_q",
        "else": "_r",
        "void": "_s",
        "false": "_t",
        "auto": "_v",
        "true": "_w",
        "sizeof": "_x",
        "struct": "_y",
    }
    seq_replace = [
        ("std::array", "R"),
        ("std::max", "S"),
        ("std::uint32_t", "Y"),
        ("_mm256_set1_ps", "Q0"),
        ("_mm256_fmadd_ps", "Q1"),
        ("_mm256_loadu_ps", "Q2"),
        ("_mm256_storeu_ps", "Q3"),
        ("_mm256_setzero_ps", "Q4"),
        ("_mm256_sub_ps", "Q5"),
        ("_mm256_set1_epi32", "Q6"),
        ("_mm256_mul_ps", "Q7"),
        ('target("avx2,fma")', "Q8"),
        ("_mm256_i32gather_ps", "Q9"),
    ]
    used_seq: list[tuple[str, str]] = []
    used_type_alias: list[tuple[str, str]] = []
    used_macro_alias: list[tuple[str, str]] = []
    drop_using_lines = {
        "using std::int64_t;",
        "using std::int8_t;",
        "using std::size_t;",
        "using std::uint16_t;",
        "using std::uint8_t;",
    }

    # Keep only compact standard includes for specialized compact mode.
    src_lines = src.split("\n")
    has_sys_inc = False
    need_immintrin = False
    kept_lines: list[str] = []
    for ln in src_lines:
        m = inc_sys_re.match(ln)
        if m is not None:
            has_sys_inc = True
            if m.group(1) == "immintrin.h":
                need_immintrin = True
            continue
        kept_lines.append(ln)
    if has_sys_inc:
        inc_lines = ["#include<bits/stdc++.h>"]
        if need_immintrin:
            inc_lines.append("#include<immintrin.h>")
        src = "\n".join(inc_lines + kept_lines)

    def tokenize(code: str) -> list[str]:
        out_toks: list[str] = []
        i = 0
        n = len(code)
        while i < n:
            c = code[i]
            if c.isspace():
                i += 1
                continue
            if c == '"' or c == "'":
                q = c
                j = i + 1
                esc = False
                while j < n:
                    ch = code[j]
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == q:
                        j += 1
                        break
                    j += 1
                out_toks.append(code[i:j])
                i = j
                continue
            if c == "R" and i + 1 < n and code[i + 1] == '"':
                j = i + 2
                while j < n and code[j] != "(":
                    j += 1
                if j >= n:
                    out_toks.append(code[i:])
                    break
                delim = code[i + 2 : j]
                marker = ")" + delim + '"'
                k = code.find(marker, j + 1)
                if k < 0:
                    out_toks.append(code[i:])
                    break
                k += len(marker)
                out_toks.append(code[i:k])
                i = k
                continue
            if c.isalpha() or c == "_":
                j = i + 1
                while j < n and (code[j].isalnum() or code[j] == "_"):
                    j += 1
                out_toks.append(code[i:j])
                i = j
                continue
            if c.isdigit() or (c == "." and i + 1 < n and code[i + 1].isdigit()):
                j = i + 1
                while j < n:
                    ch = code[j]
                    if ch.isalnum() or ch in "._'":
                        j += 1
                        continue
                    if ch in "+-" and j > i and code[j - 1] in "eEpP":
                        j += 1
                        continue
                    break
                out_toks.append(code[i:j])
                i = j
                continue
            matched = False
            for p in multi_punct:
                if code.startswith(p, i):
                    out_toks.append(p)
                    i += len(p)
                    matched = True
                    break
            if matched:
                continue
            out_toks.append(c)
            i += 1
        return out_toks

    def is_atom(tok: str) -> bool:
        if not tok:
            return False
        if tok[0].isalpha() or tok[0] == "_" or tok[0].isdigit():
            return True
        if tok[0] == "." and len(tok) > 1 and tok[1].isdigit():
            return True
        if tok[0] in ('"', "'"):
            return True
        if tok.startswith('R"'):
            return True
        return False

    def need_sep(prev: str, cur: str) -> bool:
        if not prev or not cur:
            return False
        if is_atom(prev) and is_atom(cur):
            return True
        if prev == "/" and cur in ("*", "/"):
            return True
        if (prev + cur) in multi_set:
            return True
        return False

    out: list[str] = []
    code_lines: list[str] = []
    needs_global_std_using = False

    def flush_code() -> None:
        nonlocal needs_global_std_using
        if not code_lines:
            return
        toks = tokenize(" ".join(code_lines))
        if not toks:
            code_lines.clear()
            return
        # Shrink verbose C++ casts into equivalent C-style casts.
        rewritten: list[str] = []
        i = 0
        cast_ops = {"static_cast", "reinterpret_cast", "const_cast"}
        while i < len(toks):
            if i + 2 < len(toks) and toks[i] in cast_ops and toks[i + 1] == "<":
                d = 1
                j = i + 2
                while j < len(toks):
                    if toks[j] == "<":
                        d += 1
                    elif toks[j] == ">":
                        d -= 1
                        if d == 0:
                            break
                    j += 1
                if j < len(toks) and d == 0 and i + 2 <= j - 1:
                    rewritten.append("(")
                    rewritten.extend(toks[i + 2 : j])
                    rewritten.append(")")
                    i = j + 1
                    continue
            rewritten.append(toks[i])
            i += 1
        toks: list[str] = []
        for t in rewritten:
            if t in type_alias_tok:
                nt = type_alias_tok[t]
                if (nt, t) not in used_type_alias:
                    used_type_alias.append((nt, t))
                t = nt
            if t in macro_alias_tok:
                nt = macro_alias_tok[t]
                if (nt, t) not in used_macro_alias:
                    used_macro_alias.append((nt, t))
                toks.append(nt)
                continue
            toks.append(rename_tok.get(t, t))

        prev = toks[0]
        res: list[str] = [prev]
        for t in toks[1:]:
            if need_sep(prev, t):
                res.append(" ")
            res.append(t)
            prev = t
        buf = "".join(res)
        # Merge redundant close/open of the same namespace boundary.
        while True:
            nb = buf.replace("}namespace ahc061::exp002{", "")
            if nb == buf:
                break
            buf = nb
        for old, new in seq_replace:
            nb = buf.replace(old, new)
            if nb != buf:
                if (old, new) not in used_seq:
                    used_seq.append((old, new))
                buf = nb
        if "std::" in buf:
            needs_global_std_using = True
            buf = buf.replace("std::", "")
        if "using namespace std;" in buf:
            buf = buf.replace("using namespace std;", "")
        out.append(buf + "\n")
        code_lines.clear()

    for raw in src.split("\n"):
        s = raw.strip()
        if not s:
            continue
        if s in drop_using_lines:
            continue
        if s.startswith("#") or 'u8"' in s or 'R"' in s:
            flush_code()
            out.append(s + "\n")
            continue
        code_lines.append(s)
    flush_code()
    if used_seq or used_type_alias or used_macro_alias:
        defs = []
        for old, new in used_seq:
            if needs_global_std_using:
                old = old.replace("std::", "")
            defs.append(f"#define {new} {old}\n")
        defs.extend(f"#define {name} {orig}\n" for name, orig in used_type_alias)
        defs.extend(f"#define {name} {orig}\n" for name, orig in used_macro_alias)
        ins = 0
        while ins < len(out) and (out[ins].startswith("#include ") or out[ins].startswith("#include<")):
            ins += 1
        out[ins:ins] = defs
    if needs_global_std_using:
        ins = 0
        while ins < len(out) and (
            out[ins].startswith("#include ")
            or out[ins].startswith("#include<")
            or out[ins].startswith("#define ")
        ):
            ins += 1
        out[ins:ins] = ["using namespace std;\n"]
    return "".join(out)


def _find_matching_brace(src: str, open_brace_pos: int) -> int:
    depth = 0
    for i in range(open_brace_pos, len(src)):
        ch = src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    raise RuntimeError("[TTA] failed to find matching brace")


def _remove_optional_function(src: str, signature: str) -> tuple[str, bool]:
    pos = src.find(signature)
    if pos < 0:
        return src, False
    start = src.rfind("\n", 0, pos)
    if start < 0:
        start = 0
    else:
        start += 1
    open_pos = src.find("{", pos)
    if open_pos < 0:
        raise RuntimeError(f"[TTA] malformed function near: {signature}")
    end = _find_matching_brace(src, open_pos) + 1
    while end < len(src) and src[end] in " \t\r\n":
        end += 1
    return src[:start] + src[end:], True


def _remove_optional_struct(src: str, signature: str) -> tuple[str, bool]:
    pos = src.find(signature)
    if pos < 0:
        return src, False
    start = src.rfind("\n", 0, pos)
    if start < 0:
        start = 0
    else:
        start += 1
    open_pos = src.find("{", pos)
    if open_pos < 0:
        raise RuntimeError(f"[STRIP] malformed struct near: {signature}")
    end = _find_matching_brace(src, open_pos) + 1
    while end < len(src) and src[end] in " \t\r":
        end += 1
    if end < len(src) and src[end] == ";":
        end += 1
    while end < len(src) and src[end] in " \t\r\n":
        end += 1
    return src[:start] + src[end:], True


def _remove_optional_exact_line(src: str, line: str) -> tuple[str, bool]:
    pat = line + "\n"
    if pat in src:
        return src.replace(pat, "", 1), True
    return src, False


def _ensure_system_include(src: str, header: str) -> str:
    inc = f"#include <{header}>"
    if inc in src:
        return src
    matches = list(re.finditer(r"^#include[^\n]*\n", src, flags=re.MULTILINE))
    if not matches:
        return inc + "\n" + src
    pos = matches[-1].end()
    return src[:pos] + inc + "\n" + src[pos:]


def _replace_required_function(src: str, signature: str, replacement: str) -> str:
    pos = src.find(signature)
    if pos < 0:
        raise RuntimeError(f"[TTA] function not found: {signature}")
    start = src.rfind("\n", 0, pos)
    if start < 0:
        start = 0
    else:
        start += 1
    open_pos = src.find("{", pos)
    if open_pos < 0:
        raise RuntimeError(f"[TTA] malformed function near: {signature}")
    end = _find_matching_brace(src, open_pos) + 1
    while end < len(src) and src[end] in " \t\r\n":
        end += 1
    rep = replacement
    if not rep.endswith("\n"):
        rep += "\n"
    return src[:start] + rep + src[end:]


def _build_select_action_mode0() -> str:
    return textwrap.dedent(
        """
        static int select_action(
            const CompactModel& model,
            const std::array<float, FEATURE_INPUT_C * CELL_MAX>& board,
            const std::array<std::uint8_t, CELL_MAX>& mask) {
            std::array<std::uint8_t, CELL_MAX> legal{};
            int legal_n = 0;
            for (int idx = 0; idx < CELL_MAX; idx++) {
                if (mask[static_cast<std::size_t>(idx)]) {
                    legal[static_cast<std::size_t>(legal_n++)] = static_cast<std::uint8_t>(idx);
                }
            }
            if (legal_n == 0) {
                return cell_index(0, 0);
            }

            std::array<float, CELL_MAX> logits;
            run_policy_logits(model, board, logits);

            int best = static_cast<int>(legal[0]);
            float best_v = -std::numeric_limits<float>::infinity();
            for (int li = 0; li < legal_n; li++) {
                const int idx = static_cast<int>(legal[static_cast<std::size_t>(li)]);
                const float v = logits[static_cast<std::size_t>(idx)];
                if (v > best_v) {
                    best = idx;
                    best_v = v;
                }
            }
            return best;
        }
        """
    ).strip() + "\n"


def _build_sa0_mode0() -> str:
    return textwrap.dedent(
        """
        static int sa0(
            const CM& model,
            const A<float, FEATURE_INPUT_C * CELL_MAX>& board,
            const A<uint8_t, CELL_MAX>& mask,
            int m) {
            A<uint8_t, CELL_MAX> legal{};
            int legal_n = 0;
            for (int idx = 0; idx < CELL_MAX; idx++) {
                if (mask[static_cast<size_t>(idx)]) {
                    legal[static_cast<size_t>(legal_n++)] = static_cast<uint8_t>(idx);
                }
            }
            if (legal_n == 0) {
                return cell_index(0, 0);
            }
            A<float, CELL_MAX> lgs;
            rpl(model, board, m, lgs);
            int best = static_cast<int>(legal[0]);
            float best_v = -std::numeric_limits<float>::infinity();
            for (int li = 0; li < legal_n; li++) {
                const int idx = static_cast<int>(legal[static_cast<size_t>(li)]);
                const float v = lgs[static_cast<size_t>(idx)];
                if (v > best_v) {
                    best = idx;
                    best_v = v;
                }
            }
            return best;
        }
        """
    ).strip() + "\n"


def _build_sa0_tta(*, mode: int, k: int, auto_off_ms: int) -> str:
    if mode not in (1, 2):
        raise RuntimeError(f"[TTA] unsupported mode for TTA body: {mode}")
    if k not in (2, 4, 8):
        raise RuntimeError(f"[TTA] unsupported TTA K: {k}")

    helper = ""
    if mode == 1:
        helper = textwrap.dedent(
            """
            static inline float tta_logaddexp_mode1(float a, float b) {
                const float neg_inf = -std::numeric_limits<float>::infinity();
                if (a == neg_inf)
                    return b;
                if (b == neg_inf)
                    return a;
                if (a < b)
                    std::swap(a, b);
                return a + static_cast<float>(std::log(1.0 + std::exp(static_cast<double>(b - a))));
            }
            """
        ).strip() + "\n\n"

    auto_off_decl = ""
    auto_off_init = "const int tta_k_runtime = kTtaK;"
    if auto_off_ms > 0:
        auto_off_decl = textwrap.dedent(
            f"""
            static constexpr int kTtaAutoOffMs = {int(auto_off_ms)};
            static const std::chrono::steady_clock::time_point kTtaProgramStartCompact =
                std::chrono::steady_clock::now();

            static inline int tta_effective_k_compact() {{
                static bool tta_off_latched = false;
                if (tta_off_latched)
                    return 1;
                const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                            std::chrono::steady_clock::now() - kTtaProgramStartCompact)
                                            .count();
                if (elapsed_ms >= static_cast<long long>(kTtaAutoOffMs)) {{
                    tta_off_latched = true;
                    return 1;
                }}
                return kTtaK;
            }}
            """
        ).strip() + "\n\n"
        auto_off_init = textwrap.dedent(
            """
            const int tta_k_runtime = tta_effective_k_compact();
            if (tta_k_runtime <= 1) {
                A<float, CELL_MAX> lgs0{};
                rpl(model, board, m, lgs0);
                int best0 = static_cast<int>(legal[0]);
                float best_v0 = -std::numeric_limits<float>::infinity();
                for (int li = 0; li < legal_n; li++) {
                    const int idx = static_cast<int>(legal[static_cast<size_t>(li)]);
                    const float v = lgs0[static_cast<size_t>(idx)];
                    if (v > best_v0) {
                        best0 = idx;
                        best_v0 = v;
                    }
                }
                return best0;
            }
            """
        ).strip()

    tpl = Template(
        textwrap.dedent(
            """
            ${HELPER}static constexpr int kTtaK = ${K};
            ${AUTO_OFF_DECL}static const std::array<std::array<std::uint8_t, CELL_MAX>, 8>& tta_perm_compact() {
                static const std::array<std::array<std::uint8_t, CELL_MAX>, 8> p = [] {
                    std::array<std::array<std::uint8_t, CELL_MAX>, 8> out{};
                    for (int flip = 0; flip < 2; flip++) {
                        for (int rot = 0; rot < 4; rot++) {
                            const int tk = flip * 4 + rot;
                            for (int x = 0; x < N; x++) {
                                for (int y = 0; y < N; y++) {
                                    int tx = x;
                                    int ty = y;
                                    if (flip)
                                        ty = N - 1 - ty;
                                    for (int rr = 0; rr < rot; rr++) {
                                        const int nx = ty;
                                        const int ny = N - 1 - tx;
                                        tx = nx;
                                        ty = ny;
                                    }
                                    out[static_cast<std::size_t>(tk)][static_cast<std::size_t>(cell_index(x, y))] =
                                        static_cast<std::uint8_t>(cell_index(tx, ty));
                                }
                            }
                        }
                    }
                    return out;
                }();
                return p;
            }

            static int sa0(
                const CM& model,
                const A<float, FEATURE_INPUT_C * CELL_MAX>& board,
                const A<uint8_t, CELL_MAX>& mask,
                int m) {
                A<uint8_t, CELL_MAX> legal{};
                int legal_n = 0;
                for (int idx = 0; idx < CELL_MAX; idx++) {
                    if (mask[static_cast<size_t>(idx)]) {
                        legal[static_cast<size_t>(legal_n++)] = static_cast<uint8_t>(idx);
                    }
                }
                if (legal_n == 0) {
                    return cell_index(0, 0);
                }

                const auto& p = tta_perm_compact();
                const float neg_inf = -std::numeric_limits<float>::infinity();
                A<float, CELL_MAX> acc{};
                acc.fill(${ACC_INIT});

                A<float, FEATURE_INPUT_C * CELL_MAX> board_t{};
                A<float, CELL_MAX> lgs_t{};
                ${AUTO_OFF_INIT}

                for (int tk = 0; tk < tta_k_runtime; tk++) {
                    const auto& pk = p[static_cast<std::size_t>(tk)];
                    if (tk == 0) {
                        rpl(model, board, m, lgs_t);
                    } else {
                        for (int c = 0; c < FEATURE_INPUT_C; c++) {
                            const float* src = board.data() + static_cast<size_t>(c) * CELL_MAX;
                            float* dst = board_t.data() + static_cast<size_t>(c) * CELL_MAX;
                            for (int idx = 0; idx < CELL_MAX; idx++) {
                                const int idx_t = pk[static_cast<size_t>(idx)];
                                dst[static_cast<size_t>(idx_t)] = src[static_cast<size_t>(idx)];
                            }
                        }
                        rpl(model, board_t, m, lgs_t);
                    }

                    float max_v = neg_inf;
                    for (int li = 0; li < legal_n; li++) {
                        const int idx = static_cast<int>(legal[static_cast<size_t>(li)]);
                        const int idx_t = pk[static_cast<size_t>(idx)];
                        max_v = std::max(max_v, lgs_t[static_cast<size_t>(idx_t)]);
                    }
                    double sum = 0.0;
                    for (int li = 0; li < legal_n; li++) {
                        const int idx = static_cast<int>(legal[static_cast<size_t>(li)]);
                        const int idx_t = pk[static_cast<size_t>(idx)];
                        sum += std::exp(static_cast<double>(lgs_t[static_cast<size_t>(idx_t)] - max_v));
                    }
                    const float logz = max_v + static_cast<float>(std::log(sum));

                    for (int li = 0; li < legal_n; li++) {
                        const int idx = static_cast<int>(legal[static_cast<size_t>(li)]);
                        const int idx_t = pk[static_cast<size_t>(idx)];
                        const float v = lgs_t[static_cast<size_t>(idx_t)] - logz;
                        ${ACC_UPDATE}
                    }
                }

                int best = static_cast<int>(legal[0]);
                float best_v = neg_inf;
                for (int li = 0; li < legal_n; li++) {
                    const int idx = static_cast<int>(legal[static_cast<size_t>(li)]);
                    const float v = acc[static_cast<size_t>(idx)];
                    if (v > best_v) {
                        best = idx;
                        best_v = v;
                    }
                }
                return best;
            }
            """
        )
    )
    return tpl.substitute(
        HELPER=helper,
        K=str(k),
        AUTO_OFF_DECL=auto_off_decl,
        AUTO_OFF_INIT=auto_off_init,
        ACC_INIT=("neg_inf" if mode == 1 else "0.0F"),
        ACC_UPDATE=(
            "acc[static_cast<size_t>(idx)] = tta_logaddexp_mode1(acc[static_cast<size_t>(idx)], v);"
            if mode == 1
            else "acc[static_cast<size_t>(idx)] += v;"
        ),
    ).strip() + "\n"


def _build_select_action_tta(*, mode: int, k: int, auto_off_ms: int) -> str:
    if mode not in (1, 2):
        raise RuntimeError(f"[TTA] unsupported mode for TTA body: {mode}")
    if k not in (2, 4, 8):
        raise RuntimeError(f"[TTA] unsupported TTA K: {k}")

    helper = ""
    if mode == 1:
        helper = textwrap.dedent(
            """
            static inline float tta_logaddexp_mode1(float a, float b) {
                const float neg_inf = -std::numeric_limits<float>::infinity();
                if (a == neg_inf)
                    return b;
                if (b == neg_inf)
                    return a;
                if (a < b)
                    std::swap(a, b);
                return a + static_cast<float>(std::log(1.0 + std::exp(static_cast<double>(b - a))));
            }
            """
        ).strip() + "\n\n"

    auto_off_decl = ""
    auto_off_init = "const int tta_k_runtime = kTtaK;"
    if auto_off_ms > 0:
        auto_off_decl = textwrap.dedent(
            f"""
            static constexpr int kTtaAutoOffMs = {int(auto_off_ms)};
            static const std::chrono::steady_clock::time_point kTtaProgramStartCompact =
                std::chrono::steady_clock::now();

            static inline int tta_effective_k_compact() {{
                static bool tta_off_latched = false;
                if (tta_off_latched)
                    return 1;
                const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                            std::chrono::steady_clock::now() - kTtaProgramStartCompact)
                                            .count();
                if (elapsed_ms >= static_cast<long long>(kTtaAutoOffMs)) {{
                    tta_off_latched = true;
                    return 1;
                }}
                return kTtaK;
            }}
            """
        ).strip() + "\n\n"
        auto_off_init = textwrap.dedent(
            """
            const int tta_k_runtime = tta_effective_k_compact();
            if (tta_k_runtime <= 1) {
                std::array<float, CELL_MAX> logits0{};
                run_policy_logits(model, board, logits0);
                int best0 = static_cast<int>(legal[0]);
                float best_v0 = -std::numeric_limits<float>::infinity();
                for (int li = 0; li < legal_n; li++) {
                    const int idx = static_cast<int>(legal[static_cast<std::size_t>(li)]);
                    const float v = logits0[static_cast<std::size_t>(idx)];
                    if (v > best_v0) {
                        best0 = idx;
                        best_v0 = v;
                    }
                }
                return best0;
            }
            """
        ).strip()

    tpl = Template(
        textwrap.dedent(
            """
            ${HELPER}static constexpr int kTtaK = ${K};
            ${AUTO_OFF_DECL}static const std::array<std::array<std::uint8_t, CELL_MAX>, 8>& tta_perm_compact() {
                static const std::array<std::array<std::uint8_t, CELL_MAX>, 8> p = [] {
                    std::array<std::array<std::uint8_t, CELL_MAX>, 8> out{};
                    for (int flip = 0; flip < 2; flip++) {
                        for (int rot = 0; rot < 4; rot++) {
                            const int tk = flip * 4 + rot;
                            for (int x = 0; x < N; x++) {
                                for (int y = 0; y < N; y++) {
                                    int tx = x;
                                    int ty = y;
                                    if (flip)
                                        ty = N - 1 - ty;
                                    for (int rr = 0; rr < rot; rr++) {
                                        const int nx = ty;
                                        const int ny = N - 1 - tx;
                                        tx = nx;
                                        ty = ny;
                                    }
                                    out[static_cast<std::size_t>(tk)][static_cast<std::size_t>(cell_index(x, y))] =
                                        static_cast<std::uint8_t>(cell_index(tx, ty));
                                }
                            }
                        }
                    }
                    return out;
                }();
                return p;
            }

            static int select_action(
                const CompactModel& model,
                const std::array<float, FEATURE_INPUT_C * CELL_MAX>& board,
                const std::array<std::uint8_t, CELL_MAX>& mask) {
                std::array<std::uint8_t, CELL_MAX> legal{};
                int legal_n = 0;
                for (int idx = 0; idx < CELL_MAX; idx++) {
                    if (mask[static_cast<std::size_t>(idx)]) {
                        legal[static_cast<std::size_t>(legal_n++)] = static_cast<std::uint8_t>(idx);
                    }
                }
                if (legal_n == 0) {
                    return cell_index(0, 0);
                }

                const auto& p = tta_perm_compact();
                const float neg_inf = -std::numeric_limits<float>::infinity();
                std::array<float, CELL_MAX> acc{};
                acc.fill(${ACC_INIT});

                std::array<float, FEATURE_INPUT_C * CELL_MAX> board_t{};
                std::array<float, CELL_MAX> logits_t{};
                ${AUTO_OFF_INIT}

                for (int tk = 0; tk < tta_k_runtime; tk++) {
                    const auto& pk = p[static_cast<std::size_t>(tk)];
                    if (tk == 0) {
                        run_policy_logits(model, board, logits_t);
                    } else {
                        for (int c = 0; c < FEATURE_INPUT_C; c++) {
                            const float* src = board.data() + static_cast<std::size_t>(c) * CELL_MAX;
                            float* dst = board_t.data() + static_cast<std::size_t>(c) * CELL_MAX;
                            for (int idx = 0; idx < CELL_MAX; idx++) {
                                const int idx_t = pk[static_cast<std::size_t>(idx)];
                                dst[static_cast<std::size_t>(idx_t)] = src[static_cast<std::size_t>(idx)];
                            }
                        }
                        run_policy_logits(model, board_t, logits_t);
                    }

                    float max_v = neg_inf;
                    for (int li = 0; li < legal_n; li++) {
                        const int idx = static_cast<int>(legal[static_cast<std::size_t>(li)]);
                        const int idx_t = pk[static_cast<std::size_t>(idx)];
                        max_v = std::max(max_v, logits_t[static_cast<std::size_t>(idx_t)]);
                    }
                    double sum = 0.0;
                    for (int li = 0; li < legal_n; li++) {
                        const int idx = static_cast<int>(legal[static_cast<std::size_t>(li)]);
                        const int idx_t = pk[static_cast<std::size_t>(idx)];
                        sum += std::exp(static_cast<double>(logits_t[static_cast<std::size_t>(idx_t)] - max_v));
                    }
                    const float logz = max_v + static_cast<float>(std::log(sum));

                    for (int li = 0; li < legal_n; li++) {
                        const int idx = static_cast<int>(legal[static_cast<std::size_t>(li)]);
                        const int idx_t = pk[static_cast<std::size_t>(idx)];
                        const float v = logits_t[static_cast<std::size_t>(idx_t)] - logz;
                        ${ACC_UPDATE}
                    }
                }

                int best = static_cast<int>(legal[0]);
                float best_v = neg_inf;
                for (int li = 0; li < legal_n; li++) {
                    const int idx = static_cast<int>(legal[static_cast<std::size_t>(li)]);
                    const float v = acc[static_cast<std::size_t>(idx)];
                    if (v > best_v) {
                        best = idx;
                        best_v = v;
                    }
                }
                return best;
            }
            """
        )
    )
    return tpl.substitute(
        HELPER=helper,
        K=str(k),
        AUTO_OFF_DECL=auto_off_decl,
        AUTO_OFF_INIT=auto_off_init,
        ACC_INIT=("neg_inf" if mode == 1 else "0.0F"),
        ACC_UPDATE=(
            "acc[static_cast<std::size_t>(idx)] = tta_logaddexp_mode1(acc[static_cast<std::size_t>(idx)], v);"
            if mode == 1
            else "acc[static_cast<std::size_t>(idx)] += v;"
        ),
    ).strip() + "\n"


def _apply_tta_specialization(src: str, *, mode: int, k: int, auto_off_ms: int) -> str:
    if mode not in (0, 1, 2):
        raise RuntimeError(f"[TTA] mode must be 0/1/2, got {mode}")
    if k not in (2, 4, 8):
        raise RuntimeError(f"[TTA] k must be 2/4/8, got {k}")
    if auto_off_ms > 0 and mode != 0:
        src = _ensure_system_include(src, "chrono")

    # Drop legacy macro-config TTA block if present.
    src = re.sub(
        r"#ifndef AHC061_EXP002_TTA_MODE[\s\S]*?static constexpr int kTtaK = [^\n]*\n",
        "",
        src,
        count=1,
    )
    # Drop legacy constexpr aliases if still present.
    src = re.sub(r"^\s*static constexpr int kTtaMode[^\n]*\n", "", src, flags=re.MULTILINE)
    src = re.sub(r"^\s*static constexpr int kTtaKRaw[^\n]*\n", "", src, flags=re.MULTILINE)
    src = re.sub(r"^\s*static constexpr int kTtaK[^\n]*\n", "", src, flags=re.MULTILINE)

    # Drop old helper functions when present (they will be regenerated mode-specifically).
    src, _ = _remove_optional_function(src, "static inline float logaddexp(")
    src, _ = _remove_optional_function(src, "static const std::array<std::array<std::uint8_t, CELL_MAX>, 8>& tta_perm(")

    if "static int select_action(" in src:
        if mode == 0:
            repl = _build_select_action_mode0()
        else:
            repl = _build_select_action_tta(mode=mode, k=k, auto_off_ms=auto_off_ms)
        src = _replace_required_function(src, "static int select_action(", repl)
        return src

    if "static int sa0(" in src:
        if mode == 0:
            repl = _build_sa0_mode0()
        else:
            repl = _build_sa0_tta(mode=mode, k=k, auto_off_ms=auto_off_ms)
        src = _replace_required_function(src, "static int sa0(", repl)
        return src

    raise RuntimeError("[TTA] target action function not found (select_action/sa0)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="exps/exp002/submit")
    parser.add_argument(
        "--payload-encoding",
        type=str,
        choices=("base91", "base122", "huff122", "huff91"),
        default="base91",
        help="payload text encoding embedded into model_compact_encoded.inc",
    )
    parser.add_argument(
        "--ppconcat-preset",
        type=str,
        choices=(
            "mixq",
            "i8all",
            "fp16",
            "fp16_merge_i8",
            "fp16_custom_i8",
            "c7",
            "fp16_blockpw_i8",
            "fp16_blockpw_merge_i8",
            "fp16_blockpw_i8_merge_i4",
            "fp16_blockpw_i4_merge_i8",
            "fp16_blockpw_i4_merge_i4",
            "fp16_blockpw_custom",
            "fp8aux_blockpw_merge_i8",
            "fp8aux_custom",
            "fp8full",
        ),
        default="mixq",
        help="quantization preset for dwres_ppconcat_v1",
    )
    parser.add_argument("--ppconcat-main-front-i8-mask", type=lambda x: int(x, 0), default=0)
    parser.add_argument("--ppconcat-main-back-i8-mask", type=lambda x: int(x, 0), default=0)
    parser.add_argument("--ppconcat-player-front-i8-mask", type=lambda x: int(x, 0), default=0)
    parser.add_argument("--ppconcat-merge-main-i8", action="store_true")
    parser.add_argument("--ppconcat-merge-player-i8", action="store_true")
    parser.add_argument("--ppconcat-main-front-i4-mask", type=lambda x: int(x, 0), default=0)
    parser.add_argument("--ppconcat-main-back-i4-mask", type=lambda x: int(x, 0), default=0)
    parser.add_argument("--ppconcat-player-front-i4-mask", type=lambda x: int(x, 0), default=0)
    parser.add_argument("--ppconcat-merge-main-i4", action="store_true")
    parser.add_argument("--ppconcat-merge-player-i4", action="store_true")
    parser.add_argument("--ppconcat-fp8-aux-mask", type=lambda x: int(x, 0), default=0)
    parser.add_argument("--ppconcat-mxdw-main", action="store_true")
    parser.add_argument("--ppconcat-mxdw-player", action="store_true")
    parser.add_argument(
        "--tta-mode",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="TTA mode: 0=off, 1=sum-prob(logsumexp), 2=prod-prob(sum logp)",
    )
    parser.add_argument(
        "--tta-k",
        type=int,
        choices=(2, 4, 8),
        default=2,
        help="TTA transforms K (valid when --tta-mode is 1 or 2)",
    )
    parser.add_argument(
        "--tta-auto-off-ms",
        type=int,
        default=-1,
        help=(
            "If >0 and --tta-mode is 1/2, switch to K=1 (TTA off) after this many elapsed milliseconds "
            "from program start."
        ),
    )
    args = parser.parse_args()
    tta_mode = int(args.tta_mode)
    tta_k = int(args.tta_k)
    tta_auto_off_ms = int(args.tta_auto_off_ms)

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

    if arch_name not in (
        "dwres_v1",
        "dwres_ppconcat_v1",
        "dwres_ppconcat_full",
        "dwres_ppconcat_full_v1",
        "dwres_ppconcat_full_pcatonly",
        "dwres_ppconcat_full_pcatonly_v1",
    ):
        raise RuntimeError(
            "[MODEL] compact submit currently supports arch_name in "
            "('dwres_v1', 'dwres_ppconcat_v1', 'dwres_ppconcat_full', "
            "'dwres_ppconcat_full_v1', 'dwres_ppconcat_full_pcatonly_v1') only, "
            f"got {arch_name!r}"
        )

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
    allowed_missing_prefixes = ("value_head.", "opp_move_head.", "opp_param_head.")
    bad_missing = [k for k in missing if not any(k.startswith(p) for p in allowed_missing_prefixes)]
    if bad_missing:
        raise RuntimeError(f"[CKPT] missing keys in state_dict: {bad_missing[:5]}{' ...' if len(bad_missing) > 5 else ''}")
    model.eval()

    st = normalize_state_dict_keys(model.state_dict())

    q_parts: list[bytes] = []
    h_parts: list[bytes] = []
    template_entry_name = "solver_base_compact.cpp"
    format_name = "dwres_int8pw_f16aux_v2"
    extra_meta: dict[str, int] = {}
    use_dwpp_c7_huff122_h112b16_minimal = False
    use_dwpp_c7_huff122_h112b18_minimal = False
    use_dwpp_c7_huff122_h112bx_minimal = False
    use_dwpp_c7_huff91_h112b16_minimal = False
    use_dwpp_c7_huff91_h112b18_minimal = False
    use_dwpp_c7_huff91_h112bx_minimal = False
    use_dwpp_c7_huff91_h112b20_minimal = False
    use_dwpp_full_c7_huff91_h96b17_minimal = False
    use_dwpp_pcatonly_c7_huff91_h112b20_minimal = False
    use_dwpp_c7_huff91_special_minimal = False

    def append_i8_1x1_weight(name: str) -> None:
        w = st[name].squeeze(-1).squeeze(-1)
        q, scale = _quantize_per_output_channel(w, qmax=127)
        q_parts.append(_tensor_to_i8_bytes(q))
        h_parts.append(_tensor_to_f16_bytes(scale))

    def append_i4_1x1_weight(name: str) -> None:
        w = st[name].squeeze(-1).squeeze(-1)
        q, scale = _quantize_per_output_channel(w, qmax=7)
        q_parts.append(_tensor_to_i4_packed_bytes(q))
        h_parts.append(_tensor_to_f16_bytes(scale))

    def append_i8_vector(name: str) -> None:
        v = st[name].detach().to(torch.float32).contiguous().view(-1)
        max_abs = torch.abs(v).amax()
        if float(max_abs) > 0.0:
            scale = max_abs / 127.0
        else:
            scale = torch.tensor(1.0, dtype=torch.float32)
        q = torch.round(v / scale).clamp(-127, 127).to(torch.int8)
        q_parts.append(_tensor_to_i8_bytes(q))
        h_parts.append(_tensor_to_f16_bytes(scale.view(1)))

    def append_f16_transposed_1x1_weight(name: str) -> None:
        w = st[name].squeeze(-1).squeeze(-1).detach().to(torch.float32).contiguous()  # [out, in]
        wt = w.transpose(0, 1).contiguous()  # [in, out]
        h_parts.append(_tensor_to_f16_bytes(wt))

    def append_f16_flat(name: str) -> None:
        v = st[name].detach().to(torch.float32).contiguous().view(-1)
        h_parts.append(_tensor_to_f16_bytes(v))

    def append_fp8_transposed_1x1_weight(name: str) -> None:
        w = st[name].squeeze(-1).squeeze(-1).detach().to(torch.float32).contiguous()  # [out, in]
        wt = w.transpose(0, 1).contiguous()  # [in, out]
        q_parts.append(_tensor_to_fp8e4m3_bytes(wt))

    def append_fp8_flat(name: str) -> None:
        v = st[name].detach().to(torch.float32).contiguous().view(-1)
        q_parts.append(_tensor_to_fp8e4m3_bytes(v))

    if arch_name == "dwres_v1":
        stem_w = st["stem.0.weight"].squeeze(-1).squeeze(-1)  # [H, C]
        h_parts.append(_tensor_to_f16_bytes(stem_w.view(-1)))
        h_parts.append(_tensor_to_f16_bytes(st["stem.1.weight"]))
        h_parts.append(_tensor_to_f16_bytes(st["stem.1.bias"]))

        for b in range(blocks):
            dw_w = st[f"blocks.{b}.dw.weight"].view(hidden, 9)
            h_parts.append(_tensor_to_f16_bytes(dw_w.view(-1)))
            h_parts.append(_tensor_to_f16_bytes(st[f"blocks.{b}.gn1.weight"]))
            h_parts.append(_tensor_to_f16_bytes(st[f"blocks.{b}.gn1.bias"]))

            pw_w = st[f"blocks.{b}.pw.weight"].squeeze(-1).squeeze(-1)  # [H, H]
            pw_q, pw_scale = _quantize_per_output_channel(pw_w, qmax=127)
            q_parts.append(_tensor_to_i8_bytes(pw_q))
            h_parts.append(_tensor_to_f16_bytes(pw_scale))
            h_parts.append(_tensor_to_f16_bytes(st[f"blocks.{b}.gn2.weight"]))
            h_parts.append(_tensor_to_f16_bytes(st[f"blocks.{b}.gn2.bias"]))

        policy_w = st["policy_head.weight"].view(hidden)
        h_parts.append(_tensor_to_f16_bytes(policy_w))
        h_parts.append(_tensor_to_f16_bytes(st["policy_head.bias"].view(1)))
    else:
        if feature_id != "research_v4":
            raise RuntimeError(
                f"[MODEL] dwres_ppconcat_(full|full_pcatonly)_v1 compact expects "
                f"feature_id='research_v4', got {feature_id!r}"
            )

        template_entry_name = "solver_base_compact_ppconcat.cpp"
        is_pcatonly_arch = arch_name in ("dwres_ppconcat_full_pcatonly", "dwres_ppconcat_full_pcatonly_v1")
        pp_preset = str(args.ppconcat_preset)
        c7_alias = False
        if pp_preset == "mixq":
            format_name = "dwres_ppconcat_mixq_v1"
            pp_preset_id = 0
        elif pp_preset == "i8all":
            format_name = "dwres_ppconcat_i8all_v1"
            pp_preset_id = 1
        elif pp_preset == "fp16":
            format_name = "dwres_ppconcat_fp16_v1"
            pp_preset_id = 2
        elif pp_preset == "fp16_merge_i8":
            format_name = "dwres_ppconcat_fp16_merge_i8_v1"
            pp_preset_id = 3
        elif pp_preset == "fp16_custom_i8":
            format_name = "dwres_ppconcat_fp16_custom_i8_v1"
            pp_preset_id = 12
        elif pp_preset == "c7":
            # C7 (conservative) = fp16_custom_i8 shortcut:
            # all block pw int8, merge(main/player) int8, no i4/fp8/mxdw.
            format_name = "dwres_ppconcat_fp16_custom_i8_v1"
            pp_preset_id = 12
            c7_alias = True
        elif pp_preset == "fp16_blockpw_i8":
            format_name = "dwres_ppconcat_fp16_blockpw_i8_v1"
            pp_preset_id = 4
        elif pp_preset == "fp16_blockpw_merge_i8":
            format_name = "dwres_ppconcat_fp16_blockpw_merge_i8_v1"
            pp_preset_id = 5
        elif pp_preset == "fp16_blockpw_i8_merge_i4":
            format_name = "dwres_ppconcat_fp16_blockpw_i8_merge_i4_v1"
            pp_preset_id = 6
        elif pp_preset == "fp16_blockpw_i4_merge_i8":
            format_name = "dwres_ppconcat_fp16_blockpw_i4_merge_i8_v1"
            pp_preset_id = 7
        elif pp_preset == "fp16_blockpw_i4_merge_i4":
            format_name = "dwres_ppconcat_fp16_blockpw_i4_merge_i4_v1"
            pp_preset_id = 8
        elif pp_preset == "fp16_blockpw_custom":
            format_name = "dwres_ppconcat_fp16_blockpw_custom_v1"
            pp_preset_id = 9
        elif pp_preset == "fp8aux_blockpw_merge_i8":
            format_name = "dwres_ppconcat_fp8aux_blockpw_merge_i8_v1"
            pp_preset_id = 10
        elif pp_preset == "fp8aux_custom":
            format_name = "dwres_ppconcat_fp8aux_custom_v1"
            pp_preset_id = 11
        else:
            format_name = "dwres_ppconcat_fp8full_v1"
            pp_preset_id = 13
        use_fp16_family = pp_preset_id >= 2
        fp8_aux_mask = 0
        front_blocks = max(1, blocks // 2)
        back_blocks = max(0, blocks - front_blocks)
        blockpw_base_mode = 0  # 0: fp16, 1: int8 base (+ per-block i4 mask)
        merge_main_mode = 0  # 0: fp16, 1: int8, 2: int4
        merge_player_mode = 0  # 0: fp16, 1: int8, 2: int4
        front_mask_i8 = 0
        back_mask_i8 = 0
        player_mask_i8 = 0
        front_mask_i4 = 0
        back_mask_i4 = 0
        player_mask_i4 = 0
        mxdw_main_mode = 0
        mxdw_player_mode = 0
        if pp_preset_id == 3:
            merge_main_mode = 1
            merge_player_mode = 1
        elif pp_preset_id == 12:
            if c7_alias:
                merge_main_mode = 1
                merge_player_mode = 1
                front_mask_i8 = (1 << front_blocks) - 1
                back_mask_i8 = (1 << back_blocks) - 1
                player_mask_i8 = (1 << front_blocks) - 1
            else:
                if bool(args.ppconcat_merge_main_i4):
                    merge_main_mode = 2
                elif bool(args.ppconcat_merge_main_i8):
                    merge_main_mode = 1
                else:
                    merge_main_mode = 0
                if bool(args.ppconcat_merge_player_i4):
                    merge_player_mode = 2
                elif bool(args.ppconcat_merge_player_i8):
                    merge_player_mode = 1
                else:
                    merge_player_mode = 0
                front_mask_i8 = int(args.ppconcat_main_front_i8_mask)
                back_mask_i8 = int(args.ppconcat_main_back_i8_mask)
                player_mask_i8 = int(args.ppconcat_player_front_i8_mask)
                front_mask_i4 = int(args.ppconcat_main_front_i4_mask)
                back_mask_i4 = int(args.ppconcat_main_back_i4_mask)
                player_mask_i4 = int(args.ppconcat_player_front_i4_mask)
                fp8_aux_mask = int(args.ppconcat_fp8_aux_mask)
                mxdw_main_mode = 1 if bool(args.ppconcat_mxdw_main) else 0
                mxdw_player_mode = 1 if bool(args.ppconcat_mxdw_player) else 0
        elif pp_preset_id == 4:
            blockpw_base_mode = 1
        elif pp_preset_id == 5:
            blockpw_base_mode = 1
            merge_main_mode = 1
            merge_player_mode = 1
        elif pp_preset_id == 6:
            blockpw_base_mode = 1
            merge_main_mode = 2
            merge_player_mode = 2
        elif pp_preset_id == 7:
            blockpw_base_mode = 1
            merge_main_mode = 1
            merge_player_mode = 1
            front_mask_i4 = (1 << front_blocks) - 1
            back_mask_i4 = (1 << back_blocks) - 1
            player_mask_i4 = (1 << front_blocks) - 1
        elif pp_preset_id == 8:
            blockpw_base_mode = 1
            merge_main_mode = 2
            merge_player_mode = 2
            front_mask_i4 = (1 << front_blocks) - 1
            back_mask_i4 = (1 << back_blocks) - 1
            player_mask_i4 = (1 << front_blocks) - 1
        elif pp_preset_id == 9:
            blockpw_base_mode = 1
            merge_main_mode = 2 if bool(args.ppconcat_merge_main_i4) else 1
            merge_player_mode = 2 if bool(args.ppconcat_merge_player_i4) else 1
            front_mask_i4 = int(args.ppconcat_main_front_i4_mask)
            back_mask_i4 = int(args.ppconcat_main_back_i4_mask)
            player_mask_i4 = int(args.ppconcat_player_front_i4_mask)
        elif pp_preset_id == 10:
            blockpw_base_mode = 1
            merge_main_mode = 1
            merge_player_mode = 1
            fp8_aux_mask = 0x7F
        elif pp_preset_id == 11:
            blockpw_base_mode = 1
            merge_main_mode = 1
            merge_player_mode = 1
            fp8_aux_mask = int(args.ppconcat_fp8_aux_mask)
            mxdw_main_mode = 1 if bool(args.ppconcat_mxdw_main) else 0
            mxdw_player_mode = 1 if bool(args.ppconcat_mxdw_player) else 0
        elif pp_preset_id == 13:
            fp8_aux_mask = 0x7F
        player_hidden = int(st["player_stem_common.weight"].shape[0])

        # Backward compatibility for historical key rename:
        # player_front.* -> player_front_blocks.*
        def resolve_player_front_prefix() -> str:
            preferred = "player_front_blocks"
            legacy = "player_front"
            probe = ".0.dw.weight"
            if f"{preferred}{probe}" in st:
                return preferred
            if f"{legacy}{probe}" in st:
                return legacy
            if any(k.startswith(preferred + ".") for k in st.keys()):
                return preferred
            if any(k.startswith(legacy + ".") for k in st.keys()):
                return legacy
            raise RuntimeError("[CKPT] player front block keys not found (expected player_front_blocks.* or player_front.*)")

        player_front_prefix = resolve_player_front_prefix()

        def player_front_key(block_idx: int, suffix: str) -> str:
            return f"{player_front_prefix}.{block_idx}.{suffix}"

        f16_one = _tensor_to_f16_bytes(torch.ones(1, dtype=torch.float32))

        def append_i8_1x1_weight_or_zero(name: str, out_c: int, in_c: int) -> None:
            if name in st:
                append_i8_1x1_weight(name)
                return
            z = torch.zeros((out_c, in_c), dtype=torch.float32)
            q, scale = _quantize_per_output_channel(z, qmax=127)
            q_parts.append(_tensor_to_i8_bytes(q))
            h_parts.append(_tensor_to_f16_bytes(scale))

        def append_i4_1x1_weight_or_zero(name: str, out_c: int, in_c: int) -> None:
            if name in st:
                append_i4_1x1_weight(name)
                return
            z = torch.zeros((out_c, in_c), dtype=torch.float32)
            q, scale = _quantize_per_output_channel(z, qmax=7)
            q_parts.append(_tensor_to_i4_packed_bytes(q))
            h_parts.append(_tensor_to_f16_bytes(scale))

        def append_i8_vector_or_zero(name: str, n: int) -> None:
            if name in st:
                append_i8_vector(name)
                return
            q_parts.append(bytes(n))
            h_parts.append(f16_one)

        use_dwpp_c7_huff122_h112b16_minimal = (
            bool(c7_alias)
            and str(args.payload_encoding) == "huff122"
            and hidden == 112
            and blocks == 16
            and player_hidden == 56
        )
        use_dwpp_c7_huff122_h112b18_minimal = (
            bool(c7_alias)
            and str(args.payload_encoding) == "huff122"
            and hidden == 112
            and blocks == 18
            and player_hidden == 56
        )
        use_dwpp_c7_huff91_h112b16_minimal = (
            bool(c7_alias)
            and str(args.payload_encoding) == "huff91"
            and hidden == 112
            and blocks == 16
            and player_hidden == 56
        )
        use_dwpp_c7_huff91_h112b18_minimal = (
            bool(c7_alias)
            and str(args.payload_encoding) == "huff91"
            and hidden == 112
            and blocks == 18
            and player_hidden == 56
        )
        use_dwpp_c7_huff91_h112b20_minimal = (
            bool(c7_alias)
            and str(args.payload_encoding) == "huff91"
            and hidden == 112
            and blocks == 20
            and player_hidden == 56
        )
        use_dwpp_full_c7_huff91_h96b17_minimal = (
            bool(c7_alias)
            and arch_name in ("dwres_ppconcat_full", "dwres_ppconcat_full_v1")
            and str(args.payload_encoding) == "huff91"
            and hidden == 96
            and blocks == 17
            and player_hidden == 96
        )
        use_dwpp_pcatonly_c7_huff91_h112b20_minimal = (
            bool(c7_alias)
            and arch_name in ("dwres_ppconcat_full_pcatonly", "dwres_ppconcat_full_pcatonly_v1")
            and str(args.payload_encoding) == "huff91"
            and hidden == 112
            and blocks == 20
            and player_hidden == 112
        )
        use_dwpp_c7_huff122_h112bx_minimal = use_dwpp_c7_huff122_h112b16_minimal or use_dwpp_c7_huff122_h112b18_minimal
        use_dwpp_c7_huff91_h112bx_minimal = (
            use_dwpp_c7_huff91_h112b16_minimal
            or use_dwpp_c7_huff91_h112b18_minimal
            or use_dwpp_c7_huff91_h112b20_minimal
        )
        use_dwpp_c7_huff91_special_minimal = (
            use_dwpp_c7_huff91_h112bx_minimal
            or use_dwpp_full_c7_huff91_h96b17_minimal
            or use_dwpp_pcatonly_c7_huff91_h112b20_minimal
        )
        if use_dwpp_c7_huff122_h112b16_minimal:
            template_entry_name = "solver_base_compact_ppconcat_c7_huff122_h112b16.cpp"
        elif use_dwpp_c7_huff122_h112b18_minimal:
            template_entry_name = "solver_base_compact_ppconcat_c7_huff122_h112b18.cpp"
        elif use_dwpp_c7_huff91_h112b16_minimal:
            template_entry_name = "solver_base_compact_ppconcat_c7_huff91_h112b16.cpp"
        elif use_dwpp_c7_huff91_h112b18_minimal:
            template_entry_name = "solver_base_compact_ppconcat_c7_huff91_h112b18.cpp"
        elif use_dwpp_c7_huff91_h112b20_minimal:
            template_entry_name = "solver_base_compact_ppconcat_c7_huff91_h112b20.cpp"
        elif use_dwpp_full_c7_huff91_h96b17_minimal:
            template_entry_name = "solver_base_compact_ppconcat_c7_huff91_h96b17_full.cpp"
        elif use_dwpp_pcatonly_c7_huff91_h112b20_minimal:
            template_entry_name = "solver_base_compact_ppconcat_c7_huff91_h112b20_pcatonly.cpp"
        front_mask_i8 &= (1 << front_blocks) - 1
        back_mask_i8 &= (1 << back_blocks) - 1
        player_mask_i8 &= (1 << front_blocks) - 1
        front_mask_i4 &= (1 << front_blocks) - 1
        back_mask_i4 &= (1 << back_blocks) - 1
        player_mask_i4 &= (1 << front_blocks) - 1
        if blockpw_base_mode != 0:
            front_mask_i8 = 0
            back_mask_i8 = 0
            player_mask_i8 = 0
        if blockpw_base_mode == 0 and pp_preset_id != 12:
            front_mask_i4 = 0
            back_mask_i4 = 0
            player_mask_i4 = 0
        fp8_aux_mask &= 0x7F

        def use_fp8(bit: int) -> bool:
            return ((fp8_aux_mask >> bit) & 1) != 0

        def append_aux_transposed(name: str, fp8_bit: int) -> None:
            if use_fp8(fp8_bit):
                append_fp8_transposed_1x1_weight(name)
            else:
                append_f16_transposed_1x1_weight(name)

        def append_aux_flat(name: str, fp8_bit: int) -> None:
            if use_fp8(fp8_bit):
                append_fp8_flat(name)
            else:
                append_f16_flat(name)

        def append_aux_transposed_or_zero(name: str, fp8_bit: int, out_c: int, in_c: int) -> None:
            if name in st:
                append_aux_transposed(name, fp8_bit)
                return
            n = out_c * in_c
            if use_fp8(fp8_bit):
                q_parts.append(bytes(n))
            else:
                h_parts.append(bytes(n * 2))

        def append_aux_flat_or_zero(name: str, fp8_bit: int, n: int) -> None:
            if name in st:
                append_aux_flat(name, fp8_bit)
                return
            if use_fp8(fp8_bit):
                q_parts.append(bytes(n))
            else:
                h_parts.append(bytes(n * 2))

        def append_aux_dw_kernel(name: str, channels: int, fp8_bit: int) -> None:
            dw = st[name].view(channels, 9).detach().to(torch.float32).contiguous().view(-1)
            if use_fp8(fp8_bit):
                q_parts.append(_tensor_to_fp8e4m3_bytes(dw))
            else:
                h_parts.append(_tensor_to_f16_bytes(dw))

        def append_aux_dw_kernel_or_zero(name: str, channels: int, fp8_bit: int) -> None:
            if name in st:
                append_aux_dw_kernel(name, channels, fp8_bit)
                return
            n = channels * 9
            if use_fp8(fp8_bit):
                q_parts.append(bytes(n))
            else:
                h_parts.append(bytes(n * 2))

        def append_mxfp8_dw_kernel(name: str, channels: int) -> None:
            dw = st[name].view(channels, 9)
            q_b, s_b = _tensor9_mxfp8e4m3_per_row_bytes(dw)
            q_parts.append(q_b)
            h_parts.append(s_b)

        def append_mxfp8_dw_kernel_or_zero(name: str, channels: int) -> None:
            if name in st:
                append_mxfp8_dw_kernel(name, channels)
                return
            q_parts.append(bytes(channels * 9))
            h_parts.append(bytes(channels * 2))

        if not is_pcatonly_arch:
            if use_fp16_family:
                append_aux_transposed_or_zero("main_stem.0.weight", 0, hidden, in_channels)
                append_aux_flat_or_zero("main_stem.1.weight", 0, hidden)
                append_aux_flat_or_zero("main_stem.1.bias", 0, hidden)
            else:
                append_i8_1x1_weight_or_zero("main_stem.0.weight", hidden, in_channels)
                append_i8_vector_or_zero("main_stem.1.weight", hidden)
                append_i8_vector_or_zero("main_stem.1.bias", hidden)

            for b in range(front_blocks):
                if use_fp16_family and mxdw_main_mode != 0:
                    append_mxfp8_dw_kernel_or_zero(f"main_front.{b}.dw.weight", hidden)
                else:
                    append_aux_dw_kernel_or_zero(f"main_front.{b}.dw.weight", hidden, 1)
                if use_fp16_family:
                    append_aux_flat_or_zero(f"main_front.{b}.gn1.weight", 2, hidden)
                    append_aux_flat_or_zero(f"main_front.{b}.gn1.bias", 2, hidden)
                    if blockpw_base_mode == 0:
                        if (front_mask_i4 >> b) & 1:
                            append_i4_1x1_weight_or_zero(f"main_front.{b}.pw.weight", hidden, hidden)
                        elif (front_mask_i8 >> b) & 1:
                            append_i8_1x1_weight_or_zero(f"main_front.{b}.pw.weight", hidden, hidden)
                        else:
                            append_aux_transposed_or_zero(f"main_front.{b}.pw.weight", 2, hidden, hidden)
                    elif (front_mask_i4 >> b) & 1:
                        append_i4_1x1_weight_or_zero(f"main_front.{b}.pw.weight", hidden, hidden)
                    else:
                        append_i8_1x1_weight_or_zero(f"main_front.{b}.pw.weight", hidden, hidden)
                    append_aux_flat_or_zero(f"main_front.{b}.gn2.weight", 2, hidden)
                    append_aux_flat_or_zero(f"main_front.{b}.gn2.bias", 2, hidden)
                else:
                    append_i8_vector_or_zero(f"main_front.{b}.gn1.weight", hidden)
                    append_i8_vector_or_zero(f"main_front.{b}.gn1.bias", hidden)
                    if pp_preset_id == 0:
                        append_i4_1x1_weight_or_zero(f"main_front.{b}.pw.weight", hidden, hidden)
                    else:
                        append_i8_1x1_weight_or_zero(f"main_front.{b}.pw.weight", hidden, hidden)
                    append_i8_vector_or_zero(f"main_front.{b}.gn2.weight", hidden)
                    append_i8_vector_or_zero(f"main_front.{b}.gn2.bias", hidden)

        for b in range(back_blocks):
            if use_fp16_family and mxdw_main_mode != 0:
                append_mxfp8_dw_kernel(f"main_back.{b}.dw.weight", hidden)
            else:
                append_aux_dw_kernel(f"main_back.{b}.dw.weight", hidden, 1)
            if use_fp16_family:
                append_aux_flat(f"main_back.{b}.gn1.weight", 2)
                append_aux_flat(f"main_back.{b}.gn1.bias", 2)
                if blockpw_base_mode == 0:
                    if (back_mask_i4 >> b) & 1:
                        append_i4_1x1_weight(f"main_back.{b}.pw.weight")
                    elif (back_mask_i8 >> b) & 1:
                        append_i8_1x1_weight(f"main_back.{b}.pw.weight")
                    else:
                        append_aux_transposed(f"main_back.{b}.pw.weight", 2)
                elif (back_mask_i4 >> b) & 1:
                    append_i4_1x1_weight(f"main_back.{b}.pw.weight")
                else:
                    append_i8_1x1_weight(f"main_back.{b}.pw.weight")
                append_aux_flat(f"main_back.{b}.gn2.weight", 2)
                append_aux_flat(f"main_back.{b}.gn2.bias", 2)
            else:
                append_i8_vector(f"main_back.{b}.gn1.weight")
                append_i8_vector(f"main_back.{b}.gn1.bias")
                if pp_preset_id == 0:
                    append_i4_1x1_weight(f"main_back.{b}.pw.weight")
                else:
                    append_i8_1x1_weight(f"main_back.{b}.pw.weight")
                append_i8_vector(f"main_back.{b}.gn2.weight")
                append_i8_vector(f"main_back.{b}.gn2.bias")

        if use_fp16_family:
            append_aux_transposed("player_stem_common.weight", 3)
            append_aux_transposed("player_stem_enemy_feat.weight", 3)
            append_aux_flat("player_stem_enemy_id.weight", 3)
            append_aux_flat("player_stem_norm.weight", 3)
            append_aux_flat("player_stem_norm.bias", 3)
        else:
            if pp_preset_id == 0:
                append_i4_1x1_weight("player_stem_common.weight")
                append_i4_1x1_weight("player_stem_enemy_feat.weight")
            else:
                append_i8_1x1_weight("player_stem_common.weight")
                append_i8_1x1_weight("player_stem_enemy_feat.weight")
            enemy_id_q, enemy_id_scale = _quantize_per_output_channel(st["player_stem_enemy_id.weight"], qmax=127)
            q_parts.append(_tensor_to_i8_bytes(enemy_id_q))
            h_parts.append(_tensor_to_f16_bytes(enemy_id_scale))
            append_i8_vector("player_stem_norm.weight")
            append_i8_vector("player_stem_norm.bias")

        for b in range(front_blocks):
            if use_fp16_family:
                if mxdw_player_mode != 0:
                    append_mxfp8_dw_kernel(player_front_key(b, "dw.weight"), player_hidden)
                else:
                    append_aux_flat(player_front_key(b, "dw.weight"), 4)
                append_aux_flat(player_front_key(b, "gn1.weight"), 5)
                append_aux_flat(player_front_key(b, "gn1.bias"), 5)
                if blockpw_base_mode == 0:
                    if (player_mask_i4 >> b) & 1:
                        append_i4_1x1_weight(player_front_key(b, "pw.weight"))
                    elif (player_mask_i8 >> b) & 1:
                        append_i8_1x1_weight(player_front_key(b, "pw.weight"))
                    else:
                        append_aux_transposed(player_front_key(b, "pw.weight"), 5)
                elif (player_mask_i4 >> b) & 1:
                    append_i4_1x1_weight(player_front_key(b, "pw.weight"))
                else:
                    append_i8_1x1_weight(player_front_key(b, "pw.weight"))
                append_aux_flat(player_front_key(b, "gn2.weight"), 5)
                append_aux_flat(player_front_key(b, "gn2.bias"), 5)
            else:
                dw_q, dw_scale = _quantize_per_output_channel(st[player_front_key(b, "dw.weight")].view(player_hidden, 9), qmax=127)
                q_parts.append(_tensor_to_i8_bytes(dw_q))
                h_parts.append(_tensor_to_f16_bytes(dw_scale))
                append_i8_vector(player_front_key(b, "gn1.weight"))
                append_i8_vector(player_front_key(b, "gn1.bias"))
                if pp_preset_id == 0:
                    append_i4_1x1_weight(player_front_key(b, "pw.weight"))
                else:
                    append_i8_1x1_weight(player_front_key(b, "pw.weight"))
                append_i8_vector(player_front_key(b, "gn2.weight"))
                append_i8_vector(player_front_key(b, "gn2.bias"))

        if use_fp16_family:
            if not is_pcatonly_arch:
                if merge_main_mode == 0:
                    append_aux_transposed_or_zero("merge_fuse_main.weight", 6, hidden, hidden)
                elif merge_main_mode == 2:
                    append_i4_1x1_weight_or_zero("merge_fuse_main.weight", hidden, hidden)
                else:
                    append_i8_1x1_weight_or_zero("merge_fuse_main.weight", hidden, hidden)
            if merge_player_mode == 0:
                append_aux_transposed("merge_fuse_player.weight", 6)
            elif merge_player_mode == 2:
                append_i4_1x1_weight("merge_fuse_player.weight")
            else:
                append_i8_1x1_weight("merge_fuse_player.weight")
            append_aux_flat("merge_fuse_norm.weight", 6)
            append_aux_flat("merge_fuse_norm.bias", 6)
            append_aux_flat("policy_head.weight", 6)
            append_aux_flat("policy_head.bias", 6)
        else:
            if not is_pcatonly_arch:
                if pp_preset_id == 0:
                    append_i4_1x1_weight_or_zero("merge_fuse_main.weight", hidden, hidden)
                else:
                    append_i8_1x1_weight_or_zero("merge_fuse_main.weight", hidden, hidden)
            if pp_preset_id == 0:
                append_i4_1x1_weight("merge_fuse_player.weight")
            else:
                append_i8_1x1_weight("merge_fuse_player.weight")
            append_i8_vector("merge_fuse_norm.weight")
            append_i8_vector("merge_fuse_norm.bias")

            policy_q, policy_scale = _quantize_per_output_channel(st["policy_head.weight"].view(1, hidden), qmax=127)
            q_parts.append(_tensor_to_i8_bytes(policy_q))
            h_parts.append(_tensor_to_f16_bytes(policy_scale))
            h_parts.append(_tensor_to_f16_bytes(st["policy_head.bias"].view(1)))

        if use_dwpp_c7_huff122_h112bx_minimal:
            extra_meta = {
                "player_hidden": player_hidden,
            }
        elif use_dwpp_c7_huff91_special_minimal:
            extra_meta = {}
        else:
            extra_meta = {
                "player_hidden": player_hidden,
                "ppconcat_preset": pp_preset_id,
                "blockpw_base_mode": blockpw_base_mode,
                "blockpw_front_i8_mask": front_mask_i8,
                "blockpw_back_i8_mask": back_mask_i8,
                "blockpw_player_i8_mask": player_mask_i8,
                "blockpw_front_i4_mask": front_mask_i4,
                "blockpw_back_i4_mask": back_mask_i4,
                "blockpw_player_i4_mask": player_mask_i4,
                "merge_main_mode": merge_main_mode,
                "merge_player_mode": merge_player_mode,
                "fp8_aux_mask": fp8_aux_mask,
                "mxdw_main_mode": mxdw_main_mode,
                "mxdw_player_mode": mxdw_player_mode,
                "arch_variant": 1 if is_pcatonly_arch else 0,
            }

    q_blob = b"".join(q_parts)
    h_blob = b"".join(h_parts)
    if len(h_blob) % 2 != 0:
        raise RuntimeError("[ENC] half blob size must be even")

    expected_q = len(q_blob)
    expected_h_count = len(h_blob) // 2

    payload_encoding = str(args.payload_encoding)
    if payload_encoding == "huff91" and arch_name == "dwres_v1":
        raise RuntimeError(
            "[ENC] payload-encoding=huff91 is currently supported only for dwres_ppconcat* compact formats"
        )
    payload_codec_map = {
        "base91": _PAYLOAD_CODEC_BASE91,
        "base122": _PAYLOAD_CODEC_BASE122,
        "huff122": _PAYLOAD_CODEC_HUFF122,
        "huff91": _PAYLOAD_CODEC_HUFF91,
    }
    payload_codec = payload_codec_map[payload_encoding]
    use_huff15_for_huff91 = use_dwpp_c7_huff91_special_minimal
    q_payload = _encode_model_payload(
        q_blob,
        encoding=payload_encoding,
        use_huff15_for_huff91=use_huff15_for_huff91,
    )
    h_payload = _encode_model_payload(
        h_blob,
        encoding=payload_encoding,
        use_huff15_for_huff91=use_huff15_for_huff91,
    )

    if use_dwpp_c7_huff91_special_minimal:
        meta: dict[str, int | str] = {
            "q_bytes": len(q_blob),
            "h_bytes": len(h_blob),
        }
    else:
        meta = {
            "in_channels": in_channels,
            "hidden": hidden,
            "blocks": blocks,
            "q_bytes": len(q_blob),
            "h_bytes": len(h_blob),
        }
    if not use_dwpp_c7_huff91_special_minimal:
        meta["payload_codec"] = payload_codec
        meta["q_count"] = expected_q
        meta["h_count"] = expected_h_count
    meta.update(extra_meta)

    model_inc = out_dir / "model_compact_encoded.inc"
    _write_compact_inc(
        model_inc,
        encoded_q=q_payload,
        encoded_h=h_payload,
        meta=meta,
        short_names=use_dwpp_c7_huff91_special_minimal,
    )

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
    entry = template_dir / template_entry_name
    include_dirs = [
        out_dir,
        repo_root / "exps" / "exp002" / "cpp_core" / "include",
    ]
    bundled = _bundle_cpp(entry, include_dirs=include_dirs)
    bundled = _apply_tta_specialization(
        bundled,
        mode=tta_mode,
        k=tta_k,
        auto_off_ms=tta_auto_off_ms,
    )
    if use_dwpp_c7_huff91_special_minimal:
        # Drop unused declarations from bundled core headers in specialized compact mode.
        bundled, _ = _remove_optional_struct(bundled, "struct XorShift64")
        bundled, _ = _remove_optional_function(bundled, "constexpr bool in_bounds(int x, int y)")
        bundled, _ = _remove_optional_exact_line(bundled, "static constexpr std::array<int, 4> DX{1, -1, 0, 0};")
        bundled, _ = _remove_optional_exact_line(bundled, "static constexpr std::array<int, 4> DY{0, 0, 1, -1};")
        bundled = _compact_cpp_layout(bundled)
    main_cpp = out_dir / "Main.cpp"
    main_cpp.write_text(bundled, encoding="utf-8")

    print(f"[ENC] payload_encoding={payload_encoding}")
    if tta_mode == 0:
        print("[TTA] mode=0 (disabled)")
        if tta_auto_off_ms > 0:
            print(f"[TTA] auto_off_ms={tta_auto_off_ms} ignored (mode=0)")
    elif tta_auto_off_ms > 0:
        print(f"[TTA] mode={tta_mode} k={tta_k} auto_off_ms={tta_auto_off_ms} (switch to k=1 after threshold)")
    else:
        print(f"[TTA] mode={tta_mode} k={tta_k}")
    if use_dwpp_c7_huff122_h112b16_minimal:
        print("[ENC] using specialized minimal template: solver_base_compact_ppconcat_c7_huff122_h112b16.cpp")
    if use_dwpp_c7_huff122_h112b18_minimal:
        print("[ENC] using specialized minimal template: solver_base_compact_ppconcat_c7_huff122_h112b18.cpp")
    if use_dwpp_c7_huff91_h112b16_minimal:
        print("[ENC] using specialized minimal template: solver_base_compact_ppconcat_c7_huff91_h112b16.cpp")
    if use_dwpp_c7_huff91_h112b18_minimal:
        print("[ENC] using specialized minimal template: solver_base_compact_ppconcat_c7_huff91_h112b18.cpp")
    if use_dwpp_c7_huff91_h112b20_minimal:
        print("[ENC] using specialized minimal template: solver_base_compact_ppconcat_c7_huff91_h112b20.cpp")
    if use_dwpp_full_c7_huff91_h96b17_minimal:
        print("[ENC] using specialized minimal template: solver_base_compact_ppconcat_c7_huff91_h96b17_full.cpp")
    if use_dwpp_pcatonly_c7_huff91_h112b20_minimal:
        print("[ENC] using specialized minimal template: solver_base_compact_ppconcat_c7_huff91_h112b20_pcatonly.cpp")
    print(f"[ENC] q_raw={len(q_blob)} q_enc={len(q_payload)} h_raw={len(h_blob)} h_enc={len(h_payload)}")
    print(f"[OK] wrote: {feature_inc}")
    print(f"[OK] wrote: {model_inc}")
    print(f"[OK] wrote: {main_cpp}")


if __name__ == "__main__":
    main()
