from __future__ import annotations

import argparse

from ..cpp_ext import load_ext


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    ext = load_ext(verbose=bool(args.verbose_build))
    ids = list(ext.feature_ids())
    for fid in ids:
        c = int(ext.feature_channels(str(fid)))
        submit_ok = bool(ext.feature_submit_supported(str(fid)))
        print(f"{fid}\tchannels={c}\tsubmit_supported={submit_ok}")


if __name__ == "__main__":
    main()

