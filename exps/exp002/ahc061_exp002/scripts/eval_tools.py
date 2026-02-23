from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from ..ckpt import model_spec_from_ckpt, normalize_state_dict_keys, torch_load_maybe_weights_only
from ..env import BatchEnv, tools_input_paths
from ..models import build_policy_value_model, masked_logits


def _build_tta_perm() -> torch.Tensor:
    # Compact submit implementationと同順序:
    # k = flip * 4 + rot, flip in {0,1}, rot in {0,1,2,3}
    perm = torch.empty((8, 100), dtype=torch.long)
    for flip in range(2):
        for rot in range(4):
            k = flip * 4 + rot
            for x in range(10):
                for y in range(10):
                    tx = x
                    ty = y
                    if flip:
                        ty = 9 - ty
                    for _ in range(rot):
                        nx = ty
                        ny = 9 - tx
                        tx = nx
                        ty = ny
                    perm[k, x * 10 + y] = tx * 10 + ty
    return perm


def _apply_tta_board(board: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    # perm: idx(orig) -> idx(transformed)
    flat = board.reshape(board.shape[0], board.shape[1], 100)
    out = torch.empty_like(flat)
    out[:, :, perm] = flat
    return out.reshape_as(board)


def _apply_tta_mask(mask: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(mask)
    out[:, perm] = mask
    return out


def _masked_log_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    neg_inf = float("-inf")
    masked = logits.masked_fill(~legal_mask, neg_inf)
    # 念のため: 全て非合法の行があっても NaN を出さない
    any_legal = legal_mask.any(dim=1)
    if (~any_legal).any():
        masked = masked.clone()
        masked[~any_legal] = 0.0
    out = torch.log_softmax(masked, dim=1)
    out = out.masked_fill(~legal_mask, neg_inf)
    return out


@torch.no_grad()
def run_episode_greedy(
    env: BatchEnv,
    model: torch.nn.Module,
    device: torch.device,
    *,
    tta_mode: int = 0,
    tta_k: int = 2,
    tta_perm_cpu: torch.Tensor | None = None,
    tta_perm_device: torch.Tensor | None = None,
    action_traces: list[list[tuple[int, int]]] | None = None,
) -> None:
    model.eval()
    board = torch.empty((env.batch_size, env.feature_channels, 10, 10), dtype=torch.float32, device="cpu")
    mask = torch.empty((env.batch_size, 100), dtype=torch.uint8, device="cpu")
    if tta_mode != 0:
        if tta_perm_cpu is None or tta_perm_device is None:
            full_perm = _build_tta_perm()
            tta_perm_cpu = full_perm[:tta_k].contiguous()
            tta_perm_device = tta_perm_cpu.to(device=device)

    for _ in range(env.spec.t_max):
        env.observe_into(board, mask)
        if tta_mode == 0:
            logits, _, _, _ = model(board.to(device))
            logits = masked_logits(logits, mask.to(device))
        else:
            assert tta_perm_cpu is not None
            assert tta_perm_device is not None
            legal_orig = mask.to(device=device, dtype=torch.bool)
            if tta_mode == 1:
                acc = torch.full((env.batch_size, 100), float("-inf"), dtype=torch.float32, device=device)
            else:
                acc = torch.zeros((env.batch_size, 100), dtype=torch.float32, device=device)

            for tk in range(int(tta_k)):
                if tk == 0:
                    board_t = board
                    mask_t = mask
                else:
                    pk_cpu = tta_perm_cpu[tk]
                    board_t = _apply_tta_board(board, pk_cpu)
                    mask_t = _apply_tta_mask(mask, pk_cpu)

                logits_t, _, _, _ = model(board_t.to(device))
                legal_t = mask_t.to(device=device, dtype=torch.bool)
                logp_t = _masked_log_softmax(logits_t, legal_t)

                # compact実装と同様に、orig idx -> transformed idx の写像で戻す
                pk_dev = tta_perm_device[tk]
                logp_back = logp_t.index_select(1, pk_dev)
                if tta_mode == 1:
                    acc = torch.logaddexp(acc, logp_back)
                else:
                    acc = acc + logp_back

            logits = acc.masked_fill(~legal_orig, float("-inf"))

        actions = torch.argmax(logits, dim=1).to("cpu")
        if action_traces is not None:
            action_ids = actions.tolist()
            for i, action_id in enumerate(action_ids):
                action = int(action_id)
                action_traces[i].append((action // 10, action % 10))
        env.step(actions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--seed-begin", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=9)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-pf", action="store_true")
    parser.add_argument(
        "--copmile",
        "--compile",
        dest="copmile",
        action="store_true",
        help="Enable torch.compile with fixed mode='default' (CUDA only).",
    )
    parser.add_argument("--torch-num-threads", type=int, default=None)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)
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
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()

    if args.seed_begin > args.seed_end:
        raise RuntimeError("--seed-begin must be <= --seed-end")
    if args.batch <= 0:
        raise RuntimeError("--batch must be >= 1")
    if args.torch_num_threads is not None and int(args.torch_num_threads) <= 0:
        raise RuntimeError("--torch-num-threads must be >= 1")
    if args.torch_num_interop_threads is not None and int(args.torch_num_interop_threads) <= 0:
        raise RuntimeError("--torch-num-interop-threads must be >= 1")
    if "/" in args.run_name or "\\" in args.run_name:
        raise RuntimeError("--run-name must not contain path separators")

    if args.torch_num_threads is not None:
        torch.set_num_threads(int(args.torch_num_threads))
    if args.torch_num_interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(args.torch_num_interop_threads))
        except RuntimeError as e:
            print(f"[WARN] torch.set_num_interop_threads failed ({type(e).__name__}: {e})")

    def pick_device() -> torch.device:
        if args.device != "auto":
            return torch.device(args.device)
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            arch = f"sm_{cap[0]}{cap[1]}"
            if arch in torch.cuda.get_arch_list():
                return torch.device("cuda")
        return torch.device("cpu")

    device = pick_device()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    ckpt = torch_load_maybe_weights_only(ckpt_path)
    ms = model_spec_from_ckpt(ckpt)

    env = BatchEnv(batch_size=args.batch, feature_id=ms.feature_id, pf_enabled=not args.no_pf, verbose_build=False)
    arch_name = str(ms.arch_name)
    model = build_policy_value_model(
        arch_name,
        in_channels=int(ms.in_channels),
        hidden_channels=int(ms.hidden),
        blocks=int(ms.blocks),
        feature_id=str(ms.feature_id),
        arch_kwargs=dict(ms.arch_kwargs),
    ).to(device)
    missing, unexpected = model.load_state_dict(normalize_state_dict_keys(ckpt["model"]), strict=False)
    if unexpected:
        raise RuntimeError(f"[CKPT] unexpected keys in state_dict: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    allowed_missing_prefixes = ("opp_move_head.", "opp_param_head.")
    bad_missing = [k for k in missing if not any(k.startswith(p) for p in allowed_missing_prefixes)]
    if bad_missing:
        raise RuntimeError(f"[CKPT] missing keys in state_dict: {bad_missing[:5]}{' ...' if len(bad_missing) > 5 else ''}")
    compile_ok = False
    if bool(args.copmile):
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this torch build")
        if device.type != "cuda":
            print("[WARN] --copmile was set but device is not CUDA; skipping torch.compile.")
        else:
            try:
                model = torch.compile(model, mode="default")
                compile_ok = True
            except Exception as e:
                print(f"[WARN] torch.compile failed ({type(e).__name__}: {e}); falling back to eager.")

    repo_root = Path(__file__).resolve().parents[4]
    tests_run_dir = repo_root / "tests" / str(args.run_name)
    tests_out_dir = tests_run_dir / "out"
    tests_res_dir = tests_run_dir / "res"
    results_file = repo_root / "results" / f"{args.run_name}.res"

    if tests_run_dir.exists():
        shutil.rmtree(tests_run_dir)
    tests_out_dir.mkdir(parents=True, exist_ok=True)
    tests_res_dir.mkdir(parents=True, exist_ok=True)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    seed_ids = list(range(int(args.seed_begin), int(args.seed_end) + 1))
    paths = tools_input_paths(args.seed_begin, args.seed_end)
    if len(paths) != len(seed_ids):
        raise RuntimeError(
            f"[TOOLS] number of paths mismatch: got {len(paths)}, expected {len(seed_ids)} "
            f"(seed range {args.seed_begin}-{args.seed_end})"
        )
    score_rows: list[tuple[int, int]] = []
    tta_perm_cpu: torch.Tensor | None = None
    tta_perm_device: torch.Tensor | None = None
    if int(args.tta_mode) != 0:
        tta_perm_cpu = _build_tta_perm()[: int(args.tta_k)].contiguous()
        tta_perm_device = tta_perm_cpu.to(device=device)

    with tqdm(total=len(seed_ids), desc="eval_tools", unit="seed", dynamic_ncols=True, disable=None) as pbar:
        for i in range(0, len(paths), args.batch):
            chunk_paths = paths[i : i + args.batch]
            chunk_ids = seed_ids[i : i + args.batch]
            if len(chunk_paths) < args.batch:
                # pad with last to fill the batch, then discard extras
                pad_n = args.batch - len(chunk_paths)
                chunk_paths = chunk_paths + [chunk_paths[-1]] * pad_n
                chunk_ids = chunk_ids + [chunk_ids[-1]] * pad_n
                valid = len(seed_ids) - i
            else:
                valid = args.batch

            traces: list[list[tuple[int, int]]] = [[] for _ in range(args.batch)]
            env.reset_from_tools(chunk_paths)
            run_episode_greedy(
                env,
                model,
                device,
                tta_mode=int(args.tta_mode),
                tta_k=int(args.tta_k),
                tta_perm_cpu=tta_perm_cpu,
                tta_perm_device=tta_perm_device,
                action_traces=traces,
            )
            scores = env.official_score().to(dtype=torch.int64).tolist()
            for j in range(valid):
                seed = int(chunk_ids[j])
                score = int(scores[j])
                score_rows.append((seed, score))

                seed_name = f"{seed:04d}"
                out_path = tests_out_dir / f"{seed_name}.out"
                res_path = tests_res_dir / f"{seed_name}.res"
                out_path.write_text("".join(f"{x} {y}\n" for x, y in traces[j]), encoding="utf-8")
                res_path.write_text(f"[DATA] score = {score}\n", encoding="utf-8")
            pbar.update(valid)

    score_rows.sort(key=lambda x: x[0])
    with results_file.open("w", encoding="utf-8") as f:
        for seed, score in score_rows:
            f.write(json.dumps({"id": int(seed), "score": int(score)}))
            f.write("\n")

    mean_score = sum(score for _, score in score_rows) / max(1, len(score_rows))
    if int(args.tta_mode) == 0:
        print("[TTA] mode=0 (off)")
    else:
        print(f"[TTA] mode={int(args.tta_mode)} k={int(args.tta_k)}")
    print(f"[COMPILE] requested={bool(args.copmile)} ok={bool(compile_ok)} mode=default")
    print(
        f"[CPU] torch_num_threads={torch.get_num_threads()} "
        f"torch_num_interop_threads={torch.get_num_interop_threads()}"
    )
    print(f"tools seeds {args.seed_begin}-{args.seed_end}: mean_score={mean_score:.3f} n={len(score_rows)}")
    print(f"wrote: {results_file}")
    print(f"wrote: {tests_out_dir}")
    print(f"wrote: {tests_res_dir}")


if __name__ == "__main__":
    main()
