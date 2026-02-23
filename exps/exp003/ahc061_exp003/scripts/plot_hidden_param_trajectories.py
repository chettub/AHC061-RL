from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from exps.exp002.ahc061_exp002.ckpt import model_spec_from_ckpt, normalize_state_dict_keys, torch_load_maybe_weights_only
from exps.exp002.ahc061_exp002.env import BatchEnv
from exps.exp002.ahc061_exp002.models import build_policy_value_model, masked_logits

from ..cpp_ext import load_ext


PARAM_NAMES = ("w0", "w1", "w2", "w3", "eps")
METHOD_LABEL = {
    "pf": "PF",
    "adf_beta": "ADF+Beta",
    "adf_beta_ep": "ADF+Beta+EP",
    "hybrid_adf_rbpf": "Hybrid ADF-RBPF",
}


class _XorShift64:
    _MASK = (1 << 64) - 1
    _DEFAULT = 88172645463325252

    def __init__(self, seed: int):
        x = int(seed) & self._MASK
        self.x = x if x != 0 else self._DEFAULT

    def next_u64(self) -> int:
        x = self.x
        x ^= (x << 7) & self._MASK
        x ^= (x >> 9) & self._MASK
        self.x = x & self._MASK
        return self.x

    def next_int(self, lo: int, hi: int) -> int:
        span = int(hi - lo + 1)
        return int(lo + (self.next_u64() % span))


def _pick_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        arch = f"sm_{cap[0]}{cap[1]}"
        if arch in torch.cuda.get_arch_list():
            return torch.device("cuda")
    return torch.device("cpu")


def _parse_int_list(s: str) -> list[int]:
    parts = [p.strip() for p in str(s).replace(" ", ",").split(",") if p.strip()]
    out = [int(p) for p in parts]
    if not out:
        raise ValueError("integer list is empty")
    return out


def _parse_methods(s: str) -> list[str]:
    parts = [p.strip() for p in str(s).replace(" ", ",").split(",") if p.strip()]
    out: list[str] = []
    supported = {"pf", "adf_beta", "adf_beta_ep", "hybrid_adf_rbpf"}
    for p in parts:
        if p not in supported:
            raise ValueError(f"unknown method: {p} (supported: {sorted(supported)})")
        if p not in out:
            out.append(p)
    if not out:
        raise ValueError("methods is empty")
    return out


def _run_trace_one(
    *,
    ext,
    method: str,
    seed: int,
    t_max: int,
    prior_std: float,
    eps0: float,
    rbpf_particles: int,
) -> dict[str, np.ndarray | int]:
    if method == "pf":
        out = ext.trace_pf_estimation(int(seed), int(t_max))
    elif method == "adf_beta":
        out = ext.trace_ineq_trunc_gauss_beta_eps_estimation(int(seed), int(t_max), float(prior_std), float(eps0))
    elif method == "adf_beta_ep":
        out = ext.trace_ineq_trunc_gauss_beta_ep_estimation(int(seed), int(t_max), float(prior_std), float(eps0))
    elif method == "hybrid_adf_rbpf":
        out = ext.trace_hybrid_adf_rbpf_estimation(int(seed), int(t_max), float(prior_std), float(eps0), int(rbpf_particles))
    else:
        raise ValueError(f"unsupported method: {method}")

    m = int(out["m"])
    t = int(out["t"])
    true_param = out["true_param"].cpu().numpy().astype(np.float64, copy=False)
    pred_param = out["pred_param"].cpu().numpy().astype(np.float64, copy=False)
    if true_param.shape != (t, 5) or pred_param.shape != (t, 5):
        raise RuntimeError(
            f"invalid trace shape for method={method}: true={true_param.shape} pred={pred_param.shape} expected={(t, 5)}"
        )
    return {
        "m": m,
        "t": t,
        "true_param": true_param,
        "pred_param": pred_param,
    }


def _resolve_ckpt_path(*, repo_root: Path, ckpt: str | None, latest_0p999: bool) -> Path | None:
    if ckpt is not None:
        p = Path(ckpt).expanduser()
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        else:
            p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"ckpt not found: {p}")
        return p

    if not latest_0p999:
        return None

    cand = sorted((repo_root / "checkpoints").glob("*_0p999.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not cand:
        raise FileNotFoundError("no checkpoint matched checkpoints/*_0p999.pt")
    return cand[0].resolve()


def _load_ckpt_runner(*, ckpt_path: Path, device: torch.device, no_pf: bool) -> tuple[BatchEnv, torch.nn.Module, str]:
    ckpt = torch_load_maybe_weights_only(ckpt_path)
    ms = model_spec_from_ckpt(ckpt)

    env = BatchEnv(
        batch_size=1,
        feature_id=str(ms.feature_id),
        pf_enabled=not bool(no_pf),
        verbose_build=False,
    )
    if int(ms.in_channels) != int(env.feature_channels):
        raise RuntimeError(
            f"[CKPT] in_channels mismatch: ckpt={int(ms.in_channels)} env(feature_id={str(ms.feature_id)!r})={int(env.feature_channels)}"
        )

    model = build_policy_value_model(
        str(ms.arch_name),
        in_channels=int(ms.in_channels),
        hidden_channels=int(ms.hidden),
        blocks=int(ms.blocks),
        feature_id=str(ms.feature_id),
    ).to(device)

    missing, unexpected = model.load_state_dict(normalize_state_dict_keys(ckpt["model"]), strict=False)
    if unexpected:
        raise RuntimeError(f"[CKPT] unexpected keys in state_dict: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    bad_missing = [k for k in missing if not k.startswith("opp_move_head.") and not k.startswith("opp_param_head.")]
    if bad_missing:
        raise RuntimeError(f"[CKPT] missing keys in state_dict: {bad_missing[:5]}{' ...' if len(bad_missing) > 5 else ''}")
    if any(k.startswith("opp_param_head.") for k in missing):
        raise RuntimeError(f"[CKPT] opp_param_head is missing in {ckpt_path.name}; cannot trace hidden-parameter prediction")
    model.eval()

    label = f"CKPT {ckpt_path.name}"
    return env, model, label


@torch.inference_mode()
def _run_ckpt_trace_one(
    *,
    env: BatchEnv,
    model: torch.nn.Module,
    device: torch.device,
    seed: int,
    t_max: int,
    policy: str,
) -> dict[str, np.ndarray | int]:
    m_max = int(env.spec.m_max)
    t = int(min(int(env.spec.t_max), int(t_max)))

    board = torch.empty((1, int(env.feature_channels), 10, 10), dtype=torch.float32, device="cpu")
    mask = torch.empty((1, 100), dtype=torch.uint8, device="cpu")
    next_board = torch.empty_like(board)
    next_mask = torch.empty_like(mask)
    reward = torch.empty((1,), dtype=torch.float32, device="cpu")
    done = torch.empty((1,), dtype=torch.uint8, device="cpu")
    opp_move_dist = torch.empty((1, m_max, 100), dtype=torch.float32, device="cpu")
    opp_param_true = torch.empty((1, m_max, 5), dtype=torch.float32, device="cpu")
    opp_valid = torch.empty((1, m_max), dtype=torch.uint8, device="cpu")

    true_param = np.full((t, 5), np.nan, dtype=np.float64)
    pred_param = np.full((t, 5), np.nan, dtype=np.float64)

    env.reset_random(torch.tensor([int(seed)], dtype=torch.int64, device="cpu"))
    rng = _XorShift64(int(seed) ^ 0x1234567890ABCDEF)
    m_case = -1

    for turn in range(t):
        env.observe_into(board, mask)
        env.aux_targets_into(opp_move_dist, opp_param_true, opp_valid)

        board_dev = board.to(device, non_blocking=(device.type == "cuda"))
        mask_dev = mask.to(device, non_blocking=(device.type == "cuda"))
        logits, _, _, opp_param_logits = model(board_dev, with_aux=True)
        logits = masked_logits(logits.float(), mask_dev)

        pred_logits = opp_param_logits.float().to("cpu")[0]  # [8, 5]
        pred_w = torch.softmax(pred_logits[:, :4], dim=-1)
        pred_eps = torch.sigmoid(pred_logits[:, 4])

        valid = opp_valid[0] != 0
        n_valid = int(valid.sum().item())
        if m_case < 0:
            m_case = int(n_valid + 1)
        if n_valid > 0:
            true_w = opp_param_true[0, :, :4]
            true_eps = opp_param_true[0, :, 4]
            true_param[turn, :4] = true_w[valid].mean(dim=0).numpy()
            true_param[turn, 4] = float(true_eps[valid].mean().item())
            pred_param[turn, :4] = pred_w[valid].mean(dim=0).numpy()
            pred_param[turn, 4] = float(pred_eps[valid].mean().item())

        if turn + 1 >= t:
            break

        if policy == "greedy":
            action = torch.argmax(logits, dim=1).to(dtype=torch.int64, device="cpu")
        else:
            legal = torch.nonzero(mask[0] != 0, as_tuple=False).view(-1)
            if int(legal.numel()) <= 0:
                action = torch.zeros((1,), dtype=torch.int64, device="cpu")
            else:
                pick = rng.next_int(0, int(legal.numel()) - 1)
                action = legal[pick].view(1).to(dtype=torch.int64, device="cpu")

        env.step_observe_into(action, next_board, next_mask, reward, done)
        board, next_board = next_board, board
        mask, next_mask = next_mask, mask

    if m_case < 0:
        m_case = 0
    return {
        "m": int(m_case),
        "t": int(t),
        "true_param": true_param,
        "pred_param": pred_param,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2,3")
    parser.add_argument("--t-max", type=int, default=100)
    parser.add_argument("--methods", type=str, default="pf,adf_beta,adf_beta_ep,hybrid_adf_rbpf")
    parser.add_argument("--pf-particles", type=int, default=128)
    parser.add_argument("--prior-std", type=float, default=0.325)
    parser.add_argument("--eps0", type=float, default=0.30)
    parser.add_argument("--rbpf-particles", type=int, default=64)
    parser.add_argument("--ckpt", type=str, default=None, help="optional exp002 .pt to overlay as an extra predictor")
    parser.add_argument("--ckpt-latest-0p999", action="store_true", help="use latest checkpoints/*_0p999.pt")
    parser.add_argument("--ckpt-label", type=str, default=None)
    parser.add_argument("--ckpt-device", type=str, default="auto")
    parser.add_argument("--ckpt-policy", type=str, choices=("random_xorshift", "greedy"), default="random_xorshift")
    parser.add_argument("--ckpt-no-pf", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    seeds = _parse_int_list(str(args.seeds))
    methods = _parse_methods(str(args.methods))
    if args.t_max <= 0:
        raise ValueError("t-max must be >= 1")
    if args.pf_particles <= 0:
        raise ValueError("pf-particles must be >= 1")
    if args.prior_std <= 0:
        raise ValueError("prior-std must be > 0")
    if args.rbpf_particles <= 0:
        raise ValueError("rbpf-particles must be >= 1")

    repo_root = Path(__file__).resolve().parents[4]
    exp_dir = Path(__file__).resolve().parents[2]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        exp_dir / "artifacts" / f"plot_hidden_param_trajectories_{ts}"
        if args.out_dir is None
        else Path(str(args.out_dir)).expanduser().resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = load_ext(pf_particles=int(args.pf_particles), verbose=bool(args.verbose_build))

    ckpt_path = _resolve_ckpt_path(
        repo_root=repo_root,
        ckpt=(str(args.ckpt) if args.ckpt is not None else None),
        latest_0p999=bool(args.ckpt_latest_0p999),
    )
    ckpt_method_key: str | None = None
    ckpt_env: BatchEnv | None = None
    ckpt_model: torch.nn.Module | None = None
    ckpt_device: torch.device | None = None
    method_label = dict(METHOD_LABEL)
    if ckpt_path is not None:
        ckpt_device = _pick_device(str(args.ckpt_device))
        ckpt_env, ckpt_model, auto_label = _load_ckpt_runner(
            ckpt_path=ckpt_path,
            device=ckpt_device,
            no_pf=bool(args.ckpt_no_pf),
        )
        ckpt_method_key = "__ckpt__"
        method_label[ckpt_method_key] = str(args.ckpt_label) if args.ckpt_label is not None else auto_label
        print(f"[INFO] ckpt overlay enabled: {ckpt_path} (device={ckpt_device}, policy={args.ckpt_policy})")

    traces: dict[int, dict[str, dict[str, np.ndarray | int]]] = {}
    for seed in seeds:
        traces[seed] = {}
        for i, method in enumerate(methods, start=1):
            print(f"[TRACE] seed={seed} method={method} ({i}/{len(methods)})")
            res = _run_trace_one(
                ext=ext,
                method=method,
                seed=int(seed),
                t_max=int(args.t_max),
                prior_std=float(args.prior_std),
                eps0=float(args.eps0),
                rbpf_particles=int(args.rbpf_particles),
            )
            traces[seed][method] = res

        # Consistency check among classical methods.
        ref_m = int(traces[seed][methods[0]]["m"])
        ref_t = int(traces[seed][methods[0]]["t"])
        ref_true = np.asarray(traces[seed][methods[0]]["true_param"], dtype=np.float64)
        for method in methods[1:]:
            m_i = int(traces[seed][method]["m"])
            t_i = int(traces[seed][method]["t"])
            tru_i = np.asarray(traces[seed][method]["true_param"], dtype=np.float64)
            if m_i != ref_m or t_i != ref_t:
                raise RuntimeError(
                    f"inconsistent case metadata at seed={seed}: {methods[0]} has (m,t)=({ref_m},{ref_t}) but {method} has ({m_i},{t_i})"
                )
            if not np.allclose(tru_i, ref_true, atol=1e-12, rtol=1e-10):
                raise RuntimeError(f"inconsistent true trajectory at seed={seed} between {methods[0]} and {method}")

        if ckpt_method_key is not None:
            if ckpt_env is None or ckpt_model is None or ckpt_device is None:
                raise RuntimeError("internal error: ckpt runner is not initialized")
            ckpt_res = _run_ckpt_trace_one(
                env=ckpt_env,
                model=ckpt_model,
                device=ckpt_device,
                seed=int(seed),
                t_max=int(args.t_max),
                policy=str(args.ckpt_policy),
            )
            traces[seed][ckpt_method_key] = ckpt_res
            # Informative mismatch check (true trajectory source differs between simulators).
            tt = min(int(ckpt_res["t"]), int(ref_t))
            if tt > 0:
                c_true = np.asarray(ckpt_res["true_param"], dtype=np.float64)[:tt, :]
                r_true = ref_true[:tt, :]
                max_abs = float(np.nanmax(np.abs(c_true - r_true)))
                if max_abs > 1e-7:
                    print(f"[WARN] seed={seed}: true trajectory mismatch ext-vs-ckpt simulator max_abs={max_abs:.6e}")

    plot_methods = methods[:] + ([ckpt_method_key] if ckpt_method_key is not None else [])

    # Long CSV.
    rows: list[dict[str, object]] = []
    for seed in seeds:
        m = int(traces[seed][methods[0]]["m"])
        t_seed = min(int(traces[seed][k]["t"]) for k in plot_methods)
        true_arr = np.asarray(traces[seed][methods[0]]["true_param"], dtype=np.float64)[:t_seed, :]
        for turn in range(t_seed):
            for d, pname in enumerate(PARAM_NAMES):
                rows.append(
                    {
                        "seed": int(seed),
                        "m": int(m),
                        "turn": int(turn + 1),
                        "param": str(pname),
                        "source": "true",
                        "method": "true",
                        "value": float(true_arr[turn, d]),
                    }
                )
            for method in plot_methods:
                pred = np.asarray(traces[seed][method]["pred_param"], dtype=np.float64)[:t_seed, :]
                for d, pname in enumerate(PARAM_NAMES):
                    rows.append(
                        {
                            "seed": int(seed),
                            "m": int(m),
                            "turn": int(turn + 1),
                            "param": str(pname),
                            "source": "pred",
                            "method": str(method),
                            "value": float(pred[turn, d]),
                        }
                    )
    df_long = pd.DataFrame(rows)
    csv_long = out_dir / "trajectories_long.csv"
    df_long.to_csv(csv_long, index=False)

    # Grid plot: rows=cases(seeds), cols=parameters.
    n_rows = len(seeds)
    n_cols = len(PARAM_NAMES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.1, n_rows * 2.4), squeeze=False, sharex=False)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(3, len(plot_methods))))

    legend_handles = None
    legend_labels = None
    for r, seed in enumerate(seeds):
        m = int(traces[seed][methods[0]]["m"])
        t_seed = min(int(traces[seed][k]["t"]) for k in plot_methods)
        true_arr = np.asarray(traces[seed][methods[0]]["true_param"], dtype=np.float64)[:t_seed, :]
        x = np.arange(1, t_seed + 1, dtype=np.int32)

        for c, pname in enumerate(PARAM_NAMES):
            ax = axes[r, c]
            (line_true,) = ax.plot(x, true_arr[:, c], color="black", linewidth=1.9, label="true")
            if legend_handles is None:
                legend_handles = [line_true]
                legend_labels = ["true"]

            for i, method in enumerate(plot_methods):
                pred = np.asarray(traces[seed][method]["pred_param"], dtype=np.float64)[:t_seed, :]
                label = method_label.get(method, method)
                (line_m,) = ax.plot(x, pred[:, c], color=colors[i], linewidth=1.35, alpha=0.9, label=label)
                if legend_labels is not None and label not in legend_labels:
                    legend_handles.append(line_m)
                    legend_labels.append(label)

            if pname.startswith("w"):
                ax.set_ylim(0.0, 1.0)
            else:
                vmax = max(0.55, float(np.nanmax(true_arr[:, c])) + 0.05)
                ax.set_ylim(0.0, vmax)
            ax.grid(True, alpha=0.25)

            if r == 0:
                ax.set_title(pname)
            if c == 0:
                ax.set_ylabel(f"seed={seed}\nm={m}")
            if r == n_rows - 1:
                ax.set_xlabel("turn")

    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(len(legend_labels), 6),
            frameon=False,
            fontsize="small",
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    else:
        fig.tight_layout()
    png_grid = out_dir / "plot_cases_grid.png"
    fig.savefig(png_grid, dpi=180)

    # Mean-over-cases plot.
    t_common = min(min(int(traces[s][k]["t"]) for k in plot_methods) for s in seeds)
    true_stack = np.stack(
        [np.asarray(traces[s][methods[0]]["true_param"], dtype=np.float64)[:t_common, :] for s in seeds],
        axis=0,
    )
    true_mean = true_stack.mean(axis=0)
    pred_mean: dict[str, np.ndarray] = {}
    for method in plot_methods:
        pred_stack = np.stack(
            [np.asarray(traces[s][method]["pred_param"], dtype=np.float64)[:t_common, :] for s in seeds],
            axis=0,
        )
        pred_mean[method] = pred_stack.mean(axis=0)

    fig2, axes2 = plt.subplots(1, len(PARAM_NAMES), figsize=(len(PARAM_NAMES) * 3.1, 2.9), squeeze=False, sharex=False)
    x2 = np.arange(1, t_common + 1, dtype=np.int32)
    for c, pname in enumerate(PARAM_NAMES):
        ax = axes2[0, c]
        ax.plot(x2, true_mean[:, c], color="black", linewidth=2.0, label="true")
        for i, method in enumerate(plot_methods):
            ax.plot(x2, pred_mean[method][:, c], color=colors[i], linewidth=1.4, alpha=0.95, label=method_label.get(method, method))
        if pname.startswith("w"):
            ax.set_ylim(0.0, 1.0)
        else:
            vmax = max(0.55, float(np.nanmax(true_mean[:, c])) + 0.05)
            ax.set_ylim(0.0, vmax)
        ax.set_title(pname)
        ax.set_xlabel("turn")
        ax.grid(True, alpha=0.25)
    axes2[0, 0].set_ylabel("mean value over seeds")
    handles, labels = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False, fontsize="small")
    fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    png_mean = out_dir / "plot_mean_over_cases.png"
    fig2.savefig(png_mean, dpi=180)

    print(f"[OK] out_dir: {out_dir}")
    print(f"[OK] long_csv: {csv_long}")
    print(f"[OK] cases_grid_png: {png_grid}")
    print(f"[OK] mean_over_cases_png: {png_mean}")


if __name__ == "__main__":
    main()
