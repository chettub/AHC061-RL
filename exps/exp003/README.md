# exp003

exp003 は「隠れパラメータ推定器」の**推定のみ**を定量比較するための実験用ディレクトリです。

まずは PF（particle filter）の粒子数 `P` を変えたときに、ターンごとの推定誤差がどう変化するかを可視化します。

## 現在の推奨設定（adf_beta）

大規模比較（`exps/exp003/artifacts/large_survey_20260217_154111`）時点では、`adf_beta` の最良設定は以下です。

- `--prior-std 0.35 --eps0 0.30`

追記: 精密探索（`exps/exp003/artifacts/search_tuned_adf_hybrid_20260217_164050`）では、以下がさらに僅差で改善しました。

- `--prior-std 0.325 --eps0 0.30`

再現コマンド例:

```bash
.venv/bin/python -m exps.exp003.ahc061_exp003.scripts.bench_pf_particles \
  --methods adf_beta \
  --seed-begin 0 --seed-end 999 --t-max 100 \
  --pf-particles 128 \
  --prior-std 0.325 --eps0 0.30
```

## 使い方（PF 粒子数比較）

```bash
.venv/bin/python -m exps.exp003.ahc061_exp003.scripts.bench_pf_particles \
  --seed-begin 0 --seed-end 999 \
  --pf-particles 16,64,256,1024
```

- デフォルトでは `exps/exp003/artifacts/bench_pf_particles/` に CSV と PNG を出力します。
- C++ 拡張は粒子数ごとにビルドしてキャッシュします（初回のみ時間がかかります）。
- 追加で throughput（opponent-update/sec）の CSV/PNG も出力します。

## 比較手法 A（softmax + 決定的更新）も含める

```bash
.venv/bin/python -m exps.exp003.ahc061_exp003.scripts.bench_pf_particles \
  --methods pf,a \
  --pf-particles 16,64,256,1024 \
  --tau 0.1 --prior-std 0.5 --eps0 0.5
```

## 比較手法 A（不等式 + トランケートガウス）も含める

```bash
.venv/bin/python -m exps.exp003.ahc061_exp003.scripts.bench_pf_particles \
  --methods pf,a,ineq \
  --pf-particles 16,64,256,1024 \
  --tau 0.1 --prior-std 0.5 --eps0 0.5
```

## Eigen 固定サイズ行列版（高速化）

- `a_eigen`: A-softmax の Eigen 固定サイズ版（`Eigen::Matrix<double,3,3>`）
- `ineq_eigen`: A-ineq の Eigen 固定サイズ版（`Eigen::Matrix<double,3,3>`）

Eigen が `#include <Eigen/Dense>` できる環境であれば、自動で有効になります（見つからない場合は `EIGEN3_INCLUDE_DIR` を設定してください）。

```bash
.venv/bin/python -m exps.exp003.ahc061_exp003.scripts.bench_pf_particles \
  --methods pf,a,a_eigen,ineq,ineq_eigen \
  --pf-particles 16,64,256,1024 \
  --tau 0.1 --prior-std 0.5 --eps0 0.5
```

## 追加した提案 A/B/C/D

- `is`: 提案A（固定サポート点の deterministic importance sampling; Halton 列）
  - `--is-points` で点数 K を指定（例: `--is-points 128,256`）
- `adf`: 提案B（不等式 + トランケートガウス）※ `ineq` と同等（別名）
- `adf_beta`: 提案B（不等式 + トランケートガウス）+ εを Beta（一次尤度）で解析更新
- `softmax_full`: 提案C（合法手全体の softmax + オンラインLaplace更新）
- `grid`: 提案D（3次元グリッドのヒストグラムフィルタ）
  - `--grid-n` で解像度（1軸あたり）を指定

例（全部まとめて比較）:

```bash
.venv/bin/python -m exps.exp003.ahc061_exp003.scripts.bench_pf_particles \
  --methods pf,is,a,softmax_full,ineq,grid \
  --pf-particles 16,64,256,1024 \
  --is-points 128,256 \
  --grid-n 11 \
  --tau 0.1 --prior-std 0.5 --eps0 0.5
```

## v4 checkpoint の隠れパラメータ推定精度比較

`exp003` には、学習済み ckpt の `opp_param_head` を直接評価するベンチも追加しています。

- 指標: `mean_w_l1`, `mean_eps_abs`, `mean_mae5=(w_l1+eps_abs)/5`, `updates_per_sec`
- 出力: `summary.csv`, `per_turn.csv`, `plot_mean_mae5.png`, `plot_per_turn_mae5.png`

`*_0p999.pt` のうち更新日時が最新の 1 本だけ評価する例:

```bash
.venv/bin/python -m exps.exp003.ahc061_exp003.scripts.bench_ckpt_hidden_estimation \
  --ckpt-glob 'checkpoints/*_0p999.pt' \
  --latest-only \
  --seed-begin 0 --seed-end 999 --t-max 100 \
  --batch-size 64 --device auto
```

複数 ckpt を明示して比較する例:

```bash
.venv/bin/python -m exps.exp003.ahc061_exp003.scripts.bench_ckpt_hidden_estimation \
  --ckpts checkpoints/ckpt_exp002_106_7150.pt,checkpoints/ckpt_exp002_106_7150_ema_0p999.pt \
  --seed-begin 0 --seed-end 999 --t-max 100 \
  --batch-size 64 --device auto
```

## ckpt と既存手法を同一プロットで比較

`bench_pf_particles` の既存手法結果と、`bench_ckpt_hidden_estimation` の ckpt 結果を統合して、

- `plot_combined_mean_mae5.png`
- `plot_combined_scatter_mae5_vs_throughput.png`
- `plot_combined_per_turn_mae5.png`
- `plot_combined_mean_sqe5.png`
- `plot_combined_scatter_sqe5_vs_throughput.png`
- `plot_combined_per_turn_sqe5.png`

を出力できます。

ここで `sqe5` は `mae5` の squared 版として `sqe5 = mae5^2` を使っています。

例:

```bash
.venv/bin/python -m exps.exp003.ahc061_exp003.scripts.plot_ckpt_vs_existing \
  --methods-csv exps/exp003/artifacts/bench_pf_particles/exp003_methods_pf-adf_beta-adf_beta_ep-hybrid_adf_rbpf_seeds0-999_t100_20260219_132526.csv \
  --throughput-csv exps/exp003/artifacts/bench_pf_particles/exp003_throughput_methods_pf-adf_beta-adf_beta_ep-hybrid_adf_rbpf_seeds0-999_t100_20260219_132526.csv \
  --ckpt-summary-csv exps/exp003/artifacts/bench_ckpt_hidden_estimation_20260219_131956/summary.csv \
  --ckpt-per-turn-csv exps/exp003/artifacts/bench_ckpt_hidden_estimation_20260219_131956/per_turn.csv
```

## 真値と推定値のターン依存トレース（複数ケース）

複数 seed について、`w0..w3` と `eps` の

- 真値（黒線）
- 複数手法の推定値（色線）

を turn 軸で重ね描きできます（漸近挙動確認用）。

例:

```bash
.venv/bin/python -m exps.exp003.ahc061_exp003.scripts.plot_hidden_param_trajectories \
  --seeds 0,1,2,3,4,5 \
  --t-max 100 \
  --methods pf,adf_beta,adf_beta_ep,hybrid_adf_rbpf \
  --pf-particles 128 \
  --prior-std 0.325 --eps0 0.30 --rbpf-particles 64
```

出力:

- `trajectories_long.csv`（long形式データ）
- `plot_cases_grid.png`（行=case(seed), 列=`w0,w1,w2,w3,eps`）
- `plot_mean_over_cases.png`（複数case平均の推移）

`0p999` ckpt の推定結果を同じ図に重ねる例:

```bash
.venv/bin/python -m exps.exp003.ahc061_exp003.scripts.plot_hidden_param_trajectories \
  --seeds 0,1,2,3,4,5 \
  --t-max 100 \
  --methods pf,adf_beta,adf_beta_ep,hybrid_adf_rbpf \
  --pf-particles 128 \
  --prior-std 0.325 --eps0 0.30 --rbpf-particles 64 \
  --ckpt checkpoints/ckpt_exp002_106_7150_ema_0p999.pt \
  --ckpt-policy random_xorshift
```
