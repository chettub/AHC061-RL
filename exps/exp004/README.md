# exp004

`B(M,U)`（ケース難度の proxy）を、ローカルの `results/*.res` と `tools/in/*.txt` から解析・可視化するディレクトリです。
加えて、`train_ppo` の報酬再重み付けで使う
`b_mu_reward_table.csv` と `reward_calibration_fixed_alpha.json` も生成します。

## 実行

```bash
./.venv/bin/python -m exps.exp004.ahc061_exp004.scripts.analyze_b_mu
```

主なオプション:

- `--results-glob`: 解析対象の result ファイル（既定: `results/*.res`）
- `--tools-dir`: `tools/in` の場所（既定: `tools/in`）
- `--min-files-per-id`: その case id を採用する最小出現回数（既定: `1`）
- `--output-dir`: 出力先（既定: `exps/exp004/artifacts/b_mu_analysis`）
- `--reward-b-column`: `B(M,U)` として使う列（既定: `mean_best`）

## 相対スコア平均ヒートマップ（problem.md 準拠）

`problem.md` の相対評価スコア
`round(1e9 * 自身の絶対スコア / 全参加者中の最大絶対スコア)` に合わせて、
ローカル比較では「分母 = 参照 result 群の case ごとの最大絶対スコア」として計算します。

```bash
./.venv/bin/python -m exps.exp004.ahc061_exp004.scripts.analyze_relative_mu \
  --reference-results-glob "results/*.res" \
  --target-results-glob "results/test22.res" \
  --output-dir "exps/exp004/artifacts/relative_mu_test22"
```

主なオプション:

- `--reference-results-glob`: 相対スコア分母を作る参照 result 群（既定: `results/*.res`）
- `--target-results-glob`: ヒートマップ対象 result（必須）
- `--min-ref-files-per-id`: 採用する case id の最小参照ファイル数（既定: `1`）
- `--tools-dir`: `tools/in` の場所（既定: `tools/in`）
- `--output-dir`: 出力先（既定: `exps/exp004/artifacts/relative_mu_analysis`）

主な出力:

- `reference_case_best.csv`: case ごとの参照最大絶対スコア
- `target_relative_case_scores.csv`: target の case ごとの相対スコア
- `relative_mu_summary.csv`: target の `(M,U)` ごとの相対スコア要約
- `relative_mu_heatmap.png`: `(M,U)` ごとの平均相対スコアヒートマップ（target 1 ファイル時）

## 出力

- `file_summary.csv`: 各 `.res` ファイルの要約
- `id_summary.csv`: 各 case id の要約（best/mean/std, M, U など）
- `b_mu_summary.csv`: `(M,U)` ごとの `B(M,U)` proxy 要約
- `b_mu_reward_table.csv`: 報酬重みテーブル
  - 列: `m,u,n_case,b_value,weight_raw,weight_norm,case_prob`
  - `weight_raw=1/b_value`
  - `weight_norm` はケース分布で平均1に正規化した重み
- `reward_calibration_fixed_alpha.json`: 固定 `alpha` の較正結果
  - `alpha_fixed`
  - `old_return_rms`, `weighted_return_rms`
  - `reward_formula` など
- `b_mu_heatmaps.png`: `(M,U)` ヒートマップ（mean/median/p90/count）
- `b_mu_boxplot.png`: `(M,U)` ごとの分布 boxplot
- `b_mu_lines.png`: `M` に対する `U` 別ライン（mean + IQR）

## train_ppo 連携時のデフォルトパス（想定）

- `exps/exp004/artifacts/b_mu_analysis/b_mu_reward_table.csv`
- `exps/exp004/artifacts/b_mu_analysis/reward_calibration_fixed_alpha.json`
