# exp002

exp002 は exp001 の反省を踏まえて、実験を回しやすく / 拡張しやすく / 重み管理を簡単にするための整理版です。

## 目標

- 使いやすさ: 1コマンドで学習/再開できる
- 拡張性: `feature_id` / `arch` を増やしやすい
- 重み管理: run ディレクトリに集約して散らからない
- 蒸留: teacher と student で **入力特徴量が違っても**蒸留できる

## 性能チューニングログ

高速化の試行履歴・実測・採用/非採用判断は `exps/exp002/PERF_TUNING_LOG.md` に時系列でまとめています。

## 使い方

### 1) feature 一覧（C++側 registry）

```bash
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.list_features
```

### feature の説明（現行）

exp002 では観測特徴量を `feature_id` で切り替えます（C++ 側の registry で管理）。

共通仕様:
- 盤面入力は `float32` の `[C, 10, 10]`（各 plane は `10x10` に敷き詰め）
- 行動マスクは `[100]`（`reach` に基づき「player0 が到達可能なセル=1」）
- opponent は **スコア降順**で並び替えてから plane に書き込みます（`old_to_new`。`p=0` は不変）
  - 例: 「スコア最大の敵」が常に `p=1` になる

#### `submit_v1`（46ch, 提出対応）

- 実装: `exps/exp002/cpp_core/include/ahc061/core/features.hpp`
- チャネル構成（合計46）:
  - `v_norm`（1）: `value/1000`
  - `l_norm`（1）: `level/u_max`
  - `neutral`（1）: `owner==-1`
  - `owner[p]`（8）: 所有者 one-hot（opponent reorder 後の p）
  - `comp[p]`（8）: 連結成分（領地）マスク（reorder 後）
  - `reach[p]`（8）: 到達可能マスク（reorder 後）
  - `next[p]`（8）: 「次ターンにそのセルへ行く」分布（reorder 後）
    - `p=0` は合法手一様
    - `p>=1` は PF 有効時は PF の混合分布、PF 無効時は合法手一様
  - `score[p]`（8）: 現在スコア（`value*level`）を `500000` で正規化して `[0,1]` にクリップ（reorder 後）
  - `turn_frac`（1）: `turn/t_max`
  - `m_norm`（1）: `m/M_MAX`
  - `u_norm`（1）: `u_max/5`

#### `research_v1`（48ch, 研究用・提出非対応）

- 実装: `exps/exp002/cpp_core/include/ahc061/core/features_research_v1.hpp`
- `submit_v1` に加えて:
  - `pos0_x_norm`（1）: `ex0/(N-1)`
  - `pos0_y_norm`（1）: `ey0/(N-1)`

#### `research_v2`（65ch, 研究用・提出非対応）

- 実装: `exps/exp002/cpp_core/include/ahc061/core/features_research_v2.hpp`
- 大枠は `submit_v1` と同様ですが、次を変更/追加します:
  - `l_norm`（level plane）を `level/u_max` ではなく `level/5` にする
  - `score[p]` を `score_raw/50000`（クリップなし）にする（`score_raw = Σ value*level`）
  - `pos0_x_norm`, `pos0_y_norm` を維持（`research_v1` と同じ）
  - 距離特徴:
    - `dist_owner[p]`（8）: 各セル→「AI p の所有マス集合」までの距離（multi-source BFS、`dist/18`）
    - `dist_comp[p]`（8）: 各セル→「AI p の連結成分マス集合」までの距離（multi-source BFS、`dist/18`）
    - `dist_center`（1）: 各セル→盤面中心（`(4.5,4.5)`）までのマンハッタン距離（`(abs(x-4.5)+abs(y-4.5))/9`）

#### `research_v3`（77ch, 研究用・提出非対応）

- 実装: `exps/exp002/cpp_core/include/ahc061/core/features_research_v3.hpp`
- 基本は `research_v2` と同じですが、次を変更/追加します:
  - `next[p]` の推定:
    - PF（particle filter）混合分布ではなく、`exps/exp003` の **A-softmax（非Eigen）**推定器の推定値を用います
    - 推定器は `delta=(log(wb/wa), log(wc/wa), log(wd/wa))` のガウス近似事後 `N(mu, Σ)` を持ち、`delta` を **中心点なしのUT（2d=6点）**で周辺化して「AI-like 行動分布」を近似します
    - `eps` は点推定（推定器の責務平均）を用います
    - 係数は固定: `tau=0.1`, `prior_std=0.5`, `eps0=0.5`
  - `m` と `u_max`:
    - もともとの `m_norm` / `u_norm`（global 1ch）は **0 埋め**し、代わりに one-hot の global plane を追加します
    - `m_onehot`（7ch）: `m ∈ {2..8}`
    - `u_onehot`（5ch）: `u_max ∈ {1..5}`

#### `research_v4`（149ch, 研究用・提出非対応）

- 実装: `exps/exp002/cpp_core/include/ahc061/core/features_research_v4.hpp`
- `next[p]` の推定:
  - `p=0` は合法手一様
  - `p>=1` は `exps/exp003` の `adf_beta` 推定器で得た平均パラメータを使って AI-like 分布を計算
  - 固定パラメータ: `prior_std=0.325`, `eps0=0.30`
- チャネル順（連続 slice で取り出しやすい順）:
  - global 19ch: `v_norm`, `l_norm(=level/5)`, `neutral`, `turn_frac`, `dist_center`, `x_norm`, `y_norm`, `m_onehot(7)`, `u_onehot(5)`
  - player block 128ch: `8プレイヤー x 16ch`（`owner`, `comp`, `reach`, `next`, `score`, `dist_owner`, `dist_comp`, `owner_level_sum`, `owner_level_value_sum`, `comp_level_sum`, `comp_level_value_sum`, `est_wa_norm`, `est_wb_norm`, `est_wc_norm`, `est_wd_norm`, `est_eps`）
  - tail 2ch: `pos0_x_norm`, `pos0_y_norm`

### 2) PPO 学習

```bash
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.train_ppo \
  --feature-id submit_v1 --arch resnet_v1 --hidden 64 --blocks 6 \
  --updates 500 --batch-size 256 --lr 3e-4 \
  --save-last-every 10 --save-every 50
```

#### 速度最適化オプション（RL）

- `--memory-format {auto,nchw,channels_last}`（default: `auto`）
  - CUDA では `auto` が `local_minibatch` を見て切り替えます（`<=512` は `nchw`、それより大きい場合は `channels_last`）。
  - `dwres_v1` では `minibatch` により有利なレイアウトが変わるため、必要なら固定指定してください。
- `--rollout-cache-device {auto,cpu,gpu}`（default: `auto`）
  - PPO 更新で使う flatten 後テンソル（obs/mask/aux/adv/ret）をどこに置くか。
  - `auto` は空きVRAMと推定サイズから判定します。
  - VRAMに余裕がある環境では `gpu` が有利になりやすいです。

#### Multi-GPU（torchrun / DDP）

`train_ppo` は `torchrun` で multi GPU 実行できます（1 process / 1 GPU）。

```bash
.venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m exps.exp002.ahc061_exp002.scripts.train_ppo \
  --feature-id submit_v1 --arch resnet_v1 --hidden 64 --blocks 6 \
  --updates 500 --batch-size 1024 --minibatch 8192 --lr 3e-4
```

- `--batch-size` と `--minibatch` は **global 値**です（全 rank 合計）。
- 両方とも `WORLD_SIZE` で割り切れる必要があります（例: 4GPU なら `1024/8192`）。
- wandb / checkpoint は rank0 のみが実行します。
- eval は `online/EMA × greedy/sample` のタスク単位で rank 間分散されます。sample 評価も従来実装と一致するように乱数シードを固定しています。
- `--distributed auto` がデフォルトです。`torchrun` で起動していなくても、複数GPUが見えていれば自動で distributed 起動します。従来どおり単一プロセスに固定したい場合は `--distributed off` を指定してください。

#### arch 一覧（現行）

- 既存:
  - `resnet_v1`: 3x3 Residual trunk
  - `dwres_v1`: depthwise + pointwise residual trunk
- 新規（研究用）:
  - `dwres_gc_v1`（`+A`）:
    - Coord 埋め込み（`x,y`）を入力に追加
    - global pooling 分岐で FiLM（`gamma,beta`）を生成して trunk 特徴へ注入
  - `mbconvse_gc_v1`（`+A+B`）:
    - `+A` に加えて block を MBConv-SE（expand → depthwise → SE → project）へ変更
  - `mbconvse_gc_split_v1`（`+A+B+C`）:
    - `+A+B` に加えて stem 後を `shared -> policy/value(branch)` に部分分離
    - aux head は value 側 branch の特徴を使用
  - `dwres_ppconcat_v1`:
    - `dwres_v1` trunk を前半/後半に分割
    - `research_v4` の固定レイアウト（`global=19`, `player=8x16`, `pos0=2`）を前提に、前半で相手 branch（`p=1..7`）を追加
    - player branch の hidden はデフォルトで `main hidden` の半分
    - 各 `p=1..7` branch は次の分割入力で構成:
      - common 側: `global(19ch) + player0 block(16ch) = 35ch`
      - enemy 側: `player p block(16ch)` + `player id embedding`
    - common/enemy 投影を加算して branch front block を通し、`p=1..7` を concat
    - `concat -> 1x1 fuse -> residual add` で main branch に統合し、後半 trunk へ接続
    - `p>=m` は `m_onehot` 由来マスクで merge 前に 0 化
  - `dwres_ppconcat_full_v1`:
    - `dwres_ppconcat_v1` の player branch hidden を `main hidden` と同じ幅にした互換版
  - `dwres_ppconcat_full_pcatonly_v1`:
    - `dwres_ppconcat_full_v1` から main branch 前半（`main_stem`/`main_front`）を削除した版
    - main branch へは `p=1..7` の player branch を concat した特徴のみを入力
    - `concat -> 1x1 fuse -> main_back` で policy/value trunk を構成
  - `dwres_ppconcat_full3x3_v1`:
    - `dwres_ppconcat_v1` の `DW 3x3 + PW 1x1` ブロックを、`通常 3x3 + PW 1x1`（3x3は `groups=1`）へ置換した比較用版
    - trunk 分割 / player-branch concat / merge などの構成は `dwres_ppconcat_v1` と同一
  - `dwres_ppconcat_full3x3_pmix_gamma_v1`（research_v4 専用）:
    - `dwres_ppconcat_full3x3_v1` に対し、player branch (`p=1..7`) の途中に
      「player軸 1x1 residual mixing」を数ブロックごとに挿入
    - 相互作用は attention ではなく shared `P x P` 線形混合（`P=7`）を2段重ねで実装
    - `m_onehot` 由来の active mask を key側/出力側の両方で適用し、`p>=m` のスロット混入を抑制
    - channel-wise gate (`gamma`) と residual gate (`alpha`) で注入量を制御
  - `dwres_pxattn_noffn_v1`（research_v4 専用）:
    - 入力を `8 player` 軸に展開し、各 player に `global(19ch)+player(16ch)` を連結（35ch）
    - 各 block で「per-pixel の player self-attention（player間のみ）」を適用
    - FFN は入れず、`Add + LayerNorm` の後に `DWResidualBlock` で空間方向を処理
    - `m_onehot` から active player mask を作り、`p>=m` のスロットは無効化

#### 新規archの学習例

```bash
# +A
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.train_ppo \
  --feature-id research_v3 --arch dwres_gc_v1 --hidden 128 --blocks 16

# +A+B
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.train_ppo \
  --feature-id research_v3 --arch mbconvse_gc_v1 --hidden 128 --blocks 16

# +A+B+C
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.train_ppo \
  --feature-id research_v3 --arch mbconvse_gc_split_v1 --hidden 128 --blocks 16

# player-branch concat (research_v4 専用)
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.train_ppo \
  --feature-id research_v4 --arch dwres_ppconcat_v1 --hidden 128 --blocks 16

# player-branch concat (通常3x3版, research_v4 専用)
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.train_ppo \
  --feature-id research_v4 --arch dwres_ppconcat_full3x3_v1 --hidden 128 --blocks 16

# player-concat only trunk (main前半なし, research_v4 専用)
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.train_ppo \
  --feature-id research_v4 --arch dwres_ppconcat_full_pcatonly_v1 --hidden 128 --blocks 16

# player-branch concat + non-attention player mixing(gamma), research_v4 専用
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.train_ppo \
  --feature-id research_v4 --arch dwres_ppconcat_full3x3_pmix_gamma_v1 --hidden 128 --blocks 16

# per-pixel player attention (FFNなし, research_v4 専用)
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.train_ppo \
  --feature-id research_v4 --arch dwres_pxattn_noffn_v1 --hidden 64 --blocks 16
```

- 補助ロス（aux loss）をデフォルトで入れます（係数は調整可能）。
  - `--ent-coef`: 方策エントロピー正則化係数（デフォルト: `0.01`）
  - `--aux-opp-move-coef`: 相手の次手分布（真の `OpponentParam` から計算した分布）に対するKL（Cross Entropy）
  - `--aux-opp-param-coef`: 相手の隠れパラメータ推定（`w=wa..wd` を **和が1になるよう正規化**、`eps` はそのまま）
  - 分布/真値ターゲットの計算は C++ 側で行います（Pythonで重い計算をしない設計）
- EMA（重みの指数移動平均）は `--ema-decays` で複数指定できます（例: `--ema-decays 0.995,0.999,0.9995`）。
  - 無効化は `--ema-decays off`
- `--eval-seeds > 0` のとき、**初回評価**（upd=0 相当）も必ず実行します。
- `--eval-every K` は update 間隔です。

#### 高速化オプション

- CUDA のとき、update は bf16 AMP を利用します（無効化: `--no-amp`）。
- `torch.compile`（有効化: `--compile`）で update を高速化できます（CUDA のときのみ）。
  - `--compile-mode` でモードを指定できます（デフォルト: `default`）。
  - 初回のみコンパイル時間が乗ります（長時間学習なら回収できる想定）。
- `--no-shuffle-minibatches`
  - PPO update のミニバッチを連続sliceで処理します（sample-level shuffle を無効化）。
  - 速度優先オプションです（学習の確率的性質は弱くなります）。
- `--aux-opp-move-coef 0 --aux-opp-param-coef 0` を同時指定した場合
  - rollout での `aux_targets_into` 呼び出しを自動でスキップします（追加で高速化）。
  - 当然ですが、aux loss は学習されなくなります。

#### 高速化の推奨プリセット（`research_v3` + `dwres_v1`）

`2026-02-17` 時点の手元計測では、下記設定が
`--minibatch 1024` の設定より高い `sps` を出しやすいです。

```bash
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.train_ppo \
  --feature-id research_v3 --arch dwres_v1 --hidden 128 --blocks 16 \
  --run-name rl_research_v3 \
  --updates 1000000 --batch-size 512 --epochs 1 --minibatch 512 \
  --compile --lr 2.5e-5 \
  --save-last-every 25 --save-every 200 \
  --eval-seeds 1000 --eval-every 200 \
  --device auto --rollout-amp --no-pf --warmup-updates 20 \
  --memory-format auto --rollout-cache-device auto \
  --torch-num-threads 8 --torch-num-interop-threads 1 \
  --seed 2134890
```

さらに速度優先（auxを切る）なら、上記に次を追加します。

```bash
--aux-opp-move-coef 0 --aux-opp-param-coef 0
```

#### 実測環境と実測値（`2026-02-17`）

環境:
- OS: `Linux 6.13.0-061300-generic`
- CPU: `AMD Ryzen 9 9950X (16C/32T)`
- GPU: `NVIDIA GeForce RTX 5090`
- PyTorch: `2.10.0+cu128`（CUDA `12.8`）
- 実行時 thread 指定: `--torch-num-threads 8 --torch-num-interop-threads 1`

計測方法:
- `train_ppo` は `updates=12` で実行し、`upd=5..12` の `sps` 平均を採用。
- `wandb` は `--wandb-mode disabled`。

`train_ppo`（`research_v3 + dwres_v1`, `batch=512`, `minibatch=512`）:

```bash
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.train_ppo \
  --feature-id research_v3 --arch dwres_v1 --hidden 128 --blocks 16 \
  --updates 12 --batch-size 512 --epochs 1 --minibatch 512 \
  --compile --lr 2.5e-5 --device auto --rollout-amp --no-pf \
  --warmup-updates 20 --memory-format auto --rollout-cache-device auto \
  --torch-num-threads 8 --torch-num-interop-threads 1 \
  --save-last-every 0 --save-every 0 --eval-seeds 0 --eval-every 0 \
  --seed 2134890 --wandb-mode disabled
```

- aux on（default）: `avg_sps_after4 = 28013.4`
- aux off（`--aux-opp-move-coef 0 --aux-opp-param-coef 0`）: `avg_sps_after4 = 28705.8`

`bench_rollout`（同一モデル条件、`batch=512`, `t_max=100`）:

```bash
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.bench_rollout \
  --mode both --feature-id research_v3 --arch dwres_v1 --hidden 128 --blocks 16 \
  --batch-size 512 --t-max 100 --episodes 20 --warmup 2 \
  --device auto --compile --amp --no-pf --memory-format nchw \
  --torch-num-threads 8 --torch-num-interop-threads 1
```

- `env(step_observe_into)`: `73118.7 sps`
- `env(step_observe_into+aux_targets)`: `69267.9 sps`
- `collect_rollout`（aux targetあり）: `53939.9 sps`

```bash
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.bench_rollout \
  --mode rollout --feature-id research_v3 --arch dwres_v1 --hidden 128 --blocks 16 \
  --batch-size 512 --t-max 100 --episodes 20 --warmup 2 \
  --device auto --compile --amp --no-pf --memory-format nchw \
  --no-rollout-aux-targets \
  --torch-num-threads 8 --torch-num-interop-threads 1
```

- `collect_rollout`（`--no-rollout-aux-targets`）: `56745.3 sps`

### 3) 完全再開（optimizer/RNG含む）

`--resume` に run ディレクトリ、または `ckpt_last.pt` を直接渡せます。

```bash
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.train_ppo \
  --resume exps/exp002/artifacts/runs/20250101_000000_xxx \
  --updates 800
```

- resume 時は **ckpt優先で strict**（差分があれば上書き）です。
  - `arch/feature_id/hidden/blocks/batch_size/lr/...` は ckpt に合わせます
  - `ema-decays` も ckpt の設定に合わせます（full ckpt にEMAが入っている場合）
  - `--updates`, `--save-*`, `--eval-*`, `--wandb-*` は変更できます
- wandb を有効にしていた run は、同一 run id を使って resume します（`resume="allow"`）。

### 4) 蒸留（teacher/student で feature が違ってもOK）

teacher は `--teacher-ckpt` で指定し、student は arch/サイズ/feature を指定します。

```bash
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.distill \
  --teacher-ckpt exps/exp001/artifacts/checkpoints/ckpt_0850.pt \
  --teacher-feature-id submit_v1 \
  --student-feature-id research_v1 \
  --student-arch dwres_v1 --student-hidden 64 --student-blocks 16 \
  --updates 200 --batch-size 1024 --epochs 1 --minibatch 1024 --lr 1e-3 \
  --compile --compile-teacher --compile-mode default \
  --wandb-mode online \
  --out-ckpt exps/exp002/artifacts/checkpoints_distill/ckpt_dw64b16_from_0850.pt
```

#### 蒸留の Multi-GPU（torchrun / DDP）

`distill` も `train_ppo` と同様に `torchrun` で multi GPU 実行できます（1 process / 1 GPU）。

```bash
.venv/bin/torchrun --standalone --nproc_per_node=4 \
  -m exps.exp002.ahc061_exp002.scripts.distill \
  --teacher-ckpt exps/exp001/artifacts/checkpoints/ckpt_0850.pt \
  --teacher-feature-id submit_v1 \
  --student-feature-id research_v1 \
  --student-arch dwres_v1 --student-hidden 64 --student-blocks 16 \
  --updates 200 --batch-size 1024 --epochs 1 --minibatch 4096 --lr 1e-3 \
  --out-ckpt exps/exp002/artifacts/checkpoints_distill/ckpt_dw64b16_from_0850.pt
```

- `--batch-size` と `--minibatch` は **global 値**です（全 rank 合計）。
- 両方とも `WORLD_SIZE` で割り切れる必要があります。
- wandb / checkpoint は rank0 のみが実行します。
- `--distributed auto` がデフォルトです。`torchrun` で起動していなくても、複数GPUが見えていれば自動で distributed 起動します。単一プロセス固定は `--distributed off` を指定してください。

- 蒸留時も aux loss を入れられます（teacher 出力ではなく **真値** を教師にします）。
  - `--w-aux-opp-move`, `--w-aux-opp-param`

蒸留 ckpt は `train_ppo --init-ckpt ...` で RL 微調整できます（feature/arch/サイズが一致している必要があります）。

### 4-b) 評価（tools固定）

学習スクリプト側の `--eval-*` に加えて、単発で tools の greedy 評価を回すスクリプトがあります。
`eval_tools` は wandb を使わず、標準出力に平均スコアを表示しつつ
`psytester r <run_name>` と同じ形式で結果ファイルを作成します。

```bash
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.eval_tools \
  --ckpt exps/exp002/artifacts/runs/.../checkpoints/ckpt_last.pt \
  --run-name my_eval_run \
  --seed-begin 0 --seed-end 9 --batch 16
```

生成物:
- `results/my_eval_run.res`（JSONL: `{"id": <seed>, "score": <score>}`）
- `tests/my_eval_run/out/<seed>.out`
- `tests/my_eval_run/res/<seed>.res`

TTA を使う場合（`make_submit_compact` と同じ定義）:
- `--tta-mode 0`: TTAなし
- `--tta-mode 1`: sum-prob（`argmax_a log(sum_k p_k(a))`）
- `--tta-mode 2`: prod-prob（`argmax_a sum_k log p_k(a)`）
- `--tta-k`: 変換数 `2/4/8`（`--tta-mode 1/2` で有効）

追加オプション:
- `--copmile`（`--compile` でも可）: `torch.compile(mode="default")` を有効化（CUDA時のみ）
- `--torch-num-threads`: intra-op の CPU スレッド数
- `--torch-num-interop-threads`: inter-op の CPU スレッド数

### 4-c) ベンチ

`train_ppo` のログにも `sps`（step/sec）や `time/*`（wandb有効時）が出ますが、rollout 部分だけ切り出して計測したい場合は
`bench_rollout` を使ってください（`reset_random` を使うので tools 入力の I/O は含みません）。

```bash
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.bench_rollout \
  --mode both --feature-id research_v2 --arch dwres_v1 --hidden 256 --blocks 32 \
  --batch-size 256 --t-max 100 --episodes 50 --device auto --compile --amp
```

- `--mode env`: `step_observe_into`（環境のみ）
- `--mode env_aux`: `step_observe_into + aux_targets_into`（環境+auxターゲット生成）
- `--mode rollout`: `collect_rollout`（環境+モデル推論）
- `--no-rollout-aux-targets`: `collect_rollout` で aux ターゲット生成を省いて純粋な rollout 速度を計測

補足:
- PF（particle filter）は rollout 速度に大きく影響します。最大速度を見たいときは `--no-pf` も試してください（観測の `next[p]` が一様になります）。
- PF の粒子数は環境変数 `AHC061_PF_PARTICLES` で調整できます（小さいほど速い）。例: `AHC061_PF_PARTICLES=64 ...`。

### 5) 提出用 `Main.cpp` 生成（TorchScript埋め込み）

現状、提出対応 feature は `submit_v1` のみです（研究用 feature は `submit_supported=false`）。

```bash
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.make_submit_torchscript \
  --ckpt exps/exp002/artifacts/runs/.../checkpoints/ckpt_last.pt \
  --out-dir exps/exp002/submit
```

`exps/exp002/submit/Main.cpp` が生成されます。

#### TTA（test time augmentation）

提出側の行動選択は、盤面の回転/反転（8通り）で推論して確率を結合する TTA に対応しています。
`exps/exp002/submit/solver_base.cpp` の `AHC061_EXP002_TTA_MODE` を編集して切り替えます（生成後に `Main.cpp` を直接編集してもOK）。

- `0`: TTAなし（single forward）
- `1`: TTAあり（sum, デフォルト）
- `2`: TTAあり（prod）

#### compact版（`solver_base_compact.cpp` / `make_submit_compact.py`）

`checkpoints/ckpt_exp002_070_7650.pt` を compact 形式で埋め込む場合は次を使います。

1) `Main.cpp` 生成:

```bash
./.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.make_submit_compact \
  --ckpt checkpoints/ckpt_exp002_070_7650.pt \
  --out-dir exps/exp002/submit
```

`dwres_ppconcat_v1`（例: `test.pt`）向けには `--ppconcat-preset` で圧縮強度を切り替えできます。

```bash
# 非劣化優先（安全側）
./.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.make_submit_compact \
  --ckpt test.pt --ppconcat-preset fp16_blockpw_i8 \
  --out-dir exps/exp002/submit

# 非劣化を狙いつつさらに縮小
./.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.make_submit_compact \
  --ckpt test.pt --ppconcat-preset fp16_blockpw_merge_i8 \
  --out-dir exps/exp002/submit

# c7（保守圧縮ショートカット）
# = fp16_custom_i8 + (main/player block pw 全i8) + (merge main/player i8)
./.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.make_submit_compact \
  --ckpt test2_ver2.pt --ppconcat-preset c7 --payload-encoding huff122 \
  --out-dir exps/exp002/submit

# 強圧縮（サイズ優先、劣化リスクあり）
./.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.make_submit_compact \
  --ckpt test.pt --ppconcat-preset mixq \
  --out-dir exps/exp002/submit

# test2.pt（research_v4 + dwres_ppconcat_v1）で no-int4 / 非劣化優先かつ最小サイズ
./.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.make_submit_compact \
  --ckpt test2.pt --ppconcat-preset fp8aux_blockpw_merge_i8 \
  --out-dir exps/exp002/submit

# test2_ver2.pt（1000ケース非劣化で現状最小）
./.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.make_submit_compact \
  --ckpt test2_ver2.pt \
  --ppconcat-preset fp16_custom_i8 \
  --ppconcat-fp8-aux-mask 0x7f \
  --ppconcat-main-front-i8-mask 0xff \
  --ppconcat-main-back-i8-mask 0xff \
  --ppconcat-player-front-i8-mask 0xff \
  --ppconcat-player-front-i4-mask 0xff \
  --ppconcat-main-back-i4-mask 0x5 \
  --ppconcat-merge-main-i8 \
  --ppconcat-merge-player-i8 \
  --out-dir exps/exp002/submit
```

- `--ppconcat-preset` の選択肢:
  `mixq`, `i8all`, `fp16`, `fp16_merge_i8`, `fp16_custom_i8`, `c7`,
  `fp16_blockpw_i8`, `fp16_blockpw_merge_i8`,
  `fp16_blockpw_i8_merge_i4`, `fp16_blockpw_i4_merge_i8`, `fp16_blockpw_i4_merge_i4`,
  `fp16_blockpw_custom`, `fp8aux_blockpw_merge_i8`, `fp8aux_custom`, `fp8full`
- `--payload-encoding`:
  - `base91`（既定）
  - `base122`（より高圧縮。`Main.cpp` 文字列を `u8""` で埋め込む）
  - `huff122`（Huffman 圧縮後に `base122` 化。さらに高圧縮）
  - `huff91`（Huffman 圧縮後に `base91` 化。ASCII 安全）
    - 現状は以下の特化モードのみ対応
      - `dwres_ppconcat_v1 + c7 + hidden=112 + blocks in (16,18) + player_hidden=56`
      - `dwres_ppconcat_full_v1 + c7 + hidden=96 + blocks=17 + player_hidden=96`
- 手元1000ケース比較では、
  `test.pt` 系は `fp16_blockpw_i8` / `fp16_blockpw_merge_i8`、
  `test2.pt` 系は `fp8aux_blockpw_merge_i8`、
  `test2_ver2.pt` 系は上記 `fp16_custom_i8 + fp8_aux_mask=0x7f + main_back_i4_mask=0x5` が参照比で非劣化かつ最小でした。

`test2_ver2.pt` の同一量子化設定（1000ケース）では、
`base91: 617348 bytes` -> `base122: 583364 bytes` -> `huff122: 551323 bytes` と縮小でき、
`huff122` は `base122` 比で `-32041 bytes`（`base91` 比で `-66025 bytes`）でした。
`base122` と `huff122` の `.res` は `cmp` で完全一致でした。

1-b) `O/P/M` 5ケース比較（`seed=20000-29999`）:

- `O`: それ以外（stem/dw/gn/policy など）
- `P`: `PW(1x1)`（block pw）
- `M`: `merge`（`merge_fuse_main/player`）
- 5ケース:
  - `O=fp16,P=fp16,M=fp16` -> `--ppconcat-preset fp16`
  - `O=fp16,P=fp16,M=int8` -> `--ppconcat-preset fp16_merge_i8`
  - `O=fp16,P=int8,M=fp16` -> `--ppconcat-preset fp16_blockpw_i8`
  - `O=fp16,P=int8,M=int8` -> `--ppconcat-preset fp16_blockpw_merge_i8`
  - `O=int8,P=int8,M=int8` -> `--ppconcat-preset i8all`

```bash
CKPT="checkpoints/ckpt_distill_r4_dwpp_h112b20_from_136_7450_ema_0p99999_ver1_pt_ema_0p999.pt"
SEEDS="20000-29999"
THREADS=20

run_case () {
  RUN="$1"
  PRESET="$2"
  OUT="exps/exp002/_ablate/${RUN}"

  ./.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.make_submit_compact \
    --ckpt "$CKPT" --ppconcat-preset "$PRESET" --out-dir "$OUT"
  g++-12 -std=c++20 -O3 -w -Iexps/exp002/cpp_core/include "$OUT/Main.cpp" -o main
  ./.venv/bin/psytester r -t "$SEEDS" -m "$THREADS" "$RUN"
}

run_case ablate_q_oF_pF_mF_h112b20_v1 fp16
run_case ablate_q_oF_pF_mI_h112b20_v1 fp16_merge_i8
run_case ablate_q_oF_pI_mF_h112b20_v1 fp16_blockpw_i8
run_case ablate_q_oF_pI_mI_h112b20_v1 fp16_blockpw_merge_i8
run_case ablate_q_oI_pI_mI_h112b20_v1 i8all

./.venv/bin/psytester s --files 'ablate_q_.*_h112b20_v1\.res$'
```

`psytester r` 実行後は、各 run について `results/<run>.res` と
`tests/<run>/out`, `tests/<run>/res` が作成されます。

2) `K` ごとのコンパイル例（`TTA_MODE=1` = sum-prob）:

```bash
# K=1
g++-12 -std=c++20 -O3 -w -Iexps/exp002/cpp_core/include \
  -DAHC061_EXP002_TTA_MODE=1 -DAHC061_EXP002_TTA_K=1 \
  exps/exp002/submit/solver_base_compact.cpp -o main.k1

# K=2
g++-12 -std=c++20 -O3 -w -Iexps/exp002/cpp_core/include \
  -DAHC061_EXP002_TTA_MODE=1 -DAHC061_EXP002_TTA_K=2 \
  exps/exp002/submit/solver_base_compact.cpp -o main.k2

# K=4
g++-12 -std=c++20 -O3 -w -Iexps/exp002/cpp_core/include \
  -DAHC061_EXP002_TTA_MODE=1 -DAHC061_EXP002_TTA_K=4 \
  exps/exp002/submit/solver_base_compact.cpp -o main.k4

# K=8
g++-12 -std=c++20 -O3 -w -Iexps/exp002/cpp_core/include \
  -DAHC061_EXP002_TTA_MODE=1 -DAHC061_EXP002_TTA_K=8 \
  exps/exp002/submit/solver_base_compact.cpp -o main.k8
```

`make_submit_compact` 後の `exps/exp002/submit/Main.cpp` も同じ `-D` で切り替えできます。

```bash
g++-12 -std=c++20 -O3 -w -Iexps/exp002/cpp_core/include \
  -DAHC061_EXP002_TTA_MODE=1 -DAHC061_EXP002_TTA_K=4 \
  exps/exp002/submit/Main.cpp -o exps/exp002/submit/solver_compact_k4
```

3) 単一 seed 実行例:

```bash
./tools/target/release/tester ./main.k4 < tools/in/0000.txt > out
```

#### ローカルで（AtCoder相当の）ビルドして動作確認

このリポジトリでは `.venv` の torch/libtorch を使ってリンクします。

```bash
TORCH_DIR=$(./.venv/bin/python -c 'import torch,os; print(os.path.dirname(torch.__file__))')
g++ exps/exp002/submit/Main.cpp -o exps/exp002/submit/solver \
  -O2 -std=gnu++23 -Wall -Wextra -pedantic \
  -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
  -L"$TORCH_DIR/lib" -Wl,-rpath,"$TORCH_DIR/lib" \
  -Wl,--no-as-needed -ltorch -ltorch_cpu -lc10 -Wl,--as-needed \
  -pthread -ldl
./tools/target/release/tester exps/exp002/submit/solver < tools/in/0000.txt > out
```

#### 注意（ファイルサイズ）

AtCoder のソースサイズ制限（512 KiB）に引っかかる場合は **モデル縮小**が必要です。

```bash
wc -c exps/exp002/submit/Main.cpp
```

## 重みの置き場（run dir）

`--run-dir` を指定しない場合:

- wandb の自動名（例: `trim-haze-52`）を使う場合（`--wandb-name` 未指定、末尾番号を取得できた場合）:
  `exps/exp002/artifacts/runs/<NNN>_<timestamp>_<run-name>/`（例: `052_...`）
- それ以外:
  `exps/exp002/artifacts/runs/<timestamp>_<run-name>/`
  - `config.json`
  - `checkpoints/ckpt_last.pt`（model-only）
  - `checkpoints_ema/manifest.json`（EMA一覧: name/decay/steps）
  - `checkpoints_ema/ema_decay_0p999/ckpt_last.pt`（EMA model-only, decayごとのサブディレクトリ）
  - `checkpoints_full/ckpt_last.pt`（optimizer/RNG含む）
  - `wandb/`（wandbログ）

## 拡張ポイント

### feature を追加したい

- C++: `exps/exp002/cpp_core/include/ahc061/core/feature_registry.hpp` に feature を追加
  - 実装は `features_*.hpp` を増やして `write_from_common` を登録
  - 提出対応にする場合は `submit_supported=true` にする（提出側の C++ も対応が必要）
- Python: `scripts/list_features.py` で確認できます

### arch を追加したい

- Python: `exps/exp002/ahc061_exp002/models.py` にモデルを追加し、`build_policy_value_model()` に登録
- 提出生成（`make_submit_torchscript`）互換のため、`forward_policy(board)` を実装するか、`stem/blocks/policy_head` を持つ構造にしてください
