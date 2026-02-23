# exp001: PPO（自前実装）+ C++(torch extension) Simulator

## 目的
- `tools/in/*.txt` の **完全再現**（AI params + r1/r2）を前提に、C++で高速シミュレータ＋特徴量生成を行う。
- Python（PyTorch）側でPPO + ResNetを学習・評価できるようにする。

## 観測（入力特徴量）
- `board`: `float32 [B, C, 10, 10]`
- `mask`: `uint8 [B, 100]`（player0 の合法手マスク）
- `C = 46`

### セル index
- 盤面座標 `(x, y)`（0-index）を `idx = x*10 + y`（`0..99`）に 1 次元化して扱う
  - action も `idx`（到達セル）を直接選ぶ

### `mask`（合法手）
- `mask[idx] = 1` ⇔ player0 が `idx` に移動可能（公式ツールと同順の BFS 列挙に基づく）
  - 自分の連結成分（自領土）からのみ BFS 展開し、到達したセルのうち「他プレイヤーの駒位置」を除いたセルが合法手
  - 合法手が空になるケースは `start`（現在位置）を唯一の合法手として補う

### `board` のチャネル定義（46ch）
以下、各チャネルは `10x10` 平面。

- `p=0` は常に player0（学習する側）
- `p>=1` は「相手スロット」で、観測生成時点の `score` 降順に並べ替えられる（同点は元のID昇順で安定化）
  - つまり `p>=1` の `owner/comp/reach/next/score` は「元のプレイヤーID」ではなく「スコア順位の相手」を表す
  - `m_norm`（=m/8）で実際の人数 `m` は別途与えられる

- `0`: `V_norm = value/1000`
- `1`: `L_norm = level/u_max`
- `2`: `neutral`（`owner=-1` のセルに 1）
- `3..10`（8枚）: `owner[p]` one-hot（セルの `owner==p` に 1）
- `11..18`（8枚）: `comp[p]`（プレイヤー `p` の「駒が属する連結成分」マスク）
- `19..26`（8枚）: `reach[p]`（プレイヤー `p` の合法到達セルマスク）
- `27..34`（8枚）: `next[p]`（次の行動“確率分布”の平面）
  - `p=0`（自分）は `reach[0]` 上で一様分布
  - `p>=1`（相手）は `pf_enabled` のとき PF（粒子フィルタ）で推定した AI パラメータ混合の分布、無効なら `reach[p]` 上で一様
- `35..42`（8枚）: `score[p]/500000`（定数平面）
  - `score[p] = Σ_{owner==p} value*level` を盤面から計算し、`500000` で割った値（`[0,1]` に clamp）
  - `p>=m`（存在しないプレイヤー）の平面は 0
- `43`: `turn_frac = turn/t_max`（定数平面）
- `44`: `m_norm = m/8`（定数平面）
- `45`: `u_norm = u_max/5`（定数平面）

### `next[p]`（相手分布）の直感
- 相手AIは「合法手集合 `B`」の中から、`(wa,wb,wc,wd)` で 4カテゴリ（未占領/自領土L<U/敵L=1/敵L>=2）を比較して貪欲に選ぶ確率と、`eps` で一様ランダムに選ぶ確率の混合になっている
- PF は各相手 `p` の `(wa..wd, eps)` の事後分布（粒子の重み）を持ち、観測された相手行動から更新している
- `next[p]` はその事後分布で混合した「相手が次に踏みそうなセル」の分布

> 実装の一次情報は `exps/exp001/cpp_core/include/ahc061/core/features.hpp` と `exps/exp001/cpp_core/include/ahc061/core/rules.hpp`。

## セットアップ
- 依存はリポジトリ直下の `pyproject.toml`（torch等）を利用する想定。

## 使い方

### 1) tools再現テスト（まずこれを通す）
```bash
.venv/bin/python -m exps.exp001.ahc061_exp001.scripts.test_tools_replay --seed-begin 0 --seed-end 2
```

### 2) 学習（wandb: デフォルト online）
```bash
export WANDB_API_KEY=...  # 事前に設定（または `wandb login` 済みなら不要）
.venv/bin/python -m exps.exp001.ahc061_exp001.scripts.train_ppo --updates 5 --batch-size 64
```

#### wandb を使わない / ネット無しで回す
```bash
# ローカルに記録（API key不要）
.venv/bin/python -m exps.exp001.ahc061_exp001.scripts.train_ppo --updates 5 --batch-size 64 --wandb-mode offline

# 完全に無効（最小確認用）
.venv/bin/python -m exps.exp001.ahc061_exp001.scripts.train_ppo --updates 5 --batch-size 64 --wandb-mode disabled
```

#### tools固定の greedy 評価を定期実行（初回は必ず評価）
- `--eval-seeds N`：`tools/in/0000..(N-1)` を評価に使う（0で無効）
- `--eval-batch B`：評価のバッチサイズ
- `--eval-every K`：K update ごとに評価（0なら初回評価のみ）

```bash
.venv/bin/python -m exps.exp001.ahc061_exp001.scripts.train_ppo --updates 100 --batch-size 256 --eval-seeds 50 --eval-batch 25 --eval-every 10
```

#### 高速化オプション
- CUDA のとき、update は bf16 AMP を **デフォルトで有効**（無効化: `--no-amp`）
- さらに速くしたい場合は `torch.compile`（有効化: `--compile`）
  - 初回のみコンパイル時間が乗る（長時間学習なら回収できる）

### 2-b) 学習（checkpoint も wandb に保存）
```bash
.venv/bin/python -m exps.exp001.ahc061_exp001.scripts.train_ppo --updates 100 --batch-size 256 --compile --wandb-log-checkpoints
```

### 2-c) 学習の完全再開（model + optimizer + RNG + wandb 同一run）
- `--resume-latest` / `--resume-ckpt` で再開できる（**再開時は strict**：学習ハイパラが一致しないとエラー）。
- `--updates` は「最終的に到達したい update 数」（例：`ckpt_0300` から `--updates 1000` なら `0301..1000` を実行）。
- 完全再開用の ckpt は `exps/exp001/artifacts/checkpoints_full/` に保存される。
- 以前の ckpt（`exps/exp001/artifacts/checkpoints/` 側の model-only）には optimizer/RNG が入っていないので完全再開できない。

```bash
# 最新 ckpt から再開（wandb は同一 run に auto-resume）
.venv/bin/python -m exps.exp001.ahc061_exp001.scripts.train_ppo --updates 1000 --batch-size 256 --resume-latest

# 特定 ckpt から再開
.venv/bin/python -m exps.exp001.ahc061_exp001.scripts.train_ppo --updates 1000 --batch-size 256 --resume-ckpt exps/exp001/artifacts/checkpoints_full/ckpt_0300.pt
```

### 2-d) 古い ckpt から重みだけ初期化（optimizer/RNGは新規）
- 旧形式（`exps/exp001/artifacts/checkpoints/`）の ckpt も `--init-ckpt` でロードできる。
- optimizer/RNG/wandb run は新規なので「完全再開」ではなく、**重み初期値として使う**用途。

```bash
.venv/bin/python -m exps.exp001.ahc061_exp001.scripts.train_ppo --updates 100 --batch-size 256 --init-ckpt exps/exp001/artifacts/checkpoints/ckpt_0300.pt
```

### 2-e) 蒸留（teacher ckpt → 軽量DW学生）
AtCoder の **ファイルサイズ制限（512 KiB）**を満たす軽量モデルを作りたい場合、teacher（学習済みResNet）から student（DWモデル）へ **オンライン蒸留**する。

- 蒸留は **保存せずオンライン生成**：teacher で環境を回して状態を作り、その場で student を学習する
- policy / value を **両方最後まで**蒸留する（policy: KL、value: MSE）
- teacher ckpt が `arch_name` を持たない場合は `resnet_v1` として扱う（既存ckpt互換）

例（`ckpt_0850.pt` を teacher、DW学生 `hidden=64, blocks=16` へ軽く蒸留）:
```bash
.venv/bin/python -m exps.exp001.ahc061_exp001.scripts.distill \
  --teacher-ckpt exps/exp001/artifacts/checkpoints/ckpt_0850.pt \
  --student-arch dwres_v1 --student-hidden 64 --student-blocks 16 \
  --updates 100 --batch-size 2048 --epochs 1 --minibatch 2048 --lr 1e-3 \
  --out-ckpt exps/exp001/artifacts/checkpoints_distill/ckpt_dw64b16_from_0850_u0100.pt
```

補足:
- `--no-pf` は teacher / student / 提出で **揃える**（特徴量 `next[p]` の意味が変わるため）。
- 蒸留後の ckpt は model-only で、`make_submit_torchscript` の `--ckpt` にそのまま渡せる。
- 蒸留後の ckpt を初期値にして PPO で微調整したい場合は `--init-ckpt` を使う（optimizer/RNGは新規）。

```bash
.venv/bin/python -m exps.exp001.ahc061_exp001.scripts.train_ppo \
  --arch dwres_v1 --hidden 64 --blocks 16 \
  --updates 100 --batch-size 256 \
  --init-ckpt exps/exp001/artifacts/checkpoints_distill/ckpt_dw64b16_from_0850_u0010.pt
```

### 3) toolsで評価（wandb online 必須）
```bash
export WANDB_API_KEY=...  # 事前に設定（または `wandb login` 済みなら不要）
.venv/bin/python -m exps.exp001.ahc061_exp001.scripts.eval_tools --ckpt exps/exp001/artifacts/checkpoints/ckpt_0005.pt --seed-begin 0 --seed-end 9 --batch 16
```

### 4) rollout速度ベンチ（wandb不要）
```bash
.venv/bin/python -m exps.exp001.ahc061_exp001.scripts.bench_rollout --mode rollout --device cuda --batch-size 256 --steps 200 --pin --compile
```

### 5) 学習全体ベンチ（baseline vs optimized / wandb不要）
```bash
.venv/bin/python -m exps.exp001.ahc061_exp001.scripts.bench_train --impl both --device cuda --batch-size 64 --epochs 4 --minibatch 2048 --reps 3
```

### 6) 提出（TorchScript埋め込み / libtorch CPU推論）
AtCoder 側で（CPUのみの）libtorch が利用できる前提で、学習済みモデルを TorchScript 化して `Main.cpp` に埋め込み、C++から推論して提出する。

#### 6-a) `Main.cpp` 生成（ckpt → TorchScript → 単一ファイル）
```bash
.venv/bin/python -m exps.exp001.ahc061_exp001.scripts.make_submit_torchscript \
  --ckpt exps/exp001/artifacts/checkpoints/ckpt_0400.pt \
  --out-dir exps/exp001/submit
```

TTA（test time augmentation）:
- 提出側の行動選択は、盤面の回転/反転（8通り）で推論して確率を結合する TTA に対応している。
- `exps/exp001/submit/solver_base.cpp` の `AHC061_EXP001_TTA_MODE` を編集して切り替える（生成後に `Main.cpp` を直接編集してもOK）。
  - `0`: TTAなし（greedy）
  - `1`: TTAあり（sum, デフォルト）
  - `2`: TTAあり（prod）

生成物:
- `exps/exp001/submit/Main.cpp`（これを提出する）
- `exps/exp001/submit/model_ts_base64.inc`（中間生成物）
- `exps/exp001/submit/solver_base.cpp`（生成元。直接提出はしない）

#### 6-b) ローカルで（AtCoder相当の）ビルドして動作確認
このリポジトリでは `.venv` の torch/libtorch を使ってリンクする。
```bash
TORCH_DIR=$(./.venv/bin/python -c 'import torch,os; print(os.path.dirname(torch.__file__))')
g++ exps/exp001/submit/Main.cpp -o exps/exp001/submit/solver \
  -O2 -std=gnu++23 -Wall -Wextra -pedantic \
  -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" \
  -L"$TORCH_DIR/lib" -Wl,-rpath,"$TORCH_DIR/lib" \
  -Wl,--no-as-needed -ltorch -ltorch_cpu -lc10 -Wl,--as-needed \
  -pthread -ldl
```

`tools` の tester で1ケース動作確認:
```bash
./tools/target/release/tester exps/exp001/submit/solver < tools/in/0000.txt > out
```

#### 注意
- 生成される `Main.cpp` はモデルサイズ依存。AtCoder のソースサイズ制限（512 KiB）に引っかかる場合は **モデル縮小**が必要。
  - 例：DW学生（`dwres_v1 hidden=64 blocks=16`）を作ってから提出する（上の蒸留参照）。
  - 生成後に `wc -c exps/exp001/submit/Main.cpp` で必ず確認する。
- 提出コードは `PF有効` の特徴量（`next[p]` にPF混合分布）を使う実装になっている。学習を `--no-pf` で行った場合は整合性に注意。
