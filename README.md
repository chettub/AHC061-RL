# AHC061-RL

[THIRD プログラミングコンテスト 2026 （AtCoder Heuristic Contest 061）](https://atcoder.jp/contests/ahc061) を強化学習で攻略するためのリポジトリ

## リンク

- 実験手順の詳細: `exps/exp002/README.md`
- 振り返り記事: [BLOG_ja.md](BLOG_ja.md)

## ビジュアライザ
![seed0_gif](assets/seed0.gif)

## 学習方法（`exps/exp002` の `train_ppo`）

`train_ppo` は次のコマンドで実行します。

```bash
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.train_ppo \
  --feature-id research_v4 \
  --arch dwres_ppconcat_v1 --hidden 112 --blocks 20 \
  --run-name rl --seed 42 \
  --updates 1000000 --warmup-updates 20 \
  --batch-size 512 --minibatch 512 --epochs 1 \
  --lr 3.0e-5 --ent-coef 5e-3 \
  --ema-decays 0.999,0.9999,0.99999 \
  --compile --compile-mode default --rollout-amp \
  --device auto --memory-format nchw --rollout-cache-device auto \
  --torch-num-threads 8 --torch-num-interop-threads 1 \
  --save-last-every 10 --save-every 50 \
  --eval-seeds 1000 --eval-every 1000 \
  --no-pf --shuffle-minibatches --no-fused-step-aux \
  --wandb-mode disabled
```

Multi-GPU の例（`torchrun`）:

```bash
.venv/bin/torchrun --standalone --nproc_per_node=8 \
  -m exps.exp002.ahc061_exp002.scripts.train_ppo \
  --feature-id research_v4 \
  --arch dwres_ppconcat_v1 --hidden 112 --blocks 20 \
  --run-name rl --seed 42 \
  --updates 1000000 --warmup-updates 20 \
  --batch-size 4096 --minibatch 4096 --epochs 1 \
  --lr 3.0e-5 --ent-coef 5e-3 \
  --ema-decays 0.999,0.9999,0.99999 \
  --compile --compile-mode default --rollout-amp \
  --device auto --memory-format nchw --rollout-cache-device auto \
  --torch-num-threads 8 --torch-num-interop-threads 1 \
  --save-last-every 10 --save-every 50 \
  --eval-seeds 1000 --eval-every 1000 \
  --no-pf --shuffle-minibatches --no-fused-step-aux \
  --wandb-mode disabled
```

## サブの作成方法（提出用 `Main.cpp` 生成）

標準版（TorchScript 埋め込み）:

```bash
.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.make_submit_torchscript \
  --ckpt exps/exp002/artifacts/runs/.../checkpoints/ckpt_last.pt \
  --out-dir exps/exp002/submit
```

コンパクト版（サイズ最適化）:

```bash
./.venv/bin/python -m exps.exp002.ahc061_exp002.scripts.make_submit_compact \
  --ckpt exps/exp002/artifacts/runs/.../checkpoints/ckpt_last.pt \
  --payload-encoding huff91 \
  --ppconcat-preset c7 \
  --out-dir exps/exp002/submit
```
