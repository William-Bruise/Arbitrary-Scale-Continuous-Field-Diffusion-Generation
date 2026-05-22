# Arbitrary-Scale Continuous Field Diffusion Generation (MNIST Prototype)

这是一个可运行的 research prototype，目标是：
- 学习 MNIST 图像分布；
- diffusion 生成的是 **continuous field latent**（不是固定像素图）；
- 同一个 sample 可在任意分辨率下渲染（28/42/56/84 等）。

## 核心思想

1. 用一个高斯基函数集合表示连续场：
   \[
   f(x,y)=\sum_{k=1}^{K} a_k \exp\left(-\frac{1}{2}\frac{\|[x,y]-\mu_k\|^2}{\sigma_k^2}\right)
   \]
2. diffusion 在系数向量 \(a\in\mathbb{R}^K\) 上进行（continuous field coefficient space）。
3. 渲染时在任意输出网格上 query 坐标并求值得到图像。

> 这不是超分辨：没有 LR 输入，也没有 fixed-scale 图先生成再 resize。

## 完整训练（推荐）

```bash
python -m src.train \
  --epochs 100 \
  --batch-size 128 \
  --timesteps 200 \
  --num-basis 64 \
  --smooth-weight 1e-5 \
  --normalize-coeffs \
  --sample-every 500 \
  --ckpt-every 1000 \
  --outdir runs/full_train
```

关键新增参数：
- `--smooth-weight`: 2D TV 系数平滑正则权重（默认 `1e-5`）
- `--normalize-coeffs`: 训练前估计系数均值方差并做 z-score
- `--stats-batches`: 估计系数统计量的 batch 数（默认 `100`）

## 采样

```bash
python -m src.sample --ckpt runs/full_train/checkpoints/final.pt --outdir runs/full_train/sample_eval
```

如果 checkpoint 中包含系数标准化统计量，采样时会自动反标准化后再渲染。
