# Arbitrary-Scale Continuous Field Diffusion Generation (MNIST Prototype)

最小原型：扩散在 continuous field coefficients 上进行，并支持同一样本任意分辨率渲染。

## 推荐训练命令（小补丁版）

```bash
python -m src.train \
  --epochs 100 \
  --batch-size 128 \
  --timesteps 200 \
  --num-basis 144 \
  --sigma 0.08 \
  --hidden 512 \
  --depth 4 \
  --smooth-weight 1e-5 \
  --normalize-coeffs \
  --sample-every 500 \
  --ckpt-every 1000 \
  --outdir runs/full_train
```

## `smooth` 是什么？

训练日志里输出的 `smooth` 是 **2D coefficient grid 的 TV-like 平滑项**（未乘权重前的原始值）：

- 先把 `coeffs` 从 `[B, K]` reshape 为 `[B, S, S]`，其中 `S=sqrt(K)`
- 计算纵向和横向相邻差分的平方均值
- `smooth = mean((c[i+1,j]-c[i,j])^2) + mean((c[i,j+1]-c[i,j])^2)`

最终总损失是：

`loss = ddpm + smooth_weight * smooth`

因此你看到 `smooth` 数值很大是正常的，关键看 `smooth_weight * smooth` 的量级（例如 `1e-5 * 3000 = 0.03`）。
