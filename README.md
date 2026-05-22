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
  --normalize-coeffs \
  --sample-every 500 \
  --ckpt-every 1000 \
  --outdir runs/full_train
```

## 训练损失

当前版本按你的要求**不使用 smooth 正则**，训练损失仅为 DDPM 噪声预测 MSE：

`loss = ddpm`
