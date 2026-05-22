# Continuous-Field Diffusion Generation (Arbitrary Scale)

已支持自动下载并训练多个常用扩散对比数据集，且**保留彩色图像**（不再自动转灰度）。

## 支持数据集（自动下载）
- mnist
- fashionmnist
- kmnist
- cifar10 (RGB)
- celeba (RGB)
- stl10 (RGB)

> 说明：以上都在代码中使用 `download=True`，首次运行训练命令会自动下载到 `--data-root`。

## 训练脚本（每个数据集一份）
- `scripts/train_mnist.sh`
- `scripts/train_fashionmnist.sh`
- `scripts/train_kmnist.sh`
- `scripts/train_cifar10.sh`
- `scripts/train_celeba.sh`
- `scripts/train_stl10.sh`

运行示例：
```bash
bash scripts/train_cifar10.sh
bash scripts/train_celeba.sh
```

## 通用训练命令
```bash
python -m src.train \
  --dataset cifar10 \
  --data-root ./data \
  --image-size 32 \
  --epochs 100 \
  --batch-size 128 \
  --timesteps 200 \
  --num-basis 256 \
  --sigma 0.06 \
  --unet-base 64 \
  --normalize-coeffs \
  --outdir runs/cifar10
```

## 关键变更
- 连续场与渲染现在支持多通道（RGB）系数：`coeffs` 形状从 `[B,K]` 扩展为 `[B,C,K]`。
- 扩散目标改为 flatten 后的 `[B, C*K]`，主干 UNet 输入输出按通道数配置。
- sample/train 已兼容彩色与灰度两种场景。
