# Arbitrary-Scale Continuous Field Diffusion Generation

这是一个**详细版**说明文档，目标是让你后续自己看代码、复现实验、扩展数据集时不需要反复回忆上下文。

---

## 1. 项目目标（明确边界）

本项目做的是：

- 在**连续场 latent**（Gaussian basis coefficients）空间做 diffusion；
- 用同一个生成样本在任意 `H x W` 网格渲染；
- 支持灰度和 RGB 数据集；
- 训练/采样/eval 一体化。

本项目不做的是：

- 超分辨（LR->HR）；
- 先固定分辨率生成后再简单 resize 伪装 arbitrary-scale。

---

## 2. 数学形式与核心思想

每张图表示为连续函数：

\[
f_c(x,y)=\sum_{k=1}^{K} a_{c,k}\,\phi_k(x,y),\quad c\in\{1,...,C\}
\]

- `C`：通道数（灰度 1，RGB 3）
- `K`：Gaussian basis 数（必须是完全平方数）
- `phi_k`：固定中心与宽度 `sigma` 的高斯基函数
- `a_{c,k}`：每张图的连续场系数

扩散在 `a` 上进行，而不是在像素网格上进行。

---

## 3. 代码结构（按功能解释）

### 3.1 `src/continuous_field.py`

- `ContinuousGaussianField(num_basis, sigma, channels)`
  - `basis(coords)`：坐标到 basis 响应
  - `query(coeffs, coords)`：给定系数在任意坐标查询
  - `render(coeffs, H, W)`：任意分辨率渲染
- `fit_coeffs_to_image(...)`
  - 使用 ridge 闭式解，把离散图像拟合成 `[B,C,K]` 系数

### 3.2 `src/model.py`

- `LatentUNetDenoiser`
  - 输入：flatten 后的 `[B, C*K]` noisy coeff
  - 内部 reshape 为 `[B,C,S,S]`（`S=sqrt(K)`）
  - time-conditioned UNet 去噪
  - 输出回 `[B, C*K]`
- `TimeEmbedding`：DDPM 时间步嵌入

### 3.3 `src/diffusion.py`

- `DDPMCoefficients`
  - `q_sample` 前向加噪
  - `p_sample` 单步反推
  - `sample` 全流程采样

### 3.4 `src/dataset.py`

统一入口：`make_dataset(name, root, train, image_size)`，自动下载并缓存。

支持：
- `mnist`
- `fashionmnist`
- `kmnist`
- `cifar10`（RGB）
- `celeba`（RGB）
- `stl10`（RGB）

### 3.5 `src/train.py`

- 数据自动下载 -> 拟合 coeff -> flatten coeff -> DDPM loss 训练
- 可选 `--normalize-coeffs`
- 周期保存 checkpoint 和多分辨率样本图

### 3.6 `src/sample.py`

- 从随机噪声采样系数
- 自动读取 ckpt 超参（`channels/num_basis/sigma/unet_base/...`）
- 渲染多尺度图

### 3.7 `src/eval.py`

RGB-aware 评估：
- 用对应数据集训练一个分类器（自动下载数据）
- 统计生成样本 `mean_confidence`、`label_entropy`、`label_hist`
- 比较训练尺度与 OOD 尺度
- 同时保存每个尺度的可视化网格

### 3.8 `src/smoke_test.py`

快速连通性检查。

---

## 4. 关键张量 shape

- `image`: `[B, C, H, W]`
- `coeffs`: `[B, C, K]`
- `flat_coeffs`: `[B, C*K]`
- `x_t`: `[B, C*K]`
- `model(x_t,t)` 输出：`[B, C*K]`
- `render(coeffs,H,W)` 输出：`[B, C, H, W]`

---

## 5. 参数解释（训练）

命令模板：

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
  --sample-every 500 \
  --ckpt-every 1000 \
  --outdir runs/cifar10
```

参数含义：

- `--dataset`：数据集名
- `--data-root`：自动下载目录
- `--image-size`：输入 resize 尺寸（例如 celeba/stl10）
- `--epochs`：训练轮数
- `--batch-size`：batch
- `--timesteps`：DDPM 步数
- `--num-basis`：连续场 basis 数 K（完全平方数）
- `--sigma`：Gaussian basis 宽度（越小越锐利，过小易碎）
- `--unet-base`：UNet 基础通道数
- `--normalize-coeffs`：是否做 coeff z-score
- `--sample-every`：保存 sample 周期
- `--ckpt-every`：保存 ckpt 周期
- `--outdir`：输出目录

---

## 6. 数据集建议超参（起点）

- MNIST/FashionMNIST/KMNIST（28x28）：
  - `num-basis=144`, `sigma=0.08`, `unet-base=64`
- CIFAR10（32x32 RGB）：
  - `num-basis=256`, `sigma=0.06`, `unet-base=64~96`
- CelebA/STL10（64x64 RGB）：
  - `num-basis=256~400`, `sigma=0.05~0.06`, `unet-base=96~128`

---

## 7. 训练脚本（每个数据集一份）

- `scripts/train_mnist.sh`
- `scripts/train_fashionmnist.sh`
- `scripts/train_kmnist.sh`
- `scripts/train_cifar10.sh`
- `scripts/train_celeba.sh`
- `scripts/train_stl10.sh`

示例：

```bash
bash scripts/train_cifar10.sh
bash scripts/train_celeba.sh
```

---

## 8. 评估（含 OOD 尺度）

```bash
MPLBACKEND=Agg python -m src.eval \
  --ckpt runs/cifar10/checkpoints/final.pt \
  --dataset cifar10 \
  --data-root ./data \
  --num-samples 1024 \
  --train-size 32 \
  --ood-sizes 40,48,56,64,96 \
  --clf-epochs 2 \
  --outdir runs/cifar10/eval
```

输出：
- `metrics.txt`
- `samples_size_*.png`

说明：
- `MPLBACKEND=Agg` 可避免部分环境里 X11/图形后端警告。

---

## 9. 常见问题

### Q1. loss 很低但视觉还不好
常见原因：
- `num-basis` 太小
- `sigma` 太大
- `unet-base` 太小

优先调参顺序：
1) 提高 `num-basis`
2) 适度减小 `sigma`
3) 增大 `unet-base`

### Q2. 为什么这是 arbitrary-scale generation
因为生成对象是 continuous field coefficients，同一组 coeff 可直接 `render(H,W)` 到任意分辨率，不依赖后处理 resize。

### Q3. 数据集下载是否自动
是。当前数据集构造均在代码中 `download=True`。

---

## 10. 最小复现流程

1. 跑 smoke test
```bash
python -m src.smoke_test
```

2. 训练（先少量 epoch 验证链路）
```bash
python -m src.train --dataset mnist --epochs 5 --outdir runs/debug_mnist
```

3. 采样
```bash
python -m src.sample --ckpt runs/debug_mnist/checkpoints/final.pt --outdir runs/debug_mnist/sample
```

4. 评估
```bash
MPLBACKEND=Agg python -m src.eval --ckpt runs/debug_mnist/checkpoints/final.pt --dataset mnist --outdir runs/debug_mnist/eval
```
