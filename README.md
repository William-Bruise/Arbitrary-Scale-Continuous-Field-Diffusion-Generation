# Arbitrary-Scale Continuous Field Diffusion Generation (MNIST Prototype)

这是一个 **最小但完整可运行** 的 research prototype，目标不是超分辨（SR），而是：

- 学习 MNIST 图像分布；
- diffusion 发生在 **continuous field coefficient space**，不是 pixel-space；
- 同一个生成样本可以在任意分辨率渲染（例如 28x28 / 42x42 / 56x56 / 84x84），且不是先固定分辨率生成后再 resize。

---

## 1. 任务定义与边界

### 1.1 这是什么

本项目将一张图像表示为连续函数：

\[
f(x, y)=\sum_{k=1}^{K} a_k\,\phi_k(x,y),\quad (x,y)\in[0,1]^2
\]

其中：
- \(\phi_k\) 是固定高斯基函数（由中心 \(\mu_k\) 和宽度 \(\sigma\) 定义）；
- \(a_k\) 是该样本的连续场系数（coefficient）。

diffusion 模型学习的是系数向量 \(a\in\mathbb{R}^K\) 的分布。

### 1.2 这不是什么

- 不是 LR->HR 的超分辨；
- 不是图像修复/去噪；
- 不是 pixel-space DDPM 后再插值到更大分辨率。

同一样本在不同分辨率的输出来自 `field.render(H, W)` 对同一 continuous field 的重新采样。

---

## 2. 代码结构

- `src/dataset.py`
  - `make_mnist_dataset(...)`: 自动下载/缓存 MNIST
  - `pixel_grid_to_continuous(h,w)`: 像素网格到连续坐标域映射

- `src/continuous_field.py`
  - `ContinuousGaussianField`
    - `basis(coords)`
    - `query(coeffs, coords)`
    - `render(coeffs, H, W)`
  - `fit_coeffs_to_image(...)`: 通过闭式 ridge 解把图像拟合为系数

- `src/diffusion.py`
  - `DDPMCoefficients`
    - `q_sample`（前向加噪）
    - `p_sample`（单步反向）
    - `sample`（完整反向采样）

- `src/model.py`
  - `TimeEmbedding`
  - `CoeffDenoiser`（可配置 `hidden` 和 `depth`）

- `src/train.py`
  - 训练主循环
  - 系数 z-score 归一化（可选）
  - 周期保存多分辨率 sample / checkpoint / log
  - **当前按你的要求：损失仅 `loss = ddpm`（无 smooth）**

- `src/sample.py`
  - 从随机噪声采样系数
  - 自动读取 ckpt 中超参（`num_basis/sigma/hidden/depth`）
  - 输出多分辨率渲染拼图

- `src/smoke_test.py`
  - 快速检查渲染与扩散前向是否通

---

## 3. 方法细节

### 3.1 Continuous Gaussian field 参数化

`num_basis=K` 必须是完全平方数，代码中将其排布为 `sqrt(K) x sqrt(K)` 的中心网格。每个中心对应一个 Gaussian basis。

最终图像值由所有 basis 的线性组合给出，`render(H,W)` 在任意输出网格上查询坐标并求和得到像素值。

### 3.2 系数拟合（image -> coeffs）

训练时，每个 MNIST 样本先通过 `fit_coeffs_to_image` 得到 `coeffs`：

1. 构建坐标网格和基函数矩阵 \(\Phi\)；
2. 使用 ridge 形式的闭式解求系数；
3. 将图像值经 `logit` 映射后拟合到线性系数空间。

这一步把离散像素图像转为连续场 latent，供后续 diffusion 训练。

### 3.3 DDPM 在哪里进行

DDPM 不在像素空间，而在 `coeffs` 空间：

- 训练：随机采样 `t`，对 `coeffs` 加噪得到 `x_t`，网络预测噪声 `eps`；
- 损失：`MSE(pred_eps, eps)`；
- 采样：从 `N(0,I)` 系数噪声反推到 `coeffs`，再用 field 渲染任意分辨率。

### 3.4 可选系数标准化

`--normalize-coeffs` 时会先估计训练集系数均值/方差，再在标准化空间做 diffusion。采样后自动反标准化再渲染。

---

## 4. 安装与环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib pillow
```

---

## 5. 运行说明

### 5.1 Smoke test

```bash
python -m src.smoke_test
```

预期：打印不同分辨率渲染 shape 与扩散前向 shape。

### 5.2 训练（推荐）

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

### 5.3 采样

```bash
python -m src.sample \
  --ckpt runs/full_train/checkpoints/final.pt \
  --outdir runs/full_train/sample_eval
```

---

## 6. 关键参数解释（训练）

- `--epochs`：完整遍历数据集次数
- `--batch-size`：每步样本数
- `--timesteps`：扩散总步数 T
- `--num-basis`：连续场 basis 数量 K（完全平方数）
- `--sigma`：Gaussian basis 宽度（越小越锐利，过小可能破碎）
- `--hidden`：去噪 MLP 隐层维度
- `--depth`：去噪 MLP 隐层层数
- `--normalize-coeffs`：是否启用系数 z-score
- `--stats-batches`：估计均值方差用多少个 batch
- `--sample-every`：每 N 步保存一次多分辨率 sample
- `--ckpt-every`：每 N 步保存一次 checkpoint

---

## 7. 产物说明

训练输出目录（`--outdir`）下：

- `logs/train_log.txt`：训练日志（含 `loss/ddpm` 与 shape）
- `checkpoints/step_*.pt`、`checkpoints/final.pt`
- `samples/step_*_multires.png`、`samples/final_multires.png`

sample 脚本输出：

- `sample_eval/sample_multires.png`

---

## 8. 为什么它是 arbitrary-scale generation

- 生成变量是 continuous field coefficients；
- 渲染接口是 `render(coeffs, H, W)`，可直接在任意 `(H,W)` 查询；
- 多尺度图像来自同一个 sample 的连续查询，而非后处理 resize。

---

## 9. 常见问题（FAQ）

### Q1: loss 很低但图像不像数字？
常见原因：
- `num_basis` 太小（容量不足）
- `sigma` 太大（过平滑）
- denoiser 容量偏小

优先尝试：
1) `num_basis: 64 -> 144/256`
2) `sigma: 0.12 -> 0.08/0.06`
3) `hidden/depth` 增大

### Q2: 为什么不用 smooth 正则？
按当前设计要求，已经去掉 smooth，训练目标是标准 DDPM MSE。

### Q3: 如何验证跨尺度一致性？
看同一张 `*_multires.png` 中不同分辨率的结构是否一致（只细节采样密度变化，不应语义跳变）。

---

## 10. 最小复现实验建议

先跑：
1. `python -m src.smoke_test`
2. `python -m src.train ... --epochs 5`
3. `python -m src.sample --ckpt ...`

确认流程跑通后再拉到 100 epochs 做质量观察。
