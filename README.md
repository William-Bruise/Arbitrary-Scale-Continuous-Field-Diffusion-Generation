# Arbitrary-Scale Continuous Field Diffusion Generation (MNIST Prototype)

这是一个**最小可运行 research prototype**，目标是：
- 学习 MNIST 图像分布；
- diffusion 生成的是 **continuous field latent**（不是固定像素图）；
- 同一个 sample 可在任意分辨率下渲染（28/42/56/84 等）。

## 核心思想

1. 用一个高斯基函数集合表示连续场：
   \[
   f(x,y)=\sum_{k=1}^{K} a_k \exp\left(-\frac{1}{2}\frac{\|[x,y]-\mu_k\|^2}{\sigma_k^2}\right)
   \]
2. diffusion 在系数向量 \(a\in\mathbb{R}^K\) 上进行（可看成 continuous field coefficient space）。
3. 渲染时在任意输出网格上 query 坐标并求值得到图像。

> 这不是超分辨：没有 LR 输入，也没有 fixed-scale 图先生成再 resize。

## 目录结构

- `src/dataset.py`：MNIST 自动下载与坐标域工具
- `src/continuous_field.py`：连续 Gaussian field 表示与渲染
- `src/diffusion.py`：DDPM（作用于 field coefficients）
- `src/model.py`：time-conditioned MLP 去噪器
- `src/train.py`：训练入口（保存 ckpt、日志、多分辨率样本）
- `src/sample.py`：采样入口（随机噪声 -> coefficients -> 多分辨率渲染）
- `src/smoke_test.py`：快速连通性测试

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib pillow
```

## 运行 smoke test

```bash
python -m src.smoke_test
```

## 训练（最小）

```bash
python -m src.train --steps 1 --batch-size 8 --outdir runs/minimal
```

训练会：
- 自动下载 MNIST 到 `./data`
- 打印关键 tensor shape
- 保存：
  - `runs/minimal/checkpoints/step_*.pt`
  - `runs/minimal/samples/step_*_multires.png`
  - `runs/minimal/logs/train_log.txt`

## 采样

```bash
python -m src.sample --ckpt runs/minimal/checkpoints/step_1.pt --outdir runs/minimal/sample_eval
```

采样会生成同一个样本在多分辨率下的拼图。

## GaussianSR 借鉴边界

理论上借鉴的是：
- continuous Gaussian field 参数化思想
- arbitrary-resolution query/render 接口
- query 再聚合的渲染数据流

本仓库**没有**复现 GaussianSR 的超分辨任务设置。
