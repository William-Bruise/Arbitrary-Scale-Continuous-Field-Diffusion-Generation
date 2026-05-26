# Arbitrary-Scale Continuous Field Diffusion Generation

研究型最小可运行仓库：训练一个定义在连续坐标域 `[0,1]^2` 上的无条件图像随机函数生成模型，并用于逆问题后验推断。

## 目录
- `configs/` 训练配置
- `datasets/` 数据集加载
- `models/` 连续坐标条件扩散模型
- `trainers/` 训练循环（随机坐标子集 + 子集一致性）
- `samplers/` 任意分辨率/长宽比采样
- `inverse_problems/` 统一观测算子与后验引导
- `metrics/` 一致性指标
- `scripts/` 数据准备、训练、采样、评估与demo
- `docs/method.md` 数学建模与贝叶斯推导

## 快速开始
```bash
python scripts/prepare_data.py --datasets ffhq,celebahq --data-root ./data
python scripts/train.py --config configs/ffhq_train.yaml
python scripts/sample.py --ckpt runs/ffhq_continuous/checkpoints/final.pt --resolutions 64x64,128x128,192x96
python scripts/eval_consistency.py
python scripts/inverse_demo.py --input your.png --task inpainting --out out_inpaint.png
```

## 数据说明
`prepare_data.py` 会先尝试公开镜像下载示例并建立 `train/val` 结构；若由于许可证/权限/网络限制失败，会抛出清晰错误并提示手动放置路径，不会静默失败。

## 核心实现要点
1. 每步训练随机采样坐标集（规则或不规则），而不是固定全网格。
2. 模型结构含全局分支（整图结构）与坐标条件分支（局部生成）。
3. 通过 `S_c ⊂ S_f` 的子集损失显式约束跨尺度/跨子集一致性。
4. 采样可直接在任意坐标集上运行，支持任意分辨率与长宽比。

更多数学细节见 `docs/method.md`。
