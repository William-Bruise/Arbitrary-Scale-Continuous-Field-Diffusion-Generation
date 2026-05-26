# Continuous Random Field Diffusion (Implementation Notes)

设随机图像函数为 \(f:[0,1]^2\to\mathbb{R}^3\)。对任意有限坐标集 \(S=\{u_i\}_{i=1}^n\)，定义有限维边缘 \(p(f(S))\)。模型目标是学习一致族 \(\{p(f(S))\}_S\)，并满足投影一致性：若 \(S_c\subset S_f\)，则 \(p(f(S_c))\) 是 \(p(f(S_f))\) 的边缘。

我们实现了坐标条件扩散：在每次训练中随机采样坐标集 \(S\)（规则/不规则），取真值 \(x_0=f(S)\)，执行DDPM前向
\[
q(x_t|x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)
\]
并训练 \(\epsilon_\theta(x_t,S,t,z_g)\) 预测噪声，\(z_g\) 为由整图编码的全局结构变量。主损失：
\[
\mathcal{L}_{main}=\mathbb{E}\|\epsilon-\epsilon_\theta(x_t,S,t,z_g)\|_2^2
\]
一致性损失（子集约束）：对 \(S_c\subset S_f\) 再做一次噪声预测，
\[
\mathcal{L}_{cons}=\mathbb{E}\|\epsilon_c-\epsilon_\theta(x_t^c,S_c,t,z_g)\|_2^2
\]
总损失 \(\mathcal{L}=\mathcal{L}_{main}+\lambda\mathcal{L}_{cons}\)。

这不是“固定分辨率latent+decoder”：训练/采样原生作用于任意坐标集，不依赖固定输出网格，网格仅是查询坐标的一种特例。

## Inverse problems Bayesian view
观测模型：\(y=A(f)+\eta,\;\eta\sim\mathcal{N}(0,\sigma^2I)\)。后验：
\[
p(f|y)\propto p(y|f)p(f)
\]
我们用扩散先验近似 \(p(f)\)，采用posterior guidance：在反向采样中加入似然梯度 \(\nabla_f \log p(y|f)\propto -\frac{1}{\sigma^2}\nabla_f\|A(f)-y\|^2\)。实现中统一算子接口 \(A\) 支持 inpainting / super-resolution / denoising / sparse observation。

局限：当前公开源自动下载受许可证约束，脚本提供镜像尝试+明确手动fallback；FID仅预留接口，重点在一致性与可扩展研究骨架。
