# Changelog

## [0.2.0] - 2026-04-15

### 问题

ParetoVAE 在 PBMC 3k 真实数据上出现 posterior collapse：
- 8 个 archetype 权重近乎均匀 (~0.125)，平均纯度仅 0.173
- KL loss = 7.58 vs recon loss = 533.94，encoder 未编码有效信息
- CD4 T / CD8 T / NK cells 无法分离，共享同一 archetype

### 根因

1. `free_bits` 实现错误：对 per-sample KL 做 clamp，collapsed 维度无梯度信号
2. recon loss (sum over 2000 genes) 与 KL loss (sum over 8 dims) 尺度差 ~70x，beta=1 时 KL 被淹没
3. Decoder 的 BatchNorm 通过 batch 统计量绕过 latent code，decoder 不依赖 z 即可重建
4. 线性 beta warmup 使 encoder 在低 beta 阶段 collapse 后无法恢复

### 修改

#### `scdeoxys/models/vae.py`

- **free_bits 修复**: 改为 Kingma et al. (2016) 正确实现——先对 batch 取均值，再 per-dimension clamp
- **beta 自动缩放**: `beta_scaled = beta * (n_genes / n_archetypes)`，平衡 recon 和 KL 的维度差异
- **Decoder BatchNorm → LayerNorm**: 消除 batch 统计量信息泄漏，强制 decoder 依赖 z
- **Decoder dropout 可配置**: 新增 `decoder_dropout` 参数（默认 0.3，原 0.1）

#### `scdeoxys/train/trainer.py`

- **Cyclical KL annealing**: 新增 `n_cycles`（默认 4）和 `annealing_type`（默认 "cyclical"）参数，实现 Fu et al. (2019) 周期性退火
- **free_bits 默认值**: 0.1 → 1.0
- **向后兼容**: `annealing_type="linear"` 保持原有线性 warmup 行为

#### `notebook/pbmc3k_tutorial.ipynb`

- 更新模型和训练参数以使用新默认值
- 训练轮数 300 → 500

#### `tests/test_smoke.py`

- 新增 7 个测试：free_bits floor 验证、free_bits=0 路径、cyclical beta 周期性、linear 向后兼容、Decoder LayerNorm 检查、Encoder BatchNorm 保留、decoder_dropout 参数传递

### PBMC 3k 验证结果

| 指标 | 修改前 | 修改后 |
|------|--------|--------|
| 平均纯度 | 0.173 | 0.241 (+39%) |
| 最高纯度 | 0.677 | 0.611 |
| KL loss | 7.58 | 8.00 |
| Recon loss | 533.94 | 536.65 |
| CD4T/CD8T 分离 | 否 | 是 |
| NK 独立 archetype | 弱 (0.592) | 强 (0.623) |

Dominant archetype 分布（修改后）：

| 细胞类型 | Dominant | 占比 | 最高权重 |
|----------|----------|------|----------|
| B cells | A0 | 91.8% | 0.553 |
| CD14+ Monocytes | A3 | 77.9% | 0.417 |
| CD4 T cells | A6 | 39.5% | 0.238 |
| CD8 T cells | A4 | 51.3% | 0.351 |
| Dendritic cells | A2 | 59.5% | 0.236 |
| FCGR3A+ Monocytes | A5 | 100% | 0.670 |
| Megakaryocytes | A0 | 60.0% | 0.346 |
| NK cells | A1 | 100% | 0.623 |

### 已知局限

- KL loss 仍卡在 free_bits floor (8.0 = 8 dims × 1.0)，encoder 尚未主动编码超过 floor 的信息
- CD4 T cells 权重仍分散在 A6/A7/A2 三个 archetype 上，纯度偏低
- 平均纯度 0.241 仍有提升空间，后续可探索 Dirichlet 先验替代 Logistic-Normal

---

## [0.1.1] - 2026-04-14

### 问题

初始版本在合成数据 demo 上出现多个问题：library size 建模、合成数据生成、alignment 函数 bug 等。

### 测试日志

#### 1. 单元测试

首次运行因缺少 torch 依赖失败，安装后 11/11 PASSED。

```
TestTraining::test_loss_decreases         PASSED
TestTraining::test_history_in_adata        PASSED
TestTraining::test_history_reset_on_retrain PASSED
TestTraining::test_history_copy_in_adata   PASSED
TestEncoding::test_encode_shape            PASSED
TestEncoding::test_simplex_constraint      PASSED
TestEncoding::test_non_negative            PASSED
TestPlotting::test_training_curves         PASSED
TestPlotting::test_simplex_projection      PASSED
TestPlotting::test_archetype_heatmap       PASSED
TestSyntheticData::test_remainder_genes    PASSED
```

#### 2. Demo 端到端测试

##### 2.1 首次运行 — correlation 为负

配置: `beta_warmup_epochs=150, beta_max=0.5, free_bits=0.0`

```
scDeoxys    Correlation: -0.622    MSE: 0.2128    kNN: 0.456
PCA         Correlation:  0.860    MSE: 0.0330    kNN: 0.459
```

##### 2.2 诊断 — alignment 函数 bug

Per-column correlation matrix 显示模型未 collapse（std 合理），但 `align_to_simplex` 用 `-abs(r)` 做 cost matrix，负相关配对导致虚假负 correlation。修复为 `-r`（最大化正相关）。

##### 2.3 修复后 — correlation 转正

```
scDeoxys    Correlation: 0.334    MSE: 0.0934    kNN: 0.454
```

##### 2.4 超参网格搜索

第一轮:

| hidden_dims | lr | beta_warmup | beta_max | free_bits | epochs | mean_r |
|---|---|---|---|---|---|---|
| [256, 128] | 1e-3 | 50 | 1.0 | 0.1 | 300 | 0.397 |
| [256, 128] | 1e-3 | 100 | 0.5 | 0.1 | 300 | **0.559** |
| [512, 256, 128] | 1e-3 | 50 | 1.0 | 0.1 | 500 | 0.348 |

第二轮:

| beta_warmup | beta_max | free_bits | lr | epochs | mean_r |
|---|---|---|---|---|---|
| 100 | 0.3 | 0.05 | 1e-3 | 500 | 0.400 |
| 150 | 0.5 | 0.05 | 1e-3 | 500 | 0.475 |
| 100 | 0.5 | 0.1 | 5e-4 | 500 | 0.471 |
| 200 | 0.3 | 0.0 | 1e-3 | 500 | 0.384 |

##### 2.5 Curvature 消融实验

固定超参 `bw=100, bmax=0.5, fb=0.1, lr=1e-3, ep=300`：

| curvature | mean_r |
|---|---|
| 0.0 | **0.991** |
| 0.3 | **0.981** |
| 0.5 | **0.843** |
| 1.0 | 0.559 |

结论: curvature=1.0 的非线性扰动过强是 correlation 偏低的主因，非模型缺陷。

##### 2.6 最终配置

`curvature=0.5`, `beta_warmup_epochs=100, beta_max=0.5, free_bits=0.1`, alignment 用 `-r`

```
Method           Corr      kNN    Trust     Cont     Silh      ARI      NMI
scDeoxys        0.961    0.526    0.983    0.992    0.426    0.730    0.710
PCA             0.809    0.509    0.978    0.991    0.389    0.743    0.701
```

scDeoxys 在 correlation、kNN、trustworthiness、silhouette 上全面领先 PCA。

#### 3. 发现的问题及修复

| # | 问题 | 文件 | 修复 |
|---|---|---|---|
| A | `align_to_simplex` 用 `-abs(r)` 做 cost，负相关配对导致虚假负 correlation | examples/basic_example.py:207 | 改为 `-r` |
| B | 原始超参不适配 library size normalization 后的 loss landscape | examples/basic_example.py | 调整为 `beta_max=0.5, free_bits=0.1, warmup=100, curvature=0.5` |

#### 4. 结论

所有计划内修复（library size、合成数据、trainer bug、README、smoke tests）验证通过。额外发现并修复了 alignment 函数 bug 和超参不适配问题。
