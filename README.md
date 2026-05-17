# 联邦学习拜占庭防御项目

## 项目概述

本项目实现了一个分布式联邦学习框架，集成了**拜占庭防御机制**和**差分隐私保护**。该框架在面对恶意客户端和隐私泄露威胁时，能够保证模型训练的鲁棒性和隐私安全。

### 核心特性

- 🛡️ **拜占庭防御**：支持多种聚合策略（剪裁、中值、Krum等）抵御恶意客户端攻击
- 🔒 **差分隐私**：集成自适应差分隐私机制保护客户端数据隐私
- 📊 **自适应控制**：采用自适应稳定性控制器动态调整训练参数
- 🚀 **分布式学习**：支持多客户端并行训练
- 📈 **完整监控**：包含训练指标记录和可视化工具

---

## 项目结构

```
dara1/
├── dara.py                 # 主要联邦学习框架（带DP和自适应稳定性控制）
├── mcpr.py                 # 拜占庭防御基准实验
├── models/
│   ├── resnet_gn.py        # ResNet18模型（Group Normalization）
│   └── cnn/                # CNN模型定义
├── utils/
│   ├── adapp.py            # 自适应稳定性控制器
│   ├── adaptive_dp.py       # 自适应差分隐私机制
│   └── training_monitor.py  # 训练监控和可视化
└── README.md               # 本文件
```

---

## 文件说明

### 主程序

#### `dara.py`
- **功能**：核心联邦学习框架
- **主要配置**：
  - 客户端数量：20个
  - 拜占庭客户端：4个
  - 训练轮次：200轮
  - 本地训练周期：3个
  - 批大小：128
  - 学习率：0.02
- **关键组件**：
  - `GaussianPrivacyAccountant`：高斯隐私账户管理
  - 自适应稳定性控制
  - 差分隐私梯度裁剪
  - 拜占庭鲁棒聚合

#### `mcpr.py`
- **功能**：多种拜占庭防御方法对比实验
- **支持方法**：
  - FedAvg：标准联邦平均
  - Trimmed：剪裁聚合
  - Median：中值聚合
  - Krum：Krum聚合
  - MCPR：协调多数投票
- **配置**：50个客户端，20%恶意比例

### 模型

#### `models/resnet_gn.py`
- ResNet18模型架构
- 使用Group Normalization替代BatchNormalization
- 适配CIFAR数据集

#### `models/cnn/`
- CNN模型实现

### 工具函数

#### `utils/adapp.py`
- `AdaptiveStabilityController`：自适应稳定性控制
- 动态调整学习率和梯度

#### `utils/adaptive_dp.py`
- `apply_dp_dual_adaptive_secure`：应用自适应差分隐私
- `AdaptiveRDPAccountant`：自适应RDP隐私账户
- 支持双重自适应机制

#### `utils/training_monitor.py`
- `log_training_metrics`：记录训练指标
- `visualize_training_metrics`：可视化训练曲线

---

## 主要参数说明

### dara.py 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `NUM_CLIENTS` | 20 | 总客户端数量 |
| `BYZANTINE` | 4 | 拜占庭客户端数量 |
| `ROUNDS` | 200 | 联邦学习总轮次 |
| `LOCAL_EPOCHS` | 3 | 每轮本地训练周期 |
| `BATCH_SIZE` | 128 | 批大小 |
| `LR` | 0.02 | 客户端学习率 |
| `MOMENTUM` | 0.8 | 动量参数 |
| `DP_CLIP_NORM` | 5.0 | 差分隐私梯度剪裁范数 |
| `BASE_SIGMA` | 0.05 | 基础噪声标准差 |
| `Q_LEVELS` | 1024 | 量化级数 |

### mcpr.py 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `NUM_CLIENTS` | 50 | 总客户端数量 |
| `CLIENTS_PER_ROUND` | 40 | 每轮参与客户端 |
| `ATTACK_RATIO` | 0.2 | 恶意客户端比例 |
| `METHOD` | "trimmed" | 聚合方法 |
| `TRIM_RATIO` | 0.1 | 剪裁比例 |

---

## 算法原理

### 联邦学习框架

1. **本地训练**：每个客户端在本地数据集上训练LOCAL_EPOCHS个周期
2. **梯度上传**：客户端将梯度上传到服务器
3. **拜占庭防御**：服务器应用防御机制检测和消除恶意梯度
4. **模型聚合**：安全聚合客户端梯度更新全局模型
5. **差分隐私**：在聚合过程中加入高斯噪声保护隐私

### 拜占庭防御方法

- **Trimmed Mean**：移除最大和最小的k%梯度
- **Median**：使用中值进行聚合
- **Krum**：选择与其他梯度最相似的k个梯度
- **MCPR**：多数投票和协调机制

### 差分隐私机制

- **梯度剪裁**：将梯度范数限制在DP_CLIP_NORM内
- **高斯噪声**：添加高斯噪声实现(ε, δ)-差分隐私
- **自适应σ**：根据训练进度自适应调整噪声标准差

---

## 使用方法

### 前置要求

```bash
pip install torch torchvision numpy pandas matplotlib
```

### 运行训练

#### 运行主框架（带DP和自适应稳定性）
```bash
python dara.py
```

#### 运行对比实验
```bash
python mcpr.py
```

### 输出文件

- **dara.py 输出**：
  - 模型保存：`./fltrustmn/` 目录
  - 日志：CSV格式训练记录

- **mcpr.py 输出**：
  - 训练日志：`training_log2.csv`
  - 包含精度、损失、恶意声誉等指标

---

## 性能指标

### 监控的关键指标

1. **准确率（Accuracy）**：模型在测试集上的分类准确率
2. **损失（Loss）**：训练和测试损失
3. **恶意声誉（Malicious Reputation）**：客户端的恶意程度评分
4. **隐私预算（ε, δ）**：累积的差分隐私消耗



## 实验结果

训练过程会生成详细的CSV日志，包含：
- 每轮训练的准确率和损失
- 恶意客户端的检测和隔离情况
- 隐私预算消耗情况
- 不同聚合方法的性能对比

---

## 技术细节

### 依赖库

- **PyTorch**：深度学习框架
- **TorchVision**：计算机视觉工具（CIFAR数据集）
- **NumPy**：数值计算
- **Pandas**：数据处理
- **Matplotlib**：数据可视化

### 硬件要求

- 推荐使用GPU加速（CUDA支持）
- CPU也支持但速度会较慢
- 内存需求：至少8GB

---

## 相关研究

该项目实现了以下研究领域的方法：

1. **联邦学习**：McMahan et al., FedAvg (2016)
2. **拜占庭防御**：Yin et al., Krum (2018); Chen et al., Detoxify (2019)
3. **差分隐私**：Dwork (2006); DP-SGD; 自适应隐私预算
4. **自适应优化**：自适应学习率调度

---

## 参数调整建议

### 增加隐私保护
- 提高 `BASE_SIGMA` 值
- 降低 `DP_CLIP_NORM`
- 使用 `FIXED_DP=True` 固定隐私预算

### 改进鲁棒性
- 增加 `BYZANTINE` 客户端数量测试
- 调整聚合方法的参数
- 增加本地训练周期 `LOCAL_EPOCHS`

### 加速训练
- 增加 `CLIENTS_PER_ROUND`
- 减少 `LOCAL_EPOCHS`
- 增加 `BATCH_SIZE`

---




