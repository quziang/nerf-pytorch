# NeRF 多GPU分布式训练指南

## 概述
已成功将NeRF代码适配为支持多GPU分布式训练，使用PyTorch的DistributedDataParallel (DDP)。

## 系统要求
- 2张或更多GPU
- PyTorch 1.7+
- NCCL后端支持

## 使用方法

### 1. 自动启动（推荐）
```bash
# 使用启动脚本，自动检测GPU数量
./train_distributed.sh configs/soho.txt
```

### 2. 手动启动多GPU训练
```bash
# 使用2张GPU
torchrun --nproc_per_node=2 run_nerf.py --config configs/soho.txt

# 使用所有可用GPU
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) run_nerf.py --config configs/soho.txt
```

### 3. 单GPU训练（兼容模式）
```bash
# 直接运行，自动使用单GPU
python run_nerf.py --config configs/soho.txt
```

## 配置优化

### 多GPU训练配置建议
```
# configs/soho.txt 针对多GPU的优化参数
N_rand = 2048       # 增大batch size（每张GPU分担更多射线）
lrate = 2e-4        # 可适当增大学习率
N_iters = 200000    # 减少总迭代次数（多GPU收敛更快）
```

### 内存优化
- `N_rand`: 控制每次迭代的射线数量，多GPU可以适当增大
- `chunk`: 控制并行处理的射线数，内存不够时可减小
- `netchunk`: 控制网络前向传播的batch大小

## 性能预期

### 2× RTX 3090 (24GB each)
- **理论加速比**: ~1.8x (考虑通信开销)
- **推荐配置**:
  - N_rand: 2048-4096
  - N_samples: 64
  - N_importance: 128

### 内存使用
```
总内存需求 ≈ N_rand × (N_samples + N_importance) × 特征维度
多GPU时每张卡承担: 总内存需求 / GPU数量
```

## 监控训练

### GPU使用率监控
```bash
# 实时监控GPU状态
watch -n 1 nvidia-smi
```

### 训练日志
- 只有rank 0进程会输出日志和保存模型
- 模型权重会自动去除DDP包装后保存

## 故障排除

### 常见问题
1. **NCCL错误**: 检查GPU间通信，确保使用相同的CUDA版本
2. **内存不足**: 减小N_rand、chunk或netchunk参数
3. **进程挂起**: 检查防火墙设置，确保进程间通信正常

### 测试脚本
```bash
# 测试分布式环境
torchrun --nproc_per_node=2 test_distributed.py

# 测试单GPU兼容性  
python test_single_gpu.py
```

## 代码修改总结

### 主要改动
1. **分布式初始化**: 自动检测和初始化分布式环境
2. **模型包装**: 使用DDP包装模型
3. **数据同步**: 确保梯度正确同步
4. **日志控制**: 只在主进程输出日志和保存模型
5. **兼容性**: 保持单GPU训练完全兼容

### 新增文件
- `train_distributed.sh`: 自动启动脚本
- `test_distributed.py`: 分布式环境测试
- `DISTRIBUTED_TRAINING_GUIDE.md`: 本使用指南

## 预期效果
使用2张RTX 3090训练SPE3R soho场景：
- **训练速度**: 提升约1.8倍
- **内存使用**: 每张卡约12-16GB (N_rand=2048时)
- **收敛时间**: 从单GPU的200k迭代减少到约120k迭代

开始训练：
```bash
./train_distributed.sh configs/soho.txt
```
