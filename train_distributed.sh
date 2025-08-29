#!/bin/bash

# 分布式训练启动脚本
# 用法: ./train_distributed.sh configs/soho.txt

if [ "$#" -ne 1 ]; then
    echo "用法: $0 <config_file>"
    echo "例子: $0 configs/soho.txt"
    exit 1
fi

CONFIG_FILE=$1

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "检测到 $GPU_COUNT 张GPU"

if [ $GPU_COUNT -lt 2 ]; then
    echo "警告: 只有 $GPU_COUNT 张GPU，将使用单GPU训练"
    python run_nerf.py --config $CONFIG_FILE
else
    echo "使用 $GPU_COUNT 张GPU进行分布式训练"
    
    # 使用 torchrun (推荐的方式，替代已弃用的 torch.distributed.launch)
    torchrun --nproc_per_node=$GPU_COUNT run_nerf.py --config $CONFIG_FILE
fi
