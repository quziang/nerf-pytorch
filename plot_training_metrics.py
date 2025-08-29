#!/usr/bin/env python3
"""
训练指标可视化脚本
用于从日志文件或保存的数据中生成训练曲线图表

使用方法:
1. 从CSV文件生成图表: python plot_training_metrics.py --csv_path logs/experiment/training_data.csv
2. 从NumPy文件生成图表: python plot_training_metrics.py --npz_path logs/experiment/training_data.npz
3. 从日志文件解析数据: python plot_training_metrics.py --log_dir logs/experiment
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
import re
from glob import glob


def parse_log_files(log_dir):
    """从日志目录中解析训练数据"""
    iterations = []
    losses = []
    psnrs = []
    
    # 查找可能的日志文件
    log_files = glob(os.path.join(log_dir, "*.log"))
    if not log_files:
        # 如果没有.log文件，尝试查找其他可能的日志文件
        log_files = glob(os.path.join(log_dir, "*.txt"))
    
    if not log_files:
        print(f"No log files found in {log_dir}")
        return None, None, None
    
    print(f"Found log files: {log_files}")
    
    # 正则表达式匹配训练日志
    pattern = r"\[TRAIN\] Iter: (\d+) Loss: ([\d.e-]+)\s+PSNR: ([\d.]+)"
    
    for log_file in log_files:
        print(f"Parsing {log_file}...")
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    match = re.search(pattern, line)
                    if match:
                        iteration = int(match.group(1))
                        loss = float(match.group(2))
                        psnr = float(match.group(3))
                        
                        iterations.append(iteration)
                        losses.append(loss)
                        psnrs.append(psnr)
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
    
    if not iterations:
        print("No training data found in log files")
        return None, None, None
    
    # 按迭代次数排序
    sorted_data = sorted(zip(iterations, losses, psnrs))
    iterations, losses, psnrs = zip(*sorted_data)
    
    return list(iterations), list(losses), list(psnrs)


def load_csv_data(csv_path):
    """从CSV文件加载训练数据"""
    iterations = []
    losses = []
    psnrs = []
    
    try:
        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                iterations.append(int(row['Iteration']))
                losses.append(float(row['Loss']))
                psnrs.append(float(row['PSNR']))
        
        return iterations, losses, psnrs
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return None, None, None


def load_npz_data(npz_path):
    """从NumPy文件加载训练数据"""
    try:
        data = np.load(npz_path)
        iterations = data['iterations'].tolist()
        losses = data['losses'].tolist()
        psnrs = data['psnrs'].tolist()
        
        return iterations, losses, psnrs
    except Exception as e:
        print(f"Error loading NumPy data: {e}")
        return None, None, None


def plot_metrics(iterations, losses, psnrs, output_dir=None, experiment_name="training"):
    """生成训练指标图表"""
    if not iterations:
        print("No data to plot")
        return
    
    print(f"Plotting {len(iterations)} data points...")
    
    # 设置matplotlib中文字体支持（可选）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建分离的图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制Loss曲线
    ax1.plot(iterations, losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss vs Iteration')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')  # 使用对数尺度，因为loss通常会快速下降
    
    # 绘制PSNR曲线
    ax2.plot(iterations, psnrs, 'r-', linewidth=2, label='Training PSNR', alpha=0.8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Training PSNR vs Iteration')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if output_dir:
        plot_path = os.path.join(output_dir, f'{experiment_name}_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Separate plots saved to: {plot_path}")
        
        # 保存PDF版本
        plot_path_pdf = os.path.join(output_dir, f'{experiment_name}_metrics.pdf')
        plt.savefig(plot_path_pdf, bbox_inches='tight')
        print(f"Separate plots (PDF) saved to: {plot_path_pdf}")
    
    plt.show()
    plt.close()
    
    # 创建组合图表
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 左y轴为Loss
    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss', color=color)
    line1 = ax1.plot(iterations, losses, color=color, linewidth=2, alpha=0.8, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 右y轴为PSNR
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('PSNR (dB)', color=color)
    line2 = ax2.plot(iterations, psnrs, color=color, linewidth=2, alpha=0.8, label='PSNR')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 添加图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    
    ax1.set_title('Training Loss and PSNR vs Iteration')
    
    if output_dir:
        combined_path = os.path.join(output_dir, f'{experiment_name}_combined.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {combined_path}")
    
    plt.show()
    plt.close()


def print_statistics(iterations, losses, psnrs):
    """打印训练统计信息"""
    if not iterations:
        return
    
    print("\n=== Training Statistics ===")
    print(f"Total data points: {len(iterations)}")
    print(f"Iteration range: {min(iterations)} - {max(iterations)}")
    print(f"Final Loss: {losses[-1]:.6f}")
    print(f"Final PSNR: {psnrs[-1]:.2f} dB")
    print(f"Best PSNR: {max(psnrs):.2f} dB (at iteration {iterations[psnrs.index(max(psnrs))]})")
    print(f"Lowest Loss: {min(losses):.6f} (at iteration {iterations[losses.index(min(losses))]})")
    
    # 计算改进情况
    if len(losses) > 1:
        loss_improvement = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"Loss improvement: {loss_improvement:.1f}%")
    
    if len(psnrs) > 1:
        psnr_improvement = psnrs[-1] - psnrs[0]
        print(f"PSNR improvement: {psnr_improvement:.2f} dB")


def main():
    parser = argparse.ArgumentParser(description='Plot NeRF training metrics')
    parser.add_argument('--csv_path', type=str, help='Path to CSV file with training data')
    parser.add_argument('--npz_path', type=str, help='Path to NumPy file with training data')
    parser.add_argument('--log_dir', type=str, help='Directory containing log files to parse')
    parser.add_argument('--output_dir', type=str, help='Output directory for plots (default: same as input)')
    parser.add_argument('--experiment_name', type=str, default='training', help='Name for output files')
    
    args = parser.parse_args()
    
    iterations, losses, psnrs = None, None, None
    
    # 加载数据
    if args.csv_path:
        print(f"Loading data from CSV: {args.csv_path}")
        iterations, losses, psnrs = load_csv_data(args.csv_path)
        if not args.output_dir:
            args.output_dir = os.path.dirname(args.csv_path)
    elif args.npz_path:
        print(f"Loading data from NumPy file: {args.npz_path}")
        iterations, losses, psnrs = load_npz_data(args.npz_path)
        if not args.output_dir:
            args.output_dir = os.path.dirname(args.npz_path)
    elif args.log_dir:
        print(f"Parsing log files in: {args.log_dir}")
        iterations, losses, psnrs = parse_log_files(args.log_dir)
        if not args.output_dir:
            args.output_dir = args.log_dir
    else:
        print("No input specified. Please provide --csv_path, --npz_path, or --log_dir")
        return
    
    if iterations is None or not iterations:
        print("No training data found or loaded")
        return
    
    # 创建输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 打印统计信息
    print_statistics(iterations, losses, psnrs)
    
    # 生成图表
    plot_metrics(iterations, losses, psnrs, args.output_dir, args.experiment_name)
    
    print(f"\nPlotting completed! Check {args.output_dir} for output files.")


if __name__ == '__main__':
    main()
