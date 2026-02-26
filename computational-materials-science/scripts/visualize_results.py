#!/usr/bin/env python3
"""
计算材料学结果可视化工具

功能：
- 支持多种计算方法的结果可视化（DFT、MD、FEM、Phase Field、ML）
- 生成静态图表（PNG、PDF、SVG）
- 生成动画（GIF、MP4）

使用示例：
    # 绘制DFT能带结构
    python visualize_results.py --input EIGENVAL --type plot --method dft --output band.png

    # 生成MD轨迹动画
    python visualize_results.py --input trajectory.xyz --type animation --method md --output trajectory.gif
"""

import argparse
import sys
import os
from pathlib import Path

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import pandas as pd
except ImportError as e:
    print(f"错误：缺少必要的Python包。请运行：pip install matplotlib numpy pandas")
    print(f"详细信息：{e}")
    sys.exit(1)


class Visualizer:
    """计算结果可视化基类"""
    
    def __init__(self, input_file, output_file=None):
        self.input_file = Path(input_file)
        self.output_file = output_file
        self.data = None
        
    def load_data(self):
        """加载数据文件"""
        raise NotImplementedError
        
    def plot(self):
        """生成静态图"""
        raise NotImplementedError
        
    def animate(self):
        """生成动画"""
        raise NotImplementedError


class DFTVisualizer(Visualizer):
    """DFT计算结果可视化"""
    
    def load_data(self):
        """加载VASP EIGENVAL或类似格式文件"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"文件不存在：{self.input_file}")
        
        # 简化处理：假设是标准格式的能带数据
        # 列：k点路径、各能带能量
        try:
            self.data = np.loadtxt(self.input_file)
            print(f"成功加载数据：{self.data.shape}")
        except Exception as e:
            print(f"警告：无法直接加载数据，尝试解析文本格式")
            self._parse_eigenval()
    
    def _parse_eigenval(self):
        """解析VASP EIGENVAL文件格式"""
        with open(self.input_file, 'r') as f:
            lines = f.readlines()
        
        # EIGENVAL格式解析（简化版）
        # 实际使用时需要根据具体格式调整
        data_list = []
        for line in lines[7:]:  # 跳过文件头
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    data_list.append([float(x) for x in parts])
                except ValueError:
                    continue
        
        self.data = np.array(data_list) if data_list else None
    
    def plot(self):
        """绘制能带结构图"""
        if self.data is None or len(self.data) == 0:
            print("错误：无有效数据可绘制")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 绘制能带
        if self.data.ndim == 2 and self.data.shape[1] > 1:
            for i in range(1, self.data.shape[1]):
                ax.plot(self.data[:, 0], self.data[:, i], 'b-', linewidth=1)
        
        # 标记费米能级
        ax.axhline(y=0, color='r', linestyle='--', label='Fermi Level')
        
        ax.set_xlabel('k-path', fontsize=12)
        ax.set_ylabel('Energy (eV)', fontsize=12)
        ax.set_title('Band Structure', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output = self.output_file or 'band_structure.png'
        fig.savefig(output, dpi=300, bbox_inches='tight')
        print(f"图表已保存：{output}")
        plt.close(fig)
    
    def animate(self):
        """DFT结果通常不需要动画"""
        print("DFT结果可视化暂不支持动画模式")


class MDVisualizer(Visualizer):
    """分子动力学结果可视化"""
    
    def load_data(self):
        """加载MD轨迹或数据文件"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"文件不存在：{self.input_file}")
        
        suffix = self.input_file.suffix.lower()
        
        if suffix in ['.xyz', '.pdb']:
            self._parse_trajectory()
        elif suffix in ['.dat', '.txt', '.csv']:
            self.data = pd.read_csv(self.input_file, delim_whitespace=True)
        else:
            # 尝试通用加载
            try:
                self.data = np.loadtxt(self.input_file)
            except:
                self.data = pd.read_csv(self.input_file)
    
    def _parse_trajectory(self):
        """解析轨迹文件（简化版XYZ格式）"""
        frames = []
        current_frame = []
        
        with open(self.input_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 1 and parts[0].isdigit():
                    # 原子数量行，表示新帧开始
                    if current_frame:
                        frames.append(np.array(current_frame))
                    current_frame = []
                elif len(parts) >= 4:
                    # 原子行：元素 x y z
                    try:
                        current_frame.append([float(x) for x in parts[1:4]])
                    except ValueError:
                        continue
        
        if current_frame:
            frames.append(np.array(current_frame))
        
        self.trajectory = frames
        print(f"加载轨迹：{len(frames)} 帧")
    
    def plot(self):
        """绘制RDF、能量演化等图"""
        if hasattr(self, 'data') and self.data is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            if isinstance(self.data, pd.DataFrame):
                # 绘制所有列
                for col in self.data.columns[1:]:
                    ax.plot(self.data.iloc[:, 0], self.data[col], label=col)
            else:
                ax.plot(self.data[:, 0], self.data[:, 1])
            
            ax.set_xlabel('Time/Distance', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title('MD Results', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            output = self.output_file or 'md_results.png'
            fig.savefig(output, dpi=300, bbox_inches='tight')
            print(f"图表已保存：{output}")
            plt.close(fig)
        else:
            print("提示：轨迹数据建议使用动画模式展示")
    
    def animate(self):
        """生成轨迹动画"""
        if not hasattr(self, 'trajectory') or not self.trajectory:
            print("错误：无有效轨迹数据")
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def update(frame_idx):
            ax.clear()
            frame = self.trajectory[frame_idx]
            ax.scatter(frame[:, 0], frame[:, 1], c='blue', s=50, alpha=0.6)
            ax.set_xlim(frame[:, 0].min() - 1, frame[:, 0].max() + 1)
            ax.set_ylim(frame[:, 1].min() - 1, frame[:, 1].max() + 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Frame {frame_idx + 1}/{len(self.trajectory)}')
        
        anim = FuncAnimation(fig, update, frames=len(self.trajectory), 
                            interval=50, blit=False)
        
        output = self.output_file or 'trajectory.gif'
        anim.save(output, writer='pillow', fps=20)
        print(f"动画已保存：{output}")
        plt.close(fig)


class PhaseFieldVisualizer(Visualizer):
    """相场方法结果可视化"""
    
    def load_data(self):
        """加载相场数据"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"文件不存在：{self.input_file}")
        
        try:
            self.data = np.loadtxt(self.input_file)
        except:
            self.data = np.load(self.input_file) if self.input_file.suffix == '.npy' else None
    
    def plot(self):
        """绘制相场分布图"""
        if self.data is None:
            print("错误：无有效数据")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if self.data.ndim == 2:
            im = ax.imshow(self.data, cmap='RdBu', origin='lower')
            plt.colorbar(im, ax=ax)
        else:
            ax.plot(self.data)
        
        ax.set_title('Phase Field Distribution', fontsize=14)
        
        output = self.output_file or 'phasefield.png'
        fig.savefig(output, dpi=300, bbox_inches='tight')
        print(f"图表已保存：{output}")
        plt.close(fig)
    
    def animate(self):
        """相场演化动画"""
        print("相场动画需要时间序列数据，请确保输入包含多帧数据")


class FEMVisualizer(Visualizer):
    """有限元结果可视化"""
    
    def load_data(self):
        """加载有限元结果"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"文件不存在：{self.input_file}")
        
        # 尝试多种格式
        try:
            self.data = pd.read_csv(self.input_file)
        except:
            try:
                self.data = np.loadtxt(self.input_file)
            except:
                print("错误：无法识别的数据格式")
    
    def plot(self):
        """绘制应力应变云图等"""
        if self.data is None:
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if isinstance(self.data, pd.DataFrame):
            ax.plot(self.data.iloc[:, 0], self.data.iloc[:, 1], 'b-')
            ax.set_xlabel(self.data.columns[0])
            ax.set_ylabel(self.data.columns[1] if len(self.data.columns) > 1 else 'Value')
        else:
            ax.plot(self.data[:, 0], self.data[:, 1])
        
        ax.set_title('FEM Results', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        output = self.output_file or 'fem_results.png'
        fig.savefig(output, dpi=300, bbox_inches='tight')
        print(f"图表已保存：{output}")
        plt.close(fig)
    
    def animate(self):
        print("FEM结果动画需要时序数据")


class MLVisualizer(Visualizer):
    """机器学习结果可视化"""
    
    def load_data(self):
        """加载ML结果数据"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"文件不存在：{self.input_file}")
        
        try:
            self.data = pd.read_csv(self.input_file)
        except:
            self.data = np.loadtxt(self.input_file)
    
    def plot(self):
        """绘制训练曲线、预测结果等"""
        if self.data is None:
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if isinstance(self.data, pd.DataFrame):
            # 假设格式：epoch, train_loss, val_loss 或 actual, predicted
            if 'train_loss' in self.data.columns:
                ax.plot(self.data['epoch'], self.data['train_loss'], label='Train')
                if 'val_loss' in self.data.columns:
                    ax.plot(self.data['epoch'], self.data['val_loss'], label='Validation')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
            elif len(self.data.columns) >= 2:
                ax.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], alpha=0.5)
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
        else:
            ax.plot(self.data[:, 0], self.data[:, 1])
        
        ax.set_title('Machine Learning Results', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        output = self.output_file or 'ml_results.png'
        fig.savefig(output, dpi=300, bbox_inches='tight')
        print(f"图表已保存：{output}")
        plt.close(fig)
    
    def animate(self):
        print("ML结果暂不支持动画模式")


def get_visualizer(method, input_file, output_file):
    """根据方法类型返回对应的可视化器"""
    visualizers = {
        'dft': DFTVisualizer,
        'md': MDVisualizer,
        'phasefield': PhaseFieldVisualizer,
        'fem': FEMVisualizer,
        'ml': MLVisualizer
    }
    
    if method not in visualizers:
        raise ValueError(f"不支持的方法类型：{method}。支持的类型：{list(visualizers.keys())}")
    
    return visualizers[method](input_file, output_file)


def main():
    parser = argparse.ArgumentParser(
        description='计算材料学结果可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例：
  %(prog)s --input EIGENVAL --type plot --method dft --output band.png
  %(prog)s --input trajectory.xyz --type animation --method md --output traj.gif
  %(prog)s --input results.csv --type plot --method ml
        '''
    )
    
    parser.add_argument('--input', '-i', required=True, help='输入数据文件路径')
    parser.add_argument('--type', '-t', required=True, choices=['plot', 'animation'],
                       help='输出类型：plot（静态图）或 animation（动画）')
    parser.add_argument('--method', '-m', required=True,
                       choices=['dft', 'md', 'phasefield', 'fem', 'ml'],
                       help='计算方法类型')
    parser.add_argument('--output', '-o', default=None, help='输出文件路径')
    
    args = parser.parse_args()
    
    print(f"处理中：方法={args.method}, 类型={args.type}")
    
    try:
        visualizer = get_visualizer(args.method, args.input, args.output)
        visualizer.load_data()
        
        if args.type == 'plot':
            visualizer.plot()
        else:
            visualizer.animate()
            
    except Exception as e:
        print(f"错误：{e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
