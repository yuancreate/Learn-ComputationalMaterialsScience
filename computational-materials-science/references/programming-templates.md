# 编程模板与示例

## 目录

1. [第一性原理计算(DFT)](#第一性原理计算dft)
2. [分子动力学(MD)](#分子动力学md)
3. [有限元方法(FEM)](#有限元方法fem)
4. [相场方法](#相场方法)
5. [机器学习应用](#机器学习应用)
6. [数据处理脚本](#数据处理脚本)

---

## 第一性原理计算(DFT)

### VASP输入文件模板

#### POSCAR（晶体结构）
```
Si diamond structure
   5.43
 0.00 0.50 0.50
 0.50 0.00 0.50
 0.50 0.50 0.00
 Si
 2
Direct
 0.00 0.00 0.00
 0.25 0.25 0.25
```

#### INCAR（计算参数）
```
# 基础参数
SYSTEM = Si calculation
ENCUT = 400          # 截断能(eV)
PREC = Accurate      # 精度

# 电子结构
ISMEAR = 0           # 0=高斯展宽
SIGMA = 0.05         # 展宽宽度(eV)
EDIFF = 1E-6         # 电子收敛标准

# 离子弛豫
IBRION = 2           # 2=共轭梯度法
ISIF = 3             # 3=全优化
EDIFFG = -0.01       # 离子收敛标准(eV/A)

# 并行
KPAR = 4             # k点并行组数
NCORE = 4            # 每个核处理的能带数
```

#### KPOINTS（k点网格）
```
Automatic mesh
0
Gamma
4 4 4
0 0 0
```

---

### Quantum ESPRESSO输入模板

```fortran
&CONTROL
    calculation = 'scf'
    prefix = 'silicon'
    outdir = './tmp'
    pseudo_dir = './pseudo'
/

&SYSTEM
    ibrav = 2
    celldm(1) = 10.26
    nat = 2
    ntyp = 1
    ecutwfc = 40.0
    ecutrho = 320.0
/

&ELECTRONS
    conv_thr = 1.0d-8
/

ATOMIC_SPECIES
 Si  28.086  Si.pbe-n-rrkjus_psl.1.0.0.UPF

ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25

K_POINTS automatic
 4 4 4 0 0 0
```

---

### 后处理脚本示例

#### 提取VASP能量数据
```python
#!/usr/bin/env python3
"""提取VASP OUTCAR中的能量数据"""

import re
import numpy as np
import matplotlib.pyplot as plt

def extract_energy(outcar='OUTCAR'):
    """从OUTCAR提取能量"""
    energies = []
    
    with open(outcar, 'r') as f:
        for line in f:
            if 'free  energy   TOTEN' in line:
                energy = float(line.split()[-2])
                energies.append(energy)
    
    return np.array(energies)

def plot_energy(energies, output='energy.png'):
    """绘制能量收敛曲线"""
    plt.figure(figsize=(8, 5))
    plt.plot(energies, 'b-o', markersize=4)
    plt.xlabel('Ion Step', fontsize=12)
    plt.ylabel('Energy (eV)', fontsize=12)
    plt.title('Energy Convergence', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'能量收敛图已保存: {output}')

if __name__ == '__main__':
    energies = extract_energy()
    print(f'提取到 {len(energies)} 个能量点')
    print(f'最终能量: {energies[-1]:.6f} eV')
    plot_energy(energies)
```

---

## 分子动力学(MD)

### LAMMPS输入脚本模板

#### 基础MD模拟
```lammps
# 初始化
units           metal
dimension       3
boundary        p p p
atom_style      atomic

# 读取结构
read_data       structure.data

# 设置原子间势
pair_style      eam/alloy
pair_coeff      * * Al.eam.alloy Al

# 设置模拟参数
timestep        0.001
thermo          100
thermo_style    custom step temp pe ke etotal press vol

# 能量最小化
minimize        1.0e-4 1.0e-6 100 1000

# NPT系综
velocity        all create 300.0 12345 mom yes rot yes
fix             1 all npt temp 300 300 0.1 iso 0 0 1.0

# 输出设置
dump            1 all custom 100 traj.lammpstrj id type x y z vx vy vz

# 运行
run             10000
```

---

### Python生成LAMMPS输入

```python
#!/usr/bin/env python3
"""生成LAMMPS输入脚本"""

def generate_lammps_input(
    structure_file='structure.data',
    potential_file='Al.eam.alloy',
    temperature=300,
    steps=10000,
    output='in.lammps'
):
    """生成LAMMPS输入脚本"""
    
    template = f"""# LAMMPS输入脚本 - 自动生成
units           metal
dimension       3
boundary        p p p
atom_style      atomic

read_data       {structure_file}

pair_style      eam/alloy
pair_coeff      * * {potential_file} Al

timestep        0.001
thermo          100
thermo_style    custom step temp pe ke etotal press vol

minimize        1.0e-4 1.0e-6 100 1000

velocity        all create {temperature}.0 12345 mom yes rot yes
fix             1 all npt temp {temperature} {temperature} 0.1 iso 0 0 1.0

dump            1 all custom 100 traj.lammpstrj id type x y z

run             {steps}
"""
    
    with open(output, 'w') as f:
        f.write(template)
    
    print(f'LAMMPS输入脚本已生成: {output}')
    return output

if __name__ == '__main__':
    generate_lammps_input(
        structure_file='Al.data',
        potential_file='Al.eam.alloy',
        temperature=300,
        steps=50000
    )
```

---

### 分析MD轨迹

```python
#!/usr/bin/env python3
"""MD轨迹分析：计算径向分布函数"""

import numpy as np
import matplotlib.pyplot as plt

def compute_rdf(xyz_file, r_max=10.0, dr=0.01):
    """计算径向分布函数"""
    
    # 读取轨迹（简化版XYZ格式）
    frames = []
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            n_atoms = int(lines[i].strip())
            frame = []
            for j in range(i+2, i+2+n_atoms):
                parts = lines[j].split()
                frame.append([float(x) for x in parts[1:4]])
            frames.append(np.array(frame))
            i += 2 + n_atoms
    
    # 计算RDF
    r_bins = np.arange(0, r_max, dr)
    rdf = np.zeros(len(r_bins) - 1)
    
    # 盒子大小（需要根据实际设置）
    L = 20.0
    
    for frame in frames:
        for i in range(len(frame)):
            for j in range(i+1, len(frame)):
                # 最小镜像距离
                rij = frame[i] - frame[j]
                rij = rij - L * np.round(rij / L)
                r = np.linalg.norm(rij)
                
                if r < r_max:
                    bin_idx = int(r / dr)
                    if bin_idx < len(rdf):
                        rdf[bin_idx] += 2  # 计数两次
    
    # 归一化
    rho = len(frame) / L**3  # 数密度
    for i in range(len(rdf)):
        r_inner = i * dr
        r_outer = (i + 1) * dr
        shell_vol = 4/3 * np.pi * (r_outer**3 - r_inner**3)
        rdf[i] /= len(frames) * len(frame) * rho * shell_vol
    
    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(r_bins[:-1], rdf, 'b-', linewidth=1.5)
    plt.xlabel('r (Å)', fontsize=12)
    plt.ylabel('g(r)', fontsize=12)
    plt.title('Radial Distribution Function', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('rdf.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('RDF已保存: rdf.png')
    return r_bins[:-1], rdf

if __name__ == '__main__':
    # compute_rdf('traj.xyz')
    print('请提供轨迹文件路径')
```

---

## 有限元方法(FEM)

### FEniCS示例

```python
#!/usr/bin/env python3
"""FEniCS热传导方程求解示例"""

from fenics import *
import matplotlib.pyplot as plt

def solve_heat_equation():
    """求解二维热传导方程"""
    
    # 创建网格
    nx, ny = 50, 50
    mesh = UnitSquareMesh(nx, ny)
    
    # 定义函数空间
    V = FunctionSpace(mesh, 'P', 1)
    
    # 边界条件
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    def boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V, u_D, boundary)
    
    # 定义变分问题
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(-6.0)
    a = dot(grad(u), grad(v)) * dx
    L = f * v * dx
    
    # 求解
    u = Function(V)
    solve(a == L, u, bc)
    
    # 绘图
    plot(u, title='Temperature Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('fem_result.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算误差
    error_L2 = errornorm(u_D, u, 'L2')
    print(f'L2误差: {error_L2:.6e}')
    
    return u

if __name__ == '__main__':
    solve_heat_equation()
```

---

### 一维有限元求解器

```python
#!/usr/bin/env python3
"""一维有限元求解器"""

import numpy as np
import matplotlib.pyplot as plt

class FEM1D:
    """一维有限元求解器"""
    
    def __init__(self, n_elements, domain=(0, 1)):
        self.n_elements = n_elements
        self.domain = domain
        self.n_nodes = n_elements + 1
        self.x = np.linspace(domain[0], domain[1], self.n_nodes)
        self.K = np.zeros((self.n_nodes, self.n_nodes))
        self.F = np.zeros(self.n_nodes)
    
    def assemble_stiffness(self, k=1.0):
        """组装刚度矩阵"""
        h = (self.domain[1] - self.domain[0]) / self.n_elements
        
        for i in range(self.n_elements):
            self.K[i, i] += k / h
            self.K[i, i+1] -= k / h
            self.K[i+1, i] -= k / h
            self.K[i+1, i+1] += k / h
    
    def assemble_load(self, f_func):
        """组装载荷向量"""
        h = (self.domain[1] - self.domain[0]) / self.n_elements
        
        for i in range(self.n_elements):
            x_mid = (self.x[i] + self.x[i+1]) / 2
            f_val = f_func(x_mid)
            self.F[i] += f_val * h / 2
            self.F[i+1] += f_val * h / 2
    
    def apply_bc(self, u_left=None, u_right=None):
        """应用边界条件"""
        if u_left is not None:
            self.K[0, :] = 0
            self.K[0, 0] = 1
            self.F[0] = u_left
        
        if u_right is not None:
            self.K[-1, :] = 0
            self.K[-1, -1] = 1
            self.F[-1] = u_right
    
    def solve(self):
        """求解线性系统"""
        u = np.linalg.solve(self.K, self.F)
        return u
    
    def plot_solution(self, u, title='FEM Solution'):
        """绘制解"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.x, u, 'b-o', markersize=4)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('u', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig('fem_1d_solution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f'解已保存: fem_1d_solution.png')

if __name__ == '__main__':
    # 示例：-u'' = f, u(0)=0, u(1)=0
    fem = FEM1D(n_elements=20)
    fem.assemble_stiffness(k=1.0)
    fem.assemble_load(lambda x: np.sin(np.pi * x))
    fem.apply_bc(u_left=0, u_right=0)
    u = fem.solve()
    fem.plot_solution(u, title='FEM Solution: -u\'\' = sin(πx)')
```

---

## 相场方法

### 一维相场模型

```python
#!/usr/bin/env python3
"""一维相场模型：Spinodal分解"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PhaseField1D:
    """一维相场模型求解器"""
    
    def __init__(self, nx=200, dx=1.0, dt=0.1):
        self.nx = nx
        self.dx = dx
        self.dt = dt
        self.x = np.arange(nx) * dx
        
        # 初始化浓度场
        self.c = 0.5 + 0.1 * np.random.randn(nx)
        self.c = np.clip(self.c, 0, 1)
    
    def free_energy_derivative(self, c):
        """自由能对浓度的导数（双阱势）"""
        # f(c) = c^2(1-c)^2
        # df/dc = 2c(1-c)(1-2c)
        return 2 * c * (1 - c) * (1 - 2*c)
    
    def laplacian(self, f):
        """计算拉普拉斯算子"""
        return (np.roll(f, -1) + np.roll(f, 1) - 2*f) / (self.dx**2)
    
    def step(self):
        """时间步进（Cahn-Hilliard方程）"""
        kappa = 2.0  # 梯度能量系数
        M = 1.0      # 迁移率
        
        # 化学势 μ = df/dc - κ∇²c
        mu = self.free_energy_derivative(self.c) - kappa * self.laplacian(self.c)
        
        # ∂c/∂t = M∇²μ
        dc_dt = M * self.laplacian(mu)
        
        self.c += self.dt * dc_dt
        self.c = np.clip(self.c, 0, 1)
    
    def run(self, steps=1000):
        """运行模拟"""
        history = [self.c.copy()]
        
        for i in range(steps):
            self.step()
            if i % 50 == 0:
                history.append(self.c.copy())
        
        return np.array(history)
    
    def animate(self, history, interval=100, output='phasefield.gif'):
        """生成动画"""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        def update(frame):
            ax.clear()
            ax.plot(self.x, history[frame], 'b-', linewidth=1.5)
            ax.set_xlim(0, self.x[-1])
            ax.set_ylim(0, 1)
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('c', fontsize=12)
            ax.set_title(f'Frame {frame}/{len(history)}', fontsize=14)
            ax.grid(True, alpha=0.3)
        
        anim = FuncAnimation(fig, update, frames=len(history),
                           interval=interval, blit=False)
        anim.save(output, writer='pillow', fps=10)
        plt.close()
        print(f'动画已保存: {output}')

if __name__ == '__main__':
    pf = PhaseField1D(nx=200, dx=1.0, dt=0.01)
    history = pf.run(steps=5000)
    pf.animate(history, interval=100)
```

---

### 二维相场模型

```python
#!/usr/bin/env python3
"""二维相场模型：晶粒生长"""

import numpy as np
import matplotlib.pyplot as plt

class PhaseField2D:
    """二维多相场模型"""
    
    def __init__(self, nx=100, ny=100, dx=1.0, dt=0.1, n_grains=5):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dt = dt
        self.n_grains = n_grains
        
        # 初始化晶粒场
        self.phi = np.zeros((n_grains, nx, ny))
        for i in range(n_grains):
            cx = np.random.randint(0, nx)
            cy = np.random.randint(0, ny)
            self.phi[i] = np.exp(-((np.arange(nx)[:, None] - cx)**2 +
                                   (np.arange(ny)[None, :] - cy)**2) / 200)
    
    def laplacian(self, f):
        """二维拉普拉斯算子"""
        return (np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +
                np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) - 4*f) / (self.dx**2)
    
    def step(self):
        """时间步进"""
        alpha = 1.0  # 界面能参数
        beta = 1.0   # 动力学系数
        
        for i in range(self.n_grains):
            # 其他晶粒的影响
            phi_sum = np.sum(self.phi, axis=0) - self.phi[i]
            
            # 相场演化方程
            dphi = alpha * self.laplacian(self.phi[i]) - \
                   beta * self.phi[i] * (1 - self.phi[i]) * \
                   (self.phi[i] - 0.5 + phi_sum)
            
            self.phi[i] += self.dt * dphi
            self.phi[i] = np.clip(self.phi[i], 0, 1)
    
    def plot(self, output='phasefield_2d.png'):
        """绘制晶粒结构"""
        grain_map = np.argmax(self.phi, axis=0)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(grain_map, cmap='jet', origin='lower')
        plt.colorbar(label='Grain ID')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.title('Grain Structure', fontsize=14)
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'晶粒结构图已保存: {output}')

if __name__ == '__main__':
    pf = PhaseField2D(nx=100, ny=100, n_grains=6)
    for step in range(500):
        pf.step()
        if step % 100 == 0:
            print(f'Step {step}')
    pf.plot()
```

---

## 机器学习应用

### 材料性能预测

```python
#!/usr/bin/env python3
"""机器学习预测材料性能"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_material_model(data_file):
    """训练材料性能预测模型"""
    
    # 加载数据
    # 假设CSV格式：特征列 + 目标列
    df = pd.read_csv(data_file)
    
    # 分离特征和目标
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'MSE: {mse:.4f}')
    print(f'R²: {r2:.4f}')
    
    # 绘制预测vs实际
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual', fontsize=12)
    plt.ylabel('Predicted', fontsize=12)
    plt.title('Actual vs Predicted', fontsize=14)
    plt.savefig('ml_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 特征重要性
    importances = model.feature_importances_
    feature_names = df.columns[:-1]
    
    plt.figure(figsize=(10, 5))
    plt.bar(feature_names, importances)
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title('Feature Importance', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model, r2

if __name__ == '__main__':
    # 示例使用
    print('请提供材料数据CSV文件')
```

---

### 神经网络势函数

```python
#!/usr/bin/env python3
"""神经网络势函数训练示例"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NeuralNetworkPotential(nn.Module):
    """神经网络原子间势"""
    
    def __init__(self, input_dim=10, hidden_layers=[64, 64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_nn_potential(X_train, y_train, epochs=1000):
    """训练神经网络势"""
    
    # 转换为PyTorch张量
    X = torch.FloatTensor(X_train)
    y = torch.FloatTensor(y_train).unsqueeze(1)
    
    # 创建模型
    model = NeuralNetworkPotential(input_dim=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.6f}')
    
    return model, losses

if __name__ == '__main__':
    # 示例：生成模拟数据
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.sum(X**2, axis=1) + np.random.randn(1000) * 0.1
    
    model, losses = train_nn_potential(X, y, epochs=500)
    print('训练完成')
```

---

## 数据处理脚本

### 批量文件处理

```python
#!/usr/bin/env python3
"""批量处理计算结果"""

import os
import glob
import numpy as np
import pandas as pd

def batch_process(directory, pattern='*.dat'):
    """批量处理目录下的数据文件"""
    
    files = glob.glob(os.path.join(directory, pattern))
    results = []
    
    for filepath in files:
        try:
            data = np.loadtxt(filepath)
            
            # 提取统计量
            result = {
                'file': os.path.basename(filepath),
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data)
            }
            results.append(result)
            
        except Exception as e:
            print(f'处理 {filepath} 失败: {e}')
    
    df = pd.DataFrame(results)
    df.to_csv('batch_results.csv', index=False)
    print(f'处理完成：{len(results)} 个文件')
    return df

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        batch_process(sys.argv[1])
    else:
        print('用法: python batch_process.py <directory>')
```

---

### 数据格式转换

```python
#!/usr/bin/env python3
"""常用数据格式转换"""

import json
import pandas as pd
import numpy as np

def convert_format(input_file, output_file, input_format='csv', output_format='json'):
    """通用数据格式转换"""
    
    # 读取
    if input_format == 'csv':
        df = pd.read_csv(input_file)
    elif input_format == 'json':
        df = pd.read_json(input_file)
    elif input_format == 'xlsx':
        df = pd.read_excel(input_file)
    else:
        raise ValueError(f'不支持的输入格式: {input_format}')
    
    # 写入
    if output_format == 'csv':
        df.to_csv(output_file, index=False)
    elif output_format == 'json':
        df.to_json(output_file, orient='records', indent=2)
    elif output_format == 'xlsx':
        df.to_excel(output_file, index=False)
    else:
        raise ValueError(f'不支持的输出格式: {output_format}')
    
    print(f'转换完成: {input_file} -> {output_file}')

if __name__ == '__main__':
    # 示例
    convert_format('data.csv', 'data.json', 'csv', 'json')
```

---

## 代码编写建议

### 代码规范

1. **命名规范**：
   - 变量：`lowercase_with_underscores`
   - 类：`CamelCase`
   - 常量：`UPPERCASE_WITH_UNDERSCORES`

2. **文档字符串**：
```python
def function(arg1, arg2):
    """
    函数简述
    
    Args:
        arg1: 参数1说明
        arg2: 参数2说明
    
    Returns:
        返回值说明
    
    Raises:
        可能的异常说明
    """
    pass
```

3. **错误处理**：
```python
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f'操作失败: {e}')
    raise
```

### 性能优化

1. **向量化操作**：使用NumPy避免循环
2. **内存管理**：及时释放大数组
3. **并行计算**：使用`multiprocessing`或`joblib`

### 版本控制

```bash
# .gitignore示例
*.pyc
__pycache__/
*.out
*.err
OUTCAR
CONTCAR
*.traj
*.log
```
