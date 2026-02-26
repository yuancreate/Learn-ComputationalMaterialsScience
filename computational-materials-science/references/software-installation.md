# 计算材料学软件安装指南

## 目录

1. [第一性原理计算软件](#第一性原理计算软件)
2. [分子动力学软件](#分子动力学软件)
3. [有限元软件](#有限元软件)
4. [相场方法软件](#相场方法软件)
5. [人工智能框架](#人工智能框架)
6. [常用工具与库](#常用工具与库)

---

## 第一性原理计算软件

### VASP (Vienna Ab initio Simulation Package)

**简介**：商业软件，广泛应用于固体物理、材料科学领域的DFT计算。

**安装步骤**（Linux系统）：

1. **获取许可**：从VASP官网购买许可，下载源代码

2. **编译准备**：
```bash
# 安装依赖
sudo apt-get install build-essential gfortran libopenblas-dev libscalapack-mpi-dev

# 加载编译器模块（集群环境）
module load intel/2021
module load impi/2021
```

3. **编译VASP**：
```bash
tar -xzf vasp.6.3.0.tar.gz
cd vasp.6.3.0
cp arch/makefile.include.linux_intel makefile.include
# 根据系统修改makefile.include
make all
```

4. **环境变量配置**：
```bash
echo 'export PATH=$PATH:/path/to/vasp/bin' >> ~/.bashrc
source ~/.bashrc
```

**验证安装**：
```bash
mpirun -np 4 vasp_std
```

---

### Quantum ESPRESSO

**简介**：开源DFT计算软件包，适合学术研究。

**安装步骤**（Ubuntu）：

```bash
# 方式一：包管理器安装
sudo apt-get install quantum-espresso

# 方式二：编译安装
git clone https://gitlab.com/QEF/q-e.git
cd q-e
./configure
make all
```

**验证安装**：
```bash
pw.x -version
```

**常用组件**：
- `pw.x`：平面波DFT计算
- `ph.x`：声子计算
- `bands.x`：能带后处理
- `dos.x`：态密度计算

---

### ABINIT

**简介**：开源DFT软件，支持GW、BSE等高级计算。

**安装**：
```bash
# 使用Conda安装
conda install -c conda-forge abinit

# 或编译安装
wget https://www.abinit.org/sites/default/files/packages/abinit-9.8.2.tar.gz
tar -xzf abinit-9.8.2.tar.gz
cd abinit-9.8.2
./configure --enable-mpi
make
make install
```

---

## 分子动力学软件

### LAMMPS

**简介**：大规模原子分子并行模拟器，开源免费。

**安装步骤**：

```bash
# Ubuntu包管理器
sudo apt-get install lammps
lmp -in in.lammps

# 编译安装（推荐）
git clone -b stable https://github.com/lammps/lammps.git
cd lammps
mkdir build && cd build
cmake ../cmake \
  -D PKG_USER-REAXC=yes \
  -D PKG_MOLECULE=yes \
  -D PKG_RIGID=yes
make -j 8
```

**常用可选包**：
- `USER-REAXC`：反应力场
- `MOLECULE`：分子体系
- `RIGID`：刚体动力学
- `GPU`：GPU加速

---

### GROMACS

**简介**：生物分子分子动力学模拟主流软件。

**安装**：
```bash
# Ubuntu
sudo apt-get install gromacs

# 编译安装
wget http://ftp.gromacs.org/gromacs/gromacs-2023.tar.gz
tar -xzf gromacs-2023.tar.gz
cd gromacs-2023
mkdir build && cd build
cmake .. -DGMX_GPU=CUDA
make -j 8
make install
source /usr/local/gromacs/bin/GMXRC
```

---

### NAMD

**简介**：大规模并行分子动力学程序。

**安装**：
```bash
# 下载预编译版本
wget https://www.ks.uiuc.edu/Research/namd/2.14/download/NAMD_2.14_Linux-x86_64.tar.gz
tar -xzf NAMD_2.14_Linux-x86_64.tar.gz
export PATH=$PATH:$(pwd)/NAMD_2.14_Linux-x86_64
```

---

## 有限元软件

### FEniCS

**简介**：开源有限元计算平台，支持Python接口。

**安装**：
```bash
# 使用Conda（推荐）
conda create -n fenicsproject -c conda-forge fenics
conda activate fenicsproject

# Docker方式
docker pull quay.io/fenicsproject/stable:latest
```

**测试安装**：
```python
from fenics import *
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 1)
print("FEniCS安装成功!")
```

---

### COMSOL Multiphysics

**简介**：商业多物理场仿真软件。

**安装**：
1. 从官网下载安装包
2. 运行安装程序：`./setup`
3. 按向导输入许可信息
4. 选择安装模块（结构力学、流体、传热等）

---

### ANSYS / ABAQUS

**简介**：商业有限元分析软件。

**安装要点**：
- 需要购买商业许可
- 安装包括：主程序、许可服务器、文档
- Linux下通常通过图形或命令行安装器完成

---

## 相场方法软件

### MOOSE Framework

**简介**：多物理场面向对象仿真环境，开源。

**安装**：
```bash
# 使用Conda
conda create -n moose moose
conda activate moose

# 或编译安装
git clone https://github.com/idaholab/moose.git
cd moose
./scripts/update_and_rebuild_libmesh.sh
make -j 8
```

---

### 自制相场代码

对于教学目的，建议从简单的一维相场模型开始：

**依赖库**：
```bash
pip install numpy matplotlib scipy
```

**参考实现**：
```python
import numpy as np
import matplotlib.pyplot as plt

def phase_field_1d(nx=200, dx=1.0, dt=0.1, steps=1000):
    """一维相场模型示例"""
    phi = np.random.rand(nx) * 0.1 + 0.5
    
    for step in range(steps):
        laplacian = np.roll(phi, 1) + np.roll(phi, -1) - 2 * phi
        phi += dt * (laplacian - phi * (1 - phi) * (phi - 0.5))
    
    plt.plot(phi)
    plt.savefig('phase_field.png')

if __name__ == '__main__':
    phase_field_1d()
```

---

## 人工智能框架

### PyTorch

**安装**：
```bash
# CPU版本
pip install torch torchvision

# GPU版本（CUDA 11.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### TensorFlow

**安装**：
```bash
# CPU版本
pip install tensorflow

# GPU版本
pip install tensorflow[and-cuda]
```

### Scikit-learn

**安装**：
```bash
pip install scikit-learn pandas
```

---

## 常用工具与库

### 可视化工具

**VESTA**：晶体结构可视化
```bash
# Ubuntu
sudo apt-get install vesta
```

**OVITO**：分子动力学轨迹可视化
```bash
# Ubuntu
sudo apt-get install ovito
```

**XCrySDen**：电子结构可视化
```bash
sudo apt-get install xcrysden
```

---

### 数据处理

**Python科学计算栈**：
```bash
pip install numpy scipy pandas matplotlib
```

**ASE (Atomic Simulation Environment)**：
```bash
pip install ase
```

**pymatgen**：
```bash
pip install pymatgen
```

---

## 集群环境安装

在计算集群上安装软件的通用流程：

1. **加载模块**：
```bash
module load intel/2021
module load impi/2021
module load cuda/11.8
```

2. **编译安装**：
```bash
mkdir -p $HOME/software
cd $HOME/software
# 按上述各软件的编译步骤执行
```

3. **环境变量**：
```bash
echo 'export PATH=$HOME/software/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$HOME/software/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

---

## 常见问题

### 编译错误

**问题**：找不到编译器
```bash
# 解决
sudo apt-get install build-essential gfortran
```

**问题**：MPI库缺失
```bash
sudo apt-get install libopenmpi-dev openmpi-bin
```

### 运行错误

**问题**：动态库找不到
```bash
export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH
```

**问题**：MPI运行失败
```bash
# 检查MPI配置
mpirun --version
# 使用正确的MPI启动命令
mpirun -np 4 vasp_std
```

---

## 推荐安装顺序

对于初次搭建计算材料学环境，建议按以下顺序安装：

1. **基础工具**：编译器、MPI、Python
2. **可视化**：VESTA、OVITO
3. **分子动力学**：LAMMPS（易上手）
4. **第一性原理**：Quantum ESPRESSO（开源）
5. **机器学习**：PyTorch + Scikit-learn
6. **高级工具**：根据需要安装商业软件
