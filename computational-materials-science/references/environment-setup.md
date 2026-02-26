# 计算环境配置参考

## 目录

1. [个人电脑环境](#个人电脑环境)
2. [服务器环境](#服务器环境)
3. [计算集群环境](#计算集群环境)
4. [Python环境管理](#python环境管理)
5. [作业调度系统](#作业调度系统)
6. [并行计算配置](#并行计算配置)

---

## 个人电脑环境

### Windows系统

#### WSL2 + Ubuntu（推荐）

1. **启用WSL2**：
```powershell
# 以管理员身份运行PowerShell
wsl --install
wsl --set-default-version 2
```

2. **安装Ubuntu**：
```powershell
wsl --install -d Ubuntu-22.04
```

3. **配置Ubuntu环境**：
```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装编译工具
sudo apt install build-essential gfortran openmpi-bin libopenmpi-dev

# 安装Python环境
sudo apt install python3 python3-pip python3-venv
```

4. **安装图形界面支持**：
```bash
# 安装X服务器（Windows端）：VcXsrv或X410
# WSL中配置
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
```

#### 原生Windows软件

**适用于Windows的计算软件**：
- Materials Studio
- VESTA
- OVITO
- LAMMPS（Windows预编译版）
- Quantum ESPRESSO（Windows版）

---

### macOS系统

1. **安装开发工具**：
```bash
xcode-select --install
```

2. **安装Homebrew**：
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

3. **安装编译工具**：
```bash
brew install gcc open-mpi python
```

4. **安装计算软件**：
```bash
# 使用Homebrew
brew install lammps
brew install quantum-espresso

# 或使用Conda
conda install -c conda-forge lammps
```

---

### Linux系统（Ubuntu为例）

**完整环境配置脚本**：

```bash
#!/bin/bash
# 系统更新
sudo apt update && sudo apt upgrade -y

# 编译工具链
sudo apt install -y build-essential gfortran cmake git

# MPI环境
sudo apt install -y openmpi-bin libopenmpi-dev

# 数学库
sudo apt install -y libblas-dev liblapack-dev libscalapack-mpi-dev

# Python环境
sudo apt install -y python3 python3-pip python3-venv python3-dev
pip3 install --upgrade pip

# 常用Python包
pip3 install numpy scipy matplotlib pandas ase pymatgen

# 可视化工具
sudo apt install -y vesta ovito xcrysden

echo "基础环境配置完成!"
```

---

## 服务器环境

### 远程连接

**SSH配置**：
```bash
# 本地生成密钥
ssh-keygen -t rsa -b 4096

# 复制公钥到服务器
ssh-copy-id username@server.address

# 配置~/.ssh/config
Host myserver
    HostName server.address
    User username
    Port 22
    ForwardX11 yes
```

**文件传输**：
```bash
# 使用SCP
scp file.txt user@server:~/path/

# 使用Rsync（推荐）
rsync -avz local_dir/ user@server:~/remote_dir/
```

---

### 服务器端配置

**用户环境变量**（~/.bashrc）：

```bash
# 编译器
export CC=gcc
export CXX=g++
export FC=gfortran

# MPI
export PATH=/usr/lib/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/openmpi/lib:$LD_LIBRARY_PATH

# 自定义软件
export PATH=$HOME/software/bin:$PATH
export LD_LIBRARY_PATH=$HOME/software/lib:$LD_LIBRARY_PATH

# Python
export PATH=$HOME/.local/bin:$PATH
```

---

## 计算集群环境

### 模块系统

**常用模块命令**：

```bash
# 查看可用模块
module avail

# 查看已加载模块
module list

# 加载模块
module load intel/2021 impi/2021

# 卸载模块
module unload intel/2021

# 重置模块
module purge
```

**示例：配置VASP计算环境**：
```bash
module purge
module load intel/2021
module load impi/2021
module load vasp/6.3
```

---

### 存储系统

**典型集群存储结构**：
```
/home/username/        # 主目录，小容量，存放脚本配置
/work/username/        # 工作目录，中等容量，存放计算任务
/scratch/username/     # 临时目录，大容量，快速读写
```

**使用建议**：
- 计算在 `/scratch` 进行
- 结果定期备份到 `/work`
- 重要数据备份到 `/home` 或本地

---

## Python环境管理

### Conda环境（推荐）

**安装Miniconda**：
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**创建计算环境**：
```bash
# 创建DFT计算环境
conda create -n dft python=3.10
conda activate dft
conda install -c conda-forge numpy scipy matplotlib ase pymatgen

# 创建MD计算环境
conda create -n md python=3.10
conda activate md
conda install -c conda-forge numpy scipy matplotlib mdanalysis ovito

# 创建机器学习环境
conda create -n ml python=3.10
conda activate ml
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge scikit-learn pandas
```

---

### venv + pip

**创建虚拟环境**：
```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

**requirements.txt示例**：
```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
pandas>=1.3.0
ase>=3.22.0
pymatgen>=2022.0.0
scikit-learn>=1.0.0
torch>=2.0.0
```

---

## 作业调度系统

### SLURM

**常用命令**：
```bash
# 查看队列状态
squeue -u $USER

# 提交作业
sbatch job.sh

# 取消作业
scancel <job_id>

# 查看节点信息
sinfo

# 查看历史作业
sacct -u $USER
```

**作业脚本示例**：

```bash
#!/bin/bash
#SBATCH --job-name=vasp_calc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --time=72:00:00
#SBATCH --partition=compute
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load intel/2021 impi/2021 vasp/6.3

cd $SLURM_SUBMIT_DIR
mpirun -np $SLURM_NTASKS vasp_std
```

---

### PBS

**常用命令**：
```bash
# 提交作业
qsub job.pbs

# 查看作业
qstat -u $USER

# 取消作业
qdel <job_id>
```

**作业脚本示例**：

```bash
#!/bin/bash
#PBS -N vasp_calc
#PBS -l nodes=2:ppn=24
#PBS -l walltime=72:00:00
#PBS -q compute
#PBS -o $PBS_JOBID.out
#PBS -e $PBS_JOBID.err

module load intel/2021 impi/2021 vasp/6.3

cd $PBS_O_WORKDIR
mpirun -np $PBS_NP vasp_std
```

---

### LSF

**常用命令**：
```bash
# 提交作业
bsub < job.lsf

# 查看作业
bjobs -u $USER

# 取消作业
bkill <job_id>
```

---

## 并行计算配置

### MPI并行

**环境变量配置**：
```bash
# Intel MPI
export I_MPI_PMI_LIBRARY=/path/to/libpmi.so
export I_MPI_FABRICS=shm:ofi

# OpenMPI
export OMPI_MCA_btl=self,vader
export OMPI_MCA_pml=ob1
```

**运行方式**：
```bash
# 本地运行
mpirun -np 8 vasp_std

# 分布式运行
mpirun -np 48 -hostfile hosts vasp_std

# GPU节点运行
mpirun -np 4 vasp_gpu
```

---

### GPU加速

**CUDA环境配置**：
```bash
module load cuda/11.8

# 验证CUDA
nvidia-smi

# 编译CUDA程序
nvcc -o program program.cu
```

**常用GPU加速软件**：
- VASP（GPU版本）
- LAMMPS（GPU包）
- GROMACS（GPU版本）
- PyTorch

---

### 性能优化

**关键参数调优**：

1. **MPI进程数**：
```bash
# 通常设置为物理核心数
# 避免超线程，除非经验证有提升
```

2. **内存分配**：
```bash
# 绑定内存到本地节点
export I_MPI_PIN_DOMAIN=omp
```

3. **K点并行**（VASP）：
```
# INCAR中设置
KPAR = 4  # 分成4组并行计算k点
```

---

## 常见问题排查

### 环境问题

**问题**：库文件找不到
```bash
# 检查库路径
ldd executable

# 添加库路径
export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH
```

**问题**：模块冲突
```bash
# 清除所有模块重新加载
module purge
module load required/modules
```

---

### 运行问题

**问题**：MPI运行失败
```bash
# 检查MPI环境
mpirun --version
which mpirun

# 使用正确的MPI实现
module load openmpi  # 或 impi
```

**问题**：作业被终止
```bash
# 查看作业日志
cat slurm-<jobid>.err

# 常见原因：
# - 内存超限
# - 时间超限
# - 节点故障
```

---

## 环境配置检查清单

配置完成后，建议运行以下检查：

```bash
#!/bin/bash
# 环境检查脚本

echo "=== 编译器检查 ==="
gcc --version
gfortran --version

echo "=== MPI检查 ==="
mpirun --version
mpicc --version

echo "=== Python检查 ==="
python3 --version
pip3 list | grep -E "numpy|scipy|matplotlib|ase"

echo "=== CUDA检查（如适用）==="
nvidia-smi

echo "=== 模块检查（集群）==="
module list

echo "=== 磁盘空间 ==="
df -h ~
df -h /scratch 2>/dev/null || echo "/scratch不存在"

echo "环境检查完成!"
```
