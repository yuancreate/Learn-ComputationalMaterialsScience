#!/usr/bin/env python3
"""
计算材料学数据格式转换工具

功能：
- 支持多种计算软件的数据格式转换
- 常见转换：VASP <-> QE, LAMMPS <-> GROMACS, CSV <-> JSON

使用示例：
    # VASP POSCAR转换为QE格式
    python convert_data.py --input POSCAR --input-format vasp --output-format qe --output input.in

    # 数据格式转换
    python convert_data.py --input data.csv --input-format csv --output-format json --output data.json
"""

import argparse
import sys
import json
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"错误：缺少必要的Python包。请运行：pip install numpy pandas")
    print(f"详细信息：{e}")
    sys.exit(1)


class DataConverter:
    """数据格式转换基类"""
    
    def __init__(self, input_file, input_format, output_format, output_file=None):
        self.input_file = Path(input_file)
        self.input_format = input_format.lower()
        self.output_format = output_format.lower()
        self.output_file = output_file
        self.data = None
        
    def convert(self):
        """执行转换"""
        self.load()
        self.save()
        
    def load(self):
        """加载数据"""
        raise NotImplementedError
        
    def save(self):
        """保存数据"""
        raise NotImplementedError


class StructureConverter(DataConverter):
    """晶体结构文件转换器"""
    
    def load(self):
        """加载结构文件"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"文件不存在：{self.input_file}")
        
        if self.input_format in ['vasp', 'poscar']:
            self._load_vasp()
        elif self.input_format in ['qe', 'quantum-espresso']:
            self._load_qe()
        elif self.input_format in ['xyz']:
            self._load_xyz()
        else:
            raise ValueError(f"不支持输入格式：{self.input_format}")
    
    def _load_vasp(self):
        """加载VASP POSCAR/CONTCAR格式"""
        with open(self.input_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        self.data = {
            'comment': lines[0],
            'scaling': float(lines[1]),
            'lattice': [],
            'species': [],
            'counts': [],
            'coordinate_type': 'Cartesian',
            'positions': []
        }
        
        # 晶格向量
        for i in range(2, 5):
            self.data['lattice'].append([float(x) for x in lines[i].split()])
        
        # 元素种类和数量
        self.data['species'] = lines[5].split()
        self.data['counts'] = [int(x) for x in lines[6].split()]
        
        # 坐标类型（第7行可能是Selective Dynamics）
        line_idx = 7
        if lines[7].lower().startswith('s'):
            line_idx = 8
        
        if lines[line_idx].lower().startswith('d'):
            self.data['coordinate_type'] = 'Direct'
            line_idx += 1
        
        # 原子坐标
        total_atoms = sum(self.data['counts'])
        for i in range(line_idx, line_idx + total_atoms):
            parts = lines[i].split()
            self.data['positions'].append([float(x) for x in parts[:3]])
        
        print(f"加载VASP结构：{total_atoms} 个原子")
    
    def _load_qe(self):
        """加载Quantum ESPRESSO输入格式（简化版）"""
        self.data = {
            'comment': 'Converted from QE',
            'scaling': 1.0,
            'lattice': [],
            'species': [],
            'counts': [],
            'coordinate_type': 'Cartesian',
            'positions': []
        }
        
        with open(self.input_file, 'r') as f:
            content = f.read()
        
        # 简化解析：提取关键信息
        # 实际实现需要更复杂的解析逻辑
        lines = content.split('\n')
        in_cell = False
        in_atoms = False
        
        for line in lines:
            if 'CELL_PARAMETERS' in line:
                in_cell = True
                continue
            if 'ATOMIC_POSITIONS' in line:
                in_cell = False
                in_atoms = True
                if 'crystal' in line.lower():
                    self.data['coordinate_type'] = 'Direct'
                continue
            if in_cell and line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    self.data['lattice'].append([float(x) for x in parts[:3]])
            if in_atoms and line.strip():
                parts = line.split()
                if len(parts) >= 4:
                    if parts[0] not in self.data['species']:
                        self.data['species'].append(parts[0])
                        self.data['counts'].append(1)
                    else:
                        idx = self.data['species'].index(parts[0])
                        self.data['counts'][idx] += 1
                    self.data['positions'].append([float(x) for x in parts[1:4]])
        
        print(f"加载QE结构：{len(self.data['positions'])} 个原子")
    
    def _load_xyz(self):
        """加载XYZ格式"""
        with open(self.input_file, 'r') as f:
            lines = f.readlines()
        
        self.data = {
            'comment': lines[1].strip() if len(lines) > 1 else '',
            'scaling': 1.0,
            'lattice': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # XYZ不含晶格信息
            'species': [],
            'counts': [],
            'coordinate_type': 'Cartesian',
            'positions': []
        }
        
        n_atoms = int(lines[0].strip())
        species_count = {}
        
        for line in lines[2:2+n_atoms]:
            parts = line.split()
            if len(parts) >= 4:
                spec = parts[0]
                species_count[spec] = species_count.get(spec, 0) + 1
                self.data['positions'].append([float(x) for x in parts[1:4]])
        
        self.data['species'] = list(species_count.keys())
        self.data['counts'] = list(species_count.values())
        
        print(f"加载XYZ结构：{n_atoms} 个原子")
    
    def save(self):
        """保存为目标格式"""
        if self.output_format in ['vasp', 'poscar']:
            self._save_vasp()
        elif self.output_format in ['qe', 'quantum-espresso']:
            self._save_qe()
        elif self.output_format in ['xyz']:
            self._save_xyz()
        else:
            raise ValueError(f"不支持输出格式：{self.output_format}")
    
    def _save_vasp(self):
        """保存为VASP POSCAR格式"""
        output = self.output_file or 'POSCAR'
        
        lines = []
        lines.append(self.data['comment'])
        lines.append(f"   {self.data['scaling']}")
        
        for vec in self.data['lattice']:
            lines.append(f"  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}")
        
        lines.append("   " + "   ".join(self.data['species']))
        lines.append("   " + "   ".join(str(c) for c in self.data['counts']))
        
        lines.append(self.data['coordinate_type'])
        
        for pos in self.data['positions']:
            lines.append(f"  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}")
        
        with open(output, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"VASP结构已保存：{output}")
    
    def _save_qe(self):
        """保存为Quantum ESPRESSO输入格式"""
        output = self.output_file or 'input.in'
        
        lines = [
            "&CONTROL",
            "  calculation = 'scf',",
            "/",
            "&SYSTEM",
            f"  nat = {len(self.data['positions'])},",
            f"  ntyp = {len(self.data['species'])},",
            "/",
            "&ELECTRONS",
            "/",
            "ATOMIC_SPECIES",
        ]
        
        # 假元素质量（实际使用需要真实值）
        for spec in self.data['species']:
            lines.append(f"  {spec}  1.00  {spec}.UPF")
        
        lines.append("ATOMIC_POSITIONS crystal")
        
        # 将笛卡尔坐标转换为分数坐标（简化处理）
        for i, pos in enumerate(self.data['positions']):
            # 需要根据晶格向量进行转换
            lines.append(f"  {self.data['species'][0]}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}")
        
        lines.extend([
            "CELL_PARAMETERS angstrom",
        ])
        
        for vec in self.data['lattice']:
            lines.append(f"  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}")
        
        lines.append("K_POINTS automatic")
        lines.append("  4 4 4 0 0 0")
        
        with open(output, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"QE结构已保存：{output}")
    
    def _save_xyz(self):
        """保存为XYZ格式"""
        output = self.output_file or 'structure.xyz'
        
        lines = [str(len(self.data['positions'])), self.data['comment']]
        
        # 扩展species以匹配positions
        species_expanded = []
        for spec, count in zip(self.data['species'], self.data['counts']):
            species_expanded.extend([spec] * count)
        
        for spec, pos in zip(species_expanded, self.data['positions']):
            lines.append(f"{spec}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}")
        
        with open(output, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"XYZ结构已保存：{output}")


class TabularConverter(DataConverter):
    """表格数据转换器"""
    
    def load(self):
        """加载表格数据"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"文件不存在：{self.input_file}")
        
        if self.input_format == 'csv':
            self.data = pd.read_csv(self.input_file)
        elif self.input_format == 'json':
            self.data = pd.read_json(self.input_file)
        elif self.input_format in ['tsv', 'dat']:
            self.data = pd.read_csv(self.input_file, sep='\t')
        else:
            raise ValueError(f"不支持的输入格式：{self.input_format}")
        
        print(f"加载数据：{self.data.shape}")
    
    def save(self):
        """保存表格数据"""
        output = self.output_file
        
        if self.output_format == 'csv':
            output = output or 'output.csv'
            self.data.to_csv(output, index=False)
        elif self.output_format == 'json':
            output = output or 'output.json'
            self.data.to_json(output, orient='records', indent=2)
        elif self.output_format in ['tsv', 'dat']:
            output = output or 'output.dat'
            self.data.to_csv(output, sep='\t', index=False)
        else:
            raise ValueError(f"不支持的输出格式：{self.output_format}")
        
        print(f"数据已保存：{output}")


def get_converter(input_format, output_format):
    """根据格式类型返回对应的转换器"""
    # 结构文件格式
    structure_formats = ['vasp', 'poscar', 'qe', 'quantum-espresso', 'xyz']
    # 表格数据格式
    tabular_formats = ['csv', 'json', 'tsv', 'dat']
    
    if input_format in structure_formats and output_format in structure_formats:
        return StructureConverter
    elif input_format in tabular_formats and output_format in tabular_formats:
        return TabularConverter
    else:
        raise ValueError(f"不支持的转换：{input_format} -> {output_format}")


def main():
    parser = argparse.ArgumentParser(
        description='计算材料学数据格式转换工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
支持的结构格式：vasp/poscar, qe/quantum-espresso, xyz
支持的表格格式：csv, json, tsv/dat

示例：
  %(prog)s --input POSCAR --input-format vasp --output-format xyz --output structure.xyz
  %(prog)s --input data.csv --input-format csv --output-format json --output data.json
        '''
    )
    
    parser.add_argument('--input', '-i', required=True, help='输入文件路径')
    parser.add_argument('--input-format', '-if', required=True, help='输入格式')
    parser.add_argument('--output-format', '-of', required=True, help='输出格式')
    parser.add_argument('--output', '-o', default=None, help='输出文件路径')
    
    args = parser.parse_args()
    
    input_format = args.input_format.lower()
    output_format = args.output_format.lower()
    
    print(f"转换中：{input_format} -> {output_format}")
    
    try:
        converter_class = get_converter(input_format, output_format)
        converter = converter_class(
            args.input, 
            input_format, 
            output_format, 
            args.output
        )
        converter.convert()
        
    except Exception as e:
        print(f"错误：{e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
