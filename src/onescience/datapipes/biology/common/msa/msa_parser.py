"""
统一的MSA解析器

支持a3m和Stockholm格式
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence
from pathlib import Path
import re

from onescience.datapipes.biology.common.utils.file_utils import (
    open_file,
    detect_file_format,
)


@dataclass
class MSA:
    """
    统一的MSA数据格式
    
    Attributes
    ----------
    sequences : List[str]
        序列列表
    deletion_matrix : List[List[int]]
        删除矩阵（deletion matrix）
    descriptions : List[str]
        序列描述列表
    """
    sequences: List[str]
    deletion_matrix: List[List[int]]
    descriptions: List[str]
    
    def __post_init__(self):
        """验证数据一致性"""
        if not (len(self.sequences) == len(self.deletion_matrix) == len(self.descriptions)):
            raise ValueError(
                "All fields for an MSA must have the same length. "
                f"Got sequences={len(self.sequences)}, "
                f"deletion_matrix={len(self.deletion_matrix)}, "
                f"descriptions={len(self.descriptions)}"
            )
    
    def __len__(self) -> int:
        """返回序列数量"""
        return len(self.sequences)
    
    def truncate(self, max_seqs: int) -> "MSA":
        """
        截断MSA到指定序列数
        
        Parameters
        ----------
        max_seqs : int
            最大序列数
            
        Returns
        -------
        MSA
            截断后的MSA
        """
        return MSA(
            sequences=self.sequences[:max_seqs],
            deletion_matrix=self.deletion_matrix[:max_seqs],
            descriptions=self.descriptions[:max_seqs],
        )


class MSAParser:
    """
    统一的MSA解析器
    
    支持格式：
    - a3m (HH-suite格式)
    - Stockholm (Stockholm格式)
    """
    
    @staticmethod
    def parse_a3m(a3m_string: str) -> MSA:
        """
        解析a3m格式的MSA
        
        Parameters
        ----------
        a3m_string : str
            a3m格式的字符串
            
        Returns
        -------
        MSA
            解析后的MSA对象
        """
        sequences = []
        deletion_matrix = []
        descriptions = []
        
        current_seq = ""
        current_deletions = []
        current_desc = ""
        
        for line in a3m_string.splitlines():
            line = line.strip()
            
            if line.startswith(">"):
                # 保存上一个序列
                if current_seq:
                    sequences.append(current_seq)
                    deletion_matrix.append(current_deletions)
                    descriptions.append(current_desc)
                
                # 开始新序列
                current_desc = line[1:]  # 移除 '>'
                current_seq = ""
                current_deletions = []
            elif line:
                # 解析序列行（包含小写字母表示插入）
                seq_line = ""
                del_line = []
                
                for char in line:
                    if char.isupper() or char == '-':
                        # 标准字符或gap
                        seq_line += char
                        del_line.append(0)
                    elif char.islower():
                        # 小写字母表示插入，在序列中保留，删除矩阵中标记
                        seq_line += char.upper()
                        del_line.append(0)
                    elif char.isdigit():
                        # 数字表示删除的列数
                        if del_line:
                            del_line[-1] = del_line[-1] * 10 + int(char)
                        else:
                            del_line.append(int(char))
                    else:
                        # 其他字符（如空格）跳过
                        continue
                
                current_seq += seq_line
                # 扩展删除矩阵
                while len(current_deletions) < len(current_seq):
                    current_deletions.append(0)
                # 更新删除矩阵
                for i, del_val in enumerate(del_line):
                    idx = len(current_seq) - len(del_line) + i
                    if idx < len(current_deletions):
                        current_deletions[idx] = del_val
        
        # 保存最后一个序列
        if current_seq:
            sequences.append(current_seq)
            deletion_matrix.append(current_deletions)
            descriptions.append(current_desc)
        
        return MSA(
            sequences=sequences,
            deletion_matrix=deletion_matrix,
            descriptions=descriptions,
        )
    
    @staticmethod
    def parse_stockholm(sto_string: str) -> MSA:
        """
        解析Stockholm格式的MSA
        
        Parameters
        ----------
        sto_string : str
            Stockholm格式的字符串
            
        Returns
        -------
        MSA
            解析后的MSA对象
        """
        sequences = []
        deletion_matrix = []
        descriptions = []
        
        seq_dict = {}  # name -> sequence
        in_alignment = False
        
        for line in sto_string.splitlines():
            line = line.strip()
            
            if line.startswith("# STOCKHOLM"):
                in_alignment = True
                continue
            elif line.startswith("//"):
                in_alignment = False
                break
            elif line.startswith("#"):
                continue  # 跳过注释
            elif not line:
                continue  # 跳过空行
            elif in_alignment and not line.startswith("#"):
                # 解析序列行
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    sequence = parts[1]
                    
                    if name not in seq_dict:
                        seq_dict[name] = ""
                        descriptions.append(name)
                    
                    seq_dict[name] += sequence
        
        # 转换为列表格式
        for desc in descriptions:
            seq = seq_dict[desc]
            sequences.append(seq)
            
            # 计算删除矩阵（Stockholm格式中，'.'表示删除）
            deletions = []
            for char in seq:
                if char == '.':
                    deletions.append(1)
                else:
                    deletions.append(0)
            deletion_matrix.append(deletions)
        
        return MSA(
            sequences=sequences,
            deletion_matrix=deletion_matrix,
            descriptions=descriptions,
        )
    
    @staticmethod
    def parse_file(path: Path, format: Optional[str] = None) -> MSA:
        """
        从文件解析MSA（支持压缩文件）
        
        Parameters
        ----------
        path : Path
            MSA文件路径（支持.gz, .bz2, .xz压缩文件）
        format : Optional[str]
            格式（"a3m" 或 "stockholm"），如果为None则自动检测
            
        Returns
        -------
        MSA
            解析后的MSA对象
        """
        with open_file(path, 'r') as f:
            content = f.read()
        
        # 自动检测格式
        if format is None:
            # 首先尝试基于扩展名和内容检测
            detected_format = detect_file_format(path)
            if detected_format in ['a3m', 'stockholm', 'clustal', 'phylip']:
                format = detected_format
            elif '>' in content[:100] and any(c.islower() for c in content[:1000]):
                format = 'a3m'
            elif '# STOCKHOLM' in content:
                format = 'stockholm'
            else:
                raise ValueError(
                    f"Could not detect MSA format for file: {path}. "
                    f"Supported formats: a3m, stockholm, clustal, phylip"
                )
        
        if format == 'a3m':
            return MSAParser.parse_a3m(content)
        elif format == 'stockholm':
            return MSAParser.parse_stockholm(content)
        else:
            raise ValueError(
                f"Unsupported MSA format: {format}. "
                f"Currently only 'a3m' and 'stockholm' are implemented. "
                f"Detected format: {format}"
            )

