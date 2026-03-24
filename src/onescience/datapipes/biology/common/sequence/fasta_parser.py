"""
统一的FASTA解析器

从protenix和openfold中提取的共同功能
支持压缩文件（.gz, .bz2, .xz）
"""

from typing import List, Tuple, Iterator
from pathlib import Path
import re

from onescience.datapipes.biology.common.utils.file_utils import (
    open_file,
    is_compressed,
    get_base_extension,
)


class FASTAParser:
    """
    统一的FASTA解析器
    
    支持：
    - FASTA字符串解析
    - FASTA文件解析
    - 多序列FASTA解析
    """
    
    @staticmethod
    def parse(fasta_string: str) -> Tuple[List[str], List[str]]:
        """
        解析FASTA字符串
        
        Parameters
        ----------
        fasta_string : str
            FASTA格式的字符串
            
        Returns
        -------
        Tuple[List[str], List[str]]
            序列列表和描述列表
        """
        sequences = []
        descriptions = []
        index = -1
        
        for line in fasta_string.splitlines():
            line = line.strip()
            if line.startswith(">"):
                index += 1
                descriptions.append(line[1:])  # 移除 '>'
                sequences.append("")
            elif line.startswith("#"):
                continue  # 跳过注释行
            elif not line:
                continue  # 跳过空行
            else:
                sequences[index] += line
        
        return sequences, descriptions
    
    @staticmethod
    def parse_file(path: Path) -> Tuple[List[str], List[str]]:
        """
        从文件解析FASTA（支持压缩文件）
        
        Parameters
        ----------
        path : Path
            FASTA文件路径（支持.gz, .bz2, .xz压缩文件）
            
        Returns
        -------
        Tuple[List[str], List[str]]
            序列列表和描述列表
        """
        with open_file(path, 'r') as f:
            fasta_string = f.read()
        return FASTAParser.parse(fasta_string)
    
    @staticmethod
    def parse_file_stream(path: Path) -> Iterator[Tuple[str, str]]:
        """
        流式解析FASTA文件（支持大文件和压缩文件）
        
        Parameters
        ----------
        path : Path
            FASTA文件路径（支持压缩文件）
            
        Yields
        ------
        Tuple[str, str]
            (序列, 描述) 元组
        """
        current_seq = ""
        current_desc = ""
        
        with open_file(path, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith(">"):
                    # 保存上一个序列
                    if current_seq:
                        yield current_seq, current_desc
                    
                    # 开始新序列
                    current_desc = line[1:]  # 移除 '>'
                    current_seq = ""
                elif line.startswith("#"):
                    continue  # 跳过注释行
                elif line:
                    current_seq += line
        
        # 保存最后一个序列
        if current_seq:
            yield current_seq, current_desc
    
    @staticmethod
    def parse_single(fasta_string: str) -> Tuple[str, str]:
        """
        解析单个序列的FASTA
        
        Parameters
        ----------
        fasta_string : str
            FASTA格式的字符串（只包含一个序列）
            
        Returns
        -------
        Tuple[str, str]
            序列和描述
        """
        sequences, descriptions = FASTAParser.parse(fasta_string)
        if len(sequences) == 0:
            raise ValueError("No sequence found in FASTA string")
        if len(sequences) > 1:
            raise ValueError("Multiple sequences found, use parse() instead")
        return sequences[0], descriptions[0]

