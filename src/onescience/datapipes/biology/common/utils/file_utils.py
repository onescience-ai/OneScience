"""
文件工具函数

处理文件格式检测、压缩文件等
"""

from pathlib import Path
from typing import Optional, List
import gzip
import bz2
import lzma


# 文件格式扩展名定义
FASTA_EXTENSIONS = ['.fasta', '.fa', '.fna', '.fas', '.seq', '.fsa']
MSA_EXTENSIONS = {
    'a3m': ['.a3m'],
    'stockholm': ['.sto', '.stk', '.stockholm'],
    'clustal': ['.aln', '.clustal', '.clustalw'],
    'phylip': ['.phy', '.phylip'],
}
STRUCTURE_EXTENSIONS = {
    'mmcif': ['.cif', '.mmcif'],
    'pdb': ['.pdb', '.ent'],
    'mmtf': ['.mmtf'],
}
COMPRESSION_EXTENSIONS = ['.gz', '.bz2', '.xz', '.z']


def is_compressed(path: Path) -> bool:
    """
    检查文件是否压缩
    
    Parameters
    ----------
    path : Path
        文件路径
        
    Returns
    -------
    bool
        是否为压缩文件
    """
    return path.suffix.lower() in COMPRESSION_EXTENSIONS


def get_base_extension(path: Path) -> str:
    """
    获取去除压缩后缀的基础扩展名
    
    Parameters
    ----------
    path : Path
        文件路径
        
    Returns
    -------
    str
        基础扩展名（不含压缩后缀）
        
    Examples
    --------
    >>> get_base_extension(Path("data.fasta.gz"))
    '.fasta'
    >>> get_base_extension(Path("data.fasta"))
    '.fasta'
    """
    if is_compressed(path):
        # 获取去除压缩后缀的路径
        stem_path = Path(path.stem)
        return stem_path.suffix
    return path.suffix


def open_file(path: Path, mode: str = 'r', encoding: Optional[str] = None):
    """
    自动检测并打开文件（支持压缩文件）
    
    Parameters
    ----------
    path : Path
        文件路径
    mode : str
        打开模式（'r', 'w', 'rb', 'wb'等）
    encoding : Optional[str]
        文本编码（仅文本模式）
        
    Returns
    -------
    file object
        打开的文件对象
        
    Examples
    --------
    >>> with open_file(Path("data.fasta.gz")) as f:
    ...     content = f.read()
    """
    path = Path(path)
    suffix = path.suffix.lower()
    
    if suffix == '.gz':
        if 'b' in mode:
            return gzip.open(path, mode)
        else:
            return gzip.open(path, mode, encoding=encoding or 'utf-8')
    elif suffix == '.bz2':
        if 'b' in mode:
            return bz2.open(path, mode)
        else:
            return bz2.open(path, mode, encoding=encoding or 'utf-8')
    elif suffix == '.xz':
        if 'b' in mode:
            return lzma.open(path, mode)
        else:
            return lzma.open(path, mode, encoding=encoding or 'utf-8')
    else:
        if 'b' in mode:
            return open(path, mode)
        else:
            return open(path, mode, encoding=encoding or 'utf-8')


def detect_file_format(path: Path, sample_size: int = 1024) -> Optional[str]:
    """
    检测文件格式（基于扩展名和内容）
    
    Parameters
    ----------
    path : Path
        文件路径
    sample_size : int
        用于内容检测的采样大小（字节）
        
    Returns
    -------
    Optional[str]
        检测到的格式名称，如果无法检测则返回None
        
    Examples
    --------
    >>> detect_file_format(Path("data.fasta"))
    'fasta'
    >>> detect_file_format(Path("data.fasta.gz"))
    'fasta'
    """
    # 首先基于扩展名检测
    base_ext = get_base_extension(path).lower()
    
    # 检测FASTA格式
    if base_ext in FASTA_EXTENSIONS:
        return 'fasta'
    
    # 检测MSA格式
    for format_name, extensions in MSA_EXTENSIONS.items():
        if base_ext in extensions:
            return format_name
    
    # 检测结构格式
    for format_name, extensions in STRUCTURE_EXTENSIONS.items():
        if base_ext in extensions:
            return format_name
    
    # 如果扩展名无法确定，尝试基于内容检测
    try:
        with open_file(path, 'r') as f:
            sample = f.read(sample_size)
        
        # 检测FASTA（以>开头）
        if sample.strip().startswith('>'):
            return 'fasta'
        
        # 检测a3m（包含>和小写字母）
        if '>' in sample and any(c.islower() for c in sample):
            return 'a3m'
        
        # 检测Stockholm
        if '# STOCKHOLM' in sample:
            return 'stockholm'
        
        # 检测Clustal
        if 'CLUSTAL' in sample.upper():
            return 'clustal'
        
    except Exception:
        # 如果读取失败，返回None
        pass
    
    return None


def get_all_fasta_files(directory: Path, recursive: bool = False) -> List[Path]:
    """
    获取目录中所有FASTA文件（包括压缩文件）
    
    Parameters
    ----------
    directory : Path
        目录路径
    recursive : bool
        是否递归搜索
        
    Returns
    -------
    List[Path]
        FASTA文件路径列表
    """
    files = []
    
    # 搜索所有可能的FASTA扩展名
    for ext in FASTA_EXTENSIONS:
        pattern = f"**/*{ext}" if recursive else f"*{ext}"
        files.extend(directory.glob(pattern))
        
        # 也搜索压缩版本
        for comp_ext in COMPRESSION_EXTENSIONS:
            pattern = f"**/*{ext}{comp_ext}" if recursive else f"*{ext}{comp_ext}"
            files.extend(directory.glob(pattern))
    
    return sorted(set(files))  # 去重并排序


def get_all_msa_files(directory: Path, recursive: bool = False) -> List[Path]:
    """
    获取目录中所有MSA文件（包括压缩文件）
    
    Parameters
    ----------
    directory : Path
        目录路径
    recursive : bool
        是否递归搜索
        
    Returns
    -------
    List[Path]
        MSA文件路径列表
    """
    files = []
    
    # 搜索所有MSA格式
    for format_name, extensions in MSA_EXTENSIONS.items():
        for ext in extensions:
            pattern = f"**/*{ext}" if recursive else f"*{ext}"
            files.extend(directory.glob(pattern))
            
            # 也搜索压缩版本
            for comp_ext in COMPRESSION_EXTENSIONS:
                pattern = f"**/*{ext}{comp_ext}" if recursive else f"*{ext}{comp_ext}"
                files.extend(directory.glob(pattern))
    
    return sorted(set(files))  # 去重并排序


def get_all_structure_files(
    directory: Path,
    recursive: bool = False,
    allowed_formats: Optional[List[str]] = None,
) -> List[Path]:
    """
    获取目录中所有结构文件（包括压缩文件）
    
    Parameters
    ----------
    directory : Path
        目录路径
    recursive : bool
        是否递归搜索
    allowed_formats : Optional[List[str]]
        允许的结构格式名称（如 ['pdb', 'mmcif']），None 表示全部
        
    Returns
    -------
    List[Path]
        结构文件路径列表
    """
    files: List[Path] = []
    directory = Path(directory)
    
    if allowed_formats is None:
        formats = STRUCTURE_EXTENSIONS.keys()
    else:
        formats = [
            fmt for fmt in allowed_formats
            if fmt in STRUCTURE_EXTENSIONS
        ]
    
    for fmt in formats:
        extensions = STRUCTURE_EXTENSIONS[fmt]
        for ext in extensions:
            pattern = f"**/*{ext}" if recursive else f"*{ext}"
            files.extend(directory.glob(pattern))
            
            for comp_ext in COMPRESSION_EXTENSIONS:
                pattern = f"**/*{ext}{comp_ext}" if recursive else f"*{ext}{comp_ext}"
                files.extend(directory.glob(pattern))
    
    return sorted(set(files))





