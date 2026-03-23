"""通用工具函数"""

from onescience.datapipes.biology.common.utils.file_utils import (
    is_compressed,
    get_base_extension,
    open_file,
    detect_file_format,
    get_all_fasta_files,
    get_all_msa_files,
    FASTA_EXTENSIONS,
    MSA_EXTENSIONS,
    STRUCTURE_EXTENSIONS,
    COMPRESSION_EXTENSIONS,
)

__all__ = [
    "is_compressed",
    "get_base_extension",
    "open_file",
    "detect_file_format",
    "get_all_fasta_files",
    "get_all_msa_files",
    "FASTA_EXTENSIONS",
    "MSA_EXTENSIONS",
    "STRUCTURE_EXTENSIONS",
    "COMPRESSION_EXTENSIONS",
]





