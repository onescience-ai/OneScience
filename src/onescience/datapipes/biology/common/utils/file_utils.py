"""File utility functions for biology datapipes.

This module provides functions for file format detection, compressed file handling,
and working with biological data files (FASTA, MSA, structure files).
"""

from pathlib import Path
from typing import Optional, List
import gzip
import bz2
import lzma


# File format extension definitions
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
    """Check if a file is compressed.

    Args:
        path: Path to the file.

    Returns:
        True if the file is compressed, False otherwise.
    """
    return path.suffix.lower() in COMPRESSION_EXTENSIONS


def get_base_extension(path: Path) -> str:
    """Get the base extension without compression suffix.

    Args:
        path: Path to the file.

    Returns:
        The base extension (without compression suffix).

    Examples:
        >>> get_base_extension(Path("data.fasta.gz"))
        '.fasta'
        >>> get_base_extension(Path("data.fasta"))
        '.fasta'
    """
    if is_compressed(path):
        # Get the path without compression suffix
        stem_path = Path(path.stem)
        return stem_path.suffix
    return path.suffix


def open_file(path: Path, mode: str = 'r', encoding: Optional[str] = None):
    """Open a file with automatic compression detection.

    Supports gzip, bz2, and xz compressed files.

    Args:
        path: Path to the file.
        mode: File open mode ('r', 'w', 'rb', 'wb', etc.).
        encoding: Text encoding (for text mode only).

    Returns:
        A file object.

    Examples:
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
    """Detect the format of a biological data file.

    Detection is based on file extension and content analysis.

    Args:
        path: Path to the file.
        sample_size: Number of bytes to read for content-based detection.

    Returns:
        The detected format name, or None if detection fails.

    Examples:
        >>> detect_file_format(Path("data.fasta"))
        'fasta'
        >>> detect_file_format(Path("data.fasta.gz"))
        'fasta'
    """
    # First, try detection based on extension
    base_ext = get_base_extension(path).lower()

    # Check FASTA format
    if base_ext in FASTA_EXTENSIONS:
        return 'fasta'

    # Check MSA formats
    for format_name, extensions in MSA_EXTENSIONS.items():
        if base_ext in extensions:
            return format_name

    # Check structure formats
    for format_name, extensions in STRUCTURE_EXTENSIONS.items():
        if base_ext in extensions:
            return format_name

    # If extension detection fails, try content-based detection
    try:
        with open_file(path, 'r') as f:
            sample = f.read(sample_size)

        # Detect FASTA (starts with >)
        if sample.strip().startswith('>'):
            return 'fasta'

        # Detect a3m (contains > and lowercase letters)
        if '>' in sample and any(c.islower() for c in sample):
            return 'a3m'

        # Detect Stockholm format
        if '# STOCKHOLM' in sample:
            return 'stockholm'

        # Detect Clustal format
        if 'CLUSTAL' in sample.upper():
            return 'clustal'

    except Exception:
        # Return None if reading fails
        pass

    return None


def get_all_fasta_files(directory: Path, recursive: bool = False) -> List[Path]:
    """Get all FASTA files in a directory (including compressed files).

    Args:
        directory: Path to the directory.
        recursive: Whether to search recursively.

    Returns:
        A list of paths to FASTA files.
    """
    files = []

    # Search all possible FASTA extensions
    for ext in FASTA_EXTENSIONS:
        pattern = f"**/*{ext}" if recursive else f"*{ext}"
        files.extend(directory.glob(pattern))

        # Also search compressed versions
        for comp_ext in COMPRESSION_EXTENSIONS:
            pattern = f"**/*{ext}{comp_ext}" if recursive else f"*{ext}{comp_ext}"
            files.extend(directory.glob(pattern))

    return sorted(set(files))  # Remove duplicates and sort


def get_all_msa_files(directory: Path, recursive: bool = False) -> List[Path]:
    """Get all MSA files in a directory (including compressed files).

    Args:
        directory: Path to the directory.
        recursive: Whether to search recursively.

    Returns:
        A list of paths to MSA files.
    """
    files = []

    # Search all MSA formats
    for format_name, extensions in MSA_EXTENSIONS.items():
        for ext in extensions:
            pattern = f"**/*{ext}" if recursive else f"*{ext}"
            files.extend(directory.glob(pattern))

            # Also search compressed versions
            for comp_ext in COMPRESSION_EXTENSIONS:
                pattern = f"**/*{ext}{comp_ext}" if recursive else f"*{ext}{comp_ext}"
                files.extend(directory.glob(pattern))

    return sorted(set(files))  # Remove duplicates and sort


def get_all_structure_files(
    directory: Path,
    recursive: bool = False,
    allowed_formats: Optional[List[str]] = None,
) -> List[Path]:
    """Get all structure files in a directory (including compressed files).

    Args:
        directory: Path to the directory.
        recursive: Whether to search recursively.
        allowed_formats: List of allowed structure format names (e.g., ['pdb', 'mmcif']),
            or None to include all formats.

    Returns:
        A list of paths to structure files.
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
