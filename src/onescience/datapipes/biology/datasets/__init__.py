"""统一的数据集实现"""

from onescience.datapipes.biology.datasets.protein_dataset import ProteinDataset
from onescience.datapipes.biology.datasets.genome_dataset import GenomeDataset
from onescience.datapipes.biology.datasets.multimer_dataset import MultimerDataset

__all__ = [
    "ProteinDataset",
    "GenomeDataset",
    "MultimerDataset",
]

