"""Unified protein dataset base class.

Provides basic data processing capabilities. Users can inherit and implement
their own adaptation logic. Does not force dependency on adapter; users can
choose to use adapter or implement their own.
"""

from typing import Any, Dict, List, Union, Optional
from pathlib import Path

from onescience.datapipes.biology import BioDataset
from onescience.datapipes.core.config import DatasetConfig
from onescience.datapipes.biology.common.utils.file_utils import (
    get_all_mmcif_files,
    get_all_pdb_files,
    get_all_fasta_files,
)
from onescience.datapipes.biology.datasets.unified_dataset import UnifiedDataPipeline


class ProteinDataset(BioDataset):
    """Unified protein dataset base class.

    Provides basic data processing capabilities. Users can:
    1. Directly inherit and implement their own `__getitem__` method
    2. Use `UnifiedDataPipeline` for data processing
    3. Optionally use adapter (via `use_adapter=True`)

    Examples:
        Method 1: Direct inheritance, implement your own adaptation logic::

            class MyProteinDataset(ProteinDataset):
                def __getitem__(self, idx):
                    sample = self.data_list[idx]
                    # Use pipeline for processing
                    features = self.pipeline.process_sample(
                        sequence=sample.get("sequence"),
                        mmcif_path=sample.get("mmcif_path")
                    )
                    # Implement your own adaptation logic
                    return self._adapt_to_my_model(features)

        Method 2: Use optional adapter::

            config = {
                "source": {"path": "/data/proteins"},
                "data": {
                    "extra": {
                        "model_name": "alphafold2",
                        "use_adapter": True  # Enable adapter
                    }
                }
            }
            dataset = ProteinDataset(config)
    """

    def __init__(self, config: Union[DatasetConfig, Dict[str, Any]]):
        super().__init__(config)

        # Initialize unified data pipeline (optional, users can choose to use)
        use_pipeline = self.config.data.extra.get('use_pipeline', True)
        if use_pipeline:
            self.pipeline = UnifiedDataPipeline(
                use_msa=self.config.data.extra.get('use_msa', True),
                use_structure=self.config.data.extra.get('use_structure', True),
                max_msa_seqs=self.config.data.extra.get('max_msa_seqs'),
            )
        else:
            self.pipeline = None

        # Optional adapter support (not forced)
        self.adapter = None
        use_adapter = self.config.data.extra.get('use_adapter', False)
        if use_adapter:
            try:
                from onescience.datapipes.biology.adapters import get_adapter
                model_name = self.config.data.extra.get("model_name", "alphafold2")
                self.adapter = get_adapter(model_name, self.config)
                self.logger.info(f"Using adapter: {model_name}")
            except ImportError:
                self.logger.warning("Adapter module not available, continuing without adapter")

    def _init_data(self):
        """Initialize data."""
        self.data_list = self._load_data_list()
        self.logger.info(f"Loaded {len(self.data_list)} protein structures")

    def _load_data_list(self) -> List[Dict[str, Any]]:
        """Load protein data index.

        Subclasses can override this method to implement custom data loading logic.

        Returns:
            List[Dict[str, Any]]: List of data indices.
        """
        data_list = []
        data_path = Path(self.data_path)

        if data_path.is_file():
            # Single file mode
            suffix = data_path.suffix.lower()
            if suffix in ['.cif', '.mmcif']:
                data_list.append({
                    "mmcif_path": str(data_path),
                    "name": data_path.stem,
                })
            elif suffix == '.pdb':
                data_list.append({
                    "pdb_path": str(data_path),
                    "name": data_path.stem,
                })
            elif suffix in ['.fasta', '.fa', '.faa']:
                data_list.append({
                    "fasta_path": str(data_path),
                    "name": data_path.stem,
                })
        elif data_path.is_dir():
            # Directory mode - scan for all supported files
            mmcif_files = get_all_mmcif_files(data_path, recursive=False)
            for mmcif_file in mmcif_files:
                data_list.append({
                    "mmcif_path": str(mmcif_file),
                    "name": mmcif_file.stem,
                })

            pdb_files = get_all_pdb_files(data_path, recursive=False)
            for pdb_file in pdb_files:
                data_list.append({
                    "pdb_path": str(pdb_file),
                    "name": pdb_file.stem,
                })

            fasta_files = get_all_fasta_files(data_path, recursive=False)
            for fasta_file in fasta_files:
                data_list.append({
                    "fasta_path": str(fasta_file),
                    "name": fasta_file.stem,
                })

        return data_list

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample.

        Default implementation: if adapter is enabled, use adapter for processing;
        otherwise use pipeline. Subclasses should override this method to implement
their own adaptation logic.

        Args:
            idx: Sample index.

        Returns:
            Dict[str, Any]: Feature dictionary.
        """
        sample = self.data_list[idx]

        # If using adapter, process through adapter
        if self.adapter is not None:
            features = self.adapter.process_sample(sample)
            return features

        # Otherwise use pipeline for processing
        if self.pipeline is not None:
            features = self.pipeline.process_sample(
                sequence=sample.get("sequence"),
                mmcif_path=Path(sample["mmcif_path"]) if "mmcif_path" in sample else None,
                pdb_path=Path(sample["pdb_path"]) if "pdb_path" in sample else None,
                fasta_path=Path(sample["fasta_path"]) if "fasta_path" in sample else None,
            )
            return features

        return sample

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_list)
