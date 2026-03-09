"""Unified genome dataset base class.

Provides basic data processing capabilities. Users can inherit and implement
their own adaptation logic. Does not force dependency on adapter; users can
choose to use adapter or implement their own.
"""

from typing import Any, Dict, List, Union
from pathlib import Path

from onescience.datapipes.biology import BioDataset
from onescience.datapipes.core.config import DatasetConfig
from onescience.datapipes.biology.common.sequence.fasta_parser import FASTAParser
from onescience.datapipes.biology.common.sequence.sequence_encoder import (
    NucleotideEncoder,
)
from onescience.datapipes.biology.common.utils.file_utils import (
    detect_file_format,
    get_all_fasta_files,
)
from onescience.datapipes.biology.datasets.unified_dataset import UnifiedDataPipeline


class GenomeDataset(BioDataset):
    """Unified genome dataset base class.

    Provides basic data processing capabilities. Users can:
    1. Directly inherit and implement their own `__getitem__` method
    2. Use `UnifiedDataPipeline` for data processing
    3. Optionally use adapter (via `use_adapter=True`)

    Examples:
        Method 1: Direct inheritance, implement your own adaptation logic::

            class MyGenomeDataset(GenomeDataset):
                def __getitem__(self, idx):
                    sample = self.data_list[idx]
                    # Use pipeline for processing
                    features = self.pipeline.process_sample(
                        sequence=sample.get("sequence")
                    )
                    # Implement your own adaptation logic
                    return self._adapt_to_my_model(features)

        Method 2: Use optional adapter::

            config = {
                "source": {"path": "/data/genomes"},
                "data": {
                    "extra": {
                        "model_name": "evo2",
                        "sequence_type": "DNA",
                        "use_adapter": True  # Enable adapter
                    }
                }
            }
            dataset = GenomeDataset(config)
    """

    def __init__(self, config: Union[DatasetConfig, Dict[str, Any]]):
        super().__init__(config)

        # Initialize common processing modules (basic components)
        self.fasta_parser = FASTAParser()
        sequence_type = self.config.data.extra.get("sequence_type", "DNA")
        self.nt_encoder = NucleotideEncoder(sequence_type=sequence_type)

        # Initialize unified data pipeline (optional, users can choose to use)
        use_pipeline = self.config.data.extra.get('use_pipeline', True)
        if use_pipeline:
            self.pipeline = UnifiedDataPipeline(
                use_msa=self.config.data.extra.get('use_msa', False),
                use_structure=False,  # Genome data typically doesn't need structure
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
                model_name = self.config.data.extra.get("model_name", "evo2")
                self.adapter = get_adapter(model_name, self.config)
                self.logger.info(f"Using adapter: {model_name}")
            except ImportError:
                self.logger.warning("Adapter module not available, continuing without adapter")

    def _init_data(self):
        """Initialize data."""
        self.data_list = self._load_data_list()
        self.logger.info(f"Loaded {len(self.data_list)} genome sequences")

    def _load_data_list(self) -> List[Dict[str, Any]]:
        """Load genome data index.

        Subclasses can override this method to implement custom data loading logic.

        Returns:
            List[Dict[str, Any]]: List of data indices.
        """
        data_list = []
        data_path = Path(self.data_path)

        if data_path.is_file():
            file_format = detect_file_format(data_path)
            if file_format == 'fasta':
                sequences, descriptions = self.fasta_parser.parse_file(data_path)
                for seq, desc in zip(sequences, descriptions):
                    data_list.append({
                        "sequence": seq,
                        "description": desc,
                    })
        elif data_path.is_dir():
            # Use utility function to get all FASTA files (including compressed)
            fasta_files = get_all_fasta_files(data_path, recursive=False)
            for fasta_file in fasta_files:
                sequences, descriptions = self.fasta_parser.parse_file(fasta_file)
                for seq, desc in zip(sequences, descriptions):
                    data_list.append({
                        "sequence": seq,
                        "description": desc,
                        "file_path": str(fasta_file),
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
                fasta_path=Path(sample["file_path"]) if "file_path" in sample else None,
            )
            return features

        return sample

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_list)
