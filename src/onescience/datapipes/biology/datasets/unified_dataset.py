"""Unified biological data processing pipeline.

Provides standardized processing workflows for various biological data types
including sequences, structures, and MSAs.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from onescience.datapipes.biology.common.sequence.fasta_parser import FASTAParser
from onescience.datapipes.biology.common.sequence.sequence_encoder import (
    AminoAcidEncoder,
    NucleotideEncoder,
)
from onescience.datapipes.biology.common.structure.mmcif_parser import MMCIFParser
from onescience.datapipes.biology.common.structure.pdb_parser import PDBParser
from onescience.datapipes.biology.common.msa.msa_processor import MSAProcessor
from onescience.datapipes.biology.common.utils.file_utils import detect_file_format


class UnifiedDataPipeline:
    """Unified biological data processing pipeline.

    Integrates sequence, structure, and MSA processing capabilities,
    providing standardized data processing workflows.

    Args:
        use_msa: Whether to use MSA features.
        use_structure: Whether to use structure features.
        max_msa_seqs: Maximum number of MSA sequences.
        sequence_type: Sequence type ("protein" or "DNA").

    Examples:
        Basic usage::

            pipeline = UnifiedDataPipeline(
                use_msa=True,
                use_structure=True,
                max_msa_seqs=512
            )
            features = pipeline.process_sample(
                sequence="MKTLL...",
                mmcif_path=Path("/path/to/structure.cif")
            )
    """

    def __init__(
        self,
        use_msa: bool = False,
        use_structure: bool = True,
        max_msa_seqs: Optional[int] = None,
        sequence_type: str = "protein",
    ):
        self.use_msa = use_msa
        self.use_structure = use_structure
        self.max_msa_seqs = max_msa_seqs
        self.sequence_type = sequence_type

        # Initialize parsers and encoders
        self.fasta_parser = FASTAParser()
        self.mmcif_parser = MMCIFParser()
        self.pdb_parser = PDBParser()

        if sequence_type == "protein":
            self.sequence_encoder = AminoAcidEncoder()
        else:
            self.sequence_encoder = NucleotideEncoder(sequence_type=sequence_type)

        # Initialize MSA processor
        if use_msa:
            self.msa_processor = MSAProcessor(max_seqs=max_msa_seqs)

    def process_sample(
        self,
        sequence: Optional[str] = None,
        sequence_id: Optional[str] = None,
        mmcif_path: Optional[Path] = None,
        pdb_path: Optional[Path] = None,
        fasta_path: Optional[Path] = None,
        msa_path: Optional[Path] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a single sample.

        Main entry point for data processing. Automatically detects input type
        and processes accordingly.

        Args:
            sequence: Raw sequence string.
            sequence_id: Sequence identifier.
            mmcif_path: Path to mmCIF format structure file.
            pdb_path: Path to PDB format structure file.
            fasta_path: Path to FASTA format sequence file.
            msa_path: Path to MSA file.
            **kwargs: Additional parameters.

        Returns:
            Dict[str, Any]: Processed feature dictionary.
        """
        features = {
            "sequence_id": sequence_id,
        }

        # Process sequence
        if sequence is not None:
            features["sequence"] = sequence
            features["sequence_encoded"] = self.sequence_encoder.encode(sequence)
        elif fasta_path is not None:
            seq_features = self._process_fasta(fasta_path)
            features.update(seq_features)

        # Process structure
        if mmcif_path is not None:
            struct_features = self._process_mmcif(mmcif_path)
            features.update(struct_features)
        elif pdb_path is not None:
            struct_features = self._process_pdb(pdb_path)
            features.update(struct_features)

        # Process MSA
        if self.use_msa and msa_path is not None:
            msa_features = self._process_msa(msa_path)
            features.update(msa_features)

        return features

    def _process_fasta(self, fasta_path: Path) -> Dict[str, Any]:
        """Process FASTA file.

        Args:
            fasta_path: Path to FASTA file.

        Returns:
            Dict[str, Any]: Sequence features.
        """
        sequences, descriptions = self.fasta_parser.parse_file(fasta_path)

        if not sequences:
            return {}

        # Use first sequence
        sequence = sequences[0]
        description = descriptions[0] if descriptions else ""

        return {
            "sequence": sequence,
            "sequence_encoded": self.sequence_encoder.encode(sequence),
            "description": description,
        }

    def _process_mmcif(self, mmcif_path: Path) -> Dict[str, Any]:
        """Process mmCIF file.

        Args:
            mmcif_path: Path to mmCIF file.

        Returns:
            Dict[str, Any]: Structure features.
        """
        structure = self.mmcif_parser.parse_file(mmcif_path)

        return {
            "structure": structure,
            "mmcif_path": str(mmcif_path),
        }

    def _process_pdb(self, pdb_path: Path) -> Dict[str, Any]:
        """Process PDB file.

        Args:
            pdb_path: Path to PDB file.

        Returns:
            Dict[str, Any]: Structure features.
        """
        structure = self.pdb_parser.parse_file(pdb_path)

        return {
            "structure": structure,
            "pdb_path": str(pdb_path),
        }

    def _process_msa(self, msa_path: Path) -> Dict[str, Any]:
        """Process MSA file.

        Args:
            msa_path: Path to MSA file.

        Returns:
            Dict[str, Any]: MSA features.
        """
        msa_features = self.msa_processor.process(msa_path)

        return {
            "msa": msa_features,
            "msa_path": str(msa_path),
        }

    def batch_process(
        self,
        samples: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Batch process samples.

        Args:
            samples: List of sample dictionaries.
            batch_size: Batch size.

        Returns:
            List[Dict[str, Any]]: List of processed feature dictionaries.
        """
        results = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            batch_results = [self.process_sample(**sample) for sample in batch]
            results.extend(batch_results)
        return results
