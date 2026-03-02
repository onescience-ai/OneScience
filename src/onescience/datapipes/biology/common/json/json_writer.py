"""JSON writer for protein structure prediction input files.

Supports generating JSON input files for models like Protenix/AlphaFold3.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class JSONWriteConfig:
    """JSON writing configuration.

    Attributes:
        indent: Indentation level for formatting.
        ensure_ascii: Whether to escape non-ASCII characters.
        sort_keys: Whether to sort dictionary keys.
        include_name: Whether to include the name field.
        include_covalent_bonds: Whether to include covalent bonds field.
    """
    indent: int = 2
    ensure_ascii: bool = False
    sort_keys: bool = False
    include_name: bool = True
    include_covalent_bonds: bool = True


class JSONWriter:
    """Base JSON writer class.

    Provides basic JSON file writing functionality.
    """

    def __init__(self, config: Optional[JSONWriteConfig] = None):
        """Initialize the writer.

        Args:
            config: Writing configuration. Uses default if not provided.
        """
        self.config = config or JSONWriteConfig()

    def write(self,
              data: Dict[str, Any],
              output_path: Union[str, Path],
              name: Optional[str] = None) -> Path:
        """Write data to a JSON file.

        Args:
            data: Data dictionary to write.
            output_path: Output file path.
            name: Optional name to include in the data.

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data
        output_data = dict(data)

        if self.config.include_name and name:
            output_data["name"] = name

        # Remove empty covalent_bonds if configured
        if not self.config.include_covalent_bonds and "covalent_bonds" in output_data:
            del output_data["covalent_bonds"]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f,
                     indent=self.config.indent,
                     ensure_ascii=self.config.ensure_ascii,
                     sort_keys=self.config.sort_keys)

        logger.info(f"Written JSON file: {output_path}")
        return output_path

    def write_string(self, data: Dict[str, Any], name: Optional[str] = None) -> str:
        """Write data to a JSON string.

        Args:
            data: Data dictionary to write.
            name: Optional name to include in the data.

        Returns:
            JSON string.
        """
        output_data = dict(data)

        if self.config.include_name and name:
            output_data["name"] = name

        return json.dumps(output_data,
                         indent=self.config.indent,
                         ensure_ascii=self.config.ensure_ascii,
                         sort_keys=self.config.sort_keys)


class ProteinJSONWriter(JSONWriter):
    """Protein structure prediction JSON writer.

    Provides convenient methods for creating protein structure prediction input files.
    """

    def __init__(self, config: Optional[JSONWriteConfig] = None):
        """Initialize the protein JSON writer.

        Args:
            config: Writing configuration. Uses default if not provided.
        """
        super().__init__(config)

    @staticmethod
    def create_protein_entry(sequence: str,
                            count: int = 1,
                            modifications: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Create a protein chain entry.

        Args:
            sequence: Amino acid sequence.
            count: Number of copies.
            modifications: List of post-translational modifications.

        Returns:
            Protein chain entry dictionary.
        """
        entry = {
            "proteinChain": {
                "sequence": sequence,
                "count": count
            }
        }

        if modifications:
            entry["proteinChain"]["modifications"] = modifications

        return entry

    @staticmethod
    def create_dna_entry(sequence: str,
                        count: int = 1,
                        modifications: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Create a DNA sequence entry.

        Args:
            sequence: DNA sequence.
            count: Number of copies.
            modifications: List of base modifications.

        Returns:
            DNA sequence entry dictionary.
        """
        entry = {
            "dnaSequence": {
                "sequence": sequence,
                "count": count
            }
        }

        if modifications:
            entry["dnaSequence"]["modifications"] = modifications

        return entry

    @staticmethod
    def create_rna_entry(sequence: str,
                        count: int = 1,
                        modifications: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Create an RNA sequence entry.

        Args:
            sequence: RNA sequence.
            count: Number of copies.
            modifications: List of base modifications.

        Returns:
            RNA sequence entry dictionary.
        """
        entry = {
            "rnaSequence": {
                "sequence": sequence,
                "count": count
            }
        }

        if modifications:
            entry["rnaSequence"]["modifications"] = modifications

        return entry

    @staticmethod
    def create_ligand_entry(ligand: str, count: int = 1) -> Dict[str, Any]:
        """Create a ligand entry.

        Args:
            ligand: Ligand CCD code (e.g., "ATP", "HEM").
            count: Number of copies.

        Returns:
            Ligand entry dictionary.
        """
        return {
            "ligand": {
                "ligand": ligand,
                "count": count
            }
        }

    @staticmethod
    def create_ion_entry(ion: str, count: int = 1) -> Dict[str, Any]:
        """Create an ion entry.

        Args:
            ion: Ion name (e.g., "NA", "CL", "MG").
            count: Number of copies.

        Returns:
            Ion entry dictionary.
        """
        return {
            "ion": {
                "ion": ion,
                "count": count
            }
        }

    @staticmethod
    def create_modification(position: int,
                           modification_type: str,
                           entity_type: str = "protein") -> Dict[str, Any]:
        """Create a modification entry.

        Args:
            position: Modification position (1-based).
            modification_type: Modification type (e.g., "CCD_PHY" for phosphorylation).
            entity_type: Entity type ("protein", "dna", or "rna").

        Returns:
            Modification entry dictionary.
        """
        if entity_type == "protein":
            return {
                "ptmPosition": position,
                "ptmType": modification_type
            }
        else:
            return {
                "basePosition": position,
                "modificationType": modification_type
            }

    @staticmethod
    def create_covalent_bond(entity1: int, position1: int, atom1: str,
                            entity2: int, position2: int, atom2: str,
                            copy1: Optional[int] = None,
                            copy2: Optional[int] = None) -> Dict[str, Any]:
        """Create a covalent bond entry.

        Args:
            entity1: First entity ID (1-based).
            position1: First position (1-based).
            atom1: First atom name.
            entity2: Second entity ID (1-based).
            position2: Second position (1-based).
            atom2: Second atom name.
            copy1: Optional copy ID for first entity.
            copy2: Optional copy ID for second entity.

        Returns:
            Covalent bond entry dictionary.
        """
        bond = {
            "entity1": entity1,
            "position1": position1,
            "atom1": atom1,
            "entity2": entity2,
            "position2": position2,
            "atom2": atom2
        }

        if copy1 is not None:
            bond["copy1"] = copy1
        if copy2 is not None:
            bond["copy2"] = copy2

        return bond

    def write_structure(self,
                       sequences: List[Dict[str, Any]],
                       output_path: Union[str, Path],
                       name: str,
                       covalent_bonds: Optional[List[Dict[str, Any]]] = None) -> Path:
        """Write a structure definition file.

        Args:
            sequences: List of sequence entries.
            output_path: Output file path.
            name: Structure name.
            covalent_bonds: Optional list of covalent bonds.

        Returns:
            Path to the written file.
        """
        data = {
            "sequences": sequences
        }

        if covalent_bonds:
            data["covalent_bonds"] = covalent_bonds

        return self.write(data, output_path, name=name)

    def write_protein_structure(self,
                               sequence: str,
                               output_path: Union[str, Path],
                               name: str,
                               count: int = 1,
                               modifications: Optional[List[Dict[str, Any]]] = None) -> Path:
        """Write a single protein structure file.

        Args:
            sequence: Amino acid sequence.
            output_path: Output file path.
            name: Structure name.
            count: Number of copies.
            modifications: Optional list of modifications.

        Returns:
            Path to the written file.
        """
        sequences = [
            self.create_protein_entry(sequence, count, modifications)
        ]

        return self.write_structure(sequences, output_path, name)

    def write_complex(self,
                     protein_sequences: List[str],
                     output_path: Union[str, Path],
                     name: str,
                     ligands: Optional[List[str]] = None,
                     ions: Optional[List[str]] = None) -> Path:
        """Write a protein complex structure file.

        Args:
            protein_sequences: List of protein sequences.
            output_path: Output file path.
            name: Complex name.
            ligands: Optional list of ligand CCD codes.
            ions: Optional list of ion names.

        Returns:
            Path to the written file.
        """
        sequences = []

        for seq in protein_sequences:
            sequences.append(self.create_protein_entry(seq))

        if ligands:
            for ligand in ligands:
                sequences.append(self.create_ligand_entry(ligand))

        if ions:
            for ion in ions:
                sequences.append(self.create_ion_entry(ion))

        return self.write_structure(sequences, output_path, name)
