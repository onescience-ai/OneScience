"""JSON parser for protein structure prediction input files.

Supports parsing JSON input files for models like Protenix/AlphaFold3.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from biotite.structure import AtomArray, get_chain_starts

logger = logging.getLogger(__name__)


@dataclass
class JSONData:
    """JSON data container class.

    Attributes:
        name: Data name identifier.
        data: Raw JSON data dictionary.
        source_file: Source file path (if parsed from file).
    """
    name: str
    data: Dict[str, Any]
    source_file: Optional[Path] = None

    def get_sequences(self) -> List[Dict[str, Any]]:
        """Get all sequence information.

        Returns:
            List of sequence dictionaries.
        """
        return self.data.get("sequences", [])

    def get_entities(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get all entity information.

        Returns:
            List of tuples containing (entity_type, entity_info).
        """
        entities = []
        for entity_dict in self.get_sequences():
            for entity_type, entity_info in entity_dict.items():
                entities.append((entity_type, entity_info))
        return entities

    def get_covalent_bonds(self) -> List[Dict[str, Any]]:
        """Get all covalent bond information.

        Returns:
            List of covalent bond dictionaries.
        """
        return self.data.get("covalent_bonds", [])

    def get_entity_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Get entities by type.

        Args:
            entity_type: Entity type to filter by.

        Returns:
            List of entity info dictionaries matching the type.
        """
        results = []
        for entity_dict in self.get_sequences():
            if entity_type in entity_dict:
                results.append(entity_dict[entity_type])
        return results


class JSONParser:
    """Base JSON parser class.

    Provides basic JSON file parsing functionality.
    """

    @staticmethod
    def parse_file(file_path: Union[str, Path]) -> JSONData:
        """Parse a JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            Parsed JSONData object.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        name = data.get("name", file_path.stem)

        return JSONData(name=name, data=data, source_file=file_path)

    @staticmethod
    def parse_string(json_string: str, name: str = "unnamed") -> JSONData:
        """Parse a JSON string.

        Args:
            json_string: JSON content as string.
            name: Name identifier for the data.

        Returns:
            Parsed JSONData object.

        Raises:
            json.JSONDecodeError: If the string contains invalid JSON.
        """
        data = json.loads(json_string)
        return JSONData(name=name, data=data)

    @staticmethod
    def validate_structure(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate JSON data structure.

        Args:
            data: JSON data dictionary to validate.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []

        # Check required fields
        if "sequences" not in data:
            errors.append("Missing required field: 'sequences'")
            return False, errors

        sequences = data.get("sequences", [])
        if not isinstance(sequences, list):
            errors.append("'sequences' must be a list")
            return False, errors

        # Validate each entity
        for i, entity_dict in enumerate(sequences):
            if not isinstance(entity_dict, dict):
                errors.append(f"Entity {i} must be a dictionary")
                continue

            for entity_type, entity_info in entity_dict.items():
                # Check entity type
                valid_types = ["proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"]
                if entity_type not in valid_types:
                    errors.append(f"Entity {i}: unknown type '{entity_type}'")
                    continue

                # Check required fields for each type
                if entity_type in ["proteinChain", "dnaSequence", "rnaSequence"]:
                    if "sequence" not in entity_info:
                        errors.append(f"Entity {i} ({entity_type}): missing 'sequence' field")

                elif entity_type == "ligand":
                    if "ligand" not in entity_info:
                        errors.append(f"Entity {i} ({entity_type}): missing 'ligand' field")

                elif entity_type == "ion":
                    if "ion" not in entity_info:
                        errors.append(f"Entity {i} ({entity_type}): missing 'ion' field")

        # Validate covalent bonds if present
        if "covalent_bonds" in data:
            bonds = data["covalent_bonds"]
            if not isinstance(bonds, list):
                errors.append("'covalent_bonds' must be a list")
            else:
                for i, bond in enumerate(bonds):
                    required_bond_fields = ["entity1", "position1", "atom1",
                                           "entity2", "position2", "atom2"]
                    for field_name in required_bond_fields:
                        if field_name not in bond:
                            errors.append(f"Bond {i}: missing '{field_name}'")
                            break

        return len(errors) == 0, errors


class ProteinJSONParser(JSONParser):
    """Protein structure prediction JSON parser.

    Provides protein-specific parsing functionality.
    """

    @staticmethod
    def extract_sequence_info(json_data: JSONData) -> Dict[str, List[Dict[str, Any]]]:
        """Extract sequence information.

        Args:
            json_data: JSONData object to extract from.

        Returns:
            Dictionary mapping sequence types to lists of info dictionaries.
        """
        sequences = {
            "protein": [],
            "dna": [],
            "rna": [],
            "ligand": [],
            "ion": []
        }

        for entity_type, entity_info in json_data.get_entities():
            if entity_type == "proteinChain":
                sequences["protein"].append({
                    "sequence": entity_info.get("sequence", ""),
                    "count": entity_info.get("count", 1),
                    "modifications": entity_info.get("modifications", [])
                })
            elif entity_type == "dnaSequence":
                sequences["dna"].append({
                    "sequence": entity_info.get("sequence", ""),
                    "count": entity_info.get("count", 1),
                    "modifications": entity_info.get("modifications", [])
                })
            elif entity_type == "rnaSequence":
                sequences["rna"].append({
                    "sequence": entity_info.get("sequence", ""),
                    "count": entity_info.get("count", 1),
                    "modifications": entity_info.get("modifications", [])
                })
            elif entity_type == "ligand":
                sequences["ligand"].append({
                    "ligand": entity_info.get("ligand", ""),
                    "count": entity_info.get("count", 1)
                })
            elif entity_type == "ion":
                sequences["ion"].append({
                    "ion": entity_info.get("ion", ""),
                    "count": entity_info.get("count", 1)
                })

        return sequences

    @staticmethod
    def get_modifications(json_data: JSONData) -> List[Dict[str, Any]]:
        """Get all modification information.

        Args:
            json_data: JSONData object to extract from.

        Returns:
            List of modification dictionaries.
        """
        modifications = []

        for entity_type, entity_info in json_data.get_entities():
            if "modifications" in entity_info:
                entity_mods = entity_info["modifications"]
                for mod in entity_mods:
                    mod_info = {
                        "entity_type": entity_type,
                        "modification": mod
                    }
                    modifications.append(mod_info)

        return modifications

    @staticmethod
    def calculate_statistics(json_data: JSONData) -> Dict[str, Any]:
        """Calculate sequence statistics.

        Args:
            json_data: JSONData object to analyze.

        Returns:
            Dictionary containing statistics.
        """
        seq_info = ProteinJSONParser.extract_sequence_info(json_data)

        stats = {
            "total_entities": 0,
            "total_chains": 0,
            "protein_chains": 0,
            "dna_chains": 0,
            "rna_chains": 0,
            "ligands": 0,
            "ions": 0,
            "total_residues": 0,
            "modifications": 0
        }

        for protein in seq_info["protein"]:
            count = protein["count"]
            seq_len = len(protein["sequence"])
            stats["protein_chains"] += count
            stats["total_chains"] += count
            stats["total_residues"] += seq_len * count
            stats["total_entities"] += 1
            stats["modifications"] += len(protein.get("modifications", []))

        for dna in seq_info["dna"]:
            count = dna["count"]
            seq_len = len(dna["sequence"])
            stats["dna_chains"] += count
            stats["total_chains"] += count
            stats["total_residues"] += seq_len * count
            stats["total_entities"] += 1
            stats["modifications"] += len(dna.get("modifications", []))

        for rna in seq_info["rna"]:
            count = rna["count"]
            seq_len = len(rna["sequence"])
            stats["rna_chains"] += count
            stats["total_chains"] += count
            stats["total_residues"] += seq_len * count
            stats["total_entities"] += 1
            stats["modifications"] += len(rna.get("modifications", []))

        stats["ligands"] = sum(ligand["count"] for ligand in seq_info["ligand"])
        stats["ions"] = sum(ion["count"] for ion in seq_info["ion"])
        stats["total_entities"] += len(seq_info["ligand"]) + len(seq_info["ion"])
        stats["total_chains"] += stats["ligands"] + stats["ions"]

        return stats
