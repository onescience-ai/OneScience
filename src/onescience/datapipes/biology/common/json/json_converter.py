"""Unified JSON converter.

Supports conversion between different JSON formats.
"""

import copy
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from biotite.structure import AtomArray, get_chain_starts

from onescience.datapipes.biology.common.json.json_parser import JSONData, JSONParser
from onescience.datapipes.biology.common.json.json_writer import ProteinJSONWriter

logger = logging.getLogger(__name__)


class JSONConverter:
    """Unified JSON converter.

    Supported features:
    - JSON format validation and repair
    - Conversion between different version formats
    - Merging multiple JSON files
    - Splitting JSON files
    """

    @staticmethod
    def normalize(json_data: Union[Dict[str, Any], JSONData]) -> Dict[str, Any]:
        """Normalize JSON data format.

        Args:
            json_data: Input data, either a dictionary or JSONData object.

        Returns:
            Normalized data dictionary.
        """
        if isinstance(json_data, JSONData):
            data = copy.deepcopy(json_data.data)
        else:
            data = copy.deepcopy(json_data)

        # Ensure name field exists
        if "name" not in data:
            data["name"] = ""

        # Ensure sequences field exists
        if "sequences" not in data:
            data["sequences"] = []

        # Ensure covalent_bonds is a list
        if "covalent_bonds" in data and data["covalent_bonds"] is None:
            data["covalent_bonds"] = []

        return data

    @staticmethod
    def merge(json_data_list: List[Union[Dict[str, Any], JSONData]],
             merge_name: str = "merged") -> Dict[str, Any]:
        """Merge multiple JSON data objects.

        Args:
            json_data_list: List of JSON data objects to merge.
            merge_name: Name for the merged result.

        Returns:
            Merged data dictionary.
        """
        all_sequences = []
        all_bonds = []

        for json_data in json_data_list:
            if isinstance(json_data, JSONData):
                data = json_data.data
            else:
                data = json_data

            # Merge sequences
            if "sequences" in data:
                all_sequences.extend(data["sequences"])

            # Merge covalent bonds
            if "covalent_bonds" in data:
                all_bonds.extend(data["covalent_bonds"])

        merged = {
            "name": merge_name,
            "sequences": all_sequences
        }

        if all_bonds:
            merged["covalent_bonds"] = all_bonds

        return merged

    @staticmethod
    def split_by_entity(json_data: Union[Dict[str, Any], JSONData]) -> List[Dict[str, Any]]:
        """Split JSON data by entity.

        Args:
            json_data: Input data, either a dictionary or JSONData object.

        Returns:
            List of split data dictionaries, each containing one entity.
        """
        if isinstance(json_data, JSONData):
            base_name = json_data.name or "entity"
            data = json_data.data
        else:
            base_name = json_data.get("name", "entity")
            data = json_data

        sequences = data.get("sequences", [])
        result = []

        for i, entity_dict in enumerate(sequences):
            for entity_type, entity_info in entity_dict.items():
                name = f"{base_name}_{entity_type}_{i+1}"
                split_data = {
                    "name": name,
                    "sequences": [{entity_type: entity_info}]
                }
                result.append(split_data)

        return result

    @staticmethod
    def to_protenix_format(json_data: Union[Dict[str, Any], JSONData]) -> Dict[str, Any]:
        """Convert data to Protenix format.

        Args:
            json_data: Input data, either a dictionary or JSONData object.

        Returns:
            Data in Protenix format.
        """
        data = JSONConverter.normalize(json_data)

        # Protenix format requires count field for each entity in sequences
        for entity_dict in data.get("sequences", []):
            for entity_type, entity_info in entity_dict.items():
                if "count" not in entity_info:
                    entity_info["count"] = 1

        return data

    @staticmethod
    def from_protenix_format(json_data: Union[Dict[str, Any], JSONData]) -> Dict[str, Any]:
        """Convert data from Protenix format (currently already in Protenix format, for future extension).

        Args:
            json_data: Input data, either a dictionary or JSONData object.

        Returns:
            Converted data dictionary.
        """
        return JSONConverter.normalize(json_data)


class ProteinJSONConverter(JSONConverter):
    """JSON converter for protein structure prediction.

    Optimized for input formats of models like Protenix/AlphaFold3.
    """

    @staticmethod
    def merge_covalent_bonds(covalent_bonds: List[Dict[str, Any]],
                            all_entity_counts: Dict[str, int]) -> List[Dict[str, Any]]:
        """Merge covalent bonds with the same entity and position.

        Reference implementation from Protenix.

        Args:
            covalent_bonds: List of covalent bond dictionaries.
            all_entity_counts: Dictionary mapping entity IDs to chain counts.

        Returns:
            List of merged covalent bond dictionaries.
        """
        bonds_recorder = defaultdict(list)
        bonds_entity_counts = {}

        for bond_dict in covalent_bonds:
            bond_unique_string = []
            entity_counts = (
                all_entity_counts.get(str(bond_dict.get("entity1", "")), 0),
                all_entity_counts.get(str(bond_dict.get("entity2", "")), 0)
            )

            for i in range(2):
                for key in ["entity", "position", "atom"]:
                    k = f"{key}{i+1}"
                    if k in bond_dict:
                        bond_unique_string.append(str(bond_dict[k]))

            bond_unique_string = "_".join(bond_unique_string)
            bonds_recorder[bond_unique_string].append(bond_dict)
            bonds_entity_counts[bond_unique_string] = entity_counts

        merged_covalent_bonds = []
        for key, bonds in bonds_recorder.items():
            counts1, counts2 = bonds_entity_counts[key]

            if counts1 == counts2 == len(bonds) and len(bonds) > 0:
                # Can be merged
                bond_dict_copy = copy.deepcopy(bonds[0])
                # Remove copy fields
                bond_dict_copy.pop("copy1", None)
                bond_dict_copy.pop("copy2", None)
                merged_covalent_bonds.append(bond_dict_copy)
            else:
                merged_covalent_bonds.extend(bonds)

        return merged_covalent_bonds

    @staticmethod
    def extract_sequences(json_data: Union[Dict[str, Any], JSONData]) -> Dict[str, List[str]]:
        """Extract all sequence information.

        Args:
            json_data: Input data, either a dictionary or JSONData object.

        Returns:
            Dictionary of sequences grouped by type.
        """
        if isinstance(json_data, JSONData):
            data = json_data.data
        else:
            data = json_data

        sequences_by_type = {
            "proteinChain": [],
            "dnaSequence": [],
            "rnaSequence": [],
            "ligand": [],
            "ion": []
        }

        for entity_dict in data.get("sequences", []):
            for seq_type, entity_info in entity_dict.items():
                if seq_type in sequences_by_type:
                    if "sequence" in entity_info:
                        sequences_by_type[seq_type].append(entity_info["sequence"])
                    elif "ligand" in entity_info:
                        sequences_by_type[seq_type].append(entity_info["ligand"])
                    elif "ion" in entity_info:
                        sequences_by_type[seq_type].append(entity_info["ion"])

        return sequences_by_type

    @staticmethod
    def calculate_composition(json_data: Union[Dict[str, Any], JSONData]) -> Dict[str, int]:
        """Calculate structure composition.

        Args:
            json_data: Input data, either a dictionary or JSONData object.

        Returns:
            Dictionary containing composition statistics.
        """
        if isinstance(json_data, JSONData):
            data = json_data.data
        else:
            data = json_data

        composition = {
            "num_entities": 0,
            "num_chains": 0,
            "num_residues": 0,
            "num_protein_chains": 0,
            "num_dna_chains": 0,
            "num_rna_chains": 0,
            "num_ligands": 0,
            "num_ions": 0
        }

        for entity_dict in data.get("sequences", []):
            for seq_type, entity_info in entity_dict.items():
                composition["num_entities"] += 1
                count = entity_info.get("count", 1)
                composition["num_chains"] += count

                if "sequence" in entity_info:
                    seq_len = len(entity_info["sequence"])
                    composition["num_residues"] += seq_len * count

                if seq_type == "proteinChain":
                    composition["num_protein_chains"] += count
                elif seq_type == "dnaSequence":
                    composition["num_dna_chains"] += count
                elif seq_type == "rnaSequence":
                    composition["num_rna_chains"] += count
                elif seq_type == "ligand":
                    composition["num_ligands"] += count
                elif seq_type == "ion":
                    composition["num_ions"] += count

        return composition

    @staticmethod
    def add_entity_ids(json_data: Union[Dict[str, Any], JSONData]) -> Dict[str, Any]:
        """Add IDs to each entity.

        Args:
            json_data: Input data, either a dictionary or JSONData object.

        Returns:
            Data dictionary with entity IDs added.
        """
        if isinstance(json_data, JSONData):
            data = copy.deepcopy(json_data.data)
        else:
            data = copy.deepcopy(json_data)

        entity_id = 1
        for entity_dict in data.get("sequences", []):
            for entity_info in entity_dict.values():
                entity_info["entity_id"] = entity_id
                entity_id += 1

        return data

    @staticmethod
    def convert_modifications_format(json_data: Union[Dict[str, Any], JSONData],
                                    target_format: str = "protenix") -> Dict[str, Any]:
        """Convert modification format.

        Args:
            json_data: Input data, either a dictionary or JSONData object.
            target_format: Target format (currently only "protenix" is supported).

        Returns:
            Data dictionary with converted modification format.
        """
        if isinstance(json_data, JSONData):
            data = copy.deepcopy(json_data.data)
        else:
            data = copy.deepcopy(json_data)

        for entity_dict in data.get("sequences", []):
            for seq_type, entity_info in entity_dict.items():
                if "modifications" in entity_info:
                    mods = entity_info["modifications"]
                    converted_mods = []

                    for mod in mods:
                        if isinstance(mod, list) and len(mod) == 2:
                            # Old format: [position, "CCD_XXX"]
                            position, mod_type = mod
                            if seq_type == "proteinChain":
                                converted_mods.append({
                                    "ptmPosition": position,
                                    "ptmType": mod_type
                                })
                            else:
                                converted_mods.append({
                                    "basePosition": position,
                                    "modificationType": mod_type
                                })
                        elif isinstance(mod, dict):
                            # New format, keep as is
                            converted_mods.append(mod)

                    entity_info["modifications"] = converted_mods

        return data

    @staticmethod
    def create_bond_dict(entity1: int, position1: int, atom1: str,
                        entity2: int, position2: int, atom2: str,
                        copy1: Optional[int] = None,
                        copy2: Optional[int] = None) -> Dict[str, Any]:
        """Create a covalent bond dictionary.

        Args:
            entity1: First entity ID.
            position1: First position.
            atom1: First atom name.
            entity2: Second entity ID.
            position2: Second position.
            atom2: Second atom name.
            copy1: Optional copy ID for first entity.
            copy2: Optional copy ID for second entity.

        Returns:
            Covalent bond dictionary.
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

    @staticmethod
    def filter_by_entity_type(json_data: Union[Dict[str, Any], JSONData],
                             include_types: Optional[List[str]] = None,
                             exclude_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Filter data by entity type.

        Args:
            json_data: Input data, either a dictionary or JSONData object.
            include_types: List of types to include (None means include all).
            exclude_types: List of types to exclude.

        Returns:
            Filtered data dictionary.
        """
        if isinstance(json_data, JSONData):
            data = copy.deepcopy(json_data.data)
            name = json_data.name
        else:
            data = copy.deepcopy(json_data)
            name = data.get("name", "")

        include_types = include_types or ["proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"]
        exclude_types = exclude_types or []

        filtered_sequences = []
        for entity_dict in data.get("sequences", []):
            for seq_type in list(entity_dict.keys()):
                if seq_type in include_types and seq_type not in exclude_types:
                    filtered_sequences.append({seq_type: entity_dict[seq_type]})

        result = {
            "name": name,
            "sequences": filtered_sequences
        }

        # Preserve other fields
        for key in ["covalent_bonds", "assembly_id"]:
            if key in data:
                result[key] = data[key]

        return result


class JSONBatchProcessor:
    """JSON batch processor.

    Used for batch processing of multiple JSON files.
    """

    def __init__(self,
                 parser: Optional[JSONParser] = None,
                 converter: Optional[JSONConverter] = None,
                 writer: Optional[ProteinJSONWriter] = None):
        """Initialize the batch processor.

        Args:
            parser: JSON parser instance.
            converter: JSON converter instance.
            writer: JSON writer instance.
        """
        self.parser = parser or JSONParser()
        self.converter = converter or JSONConverter()
        self.writer = writer or ProteinJSONWriter()

    def batch_convert(self,
                     input_paths: List[Union[str, Path]],
                     output_dir: Union[str, Path],
                     operation: str = "normalize") -> List[Path]:
        """Batch convert JSON files.

        Args:
            input_paths: List of input file paths.
            output_dir: Output directory path.
            operation: Operation type: "normalize", "split", or "merge".

        Returns:
            List of output file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = []

        if operation == "merge":
            # Merge all files
            json_data_list = []
            names = []

            for path in input_paths:
                try:
                    json_data = self.parser.parse_file(path)
                    json_data_list.append(json_data)
                    names.append(json_data.name or Path(path).stem)
                except Exception as e:
                    logger.warning(f"Failed to parse {path}: {e}")
                    continue

            if json_data_list:
                merged = self.converter.merge(json_data_list, merge_name="merged_batch")
                output_path = output_dir / "merged.json"
                self.writer.write(merged, output_path, name="merged_batch")
                output_paths.append(output_path)

        elif operation == "split":
            # Split each file
            for path in input_paths:
                try:
                    json_data = self.parser.parse_file(path)
                    split_data_list = self.converter.split_by_entity(json_data)

                    for i, split_data in enumerate(split_data_list):
                        output_path = output_dir / f"{Path(path).stem}_entity_{i+1}.json"
                        self.writer.write(split_data, output_path, name=split_data.get("name", ""))
                        output_paths.append(output_path)

                except Exception as e:
                    logger.warning(f"Failed to process {path}: {e}")
                    continue

        else:  # normalize
            # Normalize each file
            for path in input_paths:
                try:
                    json_data = self.parser.parse_file(path)
                    normalized = self.converter.normalize(json_data)

                    output_path = output_dir / f"{Path(path).stem}_normalized.json"
                    self.writer.write(normalized, output_path, name=normalized.get("name", ""))
                    output_paths.append(output_path)

                except Exception as e:
                    logger.warning(f"Failed to process {path}: {e}")
                    continue

        return output_paths
