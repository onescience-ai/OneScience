"""Protenix model adapter for inference.

This module provides an adapter that converts unified biological data formats
into formats required by the Protenix protein structure prediction model.
Supports processing from JSON input to generate complete Protenix features.

Example:
    >>> adapter = ProtenixInferAdapter(config)
    >>> features = adapter.process_json_sample(json_data)
"""

import copy
from typing import Any, Dict, Optional, Tuple, Union
import logging

import numpy as np
import torch
from biotite.structure import AtomArray

from onescience.datapipes.biology.adapters.base_adapter import (
    BaseAdapter,
    FeatureDict,
)
from onescience.datapipes.biology.adapters.adapter_registry import register_adapter
from onescience.datapipes.biology.common.json.json_parser import JSONData
from onescience.datapipes.biology.common.structure.molecular_builder import (
    add_entity_atom_array,
    remove_leaving_atoms,
)
from onescience.datapipes.biology.common.tokenizer import (
    AtomArrayTokenizer,
    TokenArray,
)

logger = logging.getLogger(__name__)

# Import Protenix-specific modules (no unified interface available)
try:
    from onescience.datapipes.protenix.parser import AddAtomArrayAnnot
    from onescience.datapipes.protenix.featurizer import Featurizer
    from onescience.datapipes.protenix.utils import int_to_letters, make_dummy_feature
    from onescience.datapipes.protenix.msa_featurizer import InferenceMSAFeaturizer
    from onescience.datapipes.protenix.data_pipeline import DataPipeline
    from onescience.utils.protenix.torch_utils import dict_to_tensor

    _HAS_UNIFIED_INTERFACE = True
except ImportError as e:
    _HAS_UNIFIED_INTERFACE = False
    logger.warning(f"Protenix modules not available: {e}")


class ProtenixInferAdapter(BaseAdapter):
    """Adapter for Protenix model inference.

    Converts unified biological data formats into Protenix-specific feature formats.
    Supports two usage modes:
    1. Simple sample processing (sequence/msa_path) -> process_sample
    2. Complete JSON processing (Protenix format) -> process_json_sample

    Attributes:
        config: Dataset configuration.
    """

    def adapt_features(self, common_features: FeatureDict) -> FeatureDict:
        """Convert common features to Protenix format.

        Args:
            common_features: Dictionary of common biological features.

        Returns:
            FeatureDict: Dictionary of Protenix-specific features.
        """
        protenix_features = {}

        # Convert sequence features
        if "aatype" in common_features:
            aatype = common_features["aatype"]
            if aatype.ndim == 1:
                # Convert integer encoding to one-hot
                protenix_features["aatype"] = self._int_to_onehot(aatype, num_classes=22)
            else:
                protenix_features["aatype"] = aatype

        # Convert MSA features
        if "msa" in common_features:
            protenix_features["msa"] = common_features["msa"]
            protenix_features["msa_mask"] = (common_features["msa"] != 0).astype(np.float32)

        if "deletion_matrix" in common_features:
            protenix_features["deletion_matrix"] = common_features["deletion_matrix"]
            protenix_features["deletion_matrix_int"] = common_features["deletion_matrix"].astype(np.int32)

        # Add Protenix-specific features
        if "num_alignments" in common_features:
            protenix_features["num_alignments"] = common_features["num_alignments"]

        # Add residue_index if not present
        if "residue_index" not in protenix_features and "aatype" in protenix_features:
            seq_len = protenix_features["aatype"].shape[0]
            protenix_features["residue_index"] = np.arange(seq_len, dtype=np.int32)

        return protenix_features

    def process_sample(self, sample: Dict[str, Any]) -> FeatureDict:
        """Process a single sample (simple interface).

        Args:
            sample: Raw sample data containing:
                - sequence: Amino acid sequence string.
                - structure_path: Path to structure file.
                - msa_path: Path to MSA directory (containing pairing.a3m and non_pairing.a3m).

        Returns:
            FeatureDict: Protenix-formatted feature dictionary.
        """
        features = {}

        # Process sequence
        if "sequence" in sample:
            sequence = sample["sequence"]
            encoded = self.aa_encoder.encode(sequence)
            features["aatype"] = self.aa_encoder.one_hot_encode(sequence)
            features["sequence"] = sequence

        # Process MSA
        if "msa_path" in sample:
            from pathlib import Path
            msa_path = Path(sample["msa_path"])

            # If msa_path is a directory, search for .a3m files
            if msa_path.is_dir():
                # Prefer pairing.a3m, then non_pairing.a3m
                possible_files = ["pairing.a3m", "non_pairing.a3m"]
                msa_file = None
                for fname in possible_files:
                    candidate = msa_path / fname
                    if candidate.exists():
                        msa_file = candidate
                        break

                if msa_file is None:
                    # If specific files not found, try any .a3m file
                    a3m_files = list(msa_path.glob("*.a3m"))
                    if a3m_files:
                        msa_file = a3m_files[0]

                if msa_file is None:
                    raise ValueError(f"No .a3m files found in MSA directory: {msa_path}")
            else:
                msa_file = msa_path

            msa = self.msa_parser.parse_file(msa_file)
            msa_features = self.msa_featurizer.featurize(msa)
            features.update(msa_features)

        # Convert to Protenix format
        protenix_features = self.adapt_features(features)

        return protenix_features

    def process_json_sample(
        self,
        json_data: Union[Dict[str, Any], JSONData],
    ) -> Tuple[Dict[str, torch.Tensor], AtomArray, "TokenArray"]:
        """Process Protenix-format JSON sample to generate complete features.

        This is the unified JSON feature extraction interface, replacing
the functionality of protenix_json_to_feature.py.

        Args:
            json_data: Protenix-format JSON data or JSONData object.

        Returns:
            Tuple containing:
                - feature_dict: Dictionary of features as torch tensors.
                - atom_array: AtomArray object with atom information.
                - token_array: TokenArray object with token information.

        Raises:
            ImportError: If Protenix dependencies are not available.
        """
        if not _HAS_UNIFIED_INTERFACE:
            raise ImportError(
                "Unified protein utils are required for process_json_sample. "
                "Please ensure protenix dependencies are available."
            )

        # Convert to JSONData object if needed
        if isinstance(json_data, dict):
            json_data = JSONData(
                data=json_data,
                name=json_data.get("name", ""),
            )

        # Generate features using unified interface
        return self._get_features_from_json_data(json_data)

    def _get_features_from_json_data(
        self,
        json_data: JSONData,
    ) -> Tuple[Dict[str, torch.Tensor], AtomArray, "TokenArray"]:
        """Generate complete features from JSONData (internal implementation).

        Args:
            json_data: JSONData object containing molecular description.

        Returns:
            Tuple of (feature_dict, atom_array, token_array).
        """
        # Step 1: Add atom_array to each entity
        input_dict = add_entity_atom_array(copy.deepcopy(json_data.data))

        # Step 2: Get entity type mappings
        entity_poly_type = self._get_entity_poly_type(input_dict)

        # Step 3: Build complete AtomArray
        atom_array = self._build_full_atom_array(input_dict, entity_poly_type)

        # Step 4: Add covalent bonds between entities
        atom_array = self._add_bonds_between_entities(atom_array, input_dict)

        # Step 5: Convert MSE residues to MET
        atom_array = self._mse_to_met(atom_array)

        # Step 6: Add AtomArray attributes
        atom_array = self._add_atom_array_attributes(atom_array, entity_poly_type)

        # Step 7: Generate TokenArray and features
        aa_tokenizer = AtomArrayTokenizer(atom_array)
        token_array = aa_tokenizer.get_token_array()

        featurizer = Featurizer(token_array, atom_array)
        feature_dict = featurizer.get_all_input_features()

        token_array_with_frame = featurizer.get_token_frame(
            token_array=token_array,
            atom_array=atom_array,
            ref_pos=feature_dict["ref_pos"],
            ref_mask=feature_dict["ref_mask"],
        )

        # Add has_frame and frame_atom_index
        feature_dict["has_frame"] = torch.Tensor(
            token_array_with_frame.get_annotation("has_frame")
        ).long()

        feature_dict["frame_atom_index"] = torch.Tensor(
            token_array_with_frame.get_annotation("frame_atom_index")
        ).long()

        # Step 8: Add MSA features
        entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(atom_array)
        msa_features = InferenceMSAFeaturizer.make_msa_feature(
            bioassembly=input_dict["sequences"],
            entity_to_asym_id=entity_to_asym_id,
            token_array=token_array,
            atom_array=atom_array,
        )

        # Handle MSA features (add dummy if none)
        dummy_feats = ["template"]
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            feature_dict.update(msa_features)
        feature_dict = make_dummy_feature(
            features_dict=feature_dict,
            dummy_feats=dummy_feats,
        )

        # Add entity_poly_type to feature dict for later use
        feature_dict["entity_poly_type"] = entity_poly_type

        return feature_dict, atom_array, token_array

    def _get_entity_poly_type(self, input_dict: Dict[str, Any]) -> Dict[str, str]:
        """Get entity type mappings from input dictionary.

        Args:
            input_dict: Input dictionary with sequence information.

        Returns:
            Dict mapping entity IDs to polymer types.
        """
        entity_type_mapping_dict = {
            "proteinChain": "polypeptide(L)",
            "dnaSequence": "polydeoxyribonucleotide",
            "rnaSequence": "polyribonucleotide",
        }
        entity_poly_type = {}

        for idx, type2entity_dict in enumerate(input_dict["sequences"]):
            for entity_type, entity in type2entity_dict.items():
                if "sequence" in entity:
                    entity_poly_type[str(idx + 1)] = entity_type_mapping_dict[entity_type]

        return entity_poly_type

    def _build_full_atom_array(
        self,
        input_dict: Dict[str, Any],
        entity_poly_type: Dict[str, str],
    ) -> AtomArray:
        """Build complete AtomArray from input dictionary.

        Args:
            input_dict: Input dictionary with atom array information.
            entity_poly_type: Entity type mappings.

        Returns:
            AtomArray: Complete atom array with all entities.
        """
        atom_array = None
        asym_chain_idx = 0

        for idx, type2entity_dict in enumerate(input_dict["sequences"]):
            for entity_type, entity in type2entity_dict.items():
                entity_id = str(idx + 1)

                entity_atom_array = None
                for asym_chain_count in range(1, entity["count"] + 1):
                    asym_id_str = int_to_letters(asym_chain_idx + 1)
                    asym_chain = copy.deepcopy(entity["atom_array"])
                    chain_id = [asym_id_str] * len(asym_chain)
                    copy_id = [asym_chain_count] * len(asym_chain)

                    asym_chain.set_annotation("label_asym_id", chain_id)
                    asym_chain.set_annotation("auth_asym_id", chain_id)
                    asym_chain.set_annotation("chain_id", chain_id)
                    asym_chain.set_annotation("label_seq_id", asym_chain.res_id)
                    asym_chain.set_annotation("copy_id", copy_id)

                    if entity_atom_array is None:
                        entity_atom_array = asym_chain
                    else:
                        entity_atom_array += asym_chain

                    asym_chain_idx += 1

                entity_atom_array.set_annotation(
                    "label_entity_id", [entity_id] * len(entity_atom_array)
                )

                # Set hetero flag based on entity type
                if entity_type in ["proteinChain", "dnaSequence", "rnaSequence"]:
                    entity_atom_array.hetero[:] = False
                else:
                    entity_atom_array.hetero[:] = True

                if atom_array is None:
                    atom_array = entity_atom_array
                else:
                    atom_array += entity_atom_array

        return atom_array

    def _add_bonds_between_entities(
        self,
        atom_array: AtomArray,
        input_dict: Dict[str, Any],
    ) -> AtomArray:
        """Add covalent bonds between entities.

        Args:
            atom_array: AtomArray to add bonds to.
            input_dict: Input dictionary containing bond information.

        Returns:
            AtomArray: AtomArray with bonds added.
        """
        if "covalent_bonds" not in input_dict:
            return atom_array

        bond_count = {}

        for bond_info_dict in input_dict["covalent_bonds"]:
            bond_atoms = []

            for idx, side in enumerate(["left", "right"]):
                entity_id = int(
                    bond_info_dict.get(
                        f"{side}_entity", bond_info_dict.get(f"entity{idx+1}")
                    )
                )
                copy_id = bond_info_dict.get(
                    f"{side}_copy", bond_info_dict.get(f"copy{idx+1}")
                )
                position = int(
                    bond_info_dict.get(
                        f"{side}_position", bond_info_dict.get(f"position{idx+1}")
                    )
                )
                atom_name = bond_info_dict.get(
                    f"{side}_atom", bond_info_dict.get(f"atom{idx+1}")
                )

                if copy_id is not None:
                    copy_id = int(copy_id)

                # Handle SMILES atom indices
                if isinstance(atom_name, str) and atom_name.isdigit():
                    atom_name = int(atom_name)

                if isinstance(atom_name, int):
                    entity_dict = list(input_dict["sequences"][int(entity_id - 1)].values())[0]
                    atom_name = entity_dict["atom_map_to_atom_name"][atom_name]

                # Get atom indices
                atom_indices = self._get_bond_atom(
                    atom_array, entity_id, position, atom_name, copy_id
                )
                bond_atoms.append(atom_indices)

            # Create bonds
            for atom_idx1, atom_idx2 in zip(bond_atoms[0], bond_atoms[1]):
                atom_array.bonds.add_bond(atom_idx1, atom_idx2, 1)
                bond_count[atom_idx1] = bond_count.get(atom_idx1, 0) + 1
                bond_count[atom_idx2] = bond_count.get(atom_idx2, 0) + 1

        # Remove leaving atoms
        atom_array = remove_leaving_atoms(atom_array, bond_count)

        return atom_array

    def _get_bond_atom(
        self,
        atom_array: AtomArray,
        entity_id: int,
        position: int,
        atom_name: str,
        copy_id: Optional[int] = None,
    ) -> np.ndarray:
        """Get atom indices for bonding.

        Args:
            atom_array: AtomArray to search.
            entity_id: Entity ID.
            position: Residue position.
            atom_name: Name of the atom.
            copy_id: Copy ID (optional).

        Returns:
            np.ndarray: Array of matching atom indices.
        """
        entity_mask = atom_array.label_entity_id == str(entity_id)
        position_mask = atom_array.res_id == int(position)
        atom_name_mask = atom_array.atom_name == str(atom_name)

        if copy_id is not None:
            copy_mask = atom_array.copy_id == int(copy_id)
            mask = entity_mask & position_mask & atom_name_mask & copy_mask
        else:
            mask = entity_mask & position_mask & atom_name_mask

        return np.where(mask)[0]

    def _mse_to_met(self, atom_array: AtomArray) -> AtomArray:
        """Convert MSE (selenomethionine) residues to MET (methionine).

        Args:
            atom_array: AtomArray containing MSE residues.

        Returns:
            AtomArray: AtomArray with MSE converted to MET.
        """
        mse = atom_array.res_name == "MSE"
        se = mse & (atom_array.atom_name == "SE")
        atom_array.atom_name[se] = "SD"
        atom_array.element[se] = "S"
        atom_array.res_name[mse] = "MET"
        atom_array.hetero[mse] = False
        return atom_array

    def _add_atom_array_attributes(
        self,
        atom_array: AtomArray,
        entity_poly_type: Dict[str, str],
    ) -> AtomArray:
        """Add standard attributes to AtomArray.

        Args:
            atom_array: AtomArray to add attributes to.
            entity_poly_type: Entity type mappings.

        Returns:
            AtomArray: AtomArray with all standard attributes added.
        """
        atom_array = AddAtomArrayAnnot.add_token_mol_type(atom_array, entity_poly_type)
        atom_array = AddAtomArrayAnnot.add_centre_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_atom_mol_type_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_distogram_rep_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_plddt_m_rep_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_cano_seq_resname(atom_array)
        atom_array = AddAtomArrayAnnot.add_tokatom_idx(atom_array)
        atom_array = AddAtomArrayAnnot.add_modified_res_mask(atom_array)
        atom_array = AddAtomArrayAnnot.unique_chain_and_add_ids(atom_array)
        atom_array = AddAtomArrayAnnot.find_equiv_mol_and_assign_ids(
            atom_array, entity_poly_type, check_final_equiv=False
        )
        atom_array = AddAtomArrayAnnot.add_ref_space_uid(atom_array)
        return atom_array

    def _int_to_onehot(self, arr: np.ndarray, num_classes: int = 22) -> np.ndarray:
        """Convert integer encoding to one-hot encoding.

        Args:
            arr: Integer-encoded array.
            num_classes: Number of classes for one-hot encoding.

        Returns:
            np.ndarray: One-hot encoded array with shape (len(arr), num_classes).
        """
        one_hot = np.zeros((len(arr), num_classes), dtype=np.float32)
        one_hot[np.arange(len(arr)), arr] = 1.0
        return one_hot


# Register the adapter
register_adapter("protenix_infer_adapter", ProtenixInferAdapter)
