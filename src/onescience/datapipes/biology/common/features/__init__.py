"""Biological data feature processing module.

Unified feature processing interface, inspired by Protenix and OpenFold implementations.
Supports conversion to multiple model input formats.
"""

from onescience.datapipes.biology.common.features.constants import (
    # Amino acid and nucleotide mappings
    RESTYPE_1TO3,
    RESTYPE_3TO1,
    RESTYPES,
    RESTYPE_ORDER,
    RNA_NT_TO_ID,
    RNA_ID_TO_NT,
    DNA_NT_TO_ID,
    DNA_ID_TO_NT,
    STD_RESIDUES,
    STD_RESIDUES_WITH_GAP,

    # Atom-related constants
    ATOM_TYPES,
    ATOM_ORDER,
    ATOM37_VDW,
    RES_ATOMS_DICT,

    # Molecule types
    EVALUATION_CHAIN_INTERFACE,
    ENTITY_POLY_TYPE_DICT,

    # Feature names
    MSA_FEATURE_NAMES,
    SEQUENCE_FEATURE_NAMES,
    STRUCTURE_FEATURE_NAMES,
    TEMPLATE_FEATURE_NAMES,
)

from onescience.datapipes.biology.common.features.feature_base import (
    FeatureDict,
    TensorDict,
    BaseFeatureExtractor,
    FeaturePipeline,
)

from onescience.datapipes.biology.common.features.sequence_features import (
    SequenceFeatureExtractor,
    encode_sequence,
    make_sequence_features,
    restype_onehot_encode,
    create_target_feat,
)

from onescience.datapipes.biology.common.features.msa_features import (
    MSAFeatureExtractor,
    make_msa_features,
    make_msa_mask,
    create_msa_feat,
    create_deletion_matrix,
    compute_row_weights,
)

from onescience.datapipes.biology.common.features.structure_features import (
    StructureFeatureExtractor,
    make_structure_features,
    pseudo_beta_fn,
    make_pseudo_beta,
    atom37_to_frames,
    compute_distance_matrix,
    compute_contact_map,
)

from onescience.datapipes.biology.common.features.feature_utils import (
    encode_to_onehot,
    pad_features,
    crop_features,
    merge_features,
    select_features,
    cast_to_64bit_ints,
    make_one_hot,
    squeeze_features,
    add_constant_field,
)

from onescience.datapipes.biology.common.features.feature_pipeline import (
    BiologyFeaturePipeline,
    UnifiedFeaturePipeline,
    np_example_to_features,
    make_data_config,
)

from onescience.datapipes.biology.common.features.token_features import (
    # Encoding functions
    encoder,
    restype_onehot_encoded,
    elem_onehot_encoded,
    ref_atom_name_chars_encoded,
    # Frame construction
    get_prot_nuc_frame_atom_names,
    check_colinear,
    compute_frame_from_positions,
    # Token features
    get_token_features_from_annotations,
    get_reference_features,
    get_bond_features,
    classify_polymer_bonds,
    # Auxiliary features
    get_chain_perm_features,
    get_extra_features,
    get_mask_features,
    get_label_features,
    # Utility classes/functions
    TokenFeatureExtractor,
    create_atom_to_token_mapping,
    validate_frame_atoms,
)

__all__ = [
    # Constants
    "RESTYPE_1TO3",
    "RESTYPE_3TO1",
    "RESTYPES",
    "RESTYPE_ORDER",
    "RNA_NT_TO_ID",
    "RNA_ID_TO_NT",
    "DNA_NT_TO_ID",
    "DNA_ID_TO_NT",
    "STD_RESIDUES",
    "STD_RESIDUES_WITH_GAP",
    "ATOM_TYPES",
    "ATOM_ORDER",
    "ATOM37_VDW",
    "RES_ATOMS_DICT",
    "EVALUATION_CHAIN_INTERFACE",
    "ENTITY_POLY_TYPE_DICT",
    "MSA_FEATURE_NAMES",
    "SEQUENCE_FEATURE_NAMES",
    "STRUCTURE_FEATURE_NAMES",
    "TEMPLATE_FEATURE_NAMES",

    # Base
    "FeatureDict",
    "TensorDict",
    "BaseFeatureExtractor",
    "FeaturePipeline",

    # Sequence
    "SequenceFeatureExtractor",
    "encode_sequence",
    "make_sequence_features",
    "restype_onehot_encode",
    "create_target_feat",

    # MSA
    "MSAFeatureExtractor",
    "make_msa_features",
    "make_msa_mask",
    "create_msa_feat",
    "create_deletion_matrix",
    "compute_row_weights",

    # Structure
    "StructureFeatureExtractor",
    "make_structure_features",
    "pseudo_beta_fn",
    "make_pseudo_beta",
    "atom37_to_frames",
    "compute_distance_matrix",
    "compute_contact_map",

    # Utils
    "encode_to_onehot",
    "pad_features",
    "crop_features",
    "merge_features",
    "select_features",
    "cast_to_64bit_ints",
    "make_one_hot",
    "squeeze_features",
    "add_constant_field",

    # Pipeline
    "BiologyFeaturePipeline",
    "UnifiedFeaturePipeline",
    "np_example_to_features",
    "make_data_config",

    # Token Features (migrated from Protenix)
    "encoder",
    "restype_onehot_encoded",
    "elem_onehot_encoded",
    "ref_atom_name_chars_encoded",
    "get_prot_nuc_frame_atom_names",
    "check_colinear",
    "compute_frame_from_positions",
    "get_token_features_from_annotations",
    "get_reference_features",
    "get_bond_features",
    "classify_polymer_bonds",
    "get_chain_perm_features",
    "get_extra_features",
    "get_mask_features",
    "get_label_features",
    "TokenFeatureExtractor",
    "create_atom_to_token_mapping",
    "validate_frame_atoms",
]
