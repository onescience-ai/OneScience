"""General biological molecular structure builder.

Integrates core functionality from Protenix json_parser and ccd into a standalone module.
Supports building atomic-level structures from sequence/chemical descriptions.

Dependencies: biotite, numpy, rdkit
"""

import concurrent.futures
import copy
import functools
import logging
import pickle
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import biotite
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

# =============================================================================
# Encoding mapping tables
# =============================================================================

DNA_1TO3 = {
    "A": "DA", "G": "DG", "C": "DC", "T": "DT",
    "X": "DN", "I": "DI", "N": "DN", "U": "DU",
}

RNA_1TO3 = {
    "A": "A", "G": "G", "C": "C", "U": "U",
    "X": "N", "I": "I", "N": "N",
}

PROTEIN_1TO3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP",
    "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
    "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
    "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "X": "UNK",
}

ENTITY_TYPE_MAP = {
    "proteinChain": "polypeptide(L)",
    "dnaSequence": "polydeoxyribonucleotide",
    "rnaSequence": "polyribonucleotide",
    "ligand": "non-polymer",
    "ion": "non-polymer",
}

# =============================================================================
# CCD (Chemical Component Dictionary) utility functions
# =============================================================================

# Global cache
_ccd_cif_cache = None
_ccd_rdkit_mols: Dict[str, Chem.Mol] = {}

# CCD file paths (requires external configuration)
CCD_COMPONENTS_FILE: Optional[str] = None
CCD_RDKIT_MOL_FILE: Optional[str] = None


def _get_ccd_paths_from_configs() -> Tuple[Optional[str], Optional[str]]:
    """Attempt to get CCD file paths from configs.

    Returns:
        Tuple of (components_file, rdkit_mol_file) or (None, None).
    """
    try:
        from configs.configs_data import data_configs
        components_file = data_configs.get("ccd_components_file")
        rdkit_mol_file = data_configs.get("ccd_components_rdkit_mol_file")
        return components_file, rdkit_mol_file
    except ImportError:
        return None, None
    except Exception as e:
        logger.debug(f"Failed to get CCD paths from configs: {e}")
        return None, None


def set_ccd_paths(components_file: str, rdkit_mol_file: str):
    """Set CCD file paths.

    Args:
        components_file: Path to the CCD components file.
        rdkit_mol_file: Path to the RDKit mol file.
    """
    global CCD_COMPONENTS_FILE, CCD_RDKIT_MOL_FILE
    CCD_COMPONENTS_FILE = components_file
    CCD_RDKIT_MOL_FILE = rdkit_mol_file


def set_ccd_paths_from_configs(configs: Dict[str, Any]):
    """Set CCD file paths from a configuration dictionary.

    Args:
        configs: Configuration dictionary containing data.ccd_components_file
            and data.ccd_components_rdkit_mol_file.
    """
    global CCD_COMPONENTS_FILE, CCD_RDKIT_MOL_FILE

    # Support multiple configuration formats
    if "data" in configs:
        data_cfg = configs["data"]
        CCD_COMPONENTS_FILE = data_cfg.get("ccd_components_file")
        CCD_RDKIT_MOL_FILE = data_cfg.get("ccd_components_rdkit_mol_file")
    else:
        CCD_COMPONENTS_FILE = configs.get("ccd_components_file")
        CCD_RDKIT_MOL_FILE = configs.get("ccd_components_rdkit_mol_file")

    if CCD_COMPONENTS_FILE:
        logger.info(f"CCD components file set from configs: {CCD_COMPONENTS_FILE}")
    if CCD_RDKIT_MOL_FILE:
        logger.info(f"CCD RDKit mol file set from configs: {CCD_RDKIT_MOL_FILE}")


@functools.lru_cache(maxsize=1)
def _load_ccd_cif() -> Optional[pdbx.CIFFile]:
    """Load CCD components file.

    Returns:
        CIFFile object or None if loading fails.
    """
    global CCD_COMPONENTS_FILE

    # If not set, try to get from configs
    if CCD_COMPONENTS_FILE is None:
        components_file, _ = _get_ccd_paths_from_configs()
        if components_file:
            CCD_COMPONENTS_FILE = components_file
            logger.info(f"CCD_COMPONENTS_FILE auto-loaded from configs: {CCD_COMPONENTS_FILE}")
        else:
            logger.warning(
                "CCD_COMPONENTS_FILE not set. "
                "Use set_ccd_paths() or set_ccd_paths_from_configs() first."
            )
            return None

    try:
        return pdbx.CIFFile.read(CCD_COMPONENTS_FILE)
    except Exception as e:
        logger.error(f"Failed to load CCD components from {CCD_COMPONENTS_FILE}: {e}")
        return None


def _map_central_to_leaving_groups(component: AtomArray) -> Optional[Dict[str, List[List[str]]]]:
    """Map central atoms to leaving groups.

    Args:
        component: AtomArray component to process.

    Returns:
        Dictionary mapping central atom names to leaving groups, or None.
    """
    comp = component.copy()
    if comp.bonds is None:
        return {}

    central_to_leaving_groups = defaultdict(list)
    for c_idx in np.flatnonzero(~comp.leaving_atom_flag):
        bonds, _ = comp.bonds.get_bonds(c_idx)
        for l_idx in bonds:
            if comp.leaving_atom_flag[l_idx]:
                comp.bonds.remove_bond(c_idx, l_idx)
                group_idx = struc.find_connected(comp.bonds, l_idx)
                if not np.all(comp.leaving_atom_flag[group_idx]):
                    return None
                central_to_leaving_groups[comp.atom_name[c_idx]].append(
                    comp.atom_name[group_idx].tolist()
                )
    return central_to_leaving_groups


@functools.lru_cache(maxsize=1024)
def get_component_atom_array(
    ccd_code: str,
    keep_leaving_atoms: bool = False,
    keep_hydrogens: bool = False
) -> Optional[AtomArray]:
    """Get atom array for a CCD component.

    Args:
        ccd_code: CCD code identifier.
        keep_leaving_atoms: Whether to keep leaving atoms.
        keep_hydrogens: Whether to keep hydrogen atoms.

    Returns:
        AtomArray or None if not found.
    """
    ccd_cif = _load_ccd_cif()
    if ccd_cif is None:
        logger.warning(f"CCD database not available, cannot load {ccd_code}")
        return None

    if ccd_code not in ccd_cif:
        logger.warning(f"CCD code {ccd_code} not found in database")
        return None

    try:
        comp = pdbx.get_component(ccd_cif, data_block=ccd_code, use_ideal_coord=True)
    except biotite.InvalidFileError as e:
        logger.warning(f"Cannot parse {ccd_code}: {e}")
        return None

    atom_category = ccd_cif[ccd_code]["chem_comp_atom"]
    leaving_atom_flag = atom_category["pdbx_leaving_atom_flag"].as_array()
    comp.set_annotation("leaving_atom_flag", leaving_atom_flag == "Y")

    for atom_id in ["alt_atom_id", "pdbx_component_atom_id"]:
        comp.set_annotation(atom_id, atom_category[atom_id].as_array())

    if not keep_leaving_atoms:
        comp = comp[~comp.leaving_atom_flag]
    if not keep_hydrogens:
        comp = comp[~np.isin(comp.element, ["H", "D"])]

    comp.central_to_leaving_groups = _map_central_to_leaving_groups(comp)
    return comp


def get_component_rdkit_mol(ccd_code: str) -> Optional[Chem.Mol]:
    """Get RDKit molecule for a CCD component.

    Args:
        ccd_code: CCD code identifier.

    Returns:
        RDKit Mol object or None if not found.
    """
    global _ccd_rdkit_mols, CCD_RDKIT_MOL_FILE

    if _ccd_rdkit_mols:
        return _ccd_rdkit_mols.get(ccd_code)

    # If not set, try to get from configs
    if CCD_RDKIT_MOL_FILE is None:
        _, rdkit_mol_file = _get_ccd_paths_from_configs()
        if rdkit_mol_file:
            CCD_RDKIT_MOL_FILE = rdkit_mol_file
            logger.info(f"CCD_RDKIT_MOL_FILE auto-loaded from configs: {CCD_RDKIT_MOL_FILE}")
        else:
            logger.warning(
                "CCD_RDKIT_MOL_FILE not set. "
                "Use set_ccd_paths() or set_ccd_paths_from_configs() first."
            )
            return None

    rdkit_mol_pkl = Path(CCD_RDKIT_MOL_FILE)
    if rdkit_mol_pkl.exists():
        try:
            with open(rdkit_mol_pkl, "rb") as f:
                _ccd_rdkit_mols = pickle.load(f)
            return _ccd_rdkit_mols.get(ccd_code)
        except Exception as e:
            logger.error(f"Failed to load RDKit mol pickle: {e}")
            return None
    else:
        logger.warning(f"RDKit mol file not found: {rdkit_mol_pkl}")
        return None


@functools.lru_cache(maxsize=1024)
def get_ccd_ref_info(ccd_code: str) -> Optional[Dict[str, Any]]:
    """Get CCD reference information.

    Args:
        ccd_code: CCD code identifier.

    Returns:
        Dictionary with keys: ccd, atom_map, coord, mask, charge, or None.
    """
    mol = get_component_rdkit_mol(ccd_code)
    if mol is None:
        return None
    if mol.GetNumAtoms() == 0:
        return None

    try:
        conf = mol.GetConformer(mol.ref_conf_id)
        coord = conf.GetPositions()
        charge = np.array([atom.GetFormalCharge() for atom in mol.GetAtoms()])

        return {
            "ccd": ccd_code,
            "atom_map": mol.atom_map,
            "coord": coord,
            "mask": mol.ref_mask,
            "charge": charge,
        }
    except Exception as e:
        logger.warning(f"Failed to get ref info for {ccd_code}: {e}")
        return None


@functools.lru_cache(maxsize=1024)
def get_mol_type(ccd_code: str) -> str:
    """Get molecule type: protein, dna, rna, or ligand.

    Args:
        ccd_code: CCD code identifier.

    Returns:
        Molecule type as a string.
    """
    ccd_cif = _load_ccd_cif()
    if ccd_cif is None or ccd_code not in ccd_cif:
        return "ligand"

    try:
        link_type = ccd_cif[ccd_code]["chem_comp"]["type"].as_item().upper()
    except Exception:
        return "ligand"

    if "PEPTIDE" in link_type and link_type != "PEPTIDE-LIKE":
        return "protein"
    if "DNA" in link_type:
        return "dna"
    if "RNA" in link_type:
        return "rna"
    return "ligand"


def _connect_inter_residue(chain: AtomArray, res_starts: np.ndarray) -> struc.BondList:
    """Connect inter-residue chemical bonds.

    Args:
        chain: AtomArray representing the molecular chain.
        res_starts: Array of residue start indices.

    Returns:
        BondList containing inter-residue bonds.
    """
    bonds = []
    for i in range(len(res_starts) - 2):
        start_i = res_starts[i]
        start_j = res_starts[i + 1]

        res_name_i = chain.res_name[start_i]
        res_name_j = chain.res_name[start_j]

        mol_type_i = get_mol_type(res_name_i)
        mol_type_j = get_mol_type(res_name_j)

        if mol_type_i == "protein" and mol_type_j == "protein":
            c_idx = np.where((chain.res_id == chain.res_id[start_i]) & (chain.atom_name == "C"))[0]
            n_idx = np.where((chain.res_id == chain.res_id[start_j]) & (chain.atom_name == "N"))[0]
            if len(c_idx) > 0 and len(n_idx) > 0:
                bonds.append([c_idx[0], n_idx[0], 1])
        elif mol_type_i in ("dna", "rna") and mol_type_j in ("dna", "rna"):
            o3_idx = np.where((chain.res_id == chain.res_id[start_i]) & (chain.atom_name == "O3'"))[0]
            p_idx = np.where((chain.res_id == chain.res_id[start_j]) & (chain.atom_name == "P"))[0]
            if len(o3_idx) > 0 and len(p_idx) > 0:
                bonds.append([o3_idx[0], p_idx[0], 1])

    return struc.BondList(chain.array_length(), np.array(bonds, dtype=np.int32))


# =============================================================================
# Core building functions
# =============================================================================

def add_reference_features(atom_array: AtomArray) -> AtomArray:
    """Add reference features to atom array.

    Args:
        atom_array: Input AtomArray.

    Returns:
        AtomArray with added reference features (ref_pos, ref_charge, ref_mask).
    """
    atom_count = len(atom_array)
    ref_pos = np.zeros((atom_count, 3), dtype=np.float32)
    ref_charge = np.zeros(atom_count, dtype=int)
    ref_mask = np.zeros(atom_count, dtype=int)

    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for start, stop in zip(starts[:-1], starts[1:]):
        res_name = atom_array.res_name[start]
        if res_name == "UNL":
            ref_pos[start:stop] = atom_array.coord[start:stop]
            ref_charge[start:stop] = atom_array.charge[start:stop]
            ref_mask[start:stop] = 1
            continue

        ref_info = get_ccd_ref_info(res_name)
        if ref_info:
            try:
                atom_sub_idx = [ref_info["atom_map"].get(name, 0) for name in atom_array.atom_name[start:stop]]
                ref_pos[start:stop] = ref_info["coord"][atom_sub_idx]
                ref_charge[start:stop] = ref_info["charge"][atom_sub_idx]
                ref_mask[start:stop] = ref_info["mask"][atom_sub_idx]
            except Exception as e:
                logger.debug(f"Error adding ref features for {res_name}: {e}")
        else:
            logger.debug(f"No reference info for {res_name}")

    atom_array.set_annotation("ref_pos", ref_pos)
    atom_array.set_annotation("ref_charge", ref_charge)
    atom_array.set_annotation("ref_mask", ref_mask)
    return atom_array


def find_range_by_index(starts: np.ndarray, atom_index: int) -> Tuple[int, int]:
    """Find residue range by atom index.

    Args:
        starts: Array of residue start indices.
        atom_index: Target atom index.

    Returns:
        Tuple of (start, stop) indices for the residue.

    Raises:
        ValueError: If atom_index is not found.
    """
    for start, stop in zip(starts[:-1], starts[1:]):
        if start <= atom_index < stop:
            return start, stop
    raise ValueError(f"atom_index {atom_index} not found")


def remove_leaving_atoms(atom_array: AtomArray, bond_count: Dict[int, int]) -> AtomArray:
    """Remove leaving atoms based on bond count.

    Args:
        atom_array: Input AtomArray.
        bond_count: Dictionary mapping atom indices to bond counts.

    Returns:
        AtomArray with leaving atoms removed.
    """
    remove_indices = []
    res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)

    for centre_idx, b_count in bond_count.items():
        res_name = atom_array.res_name[centre_idx]
        centre_name = atom_array.atom_name[centre_idx]

        comp = get_component_atom_array(res_name, keep_leaving_atoms=True, keep_hydrogens=False)
        if comp is None:
            continue

        leaving_groups = comp.central_to_leaving_groups.get(centre_name) if hasattr(comp, 'central_to_leaving_groups') else None
        if leaving_groups is None:
            continue

        if b_count > len(leaving_groups):
            remove_groups = leaving_groups
        else:
            remove_groups = random.sample(leaving_groups, b_count)

        start, stop = find_range_by_index(res_starts, centre_idx)

        for group in remove_groups:
            for atom_name in group:
                leaving_idx = np.where(atom_array.atom_name[start:stop] == atom_name)[0]
                if len(leaving_idx) == 0:
                    continue
                remove_indices.append(leaving_idx[0] + start)

    if not remove_indices:
        return atom_array

    keep_mask = np.ones(len(atom_array), dtype=bool)
    keep_mask[remove_indices] = False
    return atom_array[keep_mask]


def _remove_non_std_ccd_leaving_atoms(atom_array: AtomArray) -> AtomArray:
    """Remove non-standard CCD leaving atoms.

    Args:
        atom_array: Input AtomArray.

    Returns:
        AtomArray with non-standard leaving atoms removed.
    """
    if len(atom_array) == 0 or atom_array.res_id[-1] == 0:
        return atom_array

    connected = np.zeros(atom_array.res_id[-1], dtype=bool)
    if atom_array.bonds is not None:
        for i, j, t in atom_array.bonds._bonds:
            if abs(atom_array.res_id[i] - atom_array.res_id[j]) == 1:
                idx = min(atom_array.res_id[i], atom_array.res_id[j])
                if idx > 0:
                    connected[idx - 1] = True

    leaving_atoms = np.zeros(len(atom_array), dtype=bool)
    for res_id, conn in enumerate(connected):
        if res_id == 0 or conn:
            continue

        res_name_i = atom_array.res_name[atom_array.res_id == res_id][0] if np.any(atom_array.res_id == res_id) else ""
        res_name_j = atom_array.res_name[atom_array.res_id == res_id + 1][0] if np.any(atom_array.res_id == res_id + 1) else ""

        warnings.warn(
            f"No proper bond between residues {res_name_i}({res_id}) and {res_name_j}({res_id+1})"
        )

        for idx, res_name in zip([res_id, res_id + 1], [res_name_i, res_name_j]):
            comp = get_component_atom_array(res_name, keep_leaving_atoms=False, keep_hydrogens=False)
            if comp is None:
                continue
            staying_atoms = comp.atom_name
            if idx == 1 and get_mol_type(res_name) in ("dna", "rna"):
                staying_atoms = np.append(staying_atoms, ["OP3"])
            if idx == atom_array.res_id[-1] and get_mol_type(res_name) == "protein":
                staying_atoms = np.append(staying_atoms, ["OXT"])
            leaving_atoms |= (atom_array.res_id == idx) & (~np.isin(atom_array.atom_name, staying_atoms))

    return atom_array[~leaving_atoms]


def _add_bonds_to_terminal_residues(atom_array: AtomArray) -> AtomArray:
    """Add bonds to terminal residues (ACE, NME, etc.).

    Args:
        atom_array: Input AtomArray.

    Returns:
        AtomArray with terminal residue bonds added.
    """
    if atom_array.res_name[0] == "ACE":
        term_res_idx = atom_array.res_id[0]
        next_res_idx = term_res_idx + 1
        term_atom_idx = np.where((atom_array.res_id == term_res_idx) & (atom_array.atom_name == "C"))[0]
        next_atom_idx = np.where((atom_array.res_id == next_res_idx) & (atom_array.atom_name == "N"))[0]
        if len(term_atom_idx) > 0 and len(next_atom_idx) > 0:
            atom_array.bonds.add_bond(term_atom_idx[0], next_atom_idx[0], 1)

    if atom_array.res_name[-1] == "NME":
        term_res_idx = atom_array.res_id[-1]
        prev_res_idx = term_res_idx - 1
        term_atom_idx = np.where((atom_array.res_id == term_res_idx) & (atom_array.atom_name == "N"))[0]
        prev_atom_idx = np.where((atom_array.res_id == prev_res_idx) & (atom_array.atom_name == "C"))[0]
        if len(prev_atom_idx) > 0 and len(term_atom_idx) > 0:
            atom_array.bonds.add_bond(prev_atom_idx[0], term_atom_idx[0], 1)

    return atom_array


def _build_polymer_atom_array(ccd_seqs: List[str]) -> AtomArray:
    """Build polymer atom array from CCD code sequence.

    Args:
        ccd_seqs: List of CCD codes for the polymer sequence.

    Returns:
        AtomArray representing the polymer.
    """
    chain = struc.AtomArray(0)
    for res_id, res_name in enumerate(ccd_seqs):
        residue = get_component_atom_array(res_name, keep_leaving_atoms=True, keep_hydrogens=False)
        if residue is None:
            logger.warning(f"Skipping unknown residue: {res_name}")
            continue
        residue.res_id[:] = res_id + 1
        chain += residue

    if len(chain) == 0:
        return chain

    res_starts = struc.get_residue_starts(chain, add_exclusive_stop=True)
    polymer_bonds = _connect_inter_residue(chain, res_starts)

    if chain.bonds is None:
        chain.bonds = polymer_bonds
    else:
        chain.bonds = chain.bonds.merge(polymer_bonds)

    chain = _add_bonds_to_terminal_residues(chain)

    bond_count = {}
    if polymer_bonds._bonds is not None and len(polymer_bonds._bonds) > 0:
        for i, j, t in polymer_bonds._bonds:
            bond_count[i] = bond_count.get(i, 0) + 1
            bond_count[j] = bond_count.get(j, 0) + 1

    chain = remove_leaving_atoms(chain, bond_count)
    chain = _remove_non_std_ccd_leaving_atoms(chain)

    return chain


def build_polymer(entity_info: Dict[str, Any]) -> Dict[str, Any]:
    """Build polymer structure.

    Args:
        entity_info: Dictionary containing polymer type and sequence info.

    Returns:
        Dictionary with atom_array key.

    Raises:
        ValueError: If polymer type is unsupported.
    """
    poly_type, info = list(entity_info.items())[0]

    if poly_type == "proteinChain":
        sequence = info.get("sequence", "")
        ccd_seqs = [PROTEIN_1TO3.get(x, "UNK") for x in sequence]

        if modifications := info.get("modifications"):
            for m in modifications:
                index = m.get("ptmPosition", m.get("position", 1)) - 1
                mtype = m.get("ptmType", m.get("type", ""))
                if mtype.startswith("CCD_") and 0 <= index < len(ccd_seqs):
                    ccd_seqs[index] = mtype[4:]

        if glycans := info.get("glycans"):
            logger.warning(f"Glycans not supported: {glycans}")

        chain_array = _build_polymer_atom_array(ccd_seqs)

    elif poly_type in ("dnaSequence", "rnaSequence"):
        sequence = info.get("sequence", "")
        map_1to3 = DNA_1TO3 if poly_type == "dnaSequence" else RNA_1TO3
        ccd_seqs = [map_1to3.get(x, "N") for x in sequence]

        if modifications := info.get("modifications"):
            for m in modifications:
                index = m.get("basePosition", m.get("position", 1)) - 1
                mtype = m.get("modificationType", m.get("type", ""))
                if mtype.startswith("CCD_") and 0 <= index < len(ccd_seqs):
                    ccd_seqs[index] = mtype[4:]

        chain_array = _build_polymer_atom_array(ccd_seqs)

    else:
        raise ValueError(f"Unsupported polymer type: {poly_type}")

    if len(chain_array) > 0:
        chain_array = add_reference_features(chain_array)

    return {"atom_array": chain_array}


def rdkit_mol_to_atom_array(mol: Chem.Mol, removeHs: bool = True) -> AtomArray:
    """Convert RDKit molecule to AtomArray.

    Args:
        mol: RDKit Mol object.
        removeHs: Whether to remove hydrogen atoms.

    Returns:
        AtomArray representation of the molecule.
    """
    atom_count = mol.GetNumAtoms()
    atom_array = AtomArray(atom_count)
    atom_array.hetero[:] = True
    atom_array.res_name[:] = "UNL"
    atom_array.add_annotation("charge", int)

    conf = mol.GetConformer()
    coord = conf.GetPositions()

    element_count = Counter()
    for i, atom in enumerate(mol.GetAtoms()):
        element = atom.GetSymbol().upper()
        element_count[element] += 1
        atom_name = f"{element}{element_count[element]}"
        atom.SetProp("name", atom_name)

        atom_array.atom_name[i] = atom_name
        atom_array.element[i] = element
        atom_array.charge[i] = atom.GetFormalCharge()
        atom_array.coord[i, :] = coord[i, :]

    bonds = []
    for bond in mol.GetBonds():
        bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    atom_array.bonds = struc.BondList(atom_count, np.array(bonds))

    if removeHs:
        atom_array = atom_array[atom_array.element != "H"]

    return atom_array


def rdkit_mol_to_atom_info(mol: Chem.Mol) -> Dict[str, Any]:
    """Convert RDKit molecule to atom information dictionary.

    Args:
        mol: RDKit Mol object.

    Returns:
        Dictionary containing atom_map_to_atom_name and atom_array.
    """
    atom_map_to_atom_name = {}
    atom_idx_to_atom_name = {}

    element_count = Counter()
    for atom in mol.GetAtoms():
        element = atom.GetSymbol().upper()
        element_count[element] += 1
        atom_name = f"{element}{element_count[element]}"
        atom.SetProp("name", atom_name)
        if atom.GetAtomMapNum() != 0:
            atom_map_to_atom_name[atom.GetAtomMapNum()] = atom_name
        atom_idx_to_atom_name[atom.GetIdx()] = atom_name

    if atom_map_to_atom_name:
        atom_info = {"atom_map_to_atom_name": atom_map_to_atom_name}
    else:
        atom_info = {"atom_map_to_atom_name": atom_idx_to_atom_name}

    atom_info["atom_array"] = rdkit_mol_to_atom_array(mol, removeHs=True)
    return atom_info


def lig_file_to_atom_info(lig_file_path: str) -> Dict[str, Any]:
    """Read atom information from ligand file.

    Args:
        lig_file_path: Path to the ligand file.

    Returns:
        Dictionary containing atom information.

    Raises:
        ValueError: If file format is unsupported or loading fails.
    """
    path = str(lig_file_path)

    if path.endswith(".mol"):
        mol = Chem.MolFromMolFile(path)
    elif path.endswith(".sdf"):
        suppl = Chem.SDMolSupplier(path)
        mol = next(suppl)
    elif path.endswith(".pdb"):
        mol = Chem.MolFromPDBFile(path)
    elif path.endswith(".mol2"):
        mol = Chem.MolFromMol2File(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    if mol is None:
        raise ValueError(f"Failed to load molecule from {path}")

    if not mol.GetConformer().Is3D():
        raise ValueError(f"No 3D conformer in {path}")

    return rdkit_mol_to_atom_info(mol)


def smiles_to_atom_info(smiles: str) -> Dict[str, Any]:
    """Build atom information from SMILES string.

    Args:
        smiles: SMILES string representation of molecule.

    Returns:
        Dictionary containing atom information.

    Raises:
        ValueError: If SMILES is invalid or conformer generation fails.
        TimeoutError: If conformer generation times out.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(AllChem.EmbedMolecule, mol)
        try:
            ret_code = future.result(timeout=90)
        except concurrent.futures.TimeoutError:
            raise TimeoutError("Conformer generation timed out")

    if ret_code != 0:
        ret_code = AllChem.EmbedMolecule(mol, useRandomCoords=True)

    if ret_code != 0:
        raise ValueError(f"Conformer generation failed for SMILES: {smiles}")

    return rdkit_mol_to_atom_info(mol)


def build_ligand(entity_info: Dict[str, Any]) -> Dict[str, Any]:
    """Build ligand or ion structure.

    Args:
        entity_info: Dictionary containing ligand or ion information.

    Returns:
        Dictionary containing atom_array.

    Raises:
        ValueError: If entity must have 'ion' or 'ligand' key.
    """
    if info := entity_info.get("ion"):
        ccd_code = [info.get("ion", "")]
    elif info := entity_info.get("ligand"):
        ligand_str = info.get("ligand", "")
        if ligand_str.startswith("CCD_"):
            ccd_code = ligand_str[4:].split("_")
        else:
            ccd_code = None
    else:
        raise ValueError("Entity must have 'ion' or 'ligand' key")

    atom_info = {}
    if ccd_code is not None:
        atom_array = struc.AtomArray(0)
        res_ids = []
        for idx, code in enumerate(ccd_code):
            ccd_atom_array = get_component_atom_array(code, keep_leaving_atoms=True, keep_hydrogens=False)
            if ccd_atom_array is None:
                logger.warning(f"CCD code not found: {code}")
                continue
            atom_array += ccd_atom_array
            res_id = idx + 1
            res_ids += [res_id] * len(ccd_atom_array)
        if len(atom_array) > 0:
            atom_array.res_id[:] = res_ids
        atom_info["atom_array"] = atom_array
    else:
        if info.get("ligand", "").startswith("FILE_"):
            lig_file_path = ligand_str[5:]
            atom_info = lig_file_to_atom_info(lig_file_path)
        else:
            atom_info = smiles_to_atom_info(ligand_str)
        atom_info["atom_array"].res_id[:] = 1

    if len(atom_info.get("atom_array", struc.AtomArray(0))) > 0:
        atom_info["atom_array"] = add_reference_features(atom_info["atom_array"])

    return atom_info


def add_entity_atom_array(single_job_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add atom_array to each entity in the job dictionary.

    Args:
        single_job_dict: Job dictionary containing sequences.

    Returns:
        Updated job dictionary with atom_array added to each entity.

    Raises:
        ValueError: If entity type is unsupported or too many SMILES ligands.
    """
    single_job_dict = copy.deepcopy(single_job_dict)
    sequences = single_job_dict.get("sequences", [])
    smiles_ligand_count = 0

    for entity_info in sequences:
        if info := entity_info.get("proteinChain"):
            atom_info = build_polymer(entity_info)
        elif info := entity_info.get("dnaSequence"):
            atom_info = build_polymer(entity_info)
        elif info := entity_info.get("rnaSequence"):
            atom_info = build_polymer(entity_info)
        elif info := entity_info.get("ligand"):
            atom_info = build_ligand(entity_info)
            if not info.get("ligand", "").startswith("CCD_"):
                smiles_ligand_count += 1
                if smiles_ligand_count > 99:
                    raise ValueError("Too many SMILES ligands (max 99)")
                atom_info["atom_array"].res_name[:] = f"l{smiles_ligand_count:02d}"
        elif info := entity_info.get("ion"):
            atom_info = build_ligand(entity_info)
        else:
            raise ValueError(
                "Entity type must be proteinChain, dnaSequence, rnaSequence, ligand or ion"
            )
        info.update(atom_info)

    return single_job_dict


# =============================================================================
# MolecularBuilder class (unified interface)
# =============================================================================

class MolecularBuilder:
    """General biological molecular structure builder.

    Provides a unified interface for building molecular structures from
    various input formats including sequences, SMILES, and CCD codes.
    """

    PROTEIN_1TO3 = PROTEIN_1TO3
    DNA_1TO3 = DNA_1TO3
    RNA_1TO3 = RNA_1TO3
    ENTITY_TYPE_MAP = ENTITY_TYPE_MAP

    @classmethod
    def set_ccd_paths(cls, components_file: str, rdkit_mol_file: str):
        """Set CCD file paths.

        Args:
            components_file: Path to the CCD components file.
            rdkit_mol_file: Path to the RDKit mol file.
        """
        set_ccd_paths(components_file, rdkit_mol_file)

    @classmethod
    def set_ccd_paths_from_configs(cls, configs: Dict[str, Any]):
        """Set CCD file paths from a configuration dictionary.

        Args:
            configs: Configuration dictionary containing either:
                - data.ccd_components_file or ccd_components_file
                - data.ccd_components_rdkit_mol_file or ccd_components_rdkit_mol_file
        """
        set_ccd_paths_from_configs(configs)

    @classmethod
    def build_from_json(cls, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build complete molecular structure from JSON description.

        Args:
            json_data: JSON dictionary containing molecular description.

        Returns:
            Dictionary with built molecular structure.
        """
        return add_entity_atom_array(copy.deepcopy(json_data))

    @classmethod
    def build_polymer(
        cls,
        sequence: str,
        polymer_type: str,
        modifications: Optional[List[Dict]] = None,
    ) -> AtomArray:
        """Build polymer structure.

        Args:
            sequence: Polymer sequence string.
            polymer_type: Type of polymer (proteinChain, dnaSequence, rnaSequence).
            modifications: Optional list of modification dictionaries.

        Returns:
            AtomArray representing the polymer.
        """
        entity_info = {polymer_type: {"sequence": sequence, "count": 1}}
        if modifications:
            if polymer_type == "proteinChain":
                entity_info[polymer_type]["modifications"] = [
                    {"ptmPosition": m.get("position", 1),
                     "ptmType": m.get("type", "")}
                    for m in modifications
                ]
            else:
                entity_info[polymer_type]["modifications"] = [
                    {"basePosition": m.get("position", 1),
                     "modificationType": m.get("type", "")}
                    for m in modifications
                ]

        result = globals()["build_polymer"](entity_info)
        return result["atom_array"]

    @classmethod
    def build_ligand(cls, ligand_id: str, ligand_type: str = "ligand") -> AtomArray:
        """Build ligand or ion structure.

        Args:
            ligand_id: Identifier for the ligand.
            ligand_type: Type of ligand ("ligand" or "ion").

        Returns:
            AtomArray representing the ligand.
        """
        entity_info = {ligand_type: {ligand_type: ligand_id, "count": 1}}
        result = globals()["build_ligand"](entity_info)
        return result["atom_array"]

    @classmethod
    def get_entity_type_mapping(cls, entity_type: str) -> str:
        """Get standard mapping for entity type.

        Args:
            entity_type: Entity type string.

        Returns:
            Standardized entity type mapping.
        """
        return ENTITY_TYPE_MAP.get(entity_type, "unknown")

    @classmethod
    def get_polymer_sequences(cls, json_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract all polymer sequences from JSON.

        Args:
            json_data: JSON dictionary containing sequences.

        Returns:
            Dictionary with categorized polymer sequences.
        """
        sequences = json_data.get("sequences", [])
        result = {"protein": [], "dna": [], "rna": [], "ligand": [], "ion": []}

        for entity_dict in sequences:
            for entity_type, entity_info in entity_dict.items():
                if entity_type == "proteinChain" and "sequence" in entity_info:
                    result["protein"].append(entity_info["sequence"])
                elif entity_type == "dnaSequence" and "sequence" in entity_info:
                    result["dna"].append(entity_info["sequence"])
                elif entity_type == "rnaSequence" and "sequence" in entity_info:
                    result["rna"].append(entity_info["sequence"])
                elif entity_type == "ligand":
                    result["ligand"].append(entity_info.get("ligand", ""))
                elif entity_type == "ion":
                    result["ion"].append(entity_info.get("ion", ""))

        return result

    @classmethod
    def validate_sequence(cls, sequence: str, seq_type: str) -> bool:
        """Validate if sequence is valid.

        Args:
            sequence: Sequence string to validate.
            seq_type: Type of sequence (proteinChain, dnaSequence, rnaSequence).

        Returns:
            True if sequence is valid, False otherwise.
        """
        if seq_type == "proteinChain":
            valid_chars = set(PROTEIN_1TO3.keys())
        elif seq_type == "dnaSequence":
            valid_chars = set(DNA_1TO3.keys())
        elif seq_type == "rnaSequence":
            valid_chars = set(RNA_1TO3.keys())
        else:
            return True
        return all(c.upper() in valid_chars for c in sequence)

    @classmethod
    def estimate_complexity(cls, json_data: Dict[str, Any]) -> Dict[str, int]:
        """Estimate molecular complexity.

        Args:
            json_data: JSON dictionary containing molecular description.

        Returns:
            Dictionary with complexity statistics.
        """
        sequences = json_data.get("sequences", [])
        stats = {
            "total_entities": len(sequences),
            "protein_chains": 0, "dna_chains": 0, "rna_chains": 0,
            "ligands": 0, "ions": 0,
            "estimated_residues": 0, "estimated_atoms": 0,
        }

        ATOMS_PER_RESIDUE = {
            "proteinChain": 7, "dnaSequence": 20, "rnaSequence": 20,
            "ligand": 20, "ion": 1
        }

        for entity_dict in sequences:
            for seq_type, seq_info in entity_dict.items():
                count = seq_info.get("count", 1)

                if seq_type == "proteinChain":
                    stats["protein_chains"] += count
                    seq_len = len(seq_info.get("sequence", ""))
                    stats["estimated_residues"] += count * seq_len
                    stats["estimated_atoms"] += count * seq_len * ATOMS_PER_RESIDUE[seq_type]
                elif seq_type in ("dnaSequence", "rnaSequence"):
                    if seq_type == "dnaSequence":
                        stats["dna_chains"] += count
                    else:
                        stats["rna_chains"] += count
                    seq_len = len(seq_info.get("sequence", ""))
                    stats["estimated_residues"] += count * seq_len
                    stats["estimated_atoms"] += count * seq_len * ATOMS_PER_RESIDUE[seq_type]
                elif seq_type == "ligand":
                    stats["ligands"] += count
                    stats["estimated_atoms"] += count * ATOMS_PER_RESIDUE[seq_type]
                elif seq_type == "ion":
                    stats["ions"] += count
                    stats["estimated_atoms"] += count * ATOMS_PER_RESIDUE[seq_type]

        return stats
