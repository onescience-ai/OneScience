# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protein data type."""
import dataclasses
from string import ascii_lowercase, ascii_uppercase
from typing import Any, Mapping, Optional

import jax
import numpy as np

from onescience.flax_models.protoken.common.residue_constants import (
    STANDARD_ATOM_MASK,
    atom_types,
    restype_1to3,
    restype_num,
    restypes,
)

alphabet_list = list(ascii_uppercase + ascii_lowercase)

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    # [num_res, num_atom_type, 3]
    atom_positions: np.ndarray

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]


def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
        prot: The protein to convert to PDB.

    Returns:
        PDB string.
    """
    restypes_ = restypes + ["X"]
    def res_1to3(r): return restype_1to3.get(
        restypes_[r], "UNK")

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors

    if np.any(aatype > restype_num):
        raise ValueError("Invalid aatypes.")

    pdb_lines.append("MODEL     1")
    atom_index = 1
    chain_id = "A"
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(
                atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            # Protein supports only C, N, O, S, this works.
            element = atom_name[0]
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_id:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the chain.
    chain_end = "TER"
    chain_termination_line = (
        f"{chain_end:<6}{atom_index:>5}      {res_1to3(aatype[-1]):>3} "
        f"{chain_id:>1}{residue_index[-1]:>4}"
    )
    pdb_lines.append(chain_termination_line)
    pdb_lines.append("ENDMDL")

    pdb_lines.append("END")
    pdb_lines.append("")
    return "\n".join(pdb_lines)


def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return STANDARD_ATOM_MASK[prot.aatype]


def from_prediction(
    features: FeatureDict, result: ModelOutput, b_factors: Optional[np.ndarray] = None
) -> Protein:
    """Assembles a protein from a prediction.

    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the protein.

    Returns:
      A protein instance.
    """
    fold_output = result["structure_module"]
    if b_factors is None:
        b_factors = np.zeros_like(
            fold_output["final_atom_mask"])

    return Protein(
        aatype=features["aatype"][0],
        atom_positions=fold_output["final_atom_positions"],
        atom_mask=fold_output["final_atom_mask"],
        residue_index=features["residue_index"][0] + 1,
        b_factors=b_factors,
    )


def renum_pdb_str(pdb_str, Ls=None, renum=True, offset=1):
    if Ls is not None:
        L_init = 0
        new_chain = {}
        for L, c in zip(Ls, alphabet_list):
            new_chain.update(
                {i: c for i in range(L_init, L_init + L)})
            L_init += L

    n, num, pdb_out = 0, offset, []
    resnum_ = None
    chain_ = None
    new_chain_ = new_chain[0]
    for line in pdb_str.split("\n"):
        if line[:4] == "ATOM":
            chain = line[21:22]
            resnum = int(line[22: 22 + 5])
        if resnum_ is None:
            resnum_ = resnum
        if chain_ is None:
            chain_ = chain
        if resnum != resnum_ or chain != chain_:
            num += resnum - resnum_
            n += 1
            resnum_, chain_ = resnum, chain
        if Ls is not None:
            if new_chain[n] != new_chain_:
                num = offset
                new_chain_ = new_chain[n]
        N = num if renum else resnum
        if Ls is None:
            pdb_out.append("%s%4i%s" %
                           (line[:22], N, line[26:]))
        else:
            pdb_out.append("%s%s%4i%s" % (
                line[:21], new_chain[n], N, line[26:]))
    return "\n".join(pdb_out)


def save_pdb_from_aux(aux, filename=None, renum_pdb=True):
    """
    save pdb coordinates (if filename provided, otherwise return as string)
    - set get_best=False, to get the last sampled sequence
    """
    aux = jax.tree_map(np.asarray, aux)
    p = {k: aux[k] for k in [
        "aatype", "residue_index", "atom_positions", "atom_mask"]}
    p["b_factors"] = 100 * p["atom_mask"] * \
        aux["plddt"][..., None]
    Ls = [len(aux["aatype"])]

    def to_pdb_str(x, n=None):
        p_str = to_pdb(Protein(**x))
        p_str = "\n".join(p_str.splitlines()[1:-2])
        if renum_pdb:
            p_str = renum_pdb_str(p_str, Ls)
        if n is not None:
            p_str = f"MODEL{n:8}\n{p_str}\nENDMDL\n"
        return p_str

    if p["atom_positions"].ndim == 4:
        if p["aatype"].ndim == 3:
            p["aatype"] = p["aatype"].argmax(-1)
        p_str = ""
        for n in range(p["atom_positions"].shape[0]):
            p_str += to_pdb_str(
                jax.tree_map(lambda x: x[n], p), n + 1)
        p_str += "END\n"
    else:
        if p["aatype"].ndim == 2:
            p["aatype"] = p["aatype"].argmax(-1)
        p_str = to_pdb_str(p)
    if filename is None:
        return p_str, Ls
    else:
        with open(filename, "w") as f:
            f.write(p_str)
