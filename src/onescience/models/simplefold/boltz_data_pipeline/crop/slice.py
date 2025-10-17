#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

# Started from https://github.com/jwohlwend/boltz, 
# licensed under MIT License, Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro. 

from dataclasses import replace
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

from boltz_data_pipeline import const
from boltz_data_pipeline.crop.cropper import Cropper
from boltz_data_pipeline.types import Tokenized


def pick_protein_token(
    tokens: np.ndarray,
) -> np.ndarray:
    """Pick protein tokens from the data.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.

    Returns
    -------
    np.ndarray
        The selected token.

    """
    protein_ids = (tokens["mol_type"] == const.chain_type_ids["PROTEIN"])
    return tokens[protein_ids]


class SliceCropper(Cropper):
    """Interpolate between contiguous and spatial crops."""

    def __init__(self,
        protein_only: bool = False,
    ) -> None:
        """Initialize the cropper.

        Modulates the type of cropping to be performed.
        Smaller neighborhoods result in more spatial
        cropping. Larger neighborhoods result in more
        continuous cropping. A mix can be achieved by
        providing a range over which to sample.

        Parameters
        ----------
        min_neighborhood : int
            The minimum neighborhood size, by default 0.
        max_neighborhood : int
            The maximum neighborhood size, by default 40.

        """
        self.protein_only = protein_only

    def crop(  # noqa: PLR0915
        self,
        data: Tokenized,
        max_tokens: int,
        random: np.random.RandomState,
        max_atoms: Optional[int] = None,
        chain_id: Optional[int] = None,
        interface_id: Optional[int] = None,
        return_indices: bool = False,
    ) -> Tokenized:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        max_tokens : int
            The maximum number of tokens to crop.
        random : np.random.RandomState
            The random state for reproducibility.
        max_atoms : int, optional
            The maximum number of atoms to consider.
        chain_id : int, optional
            The chain ID to crop.
        interface_id : int, optional
            The interface ID to crop.

        Returns
        -------
        Tokenized
            The cropped data.

        """
        # Get token data
        token_data = data.tokens
        token_bonds = data.bonds

        # Filter to resolved tokens
        valid_tokens = token_data[token_data["resolved_mask"]]

        # Check if we have any valid tokens
        if not valid_tokens.size:
            msg = "No valid tokens in structure"
            raise ValueError(msg)

        if self.protein_only:
            valid_tokens = pick_protein_token(valid_tokens)
            if not valid_tokens.size:
                msg = "No protein tokens in structure"
                raise ValueError(msg)

        if len(valid_tokens) <= max_tokens:
            # indices = np.arange(len(valid_tokens))
            cropped = set(valid_tokens["token_idx"])
        else:
            start_idx = random.randint(0, len(valid_tokens) - max_tokens)
            indices = np.arange(start_idx, start_idx + max_tokens)
            cropped = set(valid_tokens["token_idx"][indices])

        # Get the cropped tokens sorted by index
        token_data = token_data[sorted(cropped)]

        # Only keep bonds within the cropped tokens
        indices = token_data["token_idx"]
        token_bonds = token_bonds[np.isin(token_bonds["token_1"], indices)]
        token_bonds = token_bonds[np.isin(token_bonds["token_2"], indices)]

        if return_indices:
            return replace(data, tokens=token_data, bonds=token_bonds), sorted(cropped)

        # Return the cropped tokens
        return replace(data, tokens=token_data, bonds=token_bonds)
