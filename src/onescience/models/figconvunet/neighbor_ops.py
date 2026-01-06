from typing import Literal, Union

import torch
from jaxtyping import Float, Int
from torch import Tensor

@torch.no_grad()
def _knn_search(
    ref_positions: Int[Tensor, "N 3"],
    query_positions: Int[Tensor, "M 3"],
    k: int,
) -> Int[Tensor, "M K"]:
    """Perform knn search using the open3d backend."""
    assert k > 0
    assert k < ref_positions.shape[0]
    assert ref_positions.device == query_positions.device
    # Critical for multi GPU
    if ref_positions.is_cuda:
        torch.cuda.set_device(ref_positions.device)
    # Use topk to get the top k indices from distances
    dists = torch.cdist(query_positions, ref_positions)
    _, neighbors_index = torch.topk(dists, k, dim=1, largest=False)
    return neighbors_index


@torch.no_grad()
def _chunked_knn_search(
    ref_positions: Int[Tensor, "N 3"],
    query_positions: Int[Tensor, "M 3"],
    k: int,
    chunk_size: int = 4096,
):
    """Divide the out_positions into chunks and perform knn search."""
    assert k > 0
    assert k < ref_positions.shape[0]
    assert chunk_size > 0
    neighbors_index = []
    for i in range(0, query_positions.shape[0], chunk_size):
        chunk_out_positions = query_positions[i : i + chunk_size]
        chunk_neighbors_index = _knn_search(ref_positions, chunk_out_positions, k)
        neighbors_index.append(chunk_neighbors_index)
    return torch.concatenate(neighbors_index, dim=0)


@torch.no_grad()
def neighbor_knn_search(
    ref_positions: Int[Tensor, "N 3"],
    query_positions: Int[Tensor, "M 3"],
    k: int,
    search_method: Literal["chunk"] = "chunk",
    chunk_size: int = 32768,  # 2^15
) -> Int[Tensor, "M K"]:
    """
    ref_positions: [N,3]
    query_positions: [M,3]
    k: int
    """
    assert 0 < k < ref_positions.shape[0]
    assert search_method in ["chunk"]
    # Critical for multi GPU
    if ref_positions.is_cuda:
        torch.cuda.set_device(ref_positions.device)
    assert ref_positions.device == query_positions.device
    if search_method == "chunk":
        if query_positions.shape[0] < chunk_size:
            neighbors_index = _knn_search(ref_positions, query_positions, k)
        else:
            neighbors_index = _chunked_knn_search(
                ref_positions, query_positions, k, chunk_size=chunk_size
            )
    else:
        raise ValueError(f"search_method {search_method} not supported.")
    return neighbors_index


@torch.no_grad()
def batched_neighbor_knn_search(
    ref_positions: Int[Tensor, "B N 3"],
    query_positions: Int[Tensor, "B M 3"],
    k: int,
    search_method: Literal["chunk"] = "chunk",
    chunk_size: int = 4096,
) -> Int[Tensor, "B M K"]:
    """
    ref_positions: [B,N,3]
    query_positions: [B,M,3]
    k: int
    """
    assert (
        ref_positions.shape[0] == query_positions.shape[0]
    ), f"Batch size mismatch, {ref_positions.shape[0]} != {query_positions.shape[0]}"
    neighbors = []
    index_offset = 0
    for i in range(ref_positions.shape[0]):
        neighbor_index = neighbor_knn_search(
            ref_positions[i], query_positions[i], k, search_method, chunk_size
        )
        neighbors.append(neighbor_index + index_offset)
        index_offset += ref_positions.shape[1]
    return torch.stack(neighbors, dim=0)
