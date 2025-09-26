from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse
import torch
from dgl import DGLGraph
from torch.utils.data import Dataset

try:
    from sparse_dot_mkl import dot_product_mkl
except ImportError:
    import warnings

    warnings.warn(
        "sparse_dot_mkl is not installed, install using: pip install sparse_dot_mkl"
    )


_INF = 1 + 1e10


class BistrideMultiLayerGraphDataset(Dataset):
    """Wrapper over graph dataset that enables multi-layer graphs."""

    def __init__(
        self,
        dataset: Dataset,
        num_layers: int = 1,
        cache_dir: Optional[str | Path] = None,
        **kwargs,
    ):
        self.dataset = dataset
        self.num_layers = num_layers
        if cache_dir is None:
            self.cache_dir = None
        else:
            self.cache_dir = Path(
                cache_dir) / self.dataset.split
            self.cache_dir.mkdir(
                parents=True, exist_ok=True)

    def __getitem__(self, idx):
        graph = self.dataset[idx]
        # Check if MS graph is already in the cache.
        edges_and_ids = None
        if self.cache_dir is not None:
            edges_and_ids = self._load_from_cache(idx)
        if edges_and_ids is None:
            ms_graph = BistrideMultiLayerGraph(
                graph, self.num_layers)
            _, *edges_and_ids = ms_graph.get_multi_layer_graphs()

            if self.cache_dir is not None:
                self._save_to_cache(idx, edges_and_ids)
        ms_edges, ms_ids = edges_and_ids

        return {
            "graph": graph,
            "ms_edges": [torch.tensor(e, dtype=torch.long) for e in ms_edges],
            "ms_ids": [torch.tensor(ids, dtype=torch.long) for ids in ms_ids],
        }

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def _get_cache_filename(self, idx: int) -> str:
        return f"{idx:03}.cache"

    def _load_from_cache(self, idx: int) -> tuple[list, list]:
        if self.cache_dir is None or not self.cache_dir.is_dir():
            raise ValueError(
                "Cache directory is not set or does not exist.")

        filename = self.cache_dir / \
            (self._get_cache_filename(idx) + ".npz")
        if not filename.exists():
            return None
        return np.load(filename, allow_pickle=True)["edges_and_ids"]

    def _save_to_cache(self, idx: int, edges_and_ids: tuple[list, list]) -> None:
        if self.cache_dir is None or not self.cache_dir.is_dir():
            raise ValueError(
                "Cache directory is not set or does not exist.")

        filename = self.cache_dir / \
            self._get_cache_filename(idx)
        return np.savez(
            filename,
            edges_and_ids=np.asanyarray(
                edges_and_ids, dtype=object),
        )


class BistrideMultiLayerGraph:
    """Multi-layer graph."""

    def __init__(self, graph: DGLGraph, num_layers: int):
        """
        Initializes the BistrideMultiLayerGraph object.

        Parameters
        ----------
        graph: DGLGraph
            The source graph.
        num_layers: int:
            The number of layers to generate.
        """
        self.num_nodes = graph.num_nodes()
        self.num_layers = num_layers
        self.pos_mesh = graph.ndata["pos"].numpy()

        # Initialize the first layer graph
        # Flatten edges to [2, num_edges].
        edges = graph.edges()
        flattened_edges = torch.cat(
            (edges[0].view(1, -1), edges[1].view(1, -1)), dim=0
        ).numpy()
        self.m_gs = [
            Graph(flattened_edges, GraphType.FLAT_EDGE, self.num_nodes)]
        self.m_flat_es = [self.m_gs[0].get_flat_edge()]
        self.m_ids = []

        self.generate_multi_layer_graphs()

    def generate_multi_layer_graphs(self):
        """
        Generate multiple layers of graphs with pooling.
        """
        g_l = self.m_gs[0]
        pos_l = self.pos_mesh

        index_to_keep = []
        for layer in range(self.num_layers):
            n_l = self.num_nodes if layer == 0 else len(
                index_to_keep)
            index_to_keep, g_l = self.bstride_selection(
                g_l, pos_l, n_l)
            pos_l = pos_l[index_to_keep]
            self.m_gs.append(g_l)
            self.m_flat_es.append(g_l.get_flat_edge())
            self.m_ids.append(index_to_keep)

    def get_multi_layer_graphs(self):
        """
        Get the multi-layer graph structures.

        Returns:
        tuple: A tuple containing three lists:
            - m_gs (list): List of graph wrappers for each layer.
            - m_flat_es (list): List of flat edges for each layer.
            - m_ids (list): List of node indices to be pooled at each layer.
        """
        return self.m_gs, self.m_flat_es, self.m_ids

    @staticmethod
    def bstride_selection(g, pos_mesh, n):
        """
        Perform bstride selection to pool nodes and edges.

        Parameters:
        g (Graph): The graph wrapper object.
        pos_mesh (np.ndarray): The positions of the nodes in the mesh.
        n (int): The number of nodes.

        Returns:
        tuple: A tuple containing:
            - combined_idx_kept (list): List of node indices to be pooled.
            - new_g (Graph): The new graph wrapper object after pooling.
        """
        combined_idx_kept = set()
        adj_mat = g.get_sparse_adj_mat()
        adj_mat.setdiag(1)
        clusters = g.clusters

        seeds = BistrideMultiLayerGraph.nearest_center_seed(
            pos_mesh, clusters)

        for seed, c in zip(seeds, clusters):
            even, odd = set(), set()
            dist_from_central_node = g.bfs_dist(seed)

            for i, dist in enumerate(dist_from_central_node):
                if dist % 2 == 0 and dist != _INF:
                    even.add(i)
                elif dist % 2 == 1 and dist != _INF:
                    odd.add(i)

            if len(even) <= len(odd) or not odd:
                index_kept, index_rmvd = even, odd  # noqa: F841 for clarity
            else:
                index_kept, index_rmvd = odd, even  # noqa: F841 for clarity

            combined_idx_kept = combined_idx_kept.union(
                index_kept)

        combined_idx_kept = list(combined_idx_kept)
        combined_idx_kept.sort()
        adj_mat = adj_mat.tocsr().astype(float)
        adj_mat = dot_product_mkl(adj_mat, adj_mat)
        adj_mat.setdiag(0)
        new_g = BistrideMultiLayerGraph.pool_edge(
            adj_mat, n, combined_idx_kept)

        return combined_idx_kept, new_g

    @staticmethod
    def nearest_center_seed(pos_mesh, clusters):
        """
        Find the nearest center seed for each cluster.

        Parameters:
        pos_mesh (np.ndarray): The positions of the nodes in the mesh.
        clusters (list): List of clusters, each cluster is a list of node indices.

        Returns:
        list: List of seeds per cluster.
        """
        seeds = []
        for c in clusters:
            center = np.mean(pos_mesh[c], axis=0)
            delta_to_center = pos_mesh[c] - center[None, :]
            dist_to_center = np.linalg.norm(
                delta_to_center, 2, axis=-1)
            min_node = c[np.argmin(dist_to_center)]
            seeds.append(min_node)

        return seeds

    @staticmethod
    def pool_edge(adj_mat, num_nodes, idx):
        """
        Pool the edges based on the provided node indices.

        Parameters:
        adj_mat (scipy.sparse.csr_matrix): The adjacency matrix in CSR format.
        num_nodes (int): The number of nodes in the input graph.
        idx (list): List of node indices to be kept.

        Returns:
        Graph: The new graph wrapper object after pooling.
        """
        flat_e = Graph.adj_mat_to_flat_edge(adj_mat)
        idx = np.array(idx, dtype=np.int64)
        idx_new_valid = np.arange(len(idx)).astype(np.int64)
        idx_new_all = -1 * \
            np.ones(num_nodes).astype(np.int64)
        idx_new_all[idx] = idx_new_valid
        new_flat_e = -1 * \
            np.ones_like(flat_e).astype(np.int64)
        new_flat_e[0] = idx_new_all[flat_e[0]]
        new_flat_e[1] = idx_new_all[flat_e[1]]
        both_valid = np.logical_and(
            new_flat_e[0] >= 0, new_flat_e[1] >= 0)
        e_idx = np.where(both_valid)[0]
        new_flat_e = new_flat_e[:, e_idx]
        new_g = Graph(
            new_flat_e, GraphType.FLAT_EDGE, len(idx))

        return new_g


class GraphType(Enum):
    """
    Enumeration to define the types of graph representations.
    """

    FLAT_EDGE = 1
    ADJ_LIST = 2
    ADJ_MAT = 3


class Graph:
    """Convenience graph class."""

    def __init__(self, g, g_type, num_nodes):
        """
        Initialize the Graph object.

        Parameters:
        g (np.ndarray, list, or scipy.sparse.coo_matrix): The graph data in the specified format.
        g_type (GraphType): The type of the input graph representation.
        num_nodes (int): The number of nodes in the graph.
        """
        self.num_nodes = num_nodes

        if g_type == GraphType.FLAT_EDGE:
            self.flat_edges = g
        elif g_type == GraphType.ADJ_LIST:
            self.flat_edges = self.adj_list_to_flat_edge(g)
        elif g_type == GraphType.ADJ_MAT:
            self.flat_edges = self.adj_mat_to_flat_edge(g)
        else:
            raise ValueError(
                f"Unknown graph type: {g_type}")

        self.adj_list = self.get_adj_list()
        self.clusters = self.find_clusters()

    def get_flat_edge(self):
        """
        Get the flat edge representation of the graph.

        Returns:
        np.ndarray: The flat edge representation of the graph.
        """
        return self.flat_edges

    def get_adj_list(self):
        """
        Get the adjacency list representation of the graph.

        Returns:
        list: The adjacency list representation of the graph.
        """
        return self.flat_edge_to_adj_list(self.flat_edges, self.num_nodes)

    def get_sparse_adj_mat(self):
        """
        Get the sparse adjacency matrix representation of the graph.

        Returns:
        scipy.sparse.coo_matrix: The sparse adjacency matrix representation of the graph.
        """
        return self.flat_edge_to_adj_mat(self.flat_edges, self.num_nodes)

    def bfs_dist(self, seed):
        """
        Perform a Breadth-First Search (BFS) to find the shortest distance from the seed to all other nodes.

        Parameters:
        seed (int or list): The starting node(s) for BFS.

        Returns:
        np.ndarray: The shortest distance from the seed to all other nodes.
        """
        _INF = 1 + 1e10
        res = np.ones(self.num_nodes) * _INF
        visited = [False for _ in range(self.num_nodes)]
        if isinstance(seed, list):
            for s in seed:
                res[s] = 0
                visited[s] = True
            frontier = seed
        else:
            res[seed] = 0
            visited[seed] = True
            frontier = [seed]

        depth = 0
        track = [frontier]
        while frontier:
            this_level = frontier
            depth += 1
            frontier = []
            while this_level:
                f = this_level.pop(0)
                for n in self.adj_list[f]:
                    if not visited[n]:
                        visited[n] = True
                        frontier.append(n)
                        res[n] = depth
            track.append(frontier)

        return res

    def find_clusters(self):
        """
        Find connected clusters in the graph using BFS.

        Returns:
        list: A list of clusters, each cluster is a list of node indices.
        """
        _INF = 1 + 1e10
        remaining_nodes = list(range(self.num_nodes))
        clusters = []
        while remaining_nodes:
            if len(remaining_nodes) > 1:
                seed = remaining_nodes[0]
                dist = self.bfs_dist(seed)
                tmp = []
                new_remaining = []
                for n in remaining_nodes:
                    if dist[n] != _INF:
                        tmp.append(n)
                    else:
                        new_remaining.append(n)
                clusters.append(tmp)
                remaining_nodes = new_remaining
            else:
                clusters.append([remaining_nodes[0]])
                break

        return clusters

    @staticmethod
    def flat_edge_to_adj_mat(edge_list, n):
        """
        Convert a flat edge list to a sparse adjacency matrix.

        Parameters:
        edge_list (np.ndarray): The flat edge list of shape [2, num_edges].
        n (int): The number of nodes.

        Returns:
        scipy.sparse.coo_matrix: The sparse adjacency matrix.
        """
        adj_mat = scipy.sparse.coo_matrix(
            (np.ones_like(edge_list[0]), (edge_list[0], edge_list[1])), shape=(n, n)
        )
        return adj_mat

    @staticmethod
    def flat_edge_to_adj_list(edge_list, n):
        """
        Convert a flat edge list to an adjacency list.

        Parameters:
        edge_list (np.ndarray): The flat edge list of shape [2, num_edges].
        n (int): The number of nodes.

        Returns:
        list: The adjacency list.
        """
        adj_list = [[] for _ in range(n)]
        for i in range(len(edge_list[0])):
            adj_list[edge_list[0, i]].append(
                edge_list[1, i])
        return adj_list

    @staticmethod
    def adj_list_to_flat_edge(adj_list):
        """
        Convert an adjacency list to a flat edge list.

        Parameters:
        adj_list (list): The adjacency list.

        Returns:
        np.ndarray: The flat edge list of shape [2, num_edges].
        """
        edge_list = []
        for i in range(len(adj_list)):
            for n in adj_list[i]:
                edge_list.append(  # noqa: PERF401 list comprehension makes the code less clear.
                    [i, n]
                )
        return np.array(edge_list).transpose()

    @staticmethod
    def adj_mat_to_flat_edge(adj_mat):
        """
        Convert a sparse adjacency matrix to a flat edge list.

        Parameters:
        adj_mat (np.ndarray or scipy.sparse.spmatrix): The sparse adjacency matrix.

        Returns:
        np.ndarray: The flat edge list of shape [2, num_edges].
        """
        if isinstance(adj_mat, np.ndarray):
            s, r = np.where(adj_mat.astype(bool))
        elif isinstance(adj_mat, scipy.sparse.coo_matrix):
            s, r = adj_mat.row, adj_mat.col
            dat = adj_mat.data
            valid = np.where(dat.astype(bool))[0]
            s, r = s[valid], r[valid]
        elif isinstance(adj_mat, scipy.sparse.csr_matrix):
            adj_mat = scipy.sparse.coo_matrix(adj_mat)
            s, r = adj_mat.row, adj_mat.col
            dat = adj_mat.data
            valid = np.where(dat.astype(bool))[0]
            s, r = s[valid], r[valid]
        else:
            raise ValueError(
                "Unsupported adjacency matrix type in adj_mat_to_flat_edge. Now only support numpy.ndarray, scipy.sparse.coo_matrix, scipy.sparse.csr_matrix."
            )
        return np.array([s, r])
