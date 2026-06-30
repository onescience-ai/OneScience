import signal
from contextlib import contextmanager

try:
    from spyrmsd import molecule, rmsd
except ImportError as exc:
    raise ImportError(
        "spyrmsd is required for DiffDock symmetry-aware RMSD in the "
        "training inference, confidence, and evaluate paths."
    ) from exc


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def get_symmetry_rmsd(mol, coords1, coords2, mol2=None, return_permutation=False):
    with time_limit(10):
        mol = molecule.Molecule.from_rdkit(mol)
        mol2 = molecule.Molecule.from_rdkit(mol2) if mol2 is not None else mol2
        mol2_atomicnums = mol2.atomicnums if mol2 is not None else mol.atomicnums
        mol2_adjacency_matrix = mol2.adjacency_matrix if mol2 is not None else mol.adjacency_matrix
        return rmsd.symmrmsd(
            coords1,
            coords2,
            mol.atomicnums,
            mol2_atomicnums,
            mol.adjacency_matrix,
            mol2_adjacency_matrix,
            return_permutation=return_permutation,
        )
