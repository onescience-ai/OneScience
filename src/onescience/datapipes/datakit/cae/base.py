
import os
from typing import Any

import torch
import vtk

Tensor = torch.Tensor


def read_vtp(file_path: str) -> Any:  # TODO add support for older format (VTK)
    """
    Read a VTP file and return the polydata.

    Parameters
    ----------
    file_path : str
        Path to the VTP file.

    Returns
    -------
    vtkPolyData
        The polydata read from the VTP file.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    # Check if file has .vtp extension
    if not file_path.endswith(".vtp"):
        raise ValueError(f"Expected a .vtp file, got {file_path}")

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()

    # Get the polydata
    polydata = reader.GetOutput()

    # Check if polydata is valid
    if polydata is None:
        raise ValueError(f"Failed to read polydata from {file_path}")

    return polydata


def read_vtu(file_path: str) -> Any:
    """
    Read a VTU file and return the unstructured grid data.

    Parameters
    ----------
    file_path : str
        Path to the VTU file.

    Returns
    -------
    vtkUnstructuredGrid
        The unstructured grid data read from the VTU file.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    # Check if file has .vtu extension
    if not file_path.endswith(".vtu"):
        raise ValueError(f"Expected a .vtu file, got {file_path}")

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()

    # Get the unstructured grid data
    grid = reader.GetOutput()

    # Check if grid is valid
    if grid is None:
        raise ValueError(f"Failed to read unstructured grid data from {file_path}")

    return grid


def read_cgns(file_path: str) -> Any:
    """
    Read a CGNS file and return the unstructured grid data.

    Parameters
    ----------
    file_path : str
        Path to the CGNS file.

    Returns
    -------
    vtkUnstructuredGrid
        The unstructured grid data read from the CGNS file.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    # Check if file has .cgns extension
    if not file_path.endswith(".cgns"):
        raise ValueError(f"Expected a .cgns file, got {file_path}")

    reader = vtk.vtkCGNSReader()
    reader.SetFileName(file_path)
    reader.Update()

    # Get the multi-block dataset
    multi_block = reader.GetOutput()

    # Check if the multi-block dataset is valid
    if multi_block is None:
        raise ValueError(f"Failed to read multi-block data from {file_path}")

    # Extract and return the vtkUnstructuredGrid from the multi-block dataset
    return _extract_unstructured_grid(multi_block)


def _extract_unstructured_grid(
    multi_block: vtk.vtkMultiBlockDataSet,
) -> vtk.vtkUnstructuredGrid:
    """
    Extracts a vtkUnstructuredGrid from a vtkMultiBlockDataSet.

    Parameters
    ----------
    multi_block : vtk.vtkMultiBlockDataSet
        The multi-block dataset containing various data blocks.

    Returns
    -------
    vtk.vtkUnstructuredGrid
        The unstructured grid extracted from the multi-block dataset.
    """
    block = multi_block.GetBlock(0).GetBlock(0)
    if isinstance(block, vtk.vtkUnstructuredGrid):
        return block
    raise ValueError("No vtkUnstructuredGrid found in the vtkMultiBlockDataSet.")
