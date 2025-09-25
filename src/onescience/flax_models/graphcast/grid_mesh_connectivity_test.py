"""Tests for graphcast.grid_mesh_connectivity."""

import numpy as np
from absl.testing import absltest

from onescience.flax_models.graphcast import grid_mesh_connectivity, icosahedral_mesh


class GridMeshConnectivityTest(absltest.TestCase):

    def test_grid_lat_lon_to_coordinates(self):

        # Intervals of 30 degrees.
        grid_latitude = np.array([-45.0, 0.0, 45])
        grid_longitude = np.array([0.0, 90.0, 180.0, 270.0])

        inv_sqrt2 = 1 / np.sqrt(2)
        expected_coordinates = np.array(
            [
                [
                    [inv_sqrt2, 0.0, -inv_sqrt2],
                    [0.0, inv_sqrt2, -inv_sqrt2],
                    [-inv_sqrt2, 0.0, -inv_sqrt2],
                    [0.0, -inv_sqrt2, -inv_sqrt2],
                ],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                [
                    [inv_sqrt2, 0.0, inv_sqrt2],
                    [0.0, inv_sqrt2, inv_sqrt2],
                    [-inv_sqrt2, 0.0, inv_sqrt2],
                    [0.0, -inv_sqrt2, inv_sqrt2],
                ],
            ]
        )

        coordinates = grid_mesh_connectivity._grid_lat_lon_to_coordinates(
            grid_latitude, grid_longitude
        )
        np.testing.assert_allclose(expected_coordinates, coordinates, atol=1e-15)

    def test_radius_query_indices_smoke(self):
        # TODO(alvarosg): Add non-smoke test?
        grid_latitude = np.linspace(-75, 75, 6)
        grid_longitude = np.arange(12) * 30.0
        mesh = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=3)[
            -1
        ]
        grid_mesh_connectivity.radius_query_indices(
            grid_latitude=grid_latitude,
            grid_longitude=grid_longitude,
            mesh=mesh,
            radius=0.2,
        )

    def test_in_mesh_triangle_indices_smoke(self):
        # TODO(alvarosg): Add non-smoke test?
        grid_latitude = np.linspace(-75, 75, 6)
        grid_longitude = np.arange(12) * 30.0
        mesh = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=3)[
            -1
        ]
        grid_mesh_connectivity.in_mesh_triangle_indices(
            grid_latitude=grid_latitude, grid_longitude=grid_longitude, mesh=mesh
        )


if __name__ == "__main__":
    absltest.main()
