import math

import torch


def gen_data(design_domain, steps_x, steps_y):
    x_values = torch.linspace(
        design_domain["x"][0], design_domain["x"][1], steps=steps_x
    )
    y_values = torch.linspace(
        design_domain["y"][0], design_domain["y"][1], steps=steps_y
    )
    X, Y = torch.meshgrid(x_values, y_values, indexing="ij")
    X_col_domain = torch.stack([X.flatten(), Y.flatten()], dim=1)
    return X_col_domain


def get_data_fluid(problem="rugby", N_col_domain=10000, N_train=25.0, ratio=1.0):

    domain = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
    points_x = torch.linspace(domain["x"][0], domain["x"][1], N_train + 2)[1:-1]
    points_y = torch.linspace(domain["y"][0], domain["y"][1], N_train + 2)[1:-1]

    if problem == "pipebend":
        N_train_r = int(torch.floor(torch.tensor(N_train / 5)).item())  #
        N_train_b = int(torch.floor(torch.tensor(N_train / 5)).item())  #

        # Points excluding the overlapping region [0.7, 0.9] for the left boundary
        points_y_filtered = points_y[(points_y < 0.7) | (points_y > 0.9)]
        x_left = torch.stack(
            [
                domain["x"][0] * torch.ones(len(points_y_filtered)),
                points_y_filtered.squeeze(),
            ],
            dim=1,
        )
        u_left = torch.zeros_like(x_left[:, 0])
        v_left = torch.zeros_like(x_left[:, 0])
        ro_left = torch.zeros_like(x_left[:, 0])

        # New domain and points for the left boundary
        domain_r = {"x": [0.0, 0.0], "y": [0.7, 0.9]}
        points_y_r = torch.linspace(domain_r["y"][0], domain_r["y"][1], N_train_r + 2)[
            1:-1
        ]
        x_left_r = torch.stack(
            [domain_r["x"][1] * torch.ones(N_train_r), points_y_r.squeeze()], dim=1
        )

        # Parameters for the velocity profile for the left boundary
        l = domain_r["y"][1] - domain_r["y"][0]
        t = points_y_r - (domain_r["y"][0] + l / 2)
        g_bar = (
            1.0  # Example value for the magnitude of the flow velocity at the center
        )

        # Compute the velocity profile for the left boundary
        u_left_r = g_bar * (1 - (2 * t / l) ** 2)
        v_left_r = torch.zeros_like(u_left_r)
        ro_left_r = 1 + torch.zeros_like(u_left_r)

        # Concatenate the new samples for the left boundary
        x_left_combined = torch.cat((x_left, x_left_r), dim=0)
        u_left_combined = torch.cat((u_left, u_left_r), dim=0)
        v_left_combined = torch.cat((v_left, v_left_r), dim=0)
        ro_left_combined = torch.cat((ro_left, ro_left_r), dim=0)

        # Original domain and points for the bottom boundary
        points_x_filtered = points_x[(points_x < 0.7) | (points_x > 0.9)]
        x_bottom = torch.stack(
            [points_x_filtered.squeeze(), torch.zeros(len(points_x_filtered))], dim=1
        )
        u_bottom = torch.zeros_like(x_bottom[:, 0])
        v_bottom = torch.zeros_like(x_bottom[:, 0])
        ro_bottom = torch.zeros_like(x_bottom[:, 0])

        # New domain and points for the bottom boundary
        domain_b = {"x": [0.7, 0.9], "y": [0, 0]}
        points_x_b = torch.linspace(domain_b["x"][0], domain_b["x"][1], N_train_b + 2)[
            1:-1
        ]
        x_bottom_b = torch.stack([points_x_b.squeeze(), torch.zeros(N_train_b)], dim=1)

        # Parameters for the velocity profile for the bottom boundary
        l_b = domain_b["x"][1] - domain_b["x"][0]
        t_b = points_x_b - (domain_b["x"][0] + l_b / 2)

        # Compute the velocity profile for the bottom boundary
        v_bar = (
            1.0  # Example value for the magnitude of the flow velocity at the center
        )
        v_bottom_b = -v_bar * (1 - (2 * t_b / l_b) ** 2)
        u_bottom_b = torch.zeros_like(v_bottom_b)
        ro_bottom_b = 1 + torch.zeros_like(v_bottom_b)

        # Concatenate the new samples for the bottom boundary
        x_bottom_combined = torch.cat((x_bottom, x_bottom_b), dim=0)
        u_bottom_combined = torch.cat((u_bottom, u_bottom_b), dim=0)
        v_bottom_combined = torch.cat((v_bottom, v_bottom_b), dim=0)
        ro_bottom_combined = torch.cat((ro_bottom, ro_bottom_b), dim=0)

        # The combined tensors now contain the samples from both domains without overlap

        x_top = torch.stack([points_x.squeeze(), torch.ones(N_train)], dim=1)
        u_top = torch.zeros_like(x_top[:, 0])  # 5.0*torch.sin(x_top[:,0]*torch.pi)
        v_top = torch.zeros_like(x_top[:, 0])
        ro_top = torch.zeros_like(x_top[:, 0])

        points_y = torch.linspace(domain["y"][0], domain["y"][1], N_train + 2)[1:-1]
        x_right = torch.stack([torch.ones(N_train), points_y.squeeze()], dim=1)
        u_right = torch.zeros_like(x_right[:, 0])
        v_right = torch.zeros_like(x_right[:, 0])
        ro_right = torch.zeros_like(x_right[:, 0])

        # Concatenate the points from all sides to form the boundary tensor
        x_corners = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        u_corners = torch.zeros_like(x_corners[:, 0])
        v_corners = torch.zeros_like(x_corners[:, 0])
        ro_corners = torch.zeros_like(x_corners[:, 0])

        X_train = torch.cat(
            [x_top, x_left_combined, x_bottom_combined, x_right, x_corners], dim=0
        )
        U_train = torch.cat(
            [u_top, u_left_combined, u_bottom_combined, u_right, u_corners], dim=0
        )
        V_train = torch.cat(
            [v_top, v_left_combined, v_bottom_combined, v_right, v_corners], dim=0
        )
        train_ro = torch.cat(
            [ro_top, ro_left_combined, ro_bottom_combined, ro_right, ro_corners], dim=0
        )

    elif problem == "diffuser":
        N_train_r = N_train // 3  # Adjust N_train_r as needed for the new domain
        N_train_l = N_train  # Adjust N_train_b as needed for the new domain

        # New domain and points for the left boundary
        domain_l = {"x": [0.0, 0.0], "y": [0, 1]}
        points_y_r = torch.linspace(domain_l["y"][0], domain_l["y"][1], N_train_l + 2)[
            1:-1
        ]
        x_left_r = torch.stack(
            [domain_l["x"][1] * torch.ones(N_train_l), points_y_r.squeeze()], dim=1
        )

        # Parameters for the velocity profile for the left boundary
        l = domain_l["y"][1] - domain_l["y"][0]
        t = points_y_r - (domain_l["y"][0] + l / 2)
        g_bar = (
            1.0  # Example value for the magnitude of the flow velocity at the center
        )

        # Compute the velocity profile for the left boundary
        u_left_r = g_bar * (1 - (2 * t / l) ** 2)
        v_left_r = torch.zeros_like(u_left_r)
        ro_left_r = 1 + torch.zeros_like(u_left_r)

        # Original domain and points for the bottom boundary
        x_bottom = torch.stack([points_x.squeeze(), torch.zeros(len(points_x))], dim=1)
        u_bottom = torch.zeros_like(x_bottom[:, 0])
        v_bottom = torch.zeros_like(x_bottom[:, 0])
        ro_bottom = torch.zeros_like(x_bottom[:, 0])

        # The combined tensors now contain the samples from both domains without overlap

        x_top = torch.stack([points_x.squeeze(), torch.ones(N_train)], dim=1)
        u_top = torch.zeros_like(x_top[:, 0])  # 5.0*torch.sin(x_top[:,0]*torch.pi)
        v_top = torch.zeros_like(x_top[:, 0])
        ro_top = torch.zeros_like(x_top[:, 0])

        points_y_filtered = points_y[(points_y < 0.333) | (points_y > 0.666)]
        x_right = torch.stack(
            [
                domain["x"][1] * torch.ones(len(points_y_filtered)),
                points_y_filtered.squeeze(),
            ],
            dim=1,
        )
        u_right = torch.zeros_like(x_right[:, 0])
        v_right = torch.zeros_like(x_right[:, 0])
        ro_right = torch.zeros_like(x_right[:, 0])

        # New domain and points for the left boundary
        domain_r = {"x": [1, 1], "y": [0.333, 0.666]}
        points_y_l = torch.linspace(domain_r["y"][0], domain_r["y"][1], N_train_r + 2)[
            1:-1
        ]
        x_right_l = torch.stack(
            [domain_r["x"][1] * torch.ones(N_train_r), points_y_l.squeeze()], dim=1
        )

        # Parameters for the velocity profile for the left boundary
        l = domain_r["y"][1] - domain_r["y"][0]
        t = points_y_l - (domain_r["y"][0] + l / 2)
        g_bar = (
            3.0  # Example value for the magnitude of the flow velocity at the center
        )

        # Compute the velocity profile for the left boundary
        u_right_l = g_bar * (1 - (2 * t / l) ** 2)
        v_right_l = torch.zeros_like(u_right_l)
        ro_right_l = 1 + torch.zeros_like(u_right_l)

        # Concatenate the new samples for the left boundary
        x_right_combined = torch.cat((x_right, x_right_l), dim=0)
        u_right_combined = torch.cat((u_right, u_right_l), dim=0)
        v_right_combined = torch.cat((v_right, v_right_l), dim=0)
        ro_right_combined = torch.cat((ro_right, ro_right_l), dim=0)

        # Concatenate the points from all sides to form the boundary tensor
        x_corners = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        u_corners = torch.zeros_like(x_corners[:, 0])
        v_corners = torch.zeros_like(x_corners[:, 0])
        ro_corners = torch.zeros_like(x_corners[:, 0])

        X_train = torch.cat(
            [x_top, x_left_r, x_bottom, x_right_combined, x_corners], dim=0
        )
        U_train = torch.cat(
            [u_top, u_left_r, u_bottom, u_right_combined, u_corners], dim=0
        )
        V_train = torch.cat(
            [v_top, v_left_r, v_bottom, v_right_combined, v_corners], dim=0
        )
        train_ro = torch.cat(
            [ro_top, ro_left_r, ro_bottom, ro_right_combined, ro_corners], dim=0
        )

    elif problem == "rugby":

        x_bottom = torch.stack([points_x.squeeze(), torch.zeros(N_train)], dim=1)
        u_bottom = 1 + torch.zeros_like(x_bottom[:, 0])
        v_bottom = torch.zeros_like(x_bottom[:, 0])

        x_top = torch.stack([points_x.squeeze(), torch.ones(N_train)], dim=1)
        u_top = 1 + torch.zeros_like(x_top[:, 0])  # 5.0*torch.sin(x_top[:,0]*torch.pi)
        v_top = torch.zeros_like(x_top[:, 0])

        x_right = torch.stack(
            [domain["x"][1] * torch.ones(N_train), points_y.squeeze()], dim=1
        )
        u_right = 1 + torch.zeros_like(x_right[:, 0])
        v_right = torch.zeros_like(x_right[:, 0])

        x_left = torch.stack([torch.zeros(N_train), points_y.squeeze()], dim=1)
        u_left = 1 + torch.zeros_like(x_left[:, 0])
        v_left = torch.zeros_like(x_left[:, 0])

        # Concatenate the points from all sides to form the boundary tensor
        x_corners = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        u_corners = 1 + torch.zeros_like(x_corners[:, 0])
        v_corners = torch.zeros_like(x_corners[:, 0])

        X_train = torch.cat([x_top, x_right, x_bottom, x_left, x_corners], dim=0)
        U_train = torch.cat([u_top, u_right, u_bottom, u_left, u_corners], dim=0)
        V_train = torch.cat([v_top, v_right, v_bottom, v_left, v_corners], dim=0)

        X_train = torch.cat((X_train, torch.tensor([0.5, 0.5]).unsqueeze(0)), dim=0)
        X_train_U = X_train
        X_train_V = X_train
        train_ro = torch.cat((1 + 0 * V_train.clone(), torch.tensor([0])), dim=0)
        U_train = torch.cat((U_train, torch.tensor([0])), dim=0)
        V_train = torch.cat((V_train, torch.tensor([0])), dim=0)

    elif problem == "doublepipe":

        # Define the number of samples
        N_train_r = math.floor(N_train / 3)  # For the new domain

        # Define the exclusion ranges
        exclude_range1_min = 1 / 4 - 1 / 12
        exclude_range1_max = 1 / 4 + 1 / 12
        exclude_range2_min = 0.666
        exclude_range2_max = 0.666 + 1 / 6

        domain_r1 = {"x": [0.0, ratio], "y": [exclude_range1_min, exclude_range1_max]}
        domain_r2 = {"x": [0.0, ratio], "y": [exclude_range2_min, exclude_range2_max]}

        # Generate points excluding the overlapping region for the left boundary

        # Create boolean masks to exclude ranges
        mask1 = (points_y < exclude_range1_min) | (points_y > exclude_range1_max)
        mask2 = (points_y < exclude_range2_min) | (points_y > exclude_range2_max)
        mask = mask1 & mask2

        # Apply mask to points_y
        points_y_filtered = points_y[mask]

        # Create x_left and related tensors
        x_left = torch.stack(
            [
                domain["x"][0] * torch.ones(len(points_y_filtered)),
                points_y_filtered.squeeze(),
            ],
            dim=1,
        )
        u_left = torch.zeros_like(x_left[:, 0])
        v_left = torch.zeros_like(x_left[:, 0])
        ro_left = torch.zeros_like(x_left[:, 0])

        # Calculate the length of each exclusion range
        length_range1 = exclude_range1_max - exclude_range1_min
        length_range2 = exclude_range2_max - exclude_range2_min
        total_length = length_range1 + length_range2

        # Calculate the number of samples needed for each range
        num_samples_range1 = int((length_range1 / total_length) * N_train_r)
        num_samples_range2 = (
            N_train_r - num_samples_range1
        )  # Ensure total is exactly 100

        # Generate the required number of samples within each range
        points_y_r1 = torch.linspace(
            exclude_range1_min, exclude_range1_max, num_samples_range1 + 2
        )[1:-1]
        points_y_r2 = torch.linspace(
            exclude_range2_min, exclude_range2_max, num_samples_range2 + 2
        )[1:-1]

        # Stack the points to create x_left_r
        x_left_r1 = torch.stack(
            [domain_r1["x"][0] * torch.ones(num_samples_range1), points_y_r1], dim=1
        )

        # Parameters for the velocity profile for the left boundar

        l = domain_r1["y"][1] - domain_r1["y"][0]
        t = points_y_r1 - (domain_r1["y"][0] + l / 2)
        g_bar = (
            1.0  # Example value for the magnitude of the flow velocity at the center
        )

        # Compute the velocity profile for the left boundary
        u_left_r1 = g_bar * (1 - (2 * t / l) ** 2)
        v_left_r1 = torch.zeros_like(u_left_r1)
        ro_left_r1 = 1 + torch.zeros_like(u_left_r1)

        # Stack the points to create x_left_r
        x_left_r2 = torch.stack(
            [domain_r2["x"][0] * torch.ones(num_samples_range1), points_y_r2], dim=1
        )
        # Parameters for the velocity profile for the left boundary
        l = domain_r2["y"][1] - domain_r2["y"][0]
        t = points_y_r2 - (domain_r2["y"][0] + l / 2)
        g_bar = (
            1.0  # Example value for the magnitude of the flow velocity at the center
        )

        # Compute the velocity profile for the left boundary
        u_left_r2 = g_bar * (1 - (2 * t / l) ** 2)
        v_left_r2 = torch.zeros_like(u_left_r2)
        ro_left_r2 = 1 + torch.zeros_like(u_left_r2)

        # Concatenate the new samples for the left boundary
        x_left_combined = torch.cat((x_left, x_left_r1, x_left_r2), dim=0)
        u_left_combined = torch.cat((u_left, u_left_r1, u_left_r2), dim=0)
        v_left_combined = torch.cat((v_left, v_left_r1, v_left_r2), dim=0)
        ro_left_combined = torch.cat((ro_left, ro_left_r1, ro_left_r2), dim=0)

        # Original domain and points for the bottom boundary
        x_bottom = torch.stack([points_x.squeeze(), torch.zeros(len(points_x))], dim=1)
        u_bottom = torch.zeros_like(x_bottom[:, 0])
        v_bottom = torch.zeros_like(x_bottom[:, 0])
        ro_bottom = torch.zeros_like(x_bottom[:, 0])

        # The combined tensors now contain the samples from both domains without overlap
        x_top = torch.stack([points_x.squeeze(), torch.ones(N_train)], dim=1)
        u_top = torch.zeros_like(x_top[:, 0])  # 5.0*torch.sin(x_top[:,0]*torch.pi)
        v_top = torch.zeros_like(x_top[:, 0])
        ro_top = torch.zeros_like(x_top[:, 0])

        # Create x_left and related tensors
        x_right = torch.stack(
            [
                domain["x"][1] * torch.ones(len(points_y_filtered)),
                points_y_filtered.squeeze(),
            ],
            dim=1,
        )
        u_right = torch.zeros_like(x_right[:, 0])
        v_right = torch.zeros_like(x_right[:, 0])
        ro_right = torch.zeros_like(x_right[:, 0])

        # Calculate the length of each exclusion range
        length_range1 = exclude_range1_max - exclude_range1_min
        length_range2 = exclude_range2_max - exclude_range2_min
        total_length = length_range1 + length_range2

        # Calculate the number of samples needed for each range
        num_samples_range1 = int((length_range1 / total_length) * N_train_r)
        num_samples_range2 = (
            N_train_r - num_samples_range1
        )  # Ensure total is exactly 100

        # Generate the required number of samples within each range
        points_y_r1 = torch.linspace(
            exclude_range1_min, exclude_range1_max, num_samples_range1 + 2
        )[1:-1]
        points_y_r2 = torch.linspace(
            exclude_range2_min, exclude_range2_max, num_samples_range2 + 2
        )[1:-1]

        # Stack the points to create x_right_r
        x_right_r1 = torch.stack(
            [domain_r1["x"][1] * torch.ones(num_samples_range1), points_y_r1], dim=1
        )

        # Parameters for the velocity profile for the right boundar

        l = domain_r1["y"][1] - domain_r1["y"][0]
        t = points_y_r1 - (domain_r1["y"][0] + l / 2)
        g_bar = (
            1.0  # Example value for the magnitude of the flow velocity at the center
        )

        # Compute the velocity profile for the right boundary
        u_right_r1 = g_bar * (1 - (2 * t / l) ** 2)
        v_right_r1 = torch.zeros_like(u_right_r1)
        ro_right_r1 = 1 + torch.zeros_like(u_right_r1)

        # Stack the points to create x_right_r
        x_right_r2 = torch.stack(
            [domain_r2["x"][1] * torch.ones(num_samples_range1), points_y_r2], dim=1
        )
        # Parameters for the velocity profile for the right boundary
        l = domain_r2["y"][1] - domain_r2["y"][0]
        t = points_y_r2 - (domain_r2["y"][0] + l / 2)
        g_bar = (
            1.0  # Example value for the magnitude of the flow velocity at the center
        )

        # Compute the velocity profile for the right boundary
        u_right_r2 = g_bar * (1 - (2 * t / l) ** 2)
        v_right_r2 = torch.zeros_like(u_right_r2)
        ro_right_r2 = 1 + torch.zeros_like(u_right_r2)

        # Concatenate the new samples for the right boundary
        x_right_combined = torch.cat((x_right, x_right_r1, x_right_r2), dim=0)
        u_right_combined = torch.cat((u_right, u_right_r1, u_right_r2), dim=0)
        v_right_combined = torch.cat((v_right, v_right_r1, v_right_r2), dim=0)
        ro_right_combined = torch.cat((ro_right, ro_right_r1, ro_right_r2), dim=0)

        # Concatenate the points from all sides to form the boundary tensor
        x_corners = torch.tensor([[0.0, 0.0], [0.0, 1.0], [ratio, 0.0], [ratio, 1.0]])
        u_corners = torch.zeros_like(x_corners[:, 0])
        v_corners = torch.zeros_like(x_corners[:, 0])
        ro_corners = torch.zeros_like(x_corners[:, 0])

        X_train = torch.cat(
            [x_top, x_left_combined, x_bottom, x_right_combined, x_corners], dim=0
        )
        U_train = torch.cat(
            [u_top, u_left_combined, u_bottom, u_right_combined, u_corners], dim=0
        )
        V_train = torch.cat(
            [v_top, v_left_combined, v_bottom, v_right_combined, v_corners], dim=0
        )
        train_ro = torch.cat(
            [ro_top, ro_left_combined, ro_bottom, ro_right_combined, ro_corners], dim=0
        )  # , ro_center

    n = torch.floor(
        torch.sqrt(
            torch.div(
                torch.tensor(N_col_domain),
                torch.tensor(ratio).int(),
                rounding_mode="trunc",
            )
        )
    ).int()
    nx, ny = int(n * ratio), int(n)  # Change these values to your desired grid size
    X_col_domain = gen_data(design_domain=domain, steps_x=nx, steps_y=ny)

    X_train_U = X_train
    X_train_V = X_train
    X_train_ro = X_train
    X_train_P = torch.tensor([[1.0, 0]])
    train_P = torch.tensor([0])

    return (
        X_col_domain,
        [X_train_U, X_train_V, X_train_ro, X_train_P],
        [U_train, V_train, train_ro, train_P],
    )
