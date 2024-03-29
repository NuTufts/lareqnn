from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from e3nn.o3 import Irreps
import torch
import os


def plot3d_interactive(data, label=None, tight_range=False, set_range=None, colorbar=True, show_ticks=True,
                       show_grid=True, nticks=5, offset=15, show_fig=True, save=False, save_loc=None, gif_index=None):
    """
    Args:
        data (numpy.array): Data in the form of a numpy array [x, y, z, charge] has dim [1, N, 4].
        label (str): Label of data (for title).
        tight_range (bool): If True, fit range when plotting to data, adjusted by the offset.
        set_range (tuple): Set range for the axis manually, overrides tight_range if provided.
        colorbar (bool): If True, show colorbar.
        show_ticks (bool): If True, show ticks.
        show_grid (bool): If True, show grid.
        nticks (int): Number of ticks.
        offset (float or int): Offset for normalization.
        show_fig (bool): If True, show plot.
        save (bool): If True, save plot.
        save_loc (str): Location to save plot inside plots folder.
        gif_index (int): Index of picture in gif.

    Returns:
        int: 0 If function is successful.
    """
    pos3d = data[0][:, :-1].numpy()
    pos3d_x, pos3d_y, pos3d_z = pos3d.T
    charge = data[0][:, -1].numpy()

    colorbar = dict(title="Charge") if colorbar else dict(title=None)

    fig = go.Figure(data=[go.Scatter3d(
        x=pos3d_x,
        y=pos3d_y,
        z=pos3d_z,
        mode='markers',
        marker=dict(
            size=1,
            color=charge,
            colorbar=colorbar,
            colorscale='Viridis',
            opacity=0.9
        )
    )])
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=0.5, z=0.1)
    )
    axis_range = (0, 512)

    scene = dict(aspectratio=dict(x=0.7, y=0.7, z=0.7),
                 xaxis=dict(range=axis_range, nticks=nticks, showgrid=show_grid, showticklabels=show_ticks),
                 yaxis=dict(range=axis_range, nticks=nticks, showgrid=show_grid, showticklabels=show_ticks),
                 zaxis=dict(range=axis_range, nticks=nticks, showgrid=show_grid, showticklabels=show_ticks)
                 )

    if set_range is not None:
        scene["xaxis"]["range"] = set_range
        scene["yaxis"]["range"] = set_range
        scene["zaxis"]["range"] = set_range
    # If normalize, get the corresponding range for data to fill range
    elif tight_range:
        min_pos = pos3d.min(axis=0)
        max_pos = pos3d.max(axis=0)
        axis_range = np.vstack((min_pos, max_pos)).T
        axis_range.T[0] -= offset
        axis_range.T[1] += offset
        scene["xaxis"]["range"] = axis_range[0]
        scene["yaxis"]["range"] = axis_range[1]
        scene["zaxis"]["range"] = axis_range[2]
    else:
        axis_range = [0, 512]
        scene["xaxis"]["range"] = axis_range
        scene["yaxis"]["range"] = axis_range
        scene["zaxis"]["range"] = axis_range

    fig.update_layout(
        title=label,
        scene_camera=camera,
        font=dict(
            size=13,
            color="Black"
        ),
        margin=dict(t=30, r=0, l=0, b=0),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        scene=scene)

    if show_fig:
        fig.show()

    if save:
        if save_loc is None:
            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            fig.write_image(f"../plots/{now}_3D{label}.png")
        else:
            if not os.path.exists(f"../plots/3D{label}_{save_loc}"):
                os.makedirs(f"../plots/3D{label}_{save_loc}")
            fig.write_image(f"../plots/3D{label}_{save_loc}{gif_index:05d}.png")
    return 0


def plot3d(data, label=None):
    """
    Args:
        data: data in the form of a numpy array [x, y, z, charge] has dim [1, N, 4]
        label: data label

    Returns:
        0 if successful

    """
    charges = data[0][:, 3].numpy()
    pos3d = data[0][:, :3].numpy()
    # Plot a 3d plot using plt.scatter with values at location pos3d and color given by charges along with a color bar
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    points = ax.scatter(pos3d[:, 0], pos3d[:, 1], pos3d[:, 2], c=charges, cmap='viridis', alpha=0.4)
    ax.set_title(f"{label}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.colorbar(points)
    plt.show()

    return 0


def plot_projection(data, label=None, projection_axis=0, offset=20, save=False):
    """
        Args:
            data: data in the form of a numpy array [x, y, z, charge] has dim [1, N, 4]
            label: data label
            projection_axis: axis to project on, 0=x, 1=y, 2=z
            offset: amount of offset to add to the axes
            save: True if plots should be saved

        Returns:
            0 if successful
        """
    charges = data[0][:, 3].numpy()
    pos3d = data[0][:, :3].numpy()
    pos3d_projected = np.delete(pos3d, projection_axis, 1)
    stacked = np.hstack((pos3d_projected, charges.reshape(-1, 1)))
    # Plot a 2d histogram with values at location pos3d_projected and color given by charges using imshow
    axis1_range = [pos3d_projected[:, 0].min() - offset, pos3d_projected[:, 0].max() + offset]
    axis2_range = [pos3d_projected[:, 1].min() - offset, pos3d_projected[:, 1].max() + offset]
    fig, ax = plt.subplots()
    h = ax.hist2d(stacked[:, 0], stacked[:, 1], weights=charges, range=[axis1_range, axis2_range],
                  bins=[int(axis1_range[1] - axis1_range[0]), int(axis2_range[1] - axis2_range[0])], cmap='cividis',
                  cmin=0, cmax=charges.max())
    ax.set_title(f"{label} projected on axis {projection_axis}")
    fig.colorbar(h[3], ax=ax)
    if save:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        plt.savefig(f"../plots/{now}_{label}_axis{projection_axis}.png", dpi=300)
    return 0


def plot_all_projections(data, label=None, offset=20, save=False):
    """
    Args:
        data: data in the form of a numpy array [x, y, z, charge] has dim [N, 4]
        label: data label
        offset: amount of offset to add to the axes
        save: true if plots should be saved

    Returns:
        0 if successful
    """
    charges = data[0][:, 3].numpy()
    pos3d = data[0][:, :3].numpy()
    # Plot a 2d histogram with values at location pos3d_projected and color given by charges using imshow
    fig, ax = plt.subplots(1, 3, figsize=(20, 4))
    for i in range(3):
        pos3d_projected = np.delete(pos3d, i, 1)
        axis1_range = [pos3d_projected[:, 0].min() - offset, pos3d_projected[:, 0].max() + offset]
        axis2_range = [pos3d_projected[:, 1].min() - offset, pos3d_projected[:, 1].max() + offset]
        h = ax[i].hist2d(pos3d_projected[:, 0], pos3d_projected[:, 1], weights=charges,
                         range=[axis1_range, axis2_range],
                         bins=[int(axis1_range[1] - axis1_range[0]), int(axis2_range[1] - axis2_range[0])],
                         cmap='cividis', cmin=0, cmax=charges.max())
        ax[i].set_title(f"{label} projected on axis {i}")
        fig.colorbar(h[3], ax=ax[i])
    if save:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        plt.savefig(f"../plots/{now}_{label}.png", dpi=300)
    return 0


def plot_all_projections_diff(data, model, rotation, irreps_in, irreps_out, label=None, offset=20, save=False,
                         device=torch.device("cpu")):
    """
    Args:
        data: data in the form of a numpy array [x, y, z, charge] has dim [N, 4]
        model: The model to be used for predictions
        rotation: The rotation to be applied
        irreps_in: The input irreps
        irreps_out: The output irreps
        label: data label
        offset: amount of offset to add to the axes
        save: true if plots should be saved
        device: device to use for computations

    Returns:
        0 if successful
    """
    # Compute the data for each row
    rotation_tensor = torch.tensor(rotation)
    data_first_row = rotate_sparse_tensor(model(data), irreps_out, rotation_tensor, device)
    data_second_row = model(rotate_sparse_tensor(data, irreps_in, rotation_tensor, device))
    data_third_row = (data_second_row - data_first_row) / data_first_row.F.abs().max()

    equiv_error = 1 - (data_third_row.F.abs().mean() / data_first_row.F.abs().max())

    # Combine all the data into a list
    all_data = [data_first_row, data_second_row, data_third_row]

    # Create subplots
    fig, axs = plt.subplots(3, 3, figsize=(20, 12))

    plt.subplots_adjust(top=0.92)

    # Set title
    if label is not None:
        fig.suptitle(
            f"Equivariance for {label} at rotation ({rotation[0]:.2f}, {rotation[1]:.2f}, {rotation[2]:.2f}): {equiv_error:.2f}",
            fontsize=18)
    else:
        fig.suptitle(
            f"Equivariance at rotation ({rotation[0]:.2f}, {rotation[1]:.2f}, {rotation[2]:.2f}): {equiv_error:.2f}",
            fontsize=18)

    # Mapping for axis labels
    axes_labels = ['x', 'y', 'z']

    for row in range(3):
        data = all_data[row]
        charges = data.F.cpu().numpy().reshape(-1)
        pos3d = data.C[:, 1:].cpu().numpy()
        for col in range(3):
            # setup different color map for diff
            if row != 2:
                colormap = "cividis"
                divnorm = colors.Normalize(vmin=0., vmax=charges.max())
            else:
                colormap = "seismic"
                if charges.min() == 0:  # if no diff
                    divnorm = colors.Normalize(vmin=-charges.max(), vmax=charges.max())
                else:
                    divnorm = colors.TwoSlopeNorm(vmin=charges.min(), vcenter=0., vmax=charges.max())
            pos3d_projected = np.delete(pos3d, col, 1)
            axis1_range = [pos3d_projected[:, 0].min() - offset, pos3d_projected[:, 0].max() + offset]
            axis2_range = [pos3d_projected[:, 1].min() - offset, pos3d_projected[:, 1].max() + offset]
            h = axs[row, col].hist2d(pos3d_projected[:, 0], pos3d_projected[:, 1], weights=charges,
                                     range=[axis1_range, axis2_range],
                                     bins=[int(axis1_range[1] - axis1_range[0]),
                                           int(axis2_range[1] - axis2_range[0])],
                                     cmap=colormap, norm=divnorm)
            if row == 0:
                axs[row, col].set_title(f"projection on {axes_labels[col]} axis")
            if row != 2:
                fig.colorbar(h[3], ax=axs[row, col], label=r"$\sqrt{charge}$")
            else:
                fig.colorbar(h[3], ax=axs[row, col])

            # add both axis labels for each column and row using axes_labels
            axis_labels2 = axes_labels.copy()
            del axis_labels2[col]
            axs[row, col].set_xlabel(axis_labels2[0])
            axs[row, col].set_ylabel(axis_labels2[1])

        # Add a title for each row
        row_titles = ["rot(model(data))", f"model(rot(data))", "diff"]
        axs[row, 0].set_ylabel(row_titles[row], fontsize=14)

    if save:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        plt.savefig(f"plots/{now}_{label}.png", dpi=300)

    return 0


def rotate_data_non_minkowski(data, angles):
    """

    Args:
        data: data to rotate in the form of a numpy array [x, y, z, charge] has dim [1, N, 4]
        angles: angle to rotate by. rotation is done in the order y, x, y right to left

    Returns: Rotated data with shape [1, N, 4]

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    abc = torch.Tensor(angles).to(device)
    x = data[0].type(torch.float32).to(device)

    coordinates = x[:, :-1]
    coordinates = torch.einsum("ij,bj->bi", Irreps("1e").D_from_angles(*abc), coordinates)

    features = x[:, -1]

    x = torch.cat((coordinates, features.unsqueeze(1)), dim=1)
    return x.unsqueeze(0).to("cpu")