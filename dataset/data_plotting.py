from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from e3nn.o3 import Irreps
import torch
import os


def plot3d_interactive(data, label=None, tight_range=False, set_range = None, colorbar=True, show_ticks = True, show_grid = True, nticks = 5, offset=15, show_fig=True, save=False, save_loc = None, gif_index = None):
    """
        Args:
            data: data in the form of a numpy array [x, y, z, charge] has dim [1, N, 4]
            label: label of data (for title)
            tight_range: if True, fit range when plotting to data
            rotation_range: if True, make range such that the axis go from -coord_max to coord_max
            colorbar: if True, show colorbar
            show_ticks: if True, show ticks
            show_grid: if True, show grid
            nticks: number of ticks
            offset: offset for normalization
            show_fig: if True, show plot
            save: if True, save plot
            save_loc: location to save plot inside plots folder
            gif_index: index of picture in gif

        Returns:
            0 if successful

        """

    pos3d = data[0][:, :-1].numpy()
    pos3d_x, pos3d_y, pos3d_z = pos3d.T

    charge = data[0][:, -1].numpy()

    if colorbar:
        colorbar = dict(
            title="Charge"
        )
    else:
        colorbar = dict(
            title=None
        )

    fig = go.Figure(data=[go.Scatter3d(
        x=pos3d_x,
        y=pos3d_y,
        z=pos3d_z,
        mode='markers',
        marker=dict(
            size=1,
            color=charge,  # set color to an array/list of desired values
            colorbar=colorbar,
            colorscale='Viridis',  # choose a colorscale
            opacity=0.9
        )
    )])
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=0.5, z=0.1)
    )
    range = [0, 512]
    scene = dict(aspectratio=dict(x=0.7, y=0.7, z=0.7),
                 xaxis = dict(range=range, nticks=nticks, showgrid = show_grid, showticklabels=show_ticks),
                 yaxis = dict(range=range, nticks=nticks, showgrid = show_grid, showticklabels=show_ticks),
                 zaxis = dict(range=range, nticks=nticks, showgrid = show_grid,showticklabels=show_ticks)
    )

    if set_range is not None:
        scene["xaxis"]["range"] = set_range
        scene["yaxis"]["range"] = set_range
        scene["zaxis"]["range"] = set_range
    # If normalize, get the corresponding range for data to fill range
    elif tight_range:
        min_pos = pos3d.min(axis=0)
        max_pos = pos3d.max(axis=0)
        range = np.vstack((min_pos, max_pos)).T
        range.T[0] -= offset
        range.T[1] += offset
        scene["xaxis"]["range"] = range[0]
        scene["yaxis"]["range"] = range[1]
        scene["zaxis"]["range"] = range[2]
    else:
        range = [0, 512]
        scene["xaxis"]["range"] = range
        scene["yaxis"]["range"] = range
        scene["zaxis"]["range"] = range

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