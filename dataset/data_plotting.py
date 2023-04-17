from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime


def plot3d_interactive(dataiter, idx_to_class):
    """
        Args:
            dataiter: next data iteration from data_loader
            idx_to_class: amount of offset to add to the axes

        Returns:
            0 if successful

        """
    label = idx_to_class[int(dataiter[1][0])]
    pos3d = dataiter[0][0]
    fig = go.Figure(data=[go.Scatter3d(
        x=pos3d[:, 0],
        y=pos3d[:, 1],
        z=pos3d[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=pos3d[:, 3],  # set color to an array/list of desired values
            colorbar=dict(
                title="Charge"
            ),
            colorscale='Viridis',  # choose a colorscale
            opacity=0.9
        )
    )])
    fig.update_layout(
        title=label,
        scene=dict(
            xaxis=dict(nticks=6, range=[0, 512]),
            yaxis=dict(nticks=6, range=[0, 512]),
            zaxis=dict(nticks=6, range=[0, 512]),
            aspectratio=dict(x=1, y=1, z=1)))
    fig.show()
    return 0


def plot3d(dataiter, idx_to_class):
    """
    Args:
        dataiter: next data iteration from data_loader
        idx_to_class: amount of offset to add to the axes

    Returns:
        0 if successful

    """
    label = int(dataiter[1][0])
    charges = dataiter[0][0][:, 3].numpy()
    pos3d = dataiter[0][0][:, :3].numpy()
    # Plot a 3d plot using plt.scatter with values at location pos3d and color given by charges along with a color bar
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    points = ax.scatter(pos3d[:, 0], pos3d[:, 1], pos3d[:, 2], c=charges, cmap='viridis', alpha=0.4)
    ax.set_title(f"{idx_to_class[int(label)]}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.colorbar(points)
    plt.show()

    return 0


def plot_projection(dataiter, idx_to_class, projection_axis=0, offset=20, save=False):
    """
        Args:
            dataiter: next data iteration from data_loader
            idx_to_class: amount of offset to add to the axes
            projection_axis: axis to project on, 0=x, 1=y, 2=z
            offset: if True, saves the plots
            save: true if plots should be saved

        Returns:
            0 if successful
        """
    label = dataiter[1]
    charges = dataiter[0][0][:, 3].numpy()
    pos3d = dataiter[0][0][:, :3].numpy()
    pos3d_projected = np.delete(pos3d, projection_axis, 1)
    stacked = np.hstack((pos3d_projected, charges.reshape(-1, 1)))
    print(stacked.shape)
    # Plot a 2d histogram with values at location pos3d_projected and color given by charges using imshow
    axis1_range = [pos3d_projected[:, 0].min() - offset, pos3d_projected[:, 0].max() + offset]
    axis2_range = [pos3d_projected[:, 1].min() - offset, pos3d_projected[:, 1].max() + offset]
    print(axis1_range, axis2_range)
    fig, ax = plt.subplots()
    h = ax.hist2d(stacked[:, 0], stacked[:, 1], weights=charges, range=[axis1_range, axis2_range],
                  bins=[int(axis1_range[1] - axis1_range[0]), int(axis2_range[1] - axis2_range[0])], cmap='cividis',
                  cmin=0, cmax=charges.max())
    ax.set_title(f"{idx_to_class[int(label)]} projected on axis {projection_axis}")
    fig.colorbar(h[3], ax=ax)
    if save:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        plt.savefig(f"../plots/{now}_{idx_to_class[int(label)]}_axis{projection_axis}.png", dpi=300)
    return 0


def plot_all_projections(dataiter, idx_to_class, offset=20, save=False):
    """
    Args:
        dataiter: next data iteration from data_loader
        idx_to_class: amount of offset to add to the axes
        offset: if True, saves the plots
        save: true if plots should be saved

    Returns:
        0 if successful
    """
    label = dataiter[1]
    charges = dataiter[0][0][:, 3].numpy()
    pos3d = dataiter[0][0][:, :3].numpy()
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
        ax[i].set_title(f"{idx_to_class[int(label)]} projected on axis {i}")
        fig.colorbar(h[3], ax=ax[i])
    if save:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        plt.savefig(f"../plots/{now}_{idx_to_class[int(label)]}.png", dpi=300)
    return 0
