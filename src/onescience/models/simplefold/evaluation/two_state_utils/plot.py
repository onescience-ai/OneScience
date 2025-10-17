#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

from typing import Dict, Optional
import io
import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from scipy.stats import gaussian_kde
from scipy import interpolate


##################################################
FONTSIZE = 18
FIG_DPI = 300

x = np.linspace(-1.5, 1, 100)
y = np.linspace(-0.5, 2, 100)
X, Y = np.meshgrid(x, y)
##################################################


def scatterplot_apo(x, y, save_to=None, xlabel=None, ylabel=None, regplot=False):
    if len(x) == 0 or len(x) != len(y):
        raise ValueError("Invalid input data for scatter plot.")
        
    fig = plt.figure(figsize=(10, 8))
    if regplot:
        sns.regplot(x=x, y=y, color='steelblue', scatter_kws={'s': 10, 'alpha': 0.8, 'edgecolor': 'k'})
    else:     
        # Create scatter plot
        sns.scatterplot(x=x, y=y, color='steelblue', alpha=0.8, edgecolor='k')

        # Add reference line
        grid_x = np.linspace(0, 1, 100)
        plt.plot(grid_x, grid_x / 2 + 0.5, color='red', linestyle='--')

    # Set plot title and axis labels
    xlabel = xlabel if xlabel else "TM_native"
    ylabel = ylabel if ylabel else "TM_ensemble"
    plt.xlabel(xlabel, fontsize=FONTSIZE)
    plt.ylabel(ylabel, fontsize=FONTSIZE)
        
    # Set plot limits and ticks
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    if save_to is not None:
        plt.savefig(save_to, dpi=FIG_DPI)
        plt.close('all')
        return save_to
    return fig

    # Create the plot
    if regplot:
        sns.regplot(x=x, y=y, color='steelblue', scatter_kws={'s': 10, 'alpha': 0.8, 'edgecolor': 'k'})
        sns.regplot(
            x=x,
            y=y,
            color='steelblue',
            scatter_kws={'s': point_size, 'alpha': alpha, 'edgecolor': 'k'}
        )
    else:     
        # Create scatter plot
        sns.scatterplot(x=x, y=y, color='steelblue', alpha=0.8, edgecolor='k')

        sns.scatterplot(
            x=x,
            y=y,
            color='steelblue',
            alpha=alpha,
            edgecolor='k',
            s=point_size
        )
        # Add reference line
        grid_x = np.linspace(0, 1, 100)
        plt.plot(grid_x, grid_x / 2 + 0.5, color='red', linestyle='--')
        ax.plot(grid_x, grid_x / 2 + 0.5, color='red', linestyle='--', label='Reference')

    # Set plot title and axis labels
    xlabel = xlabel if xlabel else "TM_native"
    ylabel = ylabel if ylabel else "TM_ensemble"
    plt.xlabel(xlabel, fontsize=FONTSIZE)
    plt.ylabel(ylabel, fontsize=FONTSIZE)
    ax.set_xlabel(xlabel or "TM_native", fontsize=FONTSIZE)
    ax.set_ylabel(ylabel or "TM_ensemble", fontsize=FONTSIZE)
        
    # Set plot limits and ticks
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', labelsize=FONTSIZE)

    # Add legend if reference line was added
    if not regplot:
        ax.legend(fontsize=FONTSIZE-2)

    plt.tight_layout()
    if save_to is not None:
        plt.savefig(save_to, dpi=FIG_DPI)
        plt.close('all')
        plt.savefig(save_to, dpi=FIG_DPI, bbox_inches='tight')
        plt.close(fig)
        return save_to
    return fig

