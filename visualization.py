import os
import re

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import tikzplotlib
import numpy as np
import torch

import utils

cmap = plt.cm.seismic


def _save_tikz(fig_path):
    # clean up the old png files
    pattern = re.compile(os.path.basename(fig_path) + "-[0-9]+.png$")
    for file in os.listdir(os.path.dirname(fig_path)):
        if pattern.match(file):
            os.remove(os.path.join(os.path.dirname(fig_path), file))
    tikzplotlib.save("{}.tex".format(fig_path))


def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def plot_kernel(kernel,
                x_dimension_name="Pixel",
                y_dimension_name="Avg. filter diagonals",
                kernel_name='',
                save_path='',
                vmin=None,
                vmax=None,
                subplot_len=4):
    """plot function for plotting a 2D kernel and its diagonals."""

    if torch.is_tensor(kernel):
        kernel_ndarray = utils.to_numpy(kernel)
    else:
        kernel_ndarray = kernel
    kernel_ndarray = kernel_ndarray.squeeze()
        
    if kernel_ndarray.ndim not in [2, 3]:
        raise ValueError(
            "Kernel's number of dim after squeezing should be 2 or 3 but is {}"
            .format(kernel_ndarray.ndim))
    if kernel_ndarray.ndim == 2:
        kernel_ndarray = kernel_ndarray[np.newaxis, ...]
    figsize = (subplot_len * kernel_ndarray.shape[0], subplot_len * 2)
    fig, axes = plt.subplots(
        2, kernel_ndarray.shape[0],
        figsize=figsize, squeeze=False)
    for index, (col, kernel_single) in enumerate(
        zip(axes.T, kernel_ndarray)):
        print(kernel.shape)
        kernel_plot = col[0].imshow(
            kernel_single, cmap=cmap, vmin=vmin, vmax=vmax)
        col[0].set_xticks([])
        divider = make_axes_locatable(col[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.axis('off')
        middle_index = int(kernel_single.shape[0] / 2)
        diagonals = [
            kernel_single[:, middle_index],
            kernel_single[middle_index, :],
            np.diag(np.fliplr(kernel_single)),
            np.diag(kernel_single)
        ]
        avg_diagonals = np.mean(np.array(diagonals), axis=0)
        col[0].tick_params(axis='both', which='major', labelsize=18)
        col[0].tick_params(axis='both', which='minor', labelsize=16)

        simpleaxis(col[1])
        col[1].plot(avg_diagonals, color='g')
        divider = make_axes_locatable(col[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.axis('off')
        col[1].set_ylim([vmin, vmax])
        if index != 0:
            col[0].set_yticks([])
            col[1].set_xticks([])
            col[1].set_yticks([])

    axes[1][0].set_ylabel(y_dimension_name, fontsize=22)
    axes[1][0].set_xlabel(x_dimension_name, fontsize=22)
    axes[1][0].tick_params(axis='both', which='major', labelsize=18)
    axes[1][0].tick_params(axis='both', which='minor', labelsize=16)
    divider = make_axes_locatable(axes[0][-1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(kernel_plot, cax=cax)
    save_name=''
    if kernel_name:
        if type(kernel_name) == str:
            fig.suptitle(kernel_name)
            save_name='-'.join(kernel_name.lower().split(' '))
        elif type(kernel_name) == list:
            if len(kernel_name) != kernel_ndarray.shape[0]:
                raise ValueError(
                    f'len of kernel_name = {len(kernel_name)}'
                    f' is not equal to number of kernels'
                    f' {kernel_ndarray.shape[0]}')
            for name, col in zip(kernel_name, axes.T):
                col[0].set_title(name, fontsize=18)
            save_name='-'.join('-'.join(kernel_name).lower().split(' '))
            
    plt.tight_layout()
    if save_path and save_name:
        fig_path = os.path.join(
            save_path,
            f'{save_name}.pdf')
        fig.savefig(fig_path)

def plot_feedback_kernel(feedback_kernel, master_save_name="", save_path=""):
    pathways = {0: "on", 1: "off"}
    for index_1, pathway_name_1 in pathways.items():
        for index_2, pathway_name_2 in pathways.items():
            if master_save_name == "":
                save_path = ""
            else:
                kernel_name = "{0}_{1}_{2}".format(
                    pathway_name_1,
                    pathway_name_2,
                    master_save_name)
            max_magnitude = utils.to_numpy(
                torch.max(torch.abs(feedback_kernel)))
            plot_kernel(feedback_kernel[index_1, index_2],
                        kernel_name=kernel_name,
                        save_path=save_path,
                        vmin=-max_magnitude,
                        vmax=max_magnitude)


def plot_image(image, image_name, save_path=False):
    """plot in image. Can be used to show weights."""

    image_ndarray = utils.to_numpy(image) if torch.is_tensor(image) else image
    fig = plt.figure()
    plt.imshow(image_ndarray.squeeze(), cmap=plt.cm.gray)
    plt.title(image_name, fontsize=16)
    plt.tight_layout()
    if save_path:
        file_name = "_".join(image_name.split())
        fig_path = os.path.join(save_path, "{}.pdf".format(file_name))
        fig.savefig(fig_path)


def plot_all_filters(
    filterbank_tensor,
    title=None,
    kernel_name="",
    save_path="",
    vmin=None,
    vmax=None,
    cmap=plt.cm.seismic
):
    """Plot all the filters in a filter bank."""

    SUBPLOT_LENGTH = 10
    if torch.is_tensor(filterbank_tensor):
        filterbank_ndarray = utils.to_numpy(filterbank_tensor)
    else:
        filterbank_ndarray = filterbank_tensor
    filterbank = np.squeeze(filterbank_ndarray)
    if len(filterbank.shape) == 4:
        figsize = (SUBPLOT_LENGTH * filterbank.shape[1],
                   SUBPLOT_LENGTH * filterbank.shape[0])
        fig, axs = plt.subplots(
            filterbank.shape[0],
            filterbank.shape[1],
            figsize=figsize)
        for i in range(filterbank.shape[0]):
            for j in range(filterbank.shape[1]):
                axs[i, j].imshow(
                    filterbank[i, j, :, :],
                    cmap=cmap,
                    vmax=vmax,
                    vmin=vmin)
                axs[i, j].set_axis_off()
    if len(filterbank.shape) == 3:
        fig_dim = filterbank.shape[0]
        figsize = (SUBPLOT_LENGTH * fig_dim, SUBPLOT_LENGTH)
        fig, axs = plt.subplots(1, fig_dim, figsize=figsize)
        counter = 0
        for i in range(fig_dim):
            if counter >= filterbank.shape[0]:
                break
            axs[i].imshow(
                filterbank[counter, :, :],
                cmap=cmap,
                vmax=vmax,
                vmin=vmin)
            axs[i].set_axis_off()
            counter += 1
    if len(filterbank.shape) == 2:
        figsize = (SUBPLOT_LENGTH, SUBPLOT_LENGTH)
        fig, axs = plt.subplots(1, 1, figsize=figsize)
        cs = axs.imshow(filterbank, cmap=cmap)  # ,
        # vmax = max_value, vmin = min_value)
        fig.colorbar(cs)
        axs.set_axis_off()

    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.subplots_adjust(wspace=0, hspace=0)
    if save_path:
        kernel_snake_case = "-".join(kernel_name.lower().split(" "))
        fig_path = os.path.join(save_path, kernel_snake_case)
        fig.savefig(fig_path, bbox_inches='tight', pad_inches=0)
        _save_tikz(fig_path)

    return fig

def simple_plot(x, y, ax=None, xlabel='', ylabel='',xlim=None, ylim=None, **kwargs):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family']='sans-serif'
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=28)
    ax.set_ylabel(ylabel, fontsize=28)
    ax.tick_params(
        axis='both', which='major',
        labelsize=22, length=10, width=2, direction='in')
    ax.tick_params(
        axis='both', which='minor', labelsize=18)
    ax.plot(x, y, **kwargs)
    ax.grid(True, which='both', snap=True, linestyle='--')
    #ax.grid(False)
    plt.tight_layout()
    return ax