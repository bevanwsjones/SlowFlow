from matplotlib import pyplot as plt
from post_processor import graph as gr
import numpy as np

# return string name for error metric for plotting purposes
def grid_identity(gq):
    if gq == 0:
        return str('$\zeta_{no}$')
    elif gq == 1:
        return str('$\zeta_{ue}$')
    elif gq == 2:
        return str('$\zeta_{sk}$')
        # return str('$\zeta_{no}$')
    else:
        return ValueError("Invalid grid quality selection")


def set_size(_ax):
    _ax.title.set_size(20)
    _ax.xaxis.label.set_size(16)
    _ax.yaxis.label.set_size(16)
    _ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)

# plot the cartesian
def cartesian_plotter(int_error_array, bound_error_array, h):
    # plot 1 - x-gradient error - internal cells
    golden_mean = (np.sqrt(5) - 1.0) / 2.0
    # plot 1 - x-gradient error - internal cells
    fig1, (ax, ax2) = plt.subplots(ncols=2)
    ax.plot(h, int_error_array[:, 0, 0], '-o', label='$L_{1} norm$')
    ax.plot(h, int_error_array[:, 1, 0], '-ok', label='$L_{2} norm$')
    ax.plot(h, int_error_array[:, 2, 0], '-or', label='$L_{\infty} norm$')
    ax.axline((0.01, 0.0001), (0.1, 0.001), color='g', label='$\mathcal{O}(h)$', ls='--')
    ax.axline((0.01, 0.0001), (0.1, 0.01), color='b', label='$\mathcal{O}(h^2)$', ls='--')
    ax.set(xlabel='$h$', ylabel='Internal '+r'$\varepsilon_{x}$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    set_size(ax)
    dy = np.abs(np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0]))
    dx = np.abs(np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0]))
    ax.set_aspect((dx / dy) * golden_mean, adjustable='box')

    ax2.plot(h, bound_error_array[:, 0, 0], '-o', label='$L_{1} norm$')
    ax2.plot(h, bound_error_array[:, 1, 0], '-ok', label='$L_{2} norm$')
    ax2.plot(h, bound_error_array[:, 2, 0], '-or', label='$L_{\infty} norm$')
    ax2.axline((0.01, 0.001), (0.1, 0.01), color='g', label='$\mathcal{O}(h)$', ls='--')
    ax2.axline((0.01, 0.001), (0.1, 0.1), color='b', label='$\mathcal{O}(h^2)$', ls='--')
    ax2.set(xlabel='$h$', ylabel='Boundary '+r'$\varepsilon_{x}$')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlim(0.8 * np.min(h), 1.2 * np.max(h))
    ax2.set_ylim(0.8 * np.min(bound_error_array[:, :, 0]), 1.2 * np.max(bound_error_array[:, :, 0]))
    set_size(ax2)
    dy = np.abs(np.log10(ax2.get_ylim()[1]) - np.log10(ax2.get_ylim()[0]))
    dx = np.abs(np.log10(ax2.get_xlim()[1]) - np.log10(ax2.get_xlim()[0]))
    ax2.set_aspect((dx / dy) * golden_mean, adjustable='box')
    # handles, labels = ax.get_legend_handles_labels()
    # fig1.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.55, 0.25), prop={"size":12})
    fig1.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)

    fig2, (ax, ax2) = plt.subplots(ncols=2)
    ax.plot(h, int_error_array[:, 0, 1], '-o', label='$L_{1} norm$')
    ax.plot(h, int_error_array[:, 1, 1], '-ok', label='$L_{2} norm$')
    ax.plot(h, int_error_array[:, 2, 1], '-or', label='$L_{\infty} norm$')
    ax.axline((0.01, 0.0001), (0.1, 0.001), color='g', label='$\mathcal{O}(h)$', ls='--')
    ax.axline((0.01, 0.0001), (0.1, 0.01), color='b', label='$\mathcal{O}(h^2)$', ls='--')
    ax.set(xlabel='$h$', ylabel='Internal '+r'$\varepsilon_{y}$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(0.8 * np.min(h), 1.2 * np.max(h))
    ax.set_ylim(0.8 * np.min(int_error_array[:, :, 1]), 1.2 * np.max(int_error_array[:, :, 1]))
    set_size(ax)
    dy = np.abs(np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0]))
    dx = np.abs(np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0]))
    ax.set_aspect((dx / dy) * golden_mean, adjustable='box')

    ax2.plot(h, bound_error_array[:, 0, 1], '-o', label='$L_{1} norm$')
    ax2.plot(h, bound_error_array[:, 1, 1], '-ok', label='$L_{2} norm$')
    ax2.plot(h, bound_error_array[:, 2, 1], '-or', label='$L_{\infty} norm$')
    ax2.axline((0.01, 0.001), (0.1, 0.01), color='g', label='$\mathcal{O}(h)$', ls='--')
    ax2.axline((0.01, 0.001), (0.1, 0.1), color='b', label='$\mathcal{O}(h^2)$', ls='--')
    ax2.set(xlabel='$h$', ylabel='Boundary '+r'$\varepsilon_{y}$')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlim(0.8 * np.min(h), 1.2 * np.max(h))
    ax2.set_ylim(0.8 * np.min(bound_error_array[:, :, 1]), 1.2 * np.max(bound_error_array[:, :, 1]))
    set_size(ax2)
    dy = np.abs(np.log10(ax2.get_ylim()[1]) - np.log10(ax2.get_ylim()[0]))
    dx = np.abs(np.log10(ax2.get_xlim()[1]) - np.log10(ax2.get_xlim()[0]))
    ax2.set_aspect((dx / dy) * golden_mean, adjustable='box')
    # handles, labels = ax.get_legend_handles_labels()
    # fig2.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.55, 0.25), prop={"size":12})
    fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)
    return fig1, fig2

def grid_error_refine(int_error_array, bound_error_array, h, grid_quality, gq = 0):
    name = grid_identity(gq)
    golden_mean = (np.sqrt(5) - 1.0) / 2.0
    # plot 1 - x-gradient error - internal cells
    fig1, (ax, ax2) = plt.subplots(ncols=2)
    for i, i_metric in enumerate(grid_quality):
        ax.plot(h, int_error_array[i, :, 0], '-o', label=name+'='+str(i_metric))
        ax2.plot(h, bound_error_array[i, :, 0], '-o', label=name+'='+str(i_metric))
    ax.axline((0.01, 0.0001), (0.1, 0.001), color='g', label='$\mathcal{O}(h)$', ls='--')
    ax.axline((0.01, 0.0001), (0.1, 0.01), color='b', label='$\mathcal{O}(h^2)$', ls='--')
    ax.set(xlabel='$h$', ylabel='Internal '+r'$\varepsilon_{x}$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(0.5 * np.min(h), 1.2 * np.max(h))
    ax.set_ylim(0.8 * np.min(int_error_array[:, :, 0]), 1.2 * np.max(int_error_array[:, :, 0]))
    set_size(ax)
    dy = np.abs(np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0]))
    dx = np.abs(np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0]))
    ax.set_aspect((dx / dy) * golden_mean, adjustable='box')

    # plot 2 - x-gradient error - boundrary cells
    ax2.axline((0.01, 0.001), (0.1, 0.01), color='g', label='$\mathcal{O}(h)$', ls='--')
    ax2.axline((0.01, 0.001), (0.1, 0.1), color='b', label='$\mathcal{O}(h^2)$', ls='--')
    ax2.set(xlabel='$h$', ylabel='Boundary '+r'$\varepsilon_{x}$')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlim(0.5 * np.min(h), 1.2 * np.max(h))
    ax2.set_ylim(0.8 * np.min(bound_error_array[:, :, 0]), 1.2 * np.max(bound_error_array[:, :, 0]))
    set_size(ax2)
    dy = np.abs(np.log10(ax2.get_ylim()[1]) - np.log10(ax2.get_ylim()[0]))
    dx = np.abs(np.log10(ax2.get_xlim()[1]) - np.log10(ax2.get_xlim()[0]))
    ax2.set_aspect((dx / dy) * golden_mean, adjustable='box')
    # handles, labels = ax.get_legend_handles_labels()
    # fig1.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.55, 0.22), prop={"size":12})
    fig1.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)

    # plot 3 - y-gradient error - internal cells
    fig2, (ax, ax2) = plt.subplots(ncols=2)
    for i, i_metric in enumerate(grid_quality):
        ax.plot(h, int_error_array[i, :, 1], '-o', label=name+'='+str(i_metric))
        ax2.plot(h, bound_error_array[i, :, 1], '-o', label=name+'='+str(i_metric))
    ax.axline((0.01, 0.0001), (0.1, 0.001), color='g', label='$\mathcal{O}(h)$', ls='--')
    ax.axline((0.01, 0.0001), (0.1, 0.01), color='b', label='$\mathcal{O}(h^2)$', ls='--')
    ax.set(xlabel='$h$', ylabel='Internal '+r'$\varepsilon_{y}$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(0.5 * np.min(h), 1.2 * np.max(h))
    ax.set_ylim(0.8 * np.min(int_error_array[:, :, 1]), 1.2 * np.max(int_error_array[:, :, 1]))
    set_size(ax)
    dy = np.abs(np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0]))
    dx = np.abs(np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0]))
    ax.set_aspect((dx / dy) * golden_mean, adjustable='box')

    # plot 4 - y-gradient error - boundrary cells
    ax2.axline((0.01, 0.001), (0.1, 0.01), color='g', label='$\mathcal{O}(h)$', ls='--')
    ax2.axline((0.01, 0.001), (0.1, 0.1), color='b', label='$\mathcal{O}(h^2)$', ls='--')
    ax2.set(xlabel='$h$', ylabel='Boundary '+r'$\varepsilon_{y}$')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlim(0.5 * np.min(h), 1.2 * np.max(h))
    ax2.set_ylim(0.8 * np.min(bound_error_array[:, :, 1]), 1.2 * np.max(bound_error_array[:, :, 1]))

    set_size(ax2)
    dy = np.abs(np.log10(ax2.get_ylim()[1]) - np.log10(ax2.get_ylim()[0]))
    dx = np.abs(np.log10(ax2.get_xlim()[1]) - np.log10(ax2.get_xlim()[0]))
    ax2.set_aspect((dx / dy) * golden_mean, adjustable='box')
    # handles, labels = ax.get_legend_handles_labels()
    # fig2.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.55, 0.22), prop={"size":12})
    fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)
    return fig1, fig2

def grid_metric_plotter(bound_error_array, int_error_array, grid_quality_array, i_matrix, gq = 0):
    name = grid_identity(gq)
    golden_mean = (np.sqrt(5) - 1.0) / 2.0

    # plt.figure(1, figsize=(6, 6))
    fig1, (ax, ax2) = plt.subplots(ncols=2)
    # - x-gradient error - internal cells
    ax.plot(grid_quality_array[0], int_error_array[:, 1, 0], 'ok:', label='$L_{2} norm$')
    ax.set(xlabel=name, ylabel='Internal ' + r'$\varepsilon_{x}$')
    dy = ax.get_ylim()[1] - ax.get_ylim()[0]
    dx = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax.set_aspect((dx / dy) * golden_mean, adjustable='box')
    plt.legend()
    set_size(ax)

    ax2.plot(grid_quality_array[0], bound_error_array[:, 1, 0], 'ok:', label='$L_{2} norm$')
    ax2.set(xlabel=name, ylabel='Boundary ' + r'$\varepsilon_{x}$')
    dy = ax2.get_ylim()[1] - ax2.get_ylim()[0]
    dx = ax2.get_xlim()[1] - ax2.get_xlim()[0]
    ax2.set_aspect((dx / dy) * golden_mean, adjustable='box')
    plt.legend()
    set_size(ax2)
    fig1.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)

    # plt.figure(2, figsize=(6, 6))
    fig2, (ax, ax2) = plt.subplots(ncols=2)
    # - y-gradient error - internal cells
    ax.plot(grid_quality_array[0], int_error_array[:, 1, 1], 'ok:', label='$L_{2} norm$')
    ax.set(xlabel=name, ylabel='Internal ' + r'$\varepsilon_{y}$')
    dy = ax.get_ylim()[1] - ax.get_ylim()[0]
    dx = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax.set_aspect((dx / dy) * golden_mean, adjustable='box')
    plt.legend()
    set_size(ax)

    ax2.plot(grid_quality_array[0], bound_error_array[:, 1, 1], 'ok:', label='$L_{2} norm$')
    ax2.set(xlabel=name, ylabel='Boundary ' + r'$\varepsilon_{y}$')
    dy = ax2.get_ylim()[1] - ax2.get_ylim()[0]
    dx = ax2.get_xlim()[1] - ax2.get_xlim()[0]
    ax2.set_aspect((dx / dy) * golden_mean, adjustable='box')
    plt.legend()
    set_size(ax2)
    fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)
    return fig1, fig2

def triangle_plotter(error_array, h):
    # plot 1 - x-gradient error - internal cells
    golden_mean = (np.sqrt(5) - 1.0) / 2.0
    # plot 1 - x-gradient error - internal cells
    fig1, ax = plt.subplots(ncols=1)
    ax.plot(h, error_array[:, 0], '-o', label='$L_{1} norm$')
    ax.plot(h, error_array[:, 1], '-ok', label='$L_{2} norm$')
    ax.plot(h, error_array[:, 2], '-or', label='$L_{\infty} norm$')
    # ax.axline((0.01, 0.001), (0.1, 0.01), color='g', label='$\mathcal{O}(h)$', ls='--')
    # ax.axline((0.01, 0.001), (0.1, 0.1), color='b', label='$\mathcal{O}(h^2)$', ls='--')
    ax.axline((0.01, 0.01), (0.1, 0.1), color='g', label='$\mathcal{O}(h)$', ls='--')
    ax.axline((0.01, 0.01), (0.1, 1.0), color='b', label='$\mathcal{O}(h^2)$', ls='--')
    ax.set(xlabel='$h$', ylabel= r'$\varepsilon_{x}$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(0.5 * np.min(h), 1.2 * np.max(h))
    ax.set_ylim(0.8 * np.min(error_array[:, :]), 1.2 * np.max(error_array[:, :]))
    set_size(ax)
    dy = np.abs(np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0]))
    dx = np.abs(np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0]))
    ax.set_aspect((dx / dy) * golden_mean, adjustable='box')

    handles, labels = ax.get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.55, 0.10), prop={"size":12})
    fig1.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)
    return fig1

def grid_relation_refine(h, grid_qual):
    # plot 1 - x-gradient error - internal cells
    golden_mean = (np.sqrt(5) - 1.0) / 2.0
    # plot 1 - x-gradient error - internal cells
    fig1, ax = plt.subplots(ncols=1)
    ax.plot(h, grid_qual[:, 0], '-o', label='$\zeta_{no}$')
    ax.plot(h, grid_qual[:, 1], '-ok', label='$\zeta_{ue}$')
    ax.plot(h, grid_qual[:, 2], '-or', label='$\zeta_{sk}$')
    ax.set(xlabel='$h$', ylabel= r'$\zeta$')
    ax.set_xlim(0.5 * np.min(h), 1.2 * np.max(h))
    ax.set_ylim(0.8 * np.min(grid_qual[:, :]), 1.2 * np.max(grid_qual[:, :]))
    dy = ax.get_ylim()[1] - ax.get_ylim()[0]
    dx = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax.set_aspect((dx / dy) * golden_mean, adjustable='box')
    plt.legend()
    set_size(ax)
    fig1.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)
    return fig1

# def triangle_plotter(error_array, h):
#     # plot 1 - x-gradient error - internal cells
#     golden_mean = (np.sqrt(5) - 1.0) / 2.0
#     # plot 1 - x-gradient error - internal cells
#     fig1, (ax, ax2) = plt.subplots(ncols=2)
#     ax.plot(h, error_array[:, 0, 0], '-o', label='$L_{1} norm$')
#     ax.plot(h, error_array[:, 1, 0], '-ok', label='$L_{2} norm$')
#     ax.plot(h, error_array[:, 2, 0], '-or', label='$L_{\infty} norm$')
#     ax.axline((0.01, 0.001), (0.1, 0.01), color='g', label='$\mathcal{O}(h)$', ls='--')
#     ax.axline((0.01, 0.001), (0.1, 0.1), color='b', label='$\mathcal{O}(h^2)$', ls='--')
#     # ax.axline((0.01, 0.01), (0.1, 0.1), color='g', label='$\mathcal{O}(h)$', ls='--')
#     # ax.axline((0.01, 0.01), (0.1, 1.0), color='b', label='$\mathcal{O}(h^2)$', ls='--')
#     ax.set(xlabel='$h$', ylabel= r'$\varepsilon_{x}$')
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.set_xlim(0.5 * np.min(h), 1.2 * np.max(h))
#     ax.set_ylim(0.8 * np.min(error_array[:, :, 0]), 1.2 * np.max(error_array[:, :, 0]))
#     set_size(ax)
#     dy = np.abs(np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0]))
#     dx = np.abs(np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0]))
#     ax.set_aspect((dx / dy) * golden_mean, adjustable='box')
#
#     # plot 2 - x-gradient error - boundrary cells
#     ax2.plot(h, error_array[:, 0, 1], '-o', label='$L_{1} norm$')
#     ax2.plot(h, error_array[:, 1, 1], '-ok', label='$L_{2} norm$')
#     ax2.plot(h, error_array[:, 2, 1], '-or', label='$L_{\infty} norm$')
#     # ax2.axline((0.01, 0.01), (0.1, 0.1), color='g', ls='--')
#     # ax2.axline((0.01, 0.01), (0.1, 1.0), color='b', ls='--')
#     ax2.axline((0.01, 0.01), (0.1, 0.1), color='g', ls='--')
#     ax2.axline((0.01, 0.01), (0.1, 1.0), color='b', ls='--')
#     ax2.set(xlabel='$h$', ylabel=r'$\varepsilon_{y}$')
#     ax2.set_yscale('log')
#     ax2.set_xscale('log')
#     ax2.set_xlim(0.5 * np.min(h), 1.2 * np.max(h))
#     ax2.set_ylim(0.8 * np.min(error_array[:, :, 1]), 1.2 * np.max(error_array[:, :, 1]))
#     set_size(ax2)
#     dy = np.abs(np.log10(ax2.get_ylim()[1]) - np.log10(ax2.get_ylim()[0]))
#     dx = np.abs(np.log10(ax2.get_xlim()[1]) - np.log10(ax2.get_xlim()[0]))
#     ax2.set_aspect((dx / dy) * golden_mean, adjustable='box')
#
#     handles, labels = ax.get_legend_handles_labels()
#     fig1.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.55, 0.25), prop={"size":12})
#
#     fig1.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)
#     return fig1