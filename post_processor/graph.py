# ----------------------------------------------------------------------------------------------------------------------
#  This file is part of the SlowFlow distribution  (https://github.com/bevanwsjones/SlowFlow).
#  Copyright (c) 2020 Bevan Walter Stewart Jones.
#
#  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation, version 3.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
#  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this program. If not, see
#  <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------------------------------------------------
# filename: graph.py
# description: Functions of plotting fields and graphs
# ----------------------------------------------------------------------------------------------------------------------

import sys
import numpy as np

import matplotlib
import matplotlib.collections as cl
import matplotlib.pyplot as plt

# matplotlib.use("pgf")
matplotlib.rcParams.update({
    # "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': False,
    'pgf.rcfonts': True,
    'font.size': 10,
    'axes.labelsize': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})


class FieldPlotMetaData:
    figure_title = 'Title'
    figure_page_width_cm = 15.0
    label = ''
    plot_grid_max_columns = 2  # maximum number, may be lower if there are less than this many plots.

    color_bar_label = r'$\phi$'
    color_map = 'viridis'  # https://matplotlib.org/stable/tutorials/colors/colormaps.html

    add_vector_arrows = True
    vector_arrow_color = 'k'

    number_plot = 0
    number_rows = 0
    number_column = 0


def create_field_subplot_axis(_vertex_co_ordinates, _axis, _i_column, _i_axis, _meta_data):
    """
    Sets up the subplot axis labels, ticks, limits, e.t.c for field ploting. Since it is for field the some automatic
    labelling occurs.

    :param _vertex_co_ordinates:
    :param _axis:
    :param _i_column:
    :param _i_axis:
    :param _meta_data:
    :return:
    """

    # x axis
    if _meta_data.number_plot - _i_axis <= _meta_data.plot_grid_max_columns:
        _axis.set_xlabel('$x$')
    else:
        plt.setp(_axis.get_xticklabels(), visible=False)

    # y axis
    if _i_column == 0:
        _axis.set_ylabel('$y$')
    else:
        plt.setp(_axis.get_yticklabels(), visible=False)

    _axis.set_xlim(np.min(_vertex_co_ordinates[:, 0]), np.max(_vertex_co_ordinates[:, 0]))
    _axis.set_ylim(np.min(_vertex_co_ordinates[:, 1]), np.max(_vertex_co_ordinates[:, 1]))
    _axis.set_xticks(np.linspace(np.min(_vertex_co_ordinates[:, 0]), np.max(_vertex_co_ordinates[:, 0]), 5))
    _axis.set_yticks(np.linspace(np.min(_vertex_co_ordinates[:, 1]), np.max(_vertex_co_ordinates[:, 1]), 5))
    _axis.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
    _axis.set_aspect('equal', adjustable='box')
    _axis.set_frame_on(True)


def create_field_plot(_vertex_co_ordinates, _cell_vertex_connectivity, _field_variable, _axis, _meta_data):
    """
    Using the PolyCollection the field colored field variables are plotted on the mesh.

    :param _vertex_co_ordinates:
    :param _cell_vertex_connectivity:
    :param _field_variable:
    :param _axis:
    :param _meta_data:
    :return:
    """

    pc = cl.PolyCollection(_vertex_co_ordinates[_cell_vertex_connectivity], cmap=_meta_data.color_map)
    pc.set_array(_field_variable if len(_field_variable.shape) == 1 else np.linalg.norm(_field_variable, axis=1))
    _axis.add_collection(pc)


def create_vector(_cell_centroid, _field_variable, _axis, _meta_data):
    """
    For vector fields arrow heads are added to the centroid of cells to indicate the direction of the vector field.
    :param _cell_centroid:
    :param _field_variable:
    :param _axis:
    :param _meta_data:
    :return:
    """

    if len(_field_variable.shape) != 1:
        normal_vectors = _field_variable / np.linalg.norm(_field_variable, axis=1)[:, None]
        _axis.quiver(*[_cell_centroid[:, 0], _cell_centroid[:, 1]], normal_vectors[:, 0], normal_vectors[:, 1],
                     color=_meta_data.vector_arrow_color, scale=30.0)


def create_field_color_bar(_figure, _figure_grid, _field_variable, _meta_data):
    """
    Adds a color bar to the figure, automatically scales the color bar based on the min and maximun values in the parsed
    field data. For vectors the norms are used.

    :param _figure:
    :param _figure_grid:
    :param _field_variable:
    :param _meta_data:
    :return:
    """

    map = matplotlib.cm.ScalarMappable(norm=None, cmap=_meta_data.color_map)
    min_max = [sys.float_info.max, sys.float_info.min]
    for fld in _field_variable:
        if len(fld.shape) == 1:
            min_max = [min(np.min(fld), min_max[0]), max(np.max(fld), min_max[1])]
        else:
            norm = np.linalg.norm(fld, axis=1)
            min_max = [min(np.min(norm), min_max[0]), max(np.max(norm), min_max[1])]

    map.set_clim(vmin=min_max[0], vmax=min_max[1])
    formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    axis = _figure.add_subplot(_figure_grid[-1, :], in_layout=True)
    plt.colorbar(map, cax=axis, label=_meta_data.color_bar_label, orientation='horizontal', format=formatter,
                 ticklocation='bottom', ticks=np.linspace(min_max[0], min_max[1], 5))


def setup_field_figure(_meta_data):
    """
    Sets up the subplot gird, and tries to maintain a good layout no matter how many subplots are requested. It is not
    recommended to go beyond 4 plots per row.

    :param _meta_data:
    :return:
    """

    cm = 1.0 / 2.54  # centimeters in inches
    _meta_data.number_row = int(np.ceil(np.array([_meta_data.number_plot / _meta_data.plot_grid_max_columns]))[0])
    _meta_data.number_column = min(_meta_data.plot_grid_max_columns, _meta_data.number_plot)

    color_bar_height = 0.25*cm
    width = 0.95*_meta_data.figure_page_width_cm*cm
    height = ((_meta_data.figure_page_width_cm*cm)*float(_meta_data.number_row)/float(_meta_data.number_column))
    total_height = (height + color_bar_height)

    height_ratios = [1.0/float(_meta_data.number_row)*height/total_height for _ in range(_meta_data.number_row)]
    height_ratios.append(color_bar_height/total_height)

    figure = plt.figure(figsize=(width, total_height), constrained_layout=True)
    figure_grid = figure.add_gridspec(nrows=_meta_data.number_row + 1, ncols=_meta_data.number_column, figure=figure,
                                      height_ratios=height_ratios)

    ax_list = [[0 for _ in range(_meta_data.number_column)] for _ in range(_meta_data.number_row)]
    for row in range(_meta_data.number_row):
        for col in range(_meta_data.number_column):
            ax_list[row][col] = figure.add_subplot(figure_grid[row, col], in_layout=True)
    ax_list = np.asarray(ax_list)

    return figure, figure_grid, ax_list


def plot_field(_mesh, _field, _meta_data):
    """
    Given a mesh_entities and associated fields a set of field plots are created

    :param _mesh:
    :param _field:
    :param _meta_data:
    :return:
    """

    field = [_field] if type(_field) is not list else _field
    _meta_data.number_plot = len(field)

    figure, grid, axis = setup_field_figure(_meta_data)
    create_field_color_bar(figure, grid, field, _meta_data)

    i_axis = 0
    for i_row in range(_meta_data.number_row):
        for i_column in range(_meta_data.number_column):
            if i_axis >= _meta_data.number_plot:
                axis[i_row][i_column].set_visible(False)
                continue

            create_field_subplot_axis(_mesh.vertex_table.coordinate, axis[i_row][i_column], i_column, i_axis,
                                      _meta_data)
            create_field_plot(_mesh.vertex_table.coordinate, _mesh.cell_table.connected_vertex, field[i_axis],
                              axis[i_row][i_column], _meta_data)
            if _meta_data.add_vector_arrows:
                create_vector(_mesh.cell_table.centroid, field[i_axis], axis[i_row][i_column], _meta_data)

            i_axis += 1
    # plt.tight_layout()
    plt.show()
    # plt.savefig(_meta_data.figure_title + '.pdf', bbox_inches='tight')
