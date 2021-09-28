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
import string as st
import numpy as np

from matplotlib import collections as cl, pyplot as plt, rcParams, cm, ticker
from dataclasses import dataclass


# ----------------------------------------------------------------------------------------------------------------------
# General Plotting Functions
# ----------------------------------------------------------------------------------------------------------------------

def show_plot():
    """
    Shows the current figure, ensures rcParams are correctly set.
    """

    rcParams.update({
        "backend": "GTK3Agg",
        'font.family': 'serif',
        'text.usetex': False,
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
    })
    plt.show()


def save_plot(_figure_title):
    """
    Saves the current figure, ensures rcParams are correctly set. Will set to latex formatting and the name of the file
    will be the passed _figure_title. Note must be called before plt.show() otherwise the current figure is cleared and
    nothing will be saved.

    :param _figure_title: Name of the figure, will be the name of the .pdf file.
    :type _figure_title: str
    """

    rcParams.update({
        "backend": "pdf",
        'pdf.compression': 4,
        'font.family': 'serif',
        'text.usetex': True,
        'savefig.bbox': 'tight',
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
    })
    plt.savefig(_figure_title + '.pdf')


# ----------------------------------------------------------------------------------------------------------------------
# Field Plotting Functions
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class FieldPlotMetadata:
    """
    Contains the customisable settings for field data plotting.
    """

    figure_title: str = 'Title'
    figure_page_width_cm: float = 15.0
    plot_grid_max_columns: int = 2  # maximum number, may be lower if there are less than this many plots.

    is_annotate_subplot: bool = False
    color_map: str = 'viridis'  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    add_vector_arrows: bool = True
    vector_arrow_color: str = 'k'

    color_bar_label: str = r'$\phi$'

    is_show_plot: bool = True
    is_save_plot: bool = False

    number_plot: int = 0
    number_row: int = 0
    number_column: int = 0


def generate_random_field(_number_of_fields, _number_of_cells, _is_vector=None, _seed=0):
    """
    Generates a random scalar and/or vector field. For scalar fields the value lies between [-1, 1), and for vectors
    each component of the two components will be between [-1, 1).

    :param _number_of_fields: Number of fields to create.
    :type _number_of_fields: int
    :param _number_of_cells: Number of cells in the mesh, number of random scalars/vectors created for each field.
    :type _number_of_cells: int
    :param _is_vector: List of booleans, size of _number_of_fields, if an element is true a vector field is generated.
    :type _is_vector: list
    :param _seed: seed for the numpy random generator.
    :type _seed: int
    :return: list of fields with random scalars/vectors.
    :type: list
    """

    if _is_vector is None:
        _is_vector = [False for _ in range(_number_of_fields)]

    if len(_is_vector) != _number_of_fields:
        raise RuntimeError("_number_of_fields (" + str(_number_of_fields) + ") and size of _is_vector ("
                           + str(len(_is_vector)) + ") are not the same.")

    return [2.0*np.random.default_rng(_seed).random(_number_of_cells
                                                    if not _is_vector[i_field] else (_number_of_cells, 2)) - 1.0
            for i_field in range(_number_of_fields)]


def format_field_subplot_axis(_vertex_coordinates, _metadata, _axis, _i_column, _i_axis):
    """
    Sets up the subplot axis labels, ticks, limits, e.t.c for field plotting. Since it is for field the some automatic
    labelling occurs.

    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :type _vertex_coordinates: numpy.array
    :param _metadata: The figure, subplot, and field plotting metadata containing the setting to use.
    :type _metadata: FieldPlotMetadata
    :param _axis: The current matplotlib subplot axis to format.
    :type _axis: matplotlib.axes._subplots.AxesSubplot
    :param _i_column: The column index for this subplot
    :type _i_column: int
    :param _i_axis: The index of this axis.
    :type _i_axis: int
    """

    if _metadata.is_annotate_subplot:
        _axis.set_title('(' + st.ascii_lowercase[_i_axis] + ')', loc='center')

    _axis.set_xlabel('$x$', visible=_metadata.number_plot - _i_axis <= _metadata.plot_grid_max_columns)
    _axis.set_ylabel('$y$', visible=_i_column == 0)
    _axis.set_xlim(np.min(_vertex_coordinates[:, 0]), np.max(_vertex_coordinates[:, 0]))
    _axis.set_ylim(np.min(_vertex_coordinates[:, 1]), np.max(_vertex_coordinates[:, 1]))
    _axis.set_xticks(np.linspace(np.min(_vertex_coordinates[:, 0]), np.max(_vertex_coordinates[:, 0]), 5))
    _axis.set_yticks(np.linspace(np.min(_vertex_coordinates[:, 1]), np.max(_vertex_coordinates[:, 1]), 5))
    _axis.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)

    _axis.set_aspect('equal', adjustable='box')
    _axis.set_frame_on(True)


def add_field_plot(_vertex_coordinates, _cell_vertex_connectivity, _field, _metadata, _axis):
    """
    Using the PolyCollection the field colored field variables are plotted on the mesh. The color is determined by that
    set in the metadata class.

    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :type _vertex_coordinates: numpy.array
    :param _cell_vertex_connectivity:
    :param _field: The field values.
    :type _field: numpy.ndarray
    :param _metadata: The figure, subplot, and field plotting metadata containing the setting to use.
    :type _metadata: FieldPlotMetadata
    :param _axis: The current matplotlib subplot axis to add the plot.
    :type _axis: matplotlib.axes._subplots.AxesSubplot
    """

    pc = cl.PolyCollection(_vertex_coordinates[_cell_vertex_connectivity], cmap=_metadata.color_map)
    pc.set_array(_field if len(_field.shape) == 1 else np.linalg.norm(_field, axis=1))
    _axis.add_collection(pc)


def add_field_vector(_cell_centroid, _field, _metadata, _axis):
    """
    Adds direction arrows to each cell for vector fields. For each cell an arrow head is added with the 'tail' attached
    to the centroid of cell. Vector field are automatically detected by checking the length of the returned np.shape
    tuple if it is 2 it is assumed a vector field.

    :param _cell_centroid: Co-ordinates for all cell centroids in the mesh, of the form [i_cell][x, y coordinates]
    :type _cell_centroid: numpy.array
    :param _field: The field values.
    :type _field: numpy.ndarray
    :param _metadata: The figure, subplot, and field plotting metadata containing the setting to use.
    :type _metadata: FieldPlotMetadata
    :param _axis: The current matplotlib subplot axis to add the vectors.
    :type _axis: matplotlib.axes._subplots.AxesSubplot
    """

    if len(_field.shape) == 2:
        normal_vectors = _field / np.linalg.norm(_field, axis=1)[:, None]
        _axis.quiver(*[_cell_centroid[:, 0], _cell_centroid[:, 1]], normal_vectors[:, 0], normal_vectors[:, 1],
                     color=_metadata.vector_arrow_color, scale=30.0)


def add_field_color_bar(_figure, _figure_grid, _field, _metadata):
    """
    Adds a color bar to the figure, automatically scales the color bar based on the minimum and maximum values in the
    parsed field data. For vectors fields the norms are used. The color bar is added to the last grid row of the figure.
    It is thus important to leave this row free from other axis.

    :param _figure: The matplotlib figure on which to add the color bar.
    :type _figure: matplotlib.figure.Figure
    :param _figure_grid: The grid for the figure, must have the last row empty so that the color bar can be added.
    :type _figure_grid: matplotlib.gridspec.GridSpec
    :param _field: The field values.
    :type _field: numpy.ndarray
    :param _metadata: The figure, subplot, and field plotting metadata containing the setting to use.
    :type _metadata: FieldPlotMetadata
    """

    color_map = cm.ScalarMappable(norm=None, cmap=_metadata.color_map)
    min_max = [sys.float_info.max, sys.float_info.min]
    for fld in _field:
        if len(fld.shape) == 1:
            min_max = [min(np.min(fld), min_max[0]), max(np.max(fld), min_max[1])]
        else:
            norm = np.linalg.norm(fld, axis=1)
            min_max = [min(np.min(norm), min_max[0]), max(np.max(norm), min_max[1])]

    color_map.set_clim(vmin=min_max[0], vmax=min_max[1])
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    axis = _figure.add_subplot(_figure_grid[-1, :], in_layout=True)
    plt.colorbar(color_map, cax=axis, label=_metadata.color_bar_label, orientation='horizontal', format=formatter,
                 ticklocation='bottom', ticks=np.linspace(min_max[0], min_max[1], 5))


def setup_field_figure(_metadata):
    """
    Sets up the figure, subplot grid, and un-formatted axis. The primary purpose of the function is try and maintain a
    good layout no matter how many subplots are requested and the number of plots per row. Note that at present it is
    not recommended to go beyond 4 plots per row.

    :param _metadata: The figure, subplot, and field plotting metadata containing the setting to use.
    :type _metadata: FieldPlotMetadata
    :return: The matplotlib [figure, grid, and list of axis]
    :type: [matplotlib.figure.Figure, matplotlib.gridspec.GridSpec, numpy.ndarray]
    """

    centimeter = 1.0 / 2.54  # centimeters in inches
    _metadata.number_row = int(np.ceil(np.array([_metadata.number_plot / _metadata.plot_grid_max_columns]))[0])
    _metadata.number_column = min(_metadata.plot_grid_max_columns, _metadata.number_plot)

    color_bar_height = 0.25 * centimeter
    width = 0.95 * _metadata.figure_page_width_cm * centimeter
    height = ((_metadata.figure_page_width_cm * centimeter) * float(_metadata.number_row)
              / float(_metadata.number_column))
    total_height = (height + color_bar_height)

    height_ratios = [1.0 / float(_metadata.number_row) * height / total_height for _ in range(_metadata.number_row)]
    height_ratios.append(color_bar_height / total_height)

    figure = plt.figure(figsize=(width, total_height), constrained_layout=True)
    figure_grid = figure.add_gridspec(nrows=_metadata.number_row + 1, ncols=_metadata.number_column, figure=figure,
                                      height_ratios=height_ratios)

    ax_list = [[0 for _ in range(_metadata.number_column)] for _ in range(_metadata.number_row)]
    for row in range(_metadata.number_row):
        for col in range(_metadata.number_column):
            ax_list[row][col] = figure.add_subplot(figure_grid[row, col], in_layout=True)
    ax_list = np.asarray(ax_list)

    return figure, figure_grid, ax_list


def plot_field(_mesh, _field_list, _metadata=FieldPlotMetadata):
    """
    Given a mesh and associated (list of) fields a set of plots are created. The _metadata allows for some
    customisation of the plot. The plot can be saved to file or show interactively or both depending on the settings
    specified in _metadata.

    :param _mesh: The mesh containing the geometric data associated with the parsed fields.
    :type _mesh: mesh.CellCenteredMesh
    :param _field_list: List of field variables, can be mix of scalar and vector fields.
    :type _field_list: list
    :param _metadata: The figure, subplot, and field plotting metadata containing the setting to use.
    :type _metadata: FieldPlotMetadata
    """

    field = [_field_list] if type(_field_list) is not list else _field_list
    _metadata.number_plot = len(field)

    figure, grid, axis = setup_field_figure(_metadata)
    add_field_color_bar(figure, grid, field, _metadata)

    i_axis = 0
    for i_row in range(_metadata.number_row):
        for i_column in range(_metadata.number_column):
            if i_axis >= _metadata.number_plot:
                axis[i_row][i_column].set_visible(False)
                continue

            format_field_subplot_axis(_mesh.vertex_table.coordinate, _metadata, axis[i_row][i_column], i_column, i_axis)
            add_field_plot(_mesh.vertex_table.coordinate, _mesh.cell_table.connected_vertex, field[i_axis], _metadata,
                           axis[i_row][i_column])
            if _metadata.add_vector_arrows:
                add_field_vector(_mesh.cell_table.centroid, field[i_axis], _metadata, axis[i_row][i_column])

            i_axis += 1

    # Note must save before, showing - showing clears the plot and there is nothing to save.
    if _metadata.is_save_plot:
        save_plot(_metadata.figure_title)

    if _metadata.is_show_plot:
        show_plot()
