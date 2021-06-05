import numpy as np
from compressible_solver import face as fc
from compressible_solver import node as nd
from compressible_solver import vertex as vt
import meshio
import dmsh
import optimesh


def generete_1d_mesh(max_node, domain_size):
    mnode = max_node[0]
    node_table = np.array([nd.CompressibleNode(1) for i in range(mnode)], dtype=object)

    mface = (max_node[0] + 1)
    face_table = np.array([fc.Face() for i in range(mface)], dtype=object)

    print("Generating 1D mesh with:")
    print("Domain length:   " + str(domain_size[0]))
    print("Number of nodes: " + str(mnode))
    print("Number of faces: " + str(mface))

    setup_1d_mesh_connectivity(max_node, node_table, face_table)
    setup_1d_finite_volume_mesh(domain_size, max_node, node_table, face_table)
    return node_table, face_table


def setup_1d_mesh_connectivity(max_node, node_table, face_table):
    for inode in range(max_node[0]):
        node_table[inode].connected_face[0] = inode
        node_table[inode].connected_face[1] = inode + 1
        face_table[node_table[inode].connected_face[0]].connected_cell[1] = inode
        face_table[node_table[inode].connected_face[1]].connected_cell[0] = inode

        if inode == 0 or inode == (max_node[0] - 1):
            node_table[inode].boundary = True
        else:
            node_table[inode].boundary = False
        node_table[inode].corner = False

    first_last_node = np.full([max_node[0], 2], fill_value=-1, dtype=int)
    for iface, face in enumerate(face_table):
        face.boundary = face.connected_cell[0] == -1 or face.connected_cell[1] == -1

        if face.connected_cell[0] != -1 and first_last_node[face.connected_cell[0]][0] == -1:
            first_last_node[face.connected_cell[0]][0] = iface

        if face.connected_cell[1] != -1 and first_last_node[face.connected_cell[1]][0] == -1:
            first_last_node[face.connected_cell[1]][0] = iface

        if (face.connected_cell[0] != -1 and first_last_node[face.connected_cell[0]][1] < iface) or \
                (face.connected_cell[1] != -1 and first_last_node[face.connected_cell[1]][1] < iface):
            first_last_node[face.connected_cell[0]][1] = iface

        # Re-orient 'negative faces' so that boundary faces always have a node at zero.
        if face.connected_cell[0] == -1:
            face.connected_cell[0] = face.connected_cell[1]
            face.connected_cell[1] = -1

    for inode, first_last in enumerate(first_last_node):
        face_table[first_last[0]].node_first = np.append(face_table[first_last[0]].node_first, inode)
        face_table[first_last[1]].node_last = np.append(face_table[first_last[1]].node_last, inode)
    return


def setup_1d_finite_volume_mesh(domain_size, max_node, node_table, face_table):
    delta_x = domain_size[0] / float(max_node[0] - 1)
    delta_y = 1.0
    node_volume = delta_x

    inode = 0
    for inode in range(max_node[0]):
        node_table[inode].coordinate = np.array([float(inode) * delta_x])
        node_table[inode].volume = node_volume
        if inode == 0:
            node_table[inode].volume *= 0.5
        if inode == (max_node[0] - 1):
            node_table[inode].volume *= 0.5

    for iface, face in enumerate(face_table):
        face.coefficient = np.array([delta_y, 0.0]) if iface != 0 else np.array([-delta_y, 0.0])
        face.tangent = np.array([1.0, 0.0]) if not face.boundary else np.array([0.0, 0.0])

        if face.connected_cell[0] != -1 and face.connected_cell[1] != -1:
            face.length = np.linalg.norm(node_table[face.connected_cell[1]].coordinate
                                         - node_table[face.connected_cell[0]].coordinate)
            if node_table[face.connected_cell[0]].boundary and node_table[face.connected_cell[1]].boundary:
                face.coefficient *= 0.5
        elif face.connected_cell[0] != -1 and node_table[face.connected_cell[0]].corner:
            face.coefficient *= 0.5
        elif face.connected_cell[1] != -1 and node_table[face.connected_cell[1]].corner:
            face.coefficient *= 0.5
    return


# ----------------------------------------------------------------------------------------------------------------------
# 2D Mesh Generation
# ----------------------------------------------------------------------------------------------------------------------

def generate_2d_cartesian_mesh(max_node, domain_size):
    print(max_node)
    mnode = max_node[0] * max_node[1]
    node_table = np.array([nd.CompressibleNode(2) for i in range(mnode)], dtype=object)

    mface = (max_node[0]) * (max_node[1] + 1) + (max_node[0] + 1) * (max_node[1])
    face_table = np.array([fc.Face() for i in range(mface)], dtype=object)

    print("Generating 2D structured mesh with:")
    print("Domain volume:   " + str(domain_size[0] * domain_size[1]) + "(" + str(domain_size[0]) + "x" + str(
        domain_size[1]) + ")")
    print("Number of nodes: " + str(mnode) + "(" + str(max_node[0]) + "x" + str(max_node[1]) + ")")
    print("Number of faces: " + str(mface))

    setup_2d_cartesian_mesh_connectivity(max_node, node_table, face_table)
    setup_2d_finite_volume_cartesian_mesh(domain_size, max_node, node_table, face_table)

    return node_table, face_table


def setup_2d_cartesian_mesh_connectivity(max_node, node_table, face_table):
    mnode = max_node[0] * max_node[1]
    first_horizontal_face = (max_node[0] + 1) * (max_node[1])

    inode = 0
    for inodey in range(max_node[1]):
        for inodex in range(max_node[0]):

            node_table[inode].connected_face[0] = inode + inodey
            node_table[inode].connected_face[1] = inode + inodey + 1
            node_table[inode].connected_face[2] = first_horizontal_face + inode
            node_table[inode].connected_face[3] = first_horizontal_face + inode + max_node[0]

            face_table[node_table[inode].connected_face[0]].connected_cell[1] = inode
            face_table[node_table[inode].connected_face[1]].connected_cell[0] = inode
            face_table[node_table[inode].connected_face[2]].connected_cell[1] = inode
            face_table[node_table[inode].connected_face[3]].connected_cell[0] = inode

            if inodex == 0 or inodex == (max_node[0] - 1) or inodey == 0 or inodey == (max_node[1] - 1):
                node_table[inode].boundary = True
                node_table[inode].corner = (inode == 0) or (inode == max_node[0] - 1) or \
                                           (inode == max_node[0] * (max_node[1] - 1)) or (inode == mnode - 1)
            else:
                node_table[inode].boundary = False
                node_table[inode].corner = False
            inode = inode + 1

    first_last_node = np.full([mnode, 2], fill_value=-1, dtype=int)
    for iface, face in enumerate(face_table):
        face.boundary = face.connected_cell[0] == -1 or face.connected_cell[1] == -1

        if face.connected_cell[0] != -1 and first_last_node[face.connected_cell[0]][0] == -1:
            first_last_node[face.connected_cell[0]][0] = iface

        if face.connected_cell[1] != -1 and first_last_node[face.connected_cell[1]][0] == -1:
            first_last_node[face.connected_cell[1]][0] = iface

        if (face.connected_cell[0] != -1 and first_last_node[face.connected_cell[0]][1] < iface) or \
                (face.connected_cell[1] != -1 and first_last_node[face.connected_cell[1]][1] < iface):
            first_last_node[face.connected_cell[0]][1] = iface

        # Re-orient 'negative faces' so that boundary faces always have a node at zero.
        if face.connected_cell[0] == -1:
            face.connected_cell[0] = face.connected_cell[1]
            face.connected_cell[1] = -1

    for inode, first_last in enumerate(first_last_node):
        face_table[first_last[0]].node_first = np.append(face_table[first_last[0]].node_first, inode)
        face_table[first_last[1]].node_last = np.append(face_table[first_last[1]].node_last, inode)
    return


def setup_2d_finite_volume_cartesian_mesh(domain_size, max_node, node_table, face_table):
    delta_x = domain_size[0] / float(max_node[0] - 1)
    delta_y = domain_size[1] / float(max_node[1] - 1)
    node_volume = delta_x * delta_y
    first_horizontal_face = (max_node[0] + 1) * (max_node[1])

    inode = 0
    for inodey in range(max_node[1]):
        for inodex in range(max_node[0]):
            node_table[inode].coordinate = np.array([float(inodex) * delta_x, float(inodey) * delta_y])
            node_table[inode].volume = node_volume
            if inodex == 0:
                node_table[inode].volume *= 0.5
            if inodex == (max_node[0] - 1):
                node_table[inode].volume *= 0.5
            if inodey == 0:
                node_table[inode].volume *= 0.5
            if inodey == (max_node[1] - 1):
                node_table[inode].volume *= 0.5
            inode += 1

    for iface, face in enumerate(face_table):
        if iface < first_horizontal_face:
            is_negative_face = node_table[face.connected_cell[0]].coordinate[0] == 0 and face.boundary
            face.coefficient = np.array([0.0, delta_y]) if not is_negative_face else np.array([0.0, -delta_y])
            face.tangent = np.array([1.0, 0.0]) if not face.boundary else np.array([0.0, 0.0])
        else:
            is_negative_face = node_table[face.connected_cell[0]].coordinate[1] == 0 and face.boundary
            face.coefficient = np.array([delta_x, 0.0]) if not is_negative_face else np.array([-delta_x, 0.0])
            face.tangent = np.array([0.0, 1.0]) if not face.boundary else np.array([0.0, 0.0])

        if face.connected_cell[0] != -1 and face.connected_cell[1] != -1:
            face.length = np.linalg.norm(node_table[face.connected_cell[1]].coordinate
                                         - node_table[face.connected_cell[0]].coordinate)
            if node_table[face.connected_cell[0]].boundary and node_table[face.connected_cell[1]].boundary:
                face.coefficient *= 0.5
        elif face.connected_cell[0] != -1 and node_table[face.connected_cell[0]].corner:
            face.coefficient *= 0.5
        elif face.connected_cell[1] != -1 and node_table[face.connected_cell[1]].corner:
            face.coefficient *= 0.5
    return

# ----------------------------------------------------------------------------------------------------------------------
# 1D Mesh Generation
# ----------------------------------------------------------------------------------------------------------------------


def setup_1d_unit_mesh(max_cell):
    """

    :param max_cell:
    :return:
    """

    # Size tables
    vertex_table = vt.VertexTable((max_cell + 1)*2)
    face_table = fc.FaceTable(max_cell + 1)
    cell_table = nd.CellTable(max_cell)
    cell_table.add_ghost_cells(2)
    cell_table.connected_face = np.delete(cell_table.connected_face, 2, 1)
    delta_x = 1.0/float(max_cell)

    # Setup vertex connectivity and co-ordinates
    for ivertex in range(int(vertex_table.max_vertex/2)):
        ivertex0 = 2 * ivertex
        ivertex1 = 2 * ivertex + 1
        vertex_table.coordinate[ivertex0] = np.array([float(ivertex) * delta_x, 0.0])
        vertex_table.coordinate[ivertex1] = np.array([float(ivertex) * delta_x, 1.0])

        if ivertex == 0:
            # index to the index of the first cell and first ghost cell
            vertex_table.connected_cell[ivertex0] = np.array([0, cell_table.max_cell])
            vertex_table.connected_cell[ivertex1] = np.array([0, cell_table.max_cell])
        elif ivertex == (int(vertex_table.max_vertex/2) - 1):
            # connect to the index of the last cell and last ghost cell
            vertex_table.connected_cell[ivertex0] = np.array([cell_table.max_cell - 1, cell_table.max_cell + 1])
            vertex_table.connected_cell[ivertex1] = np.array([cell_table.max_cell - 1, cell_table.max_cell + 1])
        else:
            vertex_table.connected_cell[ivertex0] = np.array([ivertex - 1, ivertex])
            vertex_table.connected_cell[ivertex1] = np.array([ivertex - 1, ivertex])

    # Setup face connectivity and coefficients
    for iface in range(face_table.max_face):
        face_table.connected_vertex[iface] = np.array([iface * 2, iface * 2 + 1])
        face_table.length[iface] = delta_x

        if iface == 0:
            face_table.boundary[iface] = True
            face_table.connected_cell[iface] = np.array([0, cell_table.max_cell])  # index to the first ghost cell
            face_table.coefficient[iface] = np.array([-1.0, 0.0])
        elif iface == (face_table.max_face - 1):
            face_table.boundary[iface] = True
            face_table.connected_cell[iface] = np.array([cell_table.max_cell - 1, cell_table.max_cell + 1])  # index to last ghost cell
            face_table.coefficient[iface] = np.array([1.0, 0.0])
        else:
            face_table.boundary[iface] = False
            face_table.connected_cell[iface] = np.array([iface - 1, iface])
            face_table.coefficient[iface] = np.array([1.0, 0.0])

    # Setup ghost cell connectivity
    for icell in range(cell_table.max_cell):
        cell_table.connected_face[icell] = np.array([icell, icell + 1])
        cell_table.volume[icell] = delta_x
        cell_table.coordinate[icell] = np.array([float(icell * 2 + 1) * 0.5 * delta_x, 0.5])

    # Hard code values for ghost cells.
    cell_table.connected_face[cell_table.max_cell] = np.array([0, -1])
    cell_table.volume[cell_table.max_cell] = delta_x
    cell_table.coordinate[cell_table.max_cell] = np.array([-0.5 * delta_x, 0.5])
    cell_table.connected_face[cell_table.max_cell + 1] = np.array([face_table.max_face - 1, -1])
    cell_table.volume[cell_table.max_cell + 1] = delta_x
    cell_table.coordinate[cell_table.max_cell + 1] = np.array([1 + 0.5 * delta_x, 0.5])

    return cell_table, face_table, vertex_table

# ----------------------------------------------------------------------------------------------------------------------
# 2D Mesh Generation Simplex Mesh
# ----------------------------------------------------------------------------------------------------------------------


def find_face_connected_cells(icell, vertex0_connected_cells, vertex1_connected_cells):
    """
    Determines the cell connected to icell given the list of cells connected to two vertices of one of the faces of
    icell. By looking for common cells in vertex0's and vertex1's connectivity the cell connected to icell can be found.
    The return value is two integers indicating the two cells connected through the common face. If no connection is
    found it is assumed that this cell face is a boundary cell face, in which case both values in the returned array are
    icell.

    :param icell: The cell index.
    :param vertex0_connected_cells: The list of cells connected to vertex 0 of the face of the cell.
    :param vertex1_connected_cells: The list of cells connected to vertex 1 of the face the cell.
    :return: [icell, iconnected cell] if internal face else [icell, icell].
    """

    cells_connected_to_face = np.empty(shape=2, dtype=int)
    is_connection_found = False

    for icell_vertex0 in vertex0_connected_cells:
        if icell_vertex0 != icell:
            connected_cell = np.where(vertex1_connected_cells == icell_vertex0)[0]

            # If the returned array is not of zero size we have found a connection.
            if len(connected_cell) != 0:
                if len(connected_cell) != 1:
                    raise ValueError("More than one face found when connecting cells.")

                cells_connected_to_face = np.array([icell, vertex1_connected_cells[connected_cell[0]]])
                is_connection_found = True

        # This is a boundary face - mark by connecting the cell to itself.
        if icell_vertex0 == vertex0_connected_cells[-1] and not is_connection_found:
            cells_connected_to_face = np.array([icell, icell])

    return cells_connected_to_face


def create_face_cell_vertex_connectivity(cells):
    """
    Given a list of cells computes the face-cell and face-vertex connectivity tables.

    :param cells: List of cells, each row is should be a list of the vertex indices.
    :return: face-cell connectivity table, face-vertex connectivity table.
    """

    face_cell_connectivity = []
    face_vertex_connectivity = []
    for icell, cell in enumerate(cells):

        for ivert in range(len(cell)):
            vert0_cell_connection = np.where(cells == cell[ivert])[0]
            vert1_cell_connection = np.where(cells == cell[(ivert + 1) % len(cell)])[0]
            face_cell_connection = find_face_connected_cells(icell, vert0_cell_connection, vert1_cell_connection)
            if face_cell_connection[0] <= face_cell_connection[1]:
                face_cell_connectivity.append(face_cell_connection)
                face_vertex_connectivity.append(np.array([cell[ivert], cell[(ivert + 1) % len(cell)]]))
    face_cell_connectivity = np.asarray(face_cell_connectivity)
    return face_cell_connectivity, face_vertex_connectivity


def add_ghost_cell(cell_table, face_table, vertex_table):
    """
    Adds ghost cells to the cell table and updates the connectivity tables for the cell, face, and vertex tables. The
    ghost cells are appended to the end of the cell table.

    :param cell_table: Cell table onto which the ghost cells will be added and connectivity updated.
    :param face_table: Face table which will be connected to the added ghost cells.
    :param vertex_table: Vertex table which will be connected to the added ghost cells.
    """

    unique, counts = np.unique(face_table.boundary, return_counts=True)
    ghost_node_start = cell_table.max_cell
    if unique[1] == True:
        cell_table.add_ghost_cells(counts[1])
    else:
        raise ValueError("No boundary faces found.")

    # loop over faces and link up the new ghost nodes
    for iface in range(face_table.max_face):
        if face_table.boundary[iface]:
            ivertex0 = face_table.connected_vertex[iface][0]
            ivertex1 = face_table.connected_vertex[iface][1]
            face_table.connected_cell[iface][1] = ghost_node_start
            cell_table.connected_face[ghost_node_start][0] = iface
            cell_table.connected_vertex[ghost_node_start][0] = ivertex0
            cell_table.connected_vertex[ghost_node_start][1] = ivertex1
            vertex_table.connected_cell[ivertex0] = np.concatenate((vertex_table.connected_cell[ivertex0], np.array([ghost_node_start])))
            vertex_table.connected_cell[ivertex1] = np.concatenate((vertex_table.connected_cell[ivertex1], np.array([ghost_node_start])))
            ghost_node_start += 1


def setup_connectivity():
    """
    Sizes the data containers to describe a cell-centred Finite-Volume mesh and creates connectivity structures
    describing the local neighbourhood of a given mesh entity (cell, face, vertex). Additionally ghost nodes are added
    and boundary status resolved.

    :return: The constructed cell, face, and vertex tables.
    """
    geo = dmsh.Polygon([[0.0, 0.0], [1.0, 0.0], [1.0 + np.cos(np.pi/3.0), np.sin(np.pi/3.0)],
                        [1.0, 2.0*np.sin(np.pi/3.0)], [0.0, 2.0*np.sin(np.pi/3.0)], [0.0 - np.cos(np.pi/3.0), np.sin(np.pi/3.0)]])
    verticies, cells = dmsh.generate(geo, 0.5)
    dmsh.helpers.show(verticies, cells, geo)  # Put on to print mesh
    face_cell_connectivity, face_vertex_connectivity = create_face_cell_vertex_connectivity(cells)

    # Size tables
    vertex_table = vt.VertexTable(len(verticies))
    cell_table = nd.CellTable(len(cells))
    face_table = fc.FaceTable(len(face_cell_connectivity))

    # Construct cell face connectivity.
    for iface in range(face_table.max_face):
        face_table.connected_cell[iface][0] = face_cell_connectivity[iface][0]
        face_table.connected_cell[iface][1] = face_cell_connectivity[iface][1]
        face_table.connected_vertex[iface][0] = face_vertex_connectivity[iface][0]
        face_table.connected_vertex[iface][1] = face_vertex_connectivity[iface][1]
        face_table.boundary[iface] = face_table.connected_cell[iface][0] == face_table.connected_cell[iface][1]

        inode0 = face_table.connected_cell[iface][0]
        inode1 = face_table.connected_cell[iface][1]

        cell_table.connected_face[inode0][np.where(cell_table.connected_face[inode0] == -1)[0][0]] = iface
        if not face_table.boundary[iface]:
            cell_table.connected_face[inode1][np.where(cell_table.connected_face[inode1] == -1)[0][0]] = iface
        else:
            cell_table.boundary[face_table.connected_cell[iface][0]] = True

    # Build the vertex_cell connectivity and copy the cell_vertex connectivity
    temp = [[] for _ in range(len(verticies))]
    for icell, cell in enumerate(cells):
        for ivertex in cell:
            temp[ivertex].append(icell)
        cell_table.connected_vertex[icell] = np.asarray(cell)

    # Copy data to the vertex table.
    for ivertex, vertex in enumerate(verticies):
        vertex_table.coordinate[ivertex] = np.array([vertex])
        vertex_table.connected_cell[ivertex] = np.asarray(temp[ivertex])

    # Add the ghost cells.
    add_ghost_cell(cell_table, face_table, vertex_table)

    print("WARNING: no cell first/last has been set up yet.")
    return cell_table, face_table, vertex_table


def setup_ghost_cell_geometry(cell_table, face_table, vertex_table):

    # loop over faces and link up the new ghost nodes
    for iface in range(face_table.max_face):
        if face_table.boundary[iface]:
            ivertex0 = face_table.connected_vertex[iface][0]
            ivertex1 = face_table.connected_vertex[iface][1]
            vect_cell_vertex = vertex_table.coordinate[ivertex0] \
                               - cell_table.coordinate[face_table.connected_cell[iface, 0]]
            face_vector = vertex_table.coordinate[ivertex1] - vertex_table.coordinate[ivertex0]
            face_vector /= np.linalg.norm(face_vector)
            vector_to_face = vect_cell_vertex - np.dot(vect_cell_vertex, face_vector) * face_vector

            cell_table.coordinate[face_table.connected_cell[iface, 1]] = \
                cell_table.coordinate[face_table.connected_cell[iface, 0]] + 2.0 * vector_to_face


def setup_finite_volume_geometry(cell_table, face_table, vertex_table):

    # Create cell geometry
    max_cell = cell_table.max_cell

    cell_table.coordinate[0:max_cell] = (vertex_table.coordinate[cell_table.connected_vertex[0:max_cell, 0]]
                                         + vertex_table.coordinate[cell_table.connected_vertex[0:max_cell, 1]]
                                         + vertex_table.coordinate[cell_table.connected_vertex[0:max_cell, 2]]) / 3.0
    cell_table.volume[0:max_cell] =\
        0.5*np.cross(vertex_table.coordinate[cell_table.connected_vertex[0:max_cell, 1]]
                     - vertex_table.coordinate[cell_table.connected_vertex[0:max_cell, 0]],
                     vertex_table.coordinate[cell_table.connected_vertex[0:max_cell, 2]]
                     - vertex_table.coordinate[cell_table.connected_vertex[0:max_cell, 0]])
    setup_ghost_cell_geometry(cell_table, face_table, vertex_table)

    # Create Face Geometry
    face_table.length[:] = np.linalg.norm(cell_table.coordinate[face_table.connected_cell[:, 1]]
                                          - cell_table.coordinate[face_table.connected_cell[:, 0]], axis=1)
    face_table.tangent[:] = (cell_table.coordinate[face_table.connected_cell[:, 1]]
                             - cell_table.coordinate[face_table.connected_cell[:, 0]])/face_table.length[:, None]
    face_normal = np.delete(np.cross(np.concatenate((vertex_table.coordinate[face_table.connected_vertex[:, 1]],
                                                     np.zeros(shape=(face_table.max_face, 1))), axis=1)
                                     - np.concatenate((vertex_table.coordinate[face_table.connected_vertex[:, 0]],
                                                       np.zeros(shape=(face_table.max_face, 1))), axis=1),
                                     np.full((face_table.max_face, 3), [0.0, 0.0, 1.0])), 2, axis=1)
    face_normal = face_normal[:]/np.linalg.norm(face_normal[:], axis=1)[:, None]
    face_table.coefficient =\
        np.linalg.norm(vertex_table.coordinate[face_table.connected_vertex[:, 1]]
                       - vertex_table.coordinate[face_table.connected_vertex[:, 0]], axis=1)[:, None]*face_normal[:]

# ----------------------------------------------------------------------------------------------------------------------
# 2D Mesh Generation Cartesian Mesh
# ----------------------------------------------------------------------------------------------------------------------

def create_cell_vertices(domain_size, number_cells):

    cells = np.zeros([number_cells[0]*number_cells[1], 4], dtype=int)
    vertex = np.zeros([(number_cells[0] + 1)*(number_cells[1] + 1), 2], dtype=float)

    cell_size = [(domain_size[1][0] - domain_size[0][0])/number_cells[0],
                 (domain_size[1][1] - domain_size[0][1])/number_cells[1]]

    ivert = 0
    print(domain_size[0])
    print([cell_size[0], cell_size[1]])
    dX = np.array((cell_size[0], cell_size[1]))
    print((domain_size[0] + dX))
    for irow in range(number_cells[1] + 1):
        for icolu in range(number_cells[0] + 1):
            vertex[ivert] = np.array((domain_size[0] + [cell_size[0]*icolu, cell_size[1]*irow]))
            ivert += 1

    print(vertex)
















