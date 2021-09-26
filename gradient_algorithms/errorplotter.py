from matplotlib import pyplot as plt

def grid_identity(gq):
    if gq == 0:
        return str("Non-orthogonal")
    elif gq == 1:
        return str("Uneven")
    elif gq == 2:
        return str("Skew")
    else:
        return ValueError("Invalid grid quality selection")

def error_plotter(bound_error_array, int_error_array, grid_quality_array, i, i_matrix, gq = 1):
    name = grid_identity(gq)
    # - x-gradient error - internal cells
    plot1 = plt.figure(1)
    plt.subplot(3, 2, 1 + 2*i)
    plt.plot(grid_quality_array[0], int_error_array[:, 0, 0], 'o',label='L1 norm')
    plt.plot(grid_quality_array[0], int_error_array[:, 1, 0], 'ok', label='L2 norm')
    plt.plot(grid_quality_array[0], int_error_array[:, 2, 0], 'or', label='Linf norm')
    plt.xlabel(name+" Metric")
    plt.ylabel("x-Gradient Error")
    #plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title(str(i_matrix)+" grid")

    plt.subplot(3, 2, 2 + 2*i)
    plt.plot(grid_quality_array[0], int_error_array[:, 0, 1], 'o',label='L1 norm')
    plt.plot(grid_quality_array[0], int_error_array[:, 1, 1], 'ok', label='L2 norm')
    plt.plot(grid_quality_array[0], int_error_array[:, 2, 1], 'or', label='Linf norm')
    plt.xlabel(name+" Metric")
    plt.ylabel("y-Gradient Error")
    #plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title(str(i_matrix)+" grid")

    plt.suptitle(name +" 2D Cartesian Mesh Internal Cells Analysis")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)

    plot2 = plt.figure(2)
    plt.subplot(3, 2, 1 + 2*i)
    plt.plot(grid_quality_array[0], bound_error_array[:, 0, 1], 'o',label='L1 norm')
    plt.plot(grid_quality_array[0], bound_error_array[:, 1, 1], 'ok', label='L2 norm')
    plt.plot(grid_quality_array[0], bound_error_array[:, 2, 1], 'or', label='Linf norm')
    plt.xlabel(name+" Metric")
    plt.ylabel("x-Gradient Error")
    #plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title(str(i_matrix)+" grid")

    plt.subplot(3, 2, 2 + 2*i)
    plt.plot(grid_quality_array[0], bound_error_array[:, 0, 0], 'o',label='L1 norm')
    plt.plot(grid_quality_array[0], bound_error_array[:, 1, 0], 'ok', label='L2 norm')
    plt.plot(grid_quality_array[0], bound_error_array[:, 2, 0], 'or', label='Linf norm')
    plt.xlabel(name+" Metric")
    plt.ylabel("y-Gradient Error")
    #plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title(str(i_matrix)+" grid")

    plt.suptitle(name +" 2D Cartesian Mesh Boundary Cells Analysis")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)
    return -1