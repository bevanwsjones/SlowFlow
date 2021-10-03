import matplotlib as mpl
#mpl.use('pgf')
from matplotlib import pyplot as plt

# return string name for error metric for plotting purposes
def grid_identity(gq):
    if gq == 0:
        return str("Non-orthogonal")
    elif gq == 1:
        return str("Uneven")
    elif gq == 2:
        return str("Skew")
    else:
        return ValueError("Invalid grid quality selection")

# plot the cartesian
def cartesian_plotter(int_error_array, bound_error_array, h):
    # plot 1 - x-gradient error - internal cells
    plt.subplot(2, 2, 1)
    plt.plot(h, int_error_array[:, 0, 0], '-o', label='L1 norm')
    plt.plot(h, int_error_array[:, 1, 0], '-ok', label='L2 norm')
    plt.plot(h, int_error_array[:, 2, 0], '-or', label='Linf norm')
    #plt.axline((.02, .0001), slope=2, color='C0', label='2nd-order', ls='--')
    #plt.axline((.02, .0001), slope=1, color='C0', label='1st-order', ls='--')
    plt.xlabel("Characteristic Length")
    plt.ylabel("x-gradient internal cell error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    # plot 2 - y-gradient error - internal cells
    plt.subplot(2, 2, 2)
    plt.plot(h, int_error_array[:, 0, 1], '-o', label='L1 norm')
    plt.plot(h, int_error_array[:, 1, 1], '-ok', label='L2 norm')
    plt.plot(h, int_error_array[:, 2, 1], '-or', label='Linf norm')
    plt.axline((.02, .0001), slope=2, color='C0', label='2nd-order', ls='--')
    plt.axline((.02, .0001), slope=1, color='C0', label='1st-order', ls='--')
    plt.xlabel("Characteristic Length")
    plt.ylabel("y-gradient internal cell error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    # plot 3 - x-gradient error - boundary cells
    plt.subplot(2, 2, 3)
    plt.plot(h, bound_error_array[:, 0, 0], '-o', label='L1 norm')
    plt.plot(h, bound_error_array[:, 1, 0], '-ok', label='L2 norm')
    plt.plot(h, bound_error_array[:, 2, 0], '-or', label='Linf norm')
    plt.axline((.02, .0001), slope=2, color='C0', label='2nd-order', ls='--')
    plt.axline((.02, .0001), slope=1, color='C0', label='1st-order', ls='--')
    plt.xlabel("Characteristic Length")
    plt.ylabel("x-gradient bound cell error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    # plot 4 - y-gradient error - boundary cells
    plt.subplot(2, 2, 4)
    plt.plot(h, bound_error_array[:, 0, 1], '-o', label='L1 norm')
    plt.plot(h, bound_error_array[:, 1, 1], '-ok', label='L2 norm')
    plt.plot(h, bound_error_array[:, 2, 1], '-or', label='Linf norm')
    plt.axline((.02, .0001), slope=2, color='C0', label='2nd-order', ls='--')
    plt.axline((.02, .0001), slope=1, color='C0', label='1st-order', ls='--')
    plt.xlabel("Characteristic Length")
    plt.ylabel("y-gradient bound cell error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.suptitle("2D Structured Cartesian Grid Error Analysis")
    plt.tight_layout
    plt.savefig('CartesianGridError.pgf', format='pgf')
    print("Complete")
    plt.show()
    return -1

def error_plotter(bound_error_array, int_error_array, grid_quality_array, i, i_matrix, gq = 0):
    name = grid_identity(gq)
    # - x-gradient error - internal cells
    plot1 = plt.figure(1)
    plt.subplot(3, 2, 1 + 2*i)
    plt.plot(grid_quality_array[0], int_error_array[:, 0, 0], 'o',label='L1 norm')
    plt.plot(grid_quality_array[0], int_error_array[:, 1, 0], 'ok', label='L2 norm')
    plt.plot(grid_quality_array[0], int_error_array[:, 2, 0], 'or', label='Linf norm')
    plt.xlabel(name+" Metric")
    plt.ylabel("x-Gradient Cell Error")
    #plt.xscale("log")
    #plt.yscale("log")
    plt.legend()
    plt.title(str(i_matrix)+" grid")

    plt.subplot(3, 2, 2 + 2*i)
    plt.plot(grid_quality_array[0], bound_error_array[:, 0, 0], 'o',label='L1 norm')
    plt.plot(grid_quality_array[0], bound_error_array[:, 1, 0], 'ok', label='L2 norm')
    plt.plot(grid_quality_array[0], bound_error_array[:, 2, 0], 'or', label='Linf norm')
    plt.xlabel(name+" Metric")
    plt.ylabel("x-Gradient Cell Error")
    #plt.xscale("log")
    #plt.yscale("log")
    plt.legend()
    plt.title(str(i_matrix)+" grid")
    plt.suptitle(name +" 2D Cartesian Mesh Internal (Left) and Boundary (Right) Cells Analysis")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)
    # plt.subplot(3, 2, 2 + 2*i)
    # plt.plot(grid_quality_array[0], int_error_array[:, 0, 1], 'o',label='L1 norm')
    # plt.plot(grid_quality_array[0], int_error_array[:, 1, 1], 'ok', label='L2 norm')
    # plt.plot(grid_quality_array[0], int_error_array[:, 2, 1], 'or', label='Linf norm')
    # plt.xlabel(name+" Metric")
    # plt.ylabel("y-Gradient Error")
    # #plt.xscale("log")
    # plt.yscale("log")
    # plt.legend()
    # plt.title(str(i_matrix)+" grid")



    plot2 = plt.figure(2)
    plt.subplot(3, 2, 1 + 2*i)
    plt.plot(grid_quality_array[0], bound_error_array[:, 0, 0], 'o',label='L1 norm')
    plt.plot(grid_quality_array[0], bound_error_array[:, 1, 0], 'ok', label='L2 norm')
    plt.plot(grid_quality_array[0], bound_error_array[:, 2, 0], 'or', label='Linf norm')
    plt.xlabel(name+" Metric")
    plt.ylabel("x-Gradient Error")
    #plt.xscale("log")
    #plt.yscale("log")
    plt.legend()
    plt.title(str(i_matrix)+" grid")

    plt.subplot(3, 2, 2 + 2*i)
    plt.plot(grid_quality_array[0], bound_error_array[:, 0, 1], 'o',label='L1 norm')
    plt.plot(grid_quality_array[0], bound_error_array[:, 1, 1], 'ok', label='L2 norm')
    plt.plot(grid_quality_array[0], bound_error_array[:, 2, 1], 'or', label='Linf norm')
    plt.xlabel(name+" Metric")
    plt.ylabel("y-Gradient Error")
    #plt.xscale("log")
    #plt.yscale("log")
    plt.legend()
    plt.title(str(i_matrix)+" grid")

    plt.suptitle(name +" 2D Cartesian Mesh Boundary Cells Analysis")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)
    return -1

def grid_error_refine(int_error_array, bound_error_array, h, grid_quality, gq = 0):
    name = grid_identity(gq)
    # plot 1 - x-gradient error - internal cells
    plt.figure(1)
    plt.subplot(1, 2, 1)
    for i, i_metric in enumerate(grid_quality):
        plt.plot(h, int_error_array[i, :, 0], '-o', label=str(i_metric))
    plt.axline((.02, .0001), slope=2, color='C0', label='by slope', ls='--')
    plt.axline((.02, .0001), slope=1, color='C0', label='by slope', ls='--')
    plt.xlabel("Characteristic Length")
    plt.ylabel("x-gradient internal cell error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    # plot 2 - y-gradient error - internal cells
    plt.subplot(1, 2, 2)
    for i, i_metric in enumerate(grid_quality):
        plt.plot(h, bound_error_array[i, :, 0], '-o', label=str(i_metric))
    plt.axline((.02, .0001), slope=2, color='C0', label='by slope', ls='--')
    plt.axline((.02, .0001), slope=1, color='C0', label='by slope', ls='--')
    plt.xlabel("Characteristic Length")
    plt.ylabel("x-gradient external cell error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.suptitle("Error for different "+name+" ratios as a function of mesh refinement")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)

    # plt.subplot(1, 2, 2)
    plt.figure(2)
    plt.subplot(1, 2, 1)
    for i, i_metric in enumerate(grid_quality):
        plt.plot(h, int_error_array[i, :, 1], '-o', label=str(i_metric))
    plt.axline((.02, .0001), slope=2, color='C0', label='by slope', ls='--')
    plt.axline((.02, .0001), slope=1, color='C0', label='by slope', ls='--')
    plt.xlabel("Characteristic Length")
    plt.ylabel("y-gradient internal cell error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    # plot 2 - y-gradient error - internal cells
    plt.subplot(1, 2, 2)
    for i, i_metric in enumerate(grid_quality):
        plt.plot(h, bound_error_array[i, :, 1], '-o', label=str(i_metric))
    plt.axline((.02, .0001), slope=2, color='C0', label='by slope', ls='--')
    plt.axline((.02, .0001), slope=1, color='C0', label='by slope', ls='--')
    plt.xlabel("Characteristic Length")
    plt.ylabel("y-gradient external cell error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.suptitle("Error for different " + name + " as a function of mesh refinement")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.1)
    return -1