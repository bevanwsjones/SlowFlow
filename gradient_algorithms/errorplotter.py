from matplotlib import pyplot as plt
from post_processor import graph as gr

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
    plt.subplots(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(h, int_error_array[:, 0, 0], '-o', label='L1 norm')
    plt.plot(h, int_error_array[:, 1, 0], '-ok', label='RMS')
    plt.plot(h, int_error_array[:, 2, 0], '-or', label='Linf norm')
    plt.axline((0.01, 0.0001), (0.1, 0.001), color='g', label='1st-order', ls='--')
    plt.axline((0.01, 0.0001), (0.1, 0.01), color='b', label='2nd-order', ls='--')
    plt.xlabel('$h$')
    plt.ylabel('Internal '+r'$\varepsilon_{x}$')
    plt.xscale("log")
    plt.yscale("log")
    # plt.xlim([5E-3, 2E-1])
    # plt.ylim([1E-5, 1E-2])
    #plt.axis('equal')
    plt.legend()
    plt.title(r'$a )$')
    plt.tight_layout()
    # plot 2 - y-gradient error - internal cells
    plt.subplot(2, 2, 2)
    plt.plot(h, int_error_array[:, 0, 1], '-o', label='L1 norm')
    plt.plot(h, int_error_array[:, 1, 1], '-ok', label='RMS')
    plt.plot(h, int_error_array[:, 2, 1], '-or', label='Linf norm')
    plt.axline((0.01, 0.00001), (0.1, 0.0001), color='g', label='1st-order', ls='--')
    plt.axline((0.01, 0.0001), (0.1, 0.01), color='b', label='2nd-order', ls='--')
    plt.xlabel('$h$')
    plt.ylabel('Internal '+r'$\varepsilon_{y}$')
    plt.xscale("log")
    plt.yscale("log")
    # plt.xlim([5E-3, 2E-1])
    # plt.ylim([1E-5, 1E-2])
    #plt.axis('equal')
    plt.legend()
    plt.title(r'$b )$')
    plt.tight_layout()
    # plot 3 - x-gradient error - boundary cells
    plt.subplot(2, 2, 3)
    plt.plot(h, bound_error_array[:, 0, 0], '-o', label='L1 norm')
    plt.plot(h, bound_error_array[:, 1, 0], '-ok', label='RMS')
    plt.plot(h, bound_error_array[:, 2, 0], '-or', label='Linf norm')
    plt.axline((0.01, 0.0001), (0.1, 0.001), color='g', label='1st-order', ls='--')
    plt.axline((0.01, 0.001), (0.1, 0.1), color='b', label='2nd-order', ls='--')
    plt.xlabel('$h$')
    plt.ylabel('Boundary '+r'$\varepsilon_{x}$')
    plt.xscale("log")
    plt.yscale("log")
    # plt.xlim([1E-3, 2E-1])
    # plt.ylim([5E-4, 5E-1])
    #plt.axis('equal')
    plt.legend()
    plt.title(r'$c )$')
    plt.tight_layout()
    # plot 4 - y-gradient error - boundary cells
    plt.subplot(2, 2, 4)
    plt.plot(h, bound_error_array[:, 0, 1], '-o', label='L1 norm')
    plt.plot(h, bound_error_array[:, 1, 1], '-ok', label='RMS')
    plt.plot(h, bound_error_array[:, 2, 1], '-or', label='Linf norm')
    plt.axline((0.01, 0.0001), (0.1, 0.001), color='g', label='1st-order', ls='--')
    plt.axline((0.01, 0.001), (0.1, 0.1), color='b', label='2nd-order', ls='--')
    plt.xlabel('$h$')
    plt.ylabel('Boundary '+r'$\varepsilon_{y}$')
    plt.xscale("log")
    plt.yscale("log")
    # plt.xlim([1E-3, 2E-1])
    # plt.ylim([5E-4, 5E-1])
    #plt.axis('equal')
    plt.title(r'$d )$')
    plt.legend()
    _figure_title = "2D_Structured_Cartesian_Grid_Error_Analysis"
    plt.tight_layout()
    print("Complete")
    gr.save_plot(_figure_title)
    gr.show_plot()
    #plt.show()
    return -1

def cartesian_plotter_v2(int_error_array, bound_error_array, h):
    fig, axs = plt.subplots(2, 2, )
    axs[0, 0].plot(h, int_error_array[:, 0, 0], '-o', label='L1 norm')
    axs[0, 0].plot(h, int_error_array[:, 1, 0], '-ok', label='RMS')
    axs[0, 0].plot(h, int_error_array[:, 2, 0], '-or', label='Linf norm')
    axs[0, 0].axline((0.0001, 0.0001), (10, 10), color='C0', label='1st-order', ls='--')
    axs[0, 0].axline((0.0001, 0.0001), (0.01, 0.1), color='C0', label='2nd-order', ls='--')
    axs[0, 0].xlabel('$h$')
    axs[0, 0].ylabel("x-gradient internal cell error")
    axs[0, 0].xscale("log")
    axs[0, 0].yscale("log")
    axs[0, 0].axis('equal')
    axs[0, 0].legend()
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
    plt.figure(1, figsize=(5, 5))
    plt.subplot(1, 2, 1)
    for i, i_metric in enumerate(grid_quality):
        plt.plot(h, int_error_array[i, :, 0], '-o', label=str(i_metric))
    plt.axline((0.01, 0.0001), (0.1, 0.001), color='g', label='1st-order', ls='--')
    plt.axline((0.01, 0.0001), (0.1, 0.01), color='b', label='2nd-order', ls='--')
    plt.xlabel('$h$')
    plt.ylabel('Internal '+r'$\varepsilon_{x}$')
    plt.xscale("log")
    plt.yscale("log")
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    # plot 2 - x-gradient error - boundrary cells
    plt.subplot(1, 2, 2)
    for i, i_metric in enumerate(grid_quality):
        plt.plot(h, bound_error_array[i, :, 0], '-o', label=str(i_metric))
    plt.axline((0.01, 0.0001), (0.1, 0.001), color='g', label='1st-order', ls='--')
    plt.axline((0.01, 0.0001), (0.1, 0.01), color='b', label='2nd-order', ls='--')
    plt.xlabel('$h$')
    plt.ylabel('Boundary '+r'$\varepsilon_{x}$')
    plt.xscale("log")
    plt.yscale("log")
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()

    # plt.subplot(1, 2, 2)
    plt.figure(2, figsize=(10, 8))
    plt.subplot(1, 2, 1)
    for i, i_metric in enumerate(grid_quality):
        plt.plot(h, int_error_array[i, :, 1], '-o', label=str(i_metric))
    plt.axline((0.01, 0.0001), (0.1, 0.001), color='g', label='1st-order', ls='--')
    plt.axline((0.01, 0.0001), (0.1, 0.01), color='b', label='2nd-order', ls='--')
    plt.xlabel('$h$')
    plt.ylabel('Internal '+r'$\varepsilon_{y}$')
    plt.xscale("log")
    plt.yscale("log")
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    # plot 2 - y-gradient error - internal cells
    plt.subplot(1, 2, 2)
    for i, i_metric in enumerate(grid_quality):
        plt.plot(h, bound_error_array[i, :, 1], '-o', label=str(i_metric))
    plt.axline((0.01, 0.0001), (0.1, 0.001), color='g', label='1st-order', ls='--')
    plt.axline((0.01, 0.0001), (0.1, 0.01), color='b', label='2nd-order', ls='--')
    plt.xlabel('$h$')
    plt.ylabel('Boundary '+r'$\varepsilon_{y}$')
    plt.xscale("log")
    plt.yscale("log")
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    return -1


def grid_metric_plotter(bound_error_array, int_error_array, grid_quality_array, i_matrix, gq = 0):
    name = grid_identity(gq)
    # - x-gradient error - internal cells
    plot1 = plt.figure(1, figsize=(6, 6))
    plt.subplot(1, 2, 1)
    plt.plot(grid_quality_array[0], int_error_array[:, 0, 0], 'o',label='L1 norm')
    plt.plot(grid_quality_array[0], int_error_array[:, 1, 0], 'ok', label='L2 norm')
    plt.plot(grid_quality_array[0], int_error_array[:, 2, 0], 'or', label='Linf norm')
    plt.xlabel(name+" Metric")
    plt.ylabel('Internal ' + r'$\varepsilon_{x}$')
    #plt.xscale("log")
    #plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    #plt.title(str(i_matrix)+" grid")

    plt.subplot(1, 2, 2)
    plt.plot(grid_quality_array[0], bound_error_array[:, 0, 0], 'o',label='L1 norm')
    plt.plot(grid_quality_array[0], bound_error_array[:, 1, 0], 'ok', label='RMS norm')
    plt.plot(grid_quality_array[0], bound_error_array[:, 2, 0], 'or', label='Linf norm')
    plt.xlabel(name+" Metric")
    plt.ylabel('Boundary ' + r'$\varepsilon_{x}$')
    #plt.xscale("log")
    #plt.yscale("log")
    plt.legend()
    #plt.title(str(i_matrix)+" grid")
    plt.tight_layout()

    plot2 = plt.figure(2, figsize=(6, 6))
    plt.subplot(1, 2, 1)
    plt.plot(grid_quality_array[0], int_error_array[:, 0, 1], 'o',label='L1 norm')
    plt.plot(grid_quality_array[0], int_error_array[:, 1, 1], 'ok', label='RMS norm')
    plt.plot(grid_quality_array[0], int_error_array[:, 2, 1], 'or', label='Linf norm')
    plt.xlabel(name+" Metric")
    plt.ylabel('Internal ' + r'$\varepsilon_{y}$')
    #plt.xscale("log")
    #plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.plot(grid_quality_array[0], bound_error_array[:, 0, 1], 'o',label='L1 norm')
    plt.plot(grid_quality_array[0], bound_error_array[:, 1, 1], 'ok', label='RMS norm')
    plt.plot(grid_quality_array[0], bound_error_array[:, 2, 1], 'or', label='Linf norm')
    plt.xlabel(name+" Metric")
    plt.ylabel('Boundary ' + r'$\varepsilon_{y}$')
    #plt.xscale("log")
    #plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    return -1