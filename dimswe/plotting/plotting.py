import matplotlib.pyplot as plt
from firedrake.pyplot import tricontourf, tricontour, tripcolor, quiver, triplot, plot
import matplotlib.animation as animation
import numpy as np

# colormap for fields drawn centred on zero (negative values of positivity diagnostics stand out)
DIVERGING_CMAP = "RdBu_r"


def centred_color_kwargs(values):
    """Return tripcolor colour arguments: a zero-centred diverging scale if values has negative entries.

    Otherwise the default sequential colormap, autoscaled, is used (an empty dict of limits).
    """
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if vmin >= 0:
        return {"cmap": "inferno"}
    limit = max(abs(vmin), abs(vmax))
    return {"cmap": DIVERGING_CMAP, "vmin": -limit, "vmax": limit}

def plot_mesh(mesh):
    fig, axes = plt.subplots()
    triplot(mesh, axes=axes)
    axes.legend()
    fig.savefig('mesh.png')

def plot_scalar2D(func, name, center_negative=False):
    """Plot a scalar field; with center_negative, fields with negative values use a zero-centred scale."""
    fig, axes = plt.subplots()
    #contours = tricontour(func, axes=axes)
    #contours = tricontourf(func, axes=axes, cmap="inferno")
    color_kwargs = centred_color_kwargs(func.dat.data_ro) if center_negative else {"cmap": "inferno"}
    contours = tripcolor(func, axes=axes, **color_kwargs)
    axes.set_aspect("equal")
    fig.colorbar(contours)
    fig.savefig(name + '.png')
    plt.close()

def animate_scalar2D(chkpoint_file, mesh, varname, noutput, name, is_vector, center_negative=False):
    """Animate a field over all outputs.

    With center_negative, a scalar field that is negative anywhere in the run is drawn on one
    zero-centred diverging scale, fixed over all frames (used for the positivity diagnostics).
    """
    fig, axes = plt.subplots()
    axes.set_aspect("equal")
    color_kwargs = {"cmap": "inferno"}
    if center_negative and not is_vector:
        values = np.concatenate([chkpoint_file.load_function(mesh, varname, idx=i).dat.data_ro for i in range(noutput)])
        color_kwargs = centred_color_kwargs(values)
    artists = []
    for i in range(noutput):
        func = chkpoint_file.load_function(mesh, varname, idx=i)
        if is_vector:
            artist = quiver(func, axes=axes)
        else:
            artist = tripcolor(func, axes=axes, **color_kwargs)
        txt = axes.text(0.02, 0.98, 't='+str(i), transform=axes.transAxes,
                      va="top", ha="left", color="w",
                      bbox=dict(facecolor="black", alpha=0.3, edgecolor="none", pad=2))
        artists.append([artist,txt])
    cbar = fig.colorbar(artists[0][0])
    ani = animation.ArtistAnimation(fig, artists, interval=50, blit=True, repeat=False) #repeat_delay=1000
    ani.save(filename=name +'.webp', writer="pillow")

def plot_vector2D_quiver(func, name):
    fig, axes = plt.subplots()
    contours = quiver(func, axes=axes)
    axes.set_aspect("equal")
    fig.colorbar(contours)
    fig.savefig(name + '.png')
    plt.close()

def plot_vector2D_mag(func, name):
    fig, axes = plt.subplots()
    contours = tripcolor(func, axes=axes)
    # contours = tricontourf(func, axes=axes, cmap="inferno")
    axes.set_aspect("equal")
    fig.colorbar(contours)
    fig.savefig(name + '-mag.png')
    plt.close()

def plot_scalar1D(func, name):

    fig, axes = plt.subplots()
    plot(func, axes=axes)
    fig.savefig(name + '.png')
    plt.close()

def plot_variable(data, name, dim, is_vector, center_negative=False):
    if dim == 1:
        plot_scalar1D(data, name)
    if dim == 2:
        if is_vector:
            plot_vector2D_quiver(data, name)
        else:
            plot_scalar2D(data, name, center_negative=center_negative)

def plot_statistic(data, name):
    """Plot a statistic time series and its change: in percent of the initial value, or absolute if that is 0."""
    plt.figure()
    plt.plot(data)
    plt.savefig(name + '.png')
    plt.close()

    plt.figure()
    if data[0] != 0:
        plt.plot((data - data[0])/data[0]*100)
        plt.savefig(name + '-fractional-change.png')
    else:
        # the fractional change is undefined (for example the minimum of a field starting at 0)
        plt.plot(data - data[0])
        plt.savefig(name + '-change.png')
    plt.close()

def plot_positivity_statistics(stats, density):
    """Plot the domain min/max of the cell mean and of the whole-cell minimum of a density over time.

    stats maps the statistic names (for example Qc_cellmin_min) to their time series; missing ones
    (no whole-cell minimum on non-box meshes) are skipped. A zero line marks the positivity bound.
    """
    fig, axes = plt.subplots()
    styles = {('cellmean', 'min'): ('C0', '-'), ('cellmean', 'max'): ('C0', '--'),
              ('cellmin', 'min'): ('C3', '-'), ('cellmin', 'max'): ('C3', '--')}
    for (measure, extremum), (color, linestyle) in styles.items():
        key = f'{density}_{measure}_{extremum}'
        if key in stats:
            axes.plot(stats[key], color=color, linestyle=linestyle, label=f'{extremum} of cell {measure[4:]}')
    axes.axhline(0.0, color='k', linewidth=0.8)
    axes.set_xlabel('statistics step')
    axes.set_title(f'{density}: cell means and whole-cell minima')
    axes.legend()
    fig.savefig(density + '-positivity.png')
    plt.close(fig)
