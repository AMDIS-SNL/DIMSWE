import matplotlib.pyplot as plt
from firedrake.pyplot import tricontourf, tricontour, tripcolor, quiver, triplot, plot
import matplotlib.animation as animation

def plot_mesh(mesh):
    fig, axes = plt.subplots()
    triplot(mesh, axes=axes)
    axes.legend()
    fig.savefig('mesh.png')

def plot_scalar2D(func, name):
    fig, axes = plt.subplots()
    #contours = tricontour(func, axes=axes)
    #contours = tricontourf(func, axes=axes, cmap="inferno")
    contours = tripcolor(func, axes=axes, cmap="inferno")
    axes.set_aspect("equal")
    fig.colorbar(contours)
    fig.savefig(name + '.png')
    plt.close()
    
def animate_scalar2D(chkpoint_file, mesh, varname, noutput, name):
    fig, axes = plt.subplots()
    axes.set_aspect("equal")
    artists = []
    for i in range(noutput):
        func = chkpoint_file.load_function(mesh, varname, idx=i)
        artist = tripcolor(func, axes=axes, cmap="inferno")
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

def plot_variable(data, name, dim, is_vector):
    if dim == 1:
        plot_scalar1D(data, name)
    if dim == 2:
        if is_vector:
            plot_vector2D_quiver(data, name)
        else:
            plot_scalar2D(data, name)

def plot_statistic(data, name):
    plt.figure()
    plt.plot(data)
    plt.savefig(name + '.png')
    plt.close()

    plt.figure()
    plt.plot((data - data[0])/data[0]*100)
    plt.savefig(name + '-fractional-change.png')
    plt.close()
