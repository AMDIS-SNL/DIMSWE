import sys
from dimswe.plotting.plot_adv_dens import plot_adv_dens
from dimswe.plotting.plot_maxwell import plot_maxwell
from dimswe.plotting.plot_elliptic import plot_elliptic, plot_elliptic_eigen, plot_convergence
from dimswe.parameters import get_parameters

if __name__ == "__main__":
    cfgfile = sys.argv[1]
    if cfgfile.endswith('summary.yaml'):
        # Convergence summary written by workflows/elliptic_convergence.py.
        print('wrote ' + str(plot_convergence(cfgfile, cfgfile[:-len('summary.yaml')] + 'convergence.png')))
        sys.exit(0)
    parameters = get_parameters(cfgfile)
    if parameters['model']['type'] == 'elliptic':
        plot_elliptic(parameters)
    elif parameters['model']['type'] == 'elliptic-eigen':
        plot_elliptic_eigen(parameters)
#ADD METRIPLECTIC STUFF HERE
    elif parameters['model']['type'] in ['advdens-cf-h1']:
        plot_adv_dens(parameters)
    elif parameters['model']['type'] in ['metriplectic']:
        if parameters['model']['hamiltonian'] in ['tswe', 'mtswe', 'ce']:
            plot_adv_dens(parameters)
#WRONG- ADD TO IT
    #elif parameters['model']['type'] == 'maxwell':
    #    plot_maxwell(parameters)
