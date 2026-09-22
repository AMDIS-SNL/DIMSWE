import sys
from .plot_adv_dens import plot_adv_dens
from .plot_maxwell import plot_maxwell
from .parameters import get_parameters

if __name__ == "__main__":
    cfgfile = sys.argv[1]
    parameters = get_parameters(cfgfile)
#ADD METRIPLECTIC STUFF HERE
    if parameters['model']['type'] in ['advdens-cf-h1']:
        plot_adv_dens(parameters)
    elif parameters['model']['type'] in ['metriplectic']:
        if parameters['model']['hamiltonian'] in ['tswe', 'mtswe', 'ce']:
            plot_adv_dens(parameters)
#WRONG- ADD TO IT
    #elif parameters['model']['type'] == 'maxwell':
    #    plot_maxwell(parameters)
