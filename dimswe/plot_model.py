import sys
from .plot_adv_dens import plot_adv_dens
from .plot_maxwell import plot_maxwell
from .parameters import get_parameters

if __name__ == "__main__":
    cfgfile = sys.argv[1]
    parameters = get_parameters(cfgfile)
    if parameters['model']['type'] in ['mtswe-cf-h1', 'tswe-cf', 'tswe-lp', 'tswe-cf-h1', 'ce-cf', 'ce-lp']:
        plot_adv_dens(parameters)
    elif parameters['model']['type'] == 'maxwell':
        plot_maxwell(parameters)
    

