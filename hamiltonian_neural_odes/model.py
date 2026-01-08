from dynamics import get_dynamics
from time_integrators import get_timestepper
from initialconditions import get_initcond
from output import Output

def run_model(parameters):
    #load settings
    nsteps = parameters['nsteps']
    dt = parameters['dt']

    #create model
    initcond = get_initcond(parameters)
    model = get_dynamics(parameters, initcond)
    timestepper = get_timestepper(parameters, model, initcond)
    output = Output(parameters, timestepper)

    #do timesteps + get relevant output

    #THESE CAN PROBABLY BE MERGED INTO INIT ROUTINES FOR MODEL AND TIMESTEPPER, I THINK
    #ALSO SEPARATING OUT CREATE VARS IS MAYBE USEFUL FOR PLOTTING?
    model.create_statistics(nsteps+1)
    timestepper.create_vars(nsteps+1)

    timestepper.set_initial_condition()
    timestepper.compute_statistics(0)
    for n in range(1,nsteps+1):
        timestepper.take_step(n, dt)
        timestepper.compute_statistics(n)
    output.output()

if __name__ == "__main__":
    from parameters import parameters
    run_model(parameters)
