class L2Objective():

    def __init__(self, truth):
        self.truth = truth

    def grad_state(self, x, params):
        return x

    def grad_params(self, x, params):
        return 0

    def gradT_state(self, x, params):
        return x.T

    def gradT_params(self, x, params):
        return 0.T

    def evaluate(self, x, params, range):
        residual = x[range] - self.truth[range]
        return 0.5 * residual**2

class PDEConstrainedOptimizerLagrangianAdjoint():
    def __init__(self, constraints, objective):
        pass

    def jac(self, params):
#do a forward sweep

#do a backward sweep

#ADD STUFF FOR INITIAL CONDITION OPTIMIZATION ALSO!
#basically this is just delta!
        pass

    def hessp(self, params, delta_params):
        pass

    def objective(self, params):
        pass
#do a forward sweep

    def optimize(self, params0):
        #LOTS OF OTHER ARGUMENTS!
        #self.dynamics.parameter_bounds()
        self.optimizer.optimize(self.objective, params0, jac=self.jac, hessp=self.hessp)


#generate "truth"
#optimize for some set of other parameters
#re-run with new parameters
