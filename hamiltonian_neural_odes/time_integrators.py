import numpy as np

class TimeIntegrator():

    def set_initial_condition(self):
        self.initcond.set_initial_condition(self.xn[0,:])

    def compute_statistics(self, i):
        self.dynamics.compute_statistics(i, self.xn[i,:])

    def get_state(self):
        return [['x', self.xn],]

    def get_statistics(self):
        return self.dynamics.get_statistics()

    def get_convergence_data(self):
        return [[]]

    def get_convergence_data_names(self):
        return []

        #EVENTUALLY ADD A NEWTON SOLVER HERE AS WELL? OR A QUASI-NEWTON?
        #specific form is going to be heavily dependent on dynamics, likely what we do here is let dynamics return a Jacobian function

class NonlinearSolver():
    def __init__(self, eps=1e-15, max_iters=50):
        self.max_iters = max_iters
        self.eps = eps

    def create_vars(self, nsteps, xk):
        self.xkp1 = np.zeros(xk.shape)
        self.niterations = np.zeros(nsteps-1,dtype=np.int32)
        self.rel_tols = np.zeros(nsteps-1)

    def get_convergence_data(self):
        return [['niterations', self.niterations], ['rel_tols', self.rel_tols]]

    def get_convergence_data_names(self):
        return ['niterations', 'rel_tols']

class NewtonSolver(NonlinearSolver):
    def __init__(self, nonlinear_func, jacobian_func):
        NonlinearSolver.__init__(self)
        self.nonlinear_func = nonlinear_func
        self.jacobian_func = jacobian_func


#THIS IS BROKEN- NEED TO FIX IT EVENTUALLY
    def solve(self, i, xk, aux_args):

        niters = 0
        rel_tol = 100.0

        while (rel_tol > self.eps and niters<self.max_iters):
            self.nonlinear_func(self.xkp1, xk, *aux_args)

            #compute relative tolerance
            rel_tol = np.linalg.norm(self.xkp1 - xk)/max(np.linalg.norm(self.xkp1),1)
            #rel_tol = np.linalg.norm(self.xn[i,:] - self.xk[:])/max(np.linalg.norm(self.xn[i,:]),1)

            xk[:] = self.xkp1[:]
            niters = niters + 1

        self.rel_tols[i-1] = rel_tol
        self.niterations[i-1] = niters


class FixedPointSolver(NonlinearSolver):
    def __init__(self, nonlinear_func):
        NonlinearSolver.__init__(self)
        self.nonlinear_func = nonlinear_func

    def solve(self, i, xk, aux_args):

        niters = 0
        rel_tol = 100.0

        #FIXED POINT LOOP- MUST BE MORE CLEVER THAN THIS EVENTUALLY...
        #ADD ANDERSON ACCELERATION AS WELL?
        while (rel_tol > self.eps and niters<self.max_iters):
            self.nonlinear_func(self.xkp1, xk, *aux_args)

            #compute relative tolerance
            rel_tol = np.linalg.norm(self.xkp1 - xk)/max(np.linalg.norm(self.xkp1),1)
            #rel_tol = np.linalg.norm(self.xn[i,:] - self.xk[:])/max(np.linalg.norm(self.xn[i,:]),1)

            xk[:] = self.xkp1[:]
            niters = niters + 1

        self.rel_tols[i-1] = rel_tol
        self.niterations[i-1] = niters

class AVF2(TimeIntegrator):
    def __init__(self, dynamics, initcond, quadpts, solver='fixedpoint'):
        self.dynamics = dynamics
        self.initcond = initcond
        self.implicit = True

        self.nquadpts = quadpts
        self.points, self.weights = np.polynomial.legendre.leggauss(quadpts)
        #renormalize to [0,1] interval
        self.points = (self.points + 1.) / 2.0
        self.weights = self.weights / 2.

        self.nonlinear_func = lambda result, xk, xn, dt: self._nonlinear_func(result, xk, xn, dt)
#ADD THIS EVENTUALLY
        self.jacobian_func = None
        if solver == 'fixedpoint':
            self.solver = FixedPointSolver(self.nonlinear_func)
        elif solver =='newton':
            self.solver = NewtonSolver(self.nonlinear_func, self.jacobian_func)
        else:
            raise ValueError('unknown solver type ' + solver)

    def get_convergence_data(self):
        return self.solver.get_convergence_data()

    def get_convergence_data_names(self):
        return self.solver.get_convergence_data_names()

    def create_vars(self, nsteps):
        self.xn = self.dynamics.create_x(size=nsteps)
        self.dhdx = self.dynamics.create_dhdx()
        self.rhs = self.dynamics.create_x()
        self.dhdx_temp = self.dynamics.create_dhdx()
        self.xq = self.dynamics.create_x()
        self.xstar = self.dynamics.create_x()
        self.xk = self.dynamics.create_x()

        self.solver.create_vars(nsteps, self.xk)

    def _nonlinear_func(self, result, xk, xn, dt):
        self.dhdx[:] = 0.0
        for q in range(self.nquadpts):
            self.xq[:] = (1.0 -self.points[q]) * xn + self.points[q] * xk
            self.dynamics.compute_dhdx(self.dhdx_temp, self.xq)
            self.dhdx[:] = self.dhdx[:] + self.weights[q] * self.dhdx_temp[:]
        self.xstar[:] = (xn + xk)/2.
        self.dynamics.compute_rhs(self.rhs, self.xstar, self.dhdx)
        result[:] = xn[:] + dt * self.rhs

    def take_step(self, i, dt):
        self.xk[:] = self.xn[i-1,:]
        self.solver.solve(i, self.xk, aux_args=[self.xn[i-1,:], dt])
        self.xn[i,:] = self.xk[:]




class RK4(TimeIntegrator):
    def __init__(self, dynamics, initcond):
        self.dynamics = dynamics
        self.initcond = initcond
        self.implicit = False


    def create_vars(self, nsteps):
        self.F1 = self.dynamics.create_x()
        self.F2 = self.dynamics.create_x()
        self.F3 = self.dynamics.create_x()
        self.F4 = self.dynamics.create_x()
        self.xn = self.dynamics.create_x(size=nsteps)
        self.x1 = self.dynamics.create_x()
        self.x2 = self.dynamics.create_x()
        self.x3 = self.dynamics.create_x()
        self.dhdx = self.dynamics.create_dhdx()

    def take_step(self, i, dt):

        self.dynamics.compute_dhdx(self.dhdx, self.xn[i-1,:])
        self.dynamics.compute_rhs(self.F1, self.xn[i-1,:], self.dhdx)
        self.x1 = self.xn[i,:] + dt/2.0*self.F1

        self.dynamics.compute_dhdx(self.dhdx, self.x1)
        self.dynamics.compute_rhs(self.F2, self.x1, self.dhdx)
        self.x2 = self.xn[i-1,:] + dt/2.0*self.F2

        self.dynamics.compute_dhdx(self.dhdx, self.x2)
        self.dynamics.compute_rhs(self.F3, self.x2, self.dhdx)
        self.x3 = self.xn[i-1,:] + dt*self.F3

        self.dynamics.compute_dhdx(self.dhdx, self.x3)
        self.dynamics.compute_rhs(self.F4, self.x3, self.dhdx)

        self.xn[i,:] = self.xn[i-1,:] + dt/6.0*(self.F1 + 2.*self.F2 + 2.*self.F3 + self.F4)

class Euler(TimeIntegrator):
    def __init__(self, dynamics, initcond):
        self.dynamics = dynamics
        self.initcond = initcond
        self.implicit = False


    def create_vars(self, nsteps):
        self.F1 = self.dynamics.create_x()
        self.xn = self.dynamics.create_x(size=nsteps)
        self.dhdx = self.dynamics.create_dhdx()

    def take_step(self, i, dt):

        self.dynamics.compute_dhdx(self.dhdx, self.xn[i-1,:])
        self.dynamics.compute_rhs(self.F1, self.xn[i-1,:], self.dhdx)
        self.xn[i,:] = self.xn[i-1,:] + dt*self.F1


class GaussSymplectic(TimeIntegrator):
    def __init__(self, dynamics, initcond):
        self.dynamics = dynamics
        self.initcond = initcond
        self.implicit = True

    def take_step(self, dt):
        pass

#ALSO ADD VARIOUS SYMPLECTIC INTEGRATORS, AND IDEALLY LIE-POISSON INTEGRATORS AS WELL
#CAREFUL WITH TIME INTEGRATION ON LIE GROUPS/ALGEBRAS...

def get_timestepper(parameters, model, initcond):
    if parameters['timestepper'] == 'AVF2':
        return AVF2(model, initcond, parameters['avf_quad_pts'])
    elif parameters['timestepper'] == 'RK4':
        return RK4(model, initcond)
    elif parameters['timestepper'] == 'Euler':
        return RK4(model, initcond)
    elif parameters['timestepper'] == 'gauss-symplectic':
        return GaussSymplectic(model, initcond)
    else:
        raise ValueError('unknown time stepper ' + parameters['timestepper'])

#ADD THIS ONE- IT IS NAT'S PROPOSED APPROACH!
#Useful to test with
class HybridizedMixed(TimeIntegrator):
    pass
