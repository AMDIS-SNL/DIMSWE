from firedrake import NonlinearVariationalProblem, NonlinearVariationalSolver, LinearVariationalProblem, LinearVariationalSolver
from .parameters import overall_solver_parameters
import numpy as np
from firedrake import Constant, inner, TestFunction, derivative, norm, assemble, Function, TestFunctions, split, TrialFunction, adjoint, action

class DummySolver():
    def solve(self):
        pass

def make_a_L(Lexpr, vartrial, varhat, dx):
    a = inner(varhat, vartrial)*dx
    L = inner(varhat, Lexpr)*dx
    return [a, L]


def get_time_integrator(name, parameters):
    if name == 'RK4':
        return RK4
    elif name == 'Euler':
        return Euler
    elif name == 'SSPRK3':
        return SSPRK3
    elif name == 'SSPRK43':
        return SSPRK43
    elif name == 'LieSplit':
        timestepper_list = parameters['timestepping']['timestepper_list']
        termlist = parameters['timestepping']['termlist']
        subcycle_list = parameters['timestepping']['subcycle_list']

        return lambda model, logger, coeffs: LieSplittingIntegrator(model, logger, coeffs, timestepper_list, termlist, subcycle_list)
    else:
        raise ValueError("time step method " + name + " is unknown")

def get_timestepper(parameters, model, logger, coeffs):
    return get_time_integrator(parameters['timestepping']['method'], parameters)(model, logger, coeffs)


class TimeStepper():


    def __init__(self, model, logger, coeffs, terms='all'):
        self.model = model
        self.logger = logger
        self.terms = terms
        self.dx = model.spaces.dx

        self.xk, self.xk_sub, self.xk_split = self.model.get_x_var('xk')

        self.lambda_var, self.lambda_sub, self.lambda_split = self.model.get_x_var('lambda')

        self.coeff, self.coeff_sub, self.coeff_split, self.coeff_trial = coeffs
        #self.grad, self.grad_sub, self.grad_split, self.grad_trial = self.model.get_coeff_var()

        self.t = Constant(1.)
        self.dt = Constant(1.)

    def set_coeff(self, coeff_val):
        self.coeff.assign(coeff_val)

class ExplicitRK(TimeStepper):
    def __init__(self, model, logger, coeffs, A, b, c, nstages, terms='all'):
        TimeStepper.__init__(self, model, logger, coeffs, terms=terms)
        self.A = A
        self.b = b
        self.c = c
        self.nstages = nstages

        self.aux_var_list = self.model.get_aux_var_list(terms=terms)

        self.Fi = []
        self.auxi = []
#NOT SURE ABOUT THESE SPLITS YET...
        self.mui = []
        self.muauxi = []
        self.li = []
        self.lauxi = []
        for i in range(nstages):
            self.auxi.append(self.model.get_aux_var('aux'+str(i)))
            self.Fi.append(self.model.get_x_var('F'+str(i)))
#NOT SURE ABOUT THESE SPLITS YET...
            self.mui.append(self.model.get_x_var('mu'+str(i)))
            self.muauxi.append(self.model.get_aux_var('muaix'+str(i)))
            self.li.append(self.model.get_x_var('li'+str(i)))
            self.lauxi.append(self.model.get_aux_var('lauxi'+str(i)))

        xhat, xhat_subs = self.model.get_x_test_vars()
        xtrial, xtrial_subs = self.model.get_x_trial_vars()

        auxhat, auxhat_subs = self.model.get_aux_test_vars()
        auxtrial, auxtrial_subs = self.model.get_aux_trial_vars()

        full_hat_subs = xhat_subs | auxhat_subs

#CAREFUL WITH SUB VS SPLIT HERE!
        self.Fsolvers = []
        self.auxsolvers = []
        for i in range(nstages):
            xi_sub = {}
            xi_split = {}
            for var in self.model.get_x_var_list():
                xi_sub[var] = self.xk_sub[var] + self.dt*sum(float(self.A[i,j]) * self.Fi[j][1][var] for j in range(self.nstages))
                xi_split[var] = self.xk_split[var] + self.dt*sum(float(self.A[i,j]) * self.Fi[j][2][var] for j in range(self.nstages))

            if self.model.has_aux():
                a_aux = 0
                L_aux = 0
                aux_expressions = self.model.compute_aux_expressions(xi_sub, self.auxi[i][2], self.t, self.coeff_sub, full_hat_subs, terms=terms)
                for var in self.model.get_aux_var_list(terms=terms):
                    a_aux = a_aux + derivative(aux_expressions[var][0], self.auxi[i][0], auxtrial)
                    L_aux = L_aux + aux_expressions[var][1]
                #print(i,residual_aux, residual_aux == 0)
                if L_aux == 0:
                    self.auxsolvers.append(DummySolver())
                    #a_aux = inner(auxtrial, auxhat) * self.dx
                    #L_aux = 0 #inner(self.auxi[i][0], auxhat) * self.dx
                else:
                    #a_aux = derivative(residual_aux, self.auxi[i][0], auxtrial)
                    #L_aux = action(a_aux, self.auxi[i][0]) - residual_aux
                    auxproblem = LinearVariationalProblem(a_aux, L_aux, self.auxi[i][0])
                    self.auxsolvers.append(LinearVariationalSolver(auxproblem, solver_parameters=overall_solver_parameters['erkstage-aux'], options_prefix = 'erk-aux'))
            else:
                self.auxsolvers.append(DummySolver())

            #this sign is due to writing things as dxdt + F(x) = 0
            #residual_F = inner(xhat, self.Fi[i][0])*self.dx +
            a_F = inner(xhat, xtrial)*self.dx
            L_F  = -model.rhs(xi_split, self.auxi[i][1], self.t, self.coeff_sub, full_hat_subs, terms=terms) #action(a_F, self.Fi[i][0]) - residual_F
            Fproblem = LinearVariationalProblem(a_F, L_F, self.Fi[i][0])
            self.Fsolvers.append(LinearVariationalSolver(Fproblem, solver_parameters=overall_solver_parameters['erkstage-f'], options_prefix = 'erk-f'))



#THIS PROBABLY REQUIRES WORKING ON THE FULL SPACE :(
#CAN WE BUILD VARIOUS STATE GRADIENTS SEPARATELY?
        #    residual = residual_aux + residual_F

        #self.gradT_state = adjoint(derivative(rhs, self.xk, xtrial))
        #if not (self.coeff is None):
        #    self.gradT_params = adjoint(derivative(rhs, self.coeff, self.coeff_trial))

        #self.muisolvers = []
        #gradrhs = 0
        #    self.Fsolvers.append(LinearVariationalSolver(Fproblem, solver_parameters=overall_solver_parameters['erkstage'], options_prefix = 'erk-f'))
        #    muirhs = action(self.gradT_state, self.li[i])
        #    muiproblem = LinearVariationalProblem(A, muirhs, self.mui[i])
        #    self.muisolvers.append(LinearVariationalSolver(muiproblem, solver_parameters=overall_solver_parameters['muistage'], options_prefix = 'erk-mui'))
#THIS IS MAYBE WRONG?
#WHAT IS THE CORRECT FORMAT ACTUALLY?
#IE WHAT IS THE GRADIENT OF THE FUNCTIONAL WRT
#Scipy optimizers want a (1,nparams) matrix (or similar)
#I bet there is no solve needed, it is just assembly of a matrix!
            #gradrhs = gradrhs - action(self.gradT_params, self.li[i])
        #Agrad = derivative(inner(self.grad, self.gradtrial)*self.dx, self.grad)
        #gradproblem = LinearVariationalProblem(Agrad, gradrhs, self.grad)
        #self.gradsolver = LinearVariationalSolver(gradproblem, solver_parameters=overall_solver_parameters['grad'], options_prefix='grad')




    def take_forward_step(self, xnp1, xnp1_sub, xn, tn, dt):

        self.t.assign(tn)
        self.dt.assign(dt)
        self.xk.assign(xn)
        for i in range(self.nstages):
            self.t.assign(self.t + self.c[i]*self.dt)
            #THIS PEICE IS A LITTLE BROKEN- WE DONT WANT TO DO THIS ANYWAYS!
            #self.model.post_step(self.xk_sub, terms=self.terms)
            self.auxsolvers[i].solve()
            self.Fsolvers[i].solve()

        xnp1.assign(xn + self.dt * sum(float(self.b[i]) * self.Fi[i][0] for i in range(self.nstages)))
        #self.model.post_step(xnp1_sub, terms=self.terms)


    def take_adjoint_step(self, grad, lambda_n, lambda_np1, tnp1, dt):

        self.dt.assign(dt)
        self.lambda_var.assign(lambda_np1)

#ALL BROKEN FOR NOW...
        #compute mui
        for i in range(self.nstages-1,-1,-1):
            self.xk.assign(self.Xi[i])
            self.aux_vars.assign(self.aux_i[i])
            self.t.assign(tnp1 - dt + self.c[i]*self.dt)
            self.muauxisolvers.solve()
            self.li[i].assign(self.dt * float(self.b[i]) * self.lambda_var + sum(float(self.A[i,j]) * self.mui[j+1] for j in range(i,self.nstages-1)))
            self.muisolvers.solve()

        #compute grad
#HOW DO WE ACTUALLY DO THIS?
        #self.gradsolver.solve()
        #grad.assign(grad + self.grad)

        #compute lambda_n
        lambda_n.assign(lambda_np1 + sum(self.mu[i] for i in range(self.nstages)) + sum(self.muaux[i] for i in range(self.nstages)))

class Euler(ExplicitRK):
    def __init__(self, model, logger, coeffs, terms='all'):
        A = np.array([[0.0,],])
        b = np.array([1.0,])
        c = np.array([0.0,])
        ExplicitRK.__init__(self, model, logger, coeffs, A, b, c, 1, terms=terms)

class RK4(ExplicitRK):
    def __init__(self, model, logger, coeffs, terms='all'):
        A = np.array([[0.0, 0.0, 0.0, 0.0,], [0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        b = np.array([1./6., 1./3., 1./3., 1./6.])
        c = np.array([0.0, 0.5, 0.5, 1.0])
        ExplicitRK.__init__(self, model, logger, coeffs, A, b, c, 4, terms=terms)

class SSPRK3(ExplicitRK):
    def __init__(self, model, logger, coeffs, terms='all'):
        A = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.25, 0.25, 0.0]])
        b = np.array([1./6., 1./6., 2./3.])
        c = np.array([0.0, 1.0, 0.5])
        ExplicitRK.__init__(self, model, logger, coeffs, A, b, c, 3, terms=terms)


class SSPRK43(ExplicitRK):
    def __init__(self, model, logger, coeffs, terms='all'):
        A = np.array([[0.0, 0.0, 0.0, 0.0,], [0.5, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [1.0/6.0, 1.0/6.0, 1.0/6.0, 0.0]])
        b = np.array([1./6., 1./6., 1./6., 3./6.])
        c = np.array([0.0, 0.5, 1.0, 0.5])
        ExplicitRK.__init__(self, model, logger, coeffs, A, b, c, 4, terms=terms)

#ONLY SIGNIFICANT DIFFERENCE HERE IS THAT YI PROBLEMS BECOME NONLINEAR BUT STILL SEPARATED, AND MUI BECOMES A TRUE LINEAR SYSTEM BUT SEPARATED!
class DIRK(TimeStepper):
    def __init__(self,):
        pass
    def take_forward_step(dt):
        pass
    def take_adjoint_step(dt):
        pass

#ONLY SIGNIFICANT DIFFERENCE HERE IS THAT YI PROBLEMS BECOME NONLINEAR AND FULLY COUPLED, AND MUI BECOMES A TRUE LINEAR SYSTEM AND COUPLED!
class ImplicitRK(TimeStepper):
    def __init__(self,):
        pass
    def take_forward_step(dt):
        pass
    def take_adjoint_step(dt):
        pass



class KGRK2(ExplicitRK):
    def __init__(self, model, logger, coeffs, terms='all'):
        A = SOMETHING
        b = SOMETHING
        c = SOMETHING
        nstages = SOMETHING
        ExplicitRK.__init__(self, model, logger, coeffs, A, b, c, nstages, terms=terms)


class KGRK3(ExplicitRK):
    def __init__(self, model, logger, coeffs, terms='all'):
        A = SOMETHING
        b = SOMETHING
        c = SOMETHING
        nstages = SOMETHING
        ExplicitRK.__init__(self, model, logger, coeffs, A, b, c, nstages, terms=terms)

class SOMEDIRK(DIRK):
    def __init__(self, model, logger, coeffs, terms='all'):
        A = SOMETHING
        b = SOMETHING
        c = SOMETHING
        nstages = SOMETHING
        DIRK.__init__(self, model, logger, coeffs, A, b, c, nstages, terms=terms)
#ADD SOME DIRK CLASSES HERE ALSO


class SOMEIMPLICITRK(ImplicitRK):
    def __init__(self, model, logger, coeffs, terms='all'):
        A = SOMETHING
        b = SOMETHING
        c = SOMETHING
        nstages = SOMETHING
        DIRK.__init__(self, model, logger, coeffs,  A, b, c, nstages, terms=terms)
#ADD SOME DIRK CLASSES HERE ALSO

class LieSplittingIntegrator():
    def __init__(self, model, logger, coeffs, timestepper_list, termlist, subcycle_list):

        self.subcycle_list = subcycle_list
        self.timestepper_list = timestepper_list
        self.termlist = termlist

        self.time_integrators = []
        for i,time_integrator_name in enumerate(timestepper_list):
            time_integrator = get_time_integrator(time_integrator_name, None)
            self.time_integrators.append(time_integrator(model, logger, coeffs, terms=termlist[i]))

        self.xk, self.xk_sub, self.xk_split = model.get_x_var('xk')
        self.lambda_k, self.lambda_sub, self.lambda_split = model.get_x_var('lambda_k')


    def take_forward_step(self, xnp1, xnp1_sub, xn, tn, dt):
        self.xk.assign(xn)
        for i,time_integrator in enumerate(self.time_integrators):
            sub_dt = dt / self.subcycle_list[i]
#PROBABLY NEED TO STORE INTERMEDIATE VALUES HERE
            for k in range(self.subcycle_list[i]):
                time_integrator.take_forward_step(self.xk, self.xk_sub, self.xk, tn + k * sub_dt, sub_dt)
        xnp1.assign(self.xk)


#THERE IS A STORAGE CHOICE THAT NEEDS TO BE MADE HERE
#BASICALLY, EACH TIMESTEPPER OBJECT ONLY KNOWS HOW TO STORA VALUES FOR A SINGLE ITERATION
#SO THE SUBCYLCING ASPECT IS BROKEN!
    def take_adjoint_step(grad, lambda_n, lambda_np1, tnp1, dt):
        self.lambda_k.assign(lambda_np1)
        for i,time_integrator in enumerate(reversed(self.time_integrators)):
            sub_dt = dt / self.subcycle_list[i]
            #LIKELY NEED TO DO SOME RECOMPUTATION OF INTERMEDIATE VALUES HERE?
#REVERSE THIS ALSO? MAYBE TNP1 IS ENOUGH TO REVERSE IT? #UNCLEAR...
            for k in range(self.subcycle_list[i]):
                time_integrator.take_adjoint_step(grad, self.lambda_k, tnp1 - k * sub_dt, sub_dt)
        lambda_n.assign(self.lambda_k)
#HOW DO WE ADD SUPPORT FOR INITIAL CONDITION SENSITIVITY?













#
# class FixedPointSolver():
#     def __init__(self, fexpr, xnp1, varspace, pre_function_callback, post_function_callback, dx):
#         self.fexpr = fexpr
#         self.xnp1 = xnp1 #this is xk
#         #self.xkp1 = xnp1.copy(deepcopy=True)
#         self.xkp1 = Function(varspace)
#         self.pre_function_callback = pre_function_callback
#         self.post_function_callback = post_function_callback
#         self.dx = dx
#
# #MOVE TO PARAMETERS AT SOME POINT
#         self.eps = 1e-12
#         self.max_iters = 50
#
#         xtest = TestFunction(varspace)
#         a = derivative(inner(xtest,self.xkp1)*self.dx, self.xkp1)
#         linearproblem = LinearVariationalProblem(a, fexpr, self.xkp1)
#         self.linearsolver = LinearVariationalSolver(linearproblem, solver_parameters=overall_solver_parameters['fixedpoint'], options_prefix = 'fixedpoint')
#
#     def solve(self):
#
#         niters = 0
#         rel_tol = 100.0
#
# #EVENTUALLY ADD ANDERSON ACCELERATION?
#         while (rel_tol > self.eps and niters<self.max_iters):
#             with self.xnp1.dat.vec_ro as vardat:
#                 self.pre_function_callback(vardat)
#
#             self.linearsolver.solve()
#             rel_tol = norm(self.xkp1 - self.xnp1) / max(norm(self.xkp1), 1.0)
#
# #SUPER UNCLEAR IF A LIMITER HERE WILL ACTUALLY WORK?
#             with self.xkp1.dat.vec_wo as xkp1:
#                     self.post_function_callback(xkp1)
#
#             with self.xkp1.dat.vec_ro as xkp1:
#                 with self.xnp1.dat.vec_wo as xnp1:
#                     xkp1.copy(xnp1)
#             niters = niters + 1
# #probably store these somewhere eventually
# #ALSO DO THIS FOR NEWTON SOLVER AS WELL!!!
#         #print(rel_tol, niters)
#
#
#
#
#
# class AVF2_Integrator(TimeStepper):
#     def __init__(self, parameters, model, initcond, logger, xn=None, terms='all'):
#         self.model = model
#         self.initcond = initcond
#         self.logger = logger
#         self.parameters = parameters
#         self.dx = model.spaces.dx
#
#
#         self.q_aux_vars = self.model.get_q_aux_vars(terms=terms)
#         self.dfdx_aux_vars = self.model.get_dfdx_aux_vars(terms=terms)
#
#         if xn is None:
#             self.xn = self.model.get_x_var('xn')
#         else:
#             self.xn = xn
#         self.xk = self.model.get_x_var('xk')
#         self.xnp1 = self.model.get_x_var('xnp1')
#
#         self.points, self.weights = np.polynomial.legendre.leggauss(parameters['timestepping']['num_avf_quad'])
#         self.npoints = parameters['timestepping']['num_avf_quad']
#         #renormalize to [0,1] interval
#         self.points = (self.points + 1.) / 2.0
#         self.weights = self.weights / 2.
#
# #CAN TRIM OR REMOVE A LOT OF THIS?
#         self.xn_sub = {}
#         self.xk_sub = {}
#         self.xnp1_sub = {}
#         self.xnp1_split = {}
#         xnp1_split_temp = split(self.xnp1)
#         for i,var in enumerate(self.model.variableset.varlist):
#             self.xn_sub[var] = self.xn.sub(i)
#             self.xnp1_sub[var] = self.xnp1.sub(i)
#             self.xnp1_split[var]  = xnp1_split_temp[i]
#             self.xk_sub[var] = self.xk.sub(i)
#
#
#         for i,var in enumerate(self.model.variableset.varlist):
#             self.q_aux_vars[var] = 0.5 * self.xnp1_split[var] + 0.5 * self.xn.sub(i)
#             #REPLACE WITH self.xn_sub here!
#
#         q_expressions = self.model.compute_q_expressions(self.q_aux_vars, terms=terms)
#         self.q_aux_solvers = []
#         for var in self.model.get_q_aux_var_list(terms=terms):
#             a, L = q_expressions[var]
#             qproblem = LinearVariationalProblem(a, L, self.q_aux_vars[var])
#             qsolver = LinearVariationalSolver(qproblem, solver_parameters=overall_solver_parameters[var], options_prefix=var)
#             self.q_aux_solvers.append(qsolver)
#
#
#         self.dfdx_aux_solvers = []
#         xq = {}
#         dfdx_expressions = {}
#         for var in self.model.get_dfdx_aux_var_list(terms=terms):
#             dfdx_expressions[var] = [0,0]
#         for i in range(self.npoints):
#             point, weight = self.points[i], self.weights[i]
#             for j,var in enumerate(self.model.variableset.varlist):
#                 xq[var] = (1. - float(point)) * self.xn.sub(j) + float(point) * self.xk.sub(j)
#             xq_dfdx_expressions = self.model.compute_dfdx_expressions(xq, terms=terms)
#             for var in self.model.get_dfdx_aux_var_list(terms=terms):
#                 a, L = make_a_L(*xq_dfdx_expressions[var], self.dx)
#                 dfdx_expressions[var][0] = dfdx_expressions[var][0] + float(weight) * a
#                 dfdx_expressions[var][1] = dfdx_expressions[var][1] + float(weight) * L
#         for var in self.model.get_dfdx_aux_var_list(terms=terms):
#             a, L = dfdx_expressions[var]
#             dfdx_problem = LinearVariationalProblem(a, L, self.dfdx_aux_vars[var])
#             dfdx_solver = LinearVariationalSolver(dfdx_problem, solver_parameters=overall_solver_parameters[var], options_prefix=var)
#             self.dfdx_aux_solvers.append(dfdx_solver)
#
#
#         xhat = self.model.variableset.get_test_var()
#         xhat_subs =  self.model.variableset.get_test_vars()
#
#         rhs_expr = model.rhs(self.q_aux_vars, self.dfdx_aux_vars, xhat_subs, terms=terms)
#
#
#         self.dt = Constant(1.)
#         self.t = Constant(1.)
#         self.tn = 0.
#
# #CANNOT REALLY DO FIXED POINT USING PETSC, OR AT LEAST I CAN'T FIGURE IT OUT
# #probably possible with some careful combination of NL solver options
#         if parameters['timestepping']['avf_solver'] == 'fixedpoint':
#             nl_expr = inner(xhat, self.xn)*self.dx - self.dt*rhs_expr
#             self.rhs_nl_solver = FixedPointSolver(nl_expr, self.xnp1, self.model.variableset.mixedspace,
#                 lambda state: self.pre_callback(state), lambda state: self.post_callback(state), self.dx)
# #SUPER UNCLEAR IF A LIMITER HERE FOR POST FUNCTION CALLBACK WILL ACTUALLY WORK?
#
#         elif parameters['timestepping']['avf_solver'] == 'qn':
#             nl_expr = inner(xhat, self.xnp1 - self.xn)*self.dx + self.dt*rhs_expr
#             rhs_linear_expr = model.linear_rhs(self.initcond.const_state, self.q_aux_vars, xhat_subs, terms=terms)
#             linear_expr = inner(xhat, self.xnp1 - self.xn)*self.dx + self.dt*rhs_linear_expr
#             J_expr = derivative(linear_expr, self.xnp1)
#             rhs_nl_problem = NonlinearVariationalProblem(nl_expr, self.xnp1, J=J_expr)
#             self.rhs_nl_solver = NonlinearVariationalSolver(rhs_nl_problem, solver_parameters=overall_solver_parameters['qn'],
#                 options_prefix = 'avf2qn', pre_function_callback=lambda state: self.pre_callback(state), post_function_callback=lambda state, jacobian: self.post_callback(state, jacobian))
#
#
#
#
#     def pre_callback(self, state):
#         with self.xk.dat.vec_wo as xk:
#             state.copy(xk)
#         for solver in self.q_aux_solvers:
#             solver.solve()
#         for solver in self.dfdx_aux_solvers:
#             solver.solve()
#
#     def initialize(self, init_xn=True):
#         if init_xn:
#             self.xn.zero()
#             varexpr = self.initcond.get_value(self.model.mesh, 0.0)
#             self.model.initialize(varexpr, self.xn)
#
#         #self.xnp1.zero()
#         #self.xk.zero()
#         self.xnp1.assign(self.xn)
# #        self.xn.dat.copy(self.xk.dat) #DONT THINK WE NEED THIS COPY
#         #self.xn.dat.copy(self.xnp1.dat)
#
# #THIS IS ACTUALLY BROKEN- UNCLEAR EXACTLY HOW TO MODIFY CURRENT STATE AFTER A STEP IS COMPUTED?
#     def post_callback(self, state, jacobian):
#         pass
#         #with self.xk.dat.vec_wo as xk:
#         #    state.copy(xk)
#         #self.model.post_step(self.xk_sub)
#
#     def take_step(self, dt):
#         self.dt.assign(dt)
#
# #HOW EXACTLY SHOULD self.t be handled here?
# #IT needs to be fed into rhs, J, etc. at the appropriate points...
# #especially BCs might be time dependent...
#         self.rhs_nl_solver.solve()
#         self.xn.assign(self.xnp1)
#         self.tn = self.tn + dt
#
#
#
# #FIX THIS EVENTUALLY
# #Here we need to somehow specify who is staggered, and also split various dHdx calcs, etc.
# class TimeStaggered_Integrator(TimeStepper):
#     def __init__(self, parameters, model, initcond, logger, xn=None, terms='all'):
#         self.model = model
#         self.initcond = initcond
#         self.logger = logger
#         self.parameters = parameters
#         self.dx = model.spaces.dx
#
# #FIX THIS
#     def initialize(self, init_xn=True):
#         if init_xn:
#             self.xn.zero()
#             varexpr = self.initcond.get_value(self.model.mesh, 0.0)
#             self.model.initialize(varexpr, self.xn)
#
#     def pre_stepA_solvers(self):
#         for solver in self.q_aux_solvers_A:
#             solver.solve()
#         for solver in self.dfdx_aux_solvers_A:
#             solver.solve()
#
#     def pre_stepB_solvers(self):
#         for solver in self.q_aux_solvers_B:
#             solver.solve()
#         for solver in self.dfdx_aux_solvers_B:
#             solver.solve()
#
# #FIX THIS
#     def take_step(self, dt):
#         self.dt.assign(dt)
# #DO WE NEED XK REALLY HERE? Probably not, can just update xn...
#         self.xk.assign(self.xn)
#         self.t.assign(self.tn)
#
# #SPLITTING IS TRICKY- PROBABLY WANT TWO SEPARATE XN VARIABLES? Kind of unclear
# #or do manual updates, etc.?
#         self.pre_stepA_solvers()
#         self.stepA_solver.solve()
#         self.xn.assign(self.xn + self.dt * self.F)
#         self.model.post_step(self.xn_sub, terms=self.terms)
#
#         self.xk.assign(self.xn)
#         self.t.assign(self.tn + dt/2.0)
#         self.pre_stepB_solvers()
#         self.stepB_solver.solve()
#         self.xn.assign(self.xn + self.dt * self.F)
#         self.model.post_step(self.xn_sub, terms=self.terms)
#
#         self.tn = self.tn + dt
#
# class RK_Integrator(TimeStepper):
#     def __init__(self, parameters, model, initcond, logger, xn=None, use_xn_as_xk=False, terms='all'):
#         self.model = model
#         self.initcond = initcond
#         self.logger = logger
#         self.parameters = parameters
#         self.terms = terms
#
#         self.dx = model.spaces.dx
#
#         self.q_aux_vars = self.model.get_q_aux_vars(terms=terms)
#         self.dfdx_aux_vars = self.model.get_dfdx_aux_vars(terms=terms)
#
#         if xn is None:
#             self.xn = self.model.get_x_var('xn')
#         else:
#             self.xn = xn
#
#         if not use_xn_as_xk:
#             self.xk = self.model.get_x_var('xk')
#         else:
#             self.xk = self.xn
#
#         self.xn_sub = {}
#         self.xk_sub = {}
#         for i,var in enumerate(self.model.variableset.varlist):
#             self.xn_sub[var] = self.xn.sub(i)
#             self.xk_sub[var] = self.xk.sub(i)
#             self.q_aux_vars[var] = self.xk.sub(i)
#
#
#         q_expressions = self.model.compute_q_expressions(self.q_aux_vars, terms=terms)
#         self.q_aux_solvers = []
#         for var in self.model.get_q_aux_var_list(terms=terms):
#             a, L = q_expressions[var]
#             qproblem = LinearVariationalProblem(a, L, self.q_aux_vars[var])
#             qsolver = LinearVariationalSolver(qproblem, solver_parameters=overall_solver_parameters[var], options_prefix=var)
#             self.q_aux_solvers.append(qsolver)
#
#         self.dfdx_aux_solvers = []
#         dfdx_expressions = self.model.compute_dfdx_expressions(self.q_aux_vars, terms=terms)
#         for var in self.model.get_dfdx_aux_var_list(terms=terms):
#             a, L = make_a_L(*dfdx_expressions[var], self.dx)
#             dfdx_problem = LinearVariationalProblem(a, L, self.dfdx_aux_vars[var])
#             dfdx_solver = LinearVariationalSolver(dfdx_problem, solver_parameters=overall_solver_parameters[var], options_prefix=var)
#             self.dfdx_aux_solvers.append(dfdx_solver)
#
#         self.dt = Constant(1.)
#         self.t = Constant(1.)
#         self.tn = 0.
#
#     def pre_step_solvers(self):
#         for solver in self.q_aux_solvers:
#             solver.solve()
#         for solver in self.dfdx_aux_solvers:
#             solver.solve()
#
#     def initialize(self, init_xn=True):
#         if init_xn:
#             self.xn.zero()
#             varexpr = self.initcond.get_value(self.model.mesh, 0.0)
#             self.model.initialize(varexpr, self.xn)
#
# class RK4_Integrator(RK_Integrator):
#
#     def __init__(self, parameters, model, initcond, logger, xn=None, terms='all'):
#         RK_Integrator.__init__(self, parameters, model, initcond, logger, xn=xn, terms=terms)
#
#         self.F1 = self.model.get_x_var('F1')
#         self.F2 = self.model.get_x_var('F2')
#         self.F3 = self.model.get_x_var('F3')
#         self.F4 = self.model.get_x_var('F4')
#
#         xhat = self.model.variableset.get_test_var()
#         xtrial = self.model.variableset.get_trial_var()
#         xhat_subs =  self.model.variableset.get_test_vars()
#
#         A = inner(xhat, xtrial)*self.dx
#         rhsproblem = -model.rhs(self.q_aux_vars, self.dfdx_aux_vars, xhat_subs, terms=terms)
#
#         F1problem = LinearVariationalProblem(A, rhsproblem, self.F1)
#         self.F1solver = LinearVariationalSolver(F1problem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'rk4-f1')
#
#         F2problem = LinearVariationalProblem(A, rhsproblem, self.F2)
#         self.F2solver = LinearVariationalSolver(F2problem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'rk4-f2')
#
#         F3problem = LinearVariationalProblem(A, rhsproblem, self.F3)
#         self.F3solver = LinearVariationalSolver(F3problem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'rk4-f3')
#
#         F4problem = LinearVariationalProblem(A, rhsproblem, self.F4)
#         self.F4solver = LinearVariationalSolver(F4problem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'rk4-f4')
#
#     def take_step(self, dt):
#         self.dt.assign(dt)
#
#         self.xk.assign(self.xn)
#         self.pre_step_solvers()
#         self.F1solver.solve()
#
#         self.xk.assign(self.xn + self.dt/2.0*self.F1)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#         self.t.assign(self.tn + self.dt/2.)
#         self.pre_step_solvers()
#         self.F2solver.solve()
#
#         self.xk.assign(self.xn + self.dt/2.0*self.F2)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#         self.t.assign(self.tn + self.dt/2.)
#         self.pre_step_solvers()
#         self.F3solver.solve()
#
#         self.xk.assign(self.xn + self.dt*self.F3)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#         self.t.assign(self.tn + self.dt)
#         self.pre_step_solvers()
#         self.F4solver.solve()
#
#         self.xn.assign(self.xn + self.dt/6. * (self.F1 + 2.*self.F2 + 2.*self.F3 + self.F4))
#         self.model.post_step(self.xn_sub, terms=self.terms)
#         self.tn = self.tn + dt
#
# #three register kinnmark + grey time integrators
# #DO THESE OFFER ANYTHING OVER THE 2 STAGE INTEGRATORS?
# #TALK TO MARK AND ANDREW STEYER
# #ie either better accuracy or more efficiency? My guess is 53 is hard to beat..
# class KGRK3_Integrator(RK_Integrator):
#
#     def __init__(self, parameters, model, initcond, logger, xn=None, terms='all'):
#         RK_Integrator.__init__(self, parameters, model, initcond, logger, xn=xn, terms=terms)
#
#         self.F1 = self.model.get_x_var('F1')
#         self.F2 = self.model.get_x_var('F2')
#
# #FIX THIS- THERE MIGHT BE TYPOS IN THE IMEX KG PAPER!
#         #set values for alpha, beta, etc.!
#         if parameters['timestepping']['nkg_stages'] == 3:
#             self.alpha = np.array([1./3., 1./3., 3./4.])
#             self.beta = np.array([0., ])
#             self.c = np.array([1./2., 1./2., 1.])
#         elif parameters['timestepping']['nkg_stages'] == 3:
#             self.alpha = np.array([1./4., 1./3., 1./2., 1.])
#             self.beta = np.array([0., ])
#             self.c = np.array([1./4., 1./3., 1./2., 1.])
#         elif parameters['timestepping']['nkg_stages'] == 3:
#             self.alpha = np.array([1./4., 1./6, 3./8., 1./2., 1.])
#             self.beta = np.array([0., ])
#             self.c = np.array([1./4., 1./6, 3./8., 1./2., 1.])
#
#         self.num_stages = self.alpha.shape[0]
#
#         xhat = self.model.variableset.get_test_var()
#         xtrial = self.model.variableset.get_trial_var()
#         xhat_subs =  self.model.variableset.get_test_vars()
#
#         A = inner(xhat, xtrial)*self.dx
#         rhsproblem = -model.rhs(self.q_aux_vars, self.dfdx_aux_vars, xhat_subs, terms=terms)
#
#         F1problem = LinearVariationalProblem(A, rhsproblem, self.F1)
#         self.F1solver = LinearVariationalSolver(F1problem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'kgrk3-f')
#
#         F2problem = LinearVariationalProblem(A, rhsproblem, self.F2)
#         self.F1solver = LinearVariationalSolver(F2problem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'kgrk3-f')
#
#
#     def take_step(self, dt):
#         self.dt.assign(dt)
#
#         self.xk.assign(self.xn)
#         self.t.assign(self.tn)
#         self.pre_step_solvers()
#         self.F1solver.solve()
#
#         for i in range(self.num_stages-1):
#             self.xk.assign(self.xn + self.dt * (self.beta[i] * self.F1 + self.alpha[i]* self.F2))
#             self.model.post_step(self.xk_sub, terms=self.terms)
#             self.t.assign(self.tn + self.c[i] * self.dt)
#             self.pre_step_solvers()
#             self.F2solver.solve()
#
#         self.xn.assign(self.xn + self.dt * (self.beta[-1] * self.F1 + self.alpha[-1] * self.F2))
#         self.model.post_step(self.xn_sub, terms=self.terms)
#         self.tn = self.tn + dt
#
# #two register kinnemark + grey time integrators
# class KGRK2_Integrator(RK_Integrator):
#
#     def __init__(self, parameters, model, initcond, logger, xn=None, terms='all'):
#         RK_Integrator.__init__(self, parameters, model, initcond, logger, xn=xn, terms=terms)
#
#         self.F = self.model.get_x_var('F')
#
#         #set values for alpha, etc.!
#         if parameters['timestepping']['kgrk2_name'] == '32': #NOT SURE THIS IS 2ND OR 1ST ORDER?
#             self.alpha = np.array([1./2., 1./2., 1.])
#             self.c = np.array([1./2., 1./2., 1.])
#         if parameters['timestepping']['kgrk2_name'] == '32a': #NOT SURE THIS IS 2ND OR 1ST ORDER?
#             self.alpha = np.array([1./3., 1./2., 1.])
#             self.c = np.array([1./3., 1./2., 1.])
#         elif parameters['timestepping']['kgrk2_name'] == '42': #NOT SURE THIS IS 2ND OR 1ST ORDER?
#             self.alpha = np.array([1./4., 1./3., 1./2., 1.])
#             self.c = np.array([1./4., 1./3., 1./2., 1.])
#         elif parameters['timestepping']['kgrk2_name'] == '52':
#             self.alpha = np.array([1./4., 1./6, 3./8., 1./2., 1.])
#             self.c = np.array([1./4., 1./6, 3./8., 1./2., 1.])
#         elif parameters['timestepping']['kgrk2_name'] == '53':
#             self.alpha = np.array([1./5., 1./5., 1./3., 1./2., 1.])
#             self.c = np.array([1./5., 1./5, 1./3., 1./2., 1.])
#         #SUPER UNCLEAR WHAT THE ORDER OF THESE METHODS ARE?
#         elif parameters['timestepping']['kgrk2_name'] == '62':
#             self.alpha = np.array([1./6., 2./15., 1./4., 1./3., 1./2., 1.])
#             self.c = np.array([1./6., 2./15., 1./4., 1./3., 1./2., 1.])
#         elif parameters['timestepping']['kgrk2_name'] == '72':
#             self.alpha = np.array([1./7., 2./21., 1./5., 8./35., 1./3., 1./2., 1.])
#             self.c = np.array([1./7., 2./21., 1./5., 8./35., 1./3., 1./2., 1.])
#         elif parameters['timestepping']['kgrk2_name'] == '82':
#             self.alpha = np.array([1./8., 1./14., 1./6., 1./6., 1./4., 1./3., 1./2., 1.])
#             self.c = np.array([1./8., 1./14., 1./6., 1./6., 1./4., 1./3., 1./2., 1.])
#         elif parameters['timestepping']['kgrk2_name'] == '92':
#             self.alpha = np.array([1./9., 1./18., 1./7., 8./63., 1./5., 5./21., 1./3., 1./2., 1.])
#             self.c = np.array([1./9., 1./18., 1./7., 8./63., 1./5., 5./21., 1./3., 1./2., 1.])
#         elif parameters['timestepping']['kgrk2_name'] == '102':
#             self.alpha = np.array([1./10., 2./45., 1./8., 1./10., 1./6., 9./50., 1./4., 1./3., 1./2., 1.])
#             self.c = np.array([1./10., 2./45., 1./8., 1./10., 1./6., 9./50., 1./4., 1./3., 1./2., 1.])
#
#
#
#
#
#         self.num_stages = self.alpha.shape[0]
#
#         xhat = self.model.variableset.get_test_var()
#         xtrial = self.model.variableset.get_trial_var()
#         xhat_subs =  self.model.variableset.get_test_vars()
#
#         A = inner(xhat, xtrial)*self.dx
#         rhsproblem = -model.rhs(self.q_aux_vars, self.dfdx_aux_vars, xhat_subs, terms=terms)
#
#         Fproblem = LinearVariationalProblem(A, rhsproblem, self.F)
#         self.Fsolver = LinearVariationalSolver(Fproblem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'kgrk2-f')
#
#     def take_step(self, dt):
#         self.dt.assign(dt)
#
#         self.xk.assign(self.xn)
#         self.t.assign(self.tn)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
#         for i in range(self.num_stages):
#             self.xk.assign(self.xn + self.dt * self.alpha[i] * self.F)
#             self.model.post_step(self.xk_sub, terms=self.terms)
#             self.t.assign(self.tn + self.c[i] * self.dt)
#             self.pre_step_solvers()
#             self.Fsolver.solve()
#
#         self.xn.assign(self.xn + self.dt * self.alpha[-1] * self.F)
#         self.model.post_step(self.xn_sub, terms=self.terms)
#         self.tn = self.tn + dt
#
# class TimeSplitIntegrator(TimeStepper):
#     def __init__(self, parameters, model, initcond, logger):
#         self.model = model
#         self.initcond = initcond
#         self.logger = logger
#         self.parameters = parameters
#
#         self.num_subcycles = parameters['timestepping']['timestepper_substeps']
#
#         termlist = parameters['timestepping']['timestepper_split_terms']
#
#         self.xn = model.get_x_var('xn')
#         self.xn_sub = {}
#         for i,var in enumerate(self.model.variableset.varlist):
#             self.xn_sub[var] = self.xn.sub(i)
#
#         self.time_integrators = []
#         for i,time_integrator_name in enumerate(parameters['timestepping']['timestepper_list']):
#             time_integrator = get_time_integrator(time_integrator_name)
#             self.time_integrators.append(time_integrator(parameters, model, initcond, logger, xn=self.xn, terms=termlist[i]))
#
#     def initialize(self):
#         self.xn.zero()
#         varexpr = self.initcond.get_value(self.model.mesh, 0.0)
#         self.model.initialize(varexpr, self.xn)
#         for time_integrator in self.time_integrators:
#             time_integrator.initialize(init_xn=False)
#
# #ADD ABILITY TO INTERLEAVE THIS ORDERING IE model, HYPER, DYNANICS, HPER, PHYSICS
# #ADD ABILITY TO SWITCH BETWEEN LIE AND STRANG SPLITTING
#     def take_step(self, dt):
#         for i,time_integrator in enumerate(self.time_integrators):
#             for k in range(self.num_subcycles[i]):
#                 time_integrator.take_step(dt/self.num_subcycles[i])
#
# class Euler_Integrator(RK_Integrator):
#
#     def __init__(self, parameters, model, initcond, logger, xn=None, terms='all'):
#         RK_Integrator.__init__(self, parameters, model, initcond, logger, xn=xn, use_xn_as_xk=True, terms=terms)
#
#         self.F1 = self.model.get_x_var('F1')
#
#         xhat = self.model.variableset.get_test_var()
#         xtrial = self.model.variableset.get_trial_var()
#         xhat_subs =  self.model.variableset.get_test_vars()
#
#         A = inner(xhat, xtrial)*self.dx
#         rhsproblem = -model.rhs(self.q_aux_vars, self.dfdx_aux_vars, xhat_subs, terms=terms)
#
#         F1problem = LinearVariationalProblem(A, rhsproblem, self.F1)
#         self.F1solver = LinearVariationalSolver(F1problem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'euler-f')
#
#     def take_step(self, dt):
#         self.dt.assign(dt)
#
#         self.t.assign(self.tn)
#         self.pre_step_solvers()
#         self.F1solver.solve()
#
#         self.xn.assign(self.xn + self.dt * self.F1)
#         self.model.post_step(self.xn_sub, terms=self.terms)
#
#         self.tn = self.tn + dt
#
# class SSPRK3_Integrator(RK_Integrator):
#     def __init__(self, parameters, model, initcond, logger, xn=None, terms='all'):
#         RK_Integrator.__init__(self, parameters, model, initcond, logger, xn=xn, terms=terms)
#
#         self.F = self.model.get_x_var('F')
#
#         xhat = self.model.variableset.get_test_var()
#         xtrial = self.model.variableset.get_trial_var()
#         xhat_subs =  self.model.variableset.get_test_vars()
#
#         A = inner(xhat, xtrial)*self.dx
#         rhsproblem = -model.rhs(self.q_aux_vars, self.dfdx_aux_vars, xhat_subs, terms=terms)
#
#         Fproblem = LinearVariationalProblem(A, rhsproblem, self.F)
#         self.Fsolver = LinearVariationalSolver(Fproblem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'ssprk3-f')
#
#     def take_step(self, dt):
#         self.dt.assign(dt)
#         self.t.assign(self.tn)
#
#         self.xk.assign(self.xn)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
# #ALL WRONG
#         self.xk.assign(self.xn + 1./2. * self.dt * self.F)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#
# #        self.t.assign(self.t + 1/2. * self.dt)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
#         self.xk.assign(self.xk + 1./2. * self.dt * self.F)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#
# #        self.t.assign(self.t + 1/2. * self.dt)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
#         self.xk.assign(2./3. * self.xn + 1./3. * self.xk + 1./6. * self.dt * self.F)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#
# #        self.t.assign(self.t + 1/2. * self.dt)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
#         self.xn.assign(self.xk + self.dt * 1./2. * self.F)
#         self.model.post_step(self.xn_sub, terms=self.terms)
#
#         self.tn = self.tn + dt
#
# class SSPRK43_Integrator(RK_Integrator):
#
#     def __init__(self, parameters, model, initcond, logger, xn=None, terms='all'):
#         RK_Integrator.__init__(self, parameters, model, initcond, logger, xn=xn, terms=terms)
#
#         self.F = self.model.get_x_var('F')
#
#         xhat = self.model.variableset.get_test_var()
#         xtrial = self.model.variableset.get_trial_var()
#         xhat_subs =  self.model.variableset.get_test_vars()
#
#         A = inner(xhat, xtrial)*self.dx
#         rhsproblem = -model.rhs(self.q_aux_vars, self.dfdx_aux_vars, xhat_subs, terms=terms)
#
#         Fproblem = LinearVariationalProblem(A, rhsproblem, self.F)
#         self.Fsolver = LinearVariationalSolver(Fproblem, solver_parameters=overall_solver_parameters['rkstage'], options_prefix = 'ssprk43-f')
#
#     def take_step(self, dt):
#         self.dt.assign(dt)
#         self.t.assign(self.tn)
#
#         self.xk.assign(self.xn)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
#         self.xk.assign(self.xn + 1./2. * self.dt * self.F)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#
# #        self.t.assign(self.t + 1/2. * self.dt)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
#         self.xk.assign(self.xk + 1./2. * self.dt * self.F)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#
# #        self.t.assign(self.t + 1/2. * self.dt)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
#         self.xk.assign(2./3. * self.xn + 1./3. * self.xk + 1./6. * self.dt * self.F)
#         self.model.post_step(self.xk_sub, terms=self.terms)
#
# #        self.t.assign(self.t + 1/2. * self.dt)
#         self.pre_step_solvers()
#         self.Fsolver.solve()
#
#         self.xn.assign(self.xk + self.dt * 1./2. * self.F)
#         self.model.post_step(self.xn_sub, terms=self.terms)
#
#         self.tn = self.tn + dt
