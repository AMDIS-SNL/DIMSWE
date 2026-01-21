from firedrake.adjoint import *
from firedrake import *
continue_annotation()

n = 30
mesh = UnitIntervalMesh(n)
timestep = Constant(1.0/n)
steps = 10

x, = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 2)
ic = project(sin(2.*pi*x), V, name="ic")

u_old = Function(V, name="u_old")
u_new = Function(V, name="u")
v = TestFunction(V)
u_old.assign(ic)
nu = Constant(0.0001)
F = ((u_new-u_old)/timestep*v
     + u_new*u_new.dx(0)*v + nu*u_new.dx(0)*v.dx(0))*dx
bc = DirichletBC(V, 0.0, "on_boundary")
problem = NonlinearVariationalProblem(F, u_new, bcs=bc)
solver = NonlinearVariationalSolver(problem)

J = assemble(ic*ic*dx)

for _ in range(steps):
    solver.solve()
    u_old.assign(u_new)
    J += assemble(u_new*u_new*dx)
pause_annotation()
print(round(J, 3))

Jhat = ReducedFunctional(J, Control(ic))

ic_new = project(sin(pi*x), V)
J_new = Jhat(ic_new)
print(round(J_new, 3))

dJ = Jhat.derivative()

get_working_tape().progress_bar = ProgressBar

dJ = Jhat.derivative()

dm = assemble(interpolate(Constant(1.), V))
rate = taylor_test(Jhat, ic, dm)
