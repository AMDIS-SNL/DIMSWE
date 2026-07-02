source ~/venv-firedrake/bin/activate
mpiexec -n $2 python -m dimswe.run_model $1
