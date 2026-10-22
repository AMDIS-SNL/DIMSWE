source ~/venv-firedrake/bin/activate
mpiexec -n $2 python ic_optimize.py $1 -log_view :flamegraph.txt:ascii_flamegraph
./flamegraph.pl --countname us flamegraph.txt > flamegraph.svg
