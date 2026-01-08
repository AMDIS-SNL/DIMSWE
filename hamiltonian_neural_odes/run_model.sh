#source ~/venv-firedrake/bin/activate

rm *.png
python model.py
python plot_results.py
