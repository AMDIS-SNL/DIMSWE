source ~/venv-firedrake/bin/activate
rm *.png *.webp
python -m dimswe.plot_model $1
