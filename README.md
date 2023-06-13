# dirichlet-v-mse
Julia code generating a comparison of NNs trained on MSE and Dirichlet loss functions. Adam is used to train both NNs, which are fully-connected with 5 hidden layers of width 5 and relu activation functions.

* fake-o_data.jl - file generating data used to create results. uses RNG with unfixed seed.
* model_dirichlet-loss.jl - fits model to data based on Dirichlet loss and generates figures in the style of model_fit_dir_*.png
* model_mse-loss.jl - fits model to data based on MSE loss and generates figures in the style of model_fit_mse_*.png
* DirichletViz.jl - contains functions for creating animations of PDF over the 3-simplex in the style of dirichlet.mp4

Observation: Occasionally, models will converge to constant functions in both cases.
