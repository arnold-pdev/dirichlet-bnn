# Repo contents:
## Dirichlet-likelihood BNN with Horseshoe priors
## 1. dirichlet-v-mse
Julia code generating a comparison of NNs trained on MSE and Dirichlet loss functions. Adam is used to train both NNs, which are fully-connected with 5 hidden layers of width 5 and relu activation functions.

* fake-o_data.jl - file generating data used to create results. uses RNG with unfixed seed.
* model_dirichlet-loss.jl - fits model to data based on Dirichlet loss and generates figures in the style of model_fit_dir_*.png
* model_mse-loss.jl - fits model to data based on MSE loss and generates figures in the style of model_fit_mse_*.png
* DirichletViz.jl - contains functions for creating animations of PDF over the 3-simplex in the style of dirichlet.mp4

Observation: Occasionally, models will converge to constant functions in both cases. DirichletViz.jl takes several (let's say 5) minutes to generate an animation. Animations are created using CairoMakie.jl through the function *dirichletanim*. The function *dirichletanime* was intended to represent training data in the animation through scatter, but is incomplete.

## 2. chemcam-bnn
This folder contains DNN/DNN-dirichlet-loss.jl, which can be used to train a deterministic neural network on ChemCam data using a Dirichlet loss function, and DNN/ChemCamDNNViz.jl, which can be used to visualize the results with lower and upper quantiles constructed from samples of $\alpha$.

Moreover, this folder contains BNN/ChemCam-ΩBNN-α.jl, which is the big, important script for using inference to learn the posterior landscape of the corresponding BNN with horseshoe priors! The horseshoe priors are implemented through a custom distribution using the "tilted-form" as described in presentation 3 (which can be found in the visualization folder). It *should be* nearly ready for parallel implementation (using Julia's mapreduce functionality). Proper sampling *might* require tweaking of HMC parameters, and other forms of MCMC that have been implemented in Turing (such as NUTS and Particle Gibbs) can be tried! Turing has many interesting features, and the definition of this Turing model should provide an interesting testbed for them.

CSV files are automatically saved at the end of the MCMC runs. They have quite long, informative names.

## 3. visualization
Includes HMC and Dirichlet animation code, as well as a stand-alone, simple implementation of 2D HMC. Presentations 1-3 are also included.

## 4. UKF timeseries
This is more or less a dump of the julia files and some figures made in the process of experimenting with learning time series dynamics. For the extra curious!
