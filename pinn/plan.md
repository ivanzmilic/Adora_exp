# Instructions for the development of Adora PINN-based inversion
# After the instructions there are more concrete steps that will be executed sequentually, with my interaction
# Each step will be coded, verified with a toy-model jupyter notebook, edited and after we are all happy with it, proceed to the next step. 

## 1) Atmosphere parametrization 

Atmosphere is represented via an MLP-based PINN, with Gaussian Fourier encoding, similar to our code in ~/codes/cloudtorch/ 

This PINN physically encodes the physical parameters: T, p, v_z, \vec(B), if we run into anything else we will add it here.

The PINN is to be *PRETRAINED* on a starting atmosphere so it can start "warm"

## 2) Fit loss

In each a random set of points is selected on a x,y grid and compared to the closest given observation. This is done in the following way: 

### 2a) For chosen x,y coordinate a z-grid is constructed in a pre-determined z range using pre-determined z-sampling. This only matters for the RT solver

### 2b) Radiative transfer equation is solved for a pre-determined wavelength grid and chosen spectral lines 

### 2c) A loss is constructed in a fasion if Chi-sq (basically weighted MSE)

## 3) Physics LOSS - HE 

In a different set of random points (akka collocation points), that are sampled randomly (or, in some other fashion) in x,y,z, hydrostatic equilibrium is evaluated 

### 3a) In the basic functionallity this he is to be evaluated only in z: dp/dz = - \rho g, \rho can be calculated via eos we made recently

### 3b) After that we will work on expanding it, and including turbulent pressure 

## 4) Physics LOSS - MHS and expansions 

This can wait for the moment 

## 5) Physics LOSS - Div B = 0

This can wait for the moment 

## 6) Regularization loss 

This can wait for the moment 

## 7) Boundary loss 

The only boundary we will prescribe right now is the gas pressure at the top boundary. We will implement it in the pinn a a soft-enforced boundary, with high weights to enforce it 

# Steps for implementation. 

Everything goes in to ~/codes/adora/pinn, every step has a max one-page pdf write up, every step has a minimal notebook to make sure what it does. 

1) Devise MLP for atmospheric parametrization. We go for MLP + Gaussian Fourier encoding. 

2) Create an example starting atmosphere, for example 128 x 128 cloned FALC, reduced to the height (-100, 1000 km)

3) Fit the MLP to the starting atmosphere to secure a warm start. 

4) Calculate the spectrum of the 6301/6302 lines from this initial model. 

5) Create a test-inversion from one of the synthetic cubes we have prepared. 

6) Calculate all the losses and test whether the derivatives make sense. 

* FIRST BIG CHECKPOINT * we will move forward after these steps are clearly implemented and made sure we run. 

* CHECKPOINT SUCCESFULLY PASSED * 

7) Make a prototype inversion in a notebook. Use all the machinery that was built to make a step by step notebook that really inverts the observed batch.

