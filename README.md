# mrf_autograd

This library provides Bloch functions that can be used to calculate Cramer Rao Bounds (CRBs) for MRI sequences.

Functions are provided for the autograd, and Tensorflow libraries. Pros and cons are as follows.

Autograd:

Pros:
* Lightweight with fewer dependencies
* More intuitive and easier to use
* Supports both forward and reverse differentiation, which makes taking gradient of CRB objective much easier

Cons:
* Must strictly adhere to the rules outlined in the Autograd package (eg no direct assignment of arrays). This adds a risk of bugs in the code, but can be avoided by verifying your gradients numerically.
* Limit on the number of parameters that you can calculate the CRB for to 3, since we cannot use the numpy inverse function for the derivative of the objective function. Calculating the matrix inverse of the Jacobian matrix using the method of adjoint matrix and cofactors is technically possible but will likely be very slow. I have added an analytical matrix inverse for the case of 3x3 matrices. Simply wrap the matrix in lists and autograd will follow them through the primitive matrix inverse claculation.

Tensorflow:

Pros:
* Tensorboard provides nice visualizations
* Supports taking derivatives of the CRB objective better
* Don't have to worry about adhering to strange assignment rules
* tf.gradients() can be used to calculate multiple derivatives at once which reuses the forward pass, unlike autograd for separate jacobian calls
* This can possibly be avoided by combining all variables to differentiate wrt in a single np.array, but this is less clear and this has not yet been tested yet

Cons:
* Some would say the graph design followed by execution is unintuitive
* Slightly more memory intensive
* Much slower, and more difficult to parallelize


TL;DR - if you only want to calculate the CRB objective, you can use either autograd or Tensorflow. For calculating the derivative of the crb using backpropagation, it is suggested to autograd


# List of Jupyter notebooks

1. OptimizeDESPOT1.ipynb / OptimizeDESS.ipynb - Builds the graphs for a simple steady state sequence and optimizes the sequence parameters. 
2. OptimizeSpinEchoT2.ipynb - Demonstration of analytical vs AD solution for CRLB
3. OptimizeMRF\_autograd.ipynb - Runs the optimization for the MR Fingerprinting sequence using CRLB calculated with autograd. The Bloch simulation is done in bloch/mrf\_spoiled\_crlb
4. PlotIntermediateSolutions.ipynb - Iterations from OptimizeMRF\_autograd.ipynb are saved to a text file in tmp/ which can be plotted using this notebook
5. MRFRuntimeTest.ipynb - Compares derivative obtained using AD and finite differences for a range of TRs


# Notes for extending functionality

* The mrf_ir_fisp_real function can be easily changed to add a phi array as the parameters. The 'real' part of the function name refers to taking the real part of the echos, but the calculation is correct for any provided phase.



