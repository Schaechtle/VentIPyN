

KL Matrices for Inference-Test
************************

Inferring over a smooth function
===================
Below we see an example of the KL divergence between the estimated predictive Gaussian and the true Gaussian. This is note included in Nose yet since I am not sure how to automate this. At the moment, I am only eyeballing it. 

We are using a version of the Neal example (see gpmem paper) and an SE-kernel plus a WN-kernel.

x-axis: number of observations.

y-axis: number of nested mh-steps.

.. image:: kl_2.png

