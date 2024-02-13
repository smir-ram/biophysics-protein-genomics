# Loss Function

## `reconstruction loss`


## `KL Divergence`

### Prior
The prior distribution typically does not depend directly on the input `x`. Instead, the prior is usually chosen to be a standard distribution that is **fixed before training** and does not change with the input. The most common choice for the prior is a standard normal or **gaussian  distribution**, $N(0,I)$, where $\mu$= 0is a zero mean vector and $I$ is the identity matrix as the covariance, indicating that the **latent variables are independent and identically distributed**.
 
 1. Why a Gaussian Distribution?
 * Training and inference computation are more tractable.
 * Facilitates application of the `reparametrization trick` and calculation of `KL divergence`
 * `Regularization` - comes from the KL Divergence loss term  which penalizes deviations from the prior distribution. Thus the organizes the latent space that effectively prevent overfitting. 
 
 2. Why *fixed* prior that does not adapt to specific inputs?
 * forces the model to learn the encoded distributions to resemble the prior ($N(0,I)$; covariance $I$ is an identity matrix, ensuring the latent variables are independent and identically distributed) and reconstruct the input variables accurately.
 
 3. Advanced Models have `conditional priors` - how do they work?
 * *blah*


