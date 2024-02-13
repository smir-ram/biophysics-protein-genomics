# **E**xcited **L**ions **D**elightedly **R**oar **L**oudly

VAEs enables compression, reconstruction and also generation of new data that is similar to the original dataset.

## Encoder
Transforms input features to a condensed parameter space (latent space). Usually a fully connected or a CNN neural network architecture depending on the problem. Provides a `mean` and `variance` for each latent variable thus quantifying the uncertinity of the input features.

## Latent Space
A probabilistic representation space defined by the `mean` and `variance` outputted by the encoder. The posterior probability $q(z|x)$ (z:latent variables *given* x:input features) is sampled from the distribution (eg. Gaussian) constructed for each latent variable with respective `mean` and `variance`. 

## Decoder
Takes the samples the latent space ($Q$) and reconstructs the inputs. The quality of reconstruction is critical to the learning process `reconstruction loss`

## Reparametriation Trick
A trick that enables the application of `gradient descent for backpropagating loss` through a neural network, which would otherwise be unfeasible; the stochastic nature of certain operations (sampling from latent space) prevents the gradient from being directly computed.

## Loss Function
Is a combination of the `reconstruction loss` - measures the the ability of the decoder to reconstruct data accurately and `KL Divergence` of the posterior ($q$) and the prior ($P(x)$ - fixed before training, usually a gaussian. This function guides the training to balance between accurate reconstruction and well-structured latent space.