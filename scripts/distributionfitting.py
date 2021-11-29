"""
Statistical inference with pyro and pytorch
"""

import torch
import pyro
import numpy as np
from scipy.stats import wasserstein_distance

# the fitting function:
def fit_dist(data, model, print_losses=False,
    trainingparams:dict={
        'stepsize':3e-5,
        'annealing':1e-5,
        'no_episodes':1500
    }, hide_trace=['block'], batchsize=100):
    """
    fits a given pyro distribution function with named parameters
    """
    # clears parameter store:
    pyro.clear_param_store()
    # data needs to be a tensor:
    data = torch.tensor(data, dtype=torch.float)
    # add model and guide:
    guide = pyro.infer.autoguide.AutoDelta(pyro.poutine.block(model, hide=hide_trace)) 
    # set up stochastic gradient descent and stochastic variational inference
    sgd = torch.optim.SGD
    optimizer = pyro.optim.ExponentialLR({'optimizer': sgd, 'optim_args': {'lr': trainingparams['stepsize']}, 'gamma': trainingparams['annealing']})
    svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO())
    # run optimization:
    loss = []
    for e in range(trainingparams['no_episodes']):
        batch_index = np.random.choice(np.arange(len(data)), batchsize)
        loss.append(
            svi.step(data[batch_index])
        )
        if print_losses:
            print(f'episode {e+1} - SVI: {loss[-1]}')
    return loss

def featurescaling(data):
    """
    standardizes the given dataset
    """
    max_value = data.max()
    min_value = data.min()
    data_standardized = (data - min_value)/(max_value - min_value)
    return data_standardized, max_value, min_value

def rescale(samples, max_value, min_value):
    """
    re-do the standardization
    """
    samples_rescaled = samples * (max_value - min_value) + min_value
    return samples_rescaled

# the distributions:
def normal(data=None, samplesize=1):
    """
    simple normal distribution
    """
    # samplesize:
    if data is not None:
        samplesize = len(data)
    # parameters:
    mu = pyro.param("normal_mu", torch.tensor(0., dtype=torch.float))
    sigma = pyro.param("normal_sigma", torch.tensor(1., dtype=torch.float), constraint=torch.distributions.constraints.positive)
    # distribution:
    with pyro.plate('observations', samplesize):
        sample = pyro.sample("normal_dist", pyro.distributions.Normal(mu, sigma), obs=data)  
    # convert to numpy:
    sample = sample.detach().numpy()
    return sample

def lognormal(data=None, samplesize=1):
    """
    simple lognormal distribution
    """
    # samplesize:
    if data is not None:
        samplesize = len(data)
    # parameters:
    mu = pyro.param("lognormal_mu", torch.tensor(0., dtype=torch.float))
    sigma = pyro.param("lognormal_sigma", torch.tensor(1., dtype=torch.float), constraint=torch.distributions.constraints.positive)
    # distribution:
    with pyro.plate('observations', samplesize):
        sample = pyro.sample("lognormal_dist", pyro.distributions.Normal(mu, sigma), obs=data)  
    # convert to numpy:
    sample = sample.detach().numpy()
    return sample

def exponential(data=None, samplesize=1):
    """
    simple exponential distribution
    """    
    # samplesize:
    if data is not None:
        samplesize = len(data)
    # parameters:
    rate = pyro.param("exponential_rate", torch.tensor(0.25, dtype=torch.float),  constraint=torch.distributions.constraints.positive)
    # distribution:
    with pyro.plate('observations', samplesize):
        sample = pyro.sample("exponential_dist", pyro.distributions.Exponential(rate), obs=data)  
    # convert to numpy:
    sample = sample.detach().numpy()
    return sample

if __name__ == "__main__":  
    distribution = normal # change for chosen distribution
    # generate random test data:
    data = np.random.normal(0,1,size=1000)
    # standardize data:
    data_standardized, maximum, minimum = featurescaling(data)    
    # fit distribution:
    loss = fit_dist(data_standardized, distribution)
    # sample from distribution:
    samples = distribution(samplesize=len(data))
    # rescale the distribution:
    samples = rescale(samples, maximum, minimum)
    #calculate the wasserstein distance:
    wd = wasserstein_distance(data, samples)
    # get the parameters:
    parameters = pyro.get_param_store()._params
    print(parameters)
    # plot results:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.histplot(data, color='orange', alpha=0.66)
    sns.histplot(samples, color='blue', alpha=0.66)
    plt.title(wd)
    plt.show()
