import numpy as np # linear algebra
import torch
import torch.nn.functional as F
import os


device = 'cpu'
vocab_size = 10

# SETTING GLOBAL RNG SEED
global_seed = 42
print(f"Setting the RNG seed to {global_seed}")
g = torch.manual_seed(global_seed)


# MLP
class Linear:
    
    def __init__(self, fan_in, fan_out, param_1, param_2, bias=True):
        self.weight = param_1
        self.bias = param_2
        
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    

class BatchNorm1d:
    
    def __init__(self, dim, param_1, param_2, param_3, param_4, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = False
        # Trained parameters
        self.gamma = param_1
        self.beta = param_2
        # Buffers (trained with momentum update)
        self.running_mean = param_3
        self.running_var = param_4
        
    def __call__(self, x):
        # Calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True) # batch mean
            xvar = x.var(0, keepdim=True, unbiased=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # Update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta, self.running_mean, self.running_var]
    
    
class Tanh:
    
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []
    

class ReLU:

    def __call__(self, x):
        self.out = torch.relu(x)
        return self.out
    
    def parameters(self):
        return []
    
    
print("Preparing to load weights")
loaded_model: dict = torch.load('/Users/deniscalin/Desktop/digit_recognizer_frontend/digit-recognizer-front-end/flask/patched_layers_300k.pt')
print("Loaded weights, setting layers")
layers = loaded_model['patched_layers']


@torch.no_grad()
def inference_function(x_batch):
    """The inference function, which takes the input tensor of 1 image, 
    and returns the label prediction and confidence score for this prediction.
    
    Args:
        x_batch | input torch tensor vector of shape [784]"""
    # Normalize the input
    x_batch = (x_batch - torch.min(x_batch)) / (torch.max(x_batch) - torch.min(x_batch))
    for layer in layers:
        x_batch = layer(x_batch)
    probs = F.softmax(x_batch, dim=-1)
    print('Probabilities: ', probs)
    ix = probs.argmax()
    conf_score = probs[0][ix]
    ix = ix.item()
    conf_score = conf_score.item() * 100
    return ix, conf_score
