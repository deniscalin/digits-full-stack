import numpy as np # linear algebra
import torch
import torch.nn.functional as F
import os


device = 'cpu'
vocab_size = 10

# SETTING GLOBAL RNG SEED
global_seed = 42
print(f"Setting the RNG seed to {global_seed}")
# g = torch.Generator().manual_seed(global_seed) # Generator for reproducibility
# Set global and generation seeds for RNGs
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
    

# Printing new parameter shapes:  
# [torch.Size([784, 512]), torch.Size([512]), torch.Size([512]), torch.Size([512]), torch.Size([1, 512]), torch.Size([1, 512]), 
# torch.Size([512, 256]), torch.Size([256]), torch.Size([256]), torch.Size([256]), torch.Size([1, 256]), torch.Size([1, 256]), 
# torch.Size([256, 128]), torch.Size([128]), torch.Size([128]), torch.Size([128]), torch.Size([1, 128]), torch.Size([1, 128]), 
# torch.Size([128, 64]), torch.Size([64]), torch.Size([64]), torch.Size([64]), torch.Size([1, 64]), torch.Size([1, 64]), 
# torch.Size([64, 10]), torch.Size([10]), torch.Size([10]), torch.Size([10]), torch.Size([1, 10]), torch.Size([1, 10])]
    
print("Preparing to load weights")
loaded_model: dict = torch.load('/Users/deniscalin/Desktop/digit_recognizer_frontend/digit-recognizer-front-end/flask/patched_layers_300k.pt')
print("Loaded weights, setting layers")
layers = loaded_model['patched_layers']

# print("Preparing to load weights")
# loaded_model: dict = torch.load('/Users/deniscalin/Desktop/digit_recognizer_frontend/digit-recognizer-front-end/flask/model_300k_params.pt')
# print("Loaded weights, setting layers")
# p = loaded_model['params']

# layers = [
#             Linear(784, 512, p[0], p[1]),       BatchNorm1d(512, p[2], p[3], p[4], p[5]), ReLU(),
#             Linear(512, 256, p[6], p[7]),       BatchNorm1d(256, p[8], p[9], p[10], p[11]), ReLU(),
#             Linear(256, 128, p[12], p[13]),       BatchNorm1d(128, p[14], p[15], p[16], p[17]), ReLU(),
#             Linear(128, 64, p[18], p[19]),        BatchNorm1d(64, p[20], p[21], p[22], p[23]),  ReLU(),
#             Linear(64, vocab_size, p[24], p[25]), BatchNorm1d(vocab_size, p[26], p[27], p[28], p[29])
#         ]


# # Set BatchNorm layers to inference mode
# for layer in layers:
#     if isinstance(layer, BatchNorm1d):
#         print(layer)
#         layer.training = False
#         print("Set BatchNorm1D training to false")
#     print(layer)

# model_obj = {'patched_layers': layers}

# torch.save(model_obj, 'patched_layers_300k.pt')

@torch.no_grad()
def inference_function(x_batch):
    """The inference function, which takes the input tensor of 1 image, 
    and returns the label prediction and confidence score for this prediction.
    
    Args:
        x_batch | input torch tensor vector of shape [784]"""
    # Normalize the input
    x_batch = (x_batch - torch.min(x_batch)) / (torch.max(x_batch) - torch.min(x_batch))
    # print("x_batch shape: ", x_batch.shape)
    # x_batch = x_batch.view(idx.shape[0], -1)
    # print("x_batch shape after reshape: ", x_batch.shape)
    for layer in layers:
        x_batch = layer(x_batch)
    # print('Logits: ', x_batch)
    # print('Logits shape: ', x_batch.shape)
    probs = F.softmax(x_batch, dim=-1)
    print('Probabilities: ', probs)
    # ix = probs.argmax()
    # ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g)
    ix = probs.argmax()
    # print("INSIDE DIGIT_RECOG_MODEL. ix: ", ix)
    # print("INSIDE DIGIT_RECOG_MODEL. ix shape: ", ix.shape)
    # print("probs shape: ", probs.shape)
    # print("ix shape: ", ix.shape)
    # print("ix: ", ix)
    conf_score = probs[0][ix]
    # print("conf_score shape: ", conf_score.shape)
    ix = ix.item()
    conf_score = conf_score.item() * 100 
    # print("Prediction: ", ix)
    return ix, conf_score
