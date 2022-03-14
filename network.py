import numpy as np
import matplotlib.pyplot as plt
from torch import distributions
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import pickle
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel

#criterion = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()


def my_loss(pred, true):
    weight_scale = true.shape[0]/torch.sum(true).detach() # calculate over all dataset
    weight = torch.ones_like(true)
    weight[true == 1] = weight_scale
    loss_f = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_f(pred, true) * weight
    
    return loss.mean()

criterion = my_loss
# save the model
def save_model_info(model, info, name='model'):
    torch.save(model.state_dict(), f'models/{name}_model.pt')
    pickle.dump(info, open(f'models/{name}_info.pkl','wb'))
    
def load_model_info(name, model):
    model.load_state_dict(torch.load(f'models/{name}_model.pt'))
    info = pickle.load(open(f'models/{name}_info.pkl', 'rb'))
    return model, info
    
    
    
def train_network(model, train_dataloader, train_dataset, test_dataloader, test_dataset, n_epochs = 100, print_freq = 10, optimizer=None):
    info = {
        'train_loss':[],
        'train_acc':[],
        'test_loss':[],
        'test_acc':[]
    }

    for epoch in range(1, n_epochs+1):  # loop over the dataset multiple times

        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs[:,0] /= 244
            inputs[:,1] /= (3.14*2)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # get statistics
            train_loss += loss.item()
            predicted = torch.sigmoid(outputs)
            train_accuracy += (torch.round(predicted) == labels).sum().item()

        # normalize
        train_loss /= len(train_dataset)
        train_accuracy /= len(train_dataset)

        info['train_acc'].append(train_accuracy)
        info['train_loss'].append(train_loss)
            
        model.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                
                inputs[:,0] /= 244
                inputs[:,1] /= (3.14*2)

                # forward + backward + optimize
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                # get statistics
                test_loss += loss.item()
                predicted = torch.sigmoid(outputs)
                test_accuracy += (torch.round(predicted) == labels).sum().item()

        # normalize
        test_loss /= len(test_dataset)
        test_accuracy /= len(test_dataset)

        info['test_acc'].append(test_accuracy)
        info['test_loss'].append(test_loss)
        
        if epoch % print_freq == 0:

            print(f'Epoch: {epoch}')
            print(f'Train loss: {train_loss} | Test losses: {test_loss}')
            print(f'Train acc: {train_accuracy} | Test acc: {test_accuracy}')

    return model, info


def train_network_GP(model, likelihood, train_dataset, test_dataset, n_epochs = 100, print_freq = 10, optimizer=None):
    info = {
        'train_loss':[],
        # 'train_acc':[],
        'test_loss':[],
        # 'test_acc':[]
    }

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # inputs, labels = train_dataset[:,:2], train_dataset[:,2]
    # inputs[:,0] /= 244
    # inputs[:,1] /= (3.14*2) 
    # train_dataset[:,0]/=244
    # train_dataset[:,1]/=(3.14*2)

    for epoch in range(1, n_epochs+1):  # loop over the dataset multiple times

        model.train()
        likelihood.train()

        train_loss = 0.0
        # train_accuracy = 0.0
        # for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
        inputs, labels = train_dataset[:,:2], train_dataset[:,2]
        # inputs[:,0] /= 244
        # inputs[:,1] /= (3.14*2)

        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = likelihood(model(inputs))
        # Calc loss and backprop gradients
        loss = -mll(output, labels).sum()
        loss.backward()
        optimizer.step()

        # get statistics
        train_loss = loss.item()

        # normalize
        train_loss /= len(train_dataset)

        info['train_loss'].append(train_loss)

        # ================================            
        model.eval()
        likelihood.eval()
        
        test_loss = 0.0
        with torch.no_grad():
            # for i, data in enumerate(test_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
            inputs, labels = test_dataset[:,:2], test_dataset[:,2]
                
            inputs[:,0] /= 244
            inputs[:,1] /= (3.14*2)

            # Output from model
            output = likelihood(model(inputs))
            # Calc loss and backprop gradients
            loss = -mll(output, labels).sum()

            # get statistics
            test_loss += loss.item()

        # normalize
        test_loss /= len(test_dataset)
        # test_accuracy /= len(test_dataset)

        # info['test_acc'].append(test_accuracy)
        info['test_loss'].append(test_loss)
        # ================================            
        
        if epoch % print_freq == 0:

            print(f'Epoch: {epoch}')
            # print(f'Train loss: {train_loss}')
            print(f'Train loss: {train_loss} | Test loss: {test_loss}')
            # print(f'Train acc: {train_accuracy} | Test acc: {test_accuracy}')

    return model, info

    
# train classifier
# class ClassifierNN(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, activation=None):
#         super(ClassifierNN, self).__init__()
#         self.input_size = input_size
#         self.hidden_size  = hidden_size
#         self.output_size = output_size
#         self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
#         self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
#         self.activation = activation
#     def forward(self, x):
#         hidden = self.fc1(x)
#         relu = F.relu(hidden)
#         if self.activation is None:
#             output = self.fc2(relu)
#         elif self.activation == 'tanh':
#             output = F.tanh( self.fc2(relu) )
#         elif self.activation == 'leaky_relu':
#             output = F.leaky_relu( self.fc2(relu) )
#         return output
    
class ClassifierNN(torch.nn.Module):
    def __init__(self, mlps, activation=None):
        super(ClassifierNN, self).__init__()
        self.total_mlps = len(mlps)
        self.mods = nn.ModuleList()
        for i in range(self.total_mlps-1):
            self.mods.append(nn.Linear(mlps[i], mlps[i+1]))
    def forward(self, x):
        for i in range(self.total_mlps-1):
            x = self.mods[i](x)
            if i != (self.total_mlps-2):
                x = F.relu(x)    
        return x


from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel

class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='RBF'):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel == 'RQ':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# class BatchedGPModel(ExactGP):
#     """Class for creating batched Gaussian Process Regression models.  Ideal candidate if
#     using GPU-based acceleration such as CUDA for training.
#     Parameters:
#         train_x (torch.tensor): The training features used for Gaussian Process
#             Regression.  These features will take shape (B * YD, N, XD), where:
#                 (i) B is the batch dimension - minibatch size
#                 (ii) N is the number of data points per GPR - the neighbors considered
#                 (iii) XD is the dimension of the features (d_state + d_action)
#                 (iv) YD is the dimension of the labels (d_reward + d_state)
#             The features of train_x are tiled YD times along the first dimension.
#         train_y (torch.tensor): The training labels used for Gaussian Process
#             Regression.  These features will take shape (B * YD, N), where:
#                 (i) B is the batch dimension - minibatch size
#                 (ii) N is the number of data points per GPR - the neighbors considered
#                 (iii) YD is the dimension of the labels (d_reward + d_state)
#             The features of train_y are stacked.
#         likelihood (gpytorch.likelihoods.GaussianLikelihood): A likelihood object
#             used for training and predicting samples with the BatchedGP model.
#         shape (int):  The batch shape used for creating this BatchedGP model.
#             This corresponds to the number of samples we wish to interpolate.
#         output_device (str):  The device on which the GPR will be trained on.
#         use_ard (bool):  Whether to use Automatic Relevance Determination (ARD)
#             for the lengthscale parameter, i.e. a weighting for each input dimension.
#             Defaults to False.
#     """
#     def __init__(self, train_x, train_y, likelihood, shape, output_device, use_ard=False):

#         # Run constructor of superclass
#         super(BatchedGPModel, self).__init__(train_x, train_y, likelihood)

#         # Determine if using ARD
#         ard_num_dims = None
#         if use_ard:
#             ard_num_dims = train_x.shape[-1]

#         # Create the mean and covariance modules
#         self.shape = torch.Size([shape])
#         self.mean_module = ConstantMean(batch_shape=self.shape)
#         self.base_kernel = RBFKernel(batch_shape=self.shape,
#                                         ard_num_dims=ard_num_dims)
#         self.covar_module = ScaleKernel(self.base_kernel,
#                                         batch_shape=self.shape,
#                                         output_device=output_device)

#     def forward(self, x):
#         """Forward pass method for making predictions through the model.  The
#         mean and covariance are each computed to produce a MV distribution.
#         Parameters:
#             x (torch.tensor): The tensor for which we predict a mean and
#                 covariance used the BatchedGP model.
#         Returns:
#             mv_normal (gpytorch.distributions.MultivariateNormal): A Multivariate
#                 Normal distribution with parameters for mean and covariance computed
#                 at x.
#         """
#         mean_x = self.mean_module(x)  # Compute the mean at x
#         covar_x = self.covar_module(x)  # Compute the covariance at x
#         return MultivariateNormal(mean_x, covar_x)

