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

    for epoch in range(1, n_epochs+1):  # loop over the dataset multiple times

        model.train()
        likelihood.train()

        train_loss = 0.0
        # train_accuracy = 0.0
        # for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
        inputs, labels = train_dataset[:,:2], train_dataset[:,2]
        inputs[:,0] /= 244
        inputs[:,1] /= (3.14*2)

        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = likelihood(model(inputs))
        # Calc loss and backprop gradients
        loss = -mll(output, labels).sum()
        loss.backward()
        optimizer.step()

        # get statistics
        # train_loss += loss.item()
        train_loss = loss.item()
        # predicted = torch.sigmoid(outputs)
        # train_accuracy += (torch.round(predicted) == labels).sum().item()

        # normalize
        train_loss /= len(train_dataset)
        # train_accuracy /= len(train_dataset)

        # info['train_acc'].append(train_accuracy)
        info['train_loss'].append(train_loss)

        # ================================            
        # model.eval()
        # likelihood.eval()
        
        # test_loss = 0.0
        # # test_accuracy = 0.0
        # with torch.no_grad():
        #     for i, data in enumerate(test_dataloader, 0):
        #         # get the inputs; data is a list of [inputs, labels]
        #         inputs, labels = data
                
        #         inputs[:,0] /= 244
        #         inputs[:,1] /= (3.14*2)

        #         # forward + backward + optimize
        #         outputs = model(inputs).squeeze()
        #         loss = criterion(outputs, labels)

        #         # get statistics
        #         test_loss += loss.item()
        #         predicted = torch.sigmoid(outputs)
        #         test_accuracy += (torch.round(predicted) == labels).sum().item()

        # # normalize
        # test_loss /= len(test_dataset)
        # # test_accuracy /= len(test_dataset)

        # # info['test_acc'].append(test_accuracy)
        # info['test_loss'].append(test_loss)
        # ================================            
        
        if epoch % print_freq == 0:

            print(f'Epoch: {epoch}')
            print(f'Train loss: {train_loss}')
            # print(f'Train loss: {train_loss} | Test losses: {test_loss}')
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
