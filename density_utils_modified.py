from warnings import simplefilter
from tqdm.auto import tqdm

import torch
import torch.optim as optim
import torch.utils.data as tdata
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import TensorDataset, DataLoader


# visualize the decision boundaries
def draw_density_ratio(ax, model, x_lim=[-10,10],
                                        y_lim=[-10,10],
                                        step_size=0.2,
                                        cmap=None,
                                        alpha=0.3,
                                        device='cpu',
                                        level=10,
                                        p_type='log_density_ratio', Np=1, Nq=1):
    if cmap is None:
        cmap = plt.cm.Paired
    
    xx, yy = np.meshgrid(np.arange(x_lim[0], x_lim[1]+step_size, step_size),
                     np.arange(y_lim[0], y_lim[1]+step_size, step_size))
    
    data = np.c_[xx.ravel(), yy.ravel()]
    dataloader = DataLoader(torch.from_numpy(data).float(), batch_size=15000, shuffle=False)
    model.eval()
    with torch.no_grad():
        zz = []
        for s in dataloader:
            s = s.to(device)
            s[:,0] /= 244
            s[:,1] /= (3.14*2)
            logit = model(s)
            
            if p_type == 'log_density_ratio':
                output = -logit*(Nq/Np)
            elif p_type == 'p_y_given_x':
                output = torch.sigmoid(logit)
            else:
                output = (1-torch.sigmoid(-logit))/torch.sigmoid(-logit)                
                    
            zz.append(output.cpu())
        
    zz=torch.cat(zz)
    Z=zz.detach().cpu().numpy().reshape(xx.shape)
    print(Z.min(), Z.max())
    return Z, ax.contourf(xx, yy, Z, cmap=cmap, alpha=alpha, level=level)


def linear(t, steps):
    return 1 - torch.FloatTensor([t/steps])
def exp(t, steps):
    return torch.exp(-torch.FloatTensor([t/steps]))
def constant(t, steps):
    return torch.FloatTensor([1.0])


# refine ONLY KL version
def refine_sample(x, D, f='KL', success_threshold=0.8,
                     eta=0.001, noise_factor=0.0001, Nq=1, Np=1, max_iterations=1000):
    
    def _velocity(x, Nq=1, Np=1, success_threshold=0.8):
        x_t = x.clone()
        x_t.requires_grad_(True)
        if x_t.grad is not None:
            x_t.grad.zero_()
            
        d_score = D(x_t)
        
        success = torch.sigmoid(d_score)
        mask_success = (success > success_threshold)
        d_score[mask_success] = 0.0

        Nq = torch.FloatTensor([Nq]).to(x_t.device)
        Np = torch.FloatTensor([Np]).to(x_t.device)
        bias_term = torch.log(Nq) - torch.log(Np)
        d_score -= bias_term

        if f == 'KL':
            s = torch.ones_like(d_score.detach())

        elif f == 'logD':
            s = 1 / (1 + d_score.detach().exp())

        elif f == 'JS':
            s = 1 / (1 + 1 / d_score.detach().exp())

        else:
            raise ValueError()

        s.expand_as(x_t)
        d_score.backward(torch.ones_like(d_score).to(x_t.device))
        grad = x_t.grad
        return s.data * grad.data, torch.sigmoid(d_score)

    
    all_x = [x.detach().cpu()]
    all_v = []
    for t in tqdm(range(1, max_iterations + 1), leave=False):
        v, success = _velocity(x, Nq=Nq, Np=Np, success_threshold=success_threshold)
        all_v.append(v.detach().cpu())
        x = x.data + eta * v +\
            np.sqrt(2*eta) * noise_factor * torch.randn_like(x)
        all_x.append(x.detach().cpu())
        mask_success = (success > success_threshold)
        if torch.sum(mask_success).detach().cpu().item() == x.shape[0]:
            print(f'All samples converged within {t} iterations.')
            break

    return all_x, all_v


# visualize the decision boundaries
def draw_density_ratio2(ax, model1,model2, x_lim=[-10,10],
                                        y_lim=[-10,10],
                                        step_size=0.2,
                                        cmap=None,
                                        level=10,
                                        alpha=0.3,
                                        device='cpu',
                                        Np1=2000,
                                        Np2=2000,
                                        Nq1=2000,
                                        Nq2=2000):
    if cmap is None:
        cmap = plt.cm.Paired
    
    xx, yy = np.meshgrid(np.arange(x_lim[0], x_lim[1]+step_size, step_size),
                     np.arange(y_lim[0], y_lim[1]+step_size, step_size))
    
    data = np.c_[xx.ravel(), yy.ravel()]
    dataloader = DataLoader(torch.from_numpy(data).float(), batch_size=15000, shuffle=False)
    model1.eval()
    model2.eval()
    with torch.no_grad():
        zz = []
        for x in dataloader:
            x = x.to(device)
            logit1 = model1(x)
            logit2 = model2(x)
            
            log_r = torch.ones_like(logit1).to(device)*torch.log( torch.Tensor( [(Nq1*Nq2)/(Np1 * Np2)] ).to(device) ) 
            log_r += - logit1 - logit2 + torch.log( 1 + Nq1/Np1*logit1.exp() + Nq2/Np2*logit2.exp() )
            zz.append(log_r.cpu())
        
    zz=torch.cat(zz)
    Z=zz.detach().cpu().numpy().reshape(xx.shape)
    print(Z.min(), Z.max())
    return Z, ax.contourf(xx, yy, Z, cmap=cmap, alpha=alpha, level=level)


# refine ONLY KL version
def refine_sample2(x, D1, D2, f='KL', success_threshold=0.8,
                     eta=0.001, noise_factor=0.0001,
                     Np1=2000,
                     Np2=2000,
                     Nq1=2000,
                     Nq2=2000,
                     max_iterations=1000):
    
    def _velocity(x, success_threshold=0.8):
        x_t = x.clone()
        x_t.requires_grad_(True)
        if x_t.grad is not None:
            x_t.grad.zero_()
            
        
        logit1 = D1(x_t)
        success1 = torch.sigmoid(logit1)
        #mask_success1 = (success1 > success_threshold)
        #logit1[mask_success1] = 0.0


        logit2 = D2(x_t)
        success2 = torch.sigmoid(logit2)
        #mask_success2 = (success2 > success_threshold)
        #logit2[mask_success2] = 0.0

        success = success1*success2
        #mean_success = torch.cat([success1.unsqueeze(-1), success2.unsqueeze(-1)], dim=-1).squeeze(-1)*0.5
        mask_success = (success > success_threshold)
        #print(mask_success)

        stop_criteria = 0
        if torch.sum(mask_success) == x_t.shape[0]:
            stop_criteria = 1

        bias_term = torch.log( torch.Tensor( [Nq1*Nq2]).to(x_t.device)) - torch.log( torch.Tensor( [Np1 * Np2] ).to(x_t.device)) 
        log_r = torch.ones_like(logit1).to(x_t.device)*bias_term
        log_r += - logit1 - logit2 + torch.log( 1 + Nq1/Np1*logit1.exp() + Nq2/Np2*logit2.exp() )

        if f == 'KL':
            s = torch.ones_like(log_r.detach())
        else:
            raise ValueError()

        s.expand_as(x_t)
        log_r.backward(torch.ones_like(log_r).to(x_t.device))
        grad = x_t.grad
        return s.data * grad.data, stop_criteria
    
    all_x = [x.detach().cpu()]
    all_v = []
    for t in tqdm(range(1, max_iterations + 1), leave=False):
        v, stop_criteria = _velocity(x, success_threshold=success_threshold)
        all_v.append(v.detach().cpu())
        x = x.data - eta * v +\
            np.sqrt(2*eta) * noise_factor * torch.randn_like(x)
        all_x.append(x.detach().cpu())
        if stop_criteria:
            print(f'All samples converged within {t} iterations.')
            break
    return all_x, all_v