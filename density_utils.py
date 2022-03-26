from tqdm.auto import tqdm

import torch
import torch.optim as optim
import torch.utils.data as tdata
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import TensorDataset, DataLoader
import gpytorch

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
def refine_sample(x, D, steps=10, f='KL',
                     eta=0.001, noise_factor=0.0001, decay_type='constant', Nq=1, Np=1):
    
    def _velocity(x, Nq=1, Np=1):
        x_t = x.clone()
        x_t.requires_grad_(True)
        if x_t.grad is not None:
            x_t.grad.zero_()
            
        
        d_score = D(x_t)
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
        return s.data * grad.data
    
    if decay_type == 'linear':
        decay_func = linear
    elif decay_type == 'exp':
        decay_func = exp
    elif decay_type == 'constant':
        decay_func = constant
    else:
        raise ValueError()
    
    all_x = [x.detach().cpu()]
    all_v = []
    for t in tqdm(range(1, steps + 1), leave=False):
        
        decay_coeff = decay_func(t, steps)
        decay_coeff = decay_coeff.to(x.device)
        v = decay_coeff*_velocity(x, Nq=Nq, Np=Np)
        all_v.append(v.detach().cpu())
        x = x.data + eta * v +\
            np.sqrt(2*eta) * noise_factor * torch.randn_like(x)
        all_x.append(x.detach().cpu())
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
def refine_sample2(x, D1, D2, steps=10, f='KL',
                     eta=0.001, noise_factor=0.0001,
                     Np1=2000,
                     Np2=2000,
                     Nq1=2000,
                     Nq2=2000):
    
    def _velocity(x):
        x_t = x.clone()
        x_t.requires_grad_(True)
        if x_t.grad is not None:
            x_t.grad.zero_()
            
        
        logit1 = D1(x_t)
        logit2 = D2(x_t)
        
        bias_term = torch.log( torch.Tensor( [Nq1*Nq2]).to(x_t.device)) - torch.log( torch.Tensor( [Np1 * Np2] ).to(x_t.device)) 
        log_r = torch.ones_like(logit1).to(x_t.device)*bias_term
        log_r += - logit1 - logit2 + torch.log( 1 + Nq1/Np1*logit1.exp() + Nq2/Np2*logit2.exp() )

        if f == 'KL':
            s = torch.ones_like(log_r.detach())
            
        elif f == 'logD':
            s = 1 / (1 + log_r.detach().exp())

        elif f == 'JS':
            s = 1 / (1 + 1 / log_r.detach().exp())

        else:
            raise ValueError()

        s.expand_as(x_t)
        log_r.backward(torch.ones_like(log_r).to(x_t.device))
        grad = x_t.grad
        return s.data * grad.data
    
    all_x = [x.detach().cpu()]
    all_v = []
    for t in tqdm(range(1, steps + 1), leave=False):
        v = _velocity(x)
        all_v.append(v.detach().cpu())
        x = x.data - eta * v +\
            np.sqrt(2*eta) * noise_factor * torch.randn_like(x)
        all_x.append(x.detach().cpu())
    return all_x, all_v


def draw_density_ratio_GP(ax, model, likelihood,
                                        x_lim=[-10,10],
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
    likelihood.eval()
    with torch.no_grad():
        zz = []
        for s in tqdm(dataloader):
            # ==================================================
            # print(f"s: {s[0]}")
            s = s.to(device)
            s[:,0] /= 244
            s[:,1] /= (3.14*2)
            # s = s.repeat(1,200)
            # print(f"s: {s}")
            # ==================================================
            # print("stuck at model")

            # logit = model(s).loc
            logit = likelihood(model(s)).loc
            # logit = likelihood(model(s)).loc[1]
            # logit = model(s)
            # print(f"logit: {torch.unique(logit)}")

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

def draw_density_ratio_GPC(ax, model, likelihood,
                                        x_lim=[-10,10],
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
    dataloader = DataLoader(torch.from_numpy(data).float(), batch_size=5, shuffle=False)
    # model = model.to(device)
    # likelihood = likelihood.to(device)
    model.eval()
    likelihood.eval()
    # with torch.no_grad():
    with gpytorch.settings.fast_pred_var(), torch.no_grad():

        zz = []
        for s in tqdm(dataloader):
            # ==================================================
            # print(f"s: {s[0]}")
            # s = s.to(device)
            s[:,0] /= 244
            s[:,1] /= (3.14*2)

            # s = s.repeat(1,200)
            # print(f"s: {s}")
            # print(f"model: {model}")
            # print(f"likelihood: {likelihood}")
            # ==================================================
            # logit = model(s).loc

            # _pred_means = model(s)
            # pred_means = _pred_means.loc
            # pred_means = likelihood(model(s)).loc
            pred_means = likelihood(model(s))
            # print(pred_means)
            # logit = pred_means[1,:] - pred_means[0,:]
            logit = pred_means.probs

            # logit = likelihood(model(s)).loc[1]
            # logit = model(s)
            # print(f"logit: {torch.unique(logit)}")

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





def refine_sample_GP(x, D, likelihood, steps=10, f='KL',
                     eta=0.001, noise_factor=0.0001, decay_type='constant', Nq=1, Np=1):
    D.eval()
    likelihood.eval()
    print(Nq, Np)

    def _velocity(x, Nq=1, Np=1):
        x_t = x.clone()
        x_t.requires_grad_(True)
        if x_t.grad is not None:
            x_t.grad.zero_()
            
        # getting logits for the class 1
        output = likelihood(D(x_t))


        d_score = output.loc.clone()
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

        s = s.view(-1, 1)
        s.expand_as(x_t)
        
        y = output.mean.sum()
        y.backward()
        grad = x_t.grad
        # print(grad)
        # print(f'grad: {grad.data}')
        return s.data * grad.data
    
    if decay_type == 'linear':
        decay_func = linear
    elif decay_type == 'exp':
        decay_func = exp
    elif decay_type == 'constant':
        decay_func = constant
    else:
        raise ValueError()
    
    all_x = [x.detach().cpu()]
    all_v = []
    for t in tqdm(range(1, steps + 1), leave=False):
        
        decay_coeff = decay_func(t, steps)
        decay_coeff = decay_coeff.to(x.device)
        # print(f"decay_coeff: {decay_coeff}")
        # print(f"_velocity: {_velocity(x, Nq=Nq, Np=Np)}")
        v = decay_coeff*_velocity(x, Nq=Nq, Np=Np)
        all_v.append(v.detach().cpu())
        # print(x.data)
        # print(f"eta * v: {eta * v}")
        # print(f"np.sqrt(2*eta): {np.sqrt(2*eta)}")
        # print(f"noise_factor: {noise_factor}")
        # print(f"torch.randn_like(x): {torch.randn_like(x)}")
        # print("===============")
        x = x.data + eta * v +\
            np.sqrt(2*eta) * noise_factor * torch.randn_like(x)
        all_x.append(x.detach().cpu())
    return all_x, all_v


def refine_sample_GPC(x, D, likelihood, steps=10, f='KL',
                     eta=0.001, noise_factor=0.0001, decay_type='constant', Nq=1, Np=1):
    D.eval()
    likelihood.eval()
    print(Nq, Np)

    def _velocity(x, Nq=1, Np=1):
        x_t = x.clone()
        x_t.requires_grad_(True)
        if x_t.grad is not None:
            x_t.grad.zero_()
            
        # getting logits for the class 1
        output = likelihood(D(x_t))


        pred_means = output.loc.clone()
        d_score = (pred_means[1,:] - pred_means[0,:]).squeeze()
        # print(d_score.size())

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

        s = s.view(-1, 1)
        s.expand_as(x_t)
        
        y = output.mean.sum()
        y.backward()
        grad = x_t.grad
        # print(grad)
        # print(f'grad: {grad.data}')
        return s.data * grad.data
    
    if decay_type == 'linear':
        decay_func = linear
    elif decay_type == 'exp':
        decay_func = exp
    elif decay_type == 'constant':
        decay_func = constant
    else:
        raise ValueError()
    
    all_x = [x.detach().cpu()]
    all_v = []
    for t in tqdm(range(1, steps + 1), leave=False):
        
        decay_coeff = decay_func(t, steps)
        decay_coeff = decay_coeff.to(x.device)
        # print(f"decay_coeff: {decay_coeff}")
        # print(f"_velocity: {_velocity(x, Nq=Nq, Np=Np)}")
        v = decay_coeff*_velocity(x, Nq=Nq, Np=Np)
        all_v.append(v.detach().cpu())
        # print(x.data)
        # print(f"eta * v: {eta * v}")
        # print(f"np.sqrt(2*eta): {np.sqrt(2*eta)}")
        # print(f"noise_factor: {noise_factor}")
        # print(f"torch.randn_like(x): {torch.randn_like(x)}")
        # print("===============")
        x = x.data + eta * v +\
            np.sqrt(2*eta) * noise_factor * torch.randn_like(x)
        all_x.append(x.detach().cpu())
    return all_x, all_v


def refine_sample2_GP(x, D1, D2, likelihood, steps=10, f='KL',
                     eta=0.001, noise_factor=0.0001,
                     Np1=2000,
                     Np2=2000,
                     Nq1=2000,
                     Nq2=2000):
    D1.eval()
    D2.eval()

    def _velocity(x):
        x_t = x.clone()
        x_t.requires_grad_(True)
        if x_t.grad is not None:
            x_t.grad.zero_()
            
        
        logit1 = D1(x_t)
        logit2 = likelihood(D2(x_t)).loc.reshape(-1,1)
        
        # print(logit1.size())
        # print(logit2.size())
        
        bias_term = torch.log( torch.Tensor( [Nq1*Nq2]).to(x_t.device)) - torch.log( torch.Tensor( [Np1 * Np2] ).to(x_t.device)) 
        log_r = torch.ones_like(logit1).to(x_t.device)*bias_term
        log_r += - logit1 - logit2 + torch.log( 1 + Nq1/Np1*logit1.exp() + Nq2/Np2*logit2.exp() )

        if f == 'KL':
            s = torch.ones_like(log_r.detach())
            
        # elif f == 'logD':
        #     s = 1 / (1 + log_r.detach().exp())

        # elif f == 'JS':
        #     s = 1 / (1 + 1 / log_r.detach().exp())

        else:
            raise ValueError()

        s.expand_as(x_t)
        log_r.backward(torch.ones_like(log_r).to(x_t.device))
        grad = x_t.grad
        return s.data * grad.data
    
    all_x = [x.detach().cpu()]
    all_v = []
    for t in tqdm(range(1, steps + 1), leave=False):
        v = _velocity(x)
        all_v.append(v.detach().cpu())
        x = x.data - eta * v +\
            np.sqrt(2*eta) * noise_factor * torch.randn_like(x)
        all_x.append(x.detach().cpu())
    return all_x, all_v


