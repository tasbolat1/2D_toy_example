from torch.utils.data import Dataset
import os
import numpy as np
import torch

def is_handle(x):
    if (x>65) and (x<135):
        return True
    return False        

def is_pos(x, angle):
    angle *= (180/np.pi)
    # check x
    x_pos = False
    if (x>=65) and (x<=135):
        x_pos = True
    if (x>=155) and (x<=180):
        x_pos = True
        
    # check alpha
    alpha_pos = False
    if (angle>=70) and (angle<=120):
        alpha_pos = True
    if (angle>=250) and (angle<=290):
        alpha_pos = True
        
    if x_pos and alpha_pos:
        return True
    
    return False   

def corr_is_pos(x, angle):
    angle *= (180/np.pi)
    # check x
    x_pos = False
    if (x>=80) and (x<=120):
        x_pos = True
    if (x>=160) and (x<=170):
        x_pos = True
        
    # check alpha
    alpha_pos = False
    if (angle>=80) and (angle<=100):
        alpha_pos = True
    if (angle>=260) and (angle<=280):
        alpha_pos = True
        
    if x_pos and alpha_pos:
        return True
    
    return False          

class ToyGraspDataset(Dataset):
    def __init__(self, root, name='data', size=1000, is_pos_label=True, device='cpu', dataset="sim2sim"):
        path = os.path.join(root, f'{name}.npy')
        #if not os.path.exists(path):
        if dataset=='sim2sim':
            self._build_dataset(path, size)
        elif dataset=='sim2real':
            self._build_corrector_dataset(path, size)
        self.data = np.load(path)
        self.transform = torch.from_numpy
        self.is_pos_label = is_pos_label
        self.device = device

    def _build_dataset(self, path, size):
        x = np.random.uniform(low=0, high=224, size=size)
        alpha = np.random.uniform(low=0, high=2*np.pi, size=size)
        data = []
        for i in range(size):
            pos_label = 0
            handle_label = 0
            if is_pos(x[i], alpha[i]):
                pos_label = 1
            if is_handle(x[i]):
                handle_label = 1
            data.append([x[i], alpha[i], pos_label, handle_label])

        dataset = np.array(data, dtype='float32')
        np.random.shuffle(dataset)
        np.save(path, dataset)


    def _build_corrector_dataset(self, path, size):
        x1 = np.random.uniform(low=65, high=135, size=int(size/2))
        x2 = np.random.uniform(low=155, high=180, size=int(size/2))
        x = np.hstack([x1,x2])
        np.random.shuffle(x)
        
        a1 = np.random.uniform(low=np.radians(70), high=np.radians(120), size=int(size/2))
        a2 = np.random.uniform(low=np.radians(250), high=np.radians(290), size=int(size/2))
        alpha = np.hstack([a1,a2])
        np.random.shuffle(alpha)
        
        
        data = []
        for i in range(size):
            pos_label = 0
            handle_label = 0
            sim_label = 0
            if corr_is_pos(x[i], alpha[i]):
                pos_label = 1
            if is_handle(x[i]):
                handle_label = 1
                
            data.append([x[i], alpha[i], pos_label, handle_label])

        dataset = np.array(data, dtype='float32')
        np.random.shuffle(dataset)
        np.save(path, dataset)    
    
    def __getitem__(self, i):
        x = self.transform(self.data[i]).to(self.device)
        if self.is_pos_label:
            return x[:2], x[2]
        else:
            return x[:2], x[3]

    def __len__(self):
        return len(self.data)