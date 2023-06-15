import numpy as np
import scipy
from scipy import fft
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import cg
from scipy.interpolate import interp2d
import time
from datetime import datetime
from timeit import default_timer
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import operator
from collections import OrderedDict
from functools import reduce
import random
import cv2
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
from  matplotlib  import cm
import imageio
from PIL import Image
import sys


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
X_list = []
iters = 0
# heatmap_params = {
#     'cmap': 'gist_rainbow',
#     'xticklabels': False,
#     'yticklabels': False
# }
heatmap_params = {
    'cmap': cm.seismic,
    'xticklabels': False,
    'yticklabels': False
}
heatmap_params = {
    'cmap': 'hot',
    'xticklabels': False,
    'yticklabels': False
}
heatmap_params = {
    'cmap': 'viridis',
    'xticklabels': False,
    'yticklabels': False
}
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul,
                    list(p.size() + (2,) if p.is_complex() else p.size()))
    return c

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=False):
        super(LpLoss, self).__init__()

        self.size_average = size_average
        self.eps = 1e-8

    def abs(self, x, y):
        num_examples = x.size()[0]
        all_norms = torch.sqrt(torch.sum((x-y)**2, dim=(1, 2, 3)))
        if self.size_average:
            return torch.mean(all_norms)
        else:
            return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.sqrt(torch.sum((x-y)**2, dim=(1, 2, 3)))
        y_norms = torch.sqrt(torch.sum((y)**2, dim=(1, 2, 3)))
        if self.size_average:
            return torch.mean(diff_norms / (y_norms+self.eps))
        else:
            return torch.sum(diff_norms / (y_norms+self.eps))
    
    def fft_loss(self,x,y):
        diff = torch.fft.fft2(x) - torch.fft.fft2(y)
        loss = torch.mean(abs(diff))
        return self.abs(torch.fft.fft2(x),torch.fft.fft2(y))
    
    def pde_loss(self,x,y,k,end=2,padding=False):
        s = x.shape[-1]
        x1 = -(forward_conv(s-1,k,y,padding)/(k*k))
        if padding:
            return self.abs(x1[:,:end],x[:,:end])
        else:
            return self.abs(x1[:,:end],x[:,:end,1:-1,1:-1])
    
    def pde_loss_rel(self,x,y,k,end=2,padding=False):
        s = x.shape[-1]
        x1 = -(forward_conv(s-1,k,y,padding)/(k*k))
        if padding:
            return self.rel(x1[:,:end],x[:,:end])
        else:
            return self.rel(x1[:,:end],x[:,:end,1:-1,1:-1])
        
    def __call__(self, x, y):
        return self.rel(x, y)


def expand_grids(q,expand_times=1):
    # N \times N --> N*expand_times \times N*expand_times
    N = q.shape[0]
    return np.pad(q,N//2*(expand_times-1))

def squeeze_grids(q, expand_times=1,flatten=True):
    # N \times N --> N/expand_times \times N/expand_times 
    N = q.shape[0]
    if len(q.shape) == 1:
        N = int(np.sqrt(N))
        q = q.reshape((N,N))
    if expand_times > 1:
        idx = ((N-1)//expand_times)//2*(expand_times-1)
        out = q[idx:-idx,idx:-idx]
    else:
        out = q
    if flatten:
        return out.reshape(-1,)
    else:
        return out

def Type_Settle(x, Type, device = 'cpu'):
    if Type == 'np':
        if isinstance(x, np.ndarray):
            output = x
        else:
            if x.is_cuda:
                output = x.cpu().detach().numpy()
            else:
                output = x.detach().numpy()
    elif Type == 'torch':
        if isinstance(x, torch.Tensor):
            output = x
        else:
            output = torch.from_numpy(x).to(torch.float32).to(device)
    return output


def Error(a, a_truth):
    """
    Relative-Error
    """
    tmp = np.linalg.norm((a - a_truth), ord=2)
    return tmp / np.linalg.norm(a_truth, ord=2)


def Round(vector, times, method = 1):
    assert method == 1 or method ==2
    if times == 0.0:
        return vector
    else:
        SHAPE = vector.shape
        vector1 = vector.reshape(-1,)
        SHAPE1 = vector1.shape[0]
        if method == 1:
            ERR = np.array([random.uniform(-times, times)
                            for i in range(SHAPE1)])
            return vector + ERR.reshape(SHAPE)* vector
        else:
            R = 0.5*(np.max(vector1)-np.min(vector1))
            ERR = np.array([random.uniform(-R*times, R*times)
                            for i in range(SHAPE1)])
            return vector + ERR.reshape(SHAPE)


def load_data(filename, device = 'cpu',angle_id = -1,
              NS_return = 'T', Usage = 'Train', output_size = 64):
    Dataset_dir = '/data/liuziyang/Programs/pde_solver/Dataset/'
    data = np.load(Dataset_dir+filename+'.npz', allow_pickle=True)    
    q = Type_Settle(data['q'], 'torch', device) # (nsample, 1, 65, 65)
    times = q.shape[-1] // output_size
    q = q[...,::times,::times]
    wave = Type_Settle(data['WAVE'][...,::times,::times], 'torch', device) 
    # (angle_for_test, nsample, 1/2, 65, 65)
    u_i = Type_Settle(data['u_i'][...,::times,::times], 'torch', device) 
    # (angle_for_test, nsample, 2, 65, 65)
    u_t = Type_Settle(data['u_t'][...,::times,::times], 'torch', device) 
    # (angle_for_test, nsample, 2, 65, 65)
    u_NS = None 
    if NS_return == 'T':
        u_NS = Type_Settle(data['u_NS'][...,::times,::times], 'torch', device) 
        # (angle_for_test, nsample, NS_length, 2, 65, 65)
        
    if angle_id >= 0:
        q = q[angle_id:(angle_id+1)]
        wave = wave[angle_id:(angle_id+1)]
        u_i = u_i[angle_id:(angle_id+1)]
        u_t = u_t[angle_id:(angle_id+1)]
        if NS_return == 'T':
            u_NS = u_NS[angle_id:(angle_id+1)]
    if NS_return == 'T':
        if Usage == 'Train':
            q = q.unsqueeze(1).repeat((1,wave.shape[0],1,1,1))
            wave = wave.permute(1,0,2,3,4)
            u_i = u_i.permute(1,0,2,3,4)
            u_NS = u_NS.permute(1,0,2,3,4,5)
            return q, wave, u_i, u_NS[:,:,0]
        elif Usage == 'Train1':
            q = q.unsqueeze(1).repeat((1,wave.shape[0],1,1,1))
            wave = wave.permute(1,0,2,3,4)
            u_t = u_t.permute(1,0,2,3,4)
            return q, wave, u_t
        elif Usage == 'Test':
            return q, wave, u_i, u_NS, u_t
    else:
        return q, wave, u_i
    
    
def heatmap_for_test(x,y,label_list,label='',loss = True):
    # Input 2d np.array/torch.Tensor
    data1 = Type_Settle(x, 'np')
    data2 = Type_Settle(y, 'np')
    if loss:
        data3 = data1-data2
        rel_err = 100*(np.linalg.norm(data3)/np.linalg.norm(data2))
        print('{}相对误差为{:.2f}%'.format(label,rel_err))

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        sns.heatmap(data1,  ax=axs[0], **heatmap_params)
        sns.heatmap(data2,  ax=axs[1], **heatmap_params)
        sns.heatmap(data3,  ax=axs[2], **heatmap_params)
        axs[0].set_title(label_list[0])
        axs[1].set_title(label_list[1])
        axs[2].set_title(label_list[2])
        fig.tight_layout()
        plt.show()
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(data1,  ax=axs[0], **heatmap_params)
        sns.heatmap(data2,  ax=axs[1], **heatmap_params)
        axs[0].set_title(label_list[0])
        axs[1].set_title(label_list[1])
        fig.tight_layout()
        plt.show()
    
    
def SEED_SET(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    
    
def callbackF(X):
    global X_list
    global iters
    X_list.append(X)
    iters += 1
    if iters >= 10 and iters < 99:
        print('iter {} completed'.format(iters),
          '       %s' % str(datetime.now())[:-7])
    else:
        print('iter {}  completed'.format(iters),
              '       %s' % str(datetime.now())[:-7])


def plot_heatmap(q_list,title,picdir,gifdir,
                 subtitle_list,percent_list,pic_list):
    n = len(q_list)
    max_value = max([abs(qq).max() for qq in q_list])
    max_value = eval('%.2f' % max_value)
    img_list,img_list_tmp = [],[]
    for i in range(n):
        if i < n - 1:
            plt.figure(figsize=(4, 4))
            plt.title(str(subtitle_list[i]) + '    ' + percent_list[i])
            q = q_list[i]
            h = sns.heatmap(q,vmin=-max_value, vmax=max_value, cbar=False, **heatmap_params)
            tmp_dir = picdir + title + '___' + str(i) + '.jpg'
            plt.savefig(tmp_dir)
            plt.close()
            img_list.append(cv2.imread(tmp_dir))
            img_list_tmp.append(cv2.imread(tmp_dir)[:, :, ::-1])
            os.remove(tmp_dir)
        else:
            plt.figure(figsize=(5, 4))
            plt.title(str(subtitle_list[i]) + '    ' + percent_list[i])
            q = q_list[i]
            h = sns.heatmap(q, vmin=-max_value, vmax=max_value, cbar=True, **heatmap_params)
            tmp_dir = picdir + title + '___' + str(i) + '.jpg'
            plt.savefig(tmp_dir)
            plt.close()
            img_list.append(cv2.imread(tmp_dir))
            os.remove(tmp_dir)
            plt.figure(figsize=(4, 4))
            plt.title(str(subtitle_list[i]) + '    ' + percent_list[i])
            q = q_list[i]
            h = sns.heatmap(q, vmin=-max_value, vmax=max_value, cbar=False, **heatmap_params)
            tmp_dir = picdir + title + '__' + str(i) + '.jpg'
            plt.savefig(tmp_dir)
            plt.close()
            img_list_tmp.append(cv2.imread(tmp_dir)[:, :, ::-1])
            os.remove(tmp_dir)
    imageio.mimsave(gifdir +title+'.gif', img_list_tmp, duration=0.5, loop=0)
    img_list1 = []
    for item in pic_list:
            img_list1.append(img_list[item])
    nn = len(img_list1)
    img1 = img_list1[0]
    for i in range(1, nn):
        img1 = np.hstack((img1, img_list1[i]))
    cv2.imwrite(picdir + title + '.jpg', img1)
    
    
def INTERPOLATE(x, in_size, out_size):
    if isinstance(x, np.ndarray):
        tmp_shape = int(np.prod(x.shape)/(in_size+1)**2)
        x1 = x.reshape((tmp_shape, in_size + 1, in_size + 1))
        l_in = np.linspace(0, 1, in_size + 1)
        l_out = np.linspace(0, 1, out_size + 1)
        if np.isrealobj(x1):
            output = np.zeros((tmp_shape, out_size + 1, out_size + 1))
            for i in range(tmp_shape):
                output[i] = interp2d(l_in, l_in, x1[i], kind='cubic')(l_out, l_out)
        else:
            output = np.zeros((tmp_shape, out_size + 1, out_size + 1), dtype = np.complex128)
            for i in range(tmp_shape):
                output_real = interp2d(l_in, l_in, x1[i].real, kind='cubic')(l_out, l_out)
                output_imag = interp2d(l_in, l_in, x1[i].imag, kind='cubic')(l_out, l_out)
                output[i] = output_real+1j*output_imag
        return output
    if isinstance(x, torch.Tensor):
        assert x.shape[-1] == in_size + 1
        x_out = F.interpolate(x, size=(out_size + 1, out_size + 1), mode='bicubic', align_corners=True)
        return x_out


def data_projection(x,N,N_rec,data_to_boundary = True, bd_num = 4, bd_type = 'D'):
    # adjacent sides
    # x.shape = (N+1)^2 / 4*(N_rec-2*N_buffer_rec-1)
    assert bd_num in [1,2,3,4]
    assert bd_type in ['D','N']
    times = N // N_rec
    N_buffer = int(N*14/128)
    N_buffer_rec = int(N_rec*14/128)
    index = [i*times for i in range(N_buffer_rec+1,N_rec- N_buffer_rec)]
    if bd_type == 'D':
        if data_to_boundary:
            x1 = x[...,:,N_buffer][...,index]
            x2 = x[...,N_buffer,:][...,index]
            x3 = x[...,:,-N_buffer-1][...,index]
            x4 = x[...,-N_buffer-1,:][...,index]
            x_bd = [x1,x2,x3,x4]
            if isinstance(x, np.ndarray):
                return np.concatenate(x_bd[0:bd_num])
            elif isinstance(x, torch.Tensor):
                return torch.cat(x_bd[0:bd_num],-1)
        else:
            N_int = N_rec - 2 * N_buffer_rec
            if isinstance(x, np.ndarray):
                output = np.zeros((N+1,N+1),dtype = np.complex128)
            elif isinstance(x, torch.Tensor):
                output = torch.zeros((x.shape[0],x.shape[1],N+1,N+1)).to(x.device)
            output[...,:,N_buffer][...,index] = x[...,0*(N_int-1):1*(N_int-1)]
            if bd_num >= 2:
                output[...,N_buffer,:][...,index] = x[...,1*(N_int-1):2*(N_int-1)]
            if bd_num >= 3:
                output[...,:,-N_buffer-1][...,index] = x[...,2*(N_int-1):3*(N_int-1)]
            if bd_num >= 4:
                output[...,-N_buffer-1,:][...,index] = x[...,3*(N_int-1):4*(N_int-1)]
            return output
    else:
        if data_to_boundary:
            x1 = (x[...,:,N_buffer-1][...,index]-x[...,:,N_buffer+1][...,index])*N/2
            x2 = (x[...,N_buffer-1,:][...,index]-x[...,N_buffer+1,:][...,index])*N/2
            x3 = (x[...,:,-N_buffer][...,index]-x[...,:,-N_buffer-2][...,index])*N/2
            x4 = (x[...,-N_buffer,:][...,index]-x[...,-N_buffer-2,:][...,index])*N/2
            x_bd = [x1,x2,x3,x4]
            if isinstance(x, np.ndarray):
                return np.concatenate(x_bd[0:bd_num])
            elif isinstance(x, torch.Tensor):
                return torch.cat(x_bd[0:bd_num],-1)
        else:
            N_int = N_rec - 2 * N_buffer_rec
            if isinstance(x, np.ndarray):
                output = np.zeros((N+1,N+1),dtype = np.complex128)
            elif isinstance(x, torch.Tensor):
                output = torch.zeros((x.shape[0],x.shape[1],N+1,N+1)).to(x.device)
            output[...,:,N_buffer-1][...,index]  += x[...,0*(N_int-1):1*(N_int-1)]*N/2
            output[...,:,N_buffer+1][...,index]  -= x[...,0*(N_int-1):1*(N_int-1)]*N/2
            if bd_num >= 2:
                output[...,N_buffer-1,:][...,index] += x[...,1*(N_int-1):2*(N_int-1)]*N/2
                output[...,N_buffer+1,:][...,index] -= x[...,1*(N_int-1):2*(N_int-1)]*N/2
            if bd_num >= 3:
                output[...,:,-N_buffer][...,index] += x[...,2*(N_int-1):3*(N_int-1)]*N/2
                output[...,:,-N_buffer-2][...,index] -= x[...,2*(N_int-1):3*(N_int-1)]*N/2
            if bd_num >= 4:
                output[...,-N_buffer,:][...,index] += x[...,3*(N_int-1):4*(N_int-1)]*N/2
                output[...,-N_buffer-2,:][...,index] -= x[...,3*(N_int-1):4*(N_int-1)]*N/2
            return output