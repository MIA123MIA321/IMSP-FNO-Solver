import numpy as np
import scipy
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import cg
from scipy import fft
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
import imageio
from PIL import Image
import sys

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
X_list = []
iters = 0
heatmap_params = {
    'cmap': 'gist_rainbow',
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


def Matrix_Gen(N, Q, k,ToTensor=False,Transpose=False):
    M = N + 1
    data1 = k * k * (1 + Q) - 4 * N * N
    data1 = np.tile(data1, 2)
    data2 = np.ones(M).reshape(-1, )
    data2[0] = 0
    data2[1] = 2
    data2 = np.tile(data2, 2)
    data2_plus = N * N * np.tile(data2, M)
    data2_minus = np.flipud(data2_plus)
    data3 = np.ones(M * M).reshape(-1, )
    data3[:M] = 0
    data3[M:2 * M] = 2
    data3_plus = N * N * data3
    data3_plus = np.tile(data3_plus, 2)
    data3_minus = np.flipud(data3_plus)
    matrix__ = np.ones((M, M))
    matrix__[0, 0] = matrix__[-1, 0] = matrix__[-1, -1] = matrix__[0, -1] = 2
    matrix__[1:-1, 1:-1] = 0
    data4 = -2 * k * matrix__.reshape(-1, ) * N
    data4_plus = np.tile(data4, 2)
    data4_minus = -data4_plus
    data = (np.c_[data1,data2_minus,data2_plus,
            data3_minus,data3_plus,data4_minus,
            data4_plus]).transpose()
    offsets = np.array([0, -1, 1, -M, M, -M * M, M * M])
    dia = dia_matrix((data, offsets), shape=(2 * M * M, 2 * M * M))
    mat = dia.tocoo()
    if Transpose:
        mat = mat.T
    if not ToTensor:
        return mat
    else:
        mat = dia.tocoo()
        values = torch.tensor(mat.data)
        indices = torch.tensor(np.array([mat.row, mat.col]), dtype=torch.long)
        shape = torch.Size(mat.shape)
        torch_sparse_mat = torch.sparse_coo_tensor(indices, values, shape)
        return torch_sparse_mat


def Diag_Gen(N, Q, k):
    M = N + 1
    diag0 = k * k * (1 + Q) - 4 * N * N+1j*2*k*N
    matrix__ = diag0.reshape((M, M))
    matrix__[0, 0]+=1j*2*k*N
    matrix__[0, -1]+=1j*2*k*N
    matrix__[-1, 0]+=1j*2*k*N
    matrix__[-1, -1]+=1j*2*k*N
    diag = matrix__.reshape(-1,)
    return torch.from_numpy(diag).to(torch.complex64)


def forward_conv(N,k,input_,padding = False):
    kernel = torch.tensor([
    [0.,   N*N,                0.],
    [N*N,  k*k-4*N*N, N*N],
    [0.,   N*N,                0.]]).to(input_.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    output_real = F.conv2d(input_[:,0].unsqueeze(1),kernel,stride=1,padding=0)
    output_imag = F.conv2d(input_[:,1].unsqueeze(1),kernel,stride=1,padding=0)
    output_ = torch.cat([output_real,output_imag],dim=1)
    if padding:
        return F.pad(output_,[1,1,1,1])
    else:
        return output_


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


def Round(vector, times):
    SHAPE = vector.shape
    vector1 = vector.reshape(-1, )
    SHAPE1 = vector1.shape[0]
    radius = (vector.max()-vector.min())/2
    ERR = np.array([random.uniform(-radius * times, radius * times)
                    for i in range(SHAPE1)])
    return vector + ERR.reshape(SHAPE)


def load_data(filename, device = 'cpu',angle_id = -1,
              NS_return = 'T', Usage = 'Train'):
    Dataset_dir = '/data/liuziyang/Programs/pde_solver/Dataset/'
    data = np.load(Dataset_dir+filename+'.npz', allow_pickle=True)    
    q = Type_Settle(data['q'], 'torch', device) # (nsample, 1, 65, 65)
    wave = Type_Settle(data['WAVE'], 'torch', device) # (angle_for_test, nsample, 1/2, 65, 65)
    u_i = Type_Settle(data['u_i'], 'torch', device) # (angle_for_test, nsample, 2, 65, 65)
    u_t = Type_Settle(data['u_t'], 'torch', device) # (angle_for_test, nsample, 2, 65, 65)
    u_NS = None 
    def flatten(x):
        SHAPE = x.shape
        if len(SHAPE)==5:
            return x.reshape(SHAPE[0] * SHAPE[1], SHAPE[2], SHAPE[3], SHAPE[4])
        elif len(SHAPE)==6:
            return x.reshape(SHAPE[0] * SHAPE[1], SHAPE[2], SHAPE[3],
                         SHAPE[4], SHAPE[5])
    if NS_return == 'T':
        u_NS = Type_Settle(data['u_NS'], 'torch', device) # (angle_for_test, nsample, NS_length, 2, 65, 65)
        
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
        rel_err = 100*(np.linalg.norm(data3)/np.linalg.norm(data1))
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


def model_eval(model, q, f,device = None, first=False):
    if first:
        out_0 = model(f[:,0:1])
        if torch.norm(f[:,1:2]) < 1e-8:
            out_1 = torch.zeros_like(out_0)
        else:
            out_1 = model(f[:,1:2])
        out = torch.stack([out_0[:,0]-out_1[:,1],out_0[:,1]+out_1[:,0]],1)
    else:
        out_0 = model(q*f[:,0:1])
        if torch.norm(f[:,1:2]) < 1e-8:
            out_1 = torch.zeros_like(out_0)
        else:
            out_1 = model(q*f[:,1:2])
        out = torch.stack([out_0[:,0]-out_1[:,1],out_0[:,1]+out_1[:,0]],1)
    if device is None:
        return out
    else:
        return out.to(device)
    
    
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
        print('iter {} completed'.format(iters),
              '        %s' % str(datetime.now())[:-7])


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