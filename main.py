import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from post_clustering import spectral_clustering, acc, nmi
import scipy.io as sio
import math
import itertools
# import h5py
from sklearn.neighbors import kneighbors_graph
from utils import load_graph, constructW_PKN, adjacent_mat
from sklearn.preprocessing import StandardScaler
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.linalg import svd
from sklearn.manifold import TSNE
import seaborn as sns

class Conv2dSamePad(nn.Module):
    """
    Implement Tensorflow's 'SAME' padding mode in Conv2d.
    When an odd number, say `m`, of pixels are need to pad, Tensorflow will pad one more column at right or one more
    row at bottom. But Pytorch will pad `m+1` pixels, i.e., Pytorch always pads in both sides.
    So we can pad the tensor in the way of Tensorflow before call the Conv2d module.
    """

    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = math.ceil(float(in_height) / float(self.stride[0]))
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)


class ConvTranspose2dSamePad(nn.Module):
    """
    This module implements the "SAME" padding mode for ConvTranspose2d as in Tensorflow.
    A tensor with width w_in, feed it to ConvTranspose2d(ci, co, kernel, stride), the width of output tensor T_nopad:
        w_nopad = (w_in - 1) * stride + kernel
    If we use padding, i.e., ConvTranspose2d(ci, co, kernel, stride, padding, output_padding), the width of T_pad:
        w_pad = (w_in - 1) * stride + kernel - (2*padding - output_padding) = w_nopad - (2*padding - output_padding)
    Yes, in ConvTranspose2d, more padding, the resulting tensor is smaller, i.e., the padding is actually deleting row/col.
    If `pad`=(2*padding - output_padding) is odd, Pytorch deletes more columns in the left, i.e., the first ceil(pad/2) and
    last `pad - ceil(pad/2)` columns of T_nopad are deleted to get T_pad.
    In contrast, Tensorflow deletes more columns in the right, i.e., the first floor(pad/2) and last `pad - floor(pad/2)`
    columns are deleted.
    For the height, Pytorch deletes more rows at top, while Tensorflow at bottom.
    In practice, we usually want `w_pad = w_in * stride`, i.e., the "SAME" padding mode in Tensorflow,
    so the number of columns to delete:
        pad = 2*padding - output_padding = kernel - stride
    We can solve the above equation and get:
        padding = ceil((kernel - stride)/2), and
        output_padding = 2*padding - (kernel - stride) which is either 1 or 0.
    But to get the same result with Tensorflow, we should delete values by ourselves instead of using padding and
    output_padding in ConvTranspose2d.
    To get there, we check the following conditions:
    If pad = kernel - stride is even, we can directly set padding=pad/2 and output_padding=0 in ConvTranspose2d.
    If pad = kernel - stride is odd, we can use ConvTranspose2d to get T_nopad, and then delete `pad` rows/columns by
    ourselves; or we can use ConvTranspose2d to delete `pad - 1` by setting `padding=(pad - 1) / 2` and `ouput_padding=0`
    and then delete the last row/column of the resulting tensor by ourselves.
    Here we implement the former case.
    This module should be called after the ConvTranspose2d module with shared kernel_size and stride values.
    And this module can only output a tensor with shape `stride * size_input`.
    A more flexible module can be found in `yaleb.py` which can output arbitrary size as specified.
    """

    def __init__(self, kernel_size, stride):
        super(ConvTranspose2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = self.kernel_size[0] - self.stride[0]
        pad_width = self.kernel_size[1] - self.stride[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]


class ConvAE(nn.Module):
    def __init__(self, channels, kernels):
        """
        :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
        :param kernels:  a list containing all kernel sizes, it should satisfy: len(kernels) = len(channels) - 1.
        """
        super(ConvAE, self).__init__()
        assert isinstance(channels, list) and isinstance(kernels, list)
        self.encoder = nn.Sequential()
        for i in range(1, len(channels)):
            #  Each layer will divide the size of feature map by 2
            self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernels[i - 1], 2))
            self.encoder.add_module('conv%d' % i,
                                    nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2))
            self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        self.decoder = nn.Sequential()
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        for i in range(len(channels) - 1):
            # Each layer will double the size of feature map
            # self.decoder.add_module('deconv%d' % (i + 1),
            #                         nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2))
            self.decoder.add_module('deconv%d' % (i + 1),
                        nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2))
            self.decoder.add_module('padd%d' % i, ConvTranspose2dSamePad(kernels[i], 2))
            self.decoder.add_module('relud%d' % i, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
    def forward(self, x):
        return self.decoder(x)
    
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        adj.clone().detach()
        # print('features', features.shape)
        # print('weight', self.weight.shape)
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        # print("support:", support.shape)
        if active:
            output = F.relu(output)
        return output
class GCNEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(GCNEncoder, self).__init__()
        self.gcn1 = GCNLayer(input_dim, 500)
        self.gcn2 = GCNLayer(500, 500)
        self.gcn3 = GCNLayer(500, 2000)
        self.gcn4 = GCNLayer(2000, feature_dim)

    def forward(self, x, adj):
        x1 = self.gcn1(x, adj)
        x1 = x1.squeeze(0)
        # print('x1', x1.shape)
        x2 = self.gcn2(x1, adj)
        x2 = x2.squeeze(0)
        x3 = self.gcn3(x2, adj)
        x3 = x3.squeeze(0)
        x4 = self.gcn4(x3, adj)
        return x4

def soft_thresholding(S, tau):
    return np.sign(S) * np.maximum(np.abs(S) - tau, 0)

def tensor_nuclear_norm_optimization(X, lambda_reg=0.1, n=165, mu=1e-5, mu1=1e7, max_iter=100, tol=1e-5):
    zero_tensor = torch.zeros(2, n, n)
    zero_tensor = zero_tensor.numpy()
    X = X - zero_tensor/mu
    X_hat = fft(X, axis=2)
    Z_hat = X_hat.copy()
    
    for iteration in range(max_iter):
        Z_hat_old = Z_hat.copy()
        
        for i in range(Z_hat.shape[2]):
            U, S, Vh = np.linalg.svd(Z_hat[:, :, i], full_matrices=False)
            
            S = soft_thresholding(S, tau=lambda_reg)
            
            Z_hat[:, :, i] = U @ np.diag(S) @ Vh
        
        Z = np.real(ifft(Z_hat, axis=2))
        zero_tensor = zero_tensor + (X - Z)
        mu = min(mu * 2, mu1)

    return Z


##common
class SelfExpression1(nn.Module):
    def __init__(self, n):
        super(SelfExpression1, self).__init__()
        self.Coefficient1 = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient1, x)
        return y
##GCN
#324,355
class SelfExpression2(nn.Module):
    def __init__(self, n, init_adj):
    # def __init__(self, n):   
        super(SelfExpression2, self).__init__()
        self.Coefficient2 = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)
        # self.A = torch.tensor(init_adj, dtype=torch.float32)
        if init_adj is not None:
            self.A = nn.Parameter(torch.tensor(init_adj, dtype=torch.float32), requires_grad=True)
        else:
            self.A = nn.Parameter(torch.rand(n, n, dtype=torch.float32), requires_grad=True)
    def forward(self, x):  # shape=[n, d]
        A = self.A.cpu().detach().numpy()
        adj = (A + A.T) / 2
        adj = adjacent_mat(adj)
        adj = torch.tensor(adj, dtype=torch.float32).to(device)
        y = torch.matmul(self.Coefficient2, adj)
        y = torch.matmul(y, x)
        # y = self.GCNencoders(x, adj)
        return y


class DTCGCSC(nn.Module):
    def __init__(self, channels, kernels, num_sample, input_size, low_feature_dim):
        super(DTCGCSC, self).__init__()
        self.n = num_sample
        self.ae = ConvAE(channels, kernels)
        self.self_expression1 = SelfExpression1(self.n)
        self.self_expression2 = SelfExpression2(self.n, adj)
        # self.self_expression2 = SelfExpression2(self.n)
        self.GCNencoders = GCNEncoder(input_size, low_feature_dim).to(device)

    ###### tensor
    def forward(self, x):  # shape=[n, c, w, h]
        z = self.ae.encoder(x)
        mu = 1e-5
        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        shape = z.shape
        z = z.view(self.n, -1)  # shape=[n, d]

        z_recon1 = self.self_expression1(z)  # shape=[n, d]
        C1 = self.self_expression1.Coefficient1
        C1_cpu = C1.detach().cpu().numpy()

        z_recon2 = self.self_expression2(z)
        C2 = self.self_expression2.Coefficient2
        C2_cpu = C2.detach().cpu().numpy()
  
        C_tensor = np.stack((C1_cpu, C2_cpu), axis=0)
        Z_optimized = tensor_nuclear_norm_optimization(C_tensor, lambda_reg=0.1, n=self.n, mu=mu)
        sum_matrix = np.zeros_like(Z_optimized[0])
        
        # 对 Z_optimized 
        for matrix in Z_optimized:
            sum_matrix += matrix
        sum_matrix = sum_matrix.T + sum_matrix
        sum_matrix = torch.tensor(sum_matrix, device=x.device)
        
        z_recon_reshape = z_recon1.view(shape)
        x_recon = self.ae.decoder(z_recon_reshape)  # shape=[n, c, w, h]
        return x_recon, z, z_recon1, z_recon2, sum_matrix
        # return x_recon, z, z_recon1, z_recon2



    def loss_fn(self, x, x_recon, z, z_recon1, z_recon2, sum_matrix, weight_coef, weight_selfExp):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        loss_coef = torch.sum(torch.pow(sum_matrix, 2))
        # loss_coef = torch.sum(torch.pow(self.self_expression2.Coefficient2, 2))
        loss_selfExp1 = F.mse_loss(z_recon1, z, reduction='sum')
        loss_selfExp2 = F.mse_loss(z_recon2, z, reduction='sum')
        loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp2 + weight_selfExp* loss_selfExp1

        return loss

accs = []
nmis = []
def train(model,  # type: DTCGCSC
          x, y, epochs, lr=1e-3, weight_coef=1.0, weight_selfExp=150, device='cuda',
          alpha=0.04, dim_subspace=12, ro=8, show=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    K = len(np.unique(y))
    for epoch in range(epochs):
        x_recon, z, z_recon1, z_recon2, sum_matrix = model(x)
        loss = model.loss_fn(x, x_recon, z, z_recon1, z_recon2, sum_matrix, weight_coef=weight_coef, weight_selfExp=weight_selfExp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        if epoch % show == 0 or epoch == epochs - 1:
            C = sum_matrix.detach().to('cpu').numpy()
            # C = model.self_expression2.Coefficient2.detach().to('cpu').numpy()
            for i in range(0, 9):
                y_pred = spectral_clustering(C, K, dim_subspace, alpha, ro)
                accs.append(acc(y, y_pred))
                nmis.append(nmi(y, y_pred))
            averageacc = np.mean(accs)
            averagenmi = np.mean(nmis)
            print('Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f' %
                (epoch, loss.item() / y_pred.shape[0], averageacc, averagenmi))
            accs.clear()
            nmis.clear()


weight_coef = [0.001, 0.01, 0.1,1, 10, 100]
weight_selfExp = [0.001, 0.01, 0.1, 1, 10, 100]

param_combinations = list(itertools.product(weight_coef, weight_selfExp))
best_metric = float('0')
best_alpha = None
best_params = None


if __name__ == "__main__":
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description='DSCNet')
    parser.add_argument('--db', default='ar',
                        choices=['coil20', 'orl', 'MNIST', 'Yale64', 'Yale32', 'Umist', 'YaleB', 'ar', 'USPS', 'MSRA25_uni'])
    parser.add_argument('--show-freq', default=1, type=int)
    parser.add_argument('--ae-weights', default=None)
    parser.add_argument('--save-dir', default='results')
    args = parser.parse_args()
    args.graph_k_save_path = 'graph.txt'
    print(args)
    import os
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    db = args.db
    if db == 'coil20':
        # load data
        data = sio.loadmat('data/COIL20.mat')
        # x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        x, y = data['fea'], data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        adj = constructW_PKN(x, 2)
        x = x.reshape((-1, 1, 32, 32))
        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 15]
        kernels = [3]
        epochs = 40
        weight_coef = 1.0
        weight_selfExp = 75
        low_feature_dim = 256
        # post clustering parameters
        alpha = 0.04  # threshold of C
        dim_subspace = 12  # dimension of each subspace
        ro = 8  #
    elif db == 'coil100':
        # load data
        data = sio.loadmat('datasets/COIL100.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        # x, y = data['fea'], data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 50]
        kernels = [5]
        epochs = 120
        low_feature_dim = 256
        # post clustering parameters
        alpha = 0.04  # threshold of C
        dim_subspace = 12  # dimension of each subspace
        ro = 8  #
    elif db == 'orl':
        # load data
        data = sio.loadmat('data/ORL_32x32.mat')
        x, y = data['fea'], data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        adj = constructW_PKN(x, 6)
        x = x.reshape((-1, 1, 32, 32))
        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 3, 3, 5]
        kernels = [3, 3, 3]
        epochs = 700
        low_feature_dim = 256
        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #
    elif db == 'Yale64':
        # load data
        data = sio.loadmat('data/Yale_64x64.mat')
        x, y = data['X'].T, data['Y']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        adj = constructW_PKN(x, 6)
        x = x.reshape((-1, 1, 64, 64))
        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 15]
        kernels = [3]
        epochs = 400
        low_feature_dim = 256

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #
    elif db == 'Yale32':
        # load data
        data = sio.loadmat('data/Yale_32x32.mat')
        x, y = data['fea'], data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        adj = constructW_PKN(x, 2)

        x = x.reshape((-1, 1, 32, 32))
        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 15]
        kernels = [3]
        epochs = 500
        low_feature_dim = 256

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #
    
    elif db == 'YaleB':
        # load data
        data = sio.loadmat('data/YaleB.mat')
        x, y = data['fea'].T, data['gnd'].T
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        adj = constructW_PKN(x, 2)

        x = x.reshape((-1, 1, 32, 32))
        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 50]
        kernels = [3]
        epochs = 280
        low_feature_dim = 256

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #

    elif db == 'Umist':
        # load data
        data = sio.loadmat('data/Umist.mat')
        x, y = data['fea'], data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        adj = constructW_PKN(x, 2)

        x = x.reshape((-1, 1, 32, 32))
        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 50]
        kernels = [3]
        epochs = 100
        low_feature_dim = 256

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #
    elif db == 'ar':
        # load data
        data = sio.loadmat('data/ar.mat')
        x, y = data['data'], data['lable'].T
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        x = np.transpose(x, (2, 0, 1))
        x = x.reshape(3120, -1)
        adj = constructW_PKN(x, 2)

        x = x.reshape((-1, 1, 50, 40))
        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 15]
        kernels = [3]
        epochs = 300
        low_feature_dim = 256

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #
    elif db == 'USPS':
        # load data
        data = sio.loadmat('data/USPS.mat')
        x, y = data['A'], data['L']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        adj = constructW_PKN(x, 2)
        
        x = x.reshape((-1, 1, 16, 16))
        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 15]
        kernels = [3]
        epochs = 40
        low_feature_dim = 256

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #
    elif db == 'MSRA25_uni':
        # load data
        data = sio.loadmat('data/MSRA25_uni.mat')
        x, y = data['X'], data['Y']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        adj = constructW_PKN(x, 6)
        
        x = x.reshape((-1, 1, 16, 16))
        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 15]
        kernels = [3]
        epochs = 500
        low_feature_dim = 256

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #
    elif db == 'MNIST':
        # load data
        data = sio.loadmat('data/MNIST.mat')
        x, y = data['fea'], data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        adj = constructW_PKN(x, 6)
        
        x = x.reshape((-1, 1, 28, 28))
        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 15]
        kernels = [3]
        epochs = 100
        low_feature_dim = 256

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #
    
    for weight_coef, weight_selfExp in param_combinations:
        
        dscnet = DTCGCSC(channels=channels, kernels=kernels, num_sample=num_sample, input_size=x.shape[1], low_feature_dim=low_feature_dim).to(device)

        train(dscnet, x, y, epochs, weight_coef=weight_coef, weight_selfExp=weight_selfExp,
          alpha=alpha, dim_subspace=dim_subspace, ro=ro, show=args.show_freq, device=device)
        print(f'Alpha: {weight_coef}, Self-Loss Weight: {weight_selfExp}')
