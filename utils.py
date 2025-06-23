import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from sklearn.neighbors import kneighbors_graph

def load_graph(graph_k_save_path, batch_data):
    
    path = graph_k_save_path

    n, _, = batch_data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)



    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def adj_norm(adj):
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj
    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)
    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), adj_tmp)
    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)
    return norm_adj


def numpy_to_torch(a, sparse=False):
    """
    numpy array to torch tensor
    :param a: the numpy array
    :param sparse: is sparse tensor or not
    :return: torch tensor
    """
    if sparse:
        a = torch.sparse.Tensor(a)
        a = a.to_sparse()
    else:
        a = torch.FloatTensor(a)
    return a


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).to(torch.float32)


class load_data(Dataset):
    def __init__(self, dataset):

        if dataset == 'cite' or dataset == 'hhar' or dataset == 'reut' \
                or dataset == 'dblp_for_np' or dataset == 'acm_for_np' or dataset == 'usps_for_np' or dataset == 'reut_for_np':
            data_cite = sio.loadmat('./data/{}.mat'.format(dataset))
            self.x = np.array(data_cite['fea'])
            self.x.astype(np.float64)
            self.y = np.array(data_cite['gnd'])
            self.y.astype(np.int64)
            self.y = self.y[:, -1]
        elif dataset == 'amap' or dataset == 'pubmed':
            load_path = "data/" + dataset + "/" + dataset
            self.x = np.load(load_path + "_feat.npy", allow_pickle=True)
            self.y = np.load(load_path + "_label.npy", allow_pickle=True)
        else:
            self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
            self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), \
               torch.from_numpy(np.array(self.y[idx])), \
               torch.from_numpy(np.array(idx))

from scipy.spatial.distance import cdist

def constructW_PKN(X, k, issymmetric=True):
    """
    Construct a similarity matrix with probabilistic k-nearest neighbors.
    
    Parameters:
        X: ndarray, shape (n, dim), each column is a data point
        k: int, number of neighbors
        issymmetric: bool, if True, set W = (W + W.T) / 2 to make W symmetric

    Returns:
        W: ndarray, shape (n, n), similarity matrix
    """
    # Transpose X to shape (n, dim) to match cdist input requirements
    n = X.shape[0]
    
    # Compute squared Euclidean distance matrix
    D = cdist(X, X, metric='sqeuclidean')
    
    # Sort distances and get indices
    idx = np.argsort(D, axis=1)
    
    # Initialize similarity matrix
    W = np.zeros((n, n))
    
    # Construct W with probabilistic k-nearest neighbors
    for i in range(n):
        # Get k nearest neighbors (skip the first one as it is the point itself)
        neighbors = idx[i, 1:k+2]
        di = D[i, neighbors]
        denom = k * di[-1] - np.sum(di[:-1]) + np.finfo(float).eps
        W[i, neighbors] = (di[-1] - di) / denom

    # Make the matrix symmetric if required
    if issymmetric:
        W = (W + W.T) / 2

    return W


def L2_distance_1(a, b):
    """
    Compute squared Euclidean distance between each pair of columns in a and b.
    
    Parameters:
        a: ndarray, shape (dim, n)
        b: ndarray, shape (dim, m)
    
    Returns:
        d: ndarray, shape (n, m), distance matrix
    """
    aa = np.sum(a * a, axis=0)
    bb = np.sum(b * b, axis=0)
    ab = np.dot(a.T, b)
    d = np.outer(aa, np.ones(len(bb))) + np.outer(np.ones(len(aa)), bb) - 2 * ab
    
    return np.maximum(d, 0)  # Ensure no negative distances due to numerical errors


def adjacent_mat(A):
        """
        Construct normlized adjacent matrix, N.B. consider only connection of k-nearest graph
        :param x: array like: n_sample * n_feature
        :return:
        """
        # A = A.toarray()
        np.fill_diagonal(A, 1)
        A = A * np.transpose(A)
        D = np.diag(np.reshape(np.sum(A, axis=1) ** -0.5, -1))
        normlized_A = np.dot(np.dot(D, A), D)
        return normlized_A

# def adjacent_mat(x, n_neighbors):
#         """
#         Construct normlized adjacent matrix, N.B. consider only connection of k-nearest graph
#         :param x: array like: n_sample * n_feature
#         :return:
#         """
#         A = kneighbors_graph(x, n_neighbors=n_neighbors, include_self=True).toarray()
#         A = A * np.transpose(A)
#         D = np.diag(np.reshape(np.sum(A, axis=1) ** -0.5, -1))
#         normlized_A = np.dot(np.dot(D, A), D)
#         return normlized_A