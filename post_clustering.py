import numpy as np
from sklearn import cluster
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from scipy.sparse import eye
from scipy.linalg import svd
from sklearn.cluster import KMeans


nmi = normalized_mutual_info_score
ami = adjusted_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


def err_rate(gt_s, s):
    return 1.0 - acc(gt_s, s)

# def simplex_proj(Y):
#     Y = Y.T
#     N, D = Y.shape
#     X = np.sort(Y, axis=1)[:, ::-1]  # 降序排序
#     Xtmp = (np.cumsum(X, axis=1) - 1) * (1 / np.arange(1, D + 1))  # 计算调整值
#     X = np.maximum(Y - Xtmp[np.arange(N), (X > Xtmp).sum(axis=1) - 1][:, np.newaxis], 0)  # 调整并确保非负
#     return X
def simplex_proj(Y, target_sum):
    Y = Y.T
    N, D = Y.shape
    X = np.sort(Y, axis=1)[:, ::-1]  # 降序排序
    # 计算调整值，使每一行的和为 target_sum
    Xtmp = (np.cumsum(X, axis=1) - target_sum) * (1 / np.arange(1, D + 1))
    # 调整并确保非负
    X = np.maximum(Y - Xtmp[np.arange(N), (X > Xtmp).sum(axis=1) - 1][:, np.newaxis], 0)
    return X

def thrC(C, alpha):
    if alpha < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while not stop and t < S.shape[0]:
                csum = csum + S[t, i]
                if csum > alpha * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d, ro):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    n = C.shape[0]
    C = 0.5 * (C + C.T)
    # C = C - np.diag(np.diag(C)) + np.eye(n, n)  # good for coil20, bad for orl
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(n))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** ro)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    # spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
    #                                       assign_labels='discretize')
    spectral = cluster.SpectralClustering(
    n_clusters=K, eigen_solver='arpack', affinity='precomputed',
    assign_labels='discretize', eigen_tol=1e-5
)
    L = np.nan_to_num(L, nan=0.0)
    if np.isnan(L).any():
        raise ValueError("处理后矩阵 C 中仍存在 NaN 值。")
    spectral.fit(L)
    grp = spectral.fit_predict(L)
    return grp, L
# def spectral_clustering1(CKSym, n):
#     N = CKSym.shape[0]
#     MAXiter = 1000  # Maximum number of iterations for KMeans
#     REPlic = 20     # Number of replications for KMeans

#     # Compute normalized symmetric Laplacian
#     DN = np.diag(1.0 / np.sqrt(np.sum(CKSym, axis=1) + np.finfo(float).eps))
#     LapN = np.eye(N) - DN @ CKSym @ DN
#     uN, sN, vN = svd(LapN)
#     kerN = vN[:, -n:]  # Select the last n columns

#     # Normalize rows of kerN
#     kerNS = np.zeros_like(kerN)
#     for i in range(N):
#         kerNS[i, :] = kerN[i, :] / (np.linalg.norm(kerN[i, :]) + np.finfo(float).eps)

#     # Perform KMeans clustering
#     kmeans = KMeans(n_clusters=n, max_iter=MAXiter, n_init=REPlic)
#     groups = kmeans.fit_predict(kerNS)

#     return groups, kerN

def spectral_clustering(C, K, d, alpha, ro):
    C = thrC(C, alpha)
    C = simplex_proj(C, 1)
    y, _ = post_proC(C, K, d, ro)
    return y
