import os
import pandas as pd
import numpy as np
import cvxpy as cp
import scipy as sp

from sklearn import manifold
from sklearn.metrics import euclidean_distances


def df_from_csv(path, n):
    df = pd.read_csv(path)
    df = df[~df['state_name'].isin(['Alaska', 'Puerto Rico', 'Hawaii'])]    
    top_layer = 150
    middle_layer = 400
    bottom_layer = n - top_layer - middle_layer
    df1 = df.iloc[0:top_layer]
    df2 = df.iloc[round(len(df)/2 - middle_layer/2): round(len(df)/2 + middle_layer/2)]
    df3 = df.iloc[len(df)-1-bottom_layer:len(df)-1]
    return pd.concat([df1, df2, df3]).reset_index(drop=True)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def positions_from_df(df):
    pos = np.zeros((len(df),2))
    for i, row in df.iterrows():
        y, x = float(row.iloc[6]), float(row.iloc[7])
        pos[i, 0] = x
        pos[i, 1] = y
    alpha = 1/1.7
    pos = scale_coordinates(pos, alpha=alpha)
    return pos
  
def prepare_noisy_samples(pos, noise_std, k):
    D = euclidean_distances(pos)
    D_sparse = create_sparse_noisy_D(D, noise_std, k)
    reduction_percent = round((np.size(D_sparse[np.where(D_sparse != 0)]) - D_sparse.shape[0])/(np.size(D[np.where(D != 0)]) - D.shape[0])*100, 2)
    print(f'Delta has {reduction_percent}% of the entries of D. (k={k})')
    return D_sparse

def perform_mds(similarities):
    seed = np.random.RandomState(seed=0)
    mds = manifold.MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        # random_state=seed,
        dissimilarity="precomputed",
        # n_jobs=1,
    )
    pos = mds.fit(similarities).embedding_
    return pos

def scale_coordinates(pos, alpha=1):
    x_coords = pos[:,0]
    y_coords = pos[:,1]
    min_x, max_x, min_y, max_y = np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)
    width_x = max_x - max_y
    width_y = max_y - min_y
    dimension = max(width_x, width_y)
    scale_factor = 1/(dimension)
    translate_x = -(min_x+max_x)/2
    translate_y = -(min_y+max_y)/2
    scaled_x = (x_coords + translate_x)*scale_factor*alpha
    scaled_y = (y_coords + translate_y)*scale_factor*alpha
    return np.vstack([scaled_x, scaled_y]).T

def create_sparse_noisy_D(D, noise_std, k):
    if noise_std == 0.0:
        D_noisy = D
    else:
        noise_matrix = np.ones(D.shape) + np.random.normal(loc=0, scale=noise_std, size=D.shape)
        noisy_distances = np.triu(D)*noise_matrix
        D_noisy = noisy_distances + noisy_distances.T

    D_sparse = np.zeros(D_noisy.shape)
    for _, row in enumerate(D_noisy):
        k_neighs =  np.sort(np.argsort(row)[0:k+1])
        D_sparse[np.ix_(k_neighs, k_neighs)] = D_noisy[np.ix_(k_neighs, k_neighs)]
    return D_sparse

def calculate_Si(i, D_sparse, k):
    row = D_sparse[i]
    k_neighs = np.sort(np.hstack([np.nonzero(row)[0][np.argsort(row[np.nonzero(row)[0]])][:k], i]))
    D_k_i = D_sparse[np.ix_(k_neighs, k_neighs)] 
    pos_k_i = perform_mds(D_k_i)
    nlsp = sp.linalg.null_space(np.vstack([pos_k_i.T, np.ones(pos_k_i.shape[0])]))
    w_i_hat = np.zeros([D_sparse.shape[0], k-pos_k_i.shape[1]])
    w_i_hat[k_neighs,:] = nlsp
    return (w_i_hat @ w_i_hat.T)

def find_S_nullspace(D_sparse, iteration_limit, target_eval, k):
    print(f'Looking for the optimal S nullspace, iteration limit: {iteration_limit}')
    for j in range(iteration_limit):
        S = np.zeros(D_sparse.shape)
        for i, _ in enumerate(D_sparse):
            S += calculate_Si(i, D_sparse, k=k)
        evl_s, evc_s =  sp.linalg.eigh(S)
        print(f"Iteration {j+1}, Eigenvalue #{target_eval}: ", np.sort(np.abs(evl_s))[target_eval])
        if j==0 or np.sort(np.abs(evl_s))[target_eval] < np.sort(np.abs(min_evl_s))[target_eval]:
            min_evl_s = evl_s
            min_evc_s = evc_s
        if np.sort(np.abs(evl_s))[2] == 0.0:
            print('Degenerate solution found!')
            break
        if j == iteration_limit-1:
            print(f'Limit reached. Best result: {np.sort(np.abs(min_evl_s))[2]}')
    
    return min_evl_s, min_evc_s

def find_position_eigenvectors(eigenvalues, eigenvectors, baseline=0, m=2):
    return eigenvectors[:, np.argsort(np.abs(baseline - eigenvalues))[1:m+1]]

def solve_SDP(phi, delta, verbose=False):
    print(f'Running SDP... (m={phi.shape[0]})')
    P = cp.Variable((phi.shape[0], phi.shape[0]), symmetric=True)
    psd_constraint = [P >> 0]
    # target_sum = cp.norm(cp.diag(phi.T @ P @ phi) - delta)
    target_sum = cp.norm(cp.diag(cp.quad_form(phi, P)) - delta)
    objective = cp.Minimize(target_sum)
    problem = cp.Problem(objective, psd_constraint)
    problem.solve(verbose=verbose)
    if P.value is not None:
        print('SDP complete.')
    else:
        print('SDP incomplete.')
    return P.value

def prepare_data_for_SDP(D_sparse, phi_pos, skip_every_m=False):
    checked_indices = set()
    delta_i_j = []
    phi_i_j = []
    for i, row in enumerate(D_sparse):
        k_neighs = np.sort(np.hstack([np.nonzero(row)[0][np.argsort(row[np.nonzero(row)[0]])][:k], i]))
        for j in k_neighs:
            if (i,j) in checked_indices or (j,i) in checked_indices:
                continue
            checked_indices.add((i,j))
            phi_i_j.append(phi_pos[i] - phi_pos[j])
            delta_i_j.append(D_sparse[i,j]**2)
    
    phi_i_j = np.array(phi_i_j).T
    delta_i_j = np.array(delta_i_j)
    if skip_every_m:
        phi_i_j = phi_i_j[0:phi_i_j.shape[0], 0:phi_i_j.shape[1]:skip_every_m]
        delta_i_j = delta_i_j[0:delta_i_j.shape[0]:skip_every_m]
    return phi_i_j, delta_i_j


def singular_values(matrix):
    _, s, _ = np.linalg.svd(matrix)
    return s[np.where(s>0)]


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if isPD(A3):
        return A3
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3

def isPD(B):
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def low_rank_approximation(matrix, rank):
    u, s, vt = np.linalg.svd(matrix)
    return u[:, :rank]@ np.diag(s)[:rank, :rank] @ vt[:rank, :]

def save_matrix(matrix, filename):
    with open(filename, 'wb') as f:
        np.save(f, matrix)
        print(f'Saved file: {filename}')

if __name__ == '__main__':
    output_dir = 'output_070424_k2'
    n = 1097
    p = 2
    k = 2
    nullspace_iteration_limit = 10
    df = df_from_csv('usa_city_distances/simplemaps_uscities_basicv1.78/uscities.csv', n)


    noise_dict = {0.0: 'no noise', 0.01: '1% noise', 0.1: '10% noise'}
    m_dict = {0.0: 3, 0.01: 4, 0.1: 5}
    target_dict = {0.0: 2, 0.01: 2, 0.1: 3}


    original_embedding = positions_from_df(df)
    save_matrix(original_embedding, os.path.join(output_dir, f'us_cities.npy'.replace(' ', '_')))

    
    for noise_std in noise_dict.keys():
        print(f'% Running LRE using p={p}, n={n}, k={k}, noise std={noise_std}')
        label = noise_dict[noise_std]
        D_sparse = prepare_noisy_samples(original_embedding, noise_std, k)
        evl_s, evc_s = find_S_nullspace(D_sparse, iteration_limit=nullspace_iteration_limit, target_eval=target_dict[noise_std], k=k)
        phi_pos = find_position_eigenvectors(evl_s, evc_s, baseline=0, m=m_dict[noise_std])
        save_matrix(evl_s, os.path.join(output_dir, f'evl_{label}.npy'.replace(' ', '_')))

        if noise_std == 0.1: # When m is large we use only half the points is phi_pos so that the SDP will not crash the kernel.
            # phi_i_j, delta_i_j = prepare_data_for_SDP(D_sparse, phi_pos, p)
            phi_i_j, delta_i_j = prepare_data_for_SDP(D_sparse, phi_pos)
        else:
            phi_i_j, delta_i_j = prepare_data_for_SDP(D_sparse, phi_pos)
        
        P = solve_SDP(phi_i_j, delta_i_j)
        save_matrix(singular_values(P), os.path.join(output_dir, f'singular_val_{label}.npy'.replace(' ', '_')))
        A = np.linalg.cholesky(nearestPD(low_rank_approximation(P, p)))[:, :p]
        r = (A.T @ phi_pos.T).T
        save_matrix(r, os.path.join(output_dir, f'r_{label}.npy'.replace(' ', '_')))
        
