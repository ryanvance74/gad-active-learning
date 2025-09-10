import numpy as np
from numpy import linalg as la
import time

def get_next_point(V, X, optimality):
    """
    Returns the optimal point as a column vector.
    V: inverse of A.T @ A
    X: the candidate set of points
    """

    
    XV = X @ V
    den = 1 + np.sum(XV * X, axis=1) # x_i^T V x_i
    if optimality == "A":
        
        num = np.linalg.norm(XV, axis=1)**2    # ||V x_i||^2
    
    elif optimality == "V":
        C = X.T @ X
        num = np.sum(XV * (XV @ C), axis=1)
    else: return None
    vals = num / den

    opt_point_idx = np.argmax(vals)
    # get the row that has the max value
    x_opt = X[opt_point_idx, :] 
    return x_opt, opt_point_idx

def sherman_update(V, b):
    Vb = V @ b
    Vb /= np.sqrt(1. + np.inner(b, Vb))
    return  V - np.outer(Vb, Vb) # am I missing the denominator?
    # V @ b @ b.T @ V
    # num = Vb @ Vb.T
    # den = 1 + np.inner(b, Vb)
    # V = V - num / den
    # return V

def greedy_sub_mod(dataset, k, eps, optimality):
    """
    dataset: the data on which to run the greedy submodular optimization.
    each row should be a data point. expected to be numpy array.
    k: the number of data points to have in the sampled dataset.
    eps: small constant used to create an intial V matrix.
    """

    d = dataset.shape[1]
    V = eps * np.eye(d)
    A = np.array([])
    B = dataset.copy()
    
    indices = []
    for _ in range(k):
        b, opt_idx = get_next_point(V, B, optimality)
        B = np.delete(B,opt_idx, axis=0)
        indices.append(opt_idx)

        # sherman-morrison update
        V = sherman_update(V, b)
        
        if A.size == 0:
            A = b.T
        else:
            A = np.vstack([A, b.T])
    return A, indices

def run_simulation_greedy_sub_mod(
        M: np.ndarray, 
        obs_set: np.ndarray,
        ctrue: np.ndarray,
        nstart: int,
        ntotal: int,
        optimality: str):
    """ 
    returns the result of the simulation along with the sample points selected and runtimes
    """
    res = {'chat':[], 'c_err':[], 'errs':[], 'MTMinv_norm':[], 'runtime':[]}
    start_time = time.perf_counter()
    if obs_set is not None:
        MTM = M[:,obs_set]
    else:
        MTM = M

    y = M @ ctrue

    samples, indices = greedy_sub_mod(MTM,ntotal,0.001, optimality)

    for i in range(nstart,ntotal+1):
        # print("Calculating OD weights for: ", i, " samples")
        # print("SAMPLES", samples[:i,:], "INDICES", indices[:i])

        MTM_sampled = samples[:i,:]
        y_sampled = y[indices[:i]]
        
        sol = np.linalg.lstsq(MTM_sampled, y_sampled, rcond=None)

        chat = np.zeros_like(ctrue, dtype=float)
        chat[obs_set] = sol[0]

        res['chat'].append(chat)
        res['c_err'].append(np.linalg.norm(ctrue-chat))
        res['errs'].append(np.linalg.norm(M@(ctrue - chat)))
        try: 
            Minv = np.linalg.pinv(MTM_sampled)
            res['MTMinv_norm'].append(np.linalg.norm(Minv,ord=2))

        except np.linalg.LinAlgError:
            res['MTMinv_norm'].append(np.nan)

        res['runtime'].append(time.perf_counter() - start_time)
    return res, samples[:,1]
