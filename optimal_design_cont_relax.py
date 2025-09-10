import numpy as np
import cvxpy as cp

def optimal_design_weights(
    X: np.ndarray,
    k: int,
    criterion: str = "A",  # "A" or "V"
    ridge: float = 1e-6,
    lb: float = 0.0,
    ub: float = 1.0,
    solver: str | None = None,
):
    """
    Solve an optimal design problem with continuous relaxation.

    Args:
        X: (n, p) design matrix (rows = candidate points).
        k: total weight budget (usually integer but continuous here).
        criterion: "A" (A-optimality) or "V" (V-optimality).
        ridge: small ridge term to ensure invertibility of information matrix.
        lb, ub: lower/upper bounds on weights w_i.
        solver: cvxpy solver.

    Returns:
        w_opt (n,): optimal weights
        obj_value (float): optimal objective value
    """

    n, p = X.shape[0], X.shape[1]
    w = cp.Variable(n, nonneg=True)

    # constraints
    constraints = [cp.sum(w) == float(k)]
    if lb > 0:
        constraints.append(w >= lb)
    if ub < np.inf:
        constraints.append(w <= ub)

    # information matrix M(w)
    I = np.eye(p)
    M = ridge * I + X.T @ cp.diag(w) @ X

    if criterion.upper() == "A":
        # A-optimal: minimize trace(M^{-1})
        objective = cp.Minimize(cp.matrix_frac(I,M))
    elif criterion.upper() == "V":
        # V-optimal: minimize average prediction variance
        objective = cp.Minimize(cp.matrix_frac(X,M))
    else:
        raise ValueError("criterion must be 'A' or 'V'")

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)

    if w.value is None:
        raise RuntimeError(f"{criterion}-optimal problem did not solve. Try a different solver or adjust ridge.")

    return np.asarray(w.value).ravel(), float(prob.value)

def run_simulation_optimal_design(
        X: np.ndarray, 
        obs_set: np.ndarray,
        ctrue: np.ndarray,
        nstart: int,
        ntotal: int, 
        criterion: str = "A",
        ridge: float = 1e-6,
        lb: float = 0.0,
        ub: float = 1.0,
        solver: str | None = None):
    
    res = {'chat':[], 'errs':[], 'MTMinv_norm':[]}
    
    if obs_set is not None:
        MTM = X[:,obs_set]
    else:
        MTM = X

    y = X @ ctrue
    print(y.shape)
    for i in range(nstart,ntotal+1):
        print("Calculating OD weights for: ", i, " samples")
        weights = optimal_design_weights(MTM,i,criterion,ridge, lb, ub, solver)[0]
        sample_indices = np.argsort(-weights)[:i]
        MTM_sampled = MTM[sample_indices,:]
        y_sampled = y[sample_indices]
        
        sol = np.linalg.lstsq(MTM_sampled, y_sampled, rcond=None)
        chat = np.zeros_like(ctrue, dtype=float)
        chat[obs_set] = sol[0]
        res['chat'].append(chat) 
        res['errs'].append(np.linalg.norm(X@(ctrue - chat)))
    return res

