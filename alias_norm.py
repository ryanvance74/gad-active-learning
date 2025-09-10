import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.legendre as leg
import numpy.polynomial.chebyshev as cheb

# number of training points
interval = (-1,1)
d_M = 100
d_U = 2000

def get_norm_modeled(M):
    M_tm = M[:,:d_M]
    M_tm_dagger = np.linalg.pinv(M_tm)
    return np.linalg.norm(M_tm_dagger, ord=2)

random_pts_norms = []
legendre_pts_norms = []
cheb_pts_norms = []
lev_scores_norms = []

x = np.linspace(-1,1, 1000)
V = cheb.chebvander(x, deg=2000)
MTM_lev = V[:,:100]
# MTU_lev = V[:,100:]

l = V.shape[0]
lev_scores = np.zeros(l)
# compute the i-th leverage score
G = np.linalg.pinv(MTM_lev.T @ MTM_lev)
for i in range(l):
    x_i = MTM_lev[i,:]
    x_i_T = x_i.T
    lev_scores[i] = x_i_T @ G @ x_i

lev_sum = np.sum(lev_scores)
probs = [t / lev_sum for t in lev_scores]

all_indices = np.random.choice(list(range(l)), size=250, replace=True, p=probs)

n_range = list(range(4,251))
random_pts = np.random.uniform(low=interval[0], high=interval[1], size=250)
for n in n_range:
    print(n)
    

    # P_n = leg.Legendre.basis(n)
    # legendre_pts = P_n.roots()
    legendre_pts = leg.leggauss(n)[0]

    cheb_pts = cheb.chebgauss(n)[0]

    random_pts_norms.append(get_norm_modeled(leg.legvander(random_pts[:n], deg=d_U)))
    legendre_pts_norms.append(get_norm_modeled(leg.legvander(legendre_pts, deg=d_U)))
    cheb_pts_norms.append(get_norm_modeled(cheb.chebvander(cheb_pts, deg=d_U)))

    curr_indices = all_indices[:n]
    
    S = np.zeros((n,l))
    for i in range(len(curr_indices)):
        j_i = curr_indices[i]
        S[i,j_i] = 1 / np.sqrt(n * probs[j_i])
    MTM_sampled = S @ MTM_lev

    lev_scores_norms.append(np.linalg.norm(np.linalg.pinv(MTM_sampled), ord=2))








plt.figure(figsize=(12,8))
plt.grid(True, which="both", ls="--")
plt.yscale('log')
plt.title('Norm of MTM\\dagger Over Differing Number of Training Points', fontsize=16)
# print(random_pts_norms)
plt.plot(n_range, random_pts_norms, color='b', label='Induced Norm with Uniformly Sampled Points')
plt.plot(n_range,legendre_pts_norms, color='r', linestyle='--', label='Induced Norm with Legendre Zeros')
plt.plot(n_range, cheb_pts_norms, color='g',label = "Induced Norm with Chebyshev Zeros" )
plt.plot(n_range, lev_scores_norms, color='black', label='Induced Norm with Leverage Score Sampling')
plt.axvline(x=100, linestyle='--', color = 'black', alpha = 0.5, label='Interpolation Threshold')

plt.xlabel('Number of Training Points')
plt.ylabel('Induced Norm of MTM\\dagger')
plt.legend(bbox_to_anchor=(1.00, 1.00), loc='upper right')
plt.savefig('spectral_norms.png')
plt.show()