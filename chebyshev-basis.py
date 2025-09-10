import numpy.polynomial.chebyshev as cheb
import numpy.polynomial.legendre as leg
import numpy as np
import matplotlib.pyplot as plt

# number of training points
interval = (-1,1)
d_M = 100
d_U = 2000

def get_norm(points):
    V = cheb.chebvander(points, deg=d_U)
    M_tm = V[:,:d_M]
    M_tu = V[:,d_M:]
    A = np.linalg.pinv(M_tm) @ M_tu
    return np.linalg.norm(A,ord=2)

random_pts_norms = []
chebyshev_pts_norm = []
n_range = list(range(4,201))
random_pts = np.random.uniform(low=interval[0], high=interval[1], size=len(n_range))
for n in n_range:
    # leg.leggauss(n)
    chebyshev_pts = cheb.chebpts2(n)

    # P_n = cheb.Chebyshev.basis(n)
    # chebyshev_pts = P_n.roots()

    random_pts_norms.append(get_norm(random_pts[:n]))
    chebyshev_pts_norm.append(get_norm(chebyshev_pts))

plt.style.use('seaborn-v0_8-whitegrid')

plt.figure(figsize=(12,8))
plt.grid(True, which="both", ls="--")
plt.yscale('log')
plt.title('Norm of the Aliasing Operator Over Differing Number of Training Points', fontsize=16)
# plt.plot(n_range, random_pts_norms, color='b', label='Induced Norm of Model With Uniformly Sampled Points as Input')
plt.plot(n_range,chebyshev_pts_norm, color='r', linestyle='--', label='Induced Norm of Model With Chebyshev Nodes as Input')
plt.axvline(x=100, linestyle='--', color = 'black', alpha = 0.5, label='Interpolation Threshold')

plt.xlabel('Number of Training Points')
plt.ylabel('Induced Norm of the Aliasing Operator')
plt.legend(bbox_to_anchor=(1.00, 1.02), loc='lower right')
plt.savefig('alias_norm_chebyshev.png')
plt.show()
