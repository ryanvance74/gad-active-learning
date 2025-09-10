import numpy as np
import numpy.polynomial.legendre as leg
import matplotlib.pyplot as plt

# number of training points
interval = (-1,1)
d_M = 100
d_U = 2000

def get_norm(points):
    V = leg.legvander(points, deg=d_U)
    M_tm = V[:,:d_M]
    M_tu = V[:,d_M:]
    A = np.linalg.pinv(M_tm) @ M_tu
    return np.linalg.norm(M_tm,ord=2)

random_pts_norms = []
legendre_pts_norms = []
n_range = list(range(4,251))
random_pts = np.random.uniform(low=interval[0], high=interval[1], size=250)
for n in n_range:
    print(n)
    

    P_n = leg.Legendre.basis(n)
    # legendre_pts = P_n.roots()
    legendre_pts = leg.leggauss(n)[0]

    random_pts_norms.append(get_norm(random_pts[:n]))
    legendre_pts_norms.append(get_norm(legendre_pts))

plt.style.use('seaborn-v0_8-whitegrid')




plt.figure(figsize=(12,8))
plt.grid(True, which="both", ls="--")
plt.yscale('log')
plt.title('Norm of the Aliasing Operator Over Differing Number of Training Points', fontsize=16)
# plt.plot(n_range, random_pts_norms, color='b', label='Induced Norm of Model With Uniformly Sampled Points as Input')
plt.plot(n_range,legendre_pts_norms, color='r', linestyle='--', label='Induced Norm of Model With Legendre Zeros as Input')
plt.axvline(x=100, linestyle='--', color = 'black', alpha = 0.5, label='Interpolation Threshold')

plt.xlabel('Number of Training Points')
plt.ylabel('Induced Norm of the Aliasing Operator')
plt.legend(bbox_to_anchor=(1.00, 1.02), loc='lower right')
plt.savefig('alias_norm_legendre.png')
plt.show()

