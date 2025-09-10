import numpy as np
import matplotlib.pyplot as plt

from numpy.polynomial import legendre as leg

# f = Polynomial.fromroots([-0.85, -0.05, -0.05, -0.05, 0.09, 0.19, 0.4, 0.8],domain=[-1,1])
# p = lambda x: 4.25718 + 13.2*x**15 - 5*x**14 + 9*x**13 + 2*x**12 + 8*x**11 - 3.5*x**10 - 2.7*x**9 + 57*x**8 + 20*x**7 - 50*x**6 + 23 *x**5 + 17*x**4 - 4*x**3 + x**2 + 2*x
p = leg.Legendre([4.25718,13.2, 5, 9, 2, 8, 3.5, 2.7, 57, 20, 50, 23, 17, 4, 1,2])
def chebyshev_dist(x):
    return 1 / np.sqrt(1 - x**2)

x = np.linspace(-1,1, 1000)
V = np.vander(x, N=2000, increasing=True)
MTM = V[:,:100]
MTU = V[:,100:]

n = V.shape[0]
lev_scores = np.zeros(n)
# compute the i-th leverage score
for i in range(n):
    x_i = MTM[i,:]
    x_i_T = x_i.T
    lev_scores[i] = x_i_T @ np.linalg.pinv(MTM.T @ MTM) @ x_i

lev_sum = np.sum(lev_scores)
probs = [t / lev_sum for t in lev_scores]

lev_risks = []
alias_scores = []
y = p(x) + np.random.normal(0, 0.02, len(x))
all_indices = np.random.choice(list(range(n)), size=200, replace=True, p=probs)

for m in range(5,200):
    curr_indices = all_indices[:m]
    
    # print(curr_indices)
    S = np.zeros((m,n))
    for i in range(len(curr_indices)):
        j_i = curr_indices[i]
        S[i,j_i] = 1 / np.sqrt(m * probs[j_i])
    A = S @ MTM

    
    b = S @ y

    beta = np.linalg.pinv(A) @ b
    # print(MTM.shape, beta.shape, y.shape)
    residual = MTM @ beta - y
    lev_risk = np.linalg.norm(residual)**2
    lev_risks.append(lev_risk)
    alias_scores.append(np.linalg.norm(np.linalg.pinv(MTM) @ MTU, ord=2))

plt.yscale("log")
plt.plot(range(5,200), lev_risks)
# plt.plot(range(5,200), alias_scores)
plt.title("Risk Over Leverage Score Sampled Data Points (No Noise)")
plt.xlabel("Number of Data Points")
plt.ylabel("Risk")
plt.axvline(x=100, color='black', linestyle='--')
plt.savefig("lev_score_sampling_risk_noise.png")
plt.show()

# plt.figure(figsize=(8,6))
# plt.xlabel("x")
# plt.ylabel(r"$\tau$")
# plt.style.use('seaborn-v0_8-whitegrid')
# plt.title("Leverage Scores Over a Grid of x Values")
# plt.plot(x, lev_scores)
# plt.savefig("lev_scores.png")
# plt.show()
# plt.clf()

# plt.figure(figsize=(8,6))
# plt.title("Leverage Scores Compared With Chebyshev Continuum Distribution")
# plt.xlabel("x")
# plt.ylabel(r"$\tau$")
# plt.plot(x, lev_scores)
# plt.plot(x, chebyshev_dist(x))
# plt.grid(True, which="both", ls="--")
# plt.savefig("lev_scores_compared_to_chebyshev.png")
# plt.show()
