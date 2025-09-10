import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.polynomial import Polynomial


# f = Polynomial.fromroots([-0.85, -0.05, -0.05, -0.05, 0.09, 0.19, 0.4, 0.8],domain=[-1,1])
f = lambda x: 4.25718 + 13.2*x**15 - 5*x**14 + 9*x**13 + 2*x**12 + 8*x**11 - 3.5*x**10 - 2.7*x**9 + 57*x**8 + 20*x**7 - 50*x**6 + 23 *x**5 + 17*x**4 - 4*x**3 + x**2 + 2*x

x_vec_all = np.linspace(0,1,1000)
y_vec_all = f(x_vec_all)#+ np.random.normal(0,0.05,len(x_vec_all))
all_points = list(zip(x_vec_all, y_vec_all))
sample_points = np.array(random.sample(list(all_points),5))
x_vec_samp = [p[0] for p in sample_points]
y_vec_samp = [p[1] for p in sample_points]

def poly_regr(vander_poly_size):
   
    vander_mat = np.vander(x_vec_samp, N=vander_poly_size, increasing=True)
    vander_mat_inv = np.linalg.pinv(vander_mat)
    theta_vec = vander_mat_inv @ y_vec_samp
    estimated_polynomial = Polynomial(theta_vec)
    y_hat_vec = estimated_polynomial(x_vec_all)

    risk = np.mean((y_vec_all-y_hat_vec)**2)

    return risk

risks = []
for n in range(1,17):
    risk = poly_regr(n)
    risks.append(risk)

plt.plot(risks)

plt.title("Risk Over Degree of Polynomials in Vandermonde Matrix (No Noise)")
plt.xlabel("Degree of Polynomial in Vandermonde Matrix Rows")
plt.ylabel("MSE")

plt.savefig("risk_plot_2.png")
plt.show()
