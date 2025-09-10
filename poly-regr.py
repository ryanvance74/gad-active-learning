import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import numpy.polynomial.legendre as legendre
import numpy.linalg as LA

domain = [-1, 1] 
MAKE_SAMPLE_GRAPHS = False

# These coefficients will be applied to the specified domain.
true_coeffs = [0, -1, 3, 2.5, -3, 2, -1.5, 10, 2, 3, -4, 2, 8, 4, 2, 4, 187, 0.2, 0.9, 0.5, 1, 0.7, 0.8, 0.5, 0.1, 0.02, 0.05, 0.01, 0.01, 0.2, 0.5, 5] 
true_degree = len(true_coeffs) - 1
true_poly = Polynomial(true_coeffs, domain=domain)

def true_function(x):
    return true_poly(x)


n_train_samples = 30
max_degree = 80
noise_level = 0.1

# --- Generate the Datasets ---
# Data is now generated using the domain variable.
np.random.seed(0) # for reproducibility
x_train = np.linspace(domain[0], domain[1], n_train_samples)
y_train = true_function(x_train) + np.random.normal(0, noise_level, n_train_samples)

# Create a larger, denser, noise-free test set to measure the true risk
x_test = np.linspace(domain[0], domain[1], 200)
y_test_true = true_function(x_test)


# --- 2. The Double Descent Experiment ---

train_risks = []
test_risks = []
spectral_norms = []
# The x-axis will be the number of parameters, which is degree + 1
num_params_list = range(1, max_degree + 2)

for p in num_params_list:
    degree = p - 1
    
    # --- Fitting using NumPy's built-in Legendre.fit ---
    # This single function handles the basis construction and fitting in a 
    # numerically stable way, replacing the manual matrix construction.
    leg_V = legendre.legvander(x_train, degree)
    leg_V_inv = np.linalg.pinv(leg_V)
    coeffs = leg_V_inv @ y_train
    fitted_poly = lambda x: legendre.legval(x,coeffs)
    # fitted_poly = Legendre.fit(x_train, y_train, deg=degree, domain=domain)
    # --- End of fitting section ---

    # --- Calculate Risk (MSE) ---
    
    # 1. Training Risk
    y_train_pred = fitted_poly(x_train)
    train_risk = np.mean((y_train - y_train_pred)**2)
    train_risks.append(train_risk)
    
    # 2. Test Risk
    y_test_pred = fitted_poly(x_test)
    test_risk = np.mean((y_test_true - y_test_pred)**2)
    test_risks.append(test_risk)

    # 3. Spectral Norm of pinv of Design Matrix
    spectral_norm = np.linalg.norm(leg_V_inv, ord=2)
    spectral_norms.append(spectral_norm)


# --- 3. Plot the Double Descent Curve ---
print(test_risks)
plt.figure(figsize=(12, 8))
plt.style.use('seaborn-v0_8-whitegrid')

plt.plot(num_params_list, train_risks, 'b-', label='Training Risk (MSE)', alpha=0.8)
plt.plot(num_params_list, test_risks, 'r-', lw=3, label='Test Risk (MSE)')
plt.plot(num_params_list, spectral_norms, 'g-', label='Spectral Norm of Pseudoinverse of Design Matrix')
plt.axvline(x=true_degree + 1, color='g', linestyle=':', lw=3, label=f'True Complexity (p={true_degree+1})')
plt.axvline(x=n_train_samples, color='k', linestyle='--', label=f'Interpolation Threshold (p=n={n_train_samples})')
plt.yscale('log')
plt.title('Risk at Different Parameter Counts', fontsize=16)
plt.xlabel('Number of Parameters (p = degree + 1)')
plt.ylabel('Mean Squared Error (Log Scale)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.ylim(bottom=1e-3) # Set a bottom limit to avoid extreme values
plt.xlim(0, max_degree + 1)
plt.savefig('risk_plot_legendre.png')
plt.show()

# --- 4. Plot visualizations of specific learned curves ---

# Select a few interesting degrees to visualize
if MAKE_SAMPLE_GRAPHS:
    degrees_to_plot = [3, 4, 15, 79] # Underfit, Good Fit, Threshold Overfit, Benign Overfit
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axs = axs.flatten()

    for i, degree in enumerate(degrees_to_plot):
        p = degree + 1
        ax = axs[i]
        
        # Fit the model for this specific degree using the stable method
        fitted_poly_viz = legendre.Legendre.fit(x_train, y_train, deg=degree, domain=domain)

        # Generate the learned curve
        y_pred_viz = fitted_poly_viz(x_test)

        # Plot the components
        ax.scatter(x_train, y_train, color='blue', s=20, alpha=0.7, label='Training Points')
        ax.plot(x_test, y_test_true, 'k-', lw=2, label='True Function')
        ax.plot(x_test, y_pred_viz, 'r-', lw=3, label='Learned Curve')
        
        ax.set_title(f'Model Fit: {p} Parameters')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.legend()

    fig.suptitle('Functions at Different Parameter Counts', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('example_functions.png')
    plt.show()
