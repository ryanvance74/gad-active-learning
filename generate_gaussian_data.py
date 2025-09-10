import numpy as np
import pandas as pd


n_samples = 1000   
n_features = 15      
# do different noise levels
# look at make blobs from sklearn
noise_std = 0.1      

# Generate feature matrix X from multivariate normal
mean = np.zeros(n_features)
cov = np.eye(n_features)  
X = np.random.multivariate_normal(mean, cov, size=n_samples)

# Generate true coefficients and compute linear target
true_coefs = np.random.randn(n_features)
y = X @ true_coefs + np.random.normal(0, noise_std, size=n_samples)


df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
df["y"] = y

df.to_csv('gaussian_dataset.csv')
