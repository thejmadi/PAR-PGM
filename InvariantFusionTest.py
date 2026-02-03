# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 22:24:02 2026

@author: tarun
"""

import numpy as np
import numpy.linalg as la
from sklearn.mixture import GaussianMixture
import scipy.stats as sci
import matplotlib.pyplot as plt
from scipy.special import logsumexp, softmax
from scipy.optimize import minimize
from scipy.stats import norm



def func(P1, P2a, P2b, w2a, w2b, db, const):
    P1a = P1 + P2a
    P1b = P1 + P2b
    val = np.sqrt(2*np.pi*P1b)*const
    val -= w2b*np.exp(-0.5*(db**2)/P1b)
    val /= (w2a*np.sqrt(P1b))
    val = np.log(np.sqrt(P1a)*val)
    val = np.sqrt(-2*P1a*val)
    return val

'''
def plot_gm_pairs(gm1, gm2, gm3, x_range=(-3, 10), n_points=1000):
    x = np.linspace(x_range[0], x_range[1], n_points)

    def gm_pdf(gm, x):
        x_reshaped = x.reshape(-1, 1)
        return np.exp(gm.score_samples(x_reshaped))

    pdf1 = gm_pdf(gm1, x)
    pdf2 = gm_pdf(gm2, x)
    pdf3 = gm_pdf(gm3, x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # GM1 vs GM2
    axes[0].plot(x, pdf1, label='GM1')
    axes[0].plot(x, pdf2, label='GM2')
    axes[0].set_title('GM1 vs GM2')
    axes[0].legend()
    axes[0].grid(True)

    # GM1 vs GM3
    axes[1].plot(x, pdf1, label='GM1')
    axes[1].plot(x, pdf3, label='GM3')
    axes[1].set_title('GM1 vs GM3')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
'''

def plot_gms_with_shaded(gm1, gm2, gm3, gm4=None, gm5=None, x_range=(-3, 10), n_points=1000):
    """
    Plots GM1, GM2, GM3 on the same figure.
    Optionally shades GM4 and GM5 with the same color as their lines.

    Parameters:
        gm1, gm2, gm3 : sklearn.mixture.GaussianMixture
            Fitted Gaussian Mixture models to plot.
        gm4, gm5 : sklearn.mixture.GaussianMixture, optional
            Distributions to shade.
        x_range : tuple
            (min, max) range for x-axis
        n_points : int
            Number of points in the x grid
    """
    x = np.linspace(x_range[0], x_range[1], n_points)

    def gm_pdf(gm, x):
        x_reshaped = x.reshape(-1, 1)
        return np.exp(gm.score_samples(x_reshaped))

    # Evaluate PDFs
    pdf1 = gm_pdf(gm1, x)
    pdf2 = gm_pdf(gm2, x)
    pdf3 = gm_pdf(gm3, x)

    # Create plot
    plt.figure(figsize=(8, 5))

    # Plot GM1, GM2, GM3
    plt.plot(x, pdf1, label='Reference 1', color='blue')
    plt.plot(x, pdf2, label='Test 2a', color='green')
    plt.plot(x, pdf3, label='Test 2b', color='red')

    # Shade GM4 if provided
    if gm4 is not None:
        pdf4 = gm_pdf(gm4, x)
        line4, = plt.plot(x, pdf4, label='Fused 3a', color='green', alpha=0)
        plt.fill_between(x, 0, pdf4, color=line4.get_color(), alpha=0.3)

    # Shade GM5 if provided
    if gm5 is not None:
        pdf5 = gm_pdf(gm5, x)
        line5, = plt.plot(x, pdf5, label='Fused 3b', color='red', alpha=0)
        plt.fill_between(x, 0, pdf5, color=line5.get_color(), alpha=0.3)

    plt.title('Reference, Test, and Fused GMs')
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.legend()
    plt.grid(True)
    plt.show()

def build_gmm_1d(means, variances, weights):
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    weights = np.asarray(weights, dtype=float)

    K = means.shape[0]

    gmm = GaussianMixture(
        n_components=K,
        covariance_type="full"
    )

    # Normalize weights
    gmm.weights_ = weights / weights.sum()

    # sklearn expects shape (K, D); here D = 1
    gmm.means_ = means.reshape(-1, 1)

    # Full covariances: shape (K, 1, 1)
    gmm.covariances_ = variances.reshape(-1, 1, 1)

    # Required precision Cholesky factors
    gmm.precisions_cholesky_ = (1.0 / np.sqrt(variances)).reshape(-1, 1, 1)

    return gmm

def fuse(gmm_a, gmm_b):
    def kalmanFusion(mu_1, cov_1, mu_2, cov_2):
        inv_cov = la.inv(cov_1 + cov_2)
        post_mu = mu_1 + cov_1 @ inv_cov @ (mu_2 - mu_1)
        post_cov = cov_1 - cov_1 @ inv_cov @ cov_1
        post_cov = (post_cov + post_cov.T) / 2
        return post_mu, post_cov
    
    K_a = gmm_a.n_components
    K_b = gmm_b.n_components
    post_weights = np.full((K_a, K_b), np.nan)
    likelihoods = np.full((K_a, K_b), np.nan)

    for ka in range(K_a):
        for kb in range(K_b):
            likelihoods[ka, kb] = sci.multivariate_normal.pdf(gmm_b.means_[kb], mean=gmm_a.means_[ka], cov=gmm_b.covariances_[kb] + gmm_a.covariances_[ka])
    
    post_weights = gmm_a.weights_[:, None] * likelihoods * gmm_b.weights_[None, :]
    post_weights /= np.sum(post_weights)
    
    post_mean = []
    post_cov = []
    for ka in range(K_a):
        for kb in range(K_b):
            post_temp = kalmanFusion(gmm_a.means_[ka], gmm_a.covariances_[ka], gmm_b.means_[kb], gmm_b.covariances_[kb])
            post_mean.append(post_temp[0][0])
            post_cov.append(post_temp[1][0])
    post_weights = list(post_weights.reshape(K_a * K_b))
    fused_gmm = build_gmm_1d(post_mean, post_cov, post_weights)
    return fused_gmm

def createThirdGM(gm_1, gm_2, K_2, K_3):
    def innerProdConstraint(x):
        weights_3 = x[:K_3]; means_3 = x[K_3:2*K_3]; variances_3 = x[2*K_3:]
        gm_3 = build_gmm_1d(means_3, variances_3, weights_3)
        inner_prod_1_3 = innerProd(gm_1, gm_3)
        return #inner_prod_1_2 - inner_prod_1_3

    # Optimization Variables: w_3, mu_3, P_3
    X = np.linspace(-5, 15, 2000000)
    constr = ({'type': 'eq', 'fun': lambda x: 1 - sum(x[:K_2])},
              {'type': 'eq', 'fun': lambda x: 1 - sum(x[3*K_2:3*K_2 + K_3])})
    
    bnds = ((0.1, 0.9),
            (0.1, 0.9),
            (1, 2),
            (2, 5), 
            (0.1, 1.9),
            (0.1, 1.9),
            
            (0.1, 0.9),
            (0.1, 0.9),
            (0, 1),
            (5, 15), 
            (0.1, 1.9),
            (0.1, 1.9))
    
    x0 = [0.5, 0.5, 1.5, 3.5, 1, 1, 0.8, 0.2, 0.5, 10, 1, 1]
    
    def obj(x):
        weights_2 = x[:K_2]; means_2 = x[K_2:2*K_2]; variances_2 = x[2*K_2:3*K_2]
        candidate_2 = build_gmm_1d(means_2, variances_2, weights_2)
        
        weights_3 = x[3*K_2:3*K_2 + K_3]; means_3 = x[3*K_2 + K_3:3*K_2 + 2*K_3]; variances_3 = x[3*K_2 + 2*K_3:]
        candidate_3 = build_gmm_1d(means_3, variances_3, weights_3)
        
        #X_2, _ = candidate_2.sample(1000000)
        logp_2 = candidate_2.score_samples(X[:,None])
        #p_2 = np.exp(logp_2)
        entropy_2 = -np.trapz(np.exp(logp_2)*logp_2, X)
        #X_3, _ = candidate_3.sample(1000000)
        logp_3 = candidate_3.score_samples(X[:,None])
        #p_2 = np.exp(logp_2)
        entropy_3 = -np.trapz(np.exp(logp_3)*logp_3, X)
        entropy_err = (entropy_2-entropy_3)**2
        
        inner_prod_1_2 = innerProd(gm_1, candidate_2)
        inner_prod_1_3 = innerProd(gm_1, candidate_3)
        inner_prod_err = (inner_prod_1_2 - inner_prod_1_3)**2
        return entropy_err + inner_prod_err
    
    r = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=constr,
                 options={'ftol': 1e-14, 'disp': True, 'maxiter': 1000, 'eps': 1e-12})
    return r

def innerProd(gm_a, gm_b):
    K_a = gm_a.n_components; K_b = gm_b.n_components
    
    inner_prod = np.full((K_a, K_b), np.nan)
    likelihoods = np.full((K_a, K_b), np.nan)
    
    for ka in range(K_a):
        mu_a = gm_a.means_[ka]; P_a = gm_a.covariances_[ka] #w_a = gm_a.weights_[ka]
        for kb in range(K_b):
            mu_b = gm_b.means_[kb]; P_b = gm_b.covariances_[kb] #w_b = gm_b.weights_[kb]
            likelihoods[ka, kb] = sci.multivariate_normal.pdf(mu_b, mean=mu_a, cov=P_a + P_b)
    inner_prod = np.sum(gm_a.weights_[:, None] * likelihoods * gm_b.weights_[None, :])
    
    return inner_prod

print("Start")
# Create control distribution
variances_1 = [1.0]
weights_1 = [1.0]
means_1 = [0.0]
gmm_1 = build_gmm_1d(means_1, variances_1, weights_1)


# Create test distribution 1
K_2 = 2
K_3 = 2

variances_2 = [1.0, 1.0]
weights_2 = [0.5, 0.5]
means_2 = [0.5, 1.5]
gmm_2 = build_gmm_1d(means_2, variances_2, weights_2)

#inner_prod_1_2_method_1 = 0.5*(np.mean(np.exp(gmm_1.score_samples(X_2))) + np.mean(np.exp(gmm_2.score_samples(X_1))))

res = createThirdGM(gmm_1, gmm_2, K_2, K_3)

weights_2 = res.x[:K_2]; means_2 = res.x[K_2:2*K_2]; variances_2 = res.x[2*K_2:3*K_2]
gmm_2 = build_gmm_1d(means_2, variances_2, weights_2)

weights_3 = res.x[3*K_2:3*K_2 + K_3]; means_3 = res.x[3*K_2 + K_3:3*K_2 + 2*K_3]; variances_3 = res.x[3*K_2 + 2*K_3:]
gmm_3 = build_gmm_1d(means_3, variances_3, weights_3)

X_2, _ = gmm_2.sample(10000000)
X_3, _ = gmm_3.sample(10000000)
gmm_2_entropy = -gmm_2.score(X_2)
gmm_3_entropy = -gmm_3.score(X_3)

print(res.x[:3*K_2])
print(res.x[3*K_2:])
print(f"P2a Inner Prod = {innerProd(gmm_1, gmm_2)}")
print(f"P2b Inner Prod = {innerProd(gmm_1, gmm_3)}")
print(f"P2a Entropy = {gmm_2_entropy}")
print(f"P2b Entropy = {gmm_3_entropy}")

#db_2 = 2
#da_2 = func(variances_1[0], variances_2[0], variances_2[1], weights_2[0], weights_2[1], db_2, IP_const)
#means_2 = [means_1[0] + da_2, means_1[0] + db_2]

#variances_3 = [1.0, 1.0]
#weights_3 = [0.5, 0.5]
#db_3 = 8.0

#da_3 = func(variances_1[0], variances_3[0], variances_3[1], weights_3[0], weights_3[1], db_3, IP_const)
#means_3 = [means_1[0] + da_3, means_1[0] + db_3]


plot_gms_with_shaded(gmm_1, gmm_2, gmm_3)

# Fuse clouds
fused_2 = fuse(gmm_1, gmm_2)
fused_3 = fuse(gmm_1, gmm_3)

# Calc entropy
entropy_fused_2 = []
entropy_fused_3 = []
n = []
for i in range(2, 9):
    n.append(i)
    fused_2_samples, _ = fused_2.sample(10**n[-1])
    fused_3_samples, _ = fused_3.sample(10**n[-1])
    
    entropy_fused_2.append(-fused_2.score(fused_2_samples))
    entropy_fused_3.append(-fused_3.score(fused_3_samples))

plt.plot(n, entropy_fused_2, c="green", label="Fused 3a")
plt.plot(n, entropy_fused_3, c="red", label="Fused 3b")
plt.ylabel("Entropy")
plt.xlabel("Log_10 Number of Samples")
plt.title("Fused Entropy vs. Number of Samples")
plt.legend()
plt.show()

plot_gms_with_shaded(gmm_1, gmm_2, gmm_3, fused_2, fused_3)

print(f"Fused cloud 2 Entropy: {entropy_fused_2}")
print(f"Fused cloud 3 Entropy: {entropy_fused_3}")
