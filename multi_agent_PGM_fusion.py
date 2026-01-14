# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 16:52:54 2025

@author: tarun
"""
import json
import numpy as np
from numpy import linalg as la
from numpy import random as rand
import scipy as sci
from scipy import io as sio
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob
from numpy.polynomial.polynomial import Polynomial as poly
import time as time
from multiprocessing import Process, Queue, Pool, cpu_count
import datetime as dt
import pymap3d
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import ot

import CoordFunctions as cf
import PlottingFunctions as plot

import Dynamics as dyn
from pathlib import Path


# Check certain file for existence
def ensureDirExists(folder_path):
    # file_path: path variable to check for existence
    Path(folder_path).mkdir(parents=True, exist_ok=True)


def combineMsmts(full_pos_datasets, full_vel_datasets, partial_datasets):
    tol = 1e-8
    n = len(full_pos_datasets)

    # Collect all timesteps
    timestep_list = np.concatenate([
        full_pos_datasets[i][:, 0] for i in range(n)
    ])

    # Sort and merge using tolerance
    timestep_list = np.sort(timestep_list)
    all_timesteps = []

    if timestep_list.size > 0:
        all_timesteps.append(timestep_list[0])
        for t in timestep_list[1:]:
            if abs(t - all_timesteps[-1]) > tol:
                all_timesteps.append(t)

    all_timesteps = np.asarray(all_timesteps)
    num_timesteps = all_timesteps.size

    # Initialize output arrays
    combined_msmts = np.full((num_timesteps, 5, n), np.nan)
    combined_msmts[:, 0, :] = all_timesteps[:, None]
    combined_msmts[:, 1, :] = 0.0  # msmt_exists flag

    combined_states = np.full((num_timesteps, 7, n), np.nan)
    combined_states[:, 0, :] = all_timesteps[:, None]

    # Loop over datasets
    for i in range(n):
        full_pos_data = full_pos_datasets[i]
        full_vel_data = full_vel_datasets[i]
        partial_data  = partial_datasets[i]

        # Match timesteps with tolerance
        ia_states = matchTimesteps(all_timesteps, full_pos_data[:, 0], tol)
        ib_states = np.where(~np.isnan(ia_states))[0]
        ia_states = ia_states[ib_states].astype(int)

        ia_msmts = matchTimesteps(all_timesteps, partial_data[:, 0], tol)
        ib_msmts = np.where(~np.isnan(ia_msmts))[0]
        ia_msmts = ia_msmts[ib_msmts].astype(int)

        # Copy in measurement data
        combined_msmts[ia_msmts, 1, i]   = 1
        combined_msmts[ia_msmts, 2:5, i] = partial_data[ib_msmts, 1:4]

        # Copy in state data (pos + vel)
        combined_states[ia_states, 1:7, i] = np.hstack((
            full_pos_data[ib_states, 1:4],
            full_vel_data[ib_states, 1:4]
        ))

    return combined_msmts, combined_states, all_timesteps


def matchTimesteps(reference_ts, query_ts, tol):
    matched_indices = np.full(len(query_ts), np.nan)

    for i, qt in enumerate(query_ts):
        diff = np.abs(reference_ts - qt)
        idx = np.where(diff < tol)[0]
        if idx.size > 0:
            matched_indices[i] = idx[0]

    return matched_indices


def stateEstCloud(num_msmt_for_IOD, ts, nfit, theta_f, range_f, combined_msmt_data, low_lim, up_lim, normalization_quantities, rng):
    msmt_existance_mask = combined_msmt_data[:, 1] == True
    #times_of_msmts = combined_msmt_data[ts - num_msmt_for_IOD:ts+1, 0]
    mu_t = combined_msmt_data[msmt_existance_mask, 2:5].reshape(-1, 1).flatten()
    
    R_x = ((range_f * combined_msmt_data[msmt_existance_mask, 2])**2).reshape(-1, 1);
    R_y_z = np.tile((theta_f * 4.84814e-6)**2, (sum(msmt_existance_mask), 1))
    R_t = np.hstack((R_x, R_y_z, R_y_z)).reshape(-1, 1)
    R_t = np.diag(R_t.flatten())
    
    #R_t = np.zeros(R_t.shape)
    data_vec = rng.multivariate_normal(mu_t, R_t).T
    
    noised_obs = np.hstack((combined_msmt_data[msmt_existance_mask, 0].reshape(-1, 1), data_vec.reshape(-1, 3)))
    for i in range(noised_obs.shape[0]):
        noised_obs[i,1] = rng.uniform(low_lim, up_lim)/normalization_quantities['dist2km']
    
    hdo = noised_obs[:, :]

    # Convert observation data into [X, Y, Z] data in the topographic frame.           
    hdR = np.zeros((hdo.shape[0],4)) # Convert quantities of hdo to [X, Y, Z]
    hdR[:,0] = hdo[:,0] # Timestamp stays the same
    hdR[:,1] = hdo[:,1] * np.cos(hdo[:,3]) * np.cos(hdo[:,2]) # Conversion to X
    hdR[:,2] = hdo[:,1] * np.cos(hdo[:,3]) * np.sin(hdo[:,2]) # Conversion to Y
    hdR[:,3] = hdo[:,1] * np.sin(hdo[:,3]) # Conversion to Z
    #matches = times_of_msmts[:, None] == hdR[:, 0][None, :]
    #times_idx_hdR = np.nonzero(np.isin(hdR[:, 0], times_of_msmts))[0]
    #[~, times_idx_hdR] = ismembertol(times_of_msmts, hdR[:,0], 1e-6);
    hdR_p = hdR[:, :] # Matrix for a partial half-day observation

    # Fit polynomials for X, Y, and Z (Cubic for X, Quadratic for X and Y)
    coeffs_X = poly.fit(hdR_p[:,0], hdR_p[:,1], 4, domain=[])
    coeffs_Y = poly.fit(hdR_p[:,0], hdR_p[:,2], 4, domain=[])
    coeffs_Z = poly.fit(hdR_p[:,0], hdR_p[:,3], 4, domain=[])
    
    # Predicted values for X, Y, and Z given the polynomial fits
    X_fit = coeffs_X(hdR_p[:,0])
    Y_fit = coeffs_Y(hdR_p[:,0])
    Z_fit = coeffs_Z(hdR_p[:,0])

    # Now that you have analytically calculated the coefficients of the fitted
    # polynomial, use them to obtain values for X_dot, Y_dot, and Z_dot.
    # 1) Plot the X_dot, Y_dot, and Z_dot values for the time points for the
    # slides. 
    # 2) Find a generic way of obtaining and plotting X_dot, Y_dot, and Z_dot
    # values given some set of [X_coeffs, Y_coeffs, Z_coeffs]. 

    coeffs_dX = coeffs_X.deriv()
    coeffs_dY = coeffs_Y.deriv()
    coeffs_dZ = coeffs_Z.deriv()

    # Predicted values for Xdot, Ydot, and Zdot given the polynomial fits
    Xdot_fit = coeffs_dX(hdR_p[:,0])
    Ydot_fit = coeffs_dY(hdR_p[:,0])
    Zdot_fit = coeffs_dZ(hdR_p[:,0])

    Xfit = np.array([[X_fit[-1], Y_fit[-1], Z_fit[-1], Xdot_fit[-1], Ydot_fit[-1], Zdot_fit[-1,]]])
    return Xfit


def termSat(T, Y):
    mu = 1.2150582e-2 # Dimensionless mass of the moon (and position of Earth w.r.t. barycenter)
    Rm = 1740/384400 # Nondimensionalized radius of the moon
    value = (np.sqrt((Y[0] + mu)**2 + Y[1]**2 + Y[2]**2) < 6371/384400) or (np.sqrt((Y[0] - (1-mu))**2 + Y[1]**2 + Y[2]**2) < Rm) # Stop when the target hits the Earth's or the Moon's surface
    return 1

def getNoisyMeas(Xtruth, R, h, rng):
    mzkm = h(Xtruth).flatten()
    zk = rng.multivariate_normal(mzkm, R)
    #zk = zk.reshape(-1,1) # Make into column vector        
    return zk

def kalmanUpdate(zk, Xcloud, R, mu_m, P_m, h):
    N = Xcloud.shape[0]
    mu_m = mu_m.reshape(1, -1)
    Zcloud = h(Xcloud)
    
    mzk_m = np.mean(Zcloud,0)

    Pxz = np.zeros((mzk_m.shape[0], P_m.shape[0]))
    Pzz = np.zeros((mzk_m.shape[0], mzk_m.shape[0]))

    for i in range(N):
        dx = (Xcloud[i,:] - mu_m).reshape([-1, 1])
        dz = (Zcloud[i,:] - mzk_m).reshape([-1, 1])
        Pxz = Pxz + dz@dx.T
        Pzz = Pzz + dz@dz.T
    

    Pxz = Pxz/N
    Pzz = Pzz/N + R

    K_k = Pxz.T@la.inv(Pzz)
    mu_p = mu_m + K_k@(zk - h(mu_m)).reshape([-1])
    P_p = P_m - K_k@Pzz@K_k.T
    
    P_p = (P_p + P_p.T)/2

    D, V = np.linalg.eig(P_p)
    D = np.diag(D)
    P_p = V@D@V.T
    return mu_p, P_p

def kalmanUpdate2(zk, Xcloud, R, mu_m, P_m, h):
    N = Xcloud.shape[0]
    # TODO: Recheck math
    mu_m = mu_m.reshape(1, -1)
    Zcloud = h(Xcloud)
    
    mzk_m = np.mean(Zcloud,0)

    #Pxz = np.zeros((mzk_m.shape[0], P_m.shape[0]))
    #Pzz = np.zeros((mzk_m.shape[0], mzk_m.shape[0]))

    #for i in range(N):
    #    dx = (Xcloud[i,:] - mu_m).reshape([-1, 1])
    #    dz = (Zcloud[i,:] - mzk_m).reshape([-1, 1])
    #    Pxz = Pxz + dz@dx.T
    #    Pzz = Pzz + dz@dz.T
    
    dx = Xcloud - mu_m
    dz = Zcloud - mzk_m
    
    Pxz = dx.T @ dz
    Pzz = dz.T @ dz

    Pxz = Pxz.T/N
    Pzz = Pzz.T/N + R
    
    K_k = Pxz.T@la.inv(Pzz)
    mu_p = mu_m + K_k@(zk - h(mu_m)).reshape([-1])
    P_p = P_m - K_k@Pzz@K_k.T
    
    P_p = (P_p + P_p.T)/2

    D, V = np.linalg.eig(P_p)
    D = np.diag(D)
    P_p = V@D@V.T
    return mu_p, P_p

def weightUpdate(wc, X_cloud, idx, zk, R, h):
    # TODO: Make sure empty rows in wc.shape[0] do not affect function
    wGains = np.full((wc.shape[0]), np.nan)
    for k in range(wc.shape[0]):
        cPts = X_cloud[idx == k, :]
        #zPoints = np.zeros((cPts.shape[0], zk.shape[0]))
        #for j in range(cPts.shape[0]):
        #    zPoints[j,:] = h(cPts[j,:])
        zPoints = h(cPts)
        zPredMean = np.mean(zPoints, axis=0)
        zPredCov = np.cov(zPoints.T) + R
        
        wGains[k] = sci.stats.multivariate_normal.pdf(zk.T, mean=zPredMean, cov=zPredCov)
    w = wc * wGains / np.sum(wc * wGains)
    return w, wGains

def drawFrom(w, mu, P, N, rng):
    wtoken = rng.random()
    pos = np.searchsorted(np.cumsum(w), wtoken)
    mu_t = mu[pos,:]
    R_t = (P[pos,:] + P[pos,:].T)/2
    x_p = rng.multivariate_normal(mu_t, R_t).astype(np.float64)
    return x_p, pos

def drawFrom2(w, mu, P, N, rng):
    
    wtoken = rng.random(N)
    pos = np.searchsorted(np.cumsum(w), wtoken)
    #mu_t = mu[pos,:]
    #R_t = (P[pos,:] + P[pos,:].transpose(0, 2, 1))/2
    #x_p = np.random.multivariate_normal(mu_t, R_t).astype(np.float64)
    z = rng.standard_normal((N, 6))
    chol = la.cholesky(P)

    x_p = mu[pos] + np.einsum("nij,nj->ni", chol[pos], z, optimize=True)
    return x_p, pos

# TODO: Check if actually needed
def getDiagCov(Xcloud):
    P = np.cov(Xcloud.T)
    ent = np.diag(P)
    return ent

def propPoint(particle_prior_synodic, t_int, interval, dynamics_model=dyn.cr3bp_dyn, ev=termSat):
    # First, convert from X_{ot} in the topocentric frame to X_{bt} in the
    # synodic frame.
    ##Xbest = backConvertSynodic(np.copy(Xest), t_int)
    
    # Next, propagate X_{bt} by a single time step and convert back to the 
    # topographic frame. Begin by calling the integrator
    
    ivp_result = sci.integrate.solve_ivp(dynamics_model, (0, interval),
    np.copy(particle_prior_synodic), method='RK45', rtol=1e-6, atol=1e-8) 
    particle_post_synodic = ivp_result.y.T[-1, :]
    #print(ivp_result.nfev)
    
    # Finish by converting back to topocentric reference frame
    ##Xm = convertToTopo(np.copy(Xbt_est), t_int + interval)
    
    return particle_post_synodic

def propagate(cloud_prior_topo, t_int, interval, obs_lat, obs_lon, obs_el, normalization_quantities):
    
    cloud_prior_synodic = cf.Topo2Synodic(np.copy(cloud_prior_topo), t_int, obs_lat, obs_lon, obs_el, normalization_quantities)
    
    termSat.terminal = True
    termSat.direction = 0
    '''
    pool_propInputs = []
    for i in range(cloud_prior_synodic.shape[0]):
        pool_propInputs.append((np.copy(cloud_prior_synodic[i,:]), t_int, interval))

    with Pool(W) as p:
        prop_Points = p.starmap(propPoint, pool_propInputs)
    '''
    cloud_post_synodic = np.zeros(cloud_prior_synodic.shape)
    starttime = time.time()
    for particle in range(cloud_prior_synodic.shape[0]):
        cloud_post_synodic[particle, :] = propPoint(np.copy(cloud_prior_synodic[particle, :]), t_int, interval)
    endtime = time.time()
    #print(endtime - starttime)
    #cloud_post_synodic = np.asarray(prop_Points)
    cloud_post_topo = cf.Synodic2Topo(np.copy(cloud_post_synodic), t_int+interval, obs_lat, obs_lon, obs_el, normalization_quantities)
    
    return cloud_post_topo


def cluster(X_cloud, K, rng, ts, load_loc2):
    # Z-Score normalized particle cloud
    X_norm, _, _ = standardize(X_cloud)
    
    # Kmeans clustering of normalized cloud
    while True:
        kmeans = KMeans(n_clusters=K, init="k-means++", random_state=int(rng.integers(0, 2**32))).fit(X_norm)
        # kmeans = KMeans(n_clusters=K, init="k-means++").fit(Xm_cloud)
        temp = kmeans.cluster_centers_
        idx = kmeans.labels_
        
        if np.all(np.bincount(idx) > 6):  # Check if all clusters have more than 6 points
            break
        else:
            K -= 1  # Reduce the number of clusters if condition is not met
    #cluster_func_idx = sci.io.loadmat(load_loc2/f"cluster_func_idx{ts}.mat")
    #idx = cluster_func_idx['idx'].flatten() - 1
    # Calculate cluster statistics
    means = np.full((K, 6), np.nan)
    covs = np.full((K, 6, 6), np.nan)
    weights = np.full((K), np.nan)
    for k in range(K):
        cluster_points = X_cloud[idx == k, :]
        #cPoints.append(cluster_points)
        means[k, :] = np.mean(cluster_points, 0) # Cell of GMM means 
        covs[k, :, :] = np.cov(cluster_points.T) # Cell of GMM covariances
        weights[k] = cluster_points.shape[0] / X_cloud.shape[0] # Vector of weights
    
    return idx, means, covs, weights, K


def standardize(X_cloud , h=None):
    mu = np.mean(X_cloud, 0)
    std = np.std(X_cloud, 0, ddof=1)
    X_norm = (X_cloud - mu)/std
    return X_norm, mu, std


def unstandardize(gmm_normalized, original_mu, original_std):
    num_components = gmm_normalized.n_components

    # Calculate unnormalized means
    unnormalized_mu = gmm_normalized.means_ * original_std + original_mu
    
    # Calculate unnormalized covariances
    diag_stds = np.diag(original_std)
    unnormalized_covs = np.zeros_like(gmm_normalized.covariances_)
    for k in range(num_components):
        unnormalized_covs[k] = diag_stds @ gmm_normalized.covariances_[k] @ diag_stds

    # Create unnormalized GMM
    gmm_unnormalized = GaussianMixture(
        n_components=num_components,
        covariance_type='full'
    )
    gmm_unnormalized.weights_ = gmm_normalized.weights_.copy()
    gmm_unnormalized.means_ = unnormalized_mu.copy()
    gmm_unnormalized.covariances_ = unnormalized_covs.copy()
    gmm_unnormalized.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(unnormalized_covs))
    return gmm_unnormalized


def fitGMM(X, rng, K_upper = 29):
    if K_upper >= 16:
        K_int = 4
    else:
        K_int = 2
    k_range = range(4, K_upper, K_int)
    best_gmm = (None, np.inf, None)
    prev = None
    
    for k in k_range:
        gmm = GaussianMixture(k, covariance_type="full", max_iter=100, n_init=1, random_state=int(rng.integers(0, 2**32)))

        if prev is not None:
            for a in ("weights_", "means_", "covariances_", "precisions_cholesky_"):
                setattr(gmm, a, getattr(prev, a)[:k])

        gmm.fit(X)
        aic = gmm.aic(X)

        if aic < best_gmm[1]:
            best_gmm = (gmm, aic, k)

        prev = gmm

    return best_gmm


def getCloudMetrics(X_cloud, X_truth, t_int, cloud_is_active, K_initial, obs_lat, obs_lon, obs_el, normalization_quantities, rng, ts, save_loc2, h=None):
    def js_divergence(gmP, gmQ, XP, XQ):
        def log_m(log_p, log_q): 
            return sci.special.logsumexp([log_p, log_q], axis=0) - np.log(2)
    
        KL_P_M = np.mean(gmP.score_samples(XP) - log_m(gmP.score_samples(XP), gmQ.score_samples(XP)))
        KL_Q_M = np.mean(gmQ.score_samples(XQ) - log_m(gmP.score_samples(XQ), gmQ.score_samples(XQ)))
        
        return 0.5 * (KL_P_M + KL_Q_M) / np.log(2)
    '''
    norm_data = sci.io.loadmat(save_loc2/f"norm_gmm_params_{ts}.mat")
    norm_weights = norm_data["norm_weights"].ravel()        # (K,)
    norm_means   = norm_data["norm_means"]                  # (K, D)
    norm_covs    = norm_data["norm_covs"]                   # (D, D, K)
    norm_covs = np.transpose(norm_covs, (2, 0, 1))
    
    unnorm_data = sci.io.loadmat(save_loc2/f"unnorm_gmm_params_{ts}.mat")
    unnorm_weights = unnorm_data["unnorm_weights"].ravel()        # (K,)
    unnorm_means   = unnorm_data["unnorm_means"]                  # (K, D)
    unnorm_covs    = unnorm_data["unnorm_covs"]                   # (D, D, K)
    unnorm_covs = np.transpose(unnorm_covs, (2, 0, 1))
    
    K, D = norm_means.shape
    
    norm_gmm_mat = GaussianMixture(
        n_components=K,
        covariance_type="full")
    # Manually set parameters
    norm_gmm_mat.weights_ = norm_weights
    norm_gmm_mat.means_ = norm_means
    norm_gmm_mat.covariances_ = norm_covs
    # sklearn requires precisions too
    norm_gmm_mat.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(norm_covs))
    
    unnorm_gmm_mat = GaussianMixture(
        n_components=K,
        covariance_type="full")
    # Manually set parameters
    unnorm_gmm_mat.weights_ = unnorm_weights
    unnorm_gmm_mat.means_ = unnorm_means
    unnorm_gmm_mat.covariances_ = unnorm_covs
    # sklearn requires precisions too
    unnorm_gmm_mat.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(unnorm_covs))
    '''
    # TODO: Rename total_num_clouds to max_num_clouds everywhere
    gmm_unnormalized = [None]*cloud_is_active.shape[0]
    
    AIC = np.full(cloud_is_active.shape, np.nan)
    K = np.full(cloud_is_active.shape, np.nan)
    gmm_truth_loglikelihood = np.full(cloud_is_active.shape, np.nan)
    gmm_entropy = np.full(cloud_is_active.shape, np.nan)
    gmm_RMSE = np.full(cloud_is_active.shape+(6,), np.nan)
    best_truth_loglikelihood = np.full(cloud_is_active.shape, np.nan)
    best_entropy = np.full(cloud_is_active.shape, np.nan)

    for ob in np.where(cloud_is_active[:, 0])[0]:
        if h is None:
            metric_truth = cf.Topo2Synodic(np.copy(X_truth[ob]), t_int, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
        else:
            metric_truth = h(np.copy(X_truth[ob]))
        for cloud in np.where(cloud_is_active[ob])[0]:
            if h is None:
                metric_cloud = cf.Topo2Synodic(np.copy(X_cloud[ob, cloud]), t_int, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
            else:
                metric_cloud = h(np.copy(X_cloud[ob, cloud]))
            # Create unnormalized gmm for analysis
            X_norm, mu, std = standardize(metric_cloud)
            gmm_normalized, AIC[ob, cloud], K[ob, cloud] = fitGMM(np.copy(X_norm), rng)
            gmm_unnormalized_temp = unstandardize(gmm_normalized, mu, std)
            if cloud == 0:
                gmm_unnormalized[ob] = gmm_unnormalized_temp
            # Calculate various metrics of GMM
            # Complete GMM metrics
            # Log-Likelihoood
            gmm_truth_loglikelihood[ob, cloud] = gmm_unnormalized_temp.score_samples(metric_truth)[0]
            # Entropy
            gmm_entropy[ob, cloud] = -gmm_unnormalized_temp.score(metric_cloud)
            #gmm_entropy = -np.mean(gmm_particles_loglikelihood)
            # Standard Deviations
            # RMSE
            if h is None:
                gmm_RMSE[ob, cloud, :] = np.sqrt(np.mean((metric_cloud - metric_truth)**2, axis=0))
            
            # Best mode metrics
            # TODO: Best mode metrics may be incorrect
            #candidate_likelihoods = gmm_unnormalized.predict_proba(X_truth) # Gives relative likelihoods, if uniform, all modes describe truth equally well/badly
            #best_k = np.argmax(candidate_likelihoods)
            #best_samples = rng.multivariate_normal(gmm_unnormalized.means_[best_k], gmm_unnormalized.covariances_[best_k], size=100_000)
            # Log-Likelihood
            best_truth_loglikelihood = 0#np.log(np.max(candidate_likelihoods))
            #best_particles_loglikelihood = sci.stats.multivariate_normal.logpdf(best_samples, mean=gmm_unnormalized.means_[best_k], cov=gmm_unnormalized.covariances_[best_k], allow_singular=True)
            #best_particles_loglikelihood = _estimate_log_gaussian_prob(best_samples, gmm_unnormalized.means_[best_k:best_k+1], gmm_unnormalized.precisions_cholesky_[best_k:best_k+1], gmm_unnormalized.covariance_type).ravel()
            # Entropy
            best_entropy = 0#np.mean(best_particles_loglikelihood)
            # Consistency
            # Standard Deviation
            # RMSE
            #best_RMSE = np.sqrt(np.mean((best_samples - X_truth)**2, axis=0))
    #K = K.astype(int)
    if h is None:
        valid_rows = np.where(cloud_is_active[:, 0])[0]
        avg_ob_ob_likeli = np.full((cloud_is_active.shape[0], cloud_is_active.shape[0]), np.nan)
        avg_ob_ob_weight_loglikeli = np.full((cloud_is_active.shape[0], cloud_is_active.shape[0]), np.nan)
        avg_cross_entropy = np.full((cloud_is_active.shape[0], cloud_is_active.shape[0]), np.nan)
        KL = np.full((cloud_is_active.shape[0], cloud_is_active.shape[0], 3), np.nan)
        JS = np.full((cloud_is_active.shape[0], cloud_is_active.shape[0]), np.nan)
        JS_marginal = np.full((cloud_is_active.shape[0], cloud_is_active.shape[0], 6), np.nan)
        ill_conditioned = np.full((cloud_is_active.shape[0], cloud_is_active.shape[0]), False)
        # TODO: Check next line
        #for idx, ob_1 in enumerate(valid_rows[0]):
        for ob_1 in range(1):
            X_cloud_1 = cf.Topo2Synodic(np.copy(X_cloud[ob_1, 0]), t_int, obs_lat[ob_1], obs_lon[ob_1], obs_el[ob_1], normalization_quantities)
            for ob_2 in valid_rows[0 + 1:]:
                X_cloud_2 = cf.Topo2Synodic(np.copy(X_cloud[ob_2, 0]), t_int, obs_lat[ob_2], obs_lon[ob_2], obs_el[ob_2], normalization_quantities)
                # Compute Ob-Ob metrics for only the original clouds of each observer
                K_1 = int(K[ob_1, 0])
                K_2 = int(K[ob_2, 0])
                ob_ob_likeli = np.full((K_1, K_2), np.nan)
                for k1 in range(K_1):
                    for k2 in range(K_2):
                        cov = gmm_unnormalized[ob_1].covariances_[k1] + gmm_unnormalized[ob_2].covariances_[k2]
                        eps = max(1e-6 - la.eigvalsh(cov).min(), 0)
                        #gmm_unnormalized[ob_1].covariances_[k1] += np.eye(6)*eps
                        #gmm_unnormalized[ob_2].covariances_[k2] += np.eye(6)*eps
                        #cov = gmm_unnormalized[ob_1].covariances_[k1] + gmm_unnormalized[ob_2].covariances_[k2]
                        try:
                            ob_ob_likeli[k1, k2] = sci.stats.multivariate_normal.logpdf(gmm_unnormalized[ob_2].means_[k2], 
                                                                                     mean=gmm_unnormalized[ob_1].means_[k1], 
                                                                                     cov=cov + np.eye(6)*eps)
                        except:
                            ill_conditioned[ob_1, ob_2] = True
                avg_ob_ob_likeli[ob_1, ob_2] = -np.mean(ob_ob_likeli)
                avg_ob_ob_weight_loglikeli[ob_1, ob_2] = -np.sum(gmm_unnormalized[ob_1].weights_[:, None] * ob_ob_likeli * gmm_unnormalized[ob_2].weights_[None, :])
                avg_cross_entropy[ob_1, ob_2] = -(gmm_unnormalized[ob_1].score(X_cloud_2) + gmm_unnormalized[ob_2].score(X_cloud_1)) / 2
                KL[ob_1, ob_2, 0] = -gmm_unnormalized[ob_2].score(X_cloud_1) + gmm_unnormalized[ob_1].score(X_cloud_1) 
                KL[ob_1, ob_2, 1] = -gmm_unnormalized[ob_1].score(X_cloud_2) + gmm_unnormalized[ob_2].score(X_cloud_2)
                KL[ob_1, ob_2, 2] = 0.5*(KL[ob_1, ob_2, 0] + KL[ob_1, ob_2, 1])
                JS[ob_1, ob_2] = js_divergence(gmm_unnormalized[ob_1], gmm_unnormalized[ob_2], X_cloud_1, X_cloud_2)
                for d in range(6):
                    JS_marginal[ob_1, ob_2, d] = sci.spatial.distance.jensenshannon(X_cloud_1[:, d], X_cloud_2[:, d])
    else:
        avg_ob_ob_likeli = None
        avg_ob_ob_weight_loglikeli = None
        avg_cross_entropy = None
        KL = None
        JS = None
        JS_marginal = None
        ill_conditioned = None
    
    return gmm_truth_loglikelihood, gmm_entropy, gmm_RMSE, best_truth_loglikelihood, best_entropy, K, X_cloud.shape[0], AIC, avg_ob_ob_likeli, avg_ob_ob_weight_loglikeli, avg_cross_entropy, KL, JS, JS_marginal, ill_conditioned


def fusionMethods(cloud_1, cloud_2, fusion_type, K, rng):
    if fusion_type == "Weight Update Algorithm":
        # Actually computing fused cloud 2 is not necessary due to symmetry of Weight Update Algorithm
        fused_cloud_1 = weightUpdateAlgorithm(cloud_1, cloud_2, K, rng)
        #fused_cloud_2 = weightUpdateAlgorithm(cloud_2, cloud_1, K, rng)
        fused_cloud_2 = None
    # Resample, probably unnecessary but doesn't hurt
    _, fused_cloud_means_1, fused_cloud_covs_1, fused_cloud_weights_1, K_1 = cluster(fused_cloud_1, K, rng, None, None)
    fused_cloud_1, _ = drawFrom2(fused_cloud_weights_1, fused_cloud_means_1, fused_cloud_covs_1, cloud_1.shape[0], rng)
    
    #_, fused_cloud_means_2, fused_cloud_covs_2, fused_cloud_weights_2, K_2 = cluster(fused_cloud_2, K, rng, None, None)
    #fused_cloud_2, _ = drawFrom2(fused_cloud_weights_2, fused_cloud_means_2, fused_cloud_covs_2, cloud_2.shape[0], rng)
    return fused_cloud_1, fused_cloud_2


def weightUpdateAlgorithm(cloud_1, cloud_2, K, rng):
    
    def kalmanFusion(mu_1, cov_1, mu_2, cov_2):
        inv_cov = la.inv(cov_1 + cov_2)
        post_mu = mu_1 + cov_1 @ inv_cov @ (mu_2 - mu_1)
        post_cov = cov_1 - cov_1 @ inv_cov @ cov_1
        post_cov = (post_cov + post_cov.T) / 2
        return post_mu, post_cov
    
    num_particles = cloud_1.shape[0]
    _, means_1, covs_1, prior_weights_1, K_1 = cluster(cloud_1, K, rng, None, None)
    _, means_2, covs_2, prior_weights_2, K_2 = cluster(cloud_2, K, rng, None, None)
    print(f"K_1, K_2 = {K_1}, {K_2}")
    
    num_tries = 0
    while False:
        try:
            num_tries += 1
            X_norm, mu, std = standardize(cloud_1)
            gmm_normalized, _, K_1 = fitGMM(np.copy(X_norm), rng, K_upper = K)
            gmm_unnormalized_temp_1 = unstandardize(gmm_normalized, mu, std)
            
            X_norm, mu, std = standardize(cloud_2)
            gmm_normalized, _, K_2 = fitGMM(np.copy(X_norm), rng, K_upper = K)
            gmm_unnormalized_temp_2 = unstandardize(gmm_normalized, mu, std)
            
            means_1 = gmm_unnormalized_temp_1.means_; covs_1 = gmm_unnormalized_temp_1.covariances_; prior_weights_1 = gmm_unnormalized_temp_1.weights_;
            means_2 = gmm_unnormalized_temp_2.means_; covs_2 = gmm_unnormalized_temp_2.covariances_; prior_weights_2 = gmm_unnormalized_temp_2.weights_;
            
            post_weights = np.full((K_1, K_2), np.nan)
            likelihoods = np.full((K_1, K_2), np.nan)

            for k1 in range(K_1):
                for k2 in range(K_2):
                    likelihoods[k1, k2] = sci.stats.multivariate_normal.pdf(means_2[k2], mean=means_1[k1], cov=covs_2[k2] + covs_1[k1])
            break
        except Exception as error:
            #print("Error: ", error)
            pass
    print(num_tries)
    
    post_weights = np.full((K_1, K_2), np.nan)
    likelihoods = np.full((K_1, K_2), np.nan)

    for k1 in range(K_1):
        for k2 in range(K_2):
            likelihoods[k1, k2] = sci.stats.multivariate_normal.pdf(means_2[k2], mean=means_1[k1], cov=covs_2[k2] + covs_1[k1])
    
    post_weights = prior_weights_1[:, None] * likelihoods * prior_weights_2[None, :]
    post_weights /= np.sum(post_weights)
    fused_cloud = []
    
    for k1 in range(K_1):
        for k2 in range(K_2):
            post_mean, post_cov = kalmanFusion(means_1[k1], covs_1[k1], means_2[k2], covs_2[k2])
            fused_cloud_temp = rng.multivariate_normal(post_mean, post_cov, int(post_weights[k1, k2]*num_particles)).astype(np.float64)
            fused_cloud.append(fused_cloud_temp)
    
    fused_cloud = np.vstack(fused_cloud)
    return fused_cloud


def main(MC_idx):
    success = False
    rng = np.random.default_rng(MC_idx)
    W = 6 # Parallelization factor
    save_loc = Path("D:/PythonProjects/EDP/PGM/ParticleFusionTest/12_15_25_meeting/Matlab2Python/Test12/pyMC2_" + str(MC_idx))
    load_loc = Path("D:/PythonProjects/EDP/PGM/ParticleFusionTest/12_15_25_meeting/Matlab2Python/Test12/OrbitData/Agent")
    load_loc2 = Path("D:/PythonProjects/EDP/PGM/ParticleFusionTest/12_15_25_meeting/Matlab2Python/Test12")
    dynamics = "CR3BP";
    if (dynamics == "CR3BP"):
        dist2km = 384400 # Kilometers per non-dimensionalized distance
        time2hr = 4.342*24 # Hours per non-dimensionalized time
        vel2kms = dist2km/(time2hr*60*60) # Kms per non-dimensionalized velocity
        normalization_quantities = {"dist2km": dist2km,
                                    "vel2kms": vel2kms,
                                    "time2hr": time2hr,
                                    "mu": 1.2150582e-2}
        #dynamics_model = @(t, x) Dynamics.cr3bp_dyn(t, x, normalization_quantities.mu);
    
    cluster_by = "FullState"
    Kn = 14 # Number of clusters (original)
    K = np.tile(Kn, (6, 6)) # Number of clusters (changeable)
    Kmax = 14 # Maximum number of clusters (Kmax = 1 for EnKF)

    plot_IOD = False
    plot_indv_clouds = [True, False] # Not recommended unless debugging
    plot_cross_observers = False # Recommended to see observers' original clouds plotted on same figure
    save_MC_metrics = True # Recommended
    
    num_IOD_particles = 15000
    total_num_agents = 2 # Expected number of agents
    num_active_obs = 0
    Lp = np.array([[15000], [15000]], dtype=int)
    num_msmt_for_IOD = np.array([[10], [10]], dtype=int)
    ts_to_perform_IOD = np.array([[0], [20]], dtype=int)
    plot_combined_clouds = np.array([[True], [False], [False], [False]], dtype=bool) # Plot all clouds of single observer on same figure. Recommended for observer 1
    num_clouds_per_agent = np.ones(total_num_agents, dtype=int)

    #partial_tsa = [sio.loadmat(f"{load_loc}{i+1}a/partial_ts.mat")["partial_ts"] for i in range(total_num_agents)]
    #full_tsa = [sio.loadmat(f"{load_loc}{i+1}a/full_ts.mat")["full_ts"] for i in range(total_num_agents)]
    #full_vtsa = [sio.loadmat(f"{load_loc}{i+1}a/full_vts.mat")["full_vts"] for i in range(total_num_agents)]
    
    
    partial_ts = [np.load(f"{load_loc}{i+1}/partial_ts.npy") for i in range(total_num_agents)] #[sio.loadmat(f"{load_loc}{i+1}/partial_ts.mat")["partial_ts"] for i in range(total_num_agents)]
    full_ts = [np.load(f"{load_loc}{i+1}/full_ts.npy") for i in range(total_num_agents)] #[sio.loadmat(f"{load_loc}{i+1}/full_ts.mat")["full_ts"] for i in range(total_num_agents)]
    full_vts = [np.load(f"{load_loc}{i+1}/full_vts.npy") for i in range(total_num_agents)] #[sio.loadmat(f"{load_loc}{i+1}/full_vts.mat")["full_vts"] for i in range(total_num_agents)]
    combined_msmt_data, combined_state_data, all_timesteps = combineMsmts(full_ts, full_vts, partial_ts)
    num_timesteps = all_timesteps.shape[0]
    
    
    # fusion_information contains various info formatted as fusion #: [observer index, observer index, fuse?, timestep to fuse]
    fusion_information = [[0, 1, True, 45, "Weight Update Algorithm", 17],
                          [0, 1, True, 45, "Weight Update Algorithm", 25],
                          [0, 1, True, 45, "Weight Update Algorithm", 33],
                          [0, 1, True, 45, "Weight Update Algorithm", 41]]#[[0, 1, False, 45], [0, 1, False, 55], [0, 1, False, 65], [0, 1, False, 75]] 
    cloud_names = [[f"Original Obs: {ob}. IOD: {all_timesteps[ts_to_perform_IOD[ob]][0]*normalization_quantities['time2hr']:.0f} hrs"] for ob in range(total_num_agents-1)]
    cloud_names.append(["Baseline Obs"])
    linestyle = [["-"] for ob in range(total_num_agents)]
    fusion_types = ["Original", "Weight Update"]
    num_new_clouds_per_agent = 1

    # College Station
    obs_lat = np.tile(30.618963, (6))
    obs_lon = np.tile(-96.339214, (6))
    obs_el = np.tile(103.8, (6))
    for ob in range(total_num_agents):
        ensureDirExists(save_loc / f"Observer{ob}" / "Topo" / "Combined")
        ensureDirExists(save_loc / f"Observer{ob}" / "Synodic" / "Combined")
        ensureDirExists(save_loc / f"Observer{ob}" / "ECI" / "Combined")
    ensureDirExists(save_loc / "CrossOb" / "Synodic")
    ensureDirExists(save_loc / "CrossOb" / "ECI")
    
    h = lambda x: np.array([np.arctan2(x[:, 1],x[:, 0]),
                            np.pi/2 - np.arccos(x[:, 2]/la.norm(x[:, :3], axis=1))]).T
    
    theta_f = 1.5 # Arc-seconds of error covariance
    range_f = np.tile(0.25, (6))
    R_weight = np.array([[theta_f*np.pi/648000, 0], [0, theta_f*np.pi/648000]])**2
    
    enforce_bounds = False
    if dynamics == "CR3BP":
        low_lim = 2*42164 # Two times the GEO Distance
        up_lim = 550000
        vel_lim = 42 # Escape velocity of the solar system
    
    
    total_num_clouds = 1 + num_new_clouds_per_agent * sum(row[2] for row in fusion_information)
    metrics = {"likelihood_state_weighted": (total_num_agents, total_num_clouds, num_timesteps),
               "likelihood_state_best": (total_num_agents, total_num_clouds, num_timesteps),
               "likelihood_msmt_weighted": (total_num_agents, total_num_clouds, num_timesteps),
               "likelihood_msmt_best": (total_num_agents, total_num_clouds, num_timesteps),
               "entropy_state_weighted": (total_num_agents, total_num_clouds, num_timesteps),
               "entropy_state_best": (total_num_agents, total_num_clouds, num_timesteps),
               "entropy_msmt_weighted": (total_num_agents, total_num_clouds, num_timesteps),
               "entropy_msmt_best": (total_num_agents, total_num_clouds, num_timesteps),
               "NEES": (total_num_agents, total_num_clouds, num_timesteps),
               #"MC_std_dev": (total_num_clouds, 1),
               "MC_consistency": (total_num_agents, total_num_clouds, num_timesteps),
               "num_cluster": (total_num_agents, total_num_clouds, num_timesteps),
               "num_particles": (total_num_agents, total_num_clouds, num_timesteps),
               #"ent1": (total_num_clouds, 6),
               "RMSE": (total_num_agents, total_num_clouds, num_timesteps, 6),
               "std_dev": (total_num_agents, total_num_clouds, num_timesteps, 6),
               #"mat_weight_metric": (total_num_clouds, 3),
               #"orig_weight_metric": (total_num_clouds, 3),
               #"cross_observer_norm": (1, 1),
               "AIC_state": (total_num_agents, total_num_clouds, num_timesteps), 
               "AIC_msmt": (total_num_agents, total_num_clouds, num_timesteps),
               "avg_ob_ob_likeli": (total_num_agents, total_num_agents, num_timesteps),
               "avg_ob_ob_weight_loglikeli": (total_num_agents, total_num_agents, num_timesteps),
               "avg_cross_entropy": (total_num_agents, total_num_agents, num_timesteps),
               "KL": (total_num_agents, total_num_agents, num_timesteps, 3),
               "JS": (total_num_agents, total_num_agents, num_timesteps),
               "JS_marginal": (total_num_agents, total_num_agents, num_timesteps, 6),
               "ill_conditioned": (total_num_agents, total_num_agents, num_timesteps)}
    
    metrics = {key: np.full(shape, np.nan) for key, shape in metrics.items()}
    
    # TODO: Prob not necessary
    fig_num = 1
    
    #X_truth = np.empty((total_num_agents), dtype=object)
    X_truth = np.full((total_num_agents, 1, 6), np.nan)
    #Xm_cloud = np.empty((total_num_agents, total_num_clouds), dtype=object)
    Xm_cloud = np.full((total_num_agents, total_num_clouds, Lp[ob, 0], 6), np.nan)
    #Xp_cloud = np.empty((total_num_agents, total_num_clouds), dtype=object)
    #Xp_cloudp = np.empty((total_num_agents, total_num_clouds), dtype=object)
    Xp_cloudp = np.full((total_num_agents, total_num_clouds, Lp[ob, 0], 6), np.nan)
    cloud_is_active = np.full((total_num_agents, total_num_clouds), False)
    # TODO: Check if should be -1
    for ts in range(num_timesteps-1):
        t_prev = all_timesteps[ts]
        #%% IOD
        for ob, is_active in enumerate(~(cloud_is_active[:, 0])):
            num_msmts_check = sum(combined_msmt_data[ts_to_perform_IOD[ob, 0]:ts+1, 1, ob]) >= num_msmt_for_IOD[ob, 0]
            if is_active and num_msmts_check:
                print(f"Performing IOD on object: {ob}")
                order_of_fit = 4
                X0_cloud_temp = np.zeros((num_IOD_particles, 6));
                
                starttime = time.time()
                for particle in range(num_IOD_particles):
                    X0_cloud_temp[particle, :] = stateEstCloud(num_msmt_for_IOD[ob, 0], 
                                                  ts, 
                                                  order_of_fit, 
                                                  theta_f, 
                                                  range_f[ob], 
                                                  combined_msmt_data[ts_to_perform_IOD[ob, 0]:ts+1, :, ob], 
                                                  low_lim, 
                                                  up_lim,
                                                  normalization_quantities,
                                                  rng)
                endtime = time.time()
                print(endtime - starttime)
                #X0_cloud_temp = sci.io.loadmat(load_loc2/"X0cloud_temp.mat")
                #X0_cloud_temp = X0_cloud_temp['X0cloud_temp']
                if enforce_bounds:
                    '''
                    X0_cloud = enforceCislunarBounds(X0_cloud_temp,
                                                    t_prev,
                                                    obs_lat[ob],
                                                    obs_lon[ob],
                                                    normalization_quantities,
                                                    low_lim,
                                                    up_lim,
                                                    vel_lim)
                    '''
                else:
                    X0_cloud = X0_cloud_temp
                # TODO: Remove most legends
                # TODO: fix plots
                X0_truth = combined_state_data[ts, 1:, ob].reshape(1, -1);
                
                plot.plotStateSpace(X0_cloud,
                                    X0_truth, 
                                    1,
                                    np.ones(X0_cloud.shape[0], dtype=int), 
                                    normalization_quantities, 
                                    f"Timestep: {t_prev*normalization_quantities['time2hr']:3.2f} Hours Obs: {ob}",
                                    save_loc/f"Observer{ob}"/"Topo"/"iodCloud.png")
                if dynamics == "CR3BP":
                    plotting_cloud = cf.Topo2Synodic(np.copy(X0_cloud), t_prev, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
                    plotting_truth = cf.Topo2Synodic(np.copy(X0_truth), t_prev, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
                    plot.plotStateSpace(plotting_cloud,
                                        plotting_truth, 
                                        1,
                                        np.ones(X0_cloud.shape[0], dtype=int), 
                                        normalization_quantities, 
                                        f"Timestep: {t_prev*normalization_quantities['time2hr']:3.2f} Hours Obs: {ob}",
                                        save_loc/f"Observer{ob}"/"Synodic"/"iodCloud.png")
                
                cloud_is_active[ob, 0] = True
                #active_mask.sort()
                num_active_obs = np.sum(cloud_is_active[:, 0])
                # TODO: Check h against matlab h
                Xp_cloudp[ob, 0] = X0_cloud.copy()
                X_truth[ob] = X0_truth.copy()
                print(X0_truth[0, :3]*normalization_quantities['dist2km'])
                print(X_truth[0, 0, :3]*normalization_quantities['dist2km'])
                
        #%% Fusion
        if num_active_obs >= 2:
            for row, (fuse_id_1, fuse_id_2, fuse, fusion_timestep, fusion_type, K_upper) in enumerate(fusion_information):
                if cloud_is_active[fuse_id_1, 0] and cloud_is_active[fuse_id_2, 0] and fuse and fusion_timestep == ts:
                    cloud_to_fuse_1 = cf.Topo2Synodic(np.copy(Xp_cloudp[fuse_id_1, 0]), t_prev, obs_lat[fuse_id_1], obs_lon[fuse_id_1], obs_el[fuse_id_1], normalization_quantities)
                    cloud_to_fuse_2 = cf.Topo2Synodic(np.copy(Xp_cloudp[fuse_id_2, 0]), t_prev, obs_lat[fuse_id_2], obs_lon[fuse_id_2], obs_el[fuse_id_2], normalization_quantities)
                    
                    fused_cloud_1, _ = fusionMethods(cloud_to_fuse_1, cloud_to_fuse_2, fusion_type, K_upper, rng)
                    
                    Xp_cloudp[fuse_id_1, num_clouds_per_agent[fuse_id_1]] = cf.Synodic2Topo(fused_cloud_1, t_prev, obs_lat[fuse_id_1], obs_lon[fuse_id_1], obs_el[fuse_id_1], normalization_quantities)
                    cloud_names[fuse_id_1].append(f"Fused Obs: {fuse_id_1} & {fuse_id_2} @ {t_prev*normalization_quantities['time2hr']:.0f} hrs")
                    linestyle[fuse_id_1].append('--')
                    cloud_is_active[fuse_id_1, num_clouds_per_agent[fuse_id_1]] = True
                    
                    fusion_information[row][2] = False
                    num_clouds_per_agent[fuse_id_1] += num_new_clouds_per_agent
                    
                    for ob in [fuse_id_1, fuse_id_2]:
                        if plot_combined_clouds[ob]:
                            plotting_cloud = np.vstack(Xp_cloudp[ob, :num_clouds_per_agent[ob]])
                            plotting_truth = np.copy(X_truth[ob])
                            plotting_idx = np.repeat(np.arange(num_clouds_per_agent[ob])+1, Lp[ob, 0])
                            plot.plotStateSpace(plotting_cloud,
                                                plotting_truth, 
                                                num_clouds_per_agent[ob],
                                                plotting_idx, 
                                                normalization_quantities, 
                                                f"Timestep: {t_prev*normalization_quantities['time2hr']:3.2f} Hours Fusion Obs: {fuse_id_1}, {fuse_id_2}",
                                                save_loc/f"Observer{ob}"/"Topo"/"Combined"/f"Timestep_{ts}_0B_combined.png",
                                                plot_cross_observers = True,
                                                cloud_names = cloud_names[ob])
                            plot.plotMsmtSpace(plotting_cloud,
                                                plotting_truth,
                                                np.array([np.nan, np.nan]),
                                                h,
                                                None,
                                                num_clouds_per_agent[ob],
                                                plotting_idx,
                                                f"Az-El Timestep: {t_prev*normalization_quantities['time2hr']:3.2f} Hours Fusion Obs: {fuse_id_1}, {fuse_id_2}",
                                                save_loc/f"Observer{ob}"/"Topo"/"Combined"/f"Timestep_{ts}_0C_cloud_combined.png",
                                                False,
                                                plot_cross_observers=True,
                                                cloud_names = cloud_names[ob])
                            
                            plotting_cloud = cf.Topo2Synodic(np.copy(plotting_cloud), t_prev, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
                            plotting_truth = cf.Topo2Synodic(np.copy(plotting_truth), t_prev, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
                            plot.plotStateSpace(plotting_cloud,
                                                plotting_truth, 
                                                num_clouds_per_agent[ob],
                                                plotting_idx, 
                                                normalization_quantities, 
                                                f"Timestep: {t_prev*normalization_quantities['time2hr']:3.2f} Hours Fusion Obs: {fuse_id_1}, {fuse_id_2}",
                                                save_loc/f"Observer{ob}"/"Synodic"/"Combined"/f"Timestep_{ts}_0B_combined.png",
                                                plot_cross_observers = True,
                                                cloud_names = cloud_names[ob])
        
                    
        #%% Calculate Metrics
        if np.any(cloud_is_active):
            metrics["likelihood_state_weighted"][:, :, ts],  metrics["entropy_state_weighted"][:, :, ts], metrics["RMSE"][:, :, ts, :], metrics["likelihood_state_best"][:, :, ts], metrics["entropy_state_best"][:, :, ts], metrics["num_cluster"][:, :, ts], metrics["num_particles"][:, :, ts], metrics["AIC_state"][:, :, ts], metrics["avg_ob_ob_likeli"][:, :, ts], metrics["avg_ob_ob_weight_loglikeli"][:, :, ts], metrics["avg_cross_entropy"][:, :, ts], metrics["KL"][:, :, ts, :], metrics["JS"][:, :, ts], metrics["JS_marginal"][:, :, ts, :], metrics["ill_conditioned"][:, :, ts] = getCloudMetrics(np.copy(Xp_cloudp), np.copy(X_truth), t_prev, cloud_is_active, K, obs_lat, obs_lon, obs_el, normalization_quantities, rng, ts+1, load_loc2)
            metrics["likelihood_msmt_weighted"][:, :, ts],  metrics["entropy_msmt_weighted"][:, :, ts], _, metrics["likelihood_msmt_best"][:, :, ts], metrics["entropy_msmt_best"][:, :, ts], _, _, metrics["AIC_msmt"][:, :, ts], _, _, _, _, _, _, _ = getCloudMetrics(np.copy(Xp_cloudp), np.copy(X_truth), t_prev, cloud_is_active, K, obs_lat, obs_lon, obs_el, normalization_quantities, rng, ts+1, load_loc2, h=h)
        
        # Check whether object is still within each cloud, return success = False if not
        for ob, cloud in zip(*np.where(cloud_is_active)):
            if metrics["likelihood_state_weighted"][ob, cloud, ts] <= -100:
                print(f"MC run failed at Ob: {ob}, Cloud: {cloud}, Time: {all_timesteps[ts+1]*normalization_quantities['time2hr']}")
                return success, ob, cloud, ts
        
        
        #%% Propagation Step
        t_prior = all_timesteps[ts+1]
        interval = t_prior - t_prev
        #Xm_cloud = np.empty((total_num_agents, total_num_clouds), dtype=object)
        Xm_cloud = np.full((total_num_agents, total_num_clouds, Lp[ob, 0], 6), np.nan)
        for ob in np.where(cloud_is_active[:, 0])[0]:
            for cloud in np.where(cloud_is_active[ob])[0]:
                #cloud_for_matlab = np.copy(Xp_cloudp[ob, cloud])
                #sci.io.savemat(load_loc2/f"cloud_for_matlab_{ts+1}.mat", {"Xp_cloudp_temp": cloud_for_matlab})
                
                Xm_cloud_temp = propagate(Xp_cloudp[ob, cloud], t_prev, interval, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
                #Xm_cloud_temp = sci.io.loadmat(load_loc2/f"propagate{ts+1}.mat")
                #Xm_cloud_temp = Xm_cloud_temp['Xm_cloud_tmp']
                if enforce_bounds:
                    print()
                    #Xm_cloud[ob][cloud] = enforceCislunarBounds(Xm_cloud_temp, t_prior, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
                else:
                    # TODO: Check if Xm_cloud changes if temp gets changed/reset
                    Xm_cloud[ob, cloud] = Xm_cloud_temp.copy()
            #if combined_state_data[ts+1, 1:, ob].reshape(1, -1) is not np.nan:
            X_truth[ob, 0, :] = combined_state_data[ts+1, 1:, 1].reshape(1, -1)#propagate(X_truth[ob], t_prev, interval, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
            #else:
            #X_truth[ob] = propagate(X_truth[ob], t_prev, interval, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
        
        print(f"Timestamp: {t_prior*normalization_quantities['time2hr']:3.2f}")
        if abs(t_prior*normalization_quantities['time2hr']-42) <= 1e-6:
            print()
        
        #%% Update Step
        #Xp_cloudp = np.empty((total_num_agents, total_num_clouds), dtype=object)
        Xp_cloudp = np.full((total_num_agents, total_num_clouds, Lp[ob, 0], 6), np.nan)
        # TODO: Make sure empty K rows are not impacting later code
        msmt_mask = combined_msmt_data[ts+1, 1, :]
        msmt = np.full((total_num_agents, 2), np.nan)
        zt_cluster_likelihood = np.full((total_num_agents, total_num_clouds, Kmax), np.nan)#np.empty((total_num_agents, total_num_clouds), dtype=object)
        
        cluster_prior_idx = np.full((total_num_agents, total_num_clouds, np.max(Lp)), np.nan)#np.empty((total_num_agents, total_num_clouds), dtype=object)
        cluster_prior_means = np.full((total_num_agents, total_num_clouds, Kmax, 6), np.nan)#np.empty((total_num_agents, total_num_clouds), dtype=object)
        cluster_prior_covs = np.full((total_num_agents, total_num_clouds, Kmax, 6, 6), np.nan)#np.empty((total_num_agents, total_num_clouds), dtype=object)
        cluster_prior_weights = np.full((total_num_agents, total_num_clouds, Kmax), np.nan)#np.empty((total_num_agents, total_num_clouds), dtype=object)
        
        cluster_post_idx = np.full((total_num_agents, total_num_clouds, np.max(Lp)), np.nan)#np.empty((total_num_agents, total_num_clouds), dtype=object)
        cluster_post_means = np.full((total_num_agents, total_num_clouds, Kmax, 6), np.nan)#np.empty((total_num_agents, total_num_clouds), dtype=object)
        cluster_post_covs = np.full((total_num_agents, total_num_clouds, Kmax, 6, 6), np.nan)#np.empty((total_num_agents, total_num_clouds), dtype=object)
        cluster_post_weights = np.full((total_num_agents, total_num_clouds, Kmax), np.nan)#np.empty((total_num_agents, total_num_clouds), dtype=object)
        
        for ob, cloud in zip(*np.where(cloud_is_active)):
            if msmt_mask[ob] == True:
                # TODO: Add next line back in
                #if K[ob, cloud] == 1:
                K[ob, cloud] = Kmax
                # Generate noisy (angles-only) measurement
                msmt[ob, :] = getNoisyMeas(X_truth[ob], R_weight, h, rng)
                
                # Calc prior statistics
                cluster_prior_idx[ob, cloud], cluster_prior_means_temp, cluster_prior_covs_temp, cluster_prior_weights_temp, K[ob, cloud] = cluster(Xm_cloud[ob, cloud], K[ob, cloud], rng, ts+2, load_loc2)
                cluster_prior_means[ob, cloud, :K[ob, cloud]] = cluster_prior_means_temp
                cluster_prior_covs[ob, cloud, :K[ob, cloud]] = cluster_prior_covs_temp
                cluster_prior_weights[ob, cloud, :K[ob, cloud]] = cluster_prior_weights_temp
                
                # Calc posterior statistics
                for k in range(K[ob, cloud]):
                    cluster_points = Xm_cloud[ob, cloud, cluster_prior_idx[ob, cloud] == k]
                    # After vectorizing kalmanUpdate(), means/Covs may be off by 1e-12
                    #cluster_post_means[ob, cloud, k], cluster_post_covs[ob, cloud, k] = kalmanUpdate(msmt[ob], cluster_points, R_weight, cluster_prior_means[ob, cloud, k], cluster_prior_covs[ob, cloud, k], h)
                    cluster_post_means[ob, cloud, k], cluster_post_covs[ob, cloud, k] = kalmanUpdate2(msmt[ob], cluster_points, R_weight, cluster_prior_means[ob, cloud, k], cluster_prior_covs[ob, cloud, k], h)
                    cluster_post_covs[ob, cloud, k] = (cluster_post_covs[ob, cloud, k] + cluster_post_covs[ob, cloud, k].T)/2
                cluster_post_weights[ob, cloud, :K[ob, cloud]], zt_cluster_likelihood[ob, cloud, :K[ob, cloud]] = weightUpdate(cluster_prior_weights[ob, cloud, :K[ob, cloud]], Xm_cloud[ob, cloud], cluster_prior_idx[ob, cloud], msmt[ob], R_weight, h)
                
                # Resample
                #Xp_cloudp[ob, cloud] = np.zeros(Xm_cloud[ob, cloud].shape)
                #for particle in range(Lp[ob, 0]):
                #    Xp_cloudp[ob, cloud][particle, :], cluster_post_idx[ob, cloud, particle] = drawFrom(cluster_post_weights[ob, cloud], cluster_post_means[ob, cloud], cluster_post_covs[ob, cloud], Lp[ob, 0], rng)#drawFrom(cluster_post_weights[ob, cloud], cluster_post_means[ob, cloud], cluster_post_covs[ob, cloud], Lp[ob, 0])
                Xp_cloudp[ob, cloud], cluster_post_idx[ob, cloud] = drawFrom2(cluster_post_weights[ob, cloud, :K[ob, cloud]], cluster_post_means[ob, cloud, :K[ob, cloud]], cluster_post_covs[ob, cloud, :K[ob, cloud]], Lp[ob, 0], rng)
                #Xp_cloudp_temp = sci.io.loadmat(load_loc2/f"drawFrom{ts+2}.mat")
                #Xp_cloudp[ob, cloud] = Xp_cloudp_temp['Xp_cloudp_temp']
                #cluster_post_idx_temp = sci.io.loadmat(load_loc2/f"drawFromIdx{ts+2}.mat")
                #cluster_post_idx[ob, cloud] = cluster_post_idx_temp['c_id_temp'].flatten() - 1
                # Add 1 to each cluster idx (plotting purposes)
                cluster_prior_idx[ob, cloud] += 1
                cluster_post_idx[ob, cloud] += 1
            else:
                # Resample
                #cloud_for_matlab = np.copy(Xm_cloud[ob, cloud])
                #sci.io.savemat(f"cloud_for_matlab_{ts+2}.mat", {"Xm_cloud_temp": cloud_for_matlab})
                Xp_cloudp[ob, cloud] = np.copy(Xm_cloud[ob, cloud])
                cluster_post_idx[ob, cloud] = np.ones((Xp_cloudp[ob, cloud].shape[0]))
            
        #metric_cloud = cf.Topo2Synodic(np.copy(Xp_cloudp[ob, cloud]), t_prev, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
        #metric_truth = cf.Topo2Synodic(np.copy(X_truth[ob]), t_prev, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
        #metrics["likelihood_state_weighted"][ob, cloud, ts+1],  metrics["entropy_state_weighted"][ob, cloud, ts+1], metrics["RMSE"][ob, cloud, ts+1, :], metrics["likelihood_state_best"][ob, cloud, ts+1], metrics["entropy_state_best"][ob, cloud, ts+1], metrics["num_cluster"][ob, cloud, ts+1], metrics["num_particles"][ob, cloud, ts+1], metrics["AIC_state"][ob, cloud, ts+1], metrics["avg_ob_ob_likeli"][ob, cloud, ts+1], metrics["avg_ob_ob_weight_loglikeli"][ob, cloud, ts+1], metrics["avg_cross_entropy"][ob, cloud, ts+1] = getCloudMetrics(np.copy(Xp_cloudp), np.copy(X_truth), t_prior, cloud_is_active, K[ob, cloud], obs_lat, obs_lon, obs_el, normalization_quantities, rng, ts+2, load_loc2)
        #metrics["likelihood_msmt_weighted"][ob, cloud, ts+1],  metrics["entropy_msmt_weighted"][ob, cloud, ts+1], _, metrics["likelihood_msmt_best"][ob, cloud, ts+1], metrics["entropy_msmt_best"][ob, cloud, ts+1], _, _, metrics["AIC_msmt"][ob, cloud, ts+1], _, _, _ = getCloudMetrics(np.copy(Xp_cloudp), np.copy(X_truth), t_prior, cloud_is_active, K[ob, cloud], obs_lat, obs_lon, obs_el, normalization_quantities, rng, ts+2, load_loc2, h=h)
        
        #%% Plot Priors
        for ob in np.where(cloud_is_active[:, 0])[0]:
            if msmt_mask[ob] == True:
                # Plot a single cloud of observer ob
                for cloud in np.where(cloud_is_active[ob])[0]:
                    if plot_indv_clouds[ob]:
                        plotting_cloud = np.copy(Xm_cloud[ob, cloud])
                        plotting_truth = np.copy(X_truth[ob])
                        plot.plotStateSpace(plotting_cloud,
                                            plotting_truth, 
                                            K[ob, cloud],
                                            cluster_prior_idx[ob, cloud], 
                                            normalization_quantities, 
                                            f"Timestep: {t_prior*normalization_quantities['time2hr']:3.2f} Hours (Prior) Obs: {ob}",
                                            save_loc/f"Observer{ob}"/"Topo"/f"Timestep_{ts+1}_1B_cloud_{cloud}.png")
                        plot.plotMsmtSpace(plotting_cloud,
                                            plotting_truth,
                                            msmt[ob],
                                            h,
                                            zt_cluster_likelihood[ob, cloud],
                                            K[ob, cloud],
                                            cluster_prior_idx[ob, cloud],
                                            f"Az-El Timestep: {t_prior*normalization_quantities['time2hr']:3.2f} Hours (Prior) Obs: {ob}",
                                            save_loc/f"Observer{ob}"/"Topo"/f"Timestep_{ts+1}_1C_cloud_{cloud}.png",
                                            msmt_mask[ob])
                        # TODO: Add ECI plots back in
                        plotting_cloud = cf.Topo2Synodic(np.copy(Xm_cloud[ob, cloud]), t_prior, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
                        plotting_truth = cf.Topo2Synodic(np.copy(X_truth[ob]), t_prior, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
                        plot.plotStateSpace(plotting_cloud,
                                            plotting_truth, 
                                            K[ob, cloud],
                                            cluster_prior_idx[ob, cloud], 
                                            normalization_quantities, 
                                            f"Timestep: {t_prior*normalization_quantities['time2hr']:3.2f} Hours (Prior) Obs: {ob}",
                                            save_loc/f"Observer{ob}"/"Synodic"/f"Timestep_{ts+1}_1B_cloud_{cloud}.png")
                
                # Plot all clouds of observer ob
                if plot_combined_clouds[ob] and np.sum(cloud_is_active[ob, :]) >= 2:
                    plotting_cloud = np.vstack(Xm_cloud[ob, :num_clouds_per_agent[ob]])
                    plotting_truth = np.copy(X_truth[ob])
                    plotting_idx = np.repeat(np.arange(num_clouds_per_agent[ob])+1, Lp[ob, 0])
                    plot.plotStateSpace(plotting_cloud,
                                        plotting_truth, 
                                        num_clouds_per_agent[ob],
                                        plotting_idx, 
                                        normalization_quantities, 
                                        f"Timestep: {t_prior*normalization_quantities['time2hr']:3.2f} Hours (Prior) Obs: {ob}",
                                        save_loc/f"Observer{ob}"/"Topo"/"Combined"/f"Timestep_{ts+1}_1B_combined.png",
                                        plot_cross_observers = True,
                                        cloud_names = cloud_names[ob])
                    plot.plotMsmtSpace(plotting_cloud,
                                        plotting_truth,
                                        msmt[ob],
                                        h,
                                        None,
                                        num_clouds_per_agent[ob],
                                        plotting_idx,
                                        f"Az-El Timestep: {t_prior*normalization_quantities['time2hr']:3.2f} Hours (Prior) Obs: {ob}",
                                        save_loc/f"Observer{ob}"/"Topo"/"Combined"/f"Timestep_{ts+1}_1C_cloud_combined.png",
                                        msmt_mask[ob],
                                        plot_cross_observers=True,
                                        cloud_names = cloud_names[ob])
                    
                    plotting_cloud = cf.Topo2Synodic(np.copy(plotting_cloud), t_prior, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
                    plotting_truth = cf.Topo2Synodic(np.copy(plotting_truth), t_prior, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
                    plot.plotStateSpace(plotting_cloud,
                                        plotting_truth, 
                                        num_clouds_per_agent[ob],
                                        plotting_idx, 
                                        normalization_quantities, 
                                        f"Timestep: {t_prior*normalization_quantities['time2hr']:3.2f} Hours (Prior) Obs: {ob}",
                                        save_loc/f"Observer{ob}"/"Synodic"/"Combined"/f"Timestep_{ts+1}_1B_combined.png",
                                        plot_cross_observers = True,
                                        cloud_names = cloud_names[ob])
        
        # Plot original clouds across all observers
        if num_active_obs >= 2 and plot_cross_observers:
            plotting_cloud = np.vstack(Xm_cloud[cloud_is_active[:, 0], 0])
            plotting_idx = np.repeat(np.asarray(np.where(cloud_is_active[:, 0]))+1, Lp[cloud_is_active[:, 0], 0])
            plotting_cloud = cf.Topo2Synodic(np.copy(plotting_cloud), t_prior, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
            plotting_truth = cf.Topo2Synodic(np.copy(X_truth[ob]), t_prior, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
            plot.plotStateSpace(plotting_cloud,
                                plotting_truth, 
                                total_num_agents,
                                plotting_idx, 
                                normalization_quantities, 
                                f"Timestep: {t_prior*normalization_quantities['time2hr']:3.2f} Hours (Prior)",
                                save_loc/"CrossOb"/"Synodic"/f"Timestep_{ts+1}_1D_cloud_{cloud}.png",
                                plot_cross_observers = True,
                                cloud_names = [row[0] for row in cloud_names])
            
        
        #%% Plot Posterior
        for ob in np.where(cloud_is_active[:, 0])[0]:
            # Plot a single cloud of observer ob
            for cloud in np.where(cloud_is_active[ob])[0]:
                if plot_indv_clouds[ob]:
                    plotting_cloud = np.copy(Xp_cloudp[ob, cloud])
                    plotting_truth = np.copy(X_truth[ob])
                    plot.plotStateSpace(plotting_cloud,
                                        plotting_truth, 
                                        K[ob, cloud],
                                        cluster_post_idx[ob, cloud], 
                                        normalization_quantities, 
                                        f"Timestep: {t_prior*normalization_quantities['time2hr']:3.2f} Hours (Post) Obs: {ob}",
                                        save_loc/f"Observer{ob}"/"Topo"/f"Timestep_{ts+1}_2B_cloud_{cloud}.png")

                    plot.plotMsmtSpace(plotting_cloud,
                                        plotting_truth,
                                        msmt[ob],
                                        h,
                                        zt_cluster_likelihood[ob, cloud],
                                        K[ob, cloud],
                                        cluster_post_idx[ob, cloud],
                                        f"Az-El Timestep: {t_prior*normalization_quantities['time2hr']:3.2f} Hours (Post) Obs: {ob}",
                                        save_loc/f"Observer{ob}"/"Topo"/f"Timestep_{ts+1}_2C_cloud_{cloud}.png",
                                        msmt_mask[ob])
                    # TODO: Add ECI plots back in
                    plotting_cloud = cf.Topo2Synodic(np.copy(Xp_cloudp[ob, cloud]), t_prior, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
                    plotting_truth = cf.Topo2Synodic(np.copy(X_truth[ob]), t_prior, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
                    plot.plotStateSpace(plotting_cloud,
                                        plotting_truth, 
                                        K[ob, cloud],
                                        cluster_post_idx[ob, cloud], 
                                        normalization_quantities, 
                                        f"Timestep: {t_prior*normalization_quantities['time2hr']:3.2f} Hours (Post) Obs: {ob}",
                                        save_loc/f"Observer{ob}"/"Synodic"/f"Timestep_{ts+1}_2B_cloud_{cloud}.png")
            
            # Plot all clouds of observer ob
            if plot_combined_clouds[ob] and np.sum(cloud_is_active[ob, :]) >= 2:
                plotting_cloud = np.vstack(Xp_cloudp[ob, :num_clouds_per_agent[ob]])
                plotting_truth = np.copy(X_truth[ob])
                plotting_idx = np.repeat(np.arange(num_clouds_per_agent[ob])+1, Lp[ob, 0])
                plot.plotStateSpace(plotting_cloud,
                                    plotting_truth, 
                                    num_clouds_per_agent[ob],
                                    plotting_idx, 
                                    normalization_quantities, 
                                    f"Timestep: {t_prior*normalization_quantities['time2hr']:3.2f} Hours (Post) Obs: {ob}",
                                    save_loc/f"Observer{ob}"/"Topo"/"Combined"/f"Timestep_{ts+1}_2B_combined.png",
                                    plot_cross_observers = True,
                                    cloud_names = cloud_names[ob])
                plot.plotMsmtSpace(plotting_cloud,
                                    plotting_truth,
                                    msmt[ob],
                                    h,
                                    None,
                                    num_clouds_per_agent[ob],
                                    plotting_idx,
                                    f"Az-El Timestep: {t_prior*normalization_quantities['time2hr']:3.2f} Hours (Post) Obs: {ob}",
                                    save_loc/f"Observer{ob}"/"Topo"/"Combined"/f"Timestep_{ts+1}_2C_cloud_combined.png",
                                    msmt_mask[ob],
                                    plot_cross_observers=True,
                                    cloud_names = cloud_names[ob])
                
                plotting_cloud = cf.Topo2Synodic(np.copy(plotting_cloud), t_prior, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
                plotting_truth = cf.Topo2Synodic(np.copy(plotting_truth), t_prior, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
                plot.plotStateSpace(plotting_cloud,
                                    plotting_truth, 
                                    num_clouds_per_agent[ob],
                                    plotting_idx, 
                                    normalization_quantities, 
                                    f"Timestep: {t_prior*normalization_quantities['time2hr']:3.2f} Hours (Post) Obs: {ob}",
                                    save_loc/f"Observer{ob}"/"Synodic"/"Combined"/f"Timestep_{ts+1}_2B_combined.png",
                                    plot_cross_observers = True,
                                    cloud_names = cloud_names[ob])
        
        # Plot original clouds across all observers
        if num_active_obs >= 2 and plot_cross_observers and np.any(msmt_mask == True):
            plotting_cloud = np.vstack(Xp_cloudp[cloud_is_active[:, 0], 0])
            plotting_idx = np.repeat(np.asarray(np.where(cloud_is_active[:, 0]))+1, Lp[cloud_is_active[:, 0], 0])
            plotting_cloud = cf.Topo2Synodic(np.copy(plotting_cloud), t_prior, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
            plotting_truth = cf.Topo2Synodic(np.copy(X_truth[ob]), t_prior, obs_lat[ob], obs_lon[ob], obs_el[ob], normalization_quantities)
            plot.plotStateSpace(plotting_cloud,
                                plotting_truth, 
                                total_num_agents,
                                plotting_idx, 
                                normalization_quantities, 
                                f"Timestep: {t_prior*normalization_quantities['time2hr']:3.2f} Hours (Post)",
                                save_loc/"CrossOb"/"Synodic"/f"Timestep_{ts+1}_2D_cloud_{cloud}.png",
                                plot_cross_observers = True,
                                cloud_names = [row[0] for row in cloud_names])
        
        
    #%% Plot and save metrics
    x = all_timesteps*normalization_quantities['time2hr']
    
    np.savez(save_loc / "metrics.npz", **metrics)
    np.savez(save_loc / "normalization_quantities.npz", **normalization_quantities)
    with open(save_loc / "cloud_names.json", 'w') as f:
        json.dump(cloud_names, f)
    with open(save_loc / "linestyle.json", 'w') as f:
        json.dump(linestyle, f)
    np.save(save_loc / "timesteps", x)

    for ob in np.where(cloud_is_active[:, 0])[0]:
        # Likelihood metrics
        plot.plotMetrics(x, metrics["likelihood_state_weighted"][ob], cloud_names[ob], linestyle[ob], save_loc, ob, "Log-Likelihood", f"GMM: Full, State Space, Log-Likelihood Ob: {ob} vs. Time", "likelihood_state_weighted.png")
        plot.plotMetrics(x, metrics["likelihood_state_best"][ob], cloud_names[ob], linestyle[ob], save_loc, ob, "Log-Likelihood", f"GMM: Best Mode, State Space, Log-Likelihood Ob: {ob} vs. Time", "likelihood_state_best.png")
        plot.plotMetrics(x, metrics["likelihood_msmt_weighted"][ob], cloud_names[ob], linestyle[ob], save_loc, ob, "Log-Likelihood", f"GMM: Full, Msmt Space, Log-Likelihood Ob: {ob} vs. Time", "likelihood_msmt_weighted.png")
        plot.plotMetrics(x, metrics["likelihood_msmt_best"][ob], cloud_names[ob], linestyle[ob], save_loc, ob, "Log-Likelihood", f"GMM: Best Mode, Msmt Space, Log-Likelihood Ob: {ob} vs. Time", "likelihood_msmt_best.png")
        
        # Entropy metrics
        plot.plotMetrics(x, metrics["entropy_state_weighted"][ob], cloud_names[ob], linestyle[ob], save_loc, ob, "Entropy", f"GMM: Full, State Space, Entropy Ob: {ob} vs. Time", "entropy_state_weighted.png")
        #plot.plotMetrics(x, metrics["entropy_state_best"][ob], cloud_names[ob], save_loc, ob, "Entropy", f"GMM: Best Mode, State Space, Entropy Ob: {ob} vs. Time", "entropy_state_best.png")
        # TODO: Look into next metric of next line
        plot.plotMetrics(x, metrics["entropy_msmt_weighted"][ob], cloud_names[ob], linestyle[ob], save_loc, ob, "Entropy", f"GMM: Full, Msmt Space, Entropy Ob: {ob} vs. Time", "entropy_msmt_weighted.png")
        plot.plotMetrics(x, metrics["entropy_msmt_best"][ob], cloud_names[ob], linestyle[ob], save_loc, ob, "Entropy", f"GMM: Best Mode, Msmt Space, Entropy Ob: {ob} vs. Time", "entropy_msmt_best.png")
        
        # AIC metrics
        plot.plotMetrics(x, metrics["AIC_state"][ob], cloud_names[ob], linestyle[ob], save_loc, ob, "AIC", f"State Space, AIC Ob: {ob} vs. Time", "AIC_state.png")
        plot.plotMetrics(x, metrics["AIC_msmt"][ob], cloud_names[ob], linestyle[ob], save_loc, ob, "AIC", f"Msmt Space, AIC Ob: {ob} vs. Time", "AIC_msmt.png")
        
        # Misc metrics
        plot.plotMetrics(x, metrics["num_cluster"][ob], cloud_names[ob], linestyle[ob], save_loc, ob, "Number of Clusters", f"Number of Metric Clusters Ob: {ob} vs. Time", "num_clusters.png")
        plot.plotMetrics(x, metrics["num_particles"][ob], cloud_names[ob], linestyle[ob], save_loc, ob, "Number of Particles", f"Number of Particles Ob: {ob} vs. Time", "num_particles.png")
        
        #plot.plotMetrics(x, metrics["ill_conditioned"][ob], cloud_names[ob], save_loc, ob, "Number of Particles", f"Number of Particles Ob: {ob} vs. Time", "num_particles.png")
        # TODO: Fix next plot
        plot.plotMetricsPerState(x, metrics["RMSE"][ob], normalization_quantities, cloud_names[ob], linestyle[ob], save_loc, ob, "RMSE (km, kms)", "RMSE Slice vs. Time", "RMSE")
    
    names = [row[0] for row in cloud_names][1:]
    ls = [row[0] for row in linestyle][1:]
    plot.plotMetrics(x, metrics["avg_ob_ob_likeli"][0, 1:], names, ls, save_loc, 0, "Ob-Ob Log-Likelihood", f"Ob {0} to Ob Likelihood vs. Time", "avg_ob_ob_likeli.png")
    plot.plotMetrics(x, metrics["avg_ob_ob_weight_loglikeli"][0, 1:], names, ls, save_loc, 0, "Ob-Ob Log-Likelihood", f"Ob {0} to Ob Weighted Log-Likelihood vs. Time", "avg_ob_ob_weight_loglikeli.png")
    plot.plotMetrics(x, metrics["avg_cross_entropy"][0, 1:], names, ls, save_loc, 0, "Cross Entropy", f"Ob {0} to Ob Cross Entropy vs. Time", "avg_cross_entropy.png")
    plot.plotMetrics(x, metrics["KL"][0, 1:, :, 0], names, ls, save_loc, 0, "KL", f"Ob {0} to Ob KL Divergence vs. Time", "KL1.png")
    plot.plotMetrics(x, metrics["KL"][0, 1:, :, 1], names, ls, save_loc, 0, "KL", f"Ob {0} to Ob KL Divergence vs. Time", "KL2.png")
    plot.plotMetrics(x, metrics["KL"][0, 1:, :, 2], names, ls, save_loc, 0, "KL", f"Ob {0} to Ob Avg KL Divergence vs. Time", "KL.png")
    plot.plotMetrics(x, metrics["JS"][0, 1:], names, ls, save_loc, 0, "JS", f"Ob {0} to Ob Jensen-Shannon Divergence vs. Time", "JS.png")
    plot.plotMetricsPerState(x, metrics["JS_marginal"][0, 1:, :, :], normalization_quantities, names, ls, save_loc, 0, "JS Slice", "JS Slice vs. Time", "JSSLice")
    
    success = True
    return success, None, None, None


def worker(MC_idx):
    try:
        success, ob, cloud, ts = main(MC_idx)
        return MC_idx, success, ob, cloud, ts
    except Exception as error:
        print(f"MC:{MC_idx} Error: ", error)
        return MC_idx, False, None, None, None


if __name__ == '__main__':
    # TODO: Save success list
    start_time = time.time()
    W = 2
    MC_indices = [10, 16, 39]#range(30, 45)
    # [0, 1, 7, 9, 11, 17, 18, 21, 22, 24, 25, 28]
    #test = main(MC_indices[0])
    test = main(MC_indices[1])
    test = main(MC_indices[2])
    #test = main(MC_indices[3])
    '''
    successes = []
    fail_ob_cloud_ts = []
    with Pool(W) as pool:
        for i, ok, ob, cloud, ts in pool.imap_unordered(worker, MC_indices, chunksize=1):
            if ok:
                successes.append(i)
            else:
                fail_ob_cloud_ts.append([i, ob, cloud, ts])
                
    success_rate = len(successes) / len(MC_indices)
    '''
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.1f}")