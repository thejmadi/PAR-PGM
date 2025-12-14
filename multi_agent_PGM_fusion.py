# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 16:52:54 2025

@author: tarun
"""

import numpy as np
from numpy import linalg as la
import scipy as sci
from scipy import io as sio
from numpy.polynomial.polynomial import Polynomial as poly
import time as time
from multiprocessing import Process
from multiprocessing import Pool, cpu_count
import datetime as dt
import pymap3d
import cr3bp_dyn as cr3bp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import CoordFunctions
from pathlib import Path


# Check certain file for existence
def ensure_dir_exists(file_path):
    # file_path: path variable to check for existence
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    

def termSat(T, Y):
    mu = 1.2150582e-2 # Dimensionless mass of the moon (and position of Earth w.r.t. barycenter)
    Rm = 1740/384400 # Nondimensionalized radius of the moon
    value = (np.sqrt((Y[0] + mu)**2 + Y[1]**2 + Y[2]**2) < 6371/384400) or (np.sqrt((Y[0] - (1-mu))**2 + Y[1]**2 + Y[2]**2) < Rm) # Stop when the target hits the Earth's or the Moon's surface
    return 1

def getNoisyMeas(Xtruth, R, h):
    mzkm = h(Xtruth)
    zk = np.random.multivariate_normal(mzkm, R)
    zk = zk.reshape([-1,1]) # Make into column vector        
    return zk

def kalmanUpdate(zk, Xcloud, R, mu_m, P_m, h):
    N = Xcloud.shape[0]
    Zcloud = np.zeros((N,zk.shape[0]))

    for i in range(N):
        Zcloud[i,:] = h(Xcloud[i,:])
    
    mzk_m = np.mean(Zcloud,0)

    Pxz = np.zeros((mzk_m.shape[0], P_m.shape[0]))
    Pzz = np.zeros((mzk_m.shape[0], mzk_m.shape[0]))

    # TODO: Vectorize
    for i in range(N):
        dx = (Xcloud[i,:] - mu_m).reshape([-1, 1])
        dz = (Zcloud[i,:] - mzk_m).reshape([-1, 1])
        Pxz = Pxz + dz@dx.T
        Pzz = Pzz + dz@dz.T
    

    Pxz = Pxz/N
    Pzz = Pzz/N + R

    K_k = Pxz.T@la.inv(Pzz)
    mu_p = mu_m + K_k@(zk - h(mu_m).reshape([-1,1])).reshape([-1])
    P_p = P_m - K_k@Pzz@K_k.T
    
    P_p = (P_p + P_p.T)/2

    D, V = np.linalg.eig(P_p)
    D = np.diag(D)
    P_p = V@D@V.T
    return mu_p, P_p

def weightUpdate(wc, cluster_points, idx, zk, R, h):
    wGains = np.zeros(wc.shape)
    for i in range(wc.shape[0]):
        cPts = cluster_points[idx == i, :]
        zPoints = np.zeros((cPts.shape[0], zk.shape[0]))
        # TODO: Vectorize
        for j in range(cPts.shape[0]):
            zPoints[j,:] = h(cPts[j,:])
        
        zPredMean = np.mean(zPoints,0)
        zPredCov = np.cov(zPoints.T) + R
        
        wGains[i, 0] = sci.stats.multivariate_normal.pdf(zk.T, mean=zPredMean, cov=zPredCov)
    w = wc * wGains / np.sum(wc * wGains)
    return w

def drawFrom(w, mu, P):
    wtoken = np.random.random()
    pos = np.searchsorted(np.cumsum(w), wtoken)
    mu_t = mu[pos,:]
    R_t = (P[pos,:] + P[pos,:].T)/2
    x_p = np.random.multivariate_normal(mu_t, R_t).astype(np.float64)
    return x_p.reshape([-1]), pos

# TODO: Check if actually needed
def getDiagCov(Xcloud):
    P = np.cov(Xcloud.T)
    ent = np.diag(P)
    return ent

def propPoint(particle_prior_synodic, t_int, interval, dyn_obj=cr3bp, ev=termSat):
    # First, convert from X_{ot} in the topocentric frame to X_{bt} in the
    # synodic frame.
    ##Xbest = backConvertSynodic(np.copy(Xest), t_int)
    
    # Next, propagate X_{bt} by a single time step and convert back to the 
    # topographic frame. Begin by calling the integrator
    
    ivp_result = sci.integrate.solve_ivp(dyn_obj.cr3bp_dyn, (0, interval),
    np.copy(particle_prior_synodic), method='BDF', events=ev, rtol=1e-6, atol=1e-8) 
    particle_post_synodic = ivp_result.y.T[-1, :]
    
    # Finish by converting back to topocentric reference frame
    ##Xm = convertToTopo(np.copy(Xbt_est), t_int + interval)
    
    return particle_post_synodic

def propagate(cloud_prior_topo, t_int, interval, obs_lat, obs_lon, obs_el):
    
    cloud_prior_synodic = Topo2Synodic(np.copy(cloud_prior_topo), t_int, obs_lat, obs_lon, obs_el)
    
    termSat.terminal = True
    termSat.direction = 0
    
    pool_propInputs = []
    for i in range(cloud_prior_synodic.shape[0]):
        pool_propInputs.append((np.copy(cloud_prior_synodic[i,:]), t_int, interval))
    
    with Pool(W) as p:
        prop_Points = p.starmap(propPoint, pool_propInputs)
    
    ##Xm_bt = np.array([r[0] for r in prop_Points])
    cloud_post_synodic = np.array([r[1] for r in prop_Points])
    cloud_post_topo = Synodic2Topo(np.copy(cloud_post_synodic), t_int+interval, obs_lat, obs_lon, obs_el)
    
    return cloud_post_topo

# TODO: Fix this function
'''
def crossObEntropy(cloud, cluster_by, Kp, num_agents):
    Kp = 10;
    gmm_unnorm = cell(1, num_agents);
    K = cell(1, num_agents);
    for ob in range(num_agents):
        [idx, Kp, ~] = cluster(cloud{ob}, cluster_by, Kp);
        cPoints = cell(Kp,1); covariances_unnorm = zeros(6, 6, Kp); means_unnorm = zeros(Kp, 6);
        w = zeros(Kp,1);
        for k = 1:Kp
            cluster_points = Xcloud{ob}(idx == k, :); 
            cPoints{k} = cluster_points; cSize = size(cPoints{k});
        
            if(cSize(1) == 1)
                covariances_unnorm(:, :, k) = zeros(6, 6);
            else
                means_unnorm(k, :) = mean(cluster_points, 1);
                covariances_unnorm(:, :, k) = cov(cluster_points); % Cell of GMM covariances 
            end
            w(k) = size(cluster_points, 1) / size(Xcloud{ob}, 1); % Vector of weights
        end
        gmm_unnorm{ob} = gmdistribution(means_unnorm, covariances_unnorm, w);
        K{ob} = gmm_unnorm{ob}.NumComponents;
    end

    likeli = zeros(K{1}, K{2});
    likeli_weight = zeros(K{1}, K{2});
    for i = 1:K{1}
        for j = 1:K{2}
            likeli(i, j) = mvnpdf(gmm_unnorm{2}.mu(j, :), gmm_unnorm{1}.mu(i, :), gmm_unnorm{1}.Sigma(:, :, i) + gmm_unnorm{2}.Sigma(:, :, j));%P_1(:, :, i) + P_2(:, :, j));
        end
    end
    
    P = likeli / sum(likeli(:));
    P_nonzero = P(P > 0);
    ob_ob_entropy = -sum(P_nonzero .* log10(P_nonzero));
    likeli_nonzero = likeli(likeli > 0);
    ob_ob_unnorm_entropy = -sum(likeli_nonzero .* log10(likeli_nonzero));
    ob_ob_l1_norm = log10(sum(abs(likeli), 'all'));
end
'''

if __name__ == '__main__':
    plt.rcParams["figure.dpi"] = 300
    MC_range = range(10)
    for MC_idx in MC_range:
        start_time = time.time()
    
        save_loc = "D:/PythonProjects/EDP/PGM/ParticleFusionTest/12_15_25_meeting/EXP_L2/EXP_VaryIODandFusionTime/Test1/MC_" + str(MC_idx);
        load_loc = "D:/PythonProjects/EDP/PGM/ParticleFusionTest/12_15_25_meeting/EXP_L2/EXP_VaryIODandFusionTime/Test1/OrbitData/Agent";
    
        dynamics = "CR3BP";
        if (dynamics == "CR3BP"):
            dist2km = 384400; # Kilometers per non-dimensionalized distance
            time2hr = 4.342*24; # Hours per non-dimensionalized time
            vel2kms = dist2km/(time2hr*60*60); # Kms per non-dimensionalized velocity
            normalization_quantities = {"dist2km": dist2km,
                                        "vel2kms": vel2kms,
                                        "time2hr": time2hr,
                                        "mu": 1.2150582e-2}
            #dynamics_model = @(t, x) Dynamics.cr3bp_dyn(t, x, normalization_quantities.mu);
        
        cluster_by = "FullState";
        Kn = 14; # Number of clusters (original)
        K = np.tile(Kn, 6, 6); # Number of clusters (changeable)
        Kmax = 14; # Maximum number of clusters (Kmax = 1 for EnKF)
        
        colors = ["red", "blue"]
        plot_IOD = False;
        plot_indv_clouds = False; # Not recommended unless debugging
        plot_cross_observers = True; # Recommended to see observers' original clouds plotted on same figure
        save_MC_metrics = True; # Recommended
        
        num_IOD_particles = 20000
        Lp = np.array([[20000], [20000], [20000]])
        total_num_agents = 3 # Expected number of agents
        num_agents = 0 # Current number of agents
        agent_is_active = np.full((total_num_agents, 1), False, dtype=bool) # Indicates whether agent at this index is active 
        active_mask = [] # Will contain indices of active agents
        num_msmt_for_IOD = np.array([[10], [10], [10]])
        ts_to_perform_IOD = np.array([[1], [21], [1]])
        plot_combined_clouds = np.array([[True], [False], [False], [False]], dtype=bool) # Plot all clouds of single observer on same figure. Recommended for observer 1
        num_clouds_per_agent = np.ones(total_num_agents, 1)
        num_clouds = num_agents * num_clouds_per_agent
        
        partial_ts = [pd.read_csv(f"{load_loc}{i+1}/partial_ts.csv").values for i in range(total_num_agents)]
        full_ts = [pd.read_csv(f"{load_loc}{i+1}/full_ts.csv").values for i in range(total_num_agents)]
        full_vts = [pd.read_csv(f"{load_loc}{i+1}/full_vts.csv").values for i in range(total_num_agents)]
        
        # TODO: combineMsmts function
        # fusion_information contains various info formatted as fusion #: [observer index, observer index, fuse?, timestep to fuse]
        fusion_information = [[0, 1, True, 45], [0, 1, True, 55], [0, 1, True, 65], [0, 1, True, 75]] 
        # TODO: Check if cloud_names is built correctly
        cloud_names = "Original Obs: " + range(max(1, total_num_agents-1)).T + ". IOD: " + str(normalization_quantities.time2hr * all_timesteps(ts_to_perform_IOD[0:-1])) + " hrs"
        cloud_names.append("Baseline Obs")
        fusion_types = ["Original", "Weight Update"]
        num_new_clouds_per_agent = 1

        # College Station
        obs_lat = np.tile(30.618963, 1, 6);
        obs_lon = np.tile(-96.339214, 1, 6);