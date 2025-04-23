# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 14:03:54 2025

@author: tarun
"""

import numpy as np
from numpy import linalg as la
import scipy as sci
from scipy import io as sio
from numpy.polynomial.polynomial import Polynomial as poly
import time as time
from multiprocessing import Pool
import datetime as dt
import pymap3d
#import termSat as tS
import cr3bp_dyn as cr3bp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def termSat(T, Y):
    mu = 1.2150582e-2 # Dimensionless mass of the moon (and position of Earth w.r.t. barycenter)
    Rm = 1740/384400 # Nondimensionalized radius of the moon
    value = (np.sqrt((Y[0] + mu)**2 + Y[1]**2 + Y[2]**2) < 6371/384400) or (np.sqrt((Y[0] - (1-mu))**2 + Y[1]**2 + Y[2]**2) < Rm) # Stop when the target hits the Earth's or the Moon's surface
    return 1


def stateEstCloud(pf, obTr, tdiff, file_path):
    noised_obs = obTr

    R_t = np.zeros(3*noised_obs.shape[0]) # We shall diagonalize this later
    mu_t = np.zeros(3*noised_obs.shape[0])

    partial_ts = np.genfromtxt(file_path + "partial_ts.csv", delimiter=',')
    
    for i in range(obTr.shape[0]):
        mu_t[3*i:3*i+3] = np.array([partial_ts[i,1], partial_ts[i,2], partial_ts[i,3]])
        R_t[3*i:3*i+3] = np.array([0.05*partial_ts[i,1], 7.2722e-6, 7.2722e-6])**2

    R_t = np.diag(R_t)
    data_vec = np.random.multivariate_normal(mu_t, R_t).reshape([-1,1])

    for i in range(noised_obs.shape[0]):
        noised_obs[i,1:4] = data_vec[3*i:3*i+3,0]

    # Extract the first continuous observation track

    hdo = [] # Matrix for a half day observation
    i = 0
    while noised_obs[i+1,0] - noised_obs[i,0] < tdiff: # Add small epsilon due to roundoff error
        hdo.append(noised_obs[i+1,:])
        i = i + 1
    
    hdo = np.asarray(hdo)
    
    # Convert observation data into [X, Y, Z] data in the topographic frame.
    
    hdR = np.zeros((hdo.shape[0],4)) # Convert quantities of hdo to [X, Y, Z]
    hdR[:,0] = hdo[:,0] # Timestamp stays the same
    hdR[:,1] = hdo[:,1] * np.cos(hdo[:,3]) * np.cos(hdo[:,2]) # Conversion to X
    hdR[:,2] = hdo[:,1] * np.cos(hdo[:,3]) * np.sin(hdo[:,2]) # Conversion to Y
    hdR[:,3] = hdo[:,1] * np.sin(hdo[:,3]) # Conversion to Z

    #pf = 0.50 # A factor between 0 to 1 describing the length of the day to interpolate [x, y]
    in_len = np.floor(pf * hdR.shape[0]+0.5).astype(int) # Length of interpolation interval
    hdR_p = hdR[:in_len,:] # Matrix for a partial half-day observation

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

def backConvertSynodic(X_ot, t_stamp):
    rot_topo = X_ot[:3] # First three components of the state vector
    vot_topo = X_ot[3:] # Last three components of the state vector

    # First step: Obtain X_{eo}**{ECI} 
    obs_lat = 30.618963
    obs_lon = -96.339214
    elevation = 103.8
    mu = 1.2150582e-2
    #, int(.1261889999956*1000*1000)
    UTC_vec_orig = dt.datetime(2024, 5, 3, 2, 41, 15, tzinfo=dt.timezone.utc) # Initial UTC vector at t_stamp = 0
    t_add_dim = t_stamp * (4.342) # Convert the time to add to a dimensional quantity
    UTC_vec = UTC_vec_orig + dt.timedelta(t_add_dim) # You will need this for calculating r_{eo} and v_{eo}
    

    delt_add_dim = t_add_dim - 1/86400
    delt_updatedUTCtime = UTC_vec_orig + dt.timedelta(delt_add_dim)

    reo_dim = pymap3d.geodetic2eci(obs_lat, obs_lon, elevation, UTC_vec)
    delt_reodim = pymap3d.geodetic2eci(obs_lat, obs_lon, elevation, delt_updatedUTCtime)
    reo_dim = np.asarray(reo_dim).reshape(3)
    delt_reodim = np.asarray(delt_reodim).reshape(3)
    veo_dim = reo_dim - delt_reodim # Finite difference

    reo_nondim = np.zeros(reo_dim.shape[0])
    veo_nondim = np.zeros(veo_dim.shape[0])
    reo_nondim[:] = reo_dim[:]/(1000*384400) # Conversion to non-dimensional units in the ECI frame
    veo_nondim[:] = veo_dim[:]*(4.342*86400)/(1000*384400) # Conversion to non-dimensional units in the ECI frame

    z_hat_topo = reo_nondim/la.norm(reo_nondim)
    x_hat_topo = np.cross(z_hat_topo, np.array([0, 0, 1]))/la.norm(np.cross(z_hat_topo, np.array([0, 0, 1])))
    y_hat_topo = np.cross(x_hat_topo, z_hat_topo)/la.norm(np.cross(x_hat_topo, z_hat_topo))
    
    A = np.vstack((x_hat_topo, y_hat_topo, z_hat_topo)) # Computing A as DCM for transforming between ECI and topographic reference frame

    dmag_dt = np.dot(reo_nondim, veo_nondim)/la.norm(reo_nondim)
    
    zhat_dot_topo = (veo_nondim * la.norm(reo_nondim) - reo_nondim * dmag_dt)/(la.norm(reo_nondim))**2
    xhat_dot_topo = np.cross(zhat_dot_topo, np.array([0, 0, 1]))/la.norm(np.cross(z_hat_topo, np.array([0, 0, 1]))) - np.dot(x_hat_topo, np.cross(zhat_dot_topo, np.array([0, 0, 1])))*x_hat_topo
    yhat_dot_topo = (np.cross(xhat_dot_topo, z_hat_topo) + np.cross(x_hat_topo, zhat_dot_topo))/la.norm(np.cross(x_hat_topo, z_hat_topo)) - np.dot(y_hat_topo, np.cross(xhat_dot_topo, z_hat_topo) + np.cross(x_hat_topo, zhat_dot_topo))*y_hat_topo

    dA_dt = np.vstack((xhat_dot_topo, yhat_dot_topo, zhat_dot_topo))

    rot_ECI = la.inv(A)@rot_topo
    vot_ECI = la.inv(A)@(vot_topo - dA_dt@rot_ECI)

    # Calculating X_{ET} in the synodic frame with our above quantities
    
    ret_ECI = reo_nondim + rot_ECI
    vet_ECI = veo_nondim + vot_ECI

    R3 = np.array([[np.cos(t_stamp), -np.sin(t_stamp), 0], [np.sin(t_stamp), np.cos(t_stamp), 0], [0, 0, 1]])
    dR3_dt = np.array([[-np.sin(t_stamp), -np.cos(t_stamp), 0], [np.cos(t_stamp), -np.sin(t_stamp), 0], [0, 0, 0]])

    ret_S = la.inv(R3)@ret_ECI
    vet_S = la.inv(R3)@(vet_ECI - dR3_dt@ret_S)

    r_be = np.array([-mu, 0, 0])
    v_be = np.array([0, 0, 0])

    r_bt = r_be + ret_S # In synodic reference frame
    v_bt = v_be + vet_S # In synodic reference frame

    X_bt = np.hstack((r_bt, v_bt))
    return X_bt

# Used for converting between X_{BT} in the synodic frame and X_{OT} in the
# topocentric frame for a single state
def convertToTopo(X_bt, t_stamp):
    # Insert code for obtaining vector between center of Earth and observer

    obs_lat = 30.618963
    obs_lon = -96.339214
    elevation = 103.8
    
    mu = 1.2150582e-2
    rbe = np.array([-mu, 0, 0]) # Position vector relating center of earth to barycenter

    UTC_vec_orig = dt.datetime(2024, 5, 3, 2, 41, 15, tzinfo=dt.timezone.utc) # Initial UTC vector at t_stamp = 0
    t_add_dim = t_stamp * (4.342) # Convert the time to add to a dimensional quantity
    UTC_vec = UTC_vec_orig + dt.timedelta(t_add_dim) # You will need this for calculating r_{eo} and v_{eo}

    delt_add_dim = t_add_dim - 1/86400
    delt_updatedUTCtime = UTC_vec_orig + dt.timedelta(delt_add_dim)

    reo_dim = pymap3d.geodetic2eci(obs_lat, obs_lon, elevation, UTC_vec)
    delt_reodim = pymap3d.geodetic2eci(obs_lat, obs_lon, elevation, delt_updatedUTCtime)
    reo_dim = np.asarray(reo_dim).reshape(3)
    delt_reodim = np.asarray(delt_reodim).reshape(3)
    veo_dim = reo_dim - delt_reodim # Finite difference

    R_z = np.array([[np.cos(t_stamp), -np.sin(t_stamp), 0], [np.sin(t_stamp), np.cos(t_stamp), 0], [0, 0, 1]])
    dRz_dt = np.array([[-np.sin(t_stamp), -np.cos(t_stamp), 0], [np.cos(t_stamp), -np.sin(t_stamp), 0], [0, 0, 0]])

    reo_nondim = np.zeros(reo_dim.shape)
    veo_nondim = np.zeros(veo_dim.shape)
    reo_nondim[:] = reo_dim[:]/(1000*384400) # Conversion to non-dimensional units in the ECI frame
    veo_nondim[:] = veo_dim[:]*(4.342*86400)/(1000*384400)

    rot_ECI = -reo_nondim + R_z@(-rbe + X_bt[:3])
    vot_ECI = -veo_nondim + R_z@(X_bt[3:]) + dRz_dt@(-rbe + X_bt[:3])

    # Finally, we convert from the ECI frame to the topographic frame

    # Step 1: Find the unit vectors governing this topocentric frame
    z_hat_topo = reo_nondim/la.norm(reo_nondim)

    x_hat_topo_unorm = np.cross(z_hat_topo, np.array([0, 0, 1])) # We choose a 
    # reference vector such as the North Pole, but we have several 
    # choices regarding the second vector
  
    x_hat_topo = x_hat_topo_unorm/la.norm(x_hat_topo_unorm) # Remember to normalize

    y_hat_topo_unorm = np.cross(x_hat_topo, z_hat_topo)
    y_hat_topo = y_hat_topo_unorm/la.norm(y_hat_topo_unorm) # Remember to normalize

    # Step 2: Convert all of the components of 'rot' from our aligned reference
    # frames to this new topocentric frame.
    
    rot_topo = np.array([np.dot(rot_ECI, x_hat_topo), np.dot(rot_ECI, y_hat_topo), np.dot(rot_ECI, z_hat_topo)])

    # Step 3: Handle the time derivatives of vot_topo = d/dt (rot_topo)
    R_topo = np.vstack((x_hat_topo, y_hat_topo, z_hat_topo)) # DCM relating ECI to topocentric coordinate frame
    dmag_dt = np.dot(reo_nondim, veo_nondim)/la.norm(reo_nondim) # How the magnitude of r_eo changes w.r.t. time
    
    zhat_dot_topo = (veo_nondim*la.norm(reo_nondim) - reo_nondim*dmag_dt)/(la.norm(reo_nondim))**2
    xhat_dot_topo = np.cross(zhat_dot_topo, np.array([0, 0, 1]))/la.norm(np.cross(z_hat_topo, np.array([0, 0, 1]))) - np.dot(x_hat_topo, np.cross(zhat_dot_topo, np.array([0, 0, 1])))*x_hat_topo
    yhat_dot_topo = (np.cross(xhat_dot_topo, z_hat_topo) + np.cross(x_hat_topo, zhat_dot_topo))/la.norm(np.cross(x_hat_topo, z_hat_topo)) - np.dot(y_hat_topo, np.cross(xhat_dot_topo, z_hat_topo) + np.cross(x_hat_topo, zhat_dot_topo))*y_hat_topo

    dA_dt = np.vstack((xhat_dot_topo, yhat_dot_topo, zhat_dot_topo))
    vot_topo = R_topo@vot_ECI + dA_dt@rot_ECI

    X_ot = np.hstack([rot_topo, vot_topo])
    return X_ot

def getNoisyMeas(Xtruth, R, h):
    mzkm = h(Xtruth)
    zk = np.random.multivariate_normal(mzkm, R)
    zk = zk.reshape([-1,1]) # Make into column vector        
    return zk

# Kalman update using particles from each cluster
def kalmanUpdate(zk, Xcloud, R, mu_m, P_m, h):
    N = Xcloud.shape[0]
    Zcloud = np.zeros((N,zk.shape[0]))

    for i in range(N):
        Zcloud[i,:] = h(Xcloud[i,:])
    
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

def getDiagCov(Xcloud):
    P = np.cov(Xcloud.T)
    ent = np.diag(P)
    return ent

def propagate(Xcloud, t_int, interval):
    Xm_cloud = np.zeros(Xcloud.shape)
    Xbt = np.zeros(Xcloud.shape)
    Xm_bt = np.zeros(Xcloud.shape)

    termSat.terminal = True
    termSat.direction = 0
    
    for i in range(Xcloud.shape[0]):
        # First, convert from X_{ot} in the topocentric frame to X_{bt} in the
        # synodic frame.
        Xbt[i,:] = backConvertSynodic(np.copy(Xcloud[i,:]), t_int)
        
        # Next, propagate each X_{bt} in your particle cloud by a single time 
        # step and convert back to the topographic frame.
        # Call ode45()
        
        # Check Syn Results
        ivp_result = sci.integrate.solve_ivp(cr3bp.cr3bp_dyn, (0, interval), np.copy(Xbt[i, :]), method='BDF', events=termSat, rtol=1e-6, atol=1e-8)
        Xm_bt[i, :] = ivp_result.y.T[-1, :]
        
        # Convert back to Topocentric
        Xm_cloud[i,:] = convertToTopo(np.copy(Xm_bt[i, :]), t_int + interval)
    return Xm_cloud


def linHx(mu):
    Hk_AZ = np.array([-mu[1]/(mu[0]**2 + mu[1]**2), mu[0]/(mu[0]**2 + mu[1]**2), 0, 0, 0, 0]) # Azimuth angle linearization
    Hk_EL = np.array([-(mu[0]*mu[2])/((mu[0]**2 + mu[1]**2 + mu[2]**2)*np.sqrt(mu[0]**2+mu[1]**2)),
                      -(mu[1]*mu[2])/((mu[0]**2 + mu[1]**2 + mu[2]**2)*np.sqrt(mu[0]**2+mu[1]**2)), 
                      np.sqrt(mu[0]**2 + mu[1]**2)/(mu[0]**2 + mu[1]**2 + mu[2]**2), 0, 0, 0])

    Hx = np.vstack((Hk_AZ, Hk_EL))
    return Hx

def getKnEntropy(Kp, Xcloud):
    mu_c = np.zeros((Kp, Xcloud.shape[1]))
    P_c = np.zeros((Kp, Xcloud.shape[1], Xcloud.shape[1]))
    w = np.zeros((Kp, 1))
    #h = lambda x: np.array([np.arctan2(x[1],x[0]), np.pi/2 - np.arccos(x[2]/np.sqrt(x[0]**2 + x[1]**2 + x[2]**2))]) # Nonlinear measurement model
    # Split propagated cloud into position and velocity data before
    # normalization.
    rc = Xcloud[:,:3]
    vc = Xcloud[:,3:]
    
    mean_rc = np.mean(rc, 0)
    mean_vc = np.mean(vc, 0)
    
    std_rc = np.std(rc,0, ddof=1)
    std_vc = np.std(vc,0, ddof=1)
    
    norm_rc = (rc - mean_rc)/la.norm(std_rc) # Normalizing the position 
    norm_vc = (vc - mean_vc)/la.norm(std_vc) # Normalizing the velocity
    
    Xm_norm = np.hstack((norm_rc, norm_vc))
    
    # Cluster using K-means clustering algorithm
    while True:
        kmeans = KMeans(n_clusters=Kp, init="k-means++").fit(Xm_norm)
        C = kmeans.cluster_centers_
        idx = kmeans.labels_
        
        if np.all(np.bincount(idx) > 6):  # Check if all clusters have more than 6 points
            break
        else:
            Kp -= 1  # Reduce the number of clusters if condition is not met
    
    # Convert cluster centers back to non-dimensionalized units
    C_unorm = np.zeros(C.shape)
    C_unorm[:, :] = C[:, :]
    C_unorm[:,:3] = (C[:,:3]*std_rc) + mean_rc # Conversion of position
    C_unorm[:,3:6] = (C[:,3:6]*std_vc) + mean_vc
    cPoints = []#cell(K,1)
    
    # Calculate covariances and weights for each cluster
    for k in range(Kp):
        cluster_points = Xcloud[idx == k, :] # Keep clustering very separate from mean, covariance, weight calculations
        cPoints.append(cluster_points)
        cSize = cPoints[k].shape
        mu_c[k, :] = np.mean(cluster_points, 0) # Cell of GMM means 
    
        if cSize[0] != 1:
            P_c[k, :, :] = np.cov(cluster_points.T) # Cell of GMM covariances
        w[k, :] = cluster_points.shape[0] / Xm_norm.shape[0] # Vector of weights
        
    wsum = 0
    for k in range(Kp):
        wsum += w[k]*la.det(P_c[k,:,:].T)
    ent = np.log(wsum)
    return ent

def plotState(data_cloud, data_truth, dist2km, vel2kms, title, save_path, save_title, K = None, idx = None, plot_cluster = False):
    fig, axs = plt.subplots(2, 3)
    fig.suptitle(title);
    fig.tight_layout()
    colors = ["Red", "royalblue", "Green", "Yellow", "Magenta", "Cyan", "Black", "#500000", "#bf5700", "#00274c"]
    if plot_cluster:
        for k in range(K):
            clusterPoints = data_cloud[idx == k, :]
            axs[0,0].scatter(dist2km*clusterPoints[:,0], dist2km*clusterPoints[:,1], label="Est", c=colors[k], s=0.2)
    else:
        axs[0,0].scatter(dist2km*data_cloud[:,0], dist2km*data_cloud[:,1], label="Est", c="royalblue", s=0.2)
    axs[0,0].scatter(dist2km*data_truth[0], dist2km*data_truth[1], label="Act", c="black", marker="x")
    axs[0,0].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    axs[0,0].set_title('X-Y')
    axs[0,0].set_xlabel('X (km)')
    axs[0,0].set_ylabel('Y (km)')
    
    if plot_cluster:
        for k in range(K):
            clusterPoints = data_cloud[idx == k, :]
            axs[0,1].scatter(dist2km*clusterPoints[:,0], dist2km*clusterPoints[:,2], label="Est", c=colors[k], s=0.2)
    else:
        axs[0,1].scatter(dist2km*data_cloud[:,0], dist2km*data_cloud[:,2], label="Est", c="royalblue", s=0.2)
    axs[0,1].scatter(dist2km*data_truth[0], dist2km*data_truth[2], label="Act", c="black", marker="x")
    axs[0,1].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    axs[0,1].set_title('X-Z')
    axs[0,1].set_xlabel('X (km)')
    axs[0,1].set_ylabel('Z (km)')

    if plot_cluster:
        for k in range(K):
            clusterPoints = data_cloud[idx == k, :]
            axs[0,2].scatter(dist2km*clusterPoints[:,1], dist2km*clusterPoints[:,2], label="Est", c=colors[k], s=0.2)
    else:    
        axs[0,2].scatter(dist2km*data_cloud[:,1], dist2km*data_cloud[:,2], label="Est", c="royalblue", s=0.2)
    axs[0,2].scatter(dist2km*data_truth[1], dist2km*data_truth[2], label="Act", c="black", marker="x")
    axs[0,2].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    axs[0,2].set_title('Y-Z')
    axs[0,2].set_xlabel('Y (km)')
    axs[0,2].set_ylabel('Z (km)')
    
    if plot_cluster:
        for k in range(K):
            clusterPoints = data_cloud[idx == k, :]
            axs[1,0].scatter(vel2kms*clusterPoints[:,3], vel2kms*clusterPoints[:,4], label="Est", c=colors[k], s=0.2)
    else:
        axs[1,0].scatter(vel2kms*data_cloud[:,3], vel2kms*data_cloud[:,4], label="Est", c="royalblue", s=0.2)
    axs[1,0].scatter(vel2kms*data_truth[3], vel2kms*data_truth[4], label="Act", c="black", marker="x")
    axs[1,0].set_title('X-Y')
    axs[1,0].set_xlabel('X (km/s)')
    axs[1,0].set_ylabel('Y (km/s)')
    
    if plot_cluster:
        for k in range(K):
            clusterPoints = data_cloud[idx == k, :]
            axs[1,1].scatter(vel2kms*clusterPoints[:,3], vel2kms*clusterPoints[:,5], label="Est", c=colors[k], s=0.2)
    else:
        axs[1,1].scatter(vel2kms*data_cloud[:,3], vel2kms*data_cloud[:,5], label="Est", c="royalblue", s=0.2)
    axs[1,1].scatter(vel2kms*data_truth[3], vel2kms*data_truth[5], label="Act", c="black", marker="x")
    axs[1,1].set_title('X-Z')
    axs[1,1].set_xlabel('X (km/s)')
    axs[1,1].set_ylabel('Z (km/s)')
    
    if plot_cluster:
        for k in range(K):
            clusterPoints = data_cloud[idx == k, :]
            axs[1,2].scatter(vel2kms*clusterPoints[:,4], vel2kms*clusterPoints[:,5], label="Est", c=colors[k], s=0.2)
    else:
        axs[1,2].scatter(vel2kms*data_cloud[:,4], vel2kms*data_cloud[:,5], label="Est", c="royalblue", s=0.2)
    axs[1,2].scatter(vel2kms*data_truth[4], vel2kms*data_truth[5], label="Act", c="black", marker="x")
    axs[1,2].set_title('Y-Z')
    axs[1,2].set_xlabel('Y (km/s)')
    axs[1,2].set_ylabel('Z (km/s)')
    
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2)
    fig.tight_layout() 
    #plt.show()
    fig.savefig(save_path + save_title + ".png")
    plt.close()

def plotEllipses(data_cloud, data_truth, dist2km, vel2kms, title, save_path, save_title, K, idx, mu_mat, P_mat):
    colors = ["Red", "royalblue", "Green", "Yellow", "Magenta", "Cyan", "Black", "#500000", "#bf5700", "#00274c"]
    fig, axs = plt.subplots(2, 3)
    fig.suptitle(title);
    
    plt.subplot(2,3,1)
    plot_dims = [0,1]
    P_marg_idx1, P_marg_idx2 = np.meshgrid(plot_dims, plot_dims)
    mu_marg = mu_mat[:, plot_dims]
    P_marg = P_mat[:, P_marg_idx1, P_marg_idx2]
    
    grid_length = 100
    X1, X2 = np.meshgrid(np.linspace(np.min(data_cloud[:, plot_dims[0]]), np.max(data_cloud[:, plot_dims[0]]), grid_length),
                         np.linspace(np.min(data_cloud[:, plot_dims[1]]), np.max(data_cloud[:, plot_dims[1]]), grid_length));
    X_grid = np.hstack((X1.flatten().reshape([-1,1]), X2.flatten().reshape([-1,1])))
    Z_cell = np.zeros((K, grid_length, grid_length))
    contours_cell = np.zeros((K, 3))
    
    for k in range(K):
        Z = np.zeros(X_grid.shape[0])
        for i in range(X_grid.shape[0]):
            X_temp = (X_grid[i,:] - mu_marg[k,:]).reshape([1,-1])
            Z[i] = np.exp(-0.5 * (X_temp @ la.inv(P_marg[k,:,:]) @ X_temp.T))[0,0]
        Z = Z.reshape(X1.shape)
        Z = Z/(2*np.pi*np.sqrt(la.det(P_marg[k,:,:])))
        Z_cell[k,:,:] = Z[:,:]
        contours_cell[k,:] = np.max(Z[:,:]) * np.exp(-0.5 * np.array([3.44, 2.3, 1])**2)  # Corresponding to sigma intervals
    contours_cell += np.ones((K, 3)) * np.array([1e-15, 2e-15, 3e-15])
    
    for k in range(K):
        axs[0,0].contour(dist2km*X1, dist2km*X2, dist2km*Z_cell[k,:,:], dist2km*contours_cell[k,:], colors=colors[k], zorder=k)
    axs[0,0].scatter(dist2km*data_truth[0], dist2km*data_truth[1], label="Act", c="black", marker="x", zorder=K)
    axs[0,0].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    axs[0,0].set_title('X-Y')
    axs[0,0].set_xlabel('X (km)')
    axs[0,0].set_ylabel('Y (km)')
    
    plt.subplot(2,3,2)
    plot_dims = [0,2]
    P_marg_idx1, P_marg_idx2 = np.meshgrid(plot_dims, plot_dims)
    mu_marg = mu_mat[:, plot_dims]
    P_marg = P_mat[:, P_marg_idx1, P_marg_idx2]
    
    grid_length = 100
    X1, X2 = np.meshgrid(np.linspace(np.min(data_cloud[:, plot_dims[0]]), np.max(data_cloud[:, plot_dims[0]]), grid_length),
                         np.linspace(np.min(data_cloud[:, plot_dims[1]]), np.max(data_cloud[:, plot_dims[1]]), grid_length));
    X_grid = np.hstack((X1.flatten().reshape([-1,1]), X2.flatten().reshape([-1,1])))
    Z_cell = np.zeros((K, grid_length, grid_length))
    contours_cell = np.zeros((K, 3))
    
    for k in range(K):
        Z = np.zeros(X_grid.shape[0])
        for i in range(X_grid.shape[0]):
            X_temp = (X_grid[i,:] - mu_marg[k,:]).reshape([1,-1])
            Z[i] = np.exp(-0.5 * (X_temp @ la.inv(P_marg[k,:,:]) @ X_temp.T))[0,0]
        Z = Z.reshape(X1.shape)
        Z = Z/(2*np.pi*np.sqrt(la.det(P_marg[k,:,:])))
        Z_cell[k,:,:] = Z[:,:]
        contours_cell[k,:] = np.max(Z[:,:]) * np.exp(-0.5 * np.array([3.44, 2.3, 1])**2)  # Corresponding to sigma intervals
    contours_cell += np.ones((K, 3)) * np.array([1e-15, 2e-15, 3e-15])

    for k in range(K):
        axs[0,1].contour(dist2km*X1, dist2km*X2, dist2km*Z_cell[k,:,:], dist2km*contours_cell[k,:], colors=colors[k], zorder=k)
    axs[0,1].scatter(dist2km*data_truth[0], dist2km*data_truth[2], label="Act", c="black", marker="x", zorder=K)
    axs[0,1].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    axs[0,1].set_title('X-Z')
    axs[0,1].set_xlabel('X (km)')
    axs[0,1].set_ylabel('Z (km)')

    plt.subplot(2,3,3)
    plot_dims = [1,2]
    P_marg_idx1, P_marg_idx2 = np.meshgrid(plot_dims, plot_dims)
    mu_marg = mu_mat[:, plot_dims]
    P_marg = P_mat[:, P_marg_idx1, P_marg_idx2]
    
    grid_length = 100
    X1, X2 = np.meshgrid(np.linspace(np.min(data_cloud[:, plot_dims[0]]), np.max(data_cloud[:, plot_dims[0]]), grid_length),
                         np.linspace(np.min(data_cloud[:, plot_dims[1]]), np.max(data_cloud[:, plot_dims[1]]), grid_length));
    X_grid = np.hstack((X1.flatten().reshape([-1,1]), X2.flatten().reshape([-1,1])))
    Z_cell = np.zeros((K, grid_length, grid_length))
    contours_cell = np.zeros((K, 3))
    
    for k in range(K):
        Z = np.zeros(X_grid.shape[0])
        for i in range(X_grid.shape[0]):
            X_temp = (X_grid[i,:] - mu_marg[k,:]).reshape([1,-1])
            Z[i] = np.exp(-0.5 * (X_temp @ la.inv(P_marg[k,:,:]) @ X_temp.T))[0,0]
        Z = Z.reshape(X1.shape)
        Z = Z/(2*np.pi*np.sqrt(la.det(P_marg[k,:,:])))
        Z_cell[k,:,:] = Z[:,:]
        contours_cell[k,:] = np.max(Z[:,:]) * np.exp(-0.5 * np.array([3.44, 2.3, 1])**2)  # Corresponding to sigma intervals
    contours_cell += np.ones((K, 3)) * np.array([1e-15, 2e-15, 3e-15])
    
    for k in range(K):
        axs[0,2].contour(dist2km*X1, dist2km*X2, dist2km*Z_cell[k,:,:], dist2km*contours_cell[k,:], colors=colors[k], zorder=k)
    axs[0,2].scatter(dist2km*data_truth[1], dist2km*data_truth[2], label="Act", c="black", marker="x", zorder=K)
    axs[0,2].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    axs[0,2].set_title('Y-Z')
    axs[0,2].set_xlabel('Y (km)')
    axs[0,2].set_ylabel('Z (km)')


    plt.subplot(2,3,4)
    plot_dims = [3,4]
    P_marg_idx1, P_marg_idx2 = np.meshgrid(plot_dims, plot_dims)
    mu_marg = mu_mat[:, plot_dims]
    P_marg = P_mat[:, P_marg_idx1, P_marg_idx2]
    
    grid_length = 100
    X1, X2 = np.meshgrid(np.linspace(np.min(data_cloud[:, plot_dims[0]]), np.max(data_cloud[:, plot_dims[0]]), grid_length),
                         np.linspace(np.min(data_cloud[:, plot_dims[1]]), np.max(data_cloud[:, plot_dims[1]]), grid_length));
    X_grid = np.hstack((X1.flatten().reshape([-1,1]), X2.flatten().reshape([-1,1])))
    Z_cell = np.zeros((K, grid_length, grid_length))
    contours_cell = np.zeros((K, 3))
    
    for k in range(K):
        Z = np.zeros(X_grid.shape[0])
        for i in range(X_grid.shape[0]):
            X_temp = (X_grid[i,:] - mu_marg[k,:]).reshape([1,-1])
            Z[i] = np.exp(-0.5 * (X_temp @ la.inv(P_marg[k,:,:]) @ X_temp.T))[0,0]
        Z = Z.reshape(X1.shape)
        Z = Z/(2*np.pi*np.sqrt(la.det(P_marg[k,:,:])))
        Z_cell[k,:,:] = Z[:,:]
        contours_cell[k,:] = np.max(Z[:,:]) * np.exp(-0.5 * np.array([3.44, 2.3, 1])**2)  # Corresponding to sigma intervals
    contours_cell += np.ones((K, 3)) * np.array([1e-15, 2e-15, 3e-15])
    
    for k in range(K):
        axs[1,0].contour(vel2kms*X1, vel2kms*X2, vel2kms*Z_cell[k,:,:], vel2kms*contours_cell[k,:], colors=colors[k], zorder=k)
    axs[1,0].scatter(vel2kms*data_truth[3], vel2kms*data_truth[4], label="Act", c="black", marker="x")
    axs[1,0].set_title('X-Y')
    axs[1,0].set_xlabel('X (km/s)')
    axs[1,0].set_ylabel('Y (km/s)')
    
    plt.subplot(2,3,5)
    plot_dims = [3,5]
    P_marg_idx1, P_marg_idx2 = np.meshgrid(plot_dims, plot_dims)
    mu_marg = mu_mat[:, plot_dims]
    P_marg = P_mat[:, P_marg_idx1, P_marg_idx2]
    
    grid_length = 100
    X1, X2 = np.meshgrid(np.linspace(np.min(data_cloud[:, plot_dims[0]]), np.max(data_cloud[:, plot_dims[0]]), grid_length),
                         np.linspace(np.min(data_cloud[:, plot_dims[1]]), np.max(data_cloud[:, plot_dims[1]]), grid_length));
    X_grid = np.hstack((X1.flatten().reshape([-1,1]), X2.flatten().reshape([-1,1])))
    Z_cell = np.zeros((K, grid_length, grid_length))
    contours_cell = np.zeros((K, 3))
    
    for k in range(K):
        Z = np.zeros(X_grid.shape[0])
        for i in range(X_grid.shape[0]):
            X_temp = (X_grid[i,:] - mu_marg[k,:]).reshape([1,-1])
            Z[i] = np.exp(-0.5 * (X_temp @ la.inv(P_marg[k,:,:]) @ X_temp.T))[0,0]
        Z = Z.reshape(X1.shape)
        Z = Z/(2*np.pi*np.sqrt(la.det(P_marg[k,:,:])))
        Z_cell[k,:,:] = Z[:,:]
        contours_cell[k,:] = np.max(Z[:,:]) * np.exp(-0.5 * np.array([3.44, 2.3, 1])**2)  # Corresponding to sigma intervals
    contours_cell += np.ones((K, 3)) * np.array([1e-15, 2e-15, 3e-15])
    
    for k in range(K):
        axs[1,1].contour(vel2kms*X1, vel2kms*X2, vel2kms*Z_cell[k,:,:], vel2kms*contours_cell[k,:], colors=colors[k], zorder=k)
    axs[1,1].scatter(vel2kms*data_truth[3], vel2kms*data_truth[5], label="Act", c="black", marker="x")
    axs[1,1].set_title('X-Z')
    axs[1,1].set_xlabel('X (km/s)')
    axs[1,1].set_ylabel('Z (km/s)')

    plt.subplot(2,3,6)
    plot_dims = [4,5]
    P_marg_idx1, P_marg_idx2 = np.meshgrid(plot_dims, plot_dims)
    mu_marg = mu_mat[:, plot_dims]
    P_marg = P_mat[:, P_marg_idx1, P_marg_idx2]
    
    grid_length = 100
    X1, X2 = np.meshgrid(np.linspace(np.min(data_cloud[:, plot_dims[0]]), np.max(data_cloud[:, plot_dims[0]]), grid_length),
                         np.linspace(np.min(data_cloud[:, plot_dims[1]]), np.max(data_cloud[:, plot_dims[1]]), grid_length));
    X_grid = np.hstack((X1.flatten().reshape([-1,1]), X2.flatten().reshape([-1,1])))
    Z_cell = np.zeros((K, grid_length, grid_length))
    contours_cell = np.zeros((K, 3))
    
    for k in range(K):
        Z = np.zeros(X_grid.shape[0])
        for i in range(X_grid.shape[0]):
            X_temp = (X_grid[i,:] - mu_marg[k,:]).reshape([1,-1])
            Z[i] = np.exp(-0.5 * (X_temp @ la.inv(P_marg[k,:,:]) @ X_temp.T))[0,0]
        Z = Z.reshape(X1.shape)
        Z = Z/(2*np.pi*np.sqrt(la.det(P_marg[k,:,:])))
        Z_cell[k,:,:] = Z[:,:]
        contours_cell[k,:] = np.max(Z[:,:]) * np.exp(-0.5 * np.array([3.44, 2.3, 1])**2)  # Corresponding to sigma intervals
    contours_cell += np.ones((K, 3)) * np.array([1e-15, 2e-15, 3e-15])
    
    for k in range(K):
        axs[1,2].contour(vel2kms*X1, vel2kms*X2, vel2kms*Z_cell[k,:,:], vel2kms*contours_cell[k,:], colors=colors[k], zorder=k)
    axs[1,2].scatter(vel2kms*data_truth[4], vel2kms*data_truth[5], label="Act", c="black", marker="x")
    axs[1,2].set_title('Y-Z')
    axs[1,2].set_xlabel('Y (km/s)')
    axs[1,2].set_ylabel('Z (km/s)')

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2)
    fig.tight_layout()
    fig.savefig(save_path + save_title + ".png")
    plt.close()
    
def plotStDev(data, dist2km, vel2kms, title, save_title, save_path):
    fig, axs = plt.subplots(2, 3)
    fig.suptitle(title);
    fig.tight_layout()
    axs[0,0].scatter(np.arange(data.shape[0]), dist2km*data[:, 0])
    axs[0,0].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    axs[0,0].set_title('X')
    axs[0,0].set_ylabel('Log sigma (km)')
    
    axs[0,1].scatter(np.arange(data.shape[0]), dist2km*data[:, 1])
    axs[0,1].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    axs[0,1].set_title('Y')
    axs[0,1].set_ylabel('Log sigma (km)')

    axs[0,2].scatter(np.arange(data.shape[0]), dist2km*data[:, 2])
    axs[0,2].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    axs[0,2].set_title('Z')
    axs[0,2].set_ylabel('Log sigma (km)')
    
    axs[1,0].scatter(np.arange(data.shape[0]), dist2km*data[:, 3])
    axs[1,0].set_title('Xdot')
    axs[1,0].set_ylabel('Log sigma (km/s)')
    
    axs[1,1].scatter(np.arange(data.shape[0]), dist2km*data[:, 4])
    axs[1,1].set_title('Ydot')
    axs[1,1].set_ylabel('Log sigma (km/s)')
    
    axs[1,2].scatter(np.arange(data.shape[0]), dist2km*data[:, 5])
    axs[1,2].set_title('Zdot')
    axs[1,2].set_ylabel('Log sigma (km/s)')
    
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.supxlabel()
    fig.legend(handles, labels, loc='upper right', ncol=2)
    fig.tight_layout() 
    fig.savefig(save_path + save_title + ".png")
    plt.close()
    
def plotAzEl(data_cloud, data_truth, K, h, save_path, save_title, idx, plot_cluster):
    colors = ["Red", "royalblue", "Green", "Yellow", "Magenta", "Cyan", "Black", "#500000", "#bf5700", "#00274c"]
    if plot_cluster:
        for k in range(K):
            clusterPoints = data_cloud[idx == k, :]
            zt = np.zeros((1,2))
            Zmcloud = np.zeros((clusterPoints.shape[0], zt.shape[1]))
            for i in range(Zmcloud.shape[0]):
                Zmcloud[i,:] = h(clusterPoints[i,:]).T
            plt.scatter(180/np.pi*Zmcloud[:,0], 180/np.pi*Zmcloud[:,1], c=colors[k], s=0.5)
    else:
        plt.scatter(180/np.pi*Zmcloud[:,0], 180/np.pi*Zmcloud[:,1], c=colors[0], s=0.5)
    Ztruth = h(data_truth).T
    plt.scatter(180/np.pi*Ztruth[0], 180/np.pi*Ztruth[1], label="Act", c="black", marker="x")
        
    plt.title('AZ-EL Python')
    plt.xlabel('Azimuth Angle (deg)')
    plt.ylabel('Elevation Angle (deg)')
    plt.savefig(save_path + save_title + ".png")
    plt.close()
    return

if __name__ == '__main__':
    # Start the clock
    plt.rcParams["figure.dpi"] = 300
    start_time = time.time()
    
    # Load noiseless observation data and other important .mat files
    file_path = "D:\\PythonProjects\\EDP\\PGM_Git\\PAR-PGM\\"
    save_path = "D:\\PythonProjects\\EDP\\PGM\\Test1\SimFigures\\"
    partial_ts = np.genfromtxt(file_path + "partial_ts.csv", delimiter=',')
    full_ts = np.genfromtxt(file_path + "full_ts.csv", delimiter=',')
    full_vts = np.genfromtxt(file_path + "full_vts.csv", delimiter=',')

    # Add observation noise to the observation data as follows:
    # Range - 5# of the current (i.e. noiseless) range
    # Azimuth - 1.5 arc sec (credit: Dr. Utkarsh Ranjan Mishra)
    # Elevation - 1.5 arc sec (credit: Dr. Utkarsh Ranjan Mishra)
    # Note: All above quantities are drawn in a zero-mean Gaussian fashion.
    
    noised_obs = np.zeros(partial_ts.shape)
    R_t = np.zeros(3*noised_obs.shape[0]) # We shall diagonalize this later
    mu_t = np.zeros(3*noised_obs.shape[0])
    noised_obs[:, :] = partial_ts[:, :]
    
    
    theta_f = 1.5 # Arc-seconds of error covariance
    R_f = 0.05 # Range percentage error covariance
    
    dist2km = 384400 # Kilometers per non-dimensionalized distance
    time2hr = 4.342*24 # Hours per non-dimensionalized time
    vel2kms = dist2km/(time2hr*60*60) # Kms per non-dimensionalized velocity
    
    for i in range(partial_ts.shape[0]):
        mu_t[3*i:3*i+3] = np.array([partial_ts[i,1], partial_ts[i,2], partial_ts[i,3]])
        R_t[3*i:3*i+3] = np.array([(0.05*partial_ts[i,1])**2, (theta_f*4.84814e-6)**2, (theta_f*4.84814e-6)**2])
    
    R_t = np.diag(R_t)
    data_vec = np.random.multivariate_normal(mu_t, R_t).reshape([-1,1])
    
    for i in range(noised_obs.shape[0]):
        noised_obs[i,1:4] = data_vec[3*i:3*i+3, 0]
    
    # Extract important time points from the noised_obs variable
    i = 1
    interval = noised_obs[1,0] - partial_ts[0,0]
    cTimes = [] # Array of important time points
    
    while i < noised_obs.shape[0]:
        if noised_obs[i,0] - noised_obs[i-1,0] > (interval+1e-11):
            cTimes.extend([noised_obs[i-1,0], noised_obs[i,0]])
        i = i + 1
    
    cTimes = np.asarray(cTimes)
    
    larger_diff = noised_obs[-1,0] - noised_obs[-2,0]
    for j in range(1, noised_obs.shape[0]):
        if (noised_obs[j,0] - noised_obs[j-1,0]) > (larger_diff+1e-11):
            cVal = noised_obs[j,0] 
            break
        else:
            cVal = noised_obs[-1, 0]
    
    # Extract the first continuous observation track
    hdo = [] # Matrix for a half day observation
    hdo.append(noised_obs[0,:])
    i = 0
    while noised_obs[i+1,0] - noised_obs[i,0] < full_ts[1,0] + 1e-15: # Add small epsilon due to roundoff error
        hdo.append(noised_obs[i+1,:])
        i = i + 1
    
    hdo = np.asarray(hdo)[1:, :]
    
    # Convert observation data into [X, Y, Z] data in the topographic frame.
    
    hdR = np.zeros((hdo.shape[0],4)) # Convert quantities of hdo to [X, Y, Z]
    hdR[:,0] = hdo[:,0] # Timestamp stays the same
    hdR[:,1] = hdo[:,1] * np.cos(hdo[:,3]) * np.cos(hdo[:,2]) # Conversion to X
    hdR[:,2] = hdo[:,1] * np.cos(hdo[:,3]) * np.sin(hdo[:,2]) # Conversion to Y
    hdR[:,3] = hdo[:,1] * np.sin(hdo[:,3]) # Conversion to Z
    
    pf = 0.25 # A factor between 0 to 1 describing the length of the day to interpolate [x, y]
    nfit = 4
    in_len = np.floor(pf * hdR.shape[0]+0.5).astype(int) # Length of interpolation interval
    
    if in_len < (nfit + 1):
        in_len = nfit + 1
        pf = in_len/hdR.shape[0]
    
    hdR_p = hdR[:in_len,:] # Matrix for a partial half-day observation
    
    # Fit polynomials for X, Y, and Z (Cubic for X, Quadratic for X and Y)
    coeffs_X = poly.fit(hdR_p[:,0], hdR_p[:,1], nfit, domain=[])
    coeffs_Y = poly.fit(hdR_p[:,0], hdR_p[:,2], nfit, domain=[])
    coeffs_Z = poly.fit(hdR_p[:,0], hdR_p[:,3], nfit, domain=[])
    
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
    
    partial_vts = []
    partial_rts = []
    j = 0
    i = 0
    while j < hdR_p.shape[0]:
        if hdR[j,0] == full_vts[i,0]: # Matching time index
            partial_vts.append(full_vts[i,:])
            partial_rts.append(full_ts[i,:])
            j = j + 1
        i = i + 1
    
    partial_vts = np.asarray(partial_vts)
    partial_rts = np.asarray(partial_rts)
    
    Xot_fitted = np.array([[X_fit[-1], Y_fit[-1], Z_fit[-1], Xdot_fit[-1], Ydot_fit[-1], Zdot_fit[-1,]]])
    Xot_truth = np.hstack((partial_rts[-1,1:], partial_vts[-1,1:]))
    
    t_truth = partial_rts[-1,0]
    idx_prop = np.where(np.equal(full_ts[:, 0],t_truth))[0]
    Xprop_truth = np.hstack((full_ts[idx_prop+1,1:], full_vts[idx_prop+1,1:])).reshape([-1])
    
    L = 1000
    Lp = 1*L
    X0cloud = np.zeros((L,6))
    for i in range(X0cloud.shape[0]):
        X0cloud[i, :] = stateEstCloud(pf, np.copy(partial_ts), partial_ts[1,0]-partial_ts[0,0]+1e-15, file_path)

    ############## Plotting ###############
    
    plotState(X0cloud, Xot_truth, dist2km, vel2kms, "Timestep " + str(t_truth*time2hr) + " Hours", save_path, "iodCloud", K= 1)

    ############## Return to Code ##############
    t_int = hdR_p[-1,0] # Time at which we are obtaining a state cloud
    tspan = np.array([0, interval]) # Integrate over just a single time step
    Xm_cloud = np.zeros((L,6))
    Xm_cloud[:, :] = X0cloud[:, :]
    Xbt = np.zeros(X0cloud.shape)
    
    termSat.terminal = True
    termSat.direction = 0
    for i in range(X0cloud.shape[0]):
        Xbt[i,:] = backConvertSynodic(X0cloud[i,:], t_int)
        # Next, propagate each X_{bt} in your particle cloud by a single time 
        # step and convert back to the topographic frame.
        # Call ode45()
        ivp_result = sci.integrate.solve_ivp(cr3bp.cr3bp_dyn, (0, interval), np.copy(Xbt[i, :]), method='BDF', events=termSat, rtol=1e-6, atol=1e-8)
        X = ivp_result.y.T
        Xm_cloud[i,:] = convertToTopo(X[-1,:].T, t_int + interval)
        # Xm_cloud[i,:] = procNoise(Xm_cloud[i,:]) # Adds process noise
    # Initialize variables
    Kn = 8 # Number of clusters (original)
    K = Kn # Number of clusters (changeable)
    Kmax = 8 # Maximum number of clusters (Kmax = 1 for EnKF)
    
    # Split propagated cloud into position and velocity data before
    # normalization.
    rc = Xm_cloud[:,:3]
    vc = Xm_cloud[:,3:]
    
    mean_rc = np.mean(rc, 0)
    mean_vc = np.mean(vc, 0)
    
    std_rc = np.std(rc,0, ddof=1)
    std_vc = np.std(vc,0, ddof=1)
    
    norm_rc = (rc - mean_rc)/la.norm(std_rc) # Normalizing the position 
    norm_vc = (vc - mean_vc)/la.norm(std_vc) # Normalizing the velocity
    
    Xm_norm = np.hstack((norm_rc, norm_vc))
    
    # Cluster using K-means clustering algorithm
    while True:
        kmeans = KMeans(n_clusters=K, init="k-means++").fit(Xm_norm)
        C = kmeans.cluster_centers_
        idx = kmeans.labels_
        
        if np.all(np.bincount(idx) > 6):  # Check if all clusters have more than 6 points
            break
        else:
            K -= 1  # Reduce the number of clusters if condition is not met
    
    mu_c = np.zeros((K, Xm_cloud.shape[1]))
    P_c = np.zeros((K, Xm_cloud.shape[1], Xm_cloud.shape[1]))
    wm = np.zeros((K, 1))
    
    # Convert cluster centers back to non-dimensionalized units
    C_unorm = np.zeros(C.shape)
    C_unorm[:, :] = C[:, :]
    C_unorm[:,:3] = (C[:,:3]*std_rc) + mean_rc # Conversion of position
    C_unorm[:,3:6] = (C[:,3:6]*std_vc) + mean_vc
    
    cPoints = []
    
    # Calculate covariances and weights for each cluster
    for k in range(K):
        cluster_points = Xm_cloud[idx == k, :] # Keep clustering very separate from mean, covariance, weight calculations
        cPoints.append(cluster_points)
        cSize = cPoints[k].shape
        mu_c[k, :] = np.mean(cluster_points, 0) # Cell of GMM means 
    
        if cSize[0] != 1:
            P_c[k, :, :] = np.cov(cluster_points.T) # Cell of GMM covariances
        wm[k, :] = cluster_points.shape[0] / Xm_norm.shape[0] # Vector of weights
    
    ############## Plotting ##############
    
    plotState(Xm_cloud, Xprop_truth, dist2km, vel2kms, "Timestep " + str(round(time2hr*full_ts[idx_prop+1,0][0],4)) + " Hours (Prior)", save_path, "Timestep_0_1B", K, idx, True)
    h = lambda x: np.array([np.arctan2(x[1],x[0]), np.pi/2 - np.arccos(x[2]/np.sqrt(x[0]**2 + x[1]**2 + x[2]**2))]) # Nonlinear measurement model
    plotAzEl(Xm_cloud, Xprop_truth, K, h, save_path, "Timestep_0_1C", idx, plot_cluster=True)
    
    ############## Return to Code ##############
    
    
    Xprop_truth = np.hstack((full_ts[idx_prop+1,1:4], full_vts[idx_prop+1,1:4])).reshape([-1])
    print('Truth State: \n')
    print(Xprop_truth)
    
    # Now that we have a GMM representing the prior distribution, we have to
    # use a Kalman update for each component: weight, mean, and covariance.
    
    # Posterior variables
    wp = np.zeros(wm.shape)
    mu_p = np.zeros(mu_c.shape)
    P_p = np.zeros(P_c.shape)
    wp[:, :] = wm[:, :]
    mu_p[:] = mu_c[:]
    P_p[:, :, :] = P_c[:, :, :]
    
    # Comment this out if you wish to use noise.
    # noised_obs = partial_ts
    
    tpr = t_int + interval # Time stamp of the prior means, weights, and covariances
    idx_meas = np.where(abs(noised_obs[:,0] - tpr) < 1e-10)[0] # Find row with time
    
    #if idx_meas.shape[0] >= 1: # i.e. there exists a measurement
    if idx_meas.shape[0] != 0:
        R_vv = np.array([[theta_f*np.pi/648000, 0], [0, theta_f*np.pi/648000]])**2
        h = lambda x: np.array([np.arctan2(x[1],x[0]), np.pi/2 - np.arccos(x[2]/np.sqrt(x[0]**2 + x[1]**2 + x[2]**2))]) # Nonlinear measurement model
        zt = getNoisyMeas(Xprop_truth, R_vv, h, 1)
    
        for i in range(K):
            # [mu_p{i}, P_p{i}] = ukfUpdate(zt, R_vv, mu_c{i}, P_c{i}, h)
            mu_p[i,:], P_p[i,:] = kalmanUpdate(zt, cPoints[i], R_vv, mu_c[i,:], P_c[i,:,:], h)
    
        # Weight update
        wp = weightUpdate(wm, Xm_cloud, idx, zt, R_vv, h, 1)
    
    else:
        for i in range(K):
            wp[i] = wm[i]
            mu_p[i,:] = mu_c[i,:]
            P_p[i,:] = P_c[i,:]
        
    Xp_cloud = np.zeros(Xm_cloud.shape)
    Xp_cloud[:,:] = Xm_cloud[:,:]
    c_id = np.zeros(Xp_cloud.shape[0])
    
    for i in range(L):
        Xp_cloud[i,:], c_id[i] = drawFrom(wp, mu_p, P_p, i+1)
    mu_pExp = np.zeros((K, mu_p.shape[1]))
    mu_pExp[:,:] = mu_p[:,:]
    
    ############## Plotting ###############
    
    plotState(Xp_cloud, Xprop_truth, dist2km, vel2kms, "Timestep " + str(round(time2hr*noised_obs[idx_meas,0][0],4)) + " Hours (Posterior)", save_path, "Timestep_0_2B", K, c_id, True)
    
    ############## Return to Code ##############
    
    # At this point, we have shown a PGM-I propagation and update step. The
    # next step is to utilize this PGM-I update across all time steps during
    # which the target is within our sensor FOV and see how the particle clouds
    # (i.e. GM components) evolve over time. If we're lucky, we should see that
    # the GMM tracks the truth over the interval.
    
    # Find and set the start and end times to simulation
    idx_meas = np.where(abs(hdR[:,0] - tpr) < 1e-10)[0]
    c_meas = 0
    interval = hdR[idx_meas,c_meas] - hdR[idx_meas-1,c_meas]
    
    idx_crit = np.where(abs(full_ts[:,0]) >= (28*24)/time2hr)[0][0] # Find the index of the last time step before a certain number of days have passed since orbit propagation
    t_end = full_ts[-1,0] # First observation of new pass + one more time step
    
    tau = -1
    idx_end = np.where(abs(full_ts[:,0] - t_end) < 1e-10)[0][0]+1
    idx_start = np.where(abs(full_ts[:,0] - tpr) < 1e-10)[0][0]
    
    l_filt = full_ts[idx_start:idx_end, 0].shape[0]+1
    
    ent2 = np.zeros((l_filt,1))
    ent1 = np.zeros((l_filt+1,mu_c.shape[1])) 
    
    ent2[0] = np.log(la.det(np.cov(X0cloud.T)))
    ent2[1] = np.log(la.det(np.cov(Xp_cloud.T)))
    ent1[0,:] = getDiagCov(X0cloud) 

    Xp_cloudp = np.zeros(Xp_cloud.shape)
    Xp_cloudp[:,:] = Xp_cloud[:, :]
    for ts in range(idx_start, idx_end-1):
        print(Xp_cloudp.shape[0])
        to = full_ts[ts,0]
        interval = full_ts[ts+1,0] - full_ts[ts,0]
    
        ent1[tau+2,:] = getDiagCov(Xp_cloudp)
        # Propagation Step
        Xm_cloud = propagate(Xp_cloudp, to, interval)
        Xprop_truth = propagate(Xprop_truth.reshape([1,-1]), to, interval)
        Xprop_truth = Xprop_truth.reshape([-1])
        # Verification Step
        tpr = to + interval # Time stamp of the prior means, weights, and covariances
        idx_meas = np.where(abs(noised_obs[:,0] - tpr) < 1e-10)[0] # Find row with time
        tau = tau + 1
        if idx_meas.shape[0] != 0:
            if tpr >= cVal:
                K = Kmax
            else:
                K = Kn
                
            rc = Xm_cloud[:,:3]
            vc = Xm_cloud[:,3:]
            
            mean_rc = np.mean(rc, 0)
            mean_vc = np.mean(vc, 0)
            
            std_rc = np.std(rc,0, ddof=1)
            std_vc = np.std(vc,0, ddof=1)
            
            norm_rc = (rc - mean_rc)/la.norm(std_rc) # Normalizing the position 
            norm_vc = (vc - mean_vc)/la.norm(std_vc) # Normalizing the velocity
            
            Xm_norm = np.hstack((norm_rc, norm_vc))
            
            # Verification Step
            idx_meas = np.where(abs(noised_obs[:,0] - tpr) < 1e-10)[0] # Find row with time
            
            print("Timestamp: " + str(round(tpr*time2hr,5)))
            
            # Cluster using K-means clustering algorithm
            while True:
                kmeans = KMeans(n_clusters=K, init="k-means++").fit(Xm_norm)
                temp = kmeans.cluster_centers_
                idx = kmeans.labels_
                
                if np.all(np.bincount(idx) > 6):  # Check if all clusters have more than 6 points
                    break
                else:
                    K -= 1  # Reduce the number of clusters if condition is not met
    
            mu_c = np.zeros((K, Xm_cloud.shape[1]))
            mu_p = np.zeros((K, Xm_cloud.shape[1]))
            P_c = np.zeros((K, Xm_cloud.shape[1], Xm_cloud.shape[1]))
            P_p = np.zeros((K, Xm_cloud.shape[1], Xm_cloud.shape[1]))
            wm = np.zeros((K, 1))
            wp = np.zeros((K, 1))
            
            cPoints = []
            
            # Calculate covariances and weights for each cluster
            for k in range(K):
                cluster_points = Xm_cloud[idx == k, :] # Keep clustering very separate from mean, covariance, weight calculations
                cPoints.append(cluster_points)
                cSize = cPoints[k].shape
                mu_c[k, :] = np.mean(cluster_points, 0) # Cell of GMM means 
            
                if cSize[0] != 1:
                    P_c[k, :, :] = np.cov(cluster_points.T) # Cell of GMM covariances
                wm[k, :] = cluster_points.shape[0] / Xm_cloud.shape[0] # Vector of weights
                
            # Extract means
            mu_mExp = np.zeros((K, mu_c.shape[1]))
            for k in range(K):
                mu_mExp[k,:] = mu_c[k,:]
        
            zc = noised_obs[idx_meas,1:4].T # Presumption: An observation occurs at this time step
            xto = zc[0]*np.cos(zc[1])*np.cos(zc[2]) 
            yto = zc[0]*np.sin(zc[1])*np.cos(zc[2]) 
            zto = zc[0]*np.sin(zc[2]) 
            rto = np.hstack([xto, yto, zto])
    
            legend_string = []
            for k in range(K):
                R_vv = np.array([[R_f*partial_ts[idx_meas,1][0], 0, 0],
                                 [0, theta_f*np.pi/648000, 0],
                                 [0, 0, theta_f*np.pi/648000]])**2
                Hxk = linHx(mu_c[k,:]) # Linearize about prior mean component
            
            if True: # Use for all time steps            
                mu_mat = np.zeros(mu_c.shape)
                P_mat = np.zeros(P_c.shape)
                mu_mat[:,:] = mu_c[:,:]
                P_mat[:,:,:] = P_c[:,:,:]
        
                ############## Plotting ###############
                plotEllipses(Xm_cloud, Xprop_truth, dist2km, vel2kms, "Timestep " + str(round(tpr*time2hr,4)) + " Hours (Prior)", save_path, "Timestep_" + str(tau) + "_1A", K, idx, mu_mat, P_mat)
                plotState(Xm_cloud, Xprop_truth, dist2km, vel2kms, "Timestep " + str(round(time2hr*noised_obs[idx_meas,0][0],4)) + " Hours (Prior)", save_path, "Timestep_" + str(tau) + "_1B", K, idx, True)

                np.savetxt("D:\\PythonProjects\\EDP\\PGM\\Test1\\cPoints" + str(ts+1) + ".txt", [cPoints[i].shape for i in range(len(cPoints))], delimiter=',', fmt='%f')
                plotAzEl(Xm_cloud, Xprop_truth, K, h, save_path, "Timestep_" + str(tau) + "_1C", idx, plot_cluster=True)
                ############## Return to Code ##############
        
            if abs(to - (t_end-interval)) < 1e-10: # At final time step possible
                # Save the a priori estimate particle cloud
                np.savetxt('aPriori_' + str(ts+1) + ".txt", Xm_cloud, delimiter=',', fmt='%f')
    
                # Extract means
                mu_mExp[:,:] = mu_c[:,:]
        
                # Show where observation lies (position only)
                if idx_meas.shape[0] != 0:
                    zc = noised_obs[idx_meas,1:4].T # Presumption: An observation occurs at this time step
                    xto = zc[0]*np.cos(zc[1])*np.cos(zc[2]) 
                    yto = zc[0]*np.sin(zc[1])*np.cos(zc[2]) 
                    zto = zc[0]*np.sin(zc[2]) 
                    rto = np.hstack((xto, yto, zto))
                
                ############## Plotting ###############
                
                ############## Return to Code ##############
            # Update Step
            R_vv = np.array([[theta_f*np.pi/648000, 0], [0, theta_f*np.pi/648000]])**2
            # Hxk = linHx(mu_c{i}); # Linearize about prior mean component
            h = lambda x: np.array([np.arctan2(x[1],x[0]), np.pi/2 - np.arccos(x[2]/np.sqrt(x[0]**2 + x[1]**2 + x[2]**2))]) # Nonlinear measurement model
            zt = getNoisyMeas(Xprop_truth, R_vv, h)
        
            for i in range(K):
                # [mu_p{i}, P_p{i}] = ukfUpdate(zt, R_vv, mu_c{i}, P_c{i}, h)
                mu_p[i,:], P_p[i,:] = kalmanUpdate(zt, cPoints[i], R_vv, mu_c[i,:], P_c[i,:,:], h)
                P_p[i, :, :] = 0.5*(P_p[i, :, :] + P_p[i, :, :].T)
                
            # Weight update
            wp = weightUpdate(wm, Xm_cloud, idx, zt, R_vv, h)
        
        else:
            print("Timestamp: " + str(round(tpr*time2hr,5)))
            Xp_cloud = np.zeros(Xm_cloud.shape)
            mu_p = np.zeros((K, Xp_cloud.shape[1]))
            cPoints = []
            
            Xp_cloud[:,:] = Xm_cloud[:,:]
            cPoints.append(Xp_cloud)
            wp = np.array([[1]])
            mu_p[0,:] = np.mean(Xp_cloud, 0)
            P_p[0,:,:] = np.cov(Xp_cloud.T)
        
        if idx_meas.shape[0] != 0:
            Xp_cloudp = np.zeros((Lp, Xprop_truth.shape[0]))
            c_id = np.zeros(Lp);
            for i in range(Lp):
                Xp_cloudp[i,:], c_id[i] = drawFrom(wp, mu_p, P_p) 
        
        else:
            K = 1
            Xp_cloudp[:,:] = Xm_cloud[:,:]
            c_id = np.zeros(Xp_cloudp.shape[0])
        
        if True:        
            # Extract means
            mu_pExp = np.zeros((K, mu_p.shape[1]))
            mu_pExp[:,:] = mu_p[:K,:]
        
            mu_mat = np.zeros(mu_pExp.shape)
            P_mat = np.zeros(P_p.shape)
            mu_mat[:,:] = mu_pExp[:,:]
            P_mat[:,:,:] = P_p[:,:,:]
            
            ############## Plotting ###############
            plotEllipses(Xp_cloudp, Xprop_truth, dist2km, vel2kms, "Timestep " + str(round(tpr*time2hr,4)) + " Hours (Post)", save_path, "Timestep_" + str(tau) + "_2A", K, c_id, mu_mat, P_mat)
            plotState(Xp_cloudp, Xprop_truth, dist2km, vel2kms, "Timestep " + str(round(tpr*time2hr,4)) + " Hours (Post)", save_path, "Timestep_" + str(tau) + "_2B", K, c_id, True)
            plotAzEl(Xp_cloudp, Xprop_truth, K, h, save_path, "Timestep_" + str(tau) + "_2C", c_id, plot_cluster=True)
            ############## Return to Code ###############
        
        if idx_meas.shape[0] != 0:
            wsum = 0
            for k in range(K):
                wsum += np.sum(wp[k,:])*la.det(P_p[k, :, :])
            ent2[tau+2] = np.log(wsum)
        else:
            if tpr >= cVal:
                Ke = Kmax # Clusters used for calculating entropy
            else:
                Ke = Kn # Clusters used for calculating entropy
            
            ent2[tau+2] = getKnEntropy(Ke, Xp_cloudp) # Get entropy as if you still are using six clusters
        if abs(tpr - cTimes[1]) < 1e-10:
            Lp = 1250
        elif abs(tpr - cTimes[3]) < 1e-10:
            Lp = 1500
        elif abs(tpr - cTimes[7]) < 1e-10:
            Lp = 2500
            #np.savetxt("Xm_cloud.txt", Xp_cloudp, delimiter=',', fmt='#f')
            #np.savetxt("t_int.txt", tpr, delimiter=',', fmt='#f')
            #np.savetxt("noised_obs.txt", noised_obs, delimiter=',', fmt='#f')
            #np.savetxt("Xtruth.txt", Xprop_truth, delimiter=',', fmt='#f')
    c_id = np.zeros(Lp)
    for i in range(Lp):
        Xp_cloudp[i,:], c_id[i] = drawFrom(wp, mu_p, P_p) 
    ent1[-1,:] = getDiagCov(Xp_cloudp)
    #ent2[-1] = []
    ############## Plotting ###############
    plotState(Xp_cloudp, Xprop_truth, dist2km, vel2kms, "Timestep " + str(round(tpr,4)), save_path, "finalDistribution_normK", K, c_id, True)
    ############## Return to Code ###############
    print("Total Time: " + str(int(time.time()-start_time)))
    
    plotStDev(ent1, dist2km, vel2kms, "StDevEvols", "Std Dev", save_path)

    plt.plot(np.arange(ent2.shape[0]), ent2)    
    plt.title('Entropy')
    plt.xlabel('Filter Step #')
    plt.ylabel('Entropy Metric')
    plt.savefig(save_path + "Entropy" + ".png")
    plt.close()