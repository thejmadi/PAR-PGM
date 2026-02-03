# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 16:54:23 2025

@author: tarun
"""
import numpy as np
from numpy import linalg as la
import datetime as dt
import pymap3d

# Used to get the location of the Earth-Observer vector at a certain time (t_stamp)
def getObserverPos(t_stamp, obs_lat, obs_lon, obs_el, norm_quantities):
    # Insert code for obtaining vector between center of Earth and observer
    UTC_vec_orig = dt.datetime(2024, 5, 3, 2, 41, 15, tzinfo=dt.timezone.utc) # Initial UTC vector at t_stamp = 0
    t_add_dim = t_stamp *  norm_quantities["time2hr"]/24 # Convert the time to add to a dimensional quantity
    UTC_vec = UTC_vec_orig + dt.timedelta(t_add_dim) # You will need this for calculating r_{eo} and v_{eo}

    
    reo_dim = pymap3d.geodetic2eci(obs_lat, obs_lon, obs_el, UTC_vec)
    reo_dim = np.asarray(reo_dim).reshape(3)
    reo_nondim = np.zeros(reo_dim.shape)
    reo_nondim[:] = reo_dim[:]/(1000*norm_quantities["dist2km"]) # Conversion to non-dimensional units in the ECI frame
    
    # Finally, we convert from the ECI frame to the topographic frame

    # Step 1: Find the unit vectors governing this topocentric frame
    z_hat_topo = reo_nondim/la.norm(reo_nondim)

    x_hat_topo_unorm = np.cross(z_hat_topo, np.array([0, 0, 1])) # We choose a 
    # reference vector such as the North Pole, but we have several 
    # choices regarding the second vector
  
    x_hat_topo = x_hat_topo_unorm/la.norm(x_hat_topo_unorm) # Remember to normalize

    y_hat_topo_unorm = np.cross(x_hat_topo, z_hat_topo)
    y_hat_topo = y_hat_topo_unorm/la.norm(y_hat_topo_unorm) # Remember to normalize
    
    reo_topo = np.array([np.dot(reo_nondim, x_hat_topo), np.dot(reo_nondim, y_hat_topo), np.dot(reo_nondim, z_hat_topo)])
    
    return reo_topo

'''
def Topo2Synodic(X_ot, t_stamp, obs_lat, obs_lon, obs_el, norm_quantities):
    # First step: Obtain X_{eo}**{ECI} 
    UTC_vec_orig = dt.datetime(2024, 5, 3, 2, 41, 15, tzinfo=dt.timezone.utc) # Initial UTC vector at t_stamp = 0
    t_add_dim = t_stamp * norm_quantities["time2hr"]/24 # Convert the time to add to a dimensional quantity
    UTC_vec = UTC_vec_orig + dt.timedelta(t_add_dim) # You will need this for calculating r_{eo} and v_{eo}
    

    delt_add_dim = t_add_dim - 1/86400
    delt_updatedUTCtime = UTC_vec_orig + dt.timedelta(delt_add_dim)

    reo_dim = pymap3d.geodetic2eci(obs_lat, obs_lon, obs_el, UTC_vec)
    delt_reodim = pymap3d.geodetic2eci(obs_lat, obs_lon, obs_el, delt_updatedUTCtime)
    reo_dim = np.asarray(reo_dim).reshape(3)
    delt_reodim = np.asarray(delt_reodim).reshape(3)
    veo_dim = reo_dim - delt_reodim # Finite difference

    reo_nondim = np.zeros(reo_dim.shape[0])
    veo_nondim = np.zeros(veo_dim.shape[0])
    reo_nondim[:] = reo_dim[:]/(1000*norm_quantities["dist2km"]) # Conversion to non-dimensional units in the ECI frame
    veo_nondim[:] = veo_dim[:]/(1000*norm_quantities["vel2kms"]) # Conversion to non-dimensional units in the ECI frame

    z_hat_topo = reo_nondim/la.norm(reo_nondim)
    x_hat_topo = np.cross(z_hat_topo, np.array([0, 0, 1]))/la.norm(np.cross(z_hat_topo, np.array([0, 0, 1])))
    y_hat_topo = np.cross(x_hat_topo, z_hat_topo)/la.norm(np.cross(x_hat_topo, z_hat_topo))
    
    A = np.vstack((x_hat_topo, y_hat_topo, z_hat_topo)) # Computing A as DCM for transforming between ECI and topographic reference frame

    dmag_dt = np.dot(reo_nondim, veo_nondim)/la.norm(reo_nondim)
    
    zhat_dot_topo = (veo_nondim * la.norm(reo_nondim) - reo_nondim * dmag_dt)/(la.norm(reo_nondim))**2
    xhat_dot_topo = np.cross(zhat_dot_topo, np.array([0, 0, 1]))/la.norm(np.cross(z_hat_topo, np.array([0, 0, 1]))) - np.dot(x_hat_topo, np.cross(zhat_dot_topo, np.array([0, 0, 1])))*x_hat_topo
    yhat_dot_topo = (np.cross(xhat_dot_topo, z_hat_topo) + np.cross(x_hat_topo, zhat_dot_topo))/la.norm(np.cross(x_hat_topo, z_hat_topo)) - np.dot(y_hat_topo, np.cross(xhat_dot_topo, z_hat_topo) + np.cross(x_hat_topo, zhat_dot_topo))*y_hat_topo

    dA_dt = np.vstack((xhat_dot_topo, yhat_dot_topo, zhat_dot_topo))
    
    num_particles = X_ot.shape[0];
    X_bt = np.zeros(X_ot.shape);
    for particle in range(num_particles):
        rot_topo = X_ot[particle, :3] # First three components of the state vector
        vot_topo = X_ot[particle, 3:] # Last three components of the state vector
        
        rot_ECI = la.inv(A)@rot_topo
        vot_ECI = la.inv(A)@(vot_topo - dA_dt@rot_ECI)
    
        # Calculating X_{ET} in the synodic frame with our above quantities
        
        ret_ECI = reo_nondim + rot_ECI
        vet_ECI = veo_nondim + vot_ECI
    
        R3 = np.array([[np.cos(t_stamp), -np.sin(t_stamp), 0], [np.sin(t_stamp), np.cos(t_stamp), 0], [0, 0, 1]])
        dR3_dt = np.array([[-np.sin(t_stamp), -np.cos(t_stamp), 0], [np.cos(t_stamp), -np.sin(t_stamp), 0], [0, 0, 0]])
    
        ret_S = la.inv(R3)@ret_ECI
        vet_S = la.inv(R3)@(vet_ECI - dR3_dt@ret_S)
    
        r_be = np.array([-norm_quantities["mu"], 0, 0])
        v_be = np.array([0, 0, 0])
    
        r_bt = r_be + ret_S # In synodic reference frame
        v_bt = v_be + vet_S # In synodic reference frame
    
        X_bt[particle, :] = np.hstack((r_bt, v_bt))
    return X_bt
'''
def Topo2Synodic(X_ot, t_stamp, obs_lat, obs_lon, obs_el, norm_quantities):
    # First step: Obtain X_{eo}**{ECI} 
    UTC_vec_orig = dt.datetime(2024, 5, 3, 2, 41, 15, tzinfo=dt.timezone.utc) # Initial UTC vector at t_stamp = 0
    t_add_dim = t_stamp * norm_quantities["time2hr"]/24 # Convert the time to add to a dimensional quantity
    UTC_vec = UTC_vec_orig + dt.timedelta(t_add_dim) # You will need this for calculating r_{eo} and v_{eo}
    

    delt_add_dim = t_add_dim - 1/86400
    delt_updatedUTCtime = UTC_vec_orig + dt.timedelta(delt_add_dim)

    reo_dim = pymap3d.geodetic2eci(obs_lat, obs_lon, obs_el, UTC_vec)
    delt_reodim = pymap3d.geodetic2eci(obs_lat, obs_lon, obs_el, delt_updatedUTCtime)
    reo_dim = np.asarray(reo_dim).reshape(3)
    delt_reodim = np.asarray(delt_reodim).reshape(3)
    veo_dim = reo_dim - delt_reodim # Finite difference

    reo_nondim = np.zeros(reo_dim.shape[0])
    veo_nondim = np.zeros(veo_dim.shape[0])
    reo_nondim[:] = reo_dim[:]/(1000*norm_quantities["dist2km"]) # Conversion to non-dimensional units in the ECI frame
    veo_nondim[:] = veo_dim[:]/(1000*norm_quantities["vel2kms"]) # Conversion to non-dimensional units in the ECI frame

    z_hat_topo = reo_nondim/la.norm(reo_nondim)
    x_hat_topo = np.cross(z_hat_topo, np.array([0, 0, 1]))/la.norm(np.cross(z_hat_topo, np.array([0, 0, 1])))
    y_hat_topo = np.cross(x_hat_topo, z_hat_topo)/la.norm(np.cross(x_hat_topo, z_hat_topo))
    
    A = np.vstack((x_hat_topo, y_hat_topo, z_hat_topo)) # Computing A as DCM for transforming between ECI and topographic reference frame

    dmag_dt = np.dot(reo_nondim, veo_nondim)/la.norm(reo_nondim)
    
    zhat_dot_topo = (veo_nondim * la.norm(reo_nondim) - reo_nondim * dmag_dt)/(la.norm(reo_nondim))**2
    xhat_dot_topo = np.cross(zhat_dot_topo, np.array([0, 0, 1]))/la.norm(np.cross(z_hat_topo, np.array([0, 0, 1]))) - np.dot(x_hat_topo, np.cross(zhat_dot_topo, np.array([0, 0, 1])))*x_hat_topo
    yhat_dot_topo = (np.cross(xhat_dot_topo, z_hat_topo) + np.cross(x_hat_topo, zhat_dot_topo))/la.norm(np.cross(x_hat_topo, z_hat_topo)) - np.dot(y_hat_topo, np.cross(xhat_dot_topo, z_hat_topo) + np.cross(x_hat_topo, zhat_dot_topo))*y_hat_topo

    dA_dt = np.vstack((xhat_dot_topo, yhat_dot_topo, zhat_dot_topo))
    
    #num_particles = X_ot.shape[0];
    X_bt = np.zeros(X_ot.shape);
    rot_topo = X_ot[:, :3] # First three components of the state vector
    vot_topo = X_ot[:, 3:] # Last three components of the state vector
    
    A_inv = la.inv(A)
    rot_ECI = rot_topo@A_inv.T
    vot_ECI = (vot_topo - rot_ECI@dA_dt.T)@A_inv.T

    # Calculating X_{ET} in the synodic frame with our above quantities
    
    ret_ECI = reo_nondim + rot_ECI
    vet_ECI = veo_nondim + vot_ECI

    R3 = np.array([[np.cos(t_stamp), -np.sin(t_stamp), 0], 
                   [np.sin(t_stamp), np.cos(t_stamp), 0], 
                   [0, 0, 1]])
    dR3_dt = np.array([[-np.sin(t_stamp), -np.cos(t_stamp), 0], 
                       [np.cos(t_stamp), -np.sin(t_stamp), 0], 
                       [0, 0, 0]])

    R3_inv = la.inv(R3)
    ret_S = ret_ECI @ R3_inv.T
    vet_S = (vet_ECI - ret_S @ dR3_dt.T) @ R3_inv.T

    r_be = np.array([-norm_quantities["mu"], 0, 0])
    v_be = np.array([0, 0, 0])

    r_bt = r_be + ret_S # In synodic reference frame
    v_bt = v_be + vet_S # In synodic reference frame

    #X_bt[particle, :] = np.hstack((r_bt, v_bt))
    X_bt = np.hstack((r_bt, v_bt))
    return X_bt
'''
def Synodic2Topo(X_bt, t_stamp, obs_lat, obs_lon, obs_el, norm_quantities):
    # Insert code for obtaining vector between center of Earth and observer
    rbe = np.array([-norm_quantities["mu"], 0, 0]) # Position vector relating center of earth to barycenter

    UTC_vec_orig = dt.datetime(2024, 5, 3, 2, 41, 15, tzinfo=dt.timezone.utc) # Initial UTC vector at t_stamp = 0
    t_add_dim = t_stamp * norm_quantities["time2hr"]/24 # Convert the time to add to a dimensional quantity
    UTC_vec = UTC_vec_orig + dt.timedelta(t_add_dim) # You will need this for calculating r_{eo} and v_{eo}

    delt_add_dim = t_add_dim - 1/86400
    delt_updatedUTCtime = UTC_vec_orig + dt.timedelta(delt_add_dim)

    reo_dim = pymap3d.geodetic2eci(obs_lat, obs_lon, obs_el, UTC_vec)
    delt_reodim = pymap3d.geodetic2eci(obs_lat, obs_lon, obs_el, delt_updatedUTCtime)
    reo_dim = np.asarray(reo_dim).reshape(3)
    delt_reodim = np.asarray(delt_reodim).reshape(3)
    veo_dim = reo_dim - delt_reodim # Finite difference

    R_z = np.array([[np.cos(t_stamp), -np.sin(t_stamp), 0], [np.sin(t_stamp), np.cos(t_stamp), 0], [0, 0, 1]])
    dRz_dt = np.array([[-np.sin(t_stamp), -np.cos(t_stamp), 0], [np.cos(t_stamp), -np.sin(t_stamp), 0], [0, 0, 0]])

    reo_nondim = np.zeros(reo_dim.shape)
    veo_nondim = np.zeros(veo_dim.shape)
    reo_nondim[:] = reo_dim[:]/(1000*norm_quantities["dist2km"]) # Conversion to non-dimensional units in the ECI frame
    veo_nondim[:] = veo_dim[:]/(1000*norm_quantities["vel2kms"])
    
    
    num_particles = X_bt.shape[0]
    X_ot = np.zeros(X_bt.shape)
    for particle in range(num_particles):
        rot_ECI = -reo_nondim + R_z@(-rbe + X_bt[particle, :3])
        vot_ECI = -veo_nondim + R_z@(X_bt[particle, 3:]) + dRz_dt@(-rbe + X_bt[particle, :3])
    
        # Finally, we convert from the ECI frame to the topographic frame
    
        # Step 1: Find the unit vectors governing this topocentric frame
        z_hat_topo = reo_nondim/la.norm(reo_nondim)
    
        x_hat_topo_unorm = np.cross(z_hat_topo, np.array([0, 0, 1])) # We choose a 
        # reference vector such as the North Pole, but we have several 
        # choices regarding the second vector
      
        x_hat_topo = x_hat_topo_unorm/la.norm(x_hat_topo_unorm) # Remember to normalize
    
        y_hat_topo_unorm = np.cross(x_hat_topo, z_hat_topo)
        y_hat_topo = y_hat_topo_unorm/la.norm(y_hat_topo_unorm) # Remember to normalize
        
        # reo_topo = np.array([np.dot(reo_nondim, x_hat_topo), np.dot(reo_nondim, y_hat_topo), np.dot(reo_nondim, z_hat_topo)])
    
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
    
        X_ot[particle, :] = np.hstack([rot_topo, vot_topo])
    return X_ot
'''

def Synodic2Topo(X_bt, t_stamp, obs_lat, obs_lon, obs_el, norm_quantities):
    # Insert code for obtaining vector between center of Earth and observer
    rbe = np.array([-norm_quantities["mu"], 0, 0]) # Position vector relating center of earth to barycenter

    UTC_vec_orig = dt.datetime(2024, 5, 3, 2, 41, 15, tzinfo=dt.timezone.utc) # Initial UTC vector at t_stamp = 0
    t_add_dim = t_stamp * norm_quantities["time2hr"]/24 # Convert the time to add to a dimensional quantity
    UTC_vec = UTC_vec_orig + dt.timedelta(t_add_dim) # You will need this for calculating r_{eo} and v_{eo}

    delt_add_dim = t_add_dim - 1/86400
    delt_updatedUTCtime = UTC_vec_orig + dt.timedelta(delt_add_dim)

    reo_dim = pymap3d.geodetic2eci(obs_lat, obs_lon, obs_el, UTC_vec)
    delt_reodim = pymap3d.geodetic2eci(obs_lat, obs_lon, obs_el, delt_updatedUTCtime)
    reo_dim = np.asarray(reo_dim).reshape(3)
    delt_reodim = np.asarray(delt_reodim).reshape(3)
    veo_dim = reo_dim - delt_reodim # Finite difference

    R_z = np.array([[np.cos(t_stamp), -np.sin(t_stamp), 0], [np.sin(t_stamp), np.cos(t_stamp), 0], [0, 0, 1]])
    dRz_dt = np.array([[-np.sin(t_stamp), -np.cos(t_stamp), 0], [np.cos(t_stamp), -np.sin(t_stamp), 0], [0, 0, 0]])

    reo_nondim = np.zeros(reo_dim.shape)
    veo_nondim = np.zeros(veo_dim.shape)
    reo_nondim[:] = reo_dim[:]/(1000*norm_quantities["dist2km"]) # Conversion to non-dimensional units in the ECI frame
    veo_nondim[:] = veo_dim[:]/(1000*norm_quantities["vel2kms"])
    
    X_ot = np.zeros(X_bt.shape)
    
    rot_ECI = (-reo_nondim + (R_z@(-rbe + X_bt[:, :3]).T).T)
    vot_ECI = (-veo_nondim + (R_z@X_bt[:, 3:].T).T + (dRz_dt@(-rbe + X_bt[:, :3]).T).T)
    
    # Finally, we convert from the ECI frame to the topographic frame

    # Step 1: Find the unit vectors governing this topocentric frame
    z_hat_topo = reo_nondim/la.norm(reo_nondim)

    x_hat_topo_unorm = np.cross(z_hat_topo, np.array([0, 0, 1])) # We choose a 
    # reference vector such as the North Pole, but we have several 
    # choices regarding the second vector
  
    x_hat_topo = x_hat_topo_unorm/la.norm(x_hat_topo_unorm) # Remember to normalize

    y_hat_topo_unorm = np.cross(x_hat_topo, z_hat_topo)
    y_hat_topo = y_hat_topo_unorm/la.norm(y_hat_topo_unorm) # Remember to normalize
    
    
    R_topo = np.vstack((x_hat_topo, y_hat_topo, z_hat_topo)) # DCM relating ECI to topocentric coordinate frame
    rot_topo = rot_ECI @ R_topo.T
    
    # Step 3: Handle the time derivatives of vot_topo = d/dt (rot_topo)
    rmag = la.norm(reo_nondim)
    dmag_dt = np.dot(reo_nondim, veo_nondim)/rmag # How the magnitude of r_eo changes w.r.t. time
    
    zhat_dot_topo = (veo_nondim*rmag - reo_nondim*dmag_dt)/rmag**2
    xhat_dot_topo = np.cross(zhat_dot_topo, np.array([0, 0, 1]))/la.norm(np.cross(z_hat_topo, np.array([0, 0, 1]))) - np.dot(x_hat_topo, np.cross(zhat_dot_topo, np.array([0, 0, 1])))*x_hat_topo
    yhat_dot_topo = (np.cross(xhat_dot_topo, z_hat_topo) + np.cross(x_hat_topo, zhat_dot_topo))/la.norm(np.cross(x_hat_topo, z_hat_topo)) - np.dot(y_hat_topo, np.cross(xhat_dot_topo, z_hat_topo) + np.cross(x_hat_topo, zhat_dot_topo))*y_hat_topo
    
    dA_dt = np.vstack((xhat_dot_topo, yhat_dot_topo, zhat_dot_topo))
    vot_topo = vot_ECI@R_topo.T + rot_ECI@dA_dt.T
    
    X_ot = np.hstack([rot_topo, vot_topo])
    return X_ot


def Synodic2ECI(X_bt, t_stamp, obs_lat, obs_lon, obs_el, norm_quantities):
    # Insert code for obtaining vector between center of Earth and observer
    rbe = np.array([-norm_quantities["mu"], 0, 0]) # Position vector relating center of earth to barycenter

    UTC_vec_orig = dt.datetime(2024, 5, 3, 2, 41, 15, tzinfo=dt.timezone.utc) # Initial UTC vector at t_stamp = 0
    t_add_dim = t_stamp * norm_quantities["time2hr"]/24 # Convert the time to add to a dimensional quantity
    UTC_vec = UTC_vec_orig + dt.timedelta(t_add_dim) # You will need this for calculating r_{eo} and v_{eo}

    delt_add_dim = t_add_dim - 1/86400
    delt_updatedUTCtime = UTC_vec_orig + dt.timedelta(delt_add_dim)

    reo_dim = pymap3d.geodetic2eci(obs_lat, obs_lon, obs_el, UTC_vec)
    delt_reodim = pymap3d.geodetic2eci(obs_lat, obs_lon, obs_el, delt_updatedUTCtime)
    reo_dim = np.asarray(reo_dim).reshape(3)
    delt_reodim = np.asarray(delt_reodim).reshape(3)
    veo_dim = reo_dim - delt_reodim # Finite difference

    R_z = np.array([[np.cos(t_stamp), -np.sin(t_stamp), 0], [np.sin(t_stamp), np.cos(t_stamp), 0], [0, 0, 1]])
    dRz_dt = np.array([[-np.sin(t_stamp), -np.cos(t_stamp), 0], [np.cos(t_stamp), -np.sin(t_stamp), 0], [0, 0, 0]])

    reo_nondim = np.zeros(reo_dim.shape)
    veo_nondim = np.zeros(veo_dim.shape)
    reo_nondim[:] = reo_dim[:]/(1000*norm_quantities["dist2km"]) # Conversion to non-dimensional units in the ECI frame
    veo_nondim[:] = veo_dim[:]/(1000*norm_quantities["vel2kms"])
    
    X_ECI = np.zeros(X_bt.shape)
    
    rot_ECI = (R_z@(-rbe + X_bt[:, :3]).T).T
    vot_ECI = (R_z@X_bt[:, 3:].T).T + (dRz_dt@(-rbe + X_bt[:, :3]).T).T
    
    X_ECI = np.hstack([rot_ECI, vot_ECI])
    return X_ECI
