# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:48:25 2025

@author: tarun, ipparanjape
"""

import numpy as np
from numpy import linalg as la
import scipy as sci
from scipy import io as sio
import datetime as dt
import pymap3d
import cr3bp_dyn as cr3bp

def termSat(T, Y):
    mu = 1.2150582e-2 # Dimensionless mass of the moon (and position of Earth w.r.t. barycenter)
    Rm = 1740/384400 # Nondimensionalized radius of the moon
    value = (np.sqrt((Y[0] + mu)**2 + Y[1]**2 + Y[2]**2) < 6371/384400) or (np.sqrt((Y[0] - (1-mu))**2 + Y[1]**2 + Y[2]**2) < Rm) # Stop when the target hits the Earth's or the Moon's surface
    return 1
    
def appendObs(zvec, N):
    M = len(zvec[:,0])
    
    if (M < N): # Verify that the number of observations in a pass is no more than N.
        N = M
        
    startIndex = np.random.randint(1, M-N+1)
    passVec = zvec[startIndex:(startIndex+N),:]
    
    return passVec

save_path = "./"
# Define initial conditions
mu = 1.2150582e-2
# x0 = [0.5-mu, 0.0455, 0, -0.5, 0.5, 0.0]' # Sample Starting Point
# x0 = [-1.00506 0 0 -0.5 0.5 0.0]' # L3 Lagrange Point
# x0 = [0.5-mu, sqrt(3/4), 0, 0, 0, 0]' # L4 Lagrange Point

# Lagrange points that we may not use
# x0 = [0.836915 0 0 0 0 0]' # L1 Lagrange Point
# x0 = [1.15568 0 0 0 0 0]' # L2 Lagrange Point
# x0 = [0.5-mu, -sqrt(3/4), 0, 0, 0, 0]' # L5 Lagrange Point

# x0 = [-0.144158380406153	-0.000697738382717277	0	0.0100115754530300	-3.45931892135987	0] # Planar Mirror Orbit "Loop-Dee-Loop" Sub-Trajectory
# x0 = np.array([1.0221, 0, -0.1821, 0, -0.1033, 0]) # 9:2 Resonant NRHO (i.e. Gateway orbit)
x0 = np.array([1.16429257222878, -0.0144369085836121, 0, -0.0389308426824481, 0.0153488211249537, 0]) # L2 Lagrange Point Approach

# Coordinate system conversions
dist2km = 384400 # Kilometers per non-dimensionalized distance
time2hr = 4.342*24 # Hours per non-dimensionalized time
vel2kms = dist2km/(time2hr*60*60) # Kms per non-dimensionalized velocity

# Multi-target tracking initializations
Nt = 2 # Number of targets
sdx = 1000 # Separation distance between targets (barycentric x-coordinate)
vdx = 0 # Difference in velocities between targets (barycentric x-coordinate)
Q0 = np.square(np.diagflat([sdx/dist2km, 0, 0, vdx/vel2kms, 0, 0]))

X0 = np.zeros((Nt, len(x0)))
X0[0,:] = np.random.multivariate_normal(x0, Q0)

for j in range(1,Nt):
    v = np.random.rand(2)
    v = v/la.norm(v)
    
    X0[j,:] = np.copy(X0[0,:])
    X0[j,0:2] = X0[0,0:2] + sdx/dist2km*v

# Define time span
tstamp1 = 0 # For long term trajectories 
# end_t = 1.5112 - tstamp1
end_t = 36/time2hr - tstamp1
tstamp = end_t
tspan = np.arange(0, end_t, 6.25e-3) # For our modified trajectory

# Call ode45()

# [t,dx_dt] = ode45(@cr3bp_dyn, tspan, x0, opts); # Assumes termination event (i.e. target enters LEO)

Dx_Dt = np.zeros((Nt, tspan.shape[0], x0.shape[0]))
Dx_Dt[:,0,:] = np.copy(X0)
t = np.zeros((tspan.shape[0],1))
    
termSat.terminal = True
termSat.direction = 0

for j in range(Nt):
    for i in range(tspan.shape[0]-1):
        ivp_res = sci.integrate.solve_ivp(cr3bp.cr3bp_dyn, (0, tspan[i+1]-tspan[i]), np.copy(X0[j,:]), method='BDF', events=termSat, rtol=1e-6, atol=1e-8)
        X0[j,:] = ivp_res.y.T[-1,:]
        Dx_Dt[j,i+1,:] = np.copy(X0[j,:])
        t[i+1] = tspan[i+1]

# tstamp = 40*24/time2hr;

# Longer-term scheduling

tstamp = t[-1] # Begin new trajectory where we left off
end_t = (40*24)/time2hr
tspan = np.arange(tstamp[0], end_t, 8/time2hr) # Schedule to take measurements once every 8 hours

X0_tmp = np.copy(np.squeeze(Dx_Dt[:,-1,:]))
# x0_tmp[:] = dx_dt[-1,:]
t = t[:-1].reshape(-1)
Dx_Dt = np.copy(Dx_Dt[:,:-1,:])

Dx_Dts = np.zeros((Nt,tspan.shape[0], x0.shape[0]))
Dx_Dts[:,0,:] = np.copy(X0_tmp) # Start at end of previous schedule

ts = np.zeros((tspan.shape[0],1))
ts[0] = tstamp

termSat.terminal = True
termSat.direction = 0

for j in range(Nt):
    for i in range(tspan.shape[0]-1):
        ivp_res = sci.integrate.solve_ivp(cr3bp.cr3bp_dyn, (tspan[i], tspan[i+1]), np.copy(X0_tmp[j,:]), method='BDF', events=termSat, rtol=1e-6, atol=1e-8)
        X0_tmp[j,:] = ivp_res.y.T[-1,:]
        Dx_Dts[j,i+1,:] = np.copy(X0_tmp[j,:])
        ts[i+1] = ivp_res.t[-1]


t = np.hstack((t, ts.reshape(-1)))
Dx_Dt = np.concatenate((Dx_Dt, Dx_Dts), axis=1)

Rb = Dx_Dt[:,:,:3] # Position evolutions from barycenter
Vb = Dx_Dt[:,:,3:] # Velocity evolutions from barycenter
rbe = np.array([[-mu], [0], [0]]) # Position vector relating center of earth to barycenter

# Insert code for obtaining vector between center of Earth and observer (Location: College Station, TX)

obs_lat = 30.618963
obs_lon = -96.339214
elevation = 103.8

# dtLCL = datetime('now', 'TimeZone','local'); # Current Local Time
# dtUTC = datetime(dtLCL, 'TimeZone','Z');     # Current UTC Time
# UTC_vec = datevec(dtUTC); # Convert to vector

UTC_vec = dt.datetime(2024, 5, 3, 2, 41, 15, tzinfo=dt.timezone.utc)
t_add_dim = tstamp1 * (4.342)
UTC_vec = UTC_vec + dt.timedelta(t_add_dim)

# reo_dim = lla2eci([obs_lat obs_lon elevation], UTC_vec); # Position vector between observer and center of Earth in meters
# reo_nondim = reo_dim/(1000*384400);

rem = np.array([[1], [0], [0]]) # Earth center - moon center 

Reo_nondim = np.zeros((Nt,t.shape[0],3))
Veo_nondim = np.zeros((Nt,t.shape[0],3)) # Array for non-dimensionalized EO velocity vectors

Rot = np.zeros(Rb.shape) # Observer - Target 
Rom = np.zeros(Rb.shape) # Observer - Moon Center
Vot = np.zeros(Rb.shape) # Observer - Target Velocity

# Conversion of Barycentric to ECI reference frame
for j in range(Nt):
    for i in range(t.shape[0]):
        t_add_nondim = np.squeeze(t[i] - tstamp1) # Time since first point of orbit
        # print(f"Current Time: {t[i,0]}, Non-dim Time Added: {t_add_nondim}, Type: {type(t_add_nondim)}")
        t_add_dim = t_add_nondim * (4.342) # Conversion to dimensionalized time
        delt_add_dim = t_add_dim - 1/86400 
    
        updated_UTCtime = UTC_vec + dt.timedelta(t_add_dim)
        
        delt_updatedUTCtime = UTC_vec + dt.timedelta(delt_add_dim)
        
        reo_dim = pymap3d.geodetic2eci(obs_lat, obs_lon, elevation, updated_UTCtime)
        delt_reodim = pymap3d.geodetic2eci(obs_lat, obs_lon, elevation, delt_updatedUTCtime)
        reo_dim = np.asarray(reo_dim).reshape(3)
        delt_reodim = np.asarray(delt_reodim).reshape(3)
        veo_dim = reo_dim - delt_reodim
        
        R_z = np.array([[np.cos(t_add_nondim + tstamp1), -np.sin(t_add_nondim + tstamp1), 0],
                        [np.sin(t_add_nondim + tstamp1), np.cos(t_add_nondim + tstamp1), 0],
                        [0, 0, 1]])
        
        dRz_dt = np.array([[-np.sin(t_add_nondim + tstamp1), -np.cos(t_add_nondim + tstamp1), 0], 
                           [np.cos(t_add_nondim + tstamp1), -np.sin(t_add_nondim + tstamp1), 0], 
                           [0, 0, 0]])
        
        Reo_nondim[j,i,:] = reo_dim[:]/(1000*384400) # Conversion to non-dimensional units and ECI frame
        Veo_nondim[j,i,:] = veo_dim[:]*(4.342*86400)/(1000*384400)
        
        # rb = np.squeeze(Rb[j,i,:])
        # vb = np.squeeze(Vb[j,i,:])
        
        
        
        Rot[j,i,:] = -np.squeeze(Reo_nondim[j,i,:]) + (R_z@(-rbe + np.squeeze(Rb[j,i,:]).reshape(rbe.shape))).reshape(-1)
        Rom[j,i,:] = -np.squeeze(Reo_nondim[j,i,:]) + (R_z@rem).reshape(-1)
        Vot[j,i,:] = -np.squeeze(Veo_nondim[j,i,:]) + (R_z@(Vb[j,i,:].reshape(rbe.shape))).reshape(-1) + (dRz_dt@(-rbe + np.squeeze(Rb[j,i,:]).reshape(rbe.shape))).reshape(-1)

# Before we obtain AZ and EL quantities, we must convert our
# observer-target vector into a topocentric frame.

Rot_topo = np.zeros((Nt, t.shape[0],Rot.shape[2]))
Rom_topo = np.zeros((Nt, t.shape[0],Rot.shape[2]))
Vot_topo = np.zeros((Nt, t.shape[0],Rot.shape[2]))

for j in range(Nt):
    reo_nondim = np.squeeze(Reo_nondim[j,:,:])
    veo_nondim = np.squeeze(Veo_nondim[j,:,:])
    rom = np.squeeze(Rom[j,:,:])
    rot = np.squeeze(Rot[j,:,:])
    vot = np.squeeze(Vot[j,:,:])
    
    for i in range(t.shape[0]):
        # Step 1: Find the unit vectors governing this topocentric frame
        z_hat_topo = reo_nondim[i,:]/la.norm(reo_nondim[i,:])
    
        x_hat_topo_unorm = np.cross(z_hat_topo, np.array([0, 0, 1])) # We choose a reference vector 
        x_hat_topo = x_hat_topo_unorm/la.norm(x_hat_topo_unorm) # Remember to normalize
        # such as the North Pole, we have several choices regarding this
        y_hat_topo_unorm = np.cross(x_hat_topo, z_hat_topo)
        y_hat_topo = y_hat_topo_unorm/la.norm(y_hat_topo_unorm) # Remember to normalize
    
        # Step 2: Convert all of the components of 'rot' from our aligned reference
        # frames to this new topocentric frame.
    
        rot_topo = np.array([np.dot(rot[i,:], x_hat_topo), np.dot(rot[i,:], y_hat_topo), np.dot(rot[i,:], z_hat_topo)])
        rom_topo = np.array([np.dot(rom[i,:], x_hat_topo), np.dot(rom[i,:], y_hat_topo), np.dot(rom[i,:], z_hat_topo)])
    
        # Step 3: Handle the time derivatives of vot_topo = d/dt (rot_topo)
        R_topo = np.vstack((x_hat_topo, y_hat_topo, z_hat_topo)) # DCM relating ECI to topocentric coordinate frame
        dmag_dt = np.dot(reo_nondim[i,:], veo_nondim[i,:])/la.norm(reo_nondim[i,:]) # How the magnitude of r_eo changes w.r.t. time
        
        zhat_dot_topo = (veo_nondim[i,:]*la.norm(reo_nondim[i,:]) - reo_nondim[i,:]*dmag_dt)/(la.norm(reo_nondim[i,:]))**2;
        xhat_dot_topo = np.cross(zhat_dot_topo, np.array([0, 0, 1]))/la.norm(np.cross(z_hat_topo, np.array([0, 0, 1]))) - np.dot(x_hat_topo, np.cross(zhat_dot_topo, np.array([0, 0, 1])))*x_hat_topo
        yhat_dot_topo = (np.cross(xhat_dot_topo, z_hat_topo) + np.cross(x_hat_topo, zhat_dot_topo))/la.norm(np.cross(x_hat_topo, z_hat_topo)) - np.dot(y_hat_topo, np.cross(xhat_dot_topo, z_hat_topo) + np.cross(x_hat_topo, zhat_dot_topo))*y_hat_topo
    
        dA_dt = np.vstack((xhat_dot_topo, yhat_dot_topo, zhat_dot_topo))
    
        vot_topo = (R_topo@vot[i,:].reshape((-1, 1)) + dA_dt@rot[i,:].reshape((-1, 1))).reshape(-1)
        
        Rot_topo[j,i,:] = np.copy(rot_topo)
        Rom_topo[j,i,:] = np.copy(rom_topo)
        Vot_topo[j,i,:] = np.copy(vot_topo)

# Due to not being able to see targets behindthe moon, design function such 
# that if the rot_topo vector passes through the Moon, then data for that 
# time step is considered invalid.

rot_valid = [[] for _ in range(Nt)]
vot_valid = [[] for _ in range(Nt)]
t_valid = [[] for _ in range(Nt)]

j = -1
Rm = 1740/384400 # Nondimensionalized radius of the moon

for k in range(Nt):
    rot_topo = np.squeeze(Rot_topo[k,:,:])
    vot_topo = np.squeeze(Vot_topo[k,:,:])
    rom_topo = np.squeeze(Rom_topo[k,:,:])
    
    for i in range(t.shape[0]):
        if (la.norm(np.cross(rot_topo[i,:], rom_topo[i,:]))/la.norm(rot_topo[i,:]) > Rm 
            and (t[i] <= tstamp or t[i] > (15*24)/time2hr)):
            j += 1
            t_valid[k].append(t[i])
            rot_valid[k].append(rot_topo[i,:])
            vot_valid[k].append(vot_topo[i,:])

for k in range(Nt):
    rot_valid[k] = np.asarray(rot_valid[k])
    vot_valid[k] = np.asarray(vot_valid[k])
    t_valid[k] = np.asarray(t_valid[k])

# Convert observer - target position vectors into range, azimuth, and
# elevation quantities

# Rho = np.zeros((rot_valid.shape[0],1))
# AZ = np.zeros((rot_valid.shape[0],1))
# EL = np.zeros((rot_valid.shape[0],1))

Rho = [[] for _ in range(Nt)]
AZ = [[] for _ in range(Nt)]
EL = [[] for _ in range(Nt)]

for j in range(Nt):
    for i in range(rot_valid[j].shape[0]):
        Rge = np.sqrt(rot_valid[j][i,0]**2 + rot_valid[j][i,1]**2 + rot_valid[j][i,2]**2)
        Rho[j].append(Rge)
        AZ[j].append(np.arctan2(rot_valid[j][i,1], rot_valid[j][i,0]))
        EL[j].append(np.pi/2 - np.arccos(rot_valid[j][i,2]/Rge))
    
    Rho[j] = np.asarray(Rho[j])
    AZ[j] = np.asarray(AZ[j])
    EL[j] = np.asarray(EL[j])
        
###### Progress Checkpoint #####

# Last Step: Due to elevation angle constraints, all t, Rho, AZ, EL data
# for which EL < 0 is considered invalid and should be discarded

full_ts = [[] for _ in range(Nt)]
partial_ts_ECI = [[] for _ in range(Nt)]
cTimes = [[] for _ in range(Nt)]

for j in range(Nt):
    full_ts[j] = np.vstack((t_valid[j], Rho[j], AZ[j], EL[j])) # Full augmented time-series vector
    full_ts[j] = full_ts[j].T

# first_msmt = np.where(full_ts[:,0] > 6)[0][0]

for j in range(Nt):
    valid_El = np.where(full_ts[j][:,3] > 0)[0]
    # valid_Az = np.where(np.abs(full_ts[:,2]) < 0.5*np.pi)[0]
    partial_ts_ECI[j] = full_ts[j][valid_El, :]

# Find the times at the beginning and end of each pass (i.e. the critical times)

for j in range(Nt):
    i = 1
    interval = full_ts[j][1,0] - full_ts[j][0,0]
    
    while i < partial_ts_ECI[j].shape[0]:
        if partial_ts_ECI[j][i,0] - partial_ts_ECI[j][i-1,0] > (interval+1e-11):
            cTimes[j].append([partial_ts_ECI[j][i-1,0], partial_ts_ECI[j][i,0]])
        i = i + 1

print(f"Critical Times: {cTimes}")

'''
# Now, we only extract P consecutive observations per pass
P = 5 # Number of consecutive observations in a single pass
q = 2 # Index for keeping track of the passes

eo1 = np.where(np.abs(cTimes[0] - partial_ts_ECI[:,0]) < (interval+1e-11))[0][0]
partial_ts_po = partial_ts_ECI[0:(eo1+1), :]

# print(f"End of First Pass Index: {eo1+1}, Time: {partial_ts_ECI[eo1+1,0]}")

while (q < len(cTimes)):
    pass_start = np.where(np.abs(cTimes[q-1] - partial_ts_ECI[:,0]) < (interval+1e-11))[0][0]
    pass_end = np.where(np.abs(cTimes[q] - partial_ts_ECI[:,0]) < (interval+1e-11))[0][0]
    obs_pass = partial_ts_ECI[pass_start:(pass_end+1), :]
    
    # print(f"Pass Starting Time: {partial_ts_ECI[pass_start,0]}, Pass Ending Time: {partial_ts_ECI[pass_end+1,0]}")
    consObs = appendObs(obs_pass, P)
    # print(f"Extracted Pass Shape: {consObs.shape}, Partial_TS_PO Shape: {partial_ts_po.shape}")
    partial_ts_po = np.vstack((partial_ts_po, consObs))
    
    q = q + 2

partial_ts_ECI = np.copy(partial_ts_po)
'''

np.savetxt(save_path + "partial_ts.csv", partial_ts_ECI, delimiter=',', fmt='%f')
np.savetxt(save_path + "full_ts.csv", np.hstack((t.reshape([-1, 1]), rot_topo)), delimiter=',', fmt='%f')
np.savetxt(save_path + "full_vts.csv", np.hstack((t.reshape([-1, 1]), vot_topo)), delimiter=',', fmt='%f')

