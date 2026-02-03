# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 12:45:17 2026

@author: tarun
"""
from pathlib import Path
import numpy as np
from numpy import linalg as la
import scipy as sci
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import Dynamics as dyn
import CoordFunctions as cf
import PlottingFunctions as plot


#%% Parameters
save_loc = Path("D:/PythonProjects/EDP/PGM/ParticleFusionTest/12_15_25_meeting/Matlab2Python/Test18/OrbitDataExp/Agent3")

orbit_choice = "L2" # Options: L2, 9:2 NRHO, Planar Mirror

# College Station
#obs_lat = 30.618963;
#obs_lon = -96.339214;
#obs_el = 103.8;
# Buenos Aires
#obs_lat = -34.5
#obs_lon = -58
#obs_el = 103.8;
# New Zealand
obs_lat = -40;
obs_lon = 175;
obs_el = 103.8;

h = lambda x: np.array([la.norm(x[:, :3], axis=1),
                        np.arctan2(x[:, 1],x[:, 0]),
                        np.pi/2 - np.arccos(x[:, 2]/la.norm(x[:, :3], axis=1))]).T
dist2km = 384400 # Kilometers per non-dimensionalized distance
time2hr = 4.342*24 # Hours per non-dimensionalized time
vel2kms = dist2km/(time2hr*60*60) # Kms per non-dimensionalized velocity
norm_quantities = {"dist2km": dist2km,
                            "vel2kms": vel2kms,
                            "time2hr": time2hr,
                            "mu": 1.2150582e-2}

#x0 = [-0.144158380406153	-0.000697738382717277	0	0.0100115754530300	-3.45931892135987	0]; # Planar Mirror Orbit "Loop-Dee-Loop" Sub-Trajectory
if orbit_choice == "9:2 NRHO":
    x0 = [1.0221, 0, -0.1821, 0, -0.1033, 0];
if orbit_choice == "L2":
    x0 = [1.16429257222878, -0.0144369085836121, 0, -0.0389308426824481, 0.0153488211249537, 0]; # L2 Lagrange Point Approach
#x0 = [1.16961297958960	-0.0154599483859532	0	-0.0506271631673632	0.00166461443708329	0]; # L2 Lagrange Point Approach (after 60 hours)


#%% Generate timespan for orbit
t_segment_starts = [0, 40, 80, 250, 300] # Segment lengths (hrs)
t_stepsize = [2, 2, 8, 8] # Step sizes for each segment (hrs)
segment_has_msmts = [True, False, False, True]

t_temp = np.concatenate([np.arange(t0, t1, dt) for t0, t1, dt in zip(t_segment_starts[:-1], t_segment_starts[1:], t_stepsize)])

t_span = np.append(t_temp, t_segment_starts[-1]) / norm_quantities['time2hr']


#%% Generate object trajectory for timespan
integration_results = sci.integrate.solve_ivp(dyn.cr3bp_dyn, (t_span[0], t_span[-1]), x0, method='RK45', t_eval=t_span, rtol=1e-6, atol=1e-8)
obj_syn = integration_results.y.T

rem = np.array([[1, 0, 0, 0, 0, 0]]) # Earth center - moon center
rbe = np.array([[-norm_quantities['mu'], 0, 0, 0, 0, 0]]) # State vector relating center of earth to barycenter

moon_eci_pos = np.full((t_span.shape[0], 3), np.nan)
obj_eci = np.full((t_span.shape[0], 6), np.nan)
obj_topo = np.full((t_span.shape[0], 6), np.nan)
for t in range(t_span.shape[0]):
    moon_eci_temp = cf.Synodic2ECI(rem+rbe, t_span[t] - t_span[0], obs_lat, obs_lon, obs_el, norm_quantities)
    moon_eci_pos[t, :] = moon_eci_temp[0, :3]
    
    obj_eci[t, :] = cf.Synodic2ECI(obj_syn[t, :].reshape(1, -1), t_span[t] - t_span[0], obs_lat, obs_lon, obs_el, norm_quantities)
    obj_topo[t, :] = cf.Synodic2Topo(obj_syn[t, :].reshape(1, -1), t_span[t] - t_span[0], obs_lat, obs_lon, obs_el, norm_quantities)

obj_msmt = h(obj_topo)

#%% Measurement mask generation

# convert to arrays
t_span = np.array(t_span)
segment_has_msmts = np.array(segment_has_msmts, dtype=bool)

segment_msmt_mask = np.zeros_like(t_span, dtype=bool)

# find indices that separate segments
indices = np.searchsorted(t_span*norm_quantities['time2hr'], t_segment_starts)

for i, flag in enumerate(segment_has_msmts):
    start_idx = indices[i]
    end_idx   = indices[i+1]
    segment_msmt_mask[start_idx:end_idx] = flag

elevation_mask = obj_msmt[:, 2] > -100

msmt_mask = np.logical_and(segment_msmt_mask, elevation_mask)

obj_syn_msmt = np.where(msmt_mask[:, None], obj_syn, np.nan)
#obj_msmt = np.where(msmt_mask[:, None], obj_msmt, np.nan)

#%% Final 

partial_ts = np.hstack((t_span[msmt_mask].reshape(-1, 1), obj_msmt[msmt_mask, :]))
full_ts = np.hstack((t_span.reshape(-1, 1), obj_topo[:, :3]))
full_vts = np.hstack((t_span.reshape(-1, 1), obj_topo[:, 3:]))

np.save(save_loc / "partial_ts", partial_ts)
np.save(save_loc / "full_ts", full_ts)
np.save(save_loc / "full_vts", full_vts)

#%% Plot Synodic Position Evolution
labels = ['x-Position', 'y-Position', 'z-Position']
colors = ['r', 'r', 'r']

fig = plt.figure(figsize=(8, 10))

for i, (label, color) in enumerate(zip(labels, colors), start=1):
    ax = plt.subplot(3, 1, i)
    ax.plot(t_span * norm_quantities["time2hr"], obj_syn[:, i-1]*norm_quantities['dist2km'], color + '-')
    #ax.scatter(t_span * norm_quantities["time2hr"], obj_syn_msmt[:, i-1]*norm_quantities['dist2km'])
    # y-axis label in scientific notation
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis='y', scilimits=(0,0))
    
    ax.set_ylabel(label + " (km)")
    #ax.set_title(f"{label}")
    ax.grid(True)
    
    if i < 3:
        ax.set_xticklabels([])

plt.xlabel("Time (hr)")
plt.suptitle(orbit_choice + " Synodic Evolution")
plt.tight_layout()
plot.save_and_close(fig, save_loc / "Evolution.png")


labels = ['Range (km)', 'Azimuth (deg)', 'Elevation (deg)']
colors = ['r', 'r', 'r']
msmt_conversion = [norm_quantities['dist2km'], 180/np.pi, 180/np.pi]
fig = plt.figure(figsize=(8, 10))

for i, (label, color) in enumerate(zip(labels, colors), start=1):
    ax = plt.subplot(3, 1, i)
    ax.scatter(partial_ts[:, 0] * norm_quantities["time2hr"], partial_ts[:, i] * msmt_conversion[i-1])
    # y-axis label in scientific notation
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    if i == 1:
        ax.ticklabel_format(style="sci", axis='y', scilimits=(0,0))
    
    ax.set_ylabel(label)
    #ax.set_title(f"{label}")
    ax.grid(True)
    
    if i < 3:
        ax.set_xticklabels([])

plt.xlabel("Time (hr)")
plt.suptitle(orbit_choice + " Observer Measurements (Noiseless)")
plt.tight_layout()
plot.save_and_close(fig, save_loc / "Observations.png")
