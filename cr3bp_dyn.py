# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 14:32:18 2025

@author: tarun
"""

import numpy as np

def cr3bp_dyn(t, x):
    # Target dynamics
    mu = 1.2150582e-2; # Dimensionless mass of the moon
    r1 = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2);
    r2 = np.sqrt((x[0] - 1 + mu)**2 + x[1]**2 + x[2]**2);
    
    cx = 1 - (1-mu)/r1**3 - mu/r2**3;
    cy = 1 - (1 - mu)/r1**3 - mu/r2**3;
    cz = -((1 - mu)/r1**3 + mu/r2**3);
    
    bx = (mu - mu**2)/r1**3 + (-mu + mu**2)/r2**3;
    
    dx_dt = np.array([x[3], x[4], x[5], cx*x[0]+2*x[4]-bx, cy*x[1]-2*x[3], cz*x[2]])
    # dx_dt = [x(4), x(5), x(6), 2*x(5) + x(1) - (1-mu)*(x(1)+mu)/r1**3 - mu*(x(1)-1+mu)/r2**3, cy*x(2)-2*x(4), cz*x(3)]';
    return dx_dt