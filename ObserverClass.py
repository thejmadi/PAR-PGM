# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:28:01 2026

@author: tarun
"""

from dataclasses import dataclass, field
import numpy as np

@dataclass
class Observer:
    name: str
    is_orig_obs: bool
    linestyle: str
    
    # Location data
    lat: float
    lon: float
    el: float
    
    max_particles: int
    K: int
    
    topo_truth: np.array = field(init=False)
    topo_cloud_prior: np.ndarray = field(init=False)
    topo_cloud_post: np.ndarray = field(init=False)
    
    plot_indv_clouds: bool
    plot_combined_clouds: bool
    
    def n_particles(self, prior_or_post) -> int:
        if prior_or_post == "prior":
            num = self.topo_cloud_prior.shape[0]
        if prior_or_post == "post":
            num = self.topo_cloud_post.shape[0]
        return num

