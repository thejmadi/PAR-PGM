# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 21:51:21 2026

@author: tarun
"""

from pathlib import Path
import json
import numpy as np
import PlottingFunctions as plot

folder_loc = Path("D:/PythonProjects/EDP/PGM/ParticleFusionTest/12_15_25_meeting/Matlab2Python/Test12/")
base_name = "pyMC2"
file_name = "metrics.npz"
linestyle = [["-", "--", "--", "--", "--"], ["-"]]
loaded = {}
normalization_quantities = {}
for folder in folder_loc.glob(f"{base_name}_*"):
    if folder.is_dir():
        metrics_path = folder / file_name
        nq_path = folder / "normalization_quantities.npz"
        if metrics_path.exists():
            with np.load(metrics_path) as data:
                loaded[folder.name] = dict(data)
            x = np.load(folder / "timesteps.npy")
            with np.load(nq_path) as data:
                normalization_quantities[folder.name] = dict(data)
            with open(folder / "cloud_names.json") as f:
                cloud_names = json.load(f)
            #with open(folder / "linestyle.json") as f:
            #    linestyle = json.load(f)
cloud_names[-1][0] = "Original Obs: 1 IOD: 40 hrs"
reshaped_loaded = {}
rearrange_list = ["likelihood_state_weighted", 
                  "likelihood_state_best", 
                  "likelihood_msmt_weighted", 
                  "likelihood_msmt_best",
                  "entropy_state_weighted",
                  "entropy_msmt_weighted",
                  "entropy_msmt_best",
                  "AIC_state",
                  "AIC_msmt",
                  "num_cluster",
                  "num_particles",
                  "RMSE"]
for folder_name, metrics in loaded.items():
    reshaped_loaded[folder_name] = {}
    
    for metric_name, arr in metrics.items():
        if metric_name in rearrange_list:
            # arr has shape (N, M, T, d)
            part1 = arr[:, 0, :]   # shape (N, T, d)
            part2 = arr[0, 1:, :]  # shape (M-1, T, d)
            
            # Stack along a new first axis (or whatever axis you prefer)
            new_arr = np.concatenate([part1, part2], axis=0)
            reshaped_loaded[folder_name][metric_name] = new_arr
        else:
            reshaped_loaded[folder_name][metric_name] = loaded[folder_name][metric_name][0, :]

part1 = [cloud_names[n][0] for n in range(len(cloud_names))]
part2 = [cloud_names[0][m] for m in range(1, len(cloud_names[0]))]
name = part1 + part2
part1 = [linestyle[n][0] for n in range(len(linestyle))]
part2 = [linestyle[0][m] for m in range(1, len(linestyle[0]))]
ls = part1 + part2

metric_names = list(next(iter(reshaped_loaded.values())).keys())
averaged_metrics = {}
for metric in metric_names:
    # Stack the metric arrays from all folders along a new axis 0
    stacked = np.stack([reshaped_loaded[folder][metric] for folder in reshaped_loaded], axis=0)
    
    # Compute mean across axis 0 (i.e., across all folders)
    averaged_metrics[metric] = np.mean(stacked, axis=0)

print()

save_loc = folder_loc/"MC_Results2"


plot.plotMetrics(x, averaged_metrics["likelihood_state_weighted"], name, ls, save_loc, 0, "Log-Likelihood", f"GMM: Full, State Space, Log-Likelihood  vs. Time", "likelihood_state_weighted.png")

plot.plotMetrics(x, averaged_metrics["likelihood_state_best"], name, ls, save_loc, 0, "Log-Likelihood", f"GMM: Best Mode, State Space, Log-Likelihood  vs. Time", "likelihood_state_best.png")
plot.plotMetrics(x, averaged_metrics["likelihood_msmt_weighted"], name, ls, save_loc, 0, "Log-Likelihood", f"GMM: Full, Msmt Space, Log-Likelihood  vs. Time", "likelihood_msmt_weighted.png")
plot.plotMetrics(x, averaged_metrics["likelihood_msmt_best"], name, ls, save_loc, 0, "Log-Likelihood", f"GMM: Best Mode, Msmt Space, Log-Likelihood  vs. Time", "likelihood_msmt_best.png")

# Entropy metrics
plot.plotMetrics(x, averaged_metrics["entropy_state_weighted"], name, ls, save_loc, 0, "Entropy", f"GMM: Full, State Space, Entropy  vs. Time", "entropy_state_weighted.png")
#plot.plotMetrics(x, averaged_metrics["entropy_state_best"], name, save_loc, 0, "Entropy", f"GMM: Best Mode, State Space, Entropy  vs. Time", "entropy_state_best.png")
plot.plotMetrics(x, averaged_metrics["entropy_msmt_weighted"], name, ls, save_loc, 0, "Entropy", f"GMM: Full, Msmt Space, Entropy  vs. Time", "entropy_msmt_weighted.png")
plot.plotMetrics(x, averaged_metrics["entropy_msmt_best"], name, ls, save_loc, 0, "Entropy", f"GMM: Best Mode, Msmt Space, Entropy  vs. Time", "entropy_msmt_best.png")

# AIC metrics
plot.plotMetrics(x, averaged_metrics["AIC_state"], name, ls, save_loc, 0, "AIC", f"State Space, AIC  vs. Time", "AIC_state.png")
plot.plotMetrics(x, averaged_metrics["AIC_msmt"], name, ls, save_loc, 0, "AIC", f"Msmt Space, AIC  vs. Time", "AIC_msmt.png")

# Misc metrics
plot.plotMetrics(x, averaged_metrics["num_cluster"], name, ls, save_loc, 0, "Number of Clusters", f"Number of Metric Clusters  vs. Time", "num_clusters.png")
plot.plotMetrics(x, averaged_metrics["num_particles"], name, ls, save_loc, 0, "Number of Particles", f"Number of Particles  vs. Time", "num_particles.png")

#plot.plotMetrics(x, averaged_metrics["ill_conditioned"], name, save_loc, 0, "Number of Particles", f"Number of Particles  vs. Time", "num_particles.png")
# TODO: Fix next plot
#plot.plotMetricsPerState(x, averaged_metrics["RMSE"], normalization_quantities, name, ls, save_loc, 0, "RMSE (km, kms)", "RMSE Slice vs. Time", "RMSE")



name = [row[0] for row in cloud_names][1:]
ls = [row[0] for row in linestyle][1:]

plot.plotMetrics(x, averaged_metrics["avg_ob_ob_likeli"][1:], name, ls, save_loc, 0, "Ob-Ob Log-Likelihood", f"Ob {0} to Ob Likelihood vs. Time", "avg_ob_ob_likeli.png")

plot.plotMetrics(x, averaged_metrics["avg_ob_ob_weight_loglikeli"][1:], name, ls, save_loc, 0, "Ob-Ob Log-Likelihood", f"Ob {0} to Ob Weighted Log-Likelihood vs. Time", "avg_ob_ob_weight_loglikeli.png")
plot.plotMetrics(x, averaged_metrics["avg_cross_entropy"][1:], name, ls, save_loc, 0, "Cross Entropy", f"Ob {0} to Ob Cross Entropy vs. Time", "avg_cross_entropy.png")
plot.plotMetrics(x, averaged_metrics["KL"][1:, :, 0], name, ls, save_loc, 0, "KL", f"Ob {0} to Ob KL Divergence vs. Time", "KL1.png")
plot.plotMetrics(x, averaged_metrics["KL"][1:, :, 1], name, ls, save_loc, 0, "KL", f"Ob {0} to Ob KL Divergence vs. Time", "KL2.png")
plot.plotMetrics(x, averaged_metrics["KL"][1:, :, 2], name, ls, save_loc, 0, "KL", f"Ob {0} to Ob Avg KL Divergence vs. Time", "KL.png")
plot.plotMetrics(x, averaged_metrics["JS"][1:], name, ls, save_loc, 0, "JS", f"Ob {0} to Ob Jensen-Shannon Divergence vs. Time", "JS.png")
#plot.plotMetricsPerState(x, averaged_metrics["JS_marginal"][0, 1:, :, :], normalization_quantities, names, ls, save_loc, 0, "JS Slice", "JS Slice vs. Time", "JSSLice")




