# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 17:50:23 2025

@author: tarun
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from scipy.stats import chi2


def create_hidden_figure(nrows=2, ncols=3):
    fig, axes = plt.subplots(nrows, ncols, figsize=plt.figaspect(nrows/ncols), constrained_layout=True)
    fig.set_visible(True)
    try:
        fig.manager.window.state('zoomed')
    except Exception:
        pass
    return fig, axes.ravel()


def save_and_close(fig, filename):
    #plt.pause(0.5)
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def apply_measurement_model(h, cloud):
    """Apply measurement function to a cloud (Nxstate_dim)."""
    return np.apply_along_axis(h, 1, cloud)



STATE_PAIR_CONFIGS = [
    (0, 1, 'X-Y',        'X (km.)',      'Y (km.)',      'dist'),
    (0, 2, 'X-Z',        'X (km.)',      'Z (km.)',      'dist'),
    (1, 2, 'Y-Z',        'Y (km.)',      'Z (km.)',      'dist'),
    (3, 4, 'Xdot-Ydot',  'Xdot (km/s)',  'Ydot (km/s)',  'vel'),
    (3, 5, 'Xdot-Zdot',  'Xdot (km/s)',  'Zdot (km/s)',  'vel'),
    (4, 5, 'Ydot-Zdot',  'Ydot (km/s)',  'Zdot (km/s)',  'vel'),
]

COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf","#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5"]

def plot_state_pairs(axs, clouds, truth, dist2km, vel2kms, labels, alpha=1.0, data_is_metric=False):
    """Generic state-pair scatter plot for multiple clouds with truth."""
    for ax, (ix, iy, title, xlabel, ylabel, scale_type) in zip(axs, STATE_PAIR_CONFIGS):
        scale = vel2kms
        if scale_type == 'dist': 
            scale = dist2km
        
        for cloud, color, label in zip(clouds, COLORS, labels):
            if cloud.shape[0] > 0:
                if data_is_metric:
                    ax.plot(
                        scale * cloud[:, ix],
                        scale * cloud[:, iy],
                        color=color,
                        label=label)
                else:
                    ax.scatter(
                        scale * cloud[:, ix],
                        scale * cloud[:, iy],
                        s=2,
                        color=color,
                        alpha=alpha,
                        label=label)

        ax.plot(
            scale * truth[0, ix],
            scale * truth[0, iy],
            'kx',
            markersize=8,
            linewidth=8,
            label='Truth')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        if scale_type == 'dist': 
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.xaxis.get_offset_text().set_fontsize(9)
        ax.yaxis.get_offset_text().set_fontsize(9)
        ax.tick_params(axis="both", labelsize=6)
        #ax.legend()



def plotMetrics(x, y_data, cloud_names, linestyle, save_loc, ob, y_label, title_str, filename):
    fig, ax = plt.subplots()
    fig.set_visible(True)
    fig.suptitle(title_str)
        
    for cloud, color, ls in zip(y_data, COLORS, linestyle):
        ax.plot(x, cloud, ls, color=color, linewidth=2)

    if y_label == 'NEES':
        NEES_lb = chi2.ppf(0.025, 6)
        NEES_ub = chi2.ppf(0.975, 6)
        ax.hlines([NEES_lb, NEES_ub], x[0], x[-1], COLORS='k', linestyles='-', label='NEES 95% CI')
        ax.set_xlabel('Filter Step #')
        ax.set_ylabel('NEES')
        ax.set_yscale('log')
        ax.legend(list(cloud_names) + ['NEES 95% CI'])
    elif y_label == 'RMSE' or y_label == 'KL' or y_label == 'Ob-Ob Log-Likelihood'or y_label == 'Cross Entropy':
        ax.set_yscale('log')

    ax.set_xlabel('Time (hr)')
    ax.set_ylabel(y_label)
    ax.legend(cloud_names, fontsize=4, loc="best")
    
    save_and_close(fig, f"{save_loc}/Observer{ob}/{filename}")



def plotMetricsPerState(x, y_data, normalization_quantities, cloud_names, linestyle, save_loc, ob, y_label, title_str, filename):
    fig, axs = plt.subplots(2, 3, sharex=True, figsize=(12, 6))
    axs = axs.flatten()
    fig.set_visible(True)
    axes_titles = ["X", "Y", "Z", "Xdot", "Ydot", "Zdot"]
    
    if "RMSE" in y_label:
        y_data[:, :, :3] *= normalization_quantities["dist2km"]
        y_data[:, :, 3:] *= normalization_quantities["vel2kms"]
    
    for i in range(6):
        for cloud, color, ls in zip(y_data[:, :, i], COLORS, linestyle):
            axs[i].plot(x, cloud, ls, color=color, linewidth=2)
            axs[i].set_title(axes_titles[i])
    
    fig.suptitle(f"{title_str}")
    fig.legend(cloud_names, fontsize=4, loc="upper right")
    fig.supxlabel("Time (hr)")
    fig.supylabel(f"{y_label}")
    fig.tight_layout()
    save_and_close(fig, f"{save_loc}/Observer{ob}/{filename}")



def plotStateSpace(cloud, truth, K, cluster_idx, normalization_quantities, plot_title, filename, plot_cross_observers=False, cloud_names=None):
    dist2km = normalization_quantities["dist2km"]
    vel2kms = normalization_quantities["vel2kms"]
    fig, axs = create_hidden_figure()
    
    # Prepare clouds per cluster
    clouds = []
    labels = []
    for k in range(1, K+1):
        cluster_points = cloud[cluster_idx == k]
        clouds.append(cluster_points)
        if plot_cross_observers:
            labels.append(cloud_names[k-1])
        else:
            labels.append(f'Cluster {k}')

    plot_state_pairs(axs, clouds, truth, dist2km, vel2kms, labels)
    fig.suptitle(plot_title, fontsize=9)
    #fig.tight_layout()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    save_and_close(fig, filename)



def plotStateSpaceCombined(plotting_clouds, plotting_truth, active_cloud_mask, normalization_quantities, cloud_names, plot_title, filename):
    dist2km = normalization_quantities["dist2km"]
    vel2kms = normalization_quantities["vel2kms"]
    fig, axs = create_hidden_figure()
    
    clouds = [plotting_clouds[i] for i in active_cloud_mask]
    labels = [cloud_names[i] for i in active_cloud_mask]
    
    plot_state_pairs(axs, clouds, plotting_truth, dist2km, vel2kms, labels, alpha=0.3)
    fig.suptitle(plot_title)
    save_and_close(fig, filename)



def plotMsmtSpace(cloud, truth, zt, h, likelihoods, K, cluster_idx, plot_title, filename, msmt_exists, plot_cross_observers=False, cloud_names=None):
    deg = 180.0 / np.pi
    fig, ax = plt.subplots()
    fig.set_visible(True)
    
    for k in range(1, K+1):
        pts = cloud[cluster_idx == k]
        if pts.size == 0:
            continue
        Zmcloud = h(pts)
        if plot_cross_observers:
            ax.scatter(deg*Zmcloud[:,0], deg*Zmcloud[:,1], s=2, color=COLORS[k-1], label=cloud_names[k-1])
        else:
            ax.scatter(deg*Zmcloud[:,0], deg*Zmcloud[:,1], s=2, color=COLORS[k-1], label=f'k: {k}; w: {likelihoods[k-1]:.3f}')
    
    Ztruth = h(truth).flatten()
    ax.plot(deg*Ztruth[0], deg*Ztruth[1], 'kx', markersize=15, linewidth=3, label='Truth')
    
    if msmt_exists:
        ax.plot(deg*zt[0], deg*zt[1], 'ko', markersize=15, markerfacecolor='none', linewidth=5, label='Msmt')
    
    ax.set_title(plot_title)
    ax.ticklabel_format(style='plain', useOffset=False)
    ax.set_xlabel('Azimuth Angle (deg)')
    ax.set_ylabel('Elevation Angle (deg)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    
    save_and_close(fig, filename)


def plotMsmtSpaceCombined(plotting_clouds, plotting_truth, zt, h, num_clouds_per_agent, cloud_names, plot_title, filename, msmt_exists):
    deg = 180.0 / np.pi
    fig, ax = plt.subplots()
    fig.set_visible(True)
    
    for cloud_idx in range(num_clouds_per_agent):
        cloud = plotting_clouds[cloud_idx]
        if cloud.size == 0:
            continue
        msmt_cloud = apply_measurement_model(h, cloud)
        ax.scatter(deg*msmt_cloud[:,0], deg*msmt_cloud[:,1], color=COLORS[cloud_idx],
                   label=cloud_names[cloud_idx])
    
    Ztruth = h(plotting_truth)
    ax.plot(deg*Ztruth[0], deg*Ztruth[1], 'kx', markersize=15, linewidth=3, label='Truth')
    
    if msmt_exists:
        ax.plot(deg*zt[0], deg*zt[1], 'ko', markersize=15, markerfacecolor='none', linewidth=5, label='Msmt')
    
    ax.set_title(plot_title)
    ax.ticklabel_format(style='plain', useOffset=False)
    ax.set_xlabel('Azimuth Angle (deg)')
    ax.set_ylabel('Elevation Angle (deg)')
    ax.legend()
    
    save_and_close(fig, filename)