# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 17:50:23 2025

@author: tarun
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2


def create_hidden_figure(nrows=2, ncols=3):
    fig, axes = plt.subplots(nrows, ncols, figsize=plt.figaspect(nrows/ncols))
    fig.set_visible(False)
    try:
        fig.manager.window.state('zoomed')
    except Exception:
        pass
    return fig, axes.ravel()


def save_and_close(fig, filename):
    plt.pause(0.5)
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


def plot_state_pairs(axs, clouds, truth, dist2km, vel2kms, colors, labels, alpha=1.0):
    """Generic state-pair scatter plot for multiple clouds with truth."""
    for ax, (ix, iy, title, xlabel, ylabel, scale_type) in zip(axs, STATE_PAIR_CONFIGS):
        scale = dist2km if scale_type == 'dist' else vel2kms

        for cloud, color, label in zip(clouds, colors, labels):
            if cloud.size > 0:
                ax.scatter(
                    scale * cloud[:, ix],
                    scale * cloud[:, iy],
                    s=20,
                    color=color,
                    alpha=alpha,
                    label=label
                )

        ax.plot(
            scale * truth[ix],
            scale * truth[iy],
            'kx',
            markersize=20,
            linewidth=3,
            label='Truth'
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()



def plotMetrics(fig_num, x, y_data, cloud_names, colors, save_loc, ob, y_label, title_str, filename):
    fig, ax = plt.subplots()
    fig.set_visible(False)
    fig.suptitle(title_str % ob)
    
    for cloud, color in zip(y_data, colors):
        ax.plot(x, cloud, '--', color=color, linewidth=2)

    if y_label == 'NEES':
        NEES_lb = chi2.ppf(0.025, 6)
        NEES_ub = chi2.ppf(0.975, 6)
        ax.hlines([NEES_lb, NEES_ub], x[0], x[-1], colors='k', linestyles='--', label='NEES 95% CI')
        ax.set_xlabel('Filter Step #')
        ax.set_ylabel('NEES')
        ax.set_yscale('log')
        ax.legend(list(cloud_names) + ['NEES 95% CI'])
    elif y_label == 'RMSE':
        ax.set_yscale('log')

    ax.set_xlabel('Filter Step #')
    ax.set_ylabel(y_label)
    ax.legend(cloud_names)
    
    save_and_close(fig, f"{save_loc}/Observer{ob}/{filename}")



def plotMetricsPerState(fig_num, x, y_data, normalization_quantities, cloud_names, colors, save_loc, ob, y_label, title_str, filename):
    dist2km = normalization_quantities.dist2km
    vel2kms = normalization_quantities.vel2kms
    fig, axs = create_hidden_figure()
    
    clouds_scaled = []
    for cloud in y_data:
        scaled = np.hstack([
            dist2km * cloud[:, :3],
            vel2kms * cloud[:, 3:6]
        ])
        clouds_scaled.append(scaled)

    plot_state_pairs(axs, clouds_scaled, np.zeros(6), dist2km, vel2kms, colors, cloud_names)
    fig.suptitle(f"{title_str} Observer {ob}")
    save_and_close(fig, f"{save_loc}/Observer{ob}/{filename}")



def plotStateSpace(cloud, truth, K, cluster_idx, normalization_quantities, colors, plot_title, filename):
    dist2km = normalization_quantities.dist2km
    vel2kms = normalization_quantities.vel2kms
    fig, axs = create_hidden_figure()
    
    # Prepare clouds per cluster
    clouds = []
    labels = []
    for k in range(1, K+1):
        cluster_points = cloud[cluster_idx == k]
        clouds.append(cluster_points)
        labels.append(f'Cluster {k}')

    plot_state_pairs(axs, clouds, truth, dist2km, vel2kms, colors[:K], labels)
    fig.suptitle(plot_title)
    save_and_close(fig, filename)



def plotStateSpaceCombined(plotting_clouds, plotting_truth, active_cloud_mask, normalization_quantities, colors, cloud_names, plot_title, filename):
    dist2km = normalization_quantities.dist2km
    vel2kms = normalization_quantities.vel2kms
    fig, axs = create_hidden_figure()
    
    clouds = [plotting_clouds[i] for i in active_cloud_mask]
    labels = [cloud_names[i] for i in active_cloud_mask]
    
    plot_state_pairs(axs, clouds, plotting_truth, dist2km, vel2kms, colors, labels, alpha=0.3)
    fig.suptitle(plot_title)
    save_and_close(fig, filename)



def plotMsmtSpace(cloud, truth, zt, h, likelihoods, K, cluster_idx, colors, plot_title, filename, msmt_exists):
    deg = 180.0 / np.pi
    fig, ax = plt.subplots()
    fig.set_visible(False)
    
    for k in range(1, K+1):
        pts = cloud[cluster_idx == k]
        if pts.size == 0:
            continue
        Zmcloud = apply_measurement_model(h, pts)
        ax.scatter(deg*Zmcloud[:,0], deg*Zmcloud[:,1], color=colors[k-1],
                   label=f'k: {k}; w: {likelihoods[k-1]:.3f}')
    
    Ztruth = h(truth)
    ax.plot(deg*Ztruth[0], deg*Ztruth[1], 'kx', markersize=20, linewidth=3, label='Truth')
    
    if msmt_exists:
        ax.plot(deg*zt[0], deg*zt[1], 'ko', markersize=20, linewidth=3, label='Noisy Truth')
    
    ax.set_title(plot_title)
    ax.set_xlabel('Azimuth Angle (deg)')
    ax.set_ylabel('Elevation Angle (deg)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    
    save_and_close(fig, filename)


def plotMsmtSpaceCombined(plotting_clouds, plotting_truth, zt, h, num_clouds_per_agent, colors, cloud_names, plot_title, filename, msmt_exists):
    deg = 180.0 / np.pi
    fig, ax = plt.subplots()
    fig.set_visible(False)
    
    for cloud_idx in range(num_clouds_per_agent):
        cloud = plotting_clouds[cloud_idx]
        if cloud.size == 0:
            continue
        msmt_cloud = apply_measurement_model(h, cloud)
        ax.scatter(deg*msmt_cloud[:,0], deg*msmt_cloud[:,1], color=colors[cloud_idx],
                   label=cloud_names[cloud_idx])
    
    Ztruth = h(plotting_truth)
    ax.plot(deg*Ztruth[0], deg*Ztruth[1], 'kx', markersize=20, linewidth=3, label='Truth')
    
    if msmt_exists:
        ax.plot(deg*zt[0], deg*zt[1], 'ko', markersize=20, linewidth=3, label='Noisy Truth')
    
    ax.set_title(plot_title)
    ax.set_xlabel('Azimuth Angle (deg)')
    ax.set_ylabel('Elevation Angle (deg)')
    ax.legend()
    
    save_and_close(fig, filename)