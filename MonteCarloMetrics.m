
clear all;
tic
load_loc = "D:/PythonProjects/EDP/PGM/ParticleFusionTest/10_3_25_meeting/DilshadMetrics/Test1";

all_folders = dir(fullfile(load_loc, 'MC_*'));
subfolders = all_folders([all_folders.isdir]);  % Keep only directories
num_subfolders = length(subfolders);

dist2km = 384400; % Kilometers per non-dimensionalized distance
time2hr = 4.342*24; % Hours per non-dimensionalized time
vel2kms = dist2km/(time2hr*60*60); % Kms per non-dimensionalized velocity

%% Import Metrics
MC_consistency = {};
%MC_std_dev = {};
std_dev = {};
RMSE = {};
%best_ent2_det_cov = {};
%ent2_det_cov = {};
%best_ent2_det_cov_msmt = {};
%ent2_det_cov_msmt = {};
likelihood_metric_state_space = {};
%likelihood_metric_msmt_space = {};
num_cluster = {};
std_dev = {};
for i = 1:num_subfolders
    folder_name = subfolders(i).name;
    subfolder_path = fullfile(load_loc, folder_name);

    try
        MC_consistency{end+1} = load(fullfile(subfolder_path, 'MC_consistency.mat'), 'MC_consistency').MC_consistency;
        std_dev{end+1} = load(fullfile(subfolder_path, 'std_dev_per_state.mat'), 'std_dev').std_dev;
        RMSE{end+1} = load(fullfile(subfolder_path, 'RMSE.mat'), 'RMSE').RMSE;
        %best_ent2_det_cov{end+1} = load(fullfile(subfolder_path, 'BestEntropyState.mat'), 'best_ent2_det_cov').best_ent2_det_cov;
        %ent2_det_cov{end+1} = load(fullfile(subfolder_path, 'WeightedEntropyState.mat'), 'ent2_det_cov').ent2_det_cov;
        %best_ent2_det_cov_msmt{end+1} = load(fullfile(subfolder_path, 'BestEntropyMsmt.mat'), 'best_ent2_det_cov_msmt').best_ent2_det_cov_msmt;
        %ent2_det_cov_msmt{end+1} = load(fullfile(subfolder_path, 'WeightedEntropyMsmt.mat'), 'ent2_det_cov_msmt').ent2_det_cov_msmt;
        likelihood_metric_state_space{end+1} = load(fullfile(subfolder_path, 'Likelihood_state.mat'), 'likelihood_metric_state_space').likelihood_metric_state_space;
        %likelihood_metric_msmt_space{end+1} = load(fullfile(subfolder_path, 'Likelihood_msmt.mat'), 'likelihood_metric_msmt_space').likelihood_metric_msmt_space;
        num_cluster{end+1} = load(fullfile(subfolder_path, 'Num_cluster.mat'), 'num_cluster').num_cluster;
        %cloud_names = load(fullfile(subfolder_path, 'Num_cluster.mat'), 'cloud_names').cloud_names;
    catch
        fprintf('Failed loading data from folder %s\n', folder_name);
    end
end

%% Calculate Statistics
[MC_avg, MC_std] = average_and_std_nested_metric(MC_consistency);
[std_dev_avg, std_dev_std] = average_and_std_nested_metric(std_dev);
[RMSE_avg, RMSE_std] = average_and_std_nested_metric(RMSE);
%[best_ent2_det_cov_avg, best_ent2_det_cov_std] = average_and_std_nested_metric(best_ent2_det_cov);
%[ent2_det_cov_avg, ent2_det_cov_std] = average_and_std_nested_metric(ent2_det_cov);
%[best_ent2_det_cov_msmt_avg, best_ent2_det_cov_msmt_std] = average_and_std_nested_metric(best_ent2_det_cov_msmt);
%[ent2_det_cov_msmt_avg, ent2_det_cov_msmt_std] = average_and_std_nested_metric(ent2_det_cov_msmt);
[likelihood_metric_state_space_avg, likelihood_metric_state_space_std] = average_and_std_nested_metric(likelihood_metric_state_space);
%[likelihood_metric_msmt_space_avg, likelihood_metric_msmt_space_std] = average_and_std_nested_metric(likelihood_metric_msmt_space);
[num_cluster_avg, num_cluster_std] = average_and_std_nested_metric(num_cluster);

%% Plot Statistics
fig_num = 1;
colors = ["Red", "Blue", "Green", "Magenta", "Cyan", "Yellow", "Black", "#500000", "#bf5700", "#00274c"];

x = 0:size(MC_avg{1, 1}, 1)-1;
%plotMetricsPerState(fig_num, x, std_dev(ob, :), dist2km, vel2kms, cloud_names, colors, save_loc, ob, 'Sigma', 'Standard Deviation', 'StdDev.png');
%fig_num = fig_num + 1;
for ob = 1:size(MC_avg, 1)
    ensureDirExists(sprintf('%s/Observer%i/', load_loc, ob));
    plotMetrics(fig_num, x, MC_avg(ob, :), MC_std(ob, :), ['', '', ''], colors, load_loc, ob, 'Consistency', 'Monte Carlo Consistency Ob: %i', 'MCConsistency.png');
    fig_num = fig_num + 1;

    plotMetricsPerState(fig_num, x, std_dev_avg(ob, :), std_dev_std(ob, :), dist2km, vel2kms, ['', '', ''], colors, load_loc, ob, 'Std Dev', 'Monte Carlo Std Dev Ob: %i', 'MCStdDev.png');
    fig_num = fig_num + 1;
    plotMetricsPerState(fig_num, x, RMSE_avg(ob, :), RMSE_std(ob, :), dist2km, vel2kms, ['', '', ''], colors, load_loc, ob, 'RMSE', 'Monte Carlo RMSE Ob: %i', 'MCRMSE.png');
    fig_num = fig_num + 1;

    %plotMetrics(fig_num, x, best_ent2_det_cov_avg(ob, :), best_ent2_det_cov_std(ob, :), ['', '', ''], colors, load_loc, ob, 'Entropy', 'Monte Carlo Best Mode State Entropy Ob: %i', 'MCBestEntropyState.png');
    %fig_num = fig_num + 1;
    %plotMetrics(fig_num, x, ent2_det_cov_avg(ob, :), ent2_det_cov_std(ob, :), ['', '', ''], colors, load_loc, ob, 'Entropy', 'Monte Carlo Weighted State Entropy Ob: %i', 'MCWeightedEntropyState.png');
    %fig_num = fig_num + 1;

    %plotMetrics(fig_num, x, best_ent2_det_cov_msmt_avg(ob, :), best_ent2_det_cov_std(ob, :), ['', '', ''], colors, load_loc, ob, 'Entropy', 'Monte Carlo Best Mode Msmt Entropy Ob: %i', 'MCBestEntropyMsmt.png');
    %fig_num = fig_num + 1;
    %plotMetrics(fig_num, x, ent2_det_cov_msmt_avg(ob, :), ent2_det_cov_msmt_std(ob, :), ['', '', ''], colors, load_loc, ob, 'Entropy', 'Monte Carlo Weighted Msmt Entropy Ob: %i', 'MCWeightedEntropyMsmt.png');
    %fig_num = fig_num + 1;

    plotMetrics(fig_num, x, likelihood_metric_state_space_avg(ob, :), likelihood_metric_state_space_std(ob, :), ['', '', ''], colors, load_loc, ob, 'Log-Likelihood', 'Monte Carlo Best Mode State Log-Likelihood Ob: %i', 'MCBestLikeliState.png');
    fig_num = fig_num + 1;
    %plotMetrics(fig_num, x, likelihood_metric_msmt_space_avg(ob, :), likelihood_metric_msmt_space_std(ob, :), ['', '', ''], colors, load_loc, ob, 'Log-Likelihood', 'Monte Carlo Weighted Msmt Log-Likelihood Ob: %i', 'MCBestLikeliMsmt.png');
    %fig_num = fig_num + 1;
    plotMetrics(fig_num, x, num_cluster_avg(ob, :), num_cluster_std(ob, :), ['', '', ''], colors, load_loc, ob, 'Number of Clusters', 'Monte Carlo Number of Clusters Ob: %i', 'MCNumClusters.png');
    fig_num = fig_num + 1;
end

%% Statistics Functions

function [avg_cell, std_cell] = average_and_std_nested_metric(metric_cell)
    sample = metric_cell{1};
    [n, c] = size(sample);

    avg_cell = cell(n, c);
    std_cell = cell(n, c);

    for row = 1:n
        for col = 1:c
            matrices = cellfun(@(x) x{row, col}, metric_cell, 'UniformOutput', false);
            stack = cat(3, matrices{:});
            avg_cell{row, col} = mean(stack, 3, 'omitnan');
            std_cell{row, col} = std(stack, 0, 3, 'omitnan');  % 0 means normalization by N-1
        end
    end
end

%% Plotting Functions

function ensureDirExists(file_path)
    dir_path = fileparts(file_path);
    if ~exist(dir_path, 'dir')
        mkdir(dir_path);
    end
end


function plotMetrics(fig_num, x, y_avg, y_std, cloud_names, colors, save_loc, ob, y_label, title_str, filename)
    f = figure(fig_num);
    f.WindowState = 'maximized';
    hold on;

    % Plot each cloud's data
    num_clouds = length(y_avg);
    for cloud = 1:num_clouds
        lw = 2;
        plot(x, y_avg{cloud}, 'Color', colors(cloud), 'LineWidth', lw, 'LineStyle', '-');
        %plot(x, y_avg{cloud} + 3 * y_std{cloud}, 'Color', colors(cloud), 'LineWidth', lw, 'LineStyle', '--');
        %plot(x, y_avg{cloud} - 3 * y_std{cloud}, 'Color', colors(cloud), 'LineWidth', lw, 'LineStyle', '--');
    end
    if (strcmp(y_label, 'NEES'))
        NEES_lb = chi2inv(0.025, 6);
        NEES_ub = chi2inv(0.975, 6);
        plot(x, NEES_lb*ones(1,size(x,2)), '--k')
        plot(x, NEES_ub*ones(1,size(x,2)), '--k')
        xlabel('Filter Step #')
        ylabel('NEES')
        title('NEES Ob: %i', ob)
        legend([cloud_names, "NEES 95% CI"], 'Location', 'best')
    end
    if (strcmp(y_label, 'NEES') || strcmp(y_label, 'RMSE'))
        set(gca, 'YScale', 'log');
    end

    % Labeling
    xlabel('Filter Step #');
    ylabel(y_label);
    title(sprintf(title_str, ob));
    legend(cloud_names, 'Location', 'best');
    hold off;

    % Save
    save_path = sprintf('%s/Observer%i/%s', save_loc, ob, filename);
    drawnow;
    pause(0.5); % Optional: allows rendering to complete before saving
    exportgraphics(f, save_path, 'Resolution', 150);
    close(f);
end


function plotMetricsPerState(fig_num, x, y_data, y_std, dist2km, vel2kms, cloud_names, colors, save_loc, ob, y_label, title_str, filename)
    f = figure(fig_num);
    f.WindowState = 'maximized';
    num_clouds = length(y_data);
    subplot(2,3,1)
    hold on
    for cloud = 1:num_clouds
        plot(x, dist2km*y_data{cloud}(:,1), 'Color', colors(cloud))
    end
    xlabel('Filter Step #')
    ylabel(y_label + "_X (km.)")
    title("X " + title_str)
    set(gca, 'YScale', 'log');
    hold off

    subplot(2,3,2)
    hold on
    for cloud = 1:num_clouds
        plot(x, dist2km*y_data{cloud}(:,2), 'Color', colors(cloud))
    end
    xlabel('Filter Step #')
    ylabel(y_label + "_Y (km.)")
    title("Y " + title_str)
    set(gca, 'YScale', 'log');
    hold off
    
    subplot(2,3,3)
    hold on
    for cloud = 1:num_clouds
        plot(x, dist2km*y_data{cloud}(:,3), 'Color', colors(cloud))
    end
    xlabel('Filter Step #')
    ylabel(y_label + "_Z (km.)")
    title("Z " + title_str)
    set(gca, 'YScale', 'log');
    hold off
    
    subplot(2,3,4)
    hold on
    for cloud = 1:num_clouds
        plot(x, vel2kms*y_data{cloud}(:,4), 'Color', colors(cloud))
    end
    xlabel('Filter Step #')
    ylabel(y_label + "_Xdot (km/s)")
    title("Xdot " + title_str)
    set(gca, 'YScale', 'log');
    hold off
    
    subplot(2,3,5)
    hold on
    for cloud = 1:num_clouds
        plot(x, vel2kms*y_data{cloud}(:,5), 'Color', colors(cloud))
    end
    xlabel('Filter Step #')
    ylabel(y_label + "_Ydot (km/s)")
    title("Ydot " + title_str)
    set(gca, 'YScale', 'log');
    hold off
    
    subplot(2,3,6)
    hold on
    for cloud = 1:num_clouds
        plot(x, vel2kms*y_data{cloud}(:,6), 'Color', colors(cloud))
    end
    xlabel('Filter Step #')
    ylabel(y_label + "_Zdot (km/s)")
    title("Zdot " + title_str)
    set(gca, 'YScale', 'log');
    hold off
  
    save_path = sprintf('%s/Observer%i/%s', save_loc, ob, filename);
    legend(cloud_names, 'Location', 'best')
    drawnow;
    pause(0.5);
    exportgraphics(f, save_path, 'Resolution', 150);
    close(f);
end