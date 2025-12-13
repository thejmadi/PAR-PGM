
clear all;
tic
load_loc = "D:/PythonProjects/EDP/PGM/ParticleFusionTest/11_24_25_meeting/EXP_L2/EXP_VaryIODandFusionTime/Test5c";
save_loc = load_loc + "/MCResults";
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
best_ent2_det_cov = {};
ent2_det_cov = {};
best_ent2_det_cov_msmt = {};
ent2_det_cov_msmt = {};
cross_ob_ent = {};
cross_ob_norm = {};
likelihood_metric_state_space = {};
likelihood_metric_msmt_space = {};
num_cluster = {};
cloud_names = {};

for i = 1:num_subfolders
    folder_name = subfolders(i).name;
    subfolder_path = fullfile(load_loc, folder_name);

    try
        MC_consistency{end+1} = load(fullfile(subfolder_path, 'MC_consistency.mat'), 'MC_consistency').MC_consistency;
        std_dev{end+1} = load(fullfile(subfolder_path, 'std_dev_per_state.mat'), 'std_dev').std_dev;
        RMSE{end+1} = load(fullfile(subfolder_path, 'RMSE.mat'), 'RMSE').RMSE;
        best_ent2_det_cov{end+1} = load(fullfile(subfolder_path, 'BestEntropyState.mat'), 'best_ent2_det_cov').best_ent2_det_cov;
        ent2_det_cov{end+1} = load(fullfile(subfolder_path, 'WeightedEntropyState.mat'), 'ent2_det_cov').ent2_det_cov;
        best_ent2_det_cov_msmt{end+1} = load(fullfile(subfolder_path, 'BestEntropyMsmt.mat'), 'best_ent2_det_cov_msmt').best_ent2_det_cov_msmt;
        ent2_det_cov_msmt{end+1} = load(fullfile(subfolder_path, 'WeightedEntropyMsmt.mat'), 'ent2_det_cov_msmt').ent2_det_cov_msmt;
        cross_ob_ent{end+1} = load(fullfile(subfolder_path, 'CrossObserverEntropy.mat'), 'cross_ob_ent').cross_ob_ent;
        cross_ob_norm{end+1} = load(fullfile(subfolder_path, 'CrossObserverNorm.mat'), 'cross_ob_norm').cross_ob_norm;
        likelihood_metric_state_space{end+1} = load(fullfile(subfolder_path, 'Likelihood_state.mat'), 'likelihood_metric_state_space').likelihood_metric_state_space;
        likelihood_metric_msmt_space{end+1} = load(fullfile(subfolder_path, 'Likelihood_msmt.mat'), 'likelihood_metric_msmt_space').likelihood_metric_msmt_space;
        num_cluster{end+1} = load(fullfile(subfolder_path, 'Num_cluster.mat'), 'num_cluster').num_cluster;
        cloud_names = load(fullfile(subfolder_path, 'cloud_names.mat'), 'cloud_names').cloud_names;
    catch
        fprintf('Failed loading data from folder %s\n', folder_name);
    end
end
timesteps = load(fullfile(subfolder_path, 'ExperimentTimesteps.mat'), 'all_timesteps').all_timesteps*time2hr;

%% Calculate Statistics
%std_dev = applyLogToNestedCell(std_dev);
%RMSE = applyLogToNestedCell(RMSE);
alter_cells_bool = false;
[MC_avg, MC_std] = average_and_std_nested_metric(MC_consistency, alter_cells_bool);
[std_dev_avg, std_dev_std] = average_and_std_nested_metric(std_dev, alter_cells_bool);
[RMSE_avg, RMSE_std] = average_and_std_nested_metric(RMSE, alter_cells_bool);
[best_ent2_det_cov_avg, best_ent2_det_cov_std] = average_and_std_nested_metric(best_ent2_det_cov, alter_cells_bool);
[ent2_det_cov_avg, ent2_det_cov_std] = average_and_std_nested_metric(ent2_det_cov, alter_cells_bool);
[best_ent2_det_cov_msmt_avg, best_ent2_det_cov_msmt_std] = average_and_std_nested_metric(best_ent2_det_cov_msmt, alter_cells_bool);
[ent2_det_cov_msmt_avg, ent2_det_cov_msmt_std] = average_and_std_nested_metric(ent2_det_cov_msmt, alter_cells_bool);
[cross_ob_ent_avg, cross_ob_ent_std] = average_and_std_nested_metric(cross_ob_ent, false);
[cross_ob_norm_avg, cross_ob_norm_std] = average_and_std_nested_metric(cross_ob_norm, false);
[likelihood_metric_state_space_avg, likelihood_metric_state_space_std] = average_and_std_nested_metric(likelihood_metric_state_space, alter_cells_bool);
[likelihood_metric_msmt_space_avg, likelihood_metric_msmt_space_std] = average_and_std_nested_metric(likelihood_metric_msmt_space, alter_cells_bool);
[num_cluster_avg, num_cluster_std] = average_and_std_nested_metric(num_cluster, alter_cells_bool);
if alter_cells_bool
    cloud_names = insertObservers(cloud_names);
    cloud_names(end, :) = [];
end
%{
MC_avg(:, end+1) = MC_avg(end, 1); MC_std(:, end+1) = MC_std(end, 1);
std_dev_avg(:, end+1) = std_dev_avg(end, 1); std_dev_std(:, end+1) = std_dev_std(end, 1);
RMSE_avg(:, end+1) = RMSE_avg(end, 1); RMSE_std(:, end+1) = RMSE_std(end, 1);

best_ent2_det_cov_avg(:, end+1) = best_ent2_det_cov_avg(end, 1); best_ent2_det_cov_std(:, end+1) = best_ent2_det_cov_std(end, 1);
ent2_det_cov_avg(:, end+1) = ent2_det_cov_avg(end, 1); ent2_det_cov_std(:, end+1) = ent2_det_cov_std(end, 1);
best_ent2_det_cov_msmt_avg(:, end+1) = best_ent2_det_cov_msmt_avg(end, 1); best_ent2_det_cov_msmt_std(:, end+1) = best_ent2_det_cov_msmt_std(end, 1);
ent2_det_cov_msmt_avg(:, end+1) = ent2_det_cov_msmt_avg(end, 1); ent2_det_cov_msmt_std(:, end+1) = ent2_det_cov_msmt_std(end, 1);

cross_ob_ent_avg(:, end+1) = cross_ob_ent_avg(end, 1); cross_ob_ent_std(:, end+1) = cross_ob_ent_std(end, 1);
cross_ob_norm_avg(:, end+1) = cross_ob_norm_avg(end, 1); cross_ob_norm_std(:, end+1) = cross_ob_norm_std(end, 1);

likelihood_metric_state_space_avg(:, end+1) = likelihood_metric_state_space_avg(end, 1); likelihood_metric_state_space_std(:, end+1) = likelihood_metric_state_space_std(end, 1);
likelihood_metric_msmt_space_avg(:, end+1) = likelihood_metric_msmt_space_avg(end, 1); likelihood_metric_msmt_space_std(:, end+1) = likelihood_metric_msmt_space_std(end, 1);
num_cluster_avg(:, end+1) = num_cluster_avg(end, 1); num_cluster_std(:, end+1) = num_cluster_std(end, 1);
cloud_names(:, end+1) = cloud_names(end, 1);

MC_avg(end, :) = []; MC_std(end, :) = [];
std_dev_avg(end, :) = []; std_dev_std(end, :) = [];
RMSE_avg(end, :) = []; RMSE_std(end, :) = [];

best_ent2_det_cov_avg(end, :) = []; best_ent2_det_cov_std(end, :) = [];
ent2_det_cov_avg(end, :) = []; ent2_det_cov_std(end, :) = [];
best_ent2_det_cov_msmt_avg(end, :) = []; best_ent2_det_cov_msmt_std(end, :) = [];
ent2_det_cov_msmt_avg(end, :) = []; ent2_det_cov_msmt_std(end, :) = [];

cross_ob_ent_avg(end, :) = []; cross_ob_ent_std(end, :) = [];
cross_ob_norm_avg(end, :) = []; cross_ob_norm_std(end, :) = [];

likelihood_metric_state_space_avg(end, :) = []; likelihood_metric_state_space_std(end, :) = [];
likelihood_metric_msmt_space_avg(end, :) = []; likelihood_metric_msmt_space_std(end, :) = [];
num_cluster_avg(end, :) = []; num_cluster_avg(end, :) = [];
cloud_names(end, :) = [];

MC_avg(end, :) = []; MC_std(end, :) = [];
std_dev_avg(end, :) = []; std_dev_std(end, :) = [];
RMSE_avg(end, :) = []; RMSE_std(end, :) = [];

best_ent2_det_cov_avg(end, :) = []; best_ent2_det_cov_std(end, :) = [];
ent2_det_cov_avg(end, :) = []; ent2_det_cov_std(end, :) = [];
best_ent2_det_cov_msmt_avg(end, :) = []; best_ent2_det_cov_msmt_std(end, :) = [];
ent2_det_cov_msmt_avg(end, :) = []; ent2_det_cov_msmt_std(end, :) = [];

cross_ob_ent_avg(end, :) = []; cross_ob_ent_std(end, :) = [];
cross_ob_norm_avg(end, :) = []; cross_ob_norm_std(end, :) = [];

likelihood_metric_state_space_avg(end, :) = []; likelihood_metric_state_space_std(end, :) = [];
likelihood_metric_msmt_space_avg(end, :) = []; likelihood_metric_msmt_space_std(end, :) = [];
num_cluster_avg(end, :) = []; num_cluster_avg(end, :) = [];
cloud_names(end, :) = [];
%}
%% Plot Statistics
fig_num = 1;
colors = ["Red", "Blue", "Green", "Magenta", "Cyan", "Black", "Yellow", "#500000", "#bf5700", "#00274c"];
%cloud_names = ["Original", "Fusion 1: 120 hrs", "Fusion 2: 200 hrs", "Fusion 3: 280 hrs", "Fusion 4: 360 hrs", "Baseline"];%["Ob 1: IOD 0-1/4 Orbits", "Ob 2: IOD 0-1/4 Orbits", "Ob 3: IOD 1/4-2/4 Orbits", "Ob 4: IOD 3/4-4/4 Orbits"];
x = 0:size(MC_avg{1, 1}, 1)-1;
%plotMetricsPerState(fig_num, x, std_dev(ob, :), dist2km, vel2kms, cloud_names, colors, save_loc, ob, 'Sigma', 'Standard Deviation', 'StdDev.png');
%fig_num = fig_num + 1;

for ob = 1:size(MC_avg, 1)
    ensureDirExists(sprintf('%s/Observer%i/', save_loc, ob));
    mask = contains(cloud_names(ob, :), ["Original","Baseline"], "IgnoreCase", true);
    ls = repmat("--", size(cloud_names(ob, :)));      % default
    ls(mask) = "-";                   % dashed where match

    plotMetrics(fig_num, timesteps, MC_avg(ob, :), MC_std(ob, :), cloud_names(ob, :), colors, save_loc, ob, ls, 'Consistency', 'Monte Carlo Consistency Ob: %i', 'MC_Consistency.png', false);
    fig_num = fig_num + 1;

    plotMetricsPerState(fig_num, timesteps, std_dev_avg(ob, :), std_dev_std(ob, :), dist2km, vel2kms, cloud_names(ob, :), colors, save_loc, ob, ls, 'Std Dev', 'Monte Carlo Std Dev Ob: %i', 'MC_StdDev.png', false);
    fig_num = fig_num + 1;
    plotMetricsPerState(fig_num, timesteps, RMSE_avg(ob, :), RMSE_std(ob, :), dist2km, vel2kms, cloud_names(ob, :), colors, save_loc, ob, ls, 'RMSE', 'Monte Carlo RMSE Ob: %i', 'MC_RMSE.png', false);
    fig_num = fig_num + 1;

    plotMetrics(fig_num, timesteps, best_ent2_det_cov_avg(ob, :), best_ent2_det_cov_std(ob, :), cloud_names(ob, :), colors, save_loc, ob, ls, 'Entropy', 'Monte Carlo Best Mode State Entropy Ob: %i', 'MC_BestEntropyState.png', false);
    fig_num = fig_num + 1;
    
    plotMetrics(fig_num, timesteps, ent2_det_cov_avg(ob, :), ent2_det_cov_std(ob, :), cloud_names(ob, :), colors, save_loc, ob, ls, 'Entropy', 'Monte Carlo Weighted State Entropy Ob: %i', 'MC_WeightedEntropyState.png', false);
    fig_num = fig_num + 1;
    
    plotMetrics(fig_num, timesteps, best_ent2_det_cov_msmt_avg(ob, :), best_ent2_det_cov_std(ob, :), cloud_names(ob, :), colors, save_loc, ob, ls, 'Entropy', 'Monte Carlo Best Mode Msmt Entropy Ob: %i', 'MC_BestEntropyMsmt.png', false);
    fig_num = fig_num + 1;
    plotMetrics(fig_num, timesteps, ent2_det_cov_msmt_avg(ob, :), ent2_det_cov_msmt_std(ob, :), cloud_names(ob, :), colors, save_loc, ob, ls, 'Entropy', 'Monte Carlo Weighted Msmt Entropy Ob: %i', 'MC_WeightedEntropyMsmt.png', false);
    fig_num = fig_num + 1;

    plotMetrics(fig_num, timesteps, likelihood_metric_state_space_avg(ob, :), likelihood_metric_state_space_std(ob, :), cloud_names(ob, :), colors, save_loc, ob, ls, 'Log-Likelihood', 'Monte Carlo Best Mode State Log-Likelihood Ob: %i', 'MC_BestLikeliState.png', false);
    fig_num = fig_num + 1;
    plotMetrics(fig_num, timesteps, likelihood_metric_msmt_space_avg(ob, :), likelihood_metric_msmt_space_std(ob, :), cloud_names(ob, :), colors, save_loc, ob, ls, 'Log-Likelihood', 'Monte Carlo Best Mode Msmt Log-Likelihood Ob: %i', 'MC_BestLikeliMsmt.png', false);
    fig_num = fig_num + 1;
    plotMetrics(fig_num, timesteps, num_cluster_avg(ob, :), num_cluster_std(ob, :), cloud_names(ob, :), colors, save_loc, ob, ls, 'Number of Clusters', 'Monte Carlo Number of Clusters Ob: %i', 'MC_NumClusters.png', false);
    fig_num = fig_num + 1;
    

    
    %plotMetricDistributionHeatmap(fig_num, MC_consistency, 30, save_loc, ob, 'Monte Carlo Consistency', 'HM_MC_Consistency.png');
    %fig_num = fig_num + 1;

    %plotMetricsPerState(fig_num, std_dev, 30, save_loc, ob, 'Std Dev', 'Monte Carlo Std Dev', 'HM_MC_StdDev.png');
    %fig_num = fig_num + 1;
    %plotMetricsPerState(fig_num, RMSE, 30, save_loc, ob, 'RMSE', 'Monte Carlo RMSE', 'HM_MC_RMSE.png');
    %fig_num = fig_num + 1;
    %{
    plotMetricDistributionHeatmap(fig_num, best_ent2_det_cov, 30, save_loc, ob, 'Monte Carlo Best Mode State Entropy', 'HM_MC_BestEntropyState.png');
    fig_num = fig_num + 1;
    plotMetricDistributionHeatmap(fig_num, ent2_det_cov, 30, save_loc, ob, 'Monte Carlo Weighted State Entropy', 'HM_MC_WeightedEntropyState.png');
    fig_num = fig_num + 1;

    plotMetricDistributionHeatmap(fig_num, best_ent2_det_cov_msmt, 30, save_loc, ob, 'Monte Carlo Best Mode Msmt Entropy', 'HM_MC_BestEntropyMsmt.png');
    fig_num = fig_num + 1;
    plotMetricDistributionHeatmap(fig_num, ent2_det_cov_msmt, 30, save_loc, ob, 'Monte Carlo Weighted Msmt Entropy', 'HM_MC_WeightedEntropyMsmt.png');
    fig_num = fig_num + 1;

    plotMetricDistributionHeatmap(fig_num, likelihood_metric_state_space, 30, save_loc, ob, 'Monte Carlo Best Mode State Log-Likelihood', 'HM_MC_BestLikeliState.png');
    fig_num = fig_num + 1;
    plotMetricDistributionHeatmap(fig_num, likelihood_metric_msmt_space, 30, save_loc, ob, 'Monte Carlo Best Mode Msmt Log-Likelihood', 'HM_MC_BestLikeliMsmt.png');
    fig_num = fig_num + 1;
    plotMetricDistributionHeatmap(fig_num, num_cluster, 30, save_loc, ob, 'Monte Carlo Number of Clusters', 'HM_MC_NumClusters.png');
    fig_num = fig_num + 1;
    %}
    
end
mask = contains(cloud_names(1, :), ["Original","Baseline"], "IgnoreCase", true);
ls = repmat("--", size(cloud_names(1, :)));      % default
ls(mask) = "-";                   % dashed where match
plotMetrics(fig_num, timesteps, cross_ob_ent_avg(2:end, 1), cross_ob_ent_std(2:end, 1), cloud_names(1, 1:2), colors(1:2), save_loc, 1, ls, 'Monte Carlo Ob-Ob Normalized Likelihood Matrix Entropy', 'Entropy', 'MC_ObObEntropy.png', false);
fig_num = fig_num + 1;
plotMetrics(fig_num, timesteps, cross_ob_norm_avg(2:end, 1), cross_ob_norm_std(2:end, 1), cloud_names(1, 1:2), colors(1:2), save_loc, 1, ls, 'Monte Carlo Ob-Ob Unnormalized Likelihood Matrix L1 Norm', 'L1 Norm', 'MC_ObObNorm.png', false);
fig_num = fig_num + 1;

%plotMetricDistributionHeatmap(fig_num, cross_ob_ent, 30, save_loc, 1, 'Ob-Ob Entropy', "HM_MC_ObObEntropy.png");
%fig_num = fig_num + 1;
%plotMetricDistributionHeatmap(fig_num, cross_ob_norm, 30, save_loc, 1, 'Ob-Ob Entropy Weighted', "HM_MC_ObObNorm.png");
%fig_num = fig_num + 1;

%% Statistics Functions

function [avg_cell, std_cell] = average_and_std_nested_metric(metric_cell, alter_cells)
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
    if (alter_cells)
        avg_cell = insertObservers(avg_cell); avg_cell(end, :) = [];
        std_cell = insertObservers(std_cell); std_cell(end, :) = [];
    end
end

function out = applyLogToNestedCell(cellIn)
    % Main function
    out = recursiveLog(cellIn);

    % Nested recursive function
    function result = recursiveLog(x)
        if iscell(x)
            result = cellfun(@(y) recursiveLog(y), x, 'UniformOutput', false);
        elseif isnumeric(x)
            result = log(x);  % Use log10(x) if needed
        else
            result = x;  % Leave other data types untouched
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


function plotMetricDistributionHeatmap(fig_num, nestedCell, numBins, save_loc, ob, title_str, file_name)
    % Assumes:
    % nestedCell: (1, num_monte_carlo) cell -> (num_observers x 1) cell -> [T x S]
    % Only processes state 1
    % numBins: number of bins for the histogram

    num_mc = numel(nestedCell);
    sampleMatrix = nestedCell{1}{1};
    [num_timesteps, ~] = size(sampleMatrix);

    % Collect all state-1 values to determine bin edges
    all_values = [];
    for mc = 1:num_mc
        mat = nestedCell{mc}{ob};     % [T x S]
        all_values = [all_values; mat(:,1)];  % State 1
    end

    % Define histogram bins from min to max value
    binEdges = linspace(min(all_values), max(all_values), numBins + 1);
    binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;

    % Initialize histogram matrix: rows = bins, cols = time steps
    histMatrix = zeros(numBins, num_timesteps);

    % For each time step, gather metric values across all MC x observers
    for t = 1:num_timesteps
        valuesAtT = zeros(num_mc, 1);
        idx = 1;
        for mc = 1:num_mc
            mat = nestedCell{mc}{ob};  % [T x S]
            valuesAtT(idx) = mat(t, 1);  % Only state 1
            idx = idx + 1;
        end
        % Compute histogram
        counts = histcounts(valuesAtT, binEdges);
        histMatrix(:, t) = counts(:);
    end

    % Plot the heatmap
    f = figure(fig_num);
    f.WindowState = 'maximized';
    imagesc(1:num_timesteps, binCenters, histMatrix);
    axis xy;  % Keep lower metric values at the bottom
    colorbar;
    xlabel('Time (hrs)');
    ylabel(title_str);
    title(sprintf('Time-Varying Distribution of %s Ob: %i', title_str, ob));

    save_path = sprintf('%s/Observer%i/TV_%s', save_loc, ob, file_name);
    drawnow;
    pause(0.5); % Optional: allows rendering to complete before saving
    exportgraphics(f, save_path, 'Resolution', 150);
    close(f);
end




function plotMetrics(fig_num, x, y_avg, y_std, cloud_names, colors, save_loc, ob, ls, y_label, title_str, filename, plot_std)
    f = figure(fig_num);
    f.WindowState = 'maximized';
    hold on;
    grid on;
    % Plot each cloud's data
    num_clouds = length(y_avg);
    for cloud = 1:num_clouds
        lw = 2;
        plot(x, y_avg{cloud}, 'Color', colors(cloud), 'LineWidth', lw, 'LineStyle', ls(cloud));
        if(plot_std)
            plot(x, y_avg{cloud} + 3 * y_std{cloud}, 'Color', colors(cloud), 'LineWidth', lw, 'LineStyle', '--');
            plot(x, y_avg{cloud} - 3 * y_std{cloud}, 'Color', colors(cloud), 'LineWidth', lw, 'LineStyle', '--');
        end
    end
    if (strcmp(y_label, 'NEES'))
        NEES_lb = chi2inv(0.025, 6);
        NEES_ub = chi2inv(0.975, 6);
        plot(x, NEES_lb*ones(1,size(x,2)), '--k')
        plot(x, NEES_ub*ones(1,size(x,2)), '--k')
        xlabel('Time (hrs)')
        ylabel('NEES')
        title('NEES Ob: %i', ob)
        legend([cloud_names, "NEES 95% CI"], 'Location', 'best')
    end
    if (strcmp(y_label, 'NEES') || strcmp(y_label, 'RMSE'))
        set(gca, 'YScale', 'log');
    end

    % Labeling
    xlabel('Time (hrs)');
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


function plotMetricsPerState(fig_num, x, y_data, y_std, dist2km, vel2kms, cloud_names, colors, save_loc, ob, ls, y_label, title_str, filename, plot_std)
    f = figure(fig_num);
    f.WindowState = 'maximized';
    num_clouds = length(y_data);
    subplot(2,3,1)
    hold on
    for cloud = 1:num_clouds
        plot(x, dist2km*y_data{cloud}(:,1), 'Color', colors(cloud), 'LineStyle', ls(cloud))
        if(plot_std)
            plot(x, dist2km*(y_data{cloud}(:,1) + 3 * y_std{cloud}(:,1)), 'Color', colors(cloud),'LineStyle', '--')
            plot(x, dist2km*(y_data{cloud}(:,1) - 3 * y_std{cloud}(:,1)), 'Color', colors(cloud),'LineStyle', '--')
        end
    end
    grid on;
    xlabel('Time (hrs)')
    ylabel(y_label + "_X (km.)")
    title("X " + title_str)
    set(gca, 'YScale', 'log');
    hold off

    subplot(2,3,2)
    hold on
    for cloud = 1:num_clouds
        plot(x, dist2km*y_data{cloud}(:,2), 'Color', colors(cloud), 'LineStyle', ls(cloud))
        if(plot_std)
            plot(x, dist2km*(y_data{cloud}(:,2) + 3 * y_std{cloud}(:,2)), 'Color', colors(cloud),'LineStyle', '--')
            plot(x, dist2km*(y_data{cloud}(:,2) - 3 * y_std{cloud}(:,2)), 'Color', colors(cloud),'LineStyle', '--')
        end
    end
    grid on;
    xlabel('Time (hrs)')
    ylabel(y_label + "_Y (km.)")
    title("Y " + title_str)
    set(gca, 'YScale', 'log');
    hold off
    
    subplot(2,3,3)
    hold on
    for cloud = 1:num_clouds
        plot(x, dist2km*y_data{cloud}(:,3), 'Color', colors(cloud), 'LineStyle', ls(cloud))
        if(plot_std)
            plot(x, dist2km*(y_data{cloud}(:,3) + 3 * y_std{cloud}(:,3)), 'Color', colors(cloud),'LineStyle', '--')
            plot(x, dist2km*(y_data{cloud}(:,3) - 3 * y_std{cloud}(:,3)), 'Color', colors(cloud),'LineStyle', '--')
        end
    end
    grid on;
    xlabel('Time (hrs)')
    ylabel(y_label + "_Z (km.)")
    title("Z " + title_str)
    set(gca, 'YScale', 'log');
    hold off
    
    subplot(2,3,4)
    hold on
    for cloud = 1:num_clouds
        plot(x, vel2kms*y_data{cloud}(:,4), 'Color', colors(cloud), 'LineStyle', ls(cloud))
        if(plot_std)
            plot(x, vel2kms*(y_data{cloud}(:,4) + 3 * y_std{cloud}(:,4)), 'Color', colors(cloud),'LineStyle', '--')
            plot(x, vel2kms*(y_data{cloud}(:,4) - 3 * y_std{cloud}(:,4)), 'Color', colors(cloud),'LineStyle', '--')
        end
    end
    grid on;
    xlabel('Time (hrs)')
    ylabel(y_label + "_Xdot (km/s)")
    title("Xdot " + title_str)
    set(gca, 'YScale', 'log');
    hold off
    
    subplot(2,3,5)
    hold on
    for cloud = 1:num_clouds
        plot(x, vel2kms*y_data{cloud}(:,5), 'Color', colors(cloud), 'LineStyle', ls(cloud))
        if(plot_std)
            plot(x, vel2kms*(y_data{cloud}(:,5) + 3 * y_std{cloud}(:,5)), 'Color', colors(cloud),'LineStyle', '--')
            plot(x, vel2kms*(y_data{cloud}(:,5) - 3 * y_std{cloud}(:,5)), 'Color', colors(cloud),'LineStyle', '--')
        end
    end
    grid on;
    xlabel('Time (hrs)')
    ylabel(y_label + "_Ydot (km/s)")
    title("Ydot " + title_str)
    set(gca, 'YScale', 'log');
    hold off
    
    subplot(2,3,6)
    hold on
    for cloud = 1:num_clouds
        plot(x, vel2kms*y_data{cloud}(:,6), 'Color', colors(cloud), 'LineStyle', ls(cloud))
        if(plot_std)
            plot(x, vel2kms*(y_data{cloud}(:,6) + 3 * y_std{cloud}(:,6)), 'Color', colors(cloud),'LineStyle', '--')
            plot(x, vel2kms*(y_data{cloud}(:,6) - 3 * y_std{cloud}(:,6)), 'Color', colors(cloud),'LineStyle', '--')
        end
    end
    grid on;
    xlabel('Time (hrs)')
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

function C = insertObservers(C)
    [n, m] = size(C);

    % Extract and transpose the first column
    col = C(:,1).';   % 1 × n cell row

    % Process each row independently
    for i = 1:n
        % Left side (before the row's own column entry)
        left = col(1:i-1);

        % The row's own first-column element
        mid = col(i);

        % Right side (after the row’s own entry)
        right = col(i+1:end);

        % Construct new row
        newRow = [ left, mid, right, C(i,2:m) ];

        % Replace row
        C(i,1:numel(newRow)) = newRow;
    end
end