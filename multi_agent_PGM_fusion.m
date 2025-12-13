% Start the clock
clear all;
tic%20, 23
for mc_idx = 11:20
    clearvars -except mc_idx; close all;
    rng(mc_idx, "twister")
    save_loc = "D:/PythonProjects/EDP/PGM/ParticleFusionTest/12_15_25_meeting/EXP_L2/EXP_VaryIODandFusionTime/Test1/MC_" + num2str(mc_idx);
    load_loc = "D:/PythonProjects/EDP/PGM/ParticleFusionTest/12_15_25_meeting/EXP_L2/EXP_VaryIODandFusionTime/Test1/OrbitData/Agent";
    dynamics = "CR3BP";
    % Non-dimensionalization
    if dynamics == "CR3BP"
        dist2km = 384400; % Kilometers per non-dimensionalized distance
        time2hr = 4.342*24; % Hours per non-dimensionalized time
        vel2kms = dist2km/(time2hr*60*60); % Kms per non-dimensionalized velocity
        normalization_quantities.dist2km = dist2km;
        normalization_quantities.vel2kms = vel2kms;
        normalization_quantities.time2hr = time2hr;
        normalization_quantities.mu = 1.2150582e-2;%mu / (norm(v0)^2 * norm(r0));
        dynamics_model = @(t, x) Dynamics.cr3bp_dyn(t, x, normalization_quantities.mu);
    end
    if dynamics == "2 Body"
        normalization_quantities = load(load_loc + num2str(1) + "/normalization_quantities.mat").normalization_quantities;
        dynamics_model = @(t, x) Dynamics.two_body_dyn(t, x, normalization_quantities.mu);
    end

    cluster_by = "FullState";
    Kn = 14; % Number of clusters (original)
    K = repmat({Kn}, 6, 6); % Number of clusters (changeable)
    Kmax = 14; % Maximum number of clusters (Kmax = 1 for EnKF)
    colors = ["#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7","#000000","#999999","#117733","#88CCEE","#332288","#DDCC77","#AA4499","#44AA99", ...
        "#882255","#1B9E77","#D95F02","#7570B3","#E7298A","#66A61E","#E6AB02","#A6761D","#666666","#A6CEE3","#1F78B4","#B2DF8A","#33A02C","#FB9A99","#E31A1C","#FDBF6F", ...
        "#FF7F00","#CAB2D6","#6A3D9A","#FFFF99","#8DD3C7","#FFFFB3","#BEBADA","#FB8072","#80B1D3","#FDB462","#B3DE69","#FCCDE5","#D9D9D9","#BC80BD","#CCEBC5","#FFED6F", ...
        "#F0027F","#386CB0","#BF5B17","#7FC97F","#BEAED4","#FDC086","#955251","#B565A7","#009688","#F4511E","#5C6BC0","#26A69A","#8E24AA","#7CB342","#039BE5","#C0CA33","#D81B60"];

    contourCols = lines(Kmax);
    
    disp_diagnostics = true;
    plot_IOD = false;
    plot_indv_clouds = false;
    plot_cross_observers = true;
    save_MC_metrics = true;
    
    % Number of particles to use in IOD
    num_IOD_particles = 20000;
    Lp = [20000; 20000; 20000];
    % Num of observers
    total_num_agents = 3;
    num_agents = 0;
    agent_is_active = false(total_num_agents, 1);
    active_mask = [];
    num_msmt_for_IOD = [10; 10; 10];
    ts_to_perform_IOD = [1; 21; 1];
    plot_comb_clouds = [true; false; false; false];
    num_clouds_per_agent = ones(total_num_agents, 1);
    num_clouds = num_agents * num_clouds_per_agent;
    
    % Load noiseless observation data and other important .mat files then
    % combine
    partial_ts = cell(1, total_num_agents);
    full_ts = cell(1, total_num_agents);
    full_vts = cell(1, total_num_agents);
    for ob = 1:total_num_agents
        partial_ts{ob} = load(load_loc + num2str(ob) + "/partial_ts.mat").partial_ts; % Noiseless observation data
        full_ts{ob} = load(load_loc + num2str(ob) + "/full_ts.mat").full_ts; % Position truth (topocentric frame)
        full_vts{ob} = load(load_loc + num2str(ob) + "/full_vts.mat").full_vts; % Velocity truth (topocentric frame)
    end
    [combined_msmt_data, combined_state_data, all_timesteps] = combineMsmts(full_ts, full_vts, partial_ts);
    num_timesteps = size(all_timesteps, 1);

    %fusion_information = [1, 2, false, 80];
    fusion_information = [1, 2, true, 46;
                        1, 2, true, 56;
                        1, 2, true, 66;
                        1, 2, true, 76];
    cloud_names = "Original Obs: " + (1:max(1, total_num_agents-1))' + ". IOD: " + string(normalization_quantities.time2hr * all_timesteps(ts_to_perform_IOD(1:end-1))) + " hrs";
    cloud_names(end+1) = "Baseline Obs";
    fusion_types = ["Original", "Weight Update"];
    num_new_clouds_per_agent = 1;

    % College Station
    obs_lat = repmat({30.618963}, 1, 6);
    obs_lon = repmat({-96.339214}, 1, 6);
    % Buenos Aires
    %obs_lat{2} = -34.612979;
    %obs_lon{2} = -58.453656;
    
    % Create Folders
    for ob = 1:total_num_agents
        ensureDirExists(sprintf('%s/Observer%i/Topo/Combined/', save_loc, ob));
        ensureDirExists(sprintf('%s/Observer%i/Synodic/Combined/', save_loc, ob));
        ensureDirExists(sprintf('%s/Observer%i/ECI/Combined/', save_loc, ob));
    end
    ensureDirExists(sprintf('%s/CrossOb/Synodic/', save_loc));
    ensureDirExists(sprintf('%s/CrossOb/ECI/', save_loc));
    
    % Add observation noise to the observation data as follows:
    % Range - 5% of the current (i.e. noiseless) range
    % Azimuth - 1.5 arc sec (credit: Dr. Utkarsh Ranjan Mishra)
    % Elevation - 1.5 arc sec (credit: Dr. Utkarsh Ranjan Mishra)
    % Note: All above quantities are drawn in a zero-mean Gaussian fashion.
    h = @(x) [atan2(x(2),x(1)); pi/2 - acos(x(3)/sqrt(x(1)^2 + x(2)^2 + x(3)^2))]; % Nonlinear measurement model
    theta_f = 1.5; % Arc-seconds of error covariance
    R_f = repmat({0.25}, 1, 6); % Range percentage error covariance
    %R_f{2} = 0.25;
    
    % Limits of the cislunar domain
    enforce_bounds = false;
    if dynamics == "CR3BP"
        low_lim = (2*42164); % Two times the GEO Distance
        up_lim = 550000;
        vel_lim = 42; % Escape velocity of the solar system
    end
    if dynamics == "2 Body"
        low_lim = 6400;
        up_lim = 2*42164;
        vel_lim = 10000;
    end
    
    total_num_clouds = 1 + num_new_clouds_per_agent * sum(fusion_information(:, 3));
    likelihood_metric_state_space = repmat({NaN(num_timesteps, 1)}, total_num_agents, total_num_clouds);
    likelihood_metric_msmt_space = repmat({NaN(num_timesteps, 1)}, total_num_agents, total_num_clouds);
    best_ent2_det_cov = repmat({NaN(num_timesteps, 1)}, total_num_agents, total_num_clouds);
    ent2_det_cov = repmat({NaN(num_timesteps, 1)}, total_num_agents, total_num_clouds);
    best_ent2_det_cov_msmt = repmat({NaN(num_timesteps, 1)}, total_num_agents, total_num_clouds);
    ent2_det_cov_msmt = repmat({NaN(num_timesteps, 1)}, total_num_agents, total_num_clouds);
    ent1 = repmat({NaN(num_timesteps, 6)}, total_num_agents, total_num_clouds);
    cross_ob_ent = repmat({NaN(num_timesteps, 1)}, total_num_agents, 1);
    %cross_ob_unnorm_ent = repmat({NaN(num_timesteps, 1)}, total_num_agents-1, 1);
    cross_ob_norm = repmat({NaN(num_timesteps, 1)}, total_num_agents, 1);
    NEES = repmat({NaN(num_timesteps, 1)}, total_num_agents, total_num_clouds);
    RMSE = repmat({NaN(num_timesteps, 6)}, total_num_agents, total_num_clouds);
    std_dev = repmat({NaN(num_timesteps, 6)}, total_num_agents, total_num_clouds);
    MC_std_dev = repmat({NaN(num_timesteps, 1)}, total_num_agents, total_num_clouds);
    mat_weight_metric = repmat({NaN(num_timesteps, 3)}, total_num_agents, total_num_clouds);
    orig_weight_metric = repmat({NaN(num_timesteps, 3)}, total_num_agents, total_num_clouds);
    MC_consistency = repmat({NaN(num_timesteps, 1)}, total_num_agents, total_num_clouds);
    num_cluster = repmat({NaN(num_timesteps, 1)}, total_num_agents, total_num_clouds);
    num_particles = repmat({NaN(num_timesteps, 1)}, total_num_agents, total_num_clouds);
    
    % Looping over the times of observation for easier propagation
    fig_num = 1;
    Xprop_truth = cell(total_num_agents, 1);
    %for ob = 1:total_num_agents
    %    Xprop_truth{ob} = combined_state_data(1, 2:end, ob);
    %end
    Xm_cloud = cell(total_num_agents, total_num_clouds);
    Xp_cloud = cell(total_num_agents, total_num_clouds);
    Xp_cloudp = cell(total_num_agents, total_num_clouds);
    for ts = 1:num_timesteps-1
        t_prev = all_timesteps(ts);
        %plot_indv_clouds = false;
        %% IOD
        for ob = 1:total_num_agents
            if ((sum(combined_msmt_data(ts_to_perform_IOD(ob):ts, 2, ob)) >= num_msmt_for_IOD(ob)) && (~agent_is_active(ob)))
                disp("Performing IOD on Ob: " + num2str(ob))
                nfit = 4;
                X0cloud_temp = zeros(num_IOD_particles, 6);
                for i = 1:num_IOD_particles
                    X0cloud_temp(i,:) = stateEstCloud(num_msmt_for_IOD(ob), ts, nfit, theta_f, R_f{ob}, combined_msmt_data(:, :, ob), low_lim, up_lim, normalization_quantities);
                end
                %theta_f = 0;
                % TODO: Check whether X0cloud clears correctly
                if (enforce_bounds)
                    X0cloud = enforceCislunarBounds(X0cloud_temp, t_prev, obs_lat{ob}, obs_lon{ob}, normalization_quantities, low_lim, up_lim, vel_lim);
                else
                    X0cloud = X0cloud_temp;
                end
                if (ob == 2)
                    ob;
                    %up_lim = 3000000;
                    %combined_msmt_data(:, 2, ob) = false;
                %    X0cloud = Xp_cloudp{1, 1};%enforceCislunarBounds(X0cloud_temp, t_prev, obs_lat{ob}, obs_lon{ob}, dist2km, vel2kms, low_lim, up_lim, vel_lim);%X0cloud(1:j,:);
                end
                Xprop_truth{ob} = combined_state_data(ts, 2:end, ob);
                plotStateSpace(X0cloud, ...
                                combined_state_data(ts, 2:end, ob), ...
                                1, ...
                                ones(size(X0cloud, 1), 1), ...
                                normalization_quantities, ...
                                colors, ...
                                sprintf('Timestep: %3.4f Hours Obs: %i', [t_prev*normalization_quantities.time2hr, ob]), ...
                                sprintf('%s/Observer%i/Topo/iodCloud.png', save_loc, ob))
                if (dynamics == "CR3BP")
                    plotting_cloud = Topo2Synodic(X0cloud, t_prev, obs_lat{ob}, obs_lon{ob});
                    plotting_truth = Topo2Synodic(combined_state_data(ts, 2:end, ob), t_prev, obs_lat{ob}, obs_lon{ob});
                    plotStateSpace(plotting_cloud, ...
                                    plotting_truth, ...
                                    1, ...
                                    ones(size(plotting_cloud, 1), 1), ...
                                    normalization_quantities, ...
                                    colors, ...
                                    sprintf('Timestep: %3.4f Hours Obs: %i', [t_prev*normalization_quantities.time2hr, ob]), ...
                                    sprintf('%s/Observer%i/Synodic/iodCloud.png', save_loc, ob))
                end
                agent_is_active(ob) = true;
                active_mask = sort([active_mask, ob]);
                num_agents = num_agents + 1;
                metric_cloud = Topo2Synodic(X0cloud, t_prev, obs_lat{ob}, obs_lon{ob});
                metric_truth = Topo2Synodic(combined_state_data(ts, 2:end, ob), t_prev, obs_lat{ob}, obs_lon{ob});
                [likelihood_metric_state_space{ob, 1}(ts), best_ent2_det_cov{ob, 1}(ts), ent2_det_cov{ob, 1}(ts), NEES{ob, 1}(ts), RMSE{ob, 1}(ts, :), std_dev{ob, 1}(ts, :), MC_std_dev{ob, 1}(ts), mat_weight_metric{ob, 1}(ts, :), MC_consistency{ob, 1}(ts), num_cluster{ob, 1}(ts), num_particles{ob, 1}(ts)] = getStateSpaceMetrics(K{ob, 1}, metric_cloud, metric_truth, cluster_by);
                [likelihood_metric_msmt_space{ob, 1}(ts), best_ent2_det_cov_msmt{ob, 1}(ts), ent2_det_cov_msmt{ob, 1}(ts)] = getMsmtSpaceMetrics(K{ob, 1}, X0cloud, combined_state_data(ts, 2:end, ob), h);

                Xp_cloudp{ob, 1} = X0cloud;
                if (ob == 2)
                    %Xp_cloudp{ob, 1} = Xp_cloudp{1, 1};
                    %theta_f = 0;
                end
            end
        end
        
        % Calculate Similarity Metrics
        for ob1 = 1:1%numel(active_mask)
            for ob2 = ob1+1:numel(active_mask)
                ob1_idx = active_mask(ob1);
                ob2_idx = active_mask(ob2);
                cross_ent_clouds_1 = Topo2Synodic(Xp_cloudp{ob1_idx, 1}, t_prev, obs_lat{ob1_idx}, obs_lon{ob1_idx});
                cross_ent_clouds_2 = Topo2Synodic(Xp_cloudp{ob2_idx, 1}, t_prev, obs_lat{ob2_idx}, obs_lon{ob2_idx});
                [cross_ob_ent{ob2_idx, ob1_idx}(ts), ~, cross_ob_norm{ob2_idx, ob1_idx}(ts)] = crossObEntropy({cross_ent_clouds_1, cross_ent_clouds_2}, cluster_by, 8, 2);
            end
        end
        if (sum(agent_is_active) >= 2) % If 2 or more agents have completed IOD
            for fuse_num = 1:size(fusion_information, 1)
                if (agent_is_active(fusion_information(fuse_num, 1)) == true && agent_is_active(fusion_information(fuse_num, 2)) == true && ...
                    fusion_information(fuse_num, 3) == true && ts == fusion_information(fuse_num, 4))
                    %% Fusion
                    disp("Fusing Clouds")
                    % Convert clouds to Synodic Frame
                    fuse_id_1 = fusion_information(fuse_num, 1);
                    fuse_id_2 = fusion_information(fuse_num, 2);
                    converted_cloud_1 = Topo2Synodic(Xp_cloudp{fuse_id_1, 1}, t_prev, obs_lat{fuse_id_1}, obs_lon{fuse_id_1});
                    converted_cloud_2 = Topo2Synodic(Xp_cloudp{fuse_id_2, 1}, t_prev, obs_lat{fuse_id_2}, obs_lon{fuse_id_2});
                    
                    % Fuse Clouds
                    [p3_simple, ~, weight_update_p3_1, weight_update_p3_2, fusion_bin_edges] = fusionMethods(converted_cloud_1, converted_cloud_2, cluster_by, Kmax, 2, Lp(fuse_id_1), disp_diagnostics, save_loc, normalization_quantities);
            
                    % Convert clouds back to Topo Frame
                    Xp_cloudp{fuse_id_1, num_clouds_per_agent(fuse_id_1)+1} = Synodic2Topo(weight_update_p3_1, t_prev, obs_lat{fuse_id_1}, obs_lon{fuse_id_1});
                    %Xp_cloudp{fuse_id_2, num_clouds_per_agent(fuse_id_2)+1} = CoordFunctions.ECI2Topo(weight_update_p3_2, t_prev, obs_lat{fuse_id_2}, obs_lon{fuse_id_2}, normalization_quantities);
                    cloud_names(fuse_id_1, num_clouds_per_agent(fuse_id_1)+1) = sprintf("Fused Obs: %i & %i @ %.f hrs", fuse_id_1, fuse_id_2, t_prev*normalization_quantities.time2hr);%, sprintf("%s %3.1f", fusion_types(3), t_prev*time2hr)];
                    
                    fusion_information(fuse_num, 3) = false;
                    num_clouds_per_agent(fuse_id_1) = num_clouds_per_agent(fuse_id_1) + num_new_clouds_per_agent;
                    %num_clouds_per_agent(fuse_id_2) = num_clouds_per_agent(fuse_id_2) + num_new_clouds_per_agent;
                    
                    for ob = 1:total_num_agents
                        if ((size(Xp_cloudp, 2) > 1) && (plot_comb_clouds(ob)))
                            plotting_cloud = cell(num_clouds_per_agent(ob));
                            for cloud = 1:num_clouds_per_agent(ob)
                                plotting_cloud{cloud} = Xp_cloudp{ob, cloud};
                            end
                            plotting_truth = Xprop_truth{ob};
                            plotStateSpaceCombined(plotting_cloud, ...
                                                    plotting_truth, ...
                                                    1:num_clouds_per_agent(ob), ...
                                                    normalization_quantities, ...
                                                    colors, ...
                                                    cloud_names(ob, :), ...
                                                    sprintf('Timestep: %3.4f Hours Fusion Ob: %i', [t_prev*normalization_quantities.time2hr, ob]), ...
                                                    sprintf('%s/Observer%i/Topo/Combined/Timestep_%i_0B_combined.png', save_loc, ob, ts))
                            plotMsmtSpaceCombined(plotting_cloud, ...
                                                    plotting_truth, ...
                                                    NaN(2, 1), ...
                                                    h, ...
                                                    num_clouds_per_agent(ob), ...
                                                    colors, ...
                                                    cloud_names(ob, :), ...
                                                    sprintf('Az-El Timestep: %3.4f Hours Fusion Ob: %i', [t_prev*normalization_quantities.time2hr, ob]), ...
                                                    sprintf('%s/Observer%i/Topo/Combined/Timestep_%i_0C_combined.png', save_loc, ob, ts), ...
                                                    false)
                            
                            plotting_cloud = cell(num_clouds_per_agent(ob));
                            for cloud = 1:num_clouds_per_agent(ob)
                                plotting_cloud{cloud} = CoordFunctions.Topo2ECI(Xp_cloudp{ob, cloud}, t_prev, obs_lat{ob}, obs_lon{ob}, normalization_quantities);
                            end
                            plotting_truth = CoordFunctions.Topo2ECI(Xprop_truth{ob}, t_prev, obs_lat{ob}, obs_lon{ob}, normalization_quantities);
                            plotStateSpaceCombined(plotting_cloud, ...
                                                    plotting_truth, ...
                                                    1:num_clouds_per_agent(ob), ...
                                                    normalization_quantities, ...
                                                    colors, ...
                                                    cloud_names(ob, :), ...
                                                    sprintf('Timestep: %3.4f Hours Fusion Ob: %i', [t_prev*normalization_quantities.time2hr, ob]), ...
                                                    sprintf('%s/Observer%i/ECI/Combined/Timestep_%i_0B_combined.png', save_loc, ob, ts))
                            
                            if (dynamics == "CR3BP")
                                plotting_cloud = cell(num_clouds_per_agent(ob));
                                for cloud = 1:num_clouds_per_agent(ob)
                                    plotting_cloud{cloud} = Topo2Synodic(Xp_cloudp{ob, cloud}, t_prev, obs_lat{ob}, obs_lon{ob});
                                end
                                plotting_truth = Topo2Synodic(Xprop_truth{ob}, t_prev, obs_lat{ob}, obs_lon{ob});
                                plotStateSpaceCombined(plotting_cloud, ...
                                                        plotting_truth, ...
                                                        1:num_clouds_per_agent(ob), ...
                                                        normalization_quantities, ...
                                                        colors, ...
                                                        cloud_names(ob, :), ...
                                                        sprintf('Timestep: %3.4f Hours Fusion Ob: %i', [t_prev*normalization_quantities.time2hr, ob]), ...
                                                        sprintf('%s/Observer%i/Synodic/Combined/Timestep_%i_0B_combined.png', save_loc, ob, ts))
                            end
                        end
                    end
                end
            end
        end

        %% Propagation
        Xm_cloud = cell(total_num_agents, max(num_clouds_per_agent));
        t_prior = all_timesteps(ts+1); % Time stamp of the prior means, weights, and covariances
        interval = t_prior - t_prev;
        for ob = active_mask
            for cloud = 1:num_clouds_per_agent(ob)
                %if ob == 2
                %    plot_indv_clouds = true;
                %    ob
                %end
                ent1{ob, cloud}(ts,:) = getDiagCov(Xp_cloudp{ob, cloud});
            
                % Propagation Step
                Xm_cloud_tmp = propagate(Xp_cloudp{ob, cloud}, t_prev, interval, obs_lat{ob}, obs_lon{ob}, normalization_quantities, dynamics);
                if (enforce_bounds)
                    Xm_cloud{ob, cloud} = enforceCislunarBounds(Xm_cloud_tmp, t_prior, obs_lat{ob}, obs_lon{ob}, normalization_quantities, low_lim, up_lim, vel_lim);
                else
                    Xm_cloud{ob, cloud} = Xm_cloud_tmp;
                end
                %end
            end
            Xprop_truth{ob} = propagate(Xprop_truth{ob}, t_prev, interval, obs_lat{ob}, obs_lon{ob}, normalization_quantities, dynamics);
        end
        fprintf("Timestamp: %1.5f\n", t_prior*normalization_quantities.time2hr);
        
        if (t_prior*normalization_quantities.time2hr >= 20)
            t_prior;
        end
        %% Update Step
        %cPoints = cell(num_agents, num_clouds_per_agent, Kmax);
        %mu_c = cell(num_agents, num_clouds_per_agent, Kmax);    mu_p = cell(total_num_agents, total_num_clouds, Kmax);
        %P_c = cell(num_agents, num_clouds_per_agent, Kmax);     P_p = cell(total_num_agents, total_num_clouds, Kmax);
        %wm = cell(num_agents, num_clouds_per_agent);            wp = cell(num_agents, num_clouds_per_agent);
        idx = cell(total_num_agents, max(num_clouds_per_agent));
        c_id = cell(total_num_agents, max(num_clouds_per_agent));
        zt = cell(total_num_agents, max(num_clouds_per_agent));
        mu_p = cell(total_num_agents, max(num_clouds_per_agent), Kmax);
        P_p = cell(total_num_agents, max(num_clouds_per_agent), Kmax);
        zt_cluster_likelihood = cell(total_num_agents, max(num_clouds_per_agent), Kmax);
        for ob = active_mask
            if (combined_msmt_data(ts+1, 2, ob) == true)
                %if ob == 2
                %    ob
                %end
                %% Msmt Exists
                % Generate noisy msmt
                R_weight = [theta_f*pi/648000, 0; 0, theta_f*pi/648000].^2;
                zt{ob} = getNoisyMeas(Xprop_truth{ob}, R_weight, h);
                
                %idx = cell(total_num_agents, num_clouds_per_agent(ob));
                for cloud = 1:num_clouds_per_agent(ob)
                    [idx{ob, cloud}, K{ob, cloud}, ~] = cluster(Xm_cloud{ob, cloud}, cluster_by, K{ob, cloud});
                end
                if (ob == 2)
                    %idx{2, 1} = idx{1, 1};
                    %K{2, 1} = K{2, 1};
                end
                %mu_p = cell(total_num_agents, max(num_clouds_per_agent), Kmax);
                %P_p = cell(total_num_agents, max(num_clouds_per_agent), Kmax);
                [cPoints, mu_c, P_c, wm, wp] = calcGMMStatistics(Xm_cloud(ob, :), idx(ob, :), num_clouds_per_agent(ob), K(ob, :), Kmax);
                for cloud = 1:num_clouds_per_agent(ob)
                    %[cPoints, mu_c, P_c, wm, wp] = calcGMMStatistics(Xm_cloud, idx{ob, cloud}, K{ob, cloud}, Kmax);
                    orig_weight_metric{ob, cloud}(ts, :) = [min(wm{1, cloud}), mean(wm{1, cloud}), max(wm{1, cloud})];
                end
        
                %% Update Step
                R_vv = [theta_f*pi/648000, 0; 0, theta_f*pi/648000].^2;
                for cloud = 1:num_clouds_per_agent(ob)
                    % Update Step
                    for k = 1:K{ob, cloud}
                        [mu_p{ob, cloud, k}, P_p{ob, cloud, k}] = kalmanUpdate(zt{ob}, cPoints{1, cloud, k}, R_vv, mu_c{1, cloud, k}, P_c{1, cloud, k}, h);
                        P_p{ob, cloud, k} = (P_p{ob, cloud, k} + P_p{ob, cloud, k}')/2;
                    end
                    % Weight update
                    [wp{1, cloud}, zt_cluster_likelihood{ob, cloud}] = weightUpdate(wm{1, cloud}, Xm_cloud{ob, cloud}, idx{ob, cloud}, zt{ob}, R_vv, h);
                end
                
                for cloud = 1:num_clouds_per_agent(ob)
                    %% Resampling 1
                    Xp_cloudp_temp = zeros(Lp(ob), length(Xprop_truth{ob}));
                    c_id_temp = zeros(Lp(ob),1);
                    for i = 1:Lp(ob)
                        [Xp_cloudp_temp(i,:), c_id_temp(i)] = drawFrom(wp{1, cloud}, mu_p(ob, cloud, :), P_p(ob, cloud, :)); 
                    end
                    Xp_cloudp{ob, cloud} = Xp_cloudp_temp;
                    c_id{ob, cloud} = c_id_temp;

                    %% Metric Calculations
                    metric_cloud = Topo2Synodic(Xp_cloudp{ob, cloud}, t_prior, obs_lat{ob}, obs_lon{ob});
                    metric_truth = Topo2Synodic(Xprop_truth{ob}, t_prior, obs_lat{ob}, obs_lon{ob});
                    [likelihood_metric_state_space{ob, cloud}(ts+1), best_ent2_det_cov{ob, cloud}(ts+1), ent2_det_cov{ob, cloud}(ts+1), NEES{ob, cloud}(ts+1), RMSE{ob, cloud}(ts+1, :), std_dev{ob, cloud}(ts+1, :), MC_std_dev{ob, cloud}(ts+1), mat_weight_metric{ob, cloud}(ts+1, :), MC_consistency{ob, cloud}(ts+1), num_cluster{ob, cloud}(ts+1), num_particles{ob, cloud}(ts+1)] = getStateSpaceMetrics(K{ob, cloud}, metric_cloud, metric_truth, cluster_by);
                    [likelihood_metric_msmt_space{ob, cloud}(ts+1), best_ent2_det_cov_msmt{ob, cloud}(ts+1), ent2_det_cov_msmt{ob, cloud}(ts+1)] = getMsmtSpaceMetrics(K{ob, cloud}, Xp_cloudp{ob, cloud}, Xprop_truth{ob}, h);
                end

            else
                %% No Msmt Exists
                zt{ob} = [NaN; NaN];
                %mu_p = cell(1, num_clouds_per_agent(ob), 1); 
                %P_p = cell(1, num_clouds_per_agent(ob), 1); 
                wm = cell(1, num_clouds_per_agent(ob));
                cPoints = cell(1, num_clouds_per_agent(ob), 1);
                for cloud = 1:num_clouds_per_agent(ob)
                    % TODO: Check whether Xp_cloud should be Xp_cloudp
                    wm{1, cloud} = zeros(1, 1);
                    Xp_cloud{ob, cloud} = Xm_cloud{ob, cloud}; cPoints{1, cloud, 1} = Xp_cloud{ob, cloud};
                    wp{1, cloud} = [1];
                    mu_p{ob, cloud, 1} = mean(Xp_cloud{ob, cloud});
                    P_p{ob, cloud, 1} = cov(Xp_cloud{ob, cloud});
                    zt_cluster_likelihood{ob, cloud} = NaN(1);
                end
                
                for cloud = 1:num_clouds_per_agent(ob)
                    %% Resampling
                    K{ob, cloud} = 1;
                    Xp_cloudp{ob, cloud} = Xm_cloud{ob, cloud}; c_id{ob, cloud} = ones(size(Xp_cloudp{ob, cloud}, 1), 1);

                    %% Metric Calculations
                    metric_cloud = Topo2Synodic(Xp_cloudp{ob, cloud}, t_prior, obs_lat{ob}, obs_lon{ob});
                    metric_truth = Topo2Synodic(Xprop_truth{ob}, t_prior, obs_lat{ob}, obs_lon{ob});
                    [likelihood_metric_state_space{ob, cloud}(ts+1), best_ent2_det_cov{ob, cloud}(ts+1), ent2_det_cov{ob, cloud}(ts+1), NEES{ob, cloud}(ts+1), RMSE{ob, cloud}(ts+1, :), std_dev{ob, cloud}(ts+1, :), MC_std_dev{ob, cloud}(ts+1), mat_weight_metric{ob, cloud}(ts+1, :), MC_consistency{ob, cloud}(ts+1), num_cluster{ob, cloud}(ts+1), num_particles{ob, cloud}(ts+1)] = getStateSpaceMetrics(Kmax, metric_cloud, metric_truth, cluster_by);
                    [likelihood_metric_msmt_space{ob, cloud}(ts+1), best_ent2_det_cov_msmt{ob, cloud}(ts+1), ent2_det_cov_msmt{ob, cloud}(ts+1)] = getMsmtSpaceMetrics(K{ob, cloud}, Xp_cloudp{ob, cloud}, Xprop_truth{ob}, h);
                end
            end
        end
        if (t_prior*normalization_quantities.time2hr >= 20)
            t_prior;
        end

        %% Plot Priors

        for ob = 1:total_num_agents
            msmt_exists = combined_msmt_data(ts+1, 2, ob);
            if (msmt_exists == true && agent_is_active(ob) == true)
                for cloud = 1:num_clouds_per_agent(ob)
                    if (plot_indv_clouds)
                        plotting_cloud = Xm_cloud{ob, cloud};
                        plotting_truth = Xprop_truth{ob};
                        plotStateSpace(plotting_cloud, ...
                                        plotting_truth, ...
                                        K{ob, cloud}, ...
                                        idx{ob, cloud}, ...
                                        normalization_quantities, ...
                                        colors, ...
                                        sprintf('Timestep: %3.4f Hours (Prior) Ob: %i', [t_prior*normalization_quantities.time2hr, ob]), ...
                                        sprintf('%s/Observer%i/Topo/Timestep_%i_1B_cloud_%i.png', save_loc, ob, ts+1, cloud))
                        plotMsmtSpace(plotting_cloud, ...
                                        plotting_truth, ...
                                        zt{ob}, ...
                                        h, ...
                                        zt_cluster_likelihood{ob, cloud}, ...
                                        K{ob, cloud}, ...
                                        idx{ob, cloud}, ...
                                        colors, ...
                                        sprintf('Az-El Timestep: %3.4f Hours (Prior) Ob: %i', [t_prior*normalization_quantities.time2hr, ob]), ...
                                        sprintf('%s/Observer%i/Topo/Timestep_%i_1C_cloud_%i.png', save_loc, ob, ts+1, cloud), ...
                                        msmt_exists)
                        
                        plotting_cloud = CoordFunctions.Topo2ECI(Xm_cloud{ob, cloud}, t_prior, obs_lat{ob}, obs_lon{ob}, normalization_quantities);
                        plotting_truth = CoordFunctions.Topo2ECI(Xprop_truth{ob}, t_prior, obs_lat{ob}, obs_lon{ob}, normalization_quantities);
                        plotStateSpace(plotting_cloud, ...
                                        plotting_truth, ...
                                        K{ob, cloud}, ...
                                        idx{ob, cloud}, ...
                                        normalization_quantities, ...
                                        colors, ...
                                        sprintf('Timestep: %3.4f Hours (Prior) Ob: %i', [t_prior*normalization_quantities.time2hr, ob]), ...
                                        sprintf('%s/Observer%i/ECI/Timestep_%i_1B_cloud_%i.png', save_loc, ob, ts+1, cloud))
                        
                        if (dynamics == "CR3BP")
                            plotting_cloud = Topo2Synodic(Xm_cloud{ob, cloud}, t_prior, obs_lat{ob}, obs_lon{ob});
                            plotting_truth = Topo2Synodic(Xprop_truth{ob}, t_prior, obs_lat{ob}, obs_lon{ob});
                            plotStateSpace(plotting_cloud, ...
                                            plotting_truth, ...
                                            K{ob, cloud}, ...
                                            idx{ob, cloud}, ...
                                            normalization_quantities, ...
                                            colors, ...
                                            sprintf('Timestep: %3.4f Hours (Prior) Ob: %i', [t_prior*normalization_quantities.time2hr, ob]), ...
                                            sprintf('%s/Observer%i/Synodic/Timestep_%i_1B_cloud_%i.png', save_loc, ob, ts+1, cloud))
                        end
                    end
                end
                if ((size(Xm_cloud, 2) > 1) && (plot_comb_clouds(ob)))
                    plotting_cloud = cell(num_clouds_per_agent(ob));
                    for cloud = 1:num_clouds_per_agent(ob)
                        plotting_cloud{cloud} = Xm_cloud{ob, cloud};
                    end
                    plotting_truth = Xprop_truth{ob};
                    plotStateSpaceCombined(plotting_cloud, ...
                                            plotting_truth, ...
                                            1:num_clouds_per_agent(ob), ...
                                            normalization_quantities, ...
                                            colors, ...
                                            cloud_names(ob, :), ...
                                            sprintf('Timestep: %3.4f Hours (Prior) Ob: %i', [t_prior*normalization_quantities.time2hr, ob]), ...
                                            sprintf('%s/Observer%i/Topo/Combined/Timestep_%i_1B_combined.png', save_loc, ob, ts+1))
                    plotMsmtSpaceCombined(plotting_cloud, ...
                                            plotting_truth, ...
                                            zt{ob}, ...
                                            h, ...
                                            num_clouds_per_agent(ob), ...
                                            colors, ...
                                            cloud_names(ob, :), ...
                                            sprintf('Az-El Timestep: %3.4f Hours (Prior) Ob: %i', [t_prior*normalization_quantities.time2hr, ob]), ...
                                            sprintf('%s/Observer%i/Topo/Combined/Timestep_%i_1C_combined.png', save_loc, ob, ts+1), ...
                                            msmt_exists)
                    
                    plotting_cloud = cell(num_clouds_per_agent(ob));
                    for cloud = 1:num_clouds_per_agent(ob)
                        plotting_cloud{cloud} = CoordFunctions.Topo2ECI(Xm_cloud{ob, cloud}, t_prior, obs_lat{ob}, obs_lon{ob}, normalization_quantities);
                    end
                    plotting_truth = CoordFunctions.Topo2ECI(Xprop_truth{ob}, t_prior, obs_lat{ob}, obs_lon{ob}, normalization_quantities);
                    plotStateSpaceCombined(plotting_cloud, ...
                                            plotting_truth, ...
                                            1:num_clouds_per_agent(ob), ...
                                            normalization_quantities, ...
                                            colors, ...
                                            cloud_names(ob, :), ...
                                            sprintf('Timestep: %3.4f Hours (Prior) Ob: %i', [t_prior*normalization_quantities.time2hr, ob]), ...
                                            sprintf('%s/Observer%i/ECI/Combined/Timestep_%i_1B_combined.png', save_loc, ob, ts+1))
                    
                    if (dynamics == "CR3BP")
                        plotting_cloud = cell(num_clouds_per_agent(ob));
                        for cloud = 1:num_clouds_per_agent(ob)
                            plotting_cloud{cloud} = Topo2Synodic(Xm_cloud{ob, cloud}, t_prior, obs_lat{ob}, obs_lon{ob});
                        end
                        plotting_truth = Topo2Synodic(Xprop_truth{ob}, t_prior, obs_lat{ob}, obs_lon{ob});
                        plotStateSpaceCombined(plotting_cloud, ...
                                                plotting_truth, ...
                                                1:num_clouds_per_agent(ob), ...
                                                normalization_quantities, ...
                                                colors, ...
                                                cloud_names(ob, :), ...
                                                sprintf('Timestep: %3.4f Hours (Prior) Ob: %i', [t_prior*normalization_quantities.time2hr, ob]), ...
                                                sprintf('%s/Observer%i/Synodic/Combined/Timestep_%i_1B_combined.png', save_loc, ob, ts+1))
                    end
                end
            end
        end
        
        if (any(combined_msmt_data(ts+1, 2, :)) && num_agents >= 2 && plot_cross_observers == true)
            %{
            plotting_cloud = cell(num_agents);
            for ob = 1:num_agents
                plotting_cloud{ob} = Xm_cloud{ob, 1};
            end
            plotting_truth = Xprop_truth{ob};
            plotStateSpaceCombined(plotting_cloud, ...
                                    plotting_truth, ...
                                    2, ...
                                    dist2km, ...
                                    vel2kms, ...
                                    colors, ...
                                    ["Ob: 1", "Ob: 2"], ...
                                    sprintf('Timestep: %3.4f Hours (Prior)', [t_prior*time2hr, ob]), ...
                                    sprintf('%s/CrossOb/Topo/Timestep_%i_1D.png', save_loc, ts+1))
            %}
            %cross_legend = arrayfun(@(i) sprintf("Ob: %d", i), active_mask);
            plotting_cloud = cell(total_num_agents, 1);
            
            for ob = active_mask
                plotting_cloud{ob} = CoordFunctions.Topo2ECI(Xm_cloud{ob, 1}, t_prior, obs_lat{ob}, obs_lon{ob}, normalization_quantities);
            end
            plotting_truth = CoordFunctions.Topo2ECI(Xprop_truth{1}, t_prior, obs_lat{1}, obs_lon{1}, normalization_quantities);
            plotStateSpaceCombined(plotting_cloud, ...
                                    plotting_truth, ...
                                    active_mask, ...
                                    normalization_quantities, ...
                                    colors, ...
                                    cloud_names(:, 1)', ...
                                    sprintf('Timestep: %3.4f Hours (Prior)', [t_prior*normalization_quantities.time2hr]), ...
                                    sprintf('%s/CrossOb/ECI/Timestep_%i_1D.png', save_loc, ts+1))
            
            if (dynamics == "CR3BP")
                plotting_cloud = cell(num_agents);
                for ob = active_mask
                    plotting_cloud{ob} = Topo2Synodic(Xm_cloud{ob, 1}, t_prior, obs_lat{ob}, obs_lon{ob});
                end
                plotting_truth = Topo2Synodic(Xprop_truth{1}, t_prior, obs_lat{1}, obs_lon{1});
                plotStateSpaceCombined(plotting_cloud, ...
                                        plotting_truth, ...
                                        active_mask, ...
                                        normalization_quantities, ...
                                        colors, ...
                                        cloud_names(:, 1)', ...
                                        sprintf('Timestep: %3.4f Hours (Prior)', [t_prior*normalization_quantities.time2hr]), ...
                                        sprintf('%s/CrossOb/Synodic/Timestep_%i_1D.png', save_loc, ts+1))
            end
        end

        %% Resampling
        %{
        c_id = cell(ob, cloud);
        for ob = 1:num_agents
            for cloud = 1:num_clouds_per_agent
                if (msmt_exists)
                    Xp_cloudp_temp = zeros(Lp, length(Xprop_truth{ob}));
                    c_id_temp = zeros(Lp,1);
                    for i = 1:Lp
                        [Xp_cloudp_temp(i,:), c_id_temp(i)] = drawFrom(wp{ob, cloud}, mu_p(ob, cloud, :), P_p(ob, cloud, :)); 
                    end
                    Xp_cloudp{ob, cloud} = Xp_cloudp_temp;
                    c_id{ob, cloud} = c_id_temp;
                else
                    K{ob, cloud} = 1;
                    Xp_cloudp{ob, cloud} = Xm_cloud{ob, cloud}; c_id{ob, cloud} = ones(length(Xp_cloudp{ob, cloud}(:,1)),1);
                end
            end
        end
        %}
    
        %% Plot Posteriors
        for ob = 1:total_num_agents
            msmt_exists = combined_msmt_data(ts+1, 2, ob);
            if agent_is_active(ob) == true
                for cloud = 1:num_clouds_per_agent(ob)
                    
                    if (plot_indv_clouds)
                        plotting_cloud = Xp_cloudp{ob, cloud};
                        plotting_truth = Xprop_truth{ob};
                        plotStateSpace(plotting_cloud, ...
                                        plotting_truth, ...
                                        K{ob, cloud}, ...
                                        c_id{ob, cloud}, ...
                                        normalization_quantities, ...
                                        colors, ...
                                        sprintf('Timestep: %3.4f Hours (Posterior) Ob: %i', [t_prior*normalization_quantities.time2hr, ob]), ...
                                        sprintf('%s/Observer%i/Topo/Timestep_%i_2B_cloud_%i.png', save_loc, ob, ts+1, cloud))
                        plotMsmtSpace(plotting_cloud, ...
                                        plotting_truth, ...
                                        zt{ob}, ...
                                        h, ...
                                        zt_cluster_likelihood{ob, cloud}, ...
                                        K{ob, cloud}, ...
                                        c_id{ob, cloud}, ...
                                        colors, ...
                                        sprintf('Az-El Timestep: %3.4f Hours (Posterior) Ob: %i', [t_prior*normalization_quantities.time2hr, ob]), ...
                                        sprintf('%s/Observer%i/Topo/Timestep_%i_2C_cloud_%i.png', save_loc, ob, ts+1, cloud), ...
                                        msmt_exists)
                        
                        plotting_cloud = CoordFunctions.Topo2ECI(Xp_cloudp{ob, cloud}, t_prior, obs_lat{ob}, obs_lon{ob}, normalization_quantities);
                        plotting_truth = CoordFunctions.Topo2ECI(Xprop_truth{ob}, t_prior, obs_lat{ob}, obs_lon{ob}, normalization_quantities);
                        plotStateSpace(plotting_cloud, ...
                                        plotting_truth, ...
                                        K{ob, cloud}, ...
                                        c_id{ob, cloud}, ...
                                        normalization_quantities, ...
                                        colors, ...
                                        sprintf('Timestep: %3.4f Hours (Posterior) Ob: %i', [t_prior*normalization_quantities.time2hr, ob]), ...
                                        sprintf('%s/Observer%i/ECI/Timestep_%i_2B_cloud_%i.png', save_loc, ob, ts+1, cloud))
                        
                        if (dynamics == "CR3BP")
                            plotting_cloud = Topo2Synodic(Xp_cloudp{ob, cloud}, t_prior, obs_lat{ob}, obs_lon{ob});
                            plotting_truth = Topo2Synodic(Xprop_truth{ob}, t_prior, obs_lat{ob}, obs_lon{ob});
                            plotStateSpace(plotting_cloud, ...
                                            plotting_truth, ...
                                            K{ob, cloud}, ...
                                            c_id{ob, cloud}, ...
                                            normalization_quantities, ...
                                            colors, ...
                                            sprintf('Timestep: %3.4f Hours (Posterior) Ob: %i', [t_prior*normalization_quantities.time2hr, ob]), ...
                                            sprintf('%s/Observer%i/Synodic/Timestep_%i_2B_cloud_%i.png', save_loc, ob, ts+1, cloud))
                        end
                    end
                end
                if (plot_comb_clouds(ob))
                    
                    plotting_cloud = cell(num_clouds_per_agent(ob));
                    for cloud = 1:num_clouds_per_agent(ob)
                        plotting_cloud{cloud} = Xp_cloudp{ob, cloud};
                    end
                    plotting_truth = Xprop_truth{ob};
                    plotStateSpaceCombined(plotting_cloud, ...
                                            plotting_truth, ...
                                            1:num_clouds_per_agent(ob), ...
                                            normalization_quantities, ...
                                            colors, ...
                                            cloud_names(ob, :), ...
                                            sprintf('Timestep: %3.4f Hours (Posterior) Ob: %i', [t_prior*normalization_quantities.time2hr, ob]), ...
                                            sprintf('%s/Observer%i/Topo/Combined/Timestep_%i_2B_combined.png', save_loc, ob, ts+1))
                    plotMsmtSpaceCombined(plotting_cloud, ...
                                            plotting_truth, ...
                                            zt{ob}, ...
                                            h, ...
                                            num_clouds_per_agent(ob), ...
                                            colors, ...
                                            cloud_names(ob, :), ...
                                            sprintf('Az-El Timestep: %3.4f Hours (Posterior) Ob: %i', [t_prior*normalization_quantities.time2hr, ob]), ...
                                            sprintf('%s/Observer%i/Topo/Combined/Timestep_%i_2C_combined.png', save_loc, ob, ts+1), ...
                                            msmt_exists)
                    
                    plotting_cloud = cell(num_clouds_per_agent(ob));
                    for cloud = 1:num_clouds_per_agent(ob)
                        plotting_cloud{cloud} = CoordFunctions.Topo2ECI(Xp_cloudp{ob, cloud}, t_prior, obs_lat{ob}, obs_lon{ob}, normalization_quantities);
                    end
                    plotting_truth = CoordFunctions.Topo2ECI(Xprop_truth{ob}, t_prior, obs_lat{ob}, obs_lon{ob}, normalization_quantities);
                    plotStateSpaceCombined(plotting_cloud, ...
                                            plotting_truth, ...
                                            1:num_clouds_per_agent(ob), ...
                                            normalization_quantities, ...
                                            colors, ...
                                            cloud_names(ob, :), ...
                                            sprintf('Timestep: %3.4f Hours (Posterior) Ob: %i', [t_prior*normalization_quantities.time2hr, ob]), ...
                                            sprintf('%s/Observer%i/ECI/Combined/Timestep_%i_2B_combined.png', save_loc, ob, ts+1))
                    
                    if (dynamics == "CR3BP")
                        plotting_cloud = cell(num_clouds_per_agent(ob));
                        for cloud = 1:num_clouds_per_agent(ob)
                            plotting_cloud{cloud} = Topo2Synodic(Xp_cloudp{ob, cloud}, t_prior, obs_lat{ob}, obs_lon{ob});
                        end
                        plotting_truth = Topo2Synodic(Xprop_truth{ob}, t_prior, obs_lat{ob}, obs_lon{ob});
                        plotStateSpaceCombined(plotting_cloud, ...
                                                plotting_truth, ...
                                                1:num_clouds_per_agent(ob), ...
                                                normalization_quantities, ...
                                                colors, ...
                                                cloud_names(ob, :), ...
                                                sprintf('Timestep: %3.4f Hours (Posterior) Ob: %i', [t_prior*normalization_quantities.time2hr, ob]), ...
                                                sprintf('%s/Observer%i/Synodic/Combined/Timestep_%i_2B_combined.png', save_loc, ob, ts+1))
                    end
                end
            end
        end
        %{
        plotting_cloud = cell(num_clouds_per_agent(ob));
        for cloud = 1:num_clouds_per_agent(ob)
            plotting_cloud{cloud} = Xp_cloudp{ob, cloud};
        end
        plotting_truth = Xprop_truth{ob};
        plotStateSpaceCombined(plotting_cloud, ...
                                plotting_truth, ...
                                num_clouds_per_agent(ob), ...
                                dist2km, ...
                                vel2kms, ...
                                colors, ...
                                cloud_names, ...
                                sprintf('Timestep: %3.4f Hours (Posterior) Ob: %i', [t_prior*time2hr, ob]), ...
                                sprintf('%s/Observer%i/Topo/Combined/Timestep_%i_2B_combined.png', save_loc, ob, ts+1))
        %}
        if (num_agents >= 2 && plot_cross_observers == true)
            %cross_legend = arrayfun(@(i) sprintf("Ob: %d", i), active_mask);
            
            plotting_cloud = cell(total_num_agents, 1);
            for ob = active_mask
                plotting_cloud{ob} = CoordFunctions.Topo2ECI(Xp_cloudp{ob, 1}, t_prior, obs_lat{ob}, obs_lon{ob}, normalization_quantities);
            end
            plotting_truth = CoordFunctions.Topo2ECI(Xprop_truth{1}, t_prior, obs_lat{1}, obs_lon{1}, normalization_quantities);
            plotStateSpaceCombined(plotting_cloud, ...
                                    plotting_truth, ...
                                    active_mask, ...
                                    normalization_quantities, ...
                                    colors, ...
                                    cloud_names(:, 1)', ...
                                    sprintf('Timestep: %3.4f Hours (Posterior)', [t_prior*normalization_quantities.time2hr]), ...
                                    sprintf('%s/CrossOb/ECI/Timestep_%i_2D.png', save_loc, ts+1))
            
            if (dynamics == "CR3BP")
                plotting_cloud = cell(num_agents);
                for ob = active_mask
                    plotting_cloud{ob} = Topo2Synodic(Xp_cloudp{ob, 1}, t_prior, obs_lat{ob}, obs_lon{ob});
                end
                plotting_truth = Topo2Synodic(Xprop_truth{1}, t_prior, obs_lat{1}, obs_lon{1});
                plotStateSpaceCombined(plotting_cloud, ...
                                        plotting_truth, ...
                                        active_mask, ...
                                        normalization_quantities, ...
                                        colors, ...
                                        cloud_names(:, 1)', ...
                                        sprintf('Timestep: %3.4f Hours (Posterior)', [t_prior*normalization_quantities.time2hr]), ...
                                        sprintf('%s/CrossOb/Synodic/Timestep_%i_2D.png', save_loc, ts+1))
            end
        end
        %{
        %% Metrics
        for ob = 1:num_agents
            for cloud = 1:num_clouds_per_agent
                if (msmt_exists)
                    metric_cloud = Topo2ECI(Xp_cloudp{ob, cloud}, tpr, obs_lat{ob}, obs_lon{ob});
                    metric_truth = Topo2ECI(Xprop_truth{ob}, tpr, obs_lat{ob}, obs_lon{ob});
                    [likelihood_metric_state_space{ob, cloud}(tau+2), ent2_det_cov{ob, cloud}(tau+2), NEES{ob, cloud}(tau+2), RMSE{ob, cloud}(tau+2, :), std_dev{ob, cloud}(tau+2, :), MC_std_dev{ob, cloud}(tau+2), mat_weight_metric{ob, cloud}(tau+2, :), MC_consistency{ob, cloud}(tau+2), num_cluster{ob, cloud}(tau+2), num_particles{ob, cloud}(tau+2)] = getStateSpaceMetrics(K{ob, cloud}, metric_cloud, metric_truth, cluster_by);
                    [likelihood_metric_msmt_space{ob, cloud}(tau+2)] = getMsmtSpaceMetrics(K{ob, cloud}, Xp_cloudp{ob, cloud}, zt{ob}, h);
                else
                    metric_cloud = Topo2ECI(Xp_cloudp{ob, cloud}, tpr, obs_lat{ob}, obs_lon{ob});
                    metric_truth = Topo2ECI(Xprop_truth{ob}, tpr, obs_lat{ob}, obs_lon{ob});
                    [likelihood_metric_state_space{ob, cloud}(tau+2), ent2_det_cov{ob, cloud}(tau+2), NEES{ob, cloud}(tau+2), RMSE{ob, cloud}(tau+2, :), std_dev{ob, cloud}(tau+2, :), MC_std_dev{ob, cloud}(tau+2), mat_weight_metric{ob, cloud}(tau+2, :), MC_consistency{ob, cloud}(tau+2), num_cluster{ob, cloud}(tau+2), num_particles{ob, cloud}(tau+2)] = getStateSpaceMetrics(Kmax, metric_cloud, metric_truth, cluster_by); % Get entropy as if you still are using six clusters
                end
            end
        end
        
        %% Fusion
        if(fuse_orig_clouds(fusion_idx) == true && tau == time_of_fusion(fusion_idx))
            disp("Fusing Clouds")
            % Convert clouds to Synodic Frame
            converted_cloud_1 = backConvertSynodic(Xp_cloudp{1, 1}, tpr, obs_lat{1}, obs_lon{1});
            converted_cloud_2 = backConvertSynodic(Xp_cloudp{2, 1}, tpr, obs_lat{2}, obs_lon{2});
            
            % Fuse Clouds
            [p3_simple, ~, weight_update_p3_1, weight_update_p3_2, fusion_bin_edges] = fusionMethods(converted_cloud_1, converted_cloud_2, cluster_by, Kmax, num_agents, Lp, disp_diagnostics, save_loc, dist2km, vel2kms, fusion_idx);
    
            % Convert clouds back to Topo Frame
            Xp_cloudp{1, num_clouds_per_agent+1} = convertToTopo(p3_simple, tpr, obs_lat{1}, obs_lon{1});
            Xp_cloudp{1, num_clouds_per_agent+2} = convertToTopo(weight_update_p3_1, tpr, obs_lat{1}, obs_lon{1});
            Xp_cloudp{2, num_clouds_per_agent+1} = convertToTopo(p3_simple, tpr, obs_lat{2}, obs_lon{2});
            Xp_cloudp{2, num_clouds_per_agent+2} = convertToTopo(weight_update_p3_2, tpr, obs_lat{2}, obs_lon{2});
            cloud_names = [cloud_names, sprintf("%s %3.1f", fusion_types(2), tpr*time2hr), sprintf("%s %3.1f", fusion_types(3), tpr*time2hr)];
            
            fuse_orig_clouds(fusion_idx) = false;
            num_clouds_per_agent = num_clouds_per_agent + num_new_clouds_per_agent;
            fusion_idx = min(size(fuse_orig_clouds, 2), fusion_idx + 1);
        end
        %}
    end
    
    %% Final Plots
    
    %Xp_cloudp = cell(ob, max(num_clouds_per_agent));
    %c_id = cell(ob, max(num_clouds_per_agent));
    plotMetrics(fig_num, 0:num_timesteps-1, cross_ob_ent(2:end, 1), cloud_names, colors, save_loc, 1, 'Entropy', 'Ob-Ob Normalized Likelihood Matrix Entropy', 'ObObEntropy.png');
    fig_num = fig_num + 1;
    %plotMetrics(fig_num, 0:num_timesteps-1, cross_ob_unnorm_ent(1, :), cloud_names, colors, save_loc, ob, 'Entropy', 'Ob-Ob Unnormalized Likelihood Matrix Entropy', 'ObObUnnormEntropy.png');
    %fig_num = fig_num + 1;
    plotMetrics(fig_num, 0:num_timesteps-1, cross_ob_norm(2:end, 1), cloud_names, colors, save_loc, 1, 'L1 Norm', 'Ob-Ob Unnormalized Likelihood Matrix L1 Norm', 'ObObNorm.png');
    fig_num = fig_num + 1;

    for ob = 1:total_num_agents
        x = 0:num_timesteps-1;
        for cloud = 1:num_clouds_per_agent(ob)
            fprintf('Final State Truth:\n')
            disp(Xprop_truth{ob});
            %{
            Xp_cloudp_temp = zeros(Lp, length(Xprop_truth{ob}));
            c_id_temp = zeros(Lp,1);
            for i = 1:Lp
                [Xp_cloudp_temp(i,:), c_id_temp(i)] = drawFrom(wp{ob, cloud}, mu_p(ob, cloud, :), P_p(ob, cloud, :)); 
            end
            Xp_cloudp{ob, cloud} = Xp_cloudp_temp;
            c_id{ob, cloud} = c_id_temp;
            %}
            ent1{ob, cloud}(end,:) = getDiagCov(Xp_cloudp{ob, cloud});
        end
        
        % Plot the results
        plotMetricsPerState(fig_num, x, std_dev(ob, :), normalization_quantities, cloud_names(ob, :), colors, save_loc, ob, 'Sigma', 'Standard Deviation', 'StdDev.png');
        fig_num = fig_num + 1;
        plotMetricsPerState(fig_num, x, RMSE(ob, :), normalization_quantities, cloud_names(ob, :), colors, save_loc, ob, 'RMSE', 'RMSE', 'RMSE.png');
        fig_num = fig_num + 1;
    
        plotMetrics(fig_num, x, best_ent2_det_cov(ob, :), cloud_names(ob, :), colors, save_loc, ob, 'Best Mode Entropy', 'Best Mode Entropy (State Space) Ob: %i', 'BestEntropyState.png');
        fig_num = fig_num + 1;
        plotMetrics(fig_num, x, ent2_det_cov(ob, :), cloud_names(ob, :), colors, save_loc, ob, 'Weighted Entropy', 'Weighted Entropy (State Space) Ob: %i', 'WeightedEntropyState.png');
        fig_num = fig_num + 1;
        plotMetrics(fig_num, x, best_ent2_det_cov_msmt(ob, :), cloud_names(ob, :), colors, save_loc, ob, 'Best Mode Entropy', 'Best Mode Entropy (Msmt Space) Ob: %i', 'BestEntropyMsmt.png');
        fig_num = fig_num + 1;
        plotMetrics(fig_num, x, ent2_det_cov_msmt(ob, :), cloud_names(ob, :), colors, save_loc, ob, 'Weighted Entropy', 'Weighted Entropy (Msmt Space) Ob: %i', 'WeightedEntropyMsmt.png');
        fig_num = fig_num + 1;
        
        plotMetrics(fig_num, x, likelihood_metric_state_space(ob, :), cloud_names(ob, :), colors, save_loc, ob, 'Log-Likelihood', 'Log-Likelihood Ob: %i', 'Likelihood.png');
        fig_num = fig_num + 1;
        plotMetrics(fig_num, x, likelihood_metric_msmt_space(ob, :), cloud_names(ob, :), colors, save_loc, ob, 'Log-Likelihood Msmt Space', 'Log-Likelihood Ob: %i', 'MsmtLikelihood.png');
        fig_num = fig_num + 1;
    
        plotMetrics(fig_num, x, NEES(ob, :), cloud_names(ob, :), colors, save_loc, ob, 'NEES', 'NEES Ob: %i', 'NEES.png');
        fig_num = fig_num + 1;
        plotMetrics(fig_num, x, MC_std_dev(ob, :), cloud_names(ob, :), colors, save_loc, ob, 'Monte Carlo (Single Run) 2 Sigma Example', '2 Sigma Ob: %i', 'MC_2_sigma.png');
        fig_num = fig_num + 1;
        plotMetrics(fig_num, x, MC_consistency(ob, :), cloud_names(ob, :), colors, save_loc, ob, 'Monte Carlo (Single Run) Consistency Example', 'Consistency Ob: %i', 'MC_consistency.png');
        fig_num = fig_num + 1;
        %plotMetrics(fig_num, x, RMSE(ob, :), cloud_names, colors, save_loc, ob, 'RMSE', 'RMSE Ob: %i', 'RMSE.png');
        %fig_num = fig_num + 1;
    
        plotMetrics(fig_num, x, num_cluster(ob, :), cloud_names(ob, :), colors, save_loc, ob, 'Number of Clusters', 'Number of Clusters Ob: %i', 'NumClusters.png');
        fig_num = fig_num + 1;
        plotMetrics(fig_num, x, num_particles(ob, :), cloud_names(ob, :), colors, save_loc, ob, 'Number of Particles', 'Number of Particles Ob: %i', 'NumParticles.png');
        fig_num = fig_num + 1;
        
        f = figure(fig_num);
        fig_num = fig_num + 1;
        f.WindowState = 'maximized';
        hold on;
        for cloud = 1:num_clouds_per_agent(ob)
            %style = '--'; 
            lw = 2;
            %if cloud == chosen_method(ob)
            %    style = '-'; lw = 3;
            %end
            plot(x, mat_weight_metric{ob, cloud}(:, 1), 'LineStyle', '--', 'Color', colors(cloud), 'LineWidth', lw);
            plot(x, mat_weight_metric{ob, cloud}(:, 2), 'LineStyle', '-', 'Color', colors(cloud), 'LineWidth', lw);
            plot(x, mat_weight_metric{ob, cloud}(:, 3), 'LineStyle', '--', 'Color', colors(cloud), 'LineWidth', lw);
        end
        %plot(x, num_particles{ob, 1}', x, num_particles{ob, 2}', x, num_particles{ob, 4}', x, num_particles{ob, 4}')
        xlabel('Filter Step #')
        ylabel('Min, Max, and Avg Weights')
        title(sprintf('GMM Weights Ob: %i', ob))
        %legend(cloud_names, 'Location', 'best')
        hold off;
        sg = sprintf('%s/Observer%i/%s.png', save_loc, ob, 'GMMWeights');
        drawnow;
        pause(0.5);
        exportgraphics(f, sg, 'Resolution', 150);
        close(f);
    
        f = figure(fig_num);
        fig_num = fig_num + 1;
        f.WindowState = 'maximized';
        hold on;
        for cloud = 1:num_clouds_per_agent(ob)
            %style = '--'; 
            lw = 2;
            %if cloud == chosen_method(ob)
            %    style = '-'; lw = 3;
            %end
            plot(x, orig_weight_metric{ob, cloud}(:, 1), 'LineStyle', '--', 'Color', colors(cloud), 'LineWidth', lw);
            plot(x, orig_weight_metric{ob, cloud}(:, 2), 'LineStyle', '-', 'Color', colors(cloud), 'LineWidth', lw);
            plot(x, orig_weight_metric{ob, cloud}(:, 3), 'LineStyle', '--', 'Color', colors(cloud), 'LineWidth', lw);
        end
        %plot(x, num_particles{ob, 1}', x, num_particles{ob, 2}', x, num_particles{ob, 4}', x, num_particles{ob, 4}')
        xlabel('Filter Step #')
        ylabel('Min, Max, and Avg Weights')
        title(sprintf('GMM Prior Weights (via Clustering) Ob: %i', ob))
        %legend(cloud_names, 'Location', 'best')
        hold off;
        sg = sprintf('%s/Observer%i/%s.png', save_loc, ob, 'PriorWeights');
        drawnow;
        pause(0.5);
        exportgraphics(f, sg, 'Resolution', 150);
        close(f);
        legend_string = ['Cloud', 'Truth']
        for cloud = 1:num_clouds_per_agent(ob)
            % Plot the results
            f = figure;
            fig_num = fig_num + 1;
            subplot(2,1,1)
            hold on;
            for k = 1:K{ob, cloud}
                clusterPoints = Xp_cloudp{ob, cloud}(c_id{ob, cloud} == k, :);
                %mu_pExp(k,:) = mu_p{ob, cloud, k};
                scatter3(clusterPoints(:,1), clusterPoints(:,2), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
                hold on;
            end
            %plot3(mu_pExp(:,1), mu_pExp(:,2), mu_pExp(:,3), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
            hold on;
            plot3(Xprop_truth{ob}(1), Xprop_truth{ob}(2), Xprop_truth{ob}(3), 'x','MarkerSize', 20, 'LineWidth', 3)
            title('Posterior Distribution (Position) Ob: %i', ob);
            xlabel('X');
            ylabel('Y');
            zlabel('Z');
            legend(legend_string);
            grid on;
            view(3);
            hold off;
            
            subplot(2,1,2)
            hold on;
            for k = 1:K{ob, cloud}
                clusterPoints = Xp_cloudp{ob, cloud}(c_id{ob, cloud} == k, :);
                scatter3(clusterPoints(:,4), clusterPoints(:,5), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
                hold on;
            end
            %plot3(mu_pExp(:,4), mu_pExp(:,5), mu_pExp(:,6), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
            hold on;
            plot3(Xprop_truth{ob}(4), Xprop_truth{ob}(5), Xprop_truth{ob}(6), 'x','MarkerSize', 20, 'LineWidth', 3)
            title('Posterior Distribution (Velocity) Ob: %i', ob);
            xlabel('Vx');
            ylabel('Vy');
            zlabel('Vz');
            legend(legend_string);
            grid on;
            view(3);
            hold off;
            close(f);
        
            % Plot planar projections
            f = figure;
            fig_num = fig_num + 1;
            set(gcf, 'units','normalized','outerposition',[0 0 1 1])
            subplot(2,3,1)
            for k = 1:K{ob, cloud}
                clusterPoints = Xp_cloudp{ob, cloud}(c_id{ob, cloud} == k, :);
                scatter(clusterPoints(:,1), clusterPoints(:,2), 'filled', 'MarkerFaceColor', colors(k));
                hold on;
            end
            %plot(mu_pExp(:,1), mu_pExp(:,2), '+', 'MarkerSize', 10, 'LineWidth', 3);
            hold on;
            plot(Xprop_truth{ob}(1), Xprop_truth{ob}(2), 'kx','MarkerSize', 20, 'LineWidth', 3)
            title('X-Y');
            xlabel('X');
            ylabel('Y');
            legend(legend_string);
            hold off;
            
            subplot(2,3,2)
            for k = 1:K{ob, cloud}
                clusterPoints = Xp_cloudp{ob, cloud}(c_id{ob, cloud} == k, :);
                scatter(clusterPoints(:,1), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
                hold on;
            end
            %plot(mu_pExp(:,1), mu_pExp(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
            hold on;
            plot(Xprop_truth{ob}(1), Xprop_truth{ob}(3), 'kx','MarkerSize', 20, 'LineWidth', 3)
            title('X-Z');
            xlabel('X');
            ylabel('Z');
            legend(legend_string);
            hold off;
            
            subplot(2,3,3)
            for k = 1:K{ob, cloud}
                clusterPoints = Xp_cloudp{ob, cloud}(c_id{ob, cloud} == k, :);
                scatter(clusterPoints(:,2), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
                hold on;
            end
            %plot(mu_pExp(:,2), mu_pExp(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
            hold on;
            plot(Xprop_truth{ob}(2), Xprop_truth{ob}(3), 'kx','MarkerSize', 20, 'LineWidth', 3)
            title('Y-Z');
            xlabel('Y');
            ylabel('Z');
            legend(legend_string);
            hold off;
            
            subplot(2,3,4)
            for k = 1:K{ob, cloud}
                clusterPoints = Xp_cloudp{ob, cloud}(c_id{ob, cloud} == k, :);
                scatter(clusterPoints(:,4), clusterPoints(:,5), 'filled', 'MarkerFaceColor', colors(k));
                hold on;
            end
            %plot(mu_pExp(:,4), mu_pExp(:,5), '+', 'MarkerSize', 10, 'LineWidth', 3);
            hold on;
            plot(Xprop_truth{ob}(4), Xprop_truth{ob}(5), 'kx','MarkerSize', 20, 'LineWidth', 3)
            title('Xdot-Ydot');
            xlabel('Xdot');
            ylabel('Ydot');
            legend(legend_string);
            hold off;
            
            subplot(2,3,5)
            for k = 1:K{ob, cloud}
                clusterPoints = Xp_cloudp{ob, cloud}(c_id{ob, cloud} == k, :);
                scatter(clusterPoints(:,4), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
                hold on;
            end
            %plot(mu_pExp(:,4), mu_pExp(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3);
            hold on;
            plot(Xprop_truth{ob}(4), Xprop_truth{ob}(6), 'kx','MarkerSize', 20, 'LineWidth', 3)
            title('Xdot-Zdot');
            xlabel('Xdot');
            ylabel('Zdot');
            legend(legend_string);
            hold off;
            
            subplot(2,3,6)
            for k = 1:K{ob, cloud}
                clusterPoints = Xp_cloudp{ob, cloud}(c_id{ob, cloud} == k, :);
                scatter(clusterPoints(:,5), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
                hold on;
            end
            %plot(mu_pExp(:,5), mu_pExp(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3);
            hold on;
            plot(Xprop_truth{ob}(5), Xprop_truth{ob}(6), 'kx','MarkerSize', 20, 'LineWidth', 3)
            title('Ydot-Zdot');
            xlabel('Ydot');
            ylabel('Zdot');
            legend(legend_string);
            hold off;
            
            sg = sprintf('Timestamp: %1.5f Ob: %i', [t_prior*normalization_quantities.time2hr, ob]);
            sgtitle(sg)
            saveas(gcf, save_loc + '/Observer' + num2str(ob) + '/finalDistribution_normK_cloud_' + num2str(cloud) + '.png', 'png');
            close(f);
            % savefig(gcf, 'nextObservedTracklet_normK.fig');
            %}
            
            %%save("./Outside2/stdevs.mat", "ent1");
        end
    end
    
    save(save_loc + '/ExperimentTimesteps.mat', 'all_timesteps')
    save(save_loc + '/MC_consistency.mat', 'MC_consistency')
    %save(save_loc + '/MC_std_dev.mat', 'MC_std_dev')
    save(save_loc + '/std_dev_per_state.mat', 'std_dev')
    save(save_loc + '/RMSE.mat', 'RMSE')
    save(save_loc + '/BestEntropyState.mat', 'best_ent2_det_cov')
    save(save_loc + '/WeightedEntropyState.mat', 'ent2_det_cov')
    save(save_loc + '/BestEntropyMsmt.mat', 'best_ent2_det_cov_msmt')
    save(save_loc + '/WeightedEntropyMsmt.mat', 'ent2_det_cov_msmt')
    save(save_loc + '/CrossObserverEntropy.mat', 'cross_ob_ent')
    %save(save_loc + '/CrossObserverUnnormEntropy.mat', 'cross_ob_unnorm_ent')
    save(save_loc + '/CrossObserverNorm.mat', 'cross_ob_norm')
    save(save_loc + '/Likelihood_state.mat', 'likelihood_metric_state_space')
    save(save_loc + '/Likelihood_msmt.mat', 'likelihood_metric_msmt_space')
    save(save_loc + '/Num_cluster.mat', 'num_cluster')
    save(save_loc + '/cloud_names.mat', 'cloud_names')
    % Finish timer
    toc
end
%% Misc Functions

function Hx = linHx(mu)
    Hk_AZ = [-mu(2)/(mu(1)^2 + mu(2)^2), mu(1)/(mu(1)^2 + mu(2)^2), 0, 0, 0, 0]; % Azimuth angle linearization
    Hk_EL = [-(mu(1)*mu(3))/((mu(1)^2 + mu(2)^2 + mu(3)^2)*sqrt(mu(1)^2+mu(2)^2)), ...
             -(mu(2)*mu(3))/((mu(1)^2 + mu(2)^2 + mu(3)^2)*sqrt(mu(1)^2+mu(2)^2)), ...
             sqrt(mu(1)^2 + mu(2)^2)/(mu(1)^2 + mu(2)^2 + mu(3)^2), 0, 0, 0];

    Hx = [Hk_AZ; Hk_EL];
end

% Adds process noise to the un-noised state vector
function [Xm] = procNoise(X)
    Q = (0.000*diag(abs(X))).^2; % Process noise is 1% of each state vector component
    Xm = mvnrnd(X,Q);
end


function [x_p, pos] = drawFrom(w, mu, P)
    cmf_w = zeros(length(w),1); % Cumulative mass function of the weights
    cmf_w(1,1) = w(1);
    for j = 2:length(w)
        cmf_w(j,1) = cmf_w(j-1,1) + w(j);
    end

    wtoken = rand;
    
    % Use binary search
    left = 1;
    right = length(cmf_w(:,1));

    while (left <= right)
        mid = floor((left + right) / 2);
        if (cmf_w(mid) == wtoken)
            pos = mid;
            return
        elseif (cmf_w(mid) < wtoken)
            left = mid + 1;
        else
            right = mid - 1;
        end
    end
    pos = left;

    if(pos > length(mu))
        pos = length(mu); % Correction for rounding error
    end

    mu_t = mu{pos};
    R_t = (P{pos} + P{pos}')/2;
    x_p = mvnrnd(mu_t, R_t)';
end


function zk = getNoisyMeas(Xtruth, R, h)
    mzkm = h(Xtruth);
    zk = mvnrnd(mzkm, R);
    zk = zk'; % Make into column vector
end

%{
function [combined_msmts, combined_states, all_timesteps] = combineMsmts(full_pos_datasets, full_vel_datasets, partial_datasets)
    n = numel(full_pos_datasets);
    timestep_sets = cell(n, 1);

    % Collect all timesteps from each dataset
    for i = 1:n
        timestep_sets{i} = full_pos_datasets{i}(:,1);
    end

    % All unique timesteps across all datasets
    all_timesteps = unique(vertcat(timestep_sets{:}));
    num_timesteps = numel(all_timesteps);

    % Initialize output: [timestep, msmt_exists, m1, m2, m3] × n
    combined_msmts = NaN(num_timesteps, 5, n);
    combined_msmts(:,1,:) = repmat(all_timesteps, 1, 1, n);
    combined_msmts(:,2,:) = zeros(num_timesteps, 1, n);

    combined_states = NaN(num_timesteps, 7, n);
    combined_states(:,1,:) = repmat(all_timesteps, 1, 1, n);
    
    for i = 1:n
        full_pos_data = full_pos_datasets{i};
        full_vel_data = full_vel_datasets{i};

        [~, ia_states, ib_states] = intersect(all_timesteps, full_pos_data(:,1));
        partial_data = partial_datasets{i};
        [~, ia_msmts, ib_msmts] = intersect(all_timesteps, partial_data(:,1));

        combined_msmts(ia_msmts,2,i) = 1;
        combined_msmts(ia_msmts,3:5,i) = partial_data(ib_msmts, 2:4);
        combined_states(ia_states,2:7,i) = [full_pos_data(ib_states,2:4), full_vel_data(ib_states, 2:4)];   % copy measurements
    end
end
%}

function [combined_msmts, combined_states, all_timesteps] = combineMsmts(full_pos_datasets, full_vel_datasets, partial_datasets)
    tol = 1e-8;  % Tolerance for comparing timesteps
    n = numel(full_pos_datasets);
    timestep_list = [];

    % Collect all timesteps from each dataset
    for i = 1:n
        timestep_list = [timestep_list; full_pos_datasets{i}(:,1)];
    end

    % Sort timesteps and merge using tolerance
    timestep_list = sort(timestep_list);
    all_timesteps = [];
    if ~isempty(timestep_list)
        all_timesteps = timestep_list(1);
        for k = 2:length(timestep_list)
            if abs(timestep_list(k) - all_timesteps(end)) > tol
                all_timesteps(end+1,1) = timestep_list(k);
            end
        end
    end
    num_timesteps = numel(all_timesteps);

    % Initialize output arrays
    combined_msmts = NaN(num_timesteps, 5, n);
    combined_msmts(:,1,:) = repmat(all_timesteps, 1, 1, n);
    combined_msmts(:,2,:) = zeros(num_timesteps, 1, n);  % msmt_exists flag

    combined_states = NaN(num_timesteps, 7, n);
    combined_states(:,1,:) = repmat(all_timesteps, 1, 1, n);

    for i = 1:n
        full_pos_data = full_pos_datasets{i};
        full_vel_data = full_vel_datasets{i};
        partial_data   = partial_datasets{i};

        % Match timesteps with tolerance
        ia_states = matchTimesteps(all_timesteps, full_pos_data(:,1), tol);
        ib_states = find(~isnan(ia_states));
        ia_states = ia_states(ib_states);

        ia_msmts = matchTimesteps(all_timesteps, partial_data(:,1), tol);
        ib_msmts = find(~isnan(ia_msmts));
        ia_msmts = ia_msmts(ib_msmts);

        % Copy in the data
        combined_msmts(ia_msmts,2,i)   = 1;
        combined_msmts(ia_msmts,3:5,i) = partial_data(ib_msmts, 2:4);

        combined_states(ia_states,2:7,i) = [full_pos_data(ib_states,2:4), ...
                                            full_vel_data(ib_states,2:4)];
    end
end

% Helper function to match timesteps using a tolerance
function matched_indices = matchTimesteps(reference_ts, query_ts, tol)
    matched_indices = NaN(length(query_ts),1);
    for i = 1:length(query_ts)
        diff = abs(reference_ts - query_ts(i));
        idx = find(diff < tol, 1, 'first');
        if ~isempty(idx)
            matched_indices(i) = idx;
        end
    end
end



%% IOD Functions

function [dX_coeffs] = polyDeriv(X_coeffs)
    
    dX_coeffs = zeros(1, length(X_coeffs)-1);
    for j = length(X_coeffs):-1:2
        dX_coeffs(length(X_coeffs)+1-j) = X_coeffs(length(X_coeffs)+1-j)*(j-1);
    end
end


function [Xfit] = stateEstCloud(num_msmt_for_IOD, ts, nfit, theta_f, R_f, combined_msmt_data, low_lim, up_lim, normalization_quantities)
    msmt_existance_mask = combined_msmt_data(:, 2) == true;
    times_of_msmts = combined_msmt_data(ts - num_msmt_for_IOD+1:ts, 1);
    mu_t = reshape(combined_msmt_data(msmt_existance_mask, 3:5)', [], 1);
    
    R_x = (R_f .* combined_msmt_data(msmt_existance_mask, 3)).^2;
    R_y_z = repmat((theta_f * 4.84814e-6)^2, sum(msmt_existance_mask), 1);
    R_t = reshape([R_x R_y_z R_y_z].', [], 1); 
    R_t = diag(R_t);

    data_vec = mvnrnd(mu_t, R_t)';
    
    noised_obs2 = [combined_msmt_data(msmt_existance_mask, 1), reshape(data_vec, 3, []).'];
    noised_obs = noised_obs2;
    for i = 1:length(noised_obs2(:,1))
        %noised_obs2(i,2:4) = data_vec(3*(i-1)+1:3*(i-1)+3,1); % AZ-EL Measurements
        noised_obs(i,2) = unifrnd(low_lim, up_lim)/normalization_quantities.dist2km;
        % while(noised_obs(i,2) < low_lim/dist2km || noised_obs(i,2) > up_lim/dist2km)
        %     noised_obs(i,2) = mvnrnd(partial_ts(i,2), (R_f*partial_ts(i,2))^2);
        % end
    end
    
    hdo = noised_obs(1:end, :);

    % Convert observation data into [X, Y, Z] data in the topographic frame.
                
    hdR = zeros(length(hdo(:,1)),4); % Convert quantities of hdo to [X, Y, Z]
    hdR(:,1) = hdo(:,1); % Timestamp stays the same
    hdR(:,2) = hdo(:,2) .* cos(hdo(:,4)) .* cos(hdo(:,3)); % Conversion to X
    hdR(:,3) = hdo(:,2) .* cos(hdo(:,4)) .* sin(hdo(:,3)); % Conversion to Y
    hdR(:,4) = hdo(:,2) .* sin(hdo(:,4)); % Conversion to Z
    
    [~, times_idx_hdR] = ismembertol(times_of_msmts, hdR(:,1), 1e-6);
    hdR_p = hdR(times_idx_hdR, :); % Matrix for a partial half-day observation

    % Fit polynomials for X, Y, and Z (Cubic for X, Quadratic for X and Y)
    coeffs_X = polyfit(hdR_p(:,1), hdR_p(:,2), nfit);
    coeffs_Y = polyfit(hdR_p(:,1), hdR_p(:,3), nfit);
    coeffs_Z = polyfit(hdR_p(:,1), hdR_p(:,4), nfit);
    
    % Predicted values for X, Y, and Z given the polynomial fits
    X_fit = polyval(coeffs_X, hdR_p(:,1));
    Y_fit = polyval(coeffs_Y, hdR_p(:,1));
    Z_fit = polyval(coeffs_Z, hdR_p(:,1));

    coeffs_dX = polyDeriv(coeffs_X);
    coeffs_dY = polyDeriv(coeffs_Y);
    coeffs_dZ = polyDeriv(coeffs_Z);
    
    % Predicted values for Xdot, Ydot, and Zdot given the polynomial fits
    Xdot_fit = polyval(coeffs_dX, hdR_p(:,1));
    Ydot_fit = polyval(coeffs_dY, hdR_p(:,1));
    Zdot_fit = polyval(coeffs_dZ, hdR_p(:,1));
    
    Xfit = [X_fit(end,1); Y_fit(end,1); Z_fit(end,1); Xdot_fit(end,1); Ydot_fit(end,1); Zdot_fit(end,1)];
end


%% Coordinate Functions

function [reo_topo] = getObserverPos(t_stamp, obs_lat, obs_lon)
    % First step: Obtain X_{eo}^{ECI} 
    elevation = 103.8;

    UTC_vec_orig = [2024	5	3	2	41	15]; % Initial UTC vector at t_stamp = 0
    t_add_dim = t_stamp * (4.342); % Convert the time to add to a dimensional quantity
    UTC_vec = datevec(datetime(UTC_vec_orig) + t_add_dim); % You will need this for calculating r_{eo} and v_{eo}

    reo_dim = lla2eci([obs_lat obs_lon, elevation], UTC_vec); % ECI frame only
    reo_nondim = reo_dim'/(1000*384400); % Conversion to non-dimensional units in the ECI frame

    z_hat_topo = reo_nondim/norm(reo_nondim); % Convert to topocentric reference frame
    
    x_hat_topo_unorm = cross(z_hat_topo, [0, 0, 1]'); 
    x_hat_topo = x_hat_topo_unorm/norm(x_hat_topo_unorm); % Remember to normalize

    y_hat_topo_unorm = cross(x_hat_topo, z_hat_topo);
    y_hat_topo = y_hat_topo_unorm/norm(y_hat_topo_unorm); % Remember to normalize

    reo_topo = [dot(reo_nondim, x_hat_topo), dot(reo_nondim, y_hat_topo), dot(reo_nondim, z_hat_topo)];
end


function [X_bt] = Topo2Synodic(X_ot, t_stamp, obs_lat, obs_lon)
    % First step: Obtain X_{eo}^{ECI} 

    elevation = 103.8;
    mu = 1.2150582e-2;

    UTC_vec_orig = [2024	5	3	2	41	15]; % Initial UTC vector at t_stamp = 0
    t_add_dim = t_stamp * (4.342); % Convert the time to add to a dimensional quantity
    UTC_vec = datevec(datetime(UTC_vec_orig) + t_add_dim); % You will need this for calculating r_{eo} and v_{eo}

    delt_add_dim = t_add_dim - 1/86400;
    delt_updatedUTCtime = datetime(UTC_vec_orig) + delt_add_dim;
    delt_updatedUTCvec = datevec(delt_updatedUTCtime);

    reo_dim = lla2eci([obs_lat obs_lon, elevation], UTC_vec);
    delt_reodim = lla2eci([obs_lat obs_lon, elevation], delt_updatedUTCvec);
    veo_dim = reo_dim - delt_reodim; % Finite difference

    reo_nondim = reo_dim'/(1000*384400); % Conversion to non-dimensional units in the ECI frame
    veo_nondim = veo_dim'*(4.342*86400)/(1000*384400); % Conversion to non-dimensional units in the ECI frame

    z_hat_topo = reo_nondim/norm(reo_nondim);
    x_hat_topo = cross(z_hat_topo, [0, 0, 1]')/norm(cross(z_hat_topo, [0, 0, 1]'));
    y_hat_topo = cross(x_hat_topo, z_hat_topo)/norm(cross(x_hat_topo, z_hat_topo));
    
    A = [x_hat_topo'; y_hat_topo'; z_hat_topo']; % Computing A as DCM for transforming between ECI and topographic reference frame

    dmag_dt = dot(reo_nondim, veo_nondim)/norm(reo_nondim);
    
    zhat_dot_topo = (veo_nondim * norm(reo_nondim) - reo_nondim * dmag_dt)/(norm(reo_nondim))^2;
    xhat_dot_topo = cross(zhat_dot_topo, [0, 0, 1]')/norm(cross(z_hat_topo, [0,0,1]')) - dot(x_hat_topo, cross(zhat_dot_topo, [0, 0, 1]'))*x_hat_topo;
    yhat_dot_topo = (cross(xhat_dot_topo, z_hat_topo) + cross(x_hat_topo, zhat_dot_topo))/norm(cross(x_hat_topo, z_hat_topo)) - dot(y_hat_topo, cross(xhat_dot_topo, z_hat_topo) + cross(x_hat_topo, zhat_dot_topo))*y_hat_topo;

    dA_dt = [xhat_dot_topo'; yhat_dot_topo'; zhat_dot_topo'];

    num_particles = size(X_ot, 1);
    X_bt = zeros(size(X_ot));
    for particle = 1:num_particles
        rot_topo = X_ot(particle, 1:3)'; % First three components of the state vector
        vot_topo = X_ot(particle, 4:6)'; % Last three components of the state vector

        rot_ECI = A^(-1)*rot_topo;
        vot_ECI = A^(-1)*(vot_topo - dA_dt*rot_ECI);
    
        % Calculating X_{ET} in the synodic frame with our above quantities
        
        ret_ECI = reo_nondim + rot_ECI;
        vet_ECI = veo_nondim + vot_ECI;
    
        R3 = [cos(t_stamp), -sin(t_stamp), 0; sin(t_stamp), cos(t_stamp), 0; 0, 0, 1];
        dR3_dt = [-sin(t_stamp), -cos(t_stamp), 0; cos(t_stamp), -sin(t_stamp), 0; 0, 0, 0];
    
        ret_S = R3^(-1)*ret_ECI;
        vet_S = R3^(-1)*(vet_ECI - dR3_dt*ret_S);
    
        r_be = [-mu, 0, 0]';
        v_be = [0, 0, 0]';
    
        r_bt = r_be + ret_S; % In synodic reference frame
        v_bt = v_be + vet_S; % In synodic reference frame
    
        X_bt(particle, :) = [r_bt', v_bt'];
    end
end


% Used for converting between X_{BT} in the synodic frame and X_{OT} in the
% topocentric frame for a single state
function [X_ot] = Synodic2Topo(X_bt, t_stamp, obs_lat, obs_lon)
    % Insert code for obtaining vector between center of Earth and observer
    elevation = 103.8;
    
    mu = 1.2150582e-2;
    rbe = [-mu, 0, 0]'; % Position vector relating center of earth to barycenter

    UTC_vec_orig = [2024	5	3	2	41	15];
    t_add_dim = t_stamp * (4.342);
    UTC_vec = datevec(datetime(UTC_vec_orig) + t_add_dim);

    delt_add_dim = -1/86400;
    delt_updatedUTCtime = datetime(UTC_vec) + delt_add_dim;
    delt_updatedUTCvec = datevec(delt_updatedUTCtime);

    reo_dim = lla2eci([obs_lat obs_lon, elevation], UTC_vec);
    delt_reodim = lla2eci([obs_lat obs_lon, elevation], delt_updatedUTCvec);
    veo_dim = reo_dim - delt_reodim;

    R_z = [cos(t_stamp), -sin(t_stamp), 0; sin(t_stamp), cos(t_stamp), 0; 0, 0, 1];
    dRz_dt = [-sin(t_stamp), -cos(t_stamp), 0; cos(t_stamp), -sin(t_stamp), 0; 0, 0, 0];

    reo_nondim = reo_dim'/(1000*384400); % Conversion to non-dimensional units and ECI frame
    veo_nondim = veo_dim'*(4.342*86400)/(1000*384400);

    num_particles = size(X_bt, 1);
    X_ot = zeros(size(X_bt));
    for particle = 1:num_particles
        rot_ECI = -reo_nondim + R_z*(-rbe + X_bt(particle, 1:3)');
        vot_ECI = -veo_nondim + R_z*(X_bt(particle, 4:6)') + dRz_dt*(-rbe + X_bt(particle, 1:3)');
        % Finally, we convert from the ECI frame to the topographic frame
    
        % Step 1: Find the unit vectors governing this topocentric frame
        z_hat_topo = reo_nondim/norm(reo_nondim);
    
        x_hat_topo_unorm = cross(z_hat_topo, [0, 0, 1]'); % We choose a 
        % reference vector such as the North Pole, but we have several 
        % choices regarding the second vector
      
        x_hat_topo = x_hat_topo_unorm/norm(x_hat_topo_unorm); % Remember to normalize
    
        y_hat_topo_unorm = cross(x_hat_topo, z_hat_topo);
        y_hat_topo = y_hat_topo_unorm/norm(y_hat_topo_unorm); % Remember to normalize
    
        % Step 2: Convert all of the components of 'rot' from our aligned reference
        % frames to this new topocentric frame.
        
        rot_topo = [dot(rot_ECI, x_hat_topo), dot(rot_ECI, y_hat_topo), dot(rot_ECI, z_hat_topo)];
    
        % Step 3: Handle the time derivatives of vot_topo = d/dt (rot_topo)
        R_topo = [x_hat_topo'; y_hat_topo'; z_hat_topo']; % DCM relating ECI to topocentric coordinate frame
        dmag_dt = dot(reo_nondim, veo_nondim)/norm(reo_nondim); % How the magnitude of r_eo changes w.r.t. time
        
        zhat_dot_topo = (veo_nondim*norm(reo_nondim) - reo_nondim*dmag_dt)/(norm(reo_nondim))^2;
        xhat_dot_topo = cross(zhat_dot_topo, [0, 0, 1]')/norm(cross(z_hat_topo, [0,0,1]')) - dot(x_hat_topo, cross(zhat_dot_topo, [0, 0, 1]'))*x_hat_topo;
        yhat_dot_topo = (cross(xhat_dot_topo, z_hat_topo) + cross(x_hat_topo, zhat_dot_topo))/norm(cross(x_hat_topo, z_hat_topo)) - dot(y_hat_topo, cross(xhat_dot_topo, z_hat_topo) + cross(x_hat_topo, zhat_dot_topo))*y_hat_topo;
    
        dA_dt = [xhat_dot_topo'; yhat_dot_topo'; zhat_dot_topo'];
        vot_topo = R_topo*vot_ECI + dA_dt*rot_ECI;
    
        X_ot(particle, :) = [rot_topo, vot_topo'];
    end
end


function [X_ECI] = Topo2ECI(X_ot, t_stamp, obs_lat, obs_lon)

    % First step: Obtain X_{eo}^{ECI} 
    elevation = 103.8;
    mu = 1.2150582e-2;

    UTC_vec_orig = [2024	5	3	2	41 15];%15.1261889999956]; % Initial UTC vector at t_stamp = 0
    t_add_dim = t_stamp * (4.342); % Convert the time to add to a dimensional quantity
    UTC_vec = datevec(datetime(UTC_vec_orig) + t_add_dim); % You will need this for calculating r_{eo} and v_{eo}

    delt_add_dim = t_add_dim - 1/86400;
    delt_updatedUTCtime = datetime(UTC_vec_orig) + delt_add_dim;
    delt_updatedUTCvec = datevec(delt_updatedUTCtime);

    reo_dim = lla2eci([obs_lat obs_lon, elevation], UTC_vec);
    delt_reodim = lla2eci([obs_lat obs_lon, elevation], delt_updatedUTCvec);
    %save("./bCS/bCS_" + num2str(id) + ".mat", "reo_dim", "delt_reodim");
    veo_dim = reo_dim - delt_reodim; % Finite difference

    reo_nondim = reo_dim'/(1000*384400); % Conversion to non-dimensional units in the ECI frame
    veo_nondim = veo_dim'*(4.342*86400)/(1000*384400); % Conversion to non-dimensional units in the ECI frame

    z_hat_topo = reo_nondim/norm(reo_nondim);
    x_hat_topo = cross(z_hat_topo, [0, 0, 1]')/norm(cross(z_hat_topo, [0, 0, 1]'));
    y_hat_topo = cross(x_hat_topo, z_hat_topo)/norm(cross(x_hat_topo, z_hat_topo));
    
    A = [x_hat_topo'; y_hat_topo'; z_hat_topo']; % Computing A as DCM for transforming between ECI and topographic reference frame

    dmag_dt = dot(reo_nondim, veo_nondim)/norm(reo_nondim);
    
    zhat_dot_topo = (veo_nondim * norm(reo_nondim) - reo_nondim * dmag_dt)/(norm(reo_nondim))^2;
    xhat_dot_topo = cross(zhat_dot_topo, [0, 0, 1]')/norm(cross(z_hat_topo, [0,0,1]')) - dot(x_hat_topo, cross(zhat_dot_topo, [0, 0, 1]'))*x_hat_topo;
    yhat_dot_topo = (cross(xhat_dot_topo, z_hat_topo) + cross(x_hat_topo, zhat_dot_topo))/norm(cross(x_hat_topo, z_hat_topo)) - dot(y_hat_topo, cross(xhat_dot_topo, z_hat_topo) + cross(x_hat_topo, zhat_dot_topo))*y_hat_topo;

    dA_dt = [xhat_dot_topo'; yhat_dot_topo'; zhat_dot_topo'];
    
    num_particles = size(X_ot, 1);
    X_ECI = zeros(size(X_ot));
    for particle = 1:num_particles
        rot_topo = X_ot(particle, 1:3)'; % First three components of the state vector
        vot_topo = X_ot(particle, 4:6)'; % Last three components of the state vector

        rot_ECI = A^(-1)*rot_topo;
        vot_ECI = A^(-1)*(vot_topo - dA_dt*rot_ECI);

        ret_ECI = reo_nondim + rot_ECI;
        vet_ECI = veo_nondim + vot_ECI;
        X_ECI(particle, :) = [ret_ECI', vet_ECI'];
    end
end


%% PGM Filter Functions

function [w, wGains] = weightUpdate(wc, cluster_points, idx, zk, R, h)
    wGains = zeros(length(wc), 1);
    for i = 1:length(wc)
        cPts = cluster_points(idx == i, :);
        zPoints = zeros(length(cPts(:,1)), length(zk));
        for j = 1:length(cPts(:,1))
            zPoints(j,:) = h(cPts(j,:));
        end
        zPredMean = mean(zPoints,1);
        zPredCov = cov(zPoints) + R;
        wGains(i) = mvnpdf(zk', zPredMean, zPredCov);
    end

    w = wc .* wGains / sum(wc .* wGains);
end


function Xm_cloud = propagate(Xcloud, t_int, interval, obs_lat, obs_lon, normalization_quantities, dynamics)
    Xm_bt = zeros(size(Xcloud));
    if (dynamics == "CR3BP")
        Xbt = Topo2Synodic(Xcloud, t_int, obs_lat, obs_lon);
        dynamics_model = @(t, x) Dynamics.cr3bp_dyn(t, x, normalization_quantities.mu);
    end
    if (dynamics == "2 Body")
        Xbt = CoordFunctions.Topo2ECI(Xcloud, t_int, obs_lat, obs_lon, normalization_quantities);
        dynamics_model = @(t, x) Dynamics.two_body_dyn(t, x, normalization_quantities.mu);
    end
    num_particles = size(Xcloud, 1);
    for particle = 1:num_particles
        % Next, propagate each X_{bt} in your particle cloud by a single time 
        % step and convert back to the topographic frame.
        opts = odeset('Events', @termSat, 'RelTol', 1e-6, 'AbsTol', 1e-8); 
        [~, X] = ode15s(dynamics_model, [0 interval], Xbt(particle, :), opts);
        Xm_bt(particle, :) = X(end,:);
    end
    if (dynamics == "CR3BP")
        Xm_cloud = CoordFunctions.Synodic2Topo(Xm_bt, t_int + interval, obs_lat, obs_lon, normalization_quantities);
    end
    if (dynamics == "2 Body")
        Xm_cloud = CoordFunctions.ECI2Topo(Xm_bt, t_int + interval, obs_lat, obs_lon, normalization_quantities);
    end
end


% Kalman update using particles from each cluster
function [mu_p, P_p] = kalmanUpdate(zk, Xcloud, R, mu_m, P_m, h)
    N = size(Xcloud,1);
    Zcloud = zeros(N,length(zk));

    for i = 1:N
        Zcloud(i,:) = h(Xcloud(i,:));
    end
    mzk_m = mean(Zcloud,1)';

    Pxz = zeros(size(mzk_m, 1), size(P_m, 1));
    Pzz = zeros(size(mzk_m, 1), size(mzk_m, 1));

    for i = 1:N
        dx = Xcloud(i,:)' - mu_m';
        dz = Zcloud(i,:)' - mzk_m;

        Pxz = Pxz + dz*dx';
        Pzz = Pzz + dz*dz';
    end

    Pxz = Pxz/N;
    Pzz = Pzz/N + R;

    K_k = Pxz'/Pzz;
    % mu_p = mu_m' + K_k*(zk - mzk_m);
    % P_p = P_m - K_k*Pzz*K_k';
    
    mu_p = mu_m' + Pxz'*Pzz^(-1)*(zk - mzk_m);
    P_p = P_m - Pxz'*Pzz^(-1)*Pxz;
    
    P_p = (P_p + P_p')/2;

    [V, D] = eig(P_p);
    D = max(D,0);
    P_p = V*D*V';
end


function clipped_cloud = enforceCislunarBounds(Xm_cloud, t_prior, obs_lat, obs_lon, normalization_quantities, low_lim, up_lim, vel_lim)
    clipped_cloud = [];
    reo = CoordFunctions.getObserverPos(t_prior, obs_lat, obs_lon, normalization_quantities);
    reo = reshape(reo, [1 length(reo)]);

    for i = 1:length(Xm_cloud(:,1))
        if(norm(Xm_cloud(i,1:3) + reo)*normalization_quantities.dist2km > low_lim && ...
                norm(Xm_cloud(i,1:3) + reo)*normalization_quantities.dist2km <= up_lim && ...
                norm(Xm_cloud(i,4:6))*normalization_quantities.vel2kms < vel_lim)
            clipped_cloud = [clipped_cloud; Xm_cloud(i,:)];
        end
    end
end

%% Plotting Functions

function ensureDirExists(file_path)
% ensureDirExists - Creates the directory for a given file path if it doesn't exist.
%
% Usage:
%   ensureDirExists('/some/path/to/file.png')
%
% This will create '/some/path/to' if it doesn't already exist.

    dir_path = fileparts(file_path);
    if ~exist(dir_path, 'dir')
        mkdir(dir_path);
    end
end


function plotMetrics(fig_num, x, y_data, cloud_names, colors, save_loc, ob, y_label, title_str, filename)
    f = figure(fig_num);
    f.WindowState = 'maximized';
    hold on;

    % Plot each cloud's data
    num_clouds = length(y_data);
    for cloud = 1:num_clouds
        style = '--';
        lw = 2;
        plot(x, y_data{cloud}, style, 'Color', colors(cloud), 'LineWidth', lw);
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


function plotMetricsPerState(fig_num, x, y_data, normalization_quantities, cloud_names, colors, save_loc, ob, y_label, title_str, filename)
    dist2km = normalization_quantities.dist2km;
    vel2kms = normalization_quantities.vel2kms;
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


function plotStateSpace(cloud, truth, K, cluster_idx, normalization_quantities, colors, plot_title, filename)
    dist2km = normalization_quantities.dist2km;
    vel2kms = normalization_quantities.vel2kms;
    f = figure('visible','off','Position', get(0,'ScreenSize'));
    f.WindowState = 'maximized';

    legend_string = "Truth";

    subplot(2,3,1)
    hold on;
    for k = 1:K
        scatter(dist2km*cloud(cluster_idx == k,1), dist2km*cloud(cluster_idx == k,2), 'filled', ...
            'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    end
    plot(dist2km*truth(1), dist2km*truth(2), 'kx', ...
        'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string)
    title('X-Y');
    xlabel('X (km.)');
    ylabel('Y (km.)');
    legend(legend_string);
    hold off;

    subplot(2,3,2)
    hold on;
    for k = 1:K
        scatter(dist2km*cloud(cluster_idx == k,1), dist2km*cloud(cluster_idx == k,3), 'filled', ...
            'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    end
    plot(dist2km*truth(1), dist2km*truth(3), 'kx', ...
        'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string)
    title('X-Z');
    xlabel('X (km.)');
    ylabel('Z (km.)');
    legend(legend_string);
    hold off;
    
    subplot(2,3,3)
    hold on;
    for k = 1:K
        scatter(dist2km*cloud(cluster_idx == k,2), dist2km*cloud(cluster_idx == k,3), 'filled', ...
            'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    end
    plot(dist2km*truth(2), dist2km*truth(3), 'kx', ...
        'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string)
    title('Y-Z');
    xlabel('Y (km.)');
    ylabel('Z (km.)');
    legend(legend_string);
    hold off;
    
    subplot(2,3,4)
    hold on;
    for k = 1:K
        scatter(vel2kms*cloud(cluster_idx == k,4), vel2kms*cloud(cluster_idx == k,5), 'filled', ...
            'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    end
    plot(vel2kms*truth(4), vel2kms*truth(5), 'kx', ...
        'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string)

    title('Xdot-Ydot');
    xlabel('Xdot (km/s)');
    ylabel('Ydot (km/s)');
    legend(legend_string);
    hold off;
    
    subplot(2,3,5)
    hold on;
    for k = 1:K
        scatter(vel2kms*cloud(cluster_idx == k,4), vel2kms*cloud(cluster_idx == k,6), 'filled', ...
            'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    end
    plot(vel2kms*truth(4), vel2kms*truth(6), 'kx', ...
        'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string)
    title('Xdot-Zdot');
    xlabel('Xdot (km/s)');
    ylabel('Zdot (km/s)');
    legend(legend_string);
    hold off;
    
    subplot(2,3,6)
    hold on;
    for k = 1:K
        scatter(vel2kms*cloud(cluster_idx == k,5), vel2kms*cloud(cluster_idx == k,6), 'filled', ...
            'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    end
    plot(vel2kms*truth(5), vel2kms*truth(6), 'kx', ...
        'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string)
    title('Ydot-Zdot');
    xlabel('Ydot (km/s)');
    ylabel('Zdot (km/s)');
    legend(legend_string);
    hold off;

    %sgt = sprintf('Timestep: %3.4f Hours (Posterior) Ob: %i', [tpr*time2hr, ob]);
    sgtitle(plot_title);
    %sg = sprintf('%s/Observer%i/ECI/Timestep_%i_2B_eci_cloud_%i.png', save_loc, ob, tau, cloud);
    drawnow;
    pause(0.5);
    exportgraphics(f, filename, 'Resolution', 150);
    close(f);
end


function plotStateSpaceCombined(plotting_clouds, plotting_truth, active_cloud_mask, normalization_quantities, colors, cloud_names, plot_title, filename)
    dist2km = normalization_quantities.dist2km;
    vel2kms = normalization_quantities.vel2kms;
    f = figure('visible','off','Position', get(0,'ScreenSize'));
    f.WindowState = 'maximized';
    
    legend_string = [cloud_names, "Truth"];

    subplot(2,3,1)
    hold on; 
    for cloud = active_cloud_mask
        scatter(dist2km*plotting_clouds{cloud}(:,1), dist2km*plotting_clouds{cloud}(:,2), 'filled', ...
        'MarkerFaceColor', colors(cloud), 'MarkerFaceAlpha', 0.3, 'DisplayName', legend_string(cloud));
    end
    plot(dist2km*plotting_truth(1), dist2km*plotting_truth(2), 'kx', ... 
        'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string(end));
    title('X-Y');
    xlabel('X (km.)');
    ylabel('Y (km.)');
    legend(legend_string);
    hold off;
    
    subplot(2,3,2)
    hold on; 
    for cloud = active_cloud_mask
        scatter(dist2km*plotting_clouds{cloud}(:,1), dist2km*plotting_clouds{cloud}(:,3), 'filled', ...
        'MarkerFaceColor', colors(cloud), 'MarkerFaceAlpha', 0.3, 'DisplayName', legend_string(cloud));
    end
    plot(dist2km*plotting_truth(1), dist2km*plotting_truth(3), 'kx', ... 
        'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string(end));
    title('X-Z');
    xlabel('X (km.)');
    ylabel('Z (km.)');
    legend(legend_string);
    hold off;
    
    subplot(2,3,3)
    hold on; 
    for cloud = active_cloud_mask
        scatter(dist2km*plotting_clouds{cloud}(:,2), dist2km*plotting_clouds{cloud}(:,3), 'filled', ...
        'MarkerFaceColor', colors(cloud), 'MarkerFaceAlpha', 0.3, 'DisplayName', legend_string(cloud));
    end
    plot(dist2km*plotting_truth(2), dist2km*plotting_truth(3), 'kx', ... 
        'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string(end));
    title('Y-Z');
    xlabel('Y (km.)');
    ylabel('Z (km.)');
    legend(legend_string);
    hold off;
    
    subplot(2,3,4)
    hold on; 
    for cloud = active_cloud_mask
        scatter(vel2kms*plotting_clouds{cloud}(:,4), vel2kms*plotting_clouds{cloud}(:,5), 'filled', ...
        'MarkerFaceColor', colors(cloud), 'MarkerFaceAlpha', 0.3, 'DisplayName', legend_string(cloud));
    end
    plot(vel2kms*plotting_truth(4), vel2kms*plotting_truth(5), 'kx', ... 
        'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string(end));
    title('Xdot-Ydot');
    xlabel('Xdot (km/s)');
    ylabel('Ydot (km/s)');
    legend(legend_string);
    hold off;
    
    subplot(2,3,5)
    hold on; 
    for cloud = active_cloud_mask
        scatter(vel2kms*plotting_clouds{cloud}(:,4), vel2kms*plotting_clouds{cloud}(:,6), 'filled', ...
        'MarkerFaceColor', colors(cloud), 'MarkerFaceAlpha', 0.3, 'DisplayName', legend_string(cloud));
    end
    plot(vel2kms*plotting_truth(4), vel2kms*plotting_truth(6), 'kx', ... 
        'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string(end));
    title('Xdot-Zdot');
    xlabel('Xdot (km/s)');
    ylabel('Zdot (km/s)');
    legend(legend_string);
    hold off;
    
    subplot(2,3,6)
    hold on; 
    for cloud = active_cloud_mask
        scatter(vel2kms*plotting_clouds{cloud}(:,5), vel2kms*plotting_clouds{cloud}(:,6), 'filled', ...
        'MarkerFaceColor', colors(cloud), 'MarkerFaceAlpha', 0.3, 'DisplayName', legend_string(cloud));
    end
    plot(vel2kms*plotting_truth(5), vel2kms*plotting_truth(6), 'kx', ... 
        'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string(end));
    title('Ydot-Zdot');
    xlabel('Ydot (km/s)');
    ylabel('Zdot (km/s)');
    legend(legend_string);
    hold off;

    %sgt = sprintf('Timestep: %3.4f Hours (Prior) Ob: %i', [tpr*time2hr, ob]);
    sgtitle(plot_title);
    %sg = sprintf('%s/Observer%i/Synodic/Timestep_%i_1B_synodic_combined.png', save_loc, ob, tau);
    drawnow;
    pause(0.5);
    exportgraphics(f, filename, 'Resolution', 150);
    close(f);
end


function plotMsmtSpace(cloud, truth, zt, h, likelihoods, K, cluster_idx, colors, plot_title, filename, msmt_exists)
    f = figure('visible','off','Position', get(0,'ScreenSize'));
    f.WindowState = 'maximized';
    
    legend_string = ["Truth"];
    hold on;
    %scatter_handles = gobjects(k,1);
    for k = 1:K
        pts = cloud(cluster_idx == k, :);
        Zmcloud = zeros(size(pts, 1), size(zt, 1));
        for i = 1:size(Zmcloud, 1)
            Zmcloud(i,:) = h(pts(i,:))';
        end
        scatter(180/pi*Zmcloud(:,1), 180/pi*Zmcloud(:,2), 'filled', ...
        'MarkerFaceColor', colors(k), 'DisplayName', sprintf('k: %i; w: %.3f', [k, likelihoods(k)]));
    end
    Ztruth = h(truth)';
    plot(180/pi*Ztruth(1), 180/pi*Ztruth(2), 'kx', ... 
    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string(1));
    if msmt_exists
        legend_string = [legend_string, "Noisy Truth"];
        plot(180/pi*zt(1), 180/pi*zt(2), 'ko', 'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string(2));
    end
    %zt_handle = plot(180/pi*zt(1), 180/pi*zt(2), 'ko', ... 
    %'MarkerSize', 20', 'LineWidth', 3, 'DisplayName', 'Noisy Truth');
    %legend([scatter_handles; truth_handle; zt_handle], 'Location', 'northeastoutside'); 
    title(plot_title)
    xlabel('Azimuth Angle (deg)')
    ylabel('Elevation Angle (deg)')

    legend('Location', 'northeastoutside');
    %sg = sprintf('%s/Observer%i/Timestep_%i_2C_cloud_%i.png', save_loc, ob, tau, cloud);
    drawnow;
    pause(0.5);
    exportgraphics(f, filename, 'Resolution', 150);
    close(f);
end


function plotMsmtSpaceCombined(plotting_clouds, plotting_truth, zt, h, num_clouds_per_agent, colors, cloud_names, plot_title, filename, msmt_exists)
    f = figure('visible','off','Position', get(0,'ScreenSize'));
    f.WindowState = 'maximized';
    legend_string = [cloud_names, "Truth"];

    hold on;
    for cloud = 1:num_clouds_per_agent
        msmt_cloud = zeros(size(plotting_clouds{cloud}, 1), 2);
        for particle = 1:size(plotting_clouds{cloud}, 1)
            msmt_cloud(particle, :) = h(plotting_clouds{cloud}(particle, :))';
        end
        scatter(180/pi*msmt_cloud(:,1), 180/pi*msmt_cloud(:,2), 'filled', 'MarkerFaceColor', colors(cloud), 'DisplayName', legend_string(cloud));
    end
    
    Ztruth = h(plotting_truth)';
    plot(180/pi*Ztruth(1), 180/pi*Ztruth(2), 'kx', 'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string(end-1));
    if msmt_exists
        legend_string = [legend_string, "Noisy Truth"];
        plot(180/pi*zt(1), 180/pi*zt(2), 'ko', 'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string(end));
    end
    title(plot_title)
    xlabel('Azimuth Angle (deg)')
    ylabel('Elevation Angle (deg)')
    
    legend(legend_string);
    %sg = sprintf('%s/Observer%i/Timestep_%i_1C_combined.png', save_loc, ob, tau);
    drawnow;
    pause(0.5);
    exportgraphics(f, filename, 'Resolution', 150);
    close(f);
end


%% Metric Functions


function [ob_ob_entropy, ob_ob_unnorm_entropy, ob_ob_l1_norm] = crossObEntropy(Xcloud, cluster_by, Kp, num_agents)
    Kp = 10;
    gmm_unnorm = cell(1, num_agents);
    K = cell(1, num_agents);
    for ob = 1:num_agents
        %{
        if(cluster_by == "FullState")
            rc = Xcloud{ob}(:,1:3);
            mean_rc = mean(rc, 1);
            std_rc = std(rc,0,1);
            norm_rc = (rc - mean_rc)./std_rc; % Normalizing the position
            vc = Xcloud{ob}(:,4:6);
            mean_vc = mean(vc, 1);
            std_vc = std(vc,0,1);
            norm_vc = (vc - mean_vc)./std_vc; % Normalizing the velocity
            Xm_norm = [norm_rc, norm_vc];
            %norm_truth = [(Xtruth(1:3)-mean_rc{ob})./std_rc{ob}, (Xtruth(4:end)-mean_vc{ob})./std_vc{ob}];
            %[whitened_cloud, W, mu] = whitenData(Xcloud);
        end
        
        [gmm_norm, Kp] = matlabGMM(Kp, Xm_norm);
       
        % Unnormalize fitgmdist gmm
        means_unnorm = gmm_norm.mu .* [std_rc, std_vc] + [mean_rc, mean_vc];
        covariances_unnorm = zeros(size(Xm_norm, 2), size(Xm_norm, 2), Kp);
        diag_std = diag([std_rc, std_vc]);
        for k = 1:Kp
            covariances_unnorm(:,:, k) = diag_std * gmm_norm.Sigma(:,:,k) * diag_std';
        end
        gmm_unnorm{ob} = gmdistribution(means_unnorm, covariances_unnorm, gmm_norm.ComponentProportion);
        K{ob} = gmm_unnorm{ob}.NumComponents;
        %}
        [idx, Kp, ~] = cluster(Xcloud{ob}, cluster_by, Kp);
        cPoints = cell(Kp,1); covariances_unnorm = zeros(6, 6, Kp); means_unnorm = zeros(Kp, 6);
        w = zeros(Kp,1);
        % Calculate covariances and weights for each cluster
        for k = 1:Kp
            cluster_points = Xcloud{ob}(idx == k, :); % Keep clustering very separate from mean, covariance, weight calculations
            cPoints{k} = cluster_points; cSize = size(cPoints{k});
        
            if(cSize(1) == 1)
                covariances_unnorm(:, :, k) = zeros(6, 6);
            else
                means_unnorm(k, :) = mean(cluster_points, 1);
                covariances_unnorm(:, :, k) = cov(cluster_points); % Cell of GMM covariances 
            end
            w(k) = size(cluster_points, 1) / size(Xcloud{ob}, 1); % Vector of weights
        end
        gmm_unnorm{ob} = gmdistribution(means_unnorm, covariances_unnorm, w);
        K{ob} = gmm_unnorm{ob}.NumComponents;
    end

    likeli = zeros(K{1}, K{2});
    likeli_weight = zeros(K{1}, K{2});
    for i = 1:K{1}
        for j = 1:K{2}
            likeli(i, j) = mvnpdf(gmm_unnorm{2}.mu(j, :), gmm_unnorm{1}.mu(i, :), gmm_unnorm{1}.Sigma(:, :, i) + gmm_unnorm{2}.Sigma(:, :, j));%P_1(:, :, i) + P_2(:, :, j));
            %likeli_weight(i, j) = gmm_unnorm{1}.ComponentProportion(i) * likeli(i, j) * gmm_unnorm{2}.ComponentProportion(j);
        end
    end
    
    P = likeli / sum(likeli(:));
    P_nonzero = P(P > 0);
    ob_ob_entropy = -sum(P_nonzero .* log10(P_nonzero));
    likeli_nonzero = likeli(likeli > 0);
    ob_ob_unnorm_entropy = -sum(likeli_nonzero .* log10(likeli_nonzero));
    %P_weight = likeli_weight / sum(likeli_weight(:));
    %P_weight_nonzero = P_weight(P_weight > 0);
    ob_ob_l1_norm = log10(sum(abs(likeli), 'all'));
end


function [truth_likelihood, best_mode_entropy, cloud_entropy, NEES, RMSE, std_dev, MC_std_dev, weights, MC_consistency, Kp, Lp] = getStateSpaceMetrics(Kp, Xcloud, Xtruth, cluster_by)
    Kp = 10;
    if(cluster_by == "Range")
        msmt_cloud = zeros(length(Xcloud), 1);
        h = @(x) [sqrt(x(1)^2 + x(2)^2 + x(3)^2)]; % Nonlinear measurement model
        for j = 1:length(Xcloud)
            msmt_cloud(j,:) = h(Xcloud(j,:));
        end
        
        mean_msmt = mean(msmt_cloud, 1);
        std_msmt = std(msmt_cloud,0,1);
        
        norm_msmt_rho = (msmt_cloud(:, 1) - mean_msmt(1))./std_msmt(1); % Normalizing the msmts
        
        Xm_norm = [norm_msmt_rho];
    end
    if(cluster_by == "Msmt")
        msmt_cloud = zeros(length(Xcloud), 2);
        h = @(x) [atan2(x(2),x(1)); pi/2 - acos(x(3)/sqrt(x(1)^2 + x(2)^2 + x(3)^2))]; % Nonlinear measurement model
        for j = 1:length(Xcloud)
            msmt_cloud(j,:) = h(Xcloud(j,:));
        end
        
        mean_msmt = mean(msmt_cloud, 1);
        std_msmt = std(msmt_cloud,0,1);
        
        norm_msmt_az = (msmt_cloud(:, 1) - mean_msmt(1))./std_msmt(1); % Normalizing the msmts
        norm_msmt_el = (msmt_cloud(:, 2) - mean_msmt(2))./std_msmt(2);
        
        Xm_norm = [norm_msmt_az, norm_msmt_el];
    end
    if(cluster_by == "FullState")
        
        rc = Xcloud(:,1:3);
        mean_rc = mean(rc, 1);
        std_rc = std(rc,0,1);
        norm_rc = (rc - mean_rc)./std_rc; % Normalizing the position
        vc = Xcloud(:,4:6);
        mean_vc = mean(vc, 1);
        std_vc = std(vc,0,1);
        norm_vc = (vc - mean_vc)./std_vc; % Normalizing the velocity
        Xm_norm = [norm_rc, norm_vc];
        norm_truth = [(Xtruth(1:3)-mean_rc)./std_rc, (Xtruth(4:end)-mean_vc)./std_vc];
        %[whitened_cloud, W, mu] = whitenData(Xcloud);
    end
    if(cluster_by == "Velocity")
        vc = Xcloud(:,4:6);
        mean_vc = mean(vc, 1);
        std_vc = std(vc,0,1);
        norm_vc = (vc - mean_vc)./std_vc; % Normalizing the velocity
        Xm_norm = [norm_vc];
    end

    [idx, Kp, ~] = cluster(Xcloud, cluster_by, Kp);
    cPoints = cell(Kp,1); P = cell(Kp,1);
    w = zeros(Kp,1);
    % Calculate covariances and weights for each cluster
    for k = 1:Kp
        cluster_points = Xcloud(idx == k, :); % Keep clustering very separate from mean, covariance, weight calculations
        cPoints{k} = cluster_points; cSize = size(cPoints{k});
    
        if(cSize(1) == 1)
            P{k} = zeros(length(w));
        else
            P{k} = cov(cluster_points); % Cell of GMM covariances 
        end
        w(k) = size(cluster_points, 1) / size(Xcloud, 1); % Vector of weights
    end
    
    Lp = length(Xcloud(:,1));
    %Kp
    [gmm_norm, Kp] = matlabGMM(Kp, Xm_norm);
    %Kp
    %gmm_norm = fitgmdist(Xm_norm, Kp, 'Start', initialize_fitgmm, 'CovarianceType', 'full', 'RegularizationValue', 1e-6, 'Options', statset('MaxIter',500, 'Display','off'));
    
    % Unnormalize fitgmdist gmm
    means_unnorm = gmm_norm.mu .* [std_rc, std_vc] + [mean_rc, mean_vc];
    covariances_unnorm = zeros(size(Xm_norm, 2), size(Xm_norm, 2), Kp);
    diag_std = diag([std_rc, std_vc]);
    for k = 1:Kp
        covariances_unnorm(:,:, k) = diag_std * gmm_norm.Sigma(:,:,k) * diag_std';
    end
    gmm_unnorm = gmdistribution(means_unnorm, covariances_unnorm, gmm_norm.ComponentProportion);
    
    % Calc various metrics
    component_likelihoods = posterior(gmm_unnorm, Xtruth);
    [~, best_mode] = max(component_likelihoods);
    best_mu = gmm_unnorm.mu(best_mode, :);
    best_cov = gmm_unnorm.Sigma(:, :, best_mode);
    %best_weight = gmm_unnorm.ComponentProportion(best_mode);
    best_samples = mvnrnd(best_mu, best_cov, 10000);

    truth_likelihood = log10(mvnpdf(Xtruth, best_mu, best_cov));

    diff = Xtruth - best_mu;
    NEES = gmm_unnorm.ComponentProportion(best_mode)* (diff * (best_cov \ diff'));  % Mahalanobis distance
    RMSE = sqrt(mean((best_samples - Xtruth).^2, 1));

    std_dev = sqrt(diag(best_cov));
    
    best_particle_likelihood = mvnpdf(best_samples, best_mu, best_cov);
    best_mode_entropy = -mean(log10(best_particle_likelihood));%log(gmm_unnorm.ComponentProportion(best_mode)*det(best_cov));
    %ent_det_cov = 0;
    for k = 1:Kp
        %ent_det_cov = ent_det_cov + gmm_unnorm.ComponentProportion(k)*det(gmm_unnorm.Sigma(:, :, k));
    end
    particle_likelihood = pdf(gmm_unnorm, Xcloud);
    cloud_entropy = -mean(log10(particle_likelihood + 1e-300));

    weights = [min(gmm_unnorm.ComponentProportion), mean(gmm_unnorm.ComponentProportion), max(gmm_unnorm.ComponentProportion)];

    MC_consistency = consistencyMetric(gmm_unnorm.ComponentProportion', best_mode, Kp);
    MC_std_dev = 0;
    for k = 1:Kp
        MC_std_dev = MC_std_dev + det(2*gmm_unnorm.Sigma(:, :, k));
    end
end

function consistency = consistencyMetric(w, truth_gmm_idx, num_modes)
    indicator = zeros(num_modes, 1);
    indicator(truth_gmm_idx) = 1;
    w_comp = 1 - w;
    surprise = (indicator - w)' * (indicator - w);
    expected_surprise = sum(w .* w_comp);
    var_expected_surprise = sum(w .* w_comp .* (w_comp.^3 + w.^3)) - expected_surprise^2;

    [wj, wk] = meshgrid(w, w);
    weight_term = wj .* wk .* (wj + wk - 3 .* wj .* wk);
    mask = ~eye(num_modes);
    var_expected_surprise = var_expected_surprise + sum(weight_term(mask));

    consistency = (surprise - expected_surprise) / sqrt(var_expected_surprise);
end


function [truth_likelihood, best_mode_entropy, cloud_entropy] = getMsmtSpaceMetrics(Kp, Xcloud, Xtruth, h)
    Kp = 10;
    msmt_cloud = zeros(length(Xcloud), 2);
    for j = 1:length(Xcloud)
        msmt_cloud(j,:) = h(Xcloud(j,:));
    end
    Xmsmt = h(Xtruth);
    mean_msmt = mean(msmt_cloud, 1);
    std_msmt = std(msmt_cloud,0,1);
    norm_msmt_az = (msmt_cloud(:, 1) - mean_msmt(1))./std_msmt(1); % Normalizing the msmts
    norm_msmt_el = (msmt_cloud(:, 2) - mean_msmt(2))./std_msmt(2);
    Xm_norm = [norm_msmt_az, norm_msmt_el];
    
    [gmm_norm, Kp] = matlabGMM(Kp, Xm_norm);

    means_unnorm = gmm_norm.mu .* std_msmt + mean_msmt;
    covariances_unnorm = zeros(size(Xm_norm, 2), size(Xm_norm, 2), Kp);
    diag_std = diag(std_msmt);
    for k = 1:Kp
        covariances_unnorm(:,:, k) = diag_std * gmm_norm.Sigma(:,:,k) * diag_std';
    end
    gmm_unnorm = gmdistribution(means_unnorm, covariances_unnorm, gmm_norm.ComponentProportion);
    
    component_likelihoods = posterior(gmm_unnorm, Xmsmt');
    [~, best_mode] = max(component_likelihoods);
    best_mu = gmm_unnorm.mu(best_mode, :);
    best_cov = gmm_unnorm.Sigma(:, :, best_mode);
    %best_weight = gmm_unnorm.ComponentProportion(best_mode);
    best_samples = mvnrnd(best_mu, best_cov, 10000);
    
    best_particle_likelihood = mvnpdf(best_samples, best_mu, best_cov);
    best_mode_entropy = -mean(log10(best_particle_likelihood));
    %best_ent_det_cov = log(gmm_unnorm.ComponentProportion(best_mode)*det(best_cov));
    %ent_det_cov = 0;
    %for k = 1:Kp
    %    ent_det_cov = ent_det_cov + gmm_unnorm.ComponentProportion(k)*det(gmm_unnorm.Sigma(:, :, k));
    %end
    particle_likelihood = pdf(gmm_unnorm, msmt_cloud);
    cloud_entropy = -mean(log10(particle_likelihood + 1e-300));
    %ent_det_cov = log(ent_det_cov);

    truth_likelihood = log10(mvnpdf(Xmsmt', best_mu, best_cov));
end


function ent = getDiagCov(Xcloud)
    P = cov(Xcloud);
    ent = diag(P);
end

%% GMM Functions

function [idx, K, C] = cluster(data, cluster_by, K)
    if(cluster_by == "Range")
        msmt_cloud = zeros(length(data), 1);
        h = @(x) [sqrt(x(1)^2 + x(2)^2 + x(3)^2)]; % Nonlinear measurement model
        for j = 1:length(data)
            msmt_cloud(j,:) = h(data(j,:));
        end
        
        mean_msmt = mean(msmt_cloud, 1);
        std_msmt = std(msmt_cloud,0,1);
        
        norm_msmt_rho = (msmt_cloud(:, 1) - mean_msmt(1))./std_msmt(1); % Normalizing the msmts
        
        Xm_norm = [norm_msmt_rho];
    end
    if(cluster_by == "Msmt")
        msmt_cloud = zeros(length(data), 2);
        h = @(x) [atan2(x(2),x(1)); pi/2 - acos(x(3)/sqrt(x(1)^2 + x(2)^2 + x(3)^2))]; % Nonlinear measurement model
        for j = 1:length(data)
            msmt_cloud(j,:) = h(data(j,:));
        end
        
        mean_msmt = mean(msmt_cloud, 1);
        std_msmt = std(msmt_cloud,0,1);
        
        norm_msmt_az = (msmt_cloud(:, 1) - mean_msmt(1))./std_msmt(1); % Normalizing the msmts
        norm_msmt_el = (msmt_cloud(:, 2) - mean_msmt(2))./std_msmt(2);
        
        Xm_norm = [norm_msmt_az, norm_msmt_el];
    end
    if cluster_by == "FullState"
        rc = data(:,1:3);
        mean_rc = mean(rc, 1);
        std_rc = std(rc,0,1);
        norm_rc = (rc - mean_rc)./std_rc; % Normalizing the position

        vc = data(:,4:6);
        mean_vc = mean(vc, 1);
        std_vc = std(vc,0,1);
        norm_vc = (vc - mean_vc)./std_vc; % Normalizing the velocity
        Xm_norm = [norm_rc, norm_vc];
    end
    if cluster_by == "Velocity"
        vc = data(:,4:6);
        mean_vc = mean(vc, 1);
        std_vc = std(vc,0,1);
        norm_vc = (vc - mean_vc)./std_vc; % Normalizing the velocity
        Xm_norm = [norm_vc];
    end
    % Cluster using K-means clustering algorithm
    [idx, C] = kmeans(Xm_norm, K, 'MaxIter', 150, 'Replicates', 2);
    num_times_clustered = 1;
    while any(histcounts(idx) <= 10) % Ensure at least 6 points in each cluster
        if num_times_clustered >= 3
            K = K - 1; 
        end
        [idx, C] = kmeans(Xm_norm, K, 'MaxIter', 150, 'Replicates', 2);
        num_times_clustered = num_times_clustered + 1;
    end
end


function [cPoints, mu_c, P_c, wm, wp] = calcGMMStatistics(Xcloud, idx, num_clouds_per_agent, K, Kmax)
    cPoints = cell(1, num_clouds_per_agent, Kmax);
    mu_c = cell(1, num_clouds_per_agent, Kmax);
    P_c = cell(1, num_clouds_per_agent, Kmax);
    wm = cell(1, num_clouds_per_agent);
    wp = cell(1, num_clouds_per_agent);
    % Calculate covariances and weights for each cluster
    for cloud = 1:num_clouds_per_agent
        %if (truth_contained(ob, cloud) == 1)
        wm_temp = zeros(K{1, cloud}, 1);
        wp{1, cloud} = wm_temp;
        for k = 1:K{1, cloud}
            cluster_points = Xcloud{1, cloud}(idx{1, cloud} == k, :); 
            cPoints{1, cloud, k} = cluster_points; 
            mu_c{1, cloud, k} = mean(cluster_points, 1); % Cell of GMM means 
            if (length(cluster_points(:,1)) == 1)
                P_c{1, cloud, k} = zeros(length(mu_c{1, cloud, k}));
            else
                P_c{1, cloud, k} = cov(cluster_points); % Cell of GMM covariances
            end
            wm_temp(k) = size(cluster_points, 1) / size(Xcloud{1, cloud}, 1); % Vector of (prior) weights
        end
        wm{1, cloud} = wm_temp;
        %end
    end
end


function [optimal_gmm, optimal_K] = matlabGMM(Kp, Xcloud)
    delta = 4;  % how far to search on either side of Kp
    K_range = max(1, Kp - delta) : (Kp + delta);  % ensure at least 1 component

    num_K = numel(K_range);
    BIC = zeros(1,num_K);
    AIC = zeros(1,num_K);
    GMModels = cell(1,num_K);
    for k = 1:num_K
        K = K_range(k);
        try
            GMModels{k} = fitgmdist(Xcloud, K, ...
            'RegularizationValue', 1e-6, ...
            'Options', statset('MaxIter',500, 'Display','off'), 'Replicates', 1);
            BIC(k) = GMModels{k}.BIC;
            AIC(k) = GMModels{k}.AIC;
        catch ME
            warning('GMM fitting failed for K = %d: %s', K, ME.message);
            BIC(k) = Inf;  % mark as invalid
        end
    end
    [~, optimal_K_idx] = min(BIC);
    optimal_gmm = GMModels{optimal_K_idx};
    optimal_K = K_range(optimal_K_idx);
end


%% Fusion Functions

function [binned_cloud, edges] = binCloud(cloud, edges_in)
    num_dims = 6;
    N = [20, 20, 20, 20, 20, 20];
    num_points = size(cloud, 1);

    if nargin < 2 || isempty(edges_in)
        edges = cell(1, num_dims);
        for i = 1:num_dims
            edges{i} = linspace(min(cloud(:, i)), max(cloud(:, i)), N(i)+1);
        end
    else
        edges = edges_in;
    end

    idxs = zeros(num_points, num_dims);
    valid = true(num_points, 1);

    for i = 1:num_dims
        idxs(:, i) = discretize(cloud(:, i), edges{i});
        valid = valid & ~isnan(idxs(:, i));
    end

    subs = idxs(valid, :);
    counts = accumarray(subs, 1, N);
    binned_cloud = counts / sum(counts(:));
end


function displayBins(binned_cloud, edges, normalization_quantities, save_loc, specific_plot_info)
    num_dims = 6;
    fprintf('Diagnostics: Binning Sum = %.6f\n', sum(binned_cloud(:)));

    % Marginal PDFs
    pdf_xy    = squeeze(sum(sum(sum(sum(binned_cloud, 3), 4), 5), 6));
    pdf_xz    = squeeze(sum(sum(sum(sum(binned_cloud, 2), 4), 5), 6));
    pdf_yz    = squeeze(sum(sum(sum(sum(binned_cloud, 1), 4), 5), 6));
    pdf_vxvy  = squeeze(sum(sum(sum(sum(binned_cloud, 1), 2), 3), 6));
    pdf_vxvz  = squeeze(sum(sum(sum(sum(binned_cloud, 1), 2), 3), 5));
    pdf_vyvz  = squeeze(sum(sum(sum(sum(binned_cloud, 1), 2), 3), 4));
    
    % Bin Centers
    centers = cell(1, num_dims);
    for i = 1:num_dims
        centers{i} = (edges{i}(1:end-1) + edges{i}(2:end)) / 2;
    end
    
    % Conversions
    for i = 1:3
        centers{i} = centers{i} * normalization_quantities.dist2km;
    end
    for i = 4:6
        centers{i} = centers{i} * normalization_quantities.vel2kms;
    end
    
    coords = {
        centers{1}, centers{2};  % x vs y
        centers{1}, centers{3};  % x vs z
        centers{2}, centers{3};  % y vs z
        centers{4}, centers{5};  % vx vs vy
        centers{4}, centers{6};  % vx vs vz
        centers{5}, centers{6};  % vy vs vz
    };
    
    projections = {pdf_xy, pdf_xz, pdf_yz, pdf_vxvy, pdf_vxvz, pdf_vyvz};
    
    titles = {'x vs y', 'x vs z', 'y vs z', 'v_x vs v_y', 'v_x vs v_z', 'v_y vs v_z'};
    xlabels = {'x [km]', 'x [km]', 'y [km]', 'v_x [km/s]', 'v_x [km/s]', 'v_y [km/s]'};
    ylabels = {'y [km]', 'z [km]', 'z [km]', 'v_y [km/s]', 'v_z [km/s]', 'v_z [km/s]'};
    
    % Plot Figs
    f = figure('Units', 'normalized');
    square_size = 0.24;
    h_spacing = 0.05;
    v_spacing = 0.1;
    left_start = 0.1;
    top_start = 0.5;
    ax = gobjects(1,6);
    
    for i = 1:6
        row = floor((i-1)/3);
        col = mod(i-1,3);
        xpos = left_start + col * (square_size + h_spacing);
        ypos = top_start - row * (square_size + v_spacing);
    
        ax(i) = axes('Position', [xpos, ypos, square_size, square_size]);
        imagesc(coords{i,1}, coords{i,2}, projections{i}');
        axis xy;
        axis square;
        title(titles{i}, 'FontSize', 10);
        xlabel(xlabels{i}, 'FontSize', 9);
        ylabel(ylabels{i}, 'FontSize', 9);
    end
    
    cb = colorbar(ax(end));
    cb.Position = [0.92 0.3 0.02 0.4];
    cb.Label.String = 'Probability Density';
    sgtitle('Binned Cloud');
    
    sg = sprintf(save_loc + "/binned_cloud_" + specific_plot_info + '.png');
    saveas(f, sg, 'png');
    close(f);
end


function [p3_resampled, p3_tallest_peaks, weight_update_p3_1, weight_update_p3_2, bin_axes]  = fusionMethods(p1, p2, cluster_by, Kmax, num_agents, num_particles, display_diagnostics, save_loc, normalization_quantities)
    bin_axes = cell(1, 3);
    p3 = vertcat(p1, p2);
    %{
    [p3_simple, bin_axes{1}] = binCloud(p3);
    displayBins(p3_simple, bin_axes{1}, dist2km, vel2kms, save_loc, "simple_" + num2str(fusion_idx))

    [binned_p1, ~] = binCloud(p1, bin_axes{1});
    [binned_p2, ~] = binCloud(p2, bin_axes{1});
    displayBins(binned_p1, bin_axes{1}, dist2km, vel2kms, save_loc, "1_" + num2str(fusion_idx))
    displayBins(binned_p2, bin_axes{1}, dist2km, vel2kms, save_loc, "2_" + num2str(fusion_idx))
    %}
    % Tallest Peaks
    %binned_max_pdf = binned_p1 + binned_p2;
    %[sorted_max_pdf, ~] = sort(binned_max_pdf(:), 'descend');
    %cumulative_sum = cumsum(sorted_max_pdf);
    %threshold_index = find(cumulative_sum >= 1, 1);
    %threshold = sorted_max_pdf(threshold_index);
    %p3_tallest_peaks = binned_max_pdf;
    %p3_tallest_peaks(p3_tallest_peaks < threshold) = 0;
    %displayBins(p3_tallest_peaks, bin_axes{1}, dist2km, vel2kms, save_loc, "tallest_" + num2str(fusion_idx))
    p3_tallest_peaks = 0;

    %[mu_p, P_p] = kalmanUpdate(zk, Xcloud, R, mu_m, P_m, h);
    [weight_update_p3_1, bin_axes{2}] = postWeights({p1, p2}, cluster_by, Kmax, num_agents, num_particles, display_diagnostics, save_loc, normalization_quantities, 1);
    [weight_update_p3_2, bin_axes{3}] = postWeights({p2, p1}, cluster_by, Kmax, num_agents, num_particles, display_diagnostics, save_loc, normalization_quantities, 2);
    
    [p3_idx, p3_K, ~] = cluster(p3, cluster_by, Kmax);
    [~, p3_mu, p3_P, p3_w, ~] = calcGMMStatistics({p3}, {p3_idx}, 1, {p3_K}, Kmax);
    p3_resampled = zeros(size(p1, 1), 6);
    for i = 1:size(p3_resampled, 1)
        [p3_resampled(i,:), ~] = drawFrom(p3_w{1}, p3_mu, p3_P); 
    end

    %if(display_diagnostics == true)
    %    disp("Diagnostics: Fusion Sums: " + num2str(sum(p3_simple, [],  "all"))\ ...
    %        + ", " + num2str(sum(p3_simple, [],  "all"))\ ...
    %        + ", " + num2str(sum(weight_update_p3_1, [],  "all"))\ ...
    %        + ", " + num2str(sum(weight_update_p3_2, [],  "all")))
    %end
end


function [new_particles, bin_axes] = postWeights(data, cluster_by, Kmax, num_agents, num_particles, display_diagnostics, save_loc, normalization_quantities, ob)
    K = {Kmax, Kmax}; % Number of clusters (changeable)
    idx = cell(1, num_agents);
    cPoints = cell(Kmax, num_agents);
    
    mu_c = cell(Kmax, num_agents);
    P_c = cell(Kmax, num_agents);
    wm = cell(1, num_agents);
    
    for ob = 1:num_agents
        [idx{ob}, K{ob}, ~] = cluster(data{ob}, cluster_by, K{ob});
        wm{ob} = zeros(K{ob}, 1);
        % Calculate covariances and weights for each cluster
        for k = 1:K{ob}
            cluster_points = data{ob}(idx{ob} == k, :); % Keep clustering very separate from mean, covariance, weight calculations
            cPoints{k, ob} = cluster_points; cSize = size(cPoints{k, ob});
            mu_c{k, ob} = mean(cluster_points, 1); % Cell of GMM means 
        
            if(cSize(1) == 1)
                P_c{k, ob} = zeros(length(K{ob}));
            else
                P_c{k, ob} = cov(cluster_points); % Cell of GMM covariances 
            end
            wm{ob}(k) = size(cluster_points, 1) / size(data{ob}, 1); % Vector of weights
        end
    end

    post_weights = zeros(K{1}, K{2});
    likeli = zeros(K{1}, K{2});
    total_weight = 0;
    for i = 1:K{1}
        for j = 1:K{2}
            likeli(i, j) = mvnpdf(mu_c{j, 2}, mu_c{i, 1}, P_c{i, 1} + P_c{j, 2});%P_1(:, :, i) + P_2(:, :, j));
        end
    end
    for i = 1:K{1}
        for j = 1:K{2}
            total_weight = total_weight + wm{1}(i)*likeli(i, j)*wm{2}(j);%w_1(i)*likeli(i, j)*w_2(j);
        end
    end
    for i = 1:K{1}
        for j = 1:K{2}
            post_weights(i, j) = wm{1}(i)*likeli(i, j)*wm{2}(j)/total_weight;
        end
    end

    % Resample
    new_particles = [];
    %new_pos = [];
    for i = 1:K{1}
        for j = 1:K{2}
            [post_mu, post_P] = KalmanFilter(mu_c{i, 1}', P_c{i, 1}, mu_c{j, 2}', P_c{j, 2});
            new_particles = [new_particles; mvnrnd(post_mu, post_P, round(num_particles*post_weights(i, j)))];
            %new_pos = [new_pos; mvnrnd(post_mu(1:3), post_P(1:3, 1:3), round(num_particles*post_weights(i, j)))];
        end
    end
    [p3, bin_axes] = binCloud(new_particles);
    %displayBins(p3, bin_axes, dist2km, vel2kms, save_loc, "Ishan" + num2str(ob) + "_" + num2str(fusion_idx))
end


function [post_mu, post_P] = KalmanFilter(mu_1, P_1, mu_2, P_2)
    post_mu = mu_1 + P_1 * inv(P_1 + P_2) * (mu_2 - mu_1);
    post_P = P_1 - P_1 * inv(P_1 + P_2) * P_1;
    post_P = (post_P + post_P') / 2;
end
