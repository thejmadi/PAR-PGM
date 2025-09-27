% Start the clock
clear all;
tic
for mc_idx = 2:2
    clear all; close all;
    rng(mc_idx, "twister")
    save_loc = "D:/PythonProjects/EDP/PGM/ParticleFusionTest/10_3_25_meeting/DilshadMetrics/Test1/MC_" + num2str(mc_idx);
    load_loc = "D:/PythonProjects/EDP/PGM/TestOrbits/2Obs/NRHO/TestOrbit2/Agent";
    
    cluster_by = "FullState";
    Kn = 8; % Number of clusters (original)
    K = repmat({Kn}, 2, 6); % Number of clusters (changeable)
    Kmax = 8; % Maximum number of clusters (Kmax = 1 for EnKF)
    colors = ["Red", "Blue", "Green", "Magenta", "Cyan", "Yellow", "Black", "#500000", "#bf5700", "#00274c"];
    contourCols = lines(Kmax);
    
    disp_diagnostics = true;
    plot_IOD = true;
    plot_indv_clouds = false;
    plot_comb_clouds = false;
    save_MC_metrics = true;
    
    % Number of particles to use in IOD
    L = 10000;
    % Num of observers
    num_agents = 2;
    num_clouds_per_agent = 1;
    num_clouds = num_agents * num_clouds_per_agent;
    num_new_clouds_per_agent = 2;
    
    % combine == 0 : don't fuse; 1 : fuse; =/=0,1 : done fusing 
    fusion_idx = 1;
    fuse_orig_clouds = [false, true]; % First entry reserved for IOD fusion
    time_of_fusion = [0, 50]; % hrs
    cloud_names = ["Original"];
    fusion_types = ["Original", "Simple", "Weight Update"];
    
    % College Station
    obs_lat{1} = 30.618963;%repmat({30.618963}, 1, 1);
    obs_lon{1} = -96.339214;%repmat({-96.339214}, 1, 1);
    % Buenos Aires
    obs_lat{2} = -34.612979;
    obs_lon{2} = -58.453656;
    
    % Create Folders
    for ob = 1:num_agents
        ensureDirExists(sprintf('%s/Observer%i/Topo/Combined/', save_loc, ob));
        ensureDirExists(sprintf('%s/Observer%i/Synodic/Combined/', save_loc, ob));
        ensureDirExists(sprintf('%s/Observer%i/ECI/Combined/', save_loc, ob));
    end
    %ensureDirExists('%s/MC_%i/', save_loc, MC_idx);
    
    % Load noiseless observation data and other important .mat files
    partial_ts = cell(1, num_agents);
    full_ts = cell(1, num_agents);
    full_vts = cell(1, num_agents);
    for ob = 1:num_agents
        partial_ts{ob} = load(load_loc + num2str(ob) + "/partial_ts.mat").partial_ts; % Noiseless observation data
        full_ts{ob} = load(load_loc + num2str(ob) + "/full_ts.mat").full_ts; % Position truth (topocentric frame)
        full_vts{ob} = load(load_loc + num2str(ob) + "/full_vts.mat").full_vts; % Velocity truth (topocentric frame)
    end
    truth_contained = ones(num_agents, num_clouds_per_agent);
    contained_failure_times = zeros(num_agents, num_clouds_per_agent);
    %partial_ts = csvread("D:/PythonProjects/EDP/PGM_Git/PAR-PGM/partial_ts.csv");
    %full_ts = csvread("D:/PythonProjects/EDP/PGM_Git/PAR-PGM/full_ts.csv");
    %full_vts = csvread("D:/PythonProjects/EDP/PGM_Git/PAR-PGM/full_vts.csv");
    draw_from_idx = 0;
    %bCS_idx = 0;
    % Add observation noise to the observation data as follows:
    % Range - 5% of the current (i.e. noiseless) range
    % Azimuth - 1.5 arc sec (credit: Dr. Utkarsh Ranjan Mishra)
    % Elevation - 1.5 arc sec (credit: Dr. Utkarsh Ranjan Mishra)
    % Note: All above quantities are drawn in a zero-mean Gaussian fashion.
    
    theta_f = 1.5; % Arc-seconds of error covariance
    R_f{1} = 0.25; % Range percentage error covariance
    R_f{2} = 0.25;
    
    % Non-dimensionalization
    dist2km = 384400; % Kilometers per non-dimensionalized distance
    time2hr = 4.342*24; % Hours per non-dimensionalized time
    vel2kms = dist2km/(time2hr*60*60); % Kms per non-dimensionalized velocity
    
    % Limits of the cislunar domain
    low_lim = (2*42164); % Two times the GEO Distance
    up_lim = 550000;
    vel_lim = 42; % Escape velocity of the solar system
    
    noised_obs = cell(1, num_agents);
    interval = cell(1, num_agents);
    cTimes = cell(1, num_agents);
    cVal = cell(1, num_agents);
    pf = cell(1, num_agents);
    hdR = cell(1, num_agents);
    hdR_p = cell(1, num_agents);
    partial_vts = cell(1, num_agents);
    partial_rts = cell(1, num_agents);
    Xot_truth = cell(1, num_agents);
    t_truth = cell(1, num_agents);
    idx_prop = cell(1, num_agents);
    c_prop = cell(1, num_agents);
    Xprop_truth = cell(1, num_agents);
    for ob = 1:num_agents
        noised_obs{ob} = partial_ts{ob};
    
    
        R_t = zeros(3*length(noised_obs{ob}(:,1)),1); % We shall diagonalize this later
        mu_t = zeros(3*length(noised_obs{ob}(:,1)),1);
        
        for i = 1:length(partial_ts{ob}(:,1))
            mu_t(3*(i-1)+1:3*(i-1)+3, 1) = [partial_ts{ob}(i,2); partial_ts{ob}(i,3); partial_ts{ob}(i,4)];
            R_t(3*(i-1)+1:3*(i-1)+3, 1) = [(R_f{ob}*partial_ts{ob}(i,2))^2; (theta_f*4.84814e-6)^2; (theta_f*4.84814e-6)^2];
        end
    
        R_t = diag(R_t);
        data_vec = mvnrnd(mu_t, R_t)';
        
        for i = 1:length(noised_obs{ob}(:,1))
            noised_obs{ob}(i,2:4) = data_vec(3*(i-1)+1:3*(i-1)+3,1);
            %noised_obs{ob}(i,2) = unifrnd(low_lim, up_lim)/dist2km;
        % while(noised_obs(i,2) < low_lim/dist2km || noised_obs(i,2) > up_lim/dist2km)
        %     noised_obs(i,2) = mvnrnd(partial_ts(i,2), (R_f*partial_ts(i,2))^2);
        % end
        end
    
        interval{ob} = noised_obs{ob}(2,1) - partial_ts{ob}(1,1);
    
        % Extract important time points from the noised_obs variable
        i = 2;
        while (i <= length(noised_obs{ob}(:,1)))
            if (noised_obs{ob}(i,1) - noised_obs{ob}(i-1,1) > (interval{ob}+1e-11))
                cTimes{ob} = [cTimes{ob}, noised_obs{ob}(i-1,1), noised_obs{ob}(i,1)];
            end
            i = i + 1;
        end
        
        largest_diff = noised_obs{ob}(2,1) - noised_obs{ob}(1,1);
        for j = 2:length(noised_obs{ob}(:,1))
            if (noised_obs{ob}(j,1) - noised_obs{ob}(j-1,1) > largest_diff+1e-11)
                largest_diff = noised_obs{ob}(j,1) - noised_obs{ob}(j-1,1);
                idx_cVal = j-1;
            end
        end
        cVal{ob} = noised_obs{ob}(idx_cVal,1);
    
        % Extract the first continuous observation track
        hdo = []; % Matrix for a half day observation
        hdo(1,:) = noised_obs{ob}(1,:);
        i = 1;
        while(noised_obs{ob}(i+1,1) - noised_obs{ob}(i,1) < full_ts{ob}(2,1) + 1e-15) % Add small epsilon due to roundoff error
            hdo(i,:) = noised_obs{ob}(i+1,:);
            i = i + 1;
        end
        
        % Convert observation data into [X, Y, Z] data in the topographic frame.
        
        hdR{ob} = zeros(length(hdo(:,1)),4); % Convert quantities of hdo to [X, Y, Z]
        hdR{ob}(:,1) = hdo(:,1); % Timestamp stays the same
        hdR{ob}(:,2) = hdo(:,2) .* cos(hdo(:,4)) .* cos(hdo(:,3)); % Conversion to X
        hdR{ob}(:,3) = hdo(:,2) .* cos(hdo(:,4)) .* sin(hdo(:,3)); % Conversion to Y
        hdR{ob}(:,4) = hdo(:,2) .* sin(hdo(:,4)); % Conversion to Z
        
        pf{ob} = 0.5; % A factor between 0 to 1 describing the length of the day to interpolate [x, y]
        nfit = 4; % Order of polynomial fitting (typically around 3-4)
        in_len = round(pf{ob} * length(hdR{ob}(:,1))); % Length of interpolation interval
    
        % Modify interpolation interval length such that you are piecing through
        % enough points.
        if (in_len < nfit + 1)
            in_len = nfit + 1;
            pf{ob} = in_len/length(hdR{ob}(:,1)); % Modify pf such that it meets minimum condition
        end
    
        hdR_p{ob} = hdR{ob}(1:in_len,:); % Matrix for a partial half-day observation
        
        % Fit polynomials for X, Y, and Z (Cubic for X, Quadratic for X and Y)
        coeffs_X = polyfit(hdR_p{ob}(:,1), hdR_p{ob}(:,2), nfit);
        coeffs_Y = polyfit(hdR_p{ob}(:,1), hdR_p{ob}(:,3), nfit);
        coeffs_Z = polyfit(hdR_p{ob}(:,1), hdR_p{ob}(:,4), nfit);
        
        % Predicted values for X, Y, and Z given the polynomial fits
        X_fit = polyval(coeffs_X, hdR_p{ob}(:,1));
        Y_fit = polyval(coeffs_Y, hdR_p{ob}(:,1));
        Z_fit = polyval(coeffs_Z, hdR_p{ob}(:,1));
        
        % Now that you have analytically calculated the coefficients of the fitted
        % polynomial, use them to obtain values for X_dot, Y_dot, and Z_dot.
        % 1) Plot the X_dot, Y_dot, and Z_dot values for the time points for the
        % slides. 
        % 2) Find a generic way of obtaining and plotting X_dot, Y_dot, and Z_dot
        % values given some set of [X_coeffs, Y_coeffs, Z_coeffs]. 
        
        coeffs_dX = polyDeriv(coeffs_X);
        coeffs_dY = polyDeriv(coeffs_Y);
        coeffs_dZ = polyDeriv(coeffs_Z);
        
        % Predicted values for Xdot, Ydot, and Zdot given the polynomial fits
        Xdot_fit = polyval(coeffs_dX, hdR_p{ob}(:,1));
        Ydot_fit = polyval(coeffs_dY, hdR_p{ob}(:,1));
        Zdot_fit = polyval(coeffs_dZ, hdR_p{ob}(:,1));
        
        partial_vts{ob} = [];
        partial_rts{ob} = [];
        j = 1;
        i = 1;
        while (j <= length(hdR_p{ob}(:,1)))
            if(hdR{ob}(j,1) == full_vts{ob}(i,1)) % Matching time index
                partial_vts{ob}(j,:) = full_vts{ob}(i,:);
                partial_rts{ob}(j,:) = full_ts{ob}(i,:);
                j = j + 1;
            end
            i = i + 1;
        end
    
        Xot_fitted = [X_fit(end,1); Y_fit(end,1); Z_fit(end,1); Xdot_fit(end,1); Ydot_fit(end,1); Zdot_fit(end,1)];
        Xot_truth{ob} = [partial_rts{ob}(end,2:4), partial_vts{ob}(end,2:4)]';
    
        t_truth{ob} = partial_rts{ob}(end,1);
        [idx_prop{ob}, c_prop{ob}] = find(full_ts{ob} == t_truth{ob});
        Xprop_truth{ob} = [full_ts{ob}(idx_prop{ob}+1,2:4), full_vts{ob}(idx_prop{ob}+1,2:4)]';
    end
    
    Lp = 1*L;
    X0cloud = cell(num_agents, num_clouds_per_agent);
    fig_num = 1;
    for ob = 1:num_agents
        for cloud = 1:num_clouds_per_agent
            obs_X0cloud = zeros(L,6);
            for i = 1:length(obs_X0cloud(:,1))
                obs_X0cloud(i,:) = stateEstCloud(pf{ob}, nfit, theta_f, R_f{ob}, partial_ts{ob}, (partial_ts{ob}(2,1) - partial_ts{ob}(1,1)) + 1e-15, low_lim, up_lim, load_loc + num2str(ob));
            end
            X0cloud{ob, cloud} = enforceCislunarBounds(obs_X0cloud, t_truth{ob}, obs_lat{ob}, obs_lon{ob}, dist2km, vel2kms, low_lim, up_lim, vel_lim);%X0cloud(1:j,:);
       
            figure(fig_num);
            fig_num = fig_num + 1;
            set(gcf, 'units','normalized','outerposition',[0 0 1 1])
            subplot(2,3,1)
            plot(dist2km*X0cloud{ob, cloud}(:,1), dist2km*X0cloud{ob, cloud}(:,2), '.')
            hold on;
            plot(dist2km*Xot_truth{ob}(1), dist2km*Xot_truth{ob}(2), 'kx','MarkerSize', 15, 'LineWidth', 3)
            title('X-Y');
            xlabel('X (km.)');
            ylabel('Y (km.)');
            legend('Estimate','Truth');
            hold off;
            
            subplot(2,3,2)
            plot(dist2km*X0cloud{ob, cloud}(:,1), dist2km*X0cloud{ob, cloud}(:,3), '.')
            hold on;
            plot(dist2km*Xot_truth{ob}(1), dist2km*Xot_truth{ob}(3), 'kx','MarkerSize', 15, 'LineWidth', 3)
            title('X-Z');
            xlabel('X (km.)');
            ylabel('Z (km.)');
            legend('Estimate','Truth');
            hold off;
            
            subplot(2,3,3)
            plot(dist2km*X0cloud{ob, cloud}(:,2), dist2km*X0cloud{ob, cloud}(:,3), '.')
            hold on;
            plot(dist2km*Xot_truth{ob}(2), dist2km*Xot_truth{ob}(3), 'kx','MarkerSize', 15, 'LineWidth', 3)
            title('Y-Z');
            xlabel('Y (km.)');
            ylabel('Z (km.)');
            legend('Estimate','Truth');
            hold off;
            
            subplot(2,3,4)
            plot(vel2kms*X0cloud{ob, cloud}(:,4), vel2kms*X0cloud{ob, cloud}(:,5), '.')
            hold on;
            plot(vel2kms*Xot_truth{ob}(4), vel2kms*Xot_truth{ob}(5), 'kx','MarkerSize', 15, 'LineWidth', 3)
            title('Xdot-Ydot');
            xlabel('Xdot (km/s)');
            ylabel('Ydot (km/s)');
            legend('Estimate','Truth');
            hold off;
            
            subplot(2,3,5)
            plot(vel2kms*X0cloud{ob, cloud}(:,4), vel2kms*X0cloud{ob, cloud}(:,6), '.')
            hold on;
            plot(vel2kms*Xot_truth{ob}(4), vel2kms*Xot_truth{ob}(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
            title('Xdot-Zdot');
            xlabel('Xdot (km/s)');
            ylabel('Zdot (km/s)');
            legend('Estimate','Truth');
            hold off;
            
            subplot(2,3,6)
            plot(vel2kms*X0cloud{ob, cloud}(:,5), vel2kms*X0cloud{ob, cloud}(:,6), '.')
            hold on;
            plot(vel2kms*Xot_truth{ob}(5), vel2kms*Xot_truth{ob}(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
            title('Ydot-Zdot');
            xlabel('Ydot (km/s)');
            ylabel('Zdot (km/s)');
            legend('Estimate','Truth');
            hold off;
            
            sg = sprintf('Timestep: %3.4f Hours Obs: %i', [t_truth{ob}*time2hr, ob]);
            sgtitle(sg);
            %savefig(gcf, "iodCloud_obs_" + num2str(ob) + "_cloud_" + num2str(cloud) + ".fig");
            saveas(gcf, save_loc + "/Observer" + num2str(ob) + "/iodCloud_cloud_" + num2str(cloud) + ".png", 'png');
            % saveas(gcf, './Simulations/Different Orbit Simulations/iodCloud.png', 'png');
        end
    end
    
    %% IOD Fusion
    if(fuse_orig_clouds(fusion_idx) == true)
        disp("Fusing Clouds")
        % Convert clouds to Synodic Frame
        converted_cloud_1 = backConvertSynodic(X0cloud{1, 1}, t_truth{1}, obs_lat{1}, obs_lon{1});
        converted_cloud_2 = backConvertSynodic(X0cloud{2, 1}, t_truth{2}, obs_lat{2}, obs_lon{2});
        
        % Fuse Clouds
        [p3_simple, ~, weight_update_p3_1, weight_update_p3_2, fusion_bin_edges] = fusionMethods(converted_cloud_1, converted_cloud_2, cluster_by, Kmax, num_agents, Lp, disp_diagnostics, save_loc, dist2km, vel2kms, fusion_idx);
        
        % Resample Simple Fusion
        [p3_idx, p3_K, ~] = cluster(p3_simple, cluster_by, Kmax);
        [~, p3_mu, p3_P, p3_w, ~] = calcGMMStatistics({p3_simple}, {p3_idx}, 1, 1, {p3_K}, Kmax);
        p3_resampled = zeros(Lp, length(Xot_truth{ob}));
        for i = 1:Lp
            [p3_resampled(i,:), ~] = drawFrom(p3_w{1}, p3_mu, p3_P); 
        end
    
        % Convert clouds back to Topo Frame
        X0cloud{1, num_clouds_per_agent+1} = convertToTopo(p3_resampled, t_truth{1}, obs_lat{1}, obs_lon{1});
        X0cloud{1, num_clouds_per_agent+2} = convertToTopo(weight_update_p3_1, t_truth{1}, obs_lat{1}, obs_lon{1});
        X0cloud{2, num_clouds_per_agent+1} = convertToTopo(p3_resampled, t_truth{2}, obs_lat{2}, obs_lon{2});
        X0cloud{2, num_clouds_per_agent+2} = convertToTopo(weight_update_p3_2, t_truth{2}, obs_lat{2}, obs_lon{2});
        cloud_names = [cloud_names, sprintf("%s %3.1f", fusion_types(2), t_truth{1}*time2hr), sprintf("%s %3.1f", fusion_types(3), t_truth{1}*time2hr)];
        
        % Add space for metrics of new clouds
        %{
        for ob = 1:num_agents
            for new_cloud = num_clouds_per_agent+1:num_clouds_per_agent + num_new_clouds_per_agent
                ent1{ob, new_cloud} = NaN(size(ent1{ob, 1}));
                ent2_det_cov_orig{ob, new_cloud} = NaN(size(ent2_det_cov_orig{ob, 1}));
                ent2_det_cov{ob, new_cloud} = NaN(size(ent2_det_cov{ob, 1}));
                ent2_diff_ent{ob, new_cloud} = NaN(size(ent2_diff_ent{ob, 1}));
                ent2_discr_ent{ob, new_cloud} = NaN(size(ent2_discr_ent{ob, 1}));
                likelihood_metric_state_space{ob, new_cloud} = NaN(size(likelihood_metric_state_space{ob, 1}));
                likelihood_metric_msmt_space{ob, new_cloud} = NaN(size(likelihood_metric_msmt_space{ob, 1}));
                mahalanobis{ob, new_cloud} = NaN(size(mahalanobis{ob, 1}));
                RMSE{ob, new_cloud} = NaN(size(RMSE{ob, 1}));
                std_dev{ob, new_cloud} = NaN(size(std_dev{ob, 1}));
                weight_metric{ob, new_cloud} = NaN(size(weight_metric{ob, 1}));
                num_cluster{ob, new_cloud} = NaN(size(num_cluster{ob, 1}));
                num_particles{ob, new_cloud} = NaN(size(num_particles{ob, 1}));
            end
        end
        %}
        num_clouds_per_agent = num_clouds_per_agent + num_new_clouds_per_agent;
        %fusion_idx = min(size(fuse_orig_clouds, 2), fusion_idx + 1);
    end
    fuse_orig_clouds(fusion_idx) = false;
    fusion_idx = fusion_idx + 1;
    
    %% IOD Plotting
    if (plot_IOD)
        for ob = 1:num_agents
            for cloud = 1:num_clouds_per_agent
                plotting_cloud = X0cloud{ob, cloud};
                plotting_truth = Xot_truth{ob}';
                idx = ones(size(plotting_cloud, 1), 1);
                plotStateSpace(plotting_cloud, ...
                                plotting_truth, ...
                                K{ob, cloud}, ...
                                idx, ...
                                dist2km, ...
                                vel2kms, ...
                                colors, ...
                                sprintf('IOD: %3.4f Hours Ob: %i', [t_truth{ob}*time2hr, ob]), ...
                                sprintf('%s/Observer%i/Topo/IOD_cloud_%i.png', save_loc, ob, cloud))
            
                plotting_cloud = Topo2ECI(X0cloud{ob, cloud}, t_truth{ob}, obs_lat{ob}, obs_lon{ob});
                plotting_truth = Topo2ECI(Xot_truth{ob}', t_truth{ob}, obs_lat{ob}, obs_lon{ob});
                plotStateSpace(plotting_cloud, ...
                                plotting_truth, ...
                                K{ob, cloud}, ...
                                idx, ...
                                dist2km, ...
                                vel2kms, ...
                                colors, ...
                                sprintf('IOD: %3.4f Hours Ob: %i', [t_truth{ob}*time2hr, ob]), ...
                                sprintf('%s/Observer%i/ECI/IOD_cloud_%i.png', save_loc, ob, cloud))
            
                plotting_cloud = backConvertSynodic(X0cloud{ob, cloud}, t_truth{ob}, obs_lat{ob}, obs_lon{ob});
                plotting_truth = backConvertSynodic(Xot_truth{ob}', t_truth{ob}, obs_lat{ob}, obs_lon{ob});
                plotStateSpace(plotting_cloud, ...
                                plotting_truth, ...
                                K{ob, cloud}, ...
                                idx, ...
                                dist2km, ...
                                vel2kms, ...
                                colors, ...
                                sprintf('IOD: %3.4f Hours Ob: %i', [t_truth{ob}*time2hr, ob]), ...
                                sprintf('%s/Observer%i/Synodic/IOD_cloud_%i.png', save_loc, ob, cloud))
            end
        end
    end
    
    if (plot_IOD)
        for ob = 1:num_agents
            plotting_cloud = cell(num_clouds_per_agent);
            for cloud = 1:num_clouds_per_agent
                plotting_cloud{cloud} = X0cloud{ob, cloud};
            end
            plotting_truth = Xot_truth{ob}';
            plotStateSpaceCombined(plotting_cloud, ...
                                    plotting_truth, ...
                                    num_clouds_per_agent, ...
                                    dist2km, ...
                                    vel2kms, ...
                                    colors, ...
                                    cloud_names, ...
                                    sprintf('IOD: %3.4f Hours Ob: %i', [t_truth{ob}*time2hr, ob]), ...
                                    sprintf('%s/Observer%i/Topo/Combined/IOD_combined.png', save_loc, ob))
            
            plotting_cloud = cell(num_clouds_per_agent);
            for cloud = 1:num_clouds_per_agent
                plotting_cloud{cloud} = Topo2ECI(X0cloud{ob, cloud}, t_truth{ob}, obs_lat{ob}, obs_lon{ob});
            end
            plotting_truth = Topo2ECI(Xot_truth{ob}', t_truth{ob}, obs_lat{ob}, obs_lon{ob});
            plotStateSpaceCombined(plotting_cloud, ...
                                    plotting_truth, ...
                                    num_clouds_per_agent, ...
                                    dist2km, ...
                                    vel2kms, ...
                                    colors, ...
                                    cloud_names, ...
                                    sprintf('IOD: %3.4f Hours Ob: %i', [t_truth{ob}*time2hr, ob]), ...
                                    sprintf('%s/Observer%i/ECI/Combined/IOD_combined.png', save_loc, ob))
        
            plotting_cloud = cell(num_clouds_per_agent);
            for cloud = 1:num_clouds_per_agent
                plotting_cloud{cloud} = backConvertSynodic(X0cloud{ob, cloud}, t_truth{ob}, obs_lat{ob}, obs_lon{ob});
            end
            plotting_truth = backConvertSynodic(Xot_truth{ob}', t_truth{ob}, obs_lat{ob}, obs_lon{ob});
            plotStateSpaceCombined(plotting_cloud, ...
                                    plotting_truth, ...
                                    num_clouds_per_agent, ...
                                    dist2km, ...
                                    vel2kms, ...
                                    colors, ...
                                    cloud_names, ...
                                    sprintf('IOD: %3.4f Hours Ob: %i', [t_truth{ob}*time2hr, ob]), ...
                                    sprintf('%s/Observer%i/Synodic/Combined/IOD_combined.png', save_loc, ob))
        end
    end
    
    %% Start of PGM Filtering
    
    h = @(x) [atan2(x(2),x(1)); pi/2 - acos(x(3)/sqrt(x(1)^2 + x(2)^2 + x(3)^2))]; % Nonlinear measurement model
    t_int = cell(1, num_agents);
    Xm_cloud = X0cloud;
    %Xbt = zeros(L, 6);
    %X_all = cell(length(X0cloud(:,1)), 1);
    %T_all = cell(length(X0cloud(:,1)), 1);
    %Xm_bt = zeros(size(X0cloud));
    for ob = 1:num_agents
        t_int{ob} = hdR_p{ob}(end,1); % Time at which we are obtaining a state cloud
        for cloud = 1:num_clouds_per_agent
            Xm_bt = zeros(size(Xm_cloud{ob, cloud}));
            Xbt = backConvertSynodic(X0cloud{ob, cloud}, t_int{ob}, obs_lat{ob}, obs_lon{ob});
            for particle = 1:length(X0cloud{ob, cloud}(:,1))
                % First, convert from X_{ot} in the topocentric frame to X_{bt} in the
                % synodic frame.
                %Xbt = backConvertSynodic(X0cloud{ob, cloud}(i,:)', t_int{ob}, obs_lat{ob}, obs_lon{ob});
                % Next, propagate each X_{bt} in your particle cloud by a single time 
                % step and convert back to the topographic frame.
                 % Call ode45()
                opts = odeset('Events', @termSat);
                [t,X] = ode45(@cr3bp_dyn, [0 interval{ob}], Xbt(particle, :), opts); % Assumes termination event (i.e. target enters LEO)
                Xm_bt(particle, :) = X(end,:);
                % Xm_cloud(i,:) = procNoise(Xm_cloud(i,:)); % Adds process noise
            end
            Xm_cloud_tmp = convertToTopo(Xm_bt, t_int{ob} + interval{ob}, obs_lat{ob}, obs_lon{ob});
            Xm_cloud{ob, cloud} = enforceCislunarBounds(Xm_cloud_tmp, t_int{ob} + interval{ob}, obs_lat{ob}, obs_lon{ob}, dist2km, vel2kms, low_lim, up_lim, vel_lim);
        end
    end
    % Initialize variables
    idx = cell(num_agents, num_clouds_per_agent);
    
    for ob = 1:num_agents
        for cloud = 1:num_clouds_per_agent
            [idx{ob, cloud}, K{ob, cloud}, ~] = cluster(Xm_cloud{ob, cloud}, cluster_by, K{ob, cloud});
        end
    end
    
    [cPoints, mu_c, P_c, wm, ~] = calcGMMStatistics(Xm_cloud, idx, num_agents, num_clouds_per_agent, K, Kmax);
    
    % Plot the results
    warning('off', 'MATLAB:legend:IgnoringExtraEntries');
    
    % Plot planar projections
    %{
    figure(2)
    subplot(2,1,1)
    hold on;
    for k = 1:K
        clusterPoints = Xm_cloud(idx == k, :);
        scatter3(dist2km*clusterPoints(:,1), dist2km*clusterPoints(:,2), dist2km*clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
        hold on;
    end
    plot3(dist2km*Xprop_truth(1), dist2km*Xprop_truth(2), dist2km*Xprop_truth(3), 'x','MarkerSize', 15, 'LineWidth', 3)
    title('k-Means Clustered Distribution (Position)');
    xlabel('X (km.)');
    ylabel('Y (km.)');
    zlabel('Z (km.)');
    
    legend_string = {};
    parfor k = 1:K
        legend_string{k} = sprintf('\\omega = %1.4f', wm(k));
    end
    % legend_string{K+1} = "Centroids";
    legend_string{K+1} = "Truth";
    legend(legend_string);
    grid on;
    view(3);
    hold off;
    
    subplot(2,1,2)
    hold on;
    for k = 1:K
        clusterPoints = Xm_cloud(idx == k, :);
        scatter3(vel2kms*clusterPoints(:,4), vel2kms*clusterPoints(:,5), vel2kms*clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
        hold on;
    end
    plot3(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(5), vel2kms*Xprop_truth(6), 'x','MarkerSize', 15, 'LineWidth', 3)
    title('k-Means Clustered Distribution (Velocity)');
    xlabel('Vx (km/s)');
    ylabel('Vy (km/s)');
    zlabel('Vz (km/s)');
    legend(legend_string);
    grid on;
    view(3);
    hold off;
    %}
    legend_string = "Truth";
    for ob = 1:num_agents
        for cloud = 1:num_clouds_per_agent
            % Plot planar projections
            figure(fig_num)
            fig_num = fig_num + 1;
            set(gcf, 'units','normalized','outerposition',[0 0 1 1])
            subplot(2,3,1)
            for k = 1:K{ob, cloud}
                clusterPoints = Xm_cloud{ob, cloud}(idx{ob, cloud} == k, :);
                scatter(dist2km*clusterPoints(:,1), dist2km*clusterPoints(:,2), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
                hold on;
            end
            % plot(dist2km*C_unorm(:,1), dist2km*C_unorm(:,2), '+', 'MarkerSize', 10, 'LineWidth', 3);
            % hold on;
            plot(dist2km*Xprop_truth{ob}(1), dist2km*Xprop_truth{ob}(2), 'kx','MarkerSize', 15, 'LineWidth', 3)
            title('X-Y');
            xlabel('X (km.)');
            ylabel('Y (km.)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,2)
            for k = 1:K{ob, cloud}
                clusterPoints = Xm_cloud{ob, cloud}(idx{ob, cloud} == k, :);
                scatter(dist2km*clusterPoints(:,1), dist2km*clusterPoints(:,3), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
                hold on;
            end
            % plot(dist2km*C_unorm(:,1), dist2km*C_unorm(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
            % hold on;
            plot(dist2km*Xprop_truth{ob}(1), dist2km*Xprop_truth{ob}(3), 'kx','MarkerSize', 15, 'LineWidth', 3)
            title('X-Z');
            xlabel('X (km.)');
            ylabel('Z (km.)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,3)
            for k = 1:K{ob, cloud}
                clusterPoints = Xm_cloud{ob, cloud}(idx{ob, cloud} == k, :);
                scatter(dist2km*clusterPoints(:,2), dist2km*clusterPoints(:,3), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
                hold on;
            end
            % plot(dist2km*C_unorm(:,2), dist2km*C_unorm(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
            % hold on;
            plot(dist2km*Xprop_truth{ob}(2), dist2km*Xprop_truth{ob}(3), 'kx','MarkerSize', 15, 'LineWidth', 3)
            title('Y-Z');
            xlabel('Y (km.)');
            ylabel('Z (km.)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,4)
            for k = 1:K{ob, cloud}
                clusterPoints = Xm_cloud{ob, cloud}(idx{ob, cloud} == k, :);
                scatter(vel2kms*clusterPoints(:,4), vel2kms*clusterPoints(:,5), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
                hold on;
            end
            % plot(vel2kms*C_unorm(:,4), vel2kms*C_unorm(:,5), '+', 'MarkerSize', 10, 'LineWidth', 3);
            % hold on;
            plot(vel2kms*Xprop_truth{ob}(4), vel2kms*Xprop_truth{ob}(5), 'kx','MarkerSize', 15, 'LineWidth', 3)
            title('Xdot-Ydot');
            xlabel('Xdot (km/s)');
            ylabel('Ydot (km/s)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,5)
            for k = 1:K{ob, cloud}
                clusterPoints = Xm_cloud{ob, cloud}(idx{ob, cloud} == k, :);
                scatter(vel2kms*clusterPoints(:,4), vel2kms*clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
                hold on;
            end
            % plot(vel2kms*C_unorm(:,4), vel2kms*C_unorm(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3);
            % hold on;
            plot(vel2kms*Xprop_truth{ob}(4), vel2kms*Xprop_truth{ob}(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
            title('Xdot-Zdot');
            xlabel('Xdot (km/s)');
            ylabel('Zdot (km/s)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,6)
            for k = 1:K{ob, cloud}
                clusterPoints = Xm_cloud{ob, cloud}(idx{ob, cloud} == k, :);
                scatter(vel2kms*clusterPoints(:,5), vel2kms*clusterPoints(:,6), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
                hold on;
            end
            % plot(vel2kms*C_unorm(:,5), vel2kms*C_unorm(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3, 'HandleVisibility', 'off');
            % hold on;
            plot(vel2kms*Xprop_truth{ob}(5), vel2kms*Xprop_truth{ob}(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
            title('Ydot-Zdot');
            xlabel('Ydot (km/s)');
            ylabel('Zdot (km/s)');
            legend(legend_string);
            hold off;
            
            sg = sprintf('Timestep: %3.4f Hours (Prior) Obs: %i', [time2hr*full_ts{ob}(idx_prop{ob}+1,1), ob]);
            sgtitle(sg);
            saveas(gcf, save_loc + "/Observer" + num2str(ob) + "/Timestep_0_1B_cloud_" + num2str(cloud) + ".png", 'png');
            % saveas(gcf, './Simulations/Different Orbit Simulations/Timestep_0_1B', 'png');
            Xprop_truth{ob} = [full_ts{ob}(idx_prop{ob}+1,2:4), full_vts{ob}(idx_prop{ob}+1,2:4)];
            fprintf('Truth State: \n');
            disp(Xprop_truth{ob});
        end
    end
    
    % Now that we have a GMM representing the prior distribution, we have to
    % use a Kalman update for each component: weight, mean, and covariance.
    
    % Posterior variables
    wp = wm;
    mu_p = mu_c;
    P_p = P_c;
    
    % Comment this out if you wish to use noise.
    % noised_obs = partial_ts;
    tpr = cell(1, num_agents);
    idx_meas = cell(1, num_agents);
    Xp_cloud = Xm_cloud;
    zt = cell(1, num_agents);
    for ob = 1:num_agents
        tpr{ob} = t_int{ob} + interval{ob}; % Time stamp of the prior means, weights, and covariances
        [idx_meas{ob}, ~] = find(abs(noised_obs{ob}(:,1) - tpr{ob}) < 1e-10); % Find row with time
        
        if (idx_meas{ob} ~= 0) % i.e. there exists a measurement
            R_vv = [theta_f*pi/648000, 0; 0, theta_f*pi/648000].^2;
            zt{ob} = getNoisyMeas(Xprop_truth{ob}, R_vv, h);
        end
        for cloud = 1:num_clouds_per_agent
            if (idx_meas{ob} ~= 0) % i.e. there exists a measurement
                for k = 1:K{ob, cloud}
                    % [mu_p{i}, P_p{i}] = ukfUpdate(zt, R_vv, mu_c{i}, P_c{i}, h);
                    [mu_p{ob, cloud, k}, P_p{ob, cloud, k}] = kalmanUpdate(zt{ob}, cPoints{ob, cloud, k}, R_vv, mu_c{ob, cloud, k}, P_c{ob, cloud, k}, h);
                end
            
                % Weight update
                wp{ob, cloud} = weightUpdate(wm{ob, cloud}, Xm_cloud{ob, cloud}, idx{ob, cloud}, zt{ob}, R_vv, h);
            
            else
                for k = 1:K{ob, cloud}
                    wp{ob, cloud}(k) = wm{ob, cloud}(k);
                    mu_p{ob, cloud, k} = mu_c{ob, cloud, k};
                    P_p{ob, cloud, k} = P_c{ob, cloud, k};
                end
            end
            
            Lp = 8000;
            c_id = zeros(length(Xp_cloud{ob, cloud}(:,1)),1);
            for i = 1:L
                [Xp_cloud{ob, cloud}(i,:), c_id(i)] = drawFrom(wp{ob, cloud}, mu_p(ob, cloud, :), P_p(ob, cloud, :)); 
            end
        
            mu_pExp = zeros(K{ob, cloud}, 6);
            %%save("./Outside2/Xp_cloud_Outside.mat", "Xp_cloud")
            aa = zeros(3);
            % Plot the results
            figure(fig_num)
            fig_num = fig_num + 1;
            subplot(2,1,1)
            hold on;
            for k = 1:K{ob, cloud}
                clusterPoints = Xp_cloud{ob, cloud}(c_id == k, :);
                mu_pExp(k,:) = mu_p{ob, cloud, k};
                scatter3(dist2km*clusterPoints(:,1), dist2km*clusterPoints(:,2), dist2km*clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
                hold on;
            end
            plot3(dist2km*mu_pExp(:,1), dist2km*mu_pExp(:,2), dist2km*mu_pExp(:,3), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
            hold on;
            plot3(dist2km*Xprop_truth{ob}(1), dist2km*Xprop_truth{ob}(2), dist2km*Xprop_truth{ob}(3), 'x','MarkerSize', 20, 'LineWidth', 3)
            title('Posterior Distribution (Position)');
            xlabel('X (km.)');
            ylabel('Y (km.)');
            zlabel('Z (km.)');
            legend(legend_string);
            grid on;
            view(3);
            hold off;
            
            subplot(2,1,2)
            hold on;
            for k = 1:K{ob, cloud}
                clusterPoints = Xp_cloud{ob, cloud}(c_id == k, :);
                scatter3(vel2kms*clusterPoints(:,4), vel2kms*clusterPoints(:,5), vel2kms*clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
                hold on;
            end
            plot3(vel2kms*mu_pExp(:,4), vel2kms*mu_pExp(:,5), vel2kms*mu_pExp(:,6), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
            hold on;
            plot3(vel2kms*Xprop_truth{ob}(4), vel2kms*Xprop_truth{ob}(5), vel2kms*Xprop_truth{ob}(6), 'x','MarkerSize', 20, 'LineWidth', 3)
            title('Posterior Distribution (Velocity)');
            xlabel('Vx (km/s)');
            ylabel('Vy (km/s)');
            zlabel('Vz (km/s)');
            legend(legend_string);
            grid on;
            view(3);
            hold off;
            
            legend_string = "Truth";
            
            % Plot planar projections
            figure(4)
            set(gcf, 'units','normalized','outerposition',[0 0 1 1])
            subplot(2,3,1)
            for k = 1:K{ob, cloud}
                clusterPoints = Xp_cloud{ob, cloud}(c_id == k, :);
                mu_pExp(k,:) = mu_p{ob, cloud, k};
                scatter(dist2km*clusterPoints(:,1), dist2km*clusterPoints(:,2), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
                hold on;
            end
            plot(dist2km*Xprop_truth{ob}(1), dist2km*Xprop_truth{ob}(2), 'kx','MarkerSize', 20, 'LineWidth', 3)
            title('X-Y');
            xlabel('X (km.)');
            ylabel('Y (km.)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,2)
            for k = 1:K{ob, cloud}
                clusterPoints = Xp_cloud{ob, cloud}(c_id == k, :);
                mu_pExp(k,:) = mu_p{ob, cloud, k};
                scatter(dist2km*clusterPoints(:,1), dist2km*clusterPoints(:,3), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
                hold on;
            end
            plot(dist2km*Xprop_truth{ob}(1), dist2km*Xprop_truth{ob}(3), 'kx','MarkerSize', 20, 'LineWidth', 3)
            title('X-Z');
            xlabel('X (km.)');
            ylabel('Z (km.)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,3)
            for k = 1:K{ob, cloud}
                clusterPoints = Xp_cloud{ob, cloud}(c_id == k, :);
                mu_pExp(k,:) = mu_p{ob, cloud, k};
                scatter(dist2km*clusterPoints(:,2), dist2km*clusterPoints(:,3), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
                hold on;
            end
            plot(dist2km*Xprop_truth{ob}(2), dist2km*Xprop_truth{ob}(3), 'kx','MarkerSize', 20, 'LineWidth', 3)
            title('Y-Z');
            xlabel('Y (km.)');
            ylabel('Z (km.)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,4)
            for k = 1:K{ob, cloud}
                clusterPoints = Xp_cloud{ob, cloud}(c_id == k, :);
                mu_pExp(k,:) = mu_p{ob, cloud, k};
                scatter(vel2kms*clusterPoints(:,4), vel2kms*clusterPoints(:,5), 'filled', ... 
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
                hold on;
            end
            % plot(vel2kms*mu_pExp(:,4), vel2kms*mu_pExp(:,5), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
            % hold on;
            plot(vel2kms*Xprop_truth{ob}(4), vel2kms*Xprop_truth{ob}(5), 'kx','MarkerSize', 20, 'LineWidth', 3)
            title('Xdot-Ydot');
            xlabel('Xdot (km/s)');
            ylabel('Ydot (km/s)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,5)
            for k = 1:K{ob, cloud}
                clusterPoints = Xp_cloud{ob, cloud}(c_id == k, :);
                mu_pExp(k,:) = mu_p{ob, cloud, k};
                scatter(vel2kms*clusterPoints(:,4), vel2kms*clusterPoints(:,6), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
                hold on;
            end
            % plot(vel2kms*mu_pExp(:,4), vel2kms*mu_pExp(:,6), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
            % hold on;
            plot(vel2kms*Xprop_truth{ob}(4), vel2kms*Xprop_truth{ob}(6), 'kx','MarkerSize', 20, 'LineWidth', 3)
            title('Xdot-Zdot');
            xlabel('Xdot (km/s)');
            ylabel('Zdot (km/s)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,6)
            for k = 1:K{ob, cloud}
                clusterPoints = Xp_cloud{ob, cloud}(c_id == k, :);
                mu_pExp(k,:) = mu_p{ob, cloud, k};
                scatter(vel2kms*clusterPoints(:,5), vel2kms*clusterPoints(:,6), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
                hold on;
            end
            % plot(vel2kms*mu_pExp(:,5), vel2kms*mu_pExp(:,6), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
            % hold on;
            plot(vel2kms*Xprop_truth{ob}(5), vel2kms*Xprop_truth{ob}(6), 'kx','MarkerSize', 20, 'LineWidth', 3)
            title('Ydot-Zdot');
            xlabel('Ydot (km/s)');
            ylabel('Zdot (km/s)');
            legend(legend_string);
            hold off;
            
            sg = sprintf('Timestep: %3.4f Hours (Posterior) Obs: %i', [time2hr*noised_obs{ob}(idx_meas{ob},1), ob]);
            sgtitle(sg);
            saveas(gcf,save_loc + '/Observer' + num2str(ob) + '/Timestep_0_2B_cloud_' + num2str(cloud) + '.png', 'png')
            % saveas(gcf,'./Simulations/Different Orbit Simulations/Timestep_0_2B.png', 'png')
        end
    end
    
    % At this point, we have shown a PGM-I propagation and update step. The
    % next step is to utilize this PGM-I update across all time steps during
    % which the target is within our sensor FOV and see how the particle clouds
    % (i.e. GM components) evolve over time. If we're lucky, we should see that
    % the GMM tracks the truth over the interval.
    
    % Find and set the start and end times to simulation
    idx_meas = cell(1, num_agents);
    c_meas = cell(1, num_agents);
    %idx_crit = cell(1, num_agents);
    t_end = cell(1, num_agents);
    idx_end = cell(1, num_agents);
    idx_start = cell(1, num_agents);
    l_filt = cell(1, num_agents);
    for ob = 1:num_agents
        [idx_meas{ob}, c_meas{ob}] = find(abs(hdR{ob}(:,1) - tpr{ob}) < 1e-10);
        interval{ob} = hdR{ob}(idx_meas{ob},c_meas{ob}) - hdR{ob}(idx_meas{ob}-1,c_meas{ob});
    
        %[idx_crit{ob}, ~] = find(abs(full_ts{ob}(:,1)) >= (28*24)/time2hr, 1, 'first'); % Find the index of the last time step before a certain number of days have passed since orbit propagation
        t_end{ob} = full_ts{ob}(end,1); % First observation of new pass + one more time step
        
        tau = 0;
        [idx_end{ob}, ~] = find(abs(full_ts{ob}(:,1) - t_end{ob}) < 1e-10);
        [idx_start{ob}, ~] = find(abs(full_ts{ob}(:,1) - tpr{ob}) < 1e-10);
        
        l_filt{ob} = length(full_ts{ob}(idx_start{ob}:idx_end{ob},1))+1;
    end
    
    likelihood_metric_state_space = cell(num_agents, num_clouds_per_agent);
    likelihood_metric_msmt_space = cell(num_agents, num_clouds_per_agent);
    ent2_det_cov = cell(num_agents, num_clouds_per_agent);
    ent1 = cell(num_agents, num_clouds_per_agent);
    mahalanobis = cell(num_agents, num_clouds_per_agent);
    RMSE = cell(num_agents, num_clouds_per_agent);
    std_dev = cell(num_agents, num_clouds_per_agent);
    MC_std_dev = cell(num_agents, num_clouds_per_agent);
    mat_weight_metric = cell(num_agents, num_clouds_per_agent);
    orig_weight_metric = cell(num_agents, num_clouds_per_agent);
    MC_consistency = cell(num_agents, num_clouds_per_agent);
    num_cluster = cell(num_agents, num_clouds_per_agent);
    num_particles = cell(num_agents, num_clouds_per_agent);
    for ob = 1:num_agents
        for cloud = 1:num_clouds_per_agent
            likelihood_metric_state_space{ob, cloud} = zeros(l_filt{ob},1);
            likelihood_metric_msmt_space{ob, cloud} = zeros(l_filt{ob},1);
            ent2_det_cov{ob, cloud} = zeros(l_filt{ob},1);
            ent1{ob, cloud} = zeros(l_filt{ob},length(mu_c{ob, cloud, 1})); 
            mahalanobis{ob, cloud} = zeros(l_filt{ob}, 1);
            RMSE{ob, cloud} = zeros(l_filt{ob}, 6);
            std_dev{ob, cloud} = zeros(l_filt{ob}, 6);
            MC_std_dev{ob, cloud} = zeros(l_filt{ob}, 1);
            mat_weight_metric{ob, cloud} = zeros(l_filt{ob}, 3);
            orig_weight_metric{ob, cloud} = zeros(l_filt{ob}, 3);
            MC_consistency{ob, cloud} = zeros(l_filt{ob}, 1);
            num_cluster{ob, cloud} = zeros(l_filt{ob}, 1);
            num_particles{ob, cloud} = zeros(l_filt{ob}, 1);
    
            metric_cloud = Topo2ECI(X0cloud{ob, cloud}, tpr{ob}, obs_lat{ob}, obs_lon{ob});
            metric_truth = Topo2ECI(Xot_truth{ob}', tpr{ob}, obs_lat{ob}, obs_lon{ob});
    
            %[truth_contained(ob, cloud), contained_failure_times(ob, cloud)] = checkIfInside(metric_cloud, metric_truth, 1);
            [likelihood_metric_state_space{ob, cloud}(1), ent2_det_cov{ob, cloud}(1), mahalanobis{ob, cloud}(1), RMSE{ob, cloud}(1, :), std_dev{ob, cloud}(1, :), MC_std_dev{ob, cloud}(1), mat_weight_metric{ob, cloud}(1, :), MC_consistency{ob, cloud}(1), num_cluster{ob, cloud}(1), num_particles{ob, cloud}(1)] = getStateSpaceMetrics(1, metric_cloud, metric_truth, cluster_by);
            
            metric_cloud = Topo2ECI(Xp_cloud{ob, cloud}, tpr{ob}, obs_lat{ob}, obs_lon{ob});
            metric_truth = Topo2ECI(Xprop_truth{ob}, tpr{ob}, obs_lat{ob}, obs_lon{ob});
    
            %[truth_contained(ob, cloud), contained_failure_times(ob, cloud)] = checkIfInside(metric_cloud, metric_truth, 2);
            [likelihood_metric_state_space{ob, cloud}(2), ent2_det_cov{ob, cloud}(2), mahalanobis{ob, cloud}(2), RMSE{ob, cloud}(2, :), std_dev{ob, cloud}(2, :), MC_std_dev{ob, cloud}(2), mat_weight_metric{ob, cloud}(2, :), MC_consistency{ob, cloud}(2), num_cluster{ob, cloud}(2), num_particles{ob, cloud}(2)] = getStateSpaceMetrics(K{ob, cloud}, metric_cloud, metric_truth, cluster_by);%log(det(cov(Xp_cloud)));
            ent1{ob, cloud}(1,:) = getDiagCov(X0cloud{ob, cloud});
        end
    end
    Xp_cloudp = Xp_cloud;
    
    % for to = tpr:interval:(t_end-1e-11) % Looping over the times of observation for easier propagation
    for ts = min(idx_start{:}):(max(idx_end{:})-1)
        plot_indv_clouds = false;
        if ((tau >= time_of_fusion(1)-3) && (tau <= time_of_fusion(1)+3))
            plot_indv_clouds = true;
        end
        if ((tau >= time_of_fusion(2)-3) && (tau <= time_of_fusion(2)+3))
            plot_indv_clouds = true;
        end
        Xm_cloud = cell(num_agents, num_clouds_per_agent);
        interval = cell(1, num_agents);
        for ob=1:num_agents
            to = full_ts{ob}(ts,1);
            interval{ob} = full_ts{ob}(ts+1,1) - to;
            for cloud = 1:num_clouds_per_agent
                %if (truth_contained(ob, cloud) == 1)
                ent1{ob, cloud}(tau+2,:) = getDiagCov(Xp_cloudp{ob, cloud});
            
                % Propagation Step
                Xm_cloud_tmp = propagate(Xp_cloudp{ob, cloud}, to, interval{ob}, obs_lat{ob}, obs_lon{ob});
                Xm_cloud{ob, cloud} = enforceCislunarBounds(Xm_cloud_tmp, to + interval{ob}, obs_lat{ob}, obs_lon{ob}, dist2km, vel2kms, low_lim, up_lim, vel_lim);
                %end
            end
            Xprop_truth{ob} = propagate(Xprop_truth{ob}, to, interval{ob}, obs_lat{ob}, obs_lon{ob});
        end
    
        tpr = to + interval{1}; % Time stamp of the prior means, weights, and covariances
        idx_meas = cell(1, num_agents);
        [idx_meas{1}, ~] = find(abs(noised_obs{1}(:,1) - tpr) < 1e-10); % Find row with time
        idx_meas{2} = idx_meas{1};
        tau = tau + 1;
        msmt_exists = idx_meas{1} ~= 0;
    
        % Verification Step
        if (msmt_exists)
            % Generate noisy msmt
            zt = cell(1, num_agents);
            for ob = 1:num_agents
                R_weight = [theta_f*pi/648000, 0; 0, theta_f*pi/648000].^2;
                zt{ob} = getNoisyMeas(Xprop_truth{ob}, R_weight, h);
            end
    
            idx = cell(num_agents, num_clouds_per_agent);
            idx_meas = cell(1, num_agents);
            for ob = 1:num_agents
                for cloud = 1:num_clouds_per_agent
                    %if (truth_contained(ob, cloud) == 1)
                    if (tpr >= cVal{ob})
                        K{ob, cloud} = Kmax;
                    else
                        K{ob, cloud} = Kn;
                    end
                    [idx{ob, cloud}, K{ob, cloud}, ~] = cluster(Xm_cloud{ob, cloud}, cluster_by, K{ob, cloud});
                    %end
                end
                % Verification Step
                [idx_meas{ob}, ~] = find(abs(noised_obs{ob}(:,1) - tpr) < 1e-10); % Find row with time
            end
            msmt_exists = idx_meas{1} ~= 0;
    
            fprintf("Timestamp: %1.5f\n", tpr*time2hr);
    
            mu_p = cell(num_agents, num_clouds_per_agent, Kmax);
            P_p = cell(num_agents, num_clouds_per_agent, Kmax);
            mu_mExp = cell(num_agents, num_clouds_per_agent);
            rto = cell(1, num_agents);
            [cPoints, mu_c, P_c, wm, wp] = calcGMMStatistics(Xm_cloud, idx, num_agents, num_clouds_per_agent, K, Kmax);
            
            for ob = 1:num_agents
                for cloud = 1:num_clouds_per_agent
                    orig_weight_metric{ob, cloud}(tau+2, :) = [min(wm{ob, cloud}), mean(wm{ob, cloud}), max(wm{ob, cloud})];
                    %if (truth_contained(ob, cloud) == 1)
                    % Extract means
                    mu_mExp{ob, cloud} = zeros(K{ob, cloud}, 6);
                    for k = 1:K{ob, cloud}
                        mu_mExp{ob, cloud}(k,:) = mu_c{ob, cloud, k};
                    end
    
                    % [idx_trth, ~] = find(abs(full_ts(:,1) - tpr) < 1e-10);
                    % Xprop_truth = [full_ts(idx_trth,2:4), full_vts(idx_trth,2:4)]';
                    zc = noised_obs{ob}(idx_meas{ob},2:4)'; % Presumption: An observation occurs at this time step
                    xto = zc(1)*cos(zc(2))*cos(zc(3)); 
                    yto = zc(1)*sin(zc(2))*cos(zc(3)); 
                    zto = zc(1)*sin(zc(3)); 
                    rto{ob} = [xto, yto, zto];
                
                    legend_string = {};
                    for k = 1:K{ob, cloud}
                        R_vv = [R_f{ob}*partial_ts{ob}(idx_meas{ob},2), 0, 0; 0 theta_f*pi/648000, 0; 0, 0, theta_f*pi/648000].^2;
                        Hxk = linHx(mu_c{ob, cloud, k}); % Linearize about prior mean component
                        legend_string{k} = sprintf('Distribution %i',k);
                        % legend_string{K+k} = sprintf('\\omega =  %1.4f, l = %1.4d', wm(k), gaussProb(zc, h(mu_c{k}), Hxk*P_c{k}*Hxk' + R_vv));
                    end
                    % legend_string{K+1} = "Centroids";
                    legend_string{K{ob, cloud}+1} = "Truth";
                    if(1) % Use for all time steps
                        legend_string{K{ob, cloud}+1} = "Truth";
    
                        if (plot_indv_clouds)
                            plotting_cloud = Xm_cloud{ob, cloud};
                            plotting_truth = Xprop_truth{ob};
                            plotStateSpace(plotting_cloud, ...
                                            plotting_truth, ...
                                            K{ob, cloud}, ...
                                            idx{ob, cloud}, ...
                                            dist2km, ...
                                            vel2kms, ...
                                            colors, ...
                                            sprintf('Timestep: %3.4f Hours (Prior) Ob: %i', [tpr*time2hr, ob]), ...
                                            sprintf('%s/Observer%i/Topo/Timestep_%i_1B_cloud_%i.png', save_loc, ob, tau, cloud))
                            plotMsmtSpace(plotting_cloud, ...
                                            plotting_truth, ...
                                            zt{ob}, ...
                                            h, ...
                                            K{ob, cloud}, ...
                                            idx{ob, cloud}, ...
                                            colors, ...
                                            sprintf('Az-El Timestep: %3.4f Hours (Prior) Ob: %i', [tpr*time2hr, ob]), ...
                                            sprintf('%s/Observer%i/Topo/Timestep_%i_1C_cloud_%i.png', save_loc, ob, tau, cloud), ...
                                            msmt_exists)
                
                            plotting_cloud = Topo2ECI(Xm_cloud{ob, cloud}, tpr, obs_lat{ob}, obs_lon{ob});
                            plotting_truth = Topo2ECI(Xprop_truth{ob}, tpr, obs_lat{ob}, obs_lon{ob});
                            plotStateSpace(plotting_cloud, ...
                                            plotting_truth, ...
                                            K{ob, cloud}, ...
                                            idx{ob, cloud}, ...
                                            dist2km, ...
                                            vel2kms, ...
                                            colors, ...
                                            sprintf('Timestep: %3.4f Hours (Prior) Ob: %i', [tpr*time2hr, ob]), ...
                                            sprintf('%s/Observer%i/ECI/Timestep_%i_1B_cloud_%i.png', save_loc, ob, tau, cloud))
                
                            plotting_cloud = backConvertSynodic(Xm_cloud{ob, cloud}, tpr, obs_lat{ob}, obs_lon{ob});
                            plotting_truth = backConvertSynodic(Xprop_truth{ob}, tpr, obs_lat{ob}, obs_lon{ob});
                            plotStateSpace(plotting_cloud, ...
                                            plotting_truth, ...
                                            K{ob, cloud}, ...
                                            idx{ob, cloud}, ...
                                            dist2km, ...
                                            vel2kms, ...
                                            colors, ...
                                            sprintf('Timestep: %3.4f Hours (Prior) Ob: %i', [tpr*time2hr, ob]), ...
                                            sprintf('%s/Observer%i/Synodic/Timestep_%i_1B_cloud_%i.png', save_loc, ob, tau, cloud))
                        end
                    end
                    %end
                end
                if ((size(Xp_cloudp, 2) > 1) && (plot_comb_clouds))
                    plotting_cloud = cell(num_clouds_per_agent);
                    for cloud = 1:num_clouds_per_agent
                        plotting_cloud{cloud} = Xm_cloud{ob, cloud};
                    end
                    plotting_truth = Xprop_truth{ob};
                    plotStateSpaceCombined(plotting_cloud, ...
                                            plotting_truth, ...
                                            num_clouds_per_agent, ...
                                            dist2km, ...
                                            vel2kms, ...
                                            colors, ...
                                            cloud_names, ...
                                            sprintf('Timestep: %3.4f Hours (Prior) Ob: %i', [tpr*time2hr, ob]), ...
                                            sprintf('%s/Observer%i/Topo/Combined/Timestep_%i_1B_combined.png', save_loc, ob, tau))
                    plotMsmtSpaceCombined(plotting_cloud, ...
                                            plotting_truth, ...
                                            zt{ob}, ...
                                            h, ...
                                            num_clouds_per_agent, ...
                                            colors, ...
                                            cloud_names, ...
                                            sprintf('Az-El Timestep: %3.4f Hours (Prior) Ob: %i', [tpr*time2hr, ob]), ...
                                            sprintf('%s/Observer%i/Topo/Combined/Timestep_%i_1C_combined.png', save_loc, ob, tau), ...
                                            msmt_exists)
    
                    plotting_cloud = cell(num_clouds_per_agent);
                    for cloud = 1:num_clouds_per_agent
                        plotting_cloud{cloud} = Topo2ECI(Xm_cloud{ob, cloud}, tpr, obs_lat{ob}, obs_lon{ob});
                    end
                    plotting_truth = Topo2ECI(Xprop_truth{ob}, tpr, obs_lat{ob}, obs_lon{ob});
                    plotStateSpaceCombined(plotting_cloud, ...
                                            plotting_truth, ...
                                            num_clouds_per_agent, ...
                                            dist2km, ...
                                            vel2kms, ...
                                            colors, ...
                                            cloud_names, ...
                                            sprintf('Timestep: %3.4f Hours (Prior) Ob: %i', [tpr*time2hr, ob]), ...
                                            sprintf('%s/Observer%i/ECI/Combined/Timestep_%i_1B_combined.png', save_loc, ob, tau))
    
                    plotting_cloud = cell(num_clouds_per_agent);
                    for cloud = 1:num_clouds_per_agent
                        plotting_cloud{cloud} = backConvertSynodic(Xm_cloud{ob, cloud}, tpr, obs_lat{ob}, obs_lon{ob});
                    end
                    plotting_truth = backConvertSynodic(Xprop_truth{ob}, tpr, obs_lat{ob}, obs_lon{ob});
                    plotStateSpaceCombined(plotting_cloud, ...
                                            plotting_truth, ...
                                            num_clouds_per_agent, ...
                                            dist2km, ...
                                            vel2kms, ...
                                            colors, ...
                                            cloud_names, ...
                                            sprintf('Timestep: %3.4f Hours (Prior) Ob: %i', [tpr*time2hr, ob]), ...
                                            sprintf('%s/Observer%i/Synodic/Combined/Timestep_%i_1B_combined.png', save_loc, ob, tau))
                end
            end
    
            %% Update Step
            R_vv = [theta_f*pi/648000, 0; 0, theta_f*pi/648000].^2;
            for ob = 1:num_agents
                for cloud = 1:num_clouds_per_agent
                    %if (truth_contained(ob, cloud) == 1)
                    % Update Step
                    % Hxk = linHx(mu_c{i}); % Linearize about prior mean component
                    for k = 1:K{ob, cloud}
                        [mu_p{ob, cloud, k}, P_p{ob, cloud, k}] = kalmanUpdate(zt{ob}, cPoints{ob, cloud, k}, R_vv, mu_c{ob, cloud, k}, P_c{ob, cloud, k}, h);
                        P_p{ob, cloud, k} = (P_p{ob, cloud, k} + P_p{ob, cloud, k}')/2;
                    end
                    % Weight update
                    wp{ob, cloud} = weightUpdate(wm{ob, cloud}, Xm_cloud{ob, cloud}, idx{ob, cloud}, zt{ob}, R_vv, h);
                    %end
                end
            end
        else
            %% No Msmt Exists
            fprintf("Timestamp: %1.5f\n", tpr*time2hr);
            mu_p = cell(ob, cloud, 1); 
            P_p = cell(ob, cloud, 1); 
            wm = cell(ob, cloud);
            cPoints = cell(ob, cloud, 1);
            for ob = 1:num_agents
                for cloud = 1:num_clouds_per_agent
                    %if (truth_contained(ob, cloud) == 1)
                    wm{ob, cloud} = zeros(1, 1);
                    if(tpr*time2hr>700)
                        length(Xm_cloud{ob, cloud})
                    end
                    Xp_cloud{ob, cloud} = Xm_cloud{ob, cloud}; cPoints{ob, cloud, 1} = Xp_cloud{ob, cloud};
                    wp{ob, cloud} = [1];
                    mu_p{ob, cloud, 1} = mean(Xp_cloud{ob, cloud});
                    P_p{ob, cloud, 1} = cov(Xp_cloud{ob, cloud});
                    %end
                end
            end
        end
    
        %% Resampling
        c_id = cell(ob, cloud);
        for ob = 1:num_agents
            for cloud = 1:num_clouds_per_agent
                %if (truth_contained(ob, cloud) == 1)
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
                %end
            end
        end
    
    
        %% Plot Posteriors
        if(1)
        % if(any(abs(tpr - cTimes) < 1e-10))j
            % [idx_trth, ~] = find(abs(full_ts(:,1) - tpr) < 1e-10);
            % Xprop_truth = [full_ts(idx_trth,2:4), full_vts(idx_trth,2:4)];
    
            % Extract means
            %mu_pExp = cell(1, num_agents);
            
            for ob = 1:num_agents
                for cloud = 1:num_clouds_per_agent
                    %if (truth_contained(ob, cloud) == 1)
                    mu_pExp = zeros(K{ob, cloud}, 6);
                    for k = 1:K{ob, cloud}
                        mu_pExp(k,:) = mu_p{ob, cloud, k};
                    end
                
                    legend_string = {};
                    for k = 1:K{ob, cloud}
                        legend_string{k} = sprintf('Contour %i', k);
                        % legend_string{K+k} = sprintf('\\omega = %1.4f', wp(k));
                    end
                    % legend_string{K+1} = "Centroids";
                    legend_string{K{ob, cloud}+1} = "Truth";
                    
                    if (plot_indv_clouds)
                        plotting_cloud = Xp_cloudp{ob, cloud};
                        plotting_truth = Xprop_truth{ob};
                        plotStateSpace(plotting_cloud, ...
                                        plotting_truth, ...
                                        K{ob, cloud}, ...
                                        c_id{ob, cloud}, ...
                                        dist2km, ...
                                        vel2kms, ...
                                        colors, ...
                                        sprintf('Timestep: %3.4f Hours (Posterior) Ob: %i', [tpr*time2hr, ob]), ...
                                        sprintf('%s/Observer%i/Topo/Timestep_%i_2B_cloud_%i.png', save_loc, ob, tau, cloud))
                        plotMsmtSpace(plotting_cloud, ...
                                        plotting_truth, ...
                                        zt{ob}, ...
                                        h, ...
                                        K{ob, cloud}, ...
                                        c_id{ob, cloud}, ...
                                        colors, ...
                                        sprintf('Az-El Timestep: %3.4f Hours (Posterior) Ob: %i', [tpr*time2hr, ob]), ...
                                        sprintf('%s/Observer%i/Topo/Timestep_%i_2C_cloud_%i.png', save_loc, ob, tau, cloud), ...
                                        msmt_exists)
        
                        plotting_cloud = Topo2ECI(Xp_cloudp{ob, cloud}, tpr, obs_lat{ob}, obs_lon{ob});
                        plotting_truth = Topo2ECI(Xprop_truth{ob}, tpr, obs_lat{ob}, obs_lon{ob});
                        plotStateSpace(plotting_cloud, ...
                                        plotting_truth, ...
                                        K{ob, cloud}, ...
                                        c_id{ob, cloud}, ...
                                        dist2km, ...
                                        vel2kms, ...
                                        colors, ...
                                        sprintf('Timestep: %3.4f Hours (Posterior) Ob: %i', [tpr*time2hr, ob]), ...
                                        sprintf('%s/Observer%i/ECI/Timestep_%i_2B_cloud_%i.png', save_loc, ob, tau, cloud))
        
                        plotting_cloud = backConvertSynodic(Xp_cloudp{ob, cloud}, tpr, obs_lat{ob}, obs_lon{ob});
                        plotting_truth = backConvertSynodic(Xprop_truth{ob}, tpr, obs_lat{ob}, obs_lon{ob});
                        plotStateSpace(plotting_cloud, ...
                                        plotting_truth, ...
                                        K{ob, cloud}, ...
                                        c_id{ob, cloud}, ...
                                        dist2km, ...
                                        vel2kms, ...
                                        colors, ...
                                        sprintf('Timestep: %3.4f Hours (Posterior) Ob: %i', [tpr*time2hr, ob]), ...
                                        sprintf('%s/Observer%i/Synodic/Timestep_%i_2B_cloud_%i.png', save_loc, ob, tau, cloud))
                    end
                    %end
                end
                if ((size(Xp_cloudp, 2) > 1) && (plot_comb_clouds))
                    
                    plotting_cloud = cell(num_clouds_per_agent);
                    for cloud = 1:num_clouds_per_agent
                        plotting_cloud{cloud} = Xp_cloudp{ob, cloud};
                    end
                    plotting_truth = Xprop_truth{ob};
                    plotStateSpaceCombined(plotting_cloud, ...
                                            plotting_truth, ...
                                            num_clouds_per_agent, ...
                                            dist2km, ...
                                            vel2kms, ...
                                            colors, ...
                                            cloud_names, ...
                                            sprintf('Timestep: %3.4f Hours (Posterior) Ob: %i', [tpr*time2hr, ob]), ...
                                            sprintf('%s/Observer%i/Topo/Combined/Timestep_%i_2B_combined.png', save_loc, ob, tau))
                    plotMsmtSpaceCombined(plotting_cloud, ...
                                            plotting_truth, ...
                                            zt{ob}, ...
                                            h, ...
                                            num_clouds_per_agent, ...
                                            colors, ...
                                            cloud_names, ...
                                            sprintf('Az-El Timestep: %3.4f Hours (Posterior) Ob: %i', [tpr*time2hr, ob]), ...
                                            sprintf('%s/Observer%i/Topo/Combined/Timestep_%i_2C_combined.png', save_loc, ob, tau), ...
                                            msmt_exists)
    
                    plotting_cloud = cell(num_clouds_per_agent);
                    for cloud = 1:num_clouds_per_agent
                        plotting_cloud{cloud} = Topo2ECI(Xp_cloudp{ob, cloud}, tpr, obs_lat{ob}, obs_lon{ob});
                    end
                    plotting_truth = Topo2ECI(Xprop_truth{ob}, tpr, obs_lat{ob}, obs_lon{ob});
                    plotStateSpaceCombined(plotting_cloud, ...
                                            plotting_truth, ...
                                            num_clouds_per_agent, ...
                                            dist2km, ...
                                            vel2kms, ...
                                            colors, ...
                                            cloud_names, ...
                                            sprintf('Timestep: %3.4f Hours (Posterior) Ob: %i', [tpr*time2hr, ob]), ...
                                            sprintf('%s/Observer%i/ECI/Combined/Timestep_%i_2B_combined.png', save_loc, ob, tau))
    
                    plotting_cloud = cell(num_clouds_per_agent);
                    for cloud = 1:num_clouds_per_agent
                        plotting_cloud{cloud} = backConvertSynodic(Xp_cloudp{ob, cloud}, tpr, obs_lat{ob}, obs_lon{ob});
                    end
                    plotting_truth = backConvertSynodic(Xprop_truth{ob}, tpr, obs_lat{ob}, obs_lon{ob});
                    plotStateSpaceCombined(plotting_cloud, ...
                                            plotting_truth, ...
                                            num_clouds_per_agent, ...
                                            dist2km, ...
                                            vel2kms, ...
                                            colors, ...
                                            cloud_names, ...
                                            sprintf('Timestep: %3.4f Hours (Posterior) Ob: %i', [tpr*time2hr, ob]), ...
                                            sprintf('%s/Observer%i/Synodic/Combined/Timestep_%i_2B_combined.png', save_loc, ob, tau))
                end
            end
        end
        % if(1)
        %{
        if(idx_meas ~= 0)
            K = Kn;
        else
            K = 1;
        end
        %}
    
        %% Metrics
        for ob = 1:num_agents
            for cloud = 1:num_clouds_per_agent
                if (tau + 2 == 20)
                    tau
                end
                %if (truth_contained(ob, cloud) == 1)
                if (msmt_exists)
                    %wsum = 0;
                    %for k = 1:K
                    %    wsum = wsum + wp(k)*det(P_p{k});
                    %end
                    %ent2(tau+2) = log(wsum);
                    metric_cloud = Topo2ECI(Xp_cloudp{ob, cloud}, tpr, obs_lat{ob}, obs_lon{ob});
                    metric_truth = Topo2ECI(Xprop_truth{ob}, tpr, obs_lat{ob}, obs_lon{ob});
                    %[truth_contained(ob, cloud), contained_failure_times(ob, cloud)] = checkIfInside(metric_cloud, metric_truth, tau);
                    [likelihood_metric_state_space{ob, cloud}(tau+2), ent2_det_cov{ob, cloud}(tau+2), mahalanobis{ob, cloud}(tau+2), RMSE{ob, cloud}(tau+2, :), std_dev{ob, cloud}(tau+2, :), MC_std_dev{ob, cloud}(tau+2), mat_weight_metric{ob, cloud}(tau+2, :), MC_consistency{ob, cloud}(tau+2), num_cluster{ob, cloud}(tau+2), num_particles{ob, cloud}(tau+2)] = getStateSpaceMetrics(K{ob, cloud}, metric_cloud, metric_truth, cluster_by);
                    [likelihood_metric_msmt_space{ob, cloud}(tau+2)] = getMsmtSpaceMetrics(K{ob, cloud}, Xp_cloudp{ob, cloud}, zt{ob}, h);
                else
                    if (tpr >= cVal{ob})
                        Ke = Kmax; % Clusters used for calculating entropy
                    else
                        Ke = Kn; % Clusters used for calculating entropy
                    end
                    metric_cloud = Topo2ECI(Xp_cloudp{ob, cloud}, tpr, obs_lat{ob}, obs_lon{ob});
                    metric_truth = Topo2ECI(Xprop_truth{ob}, tpr, obs_lat{ob}, obs_lon{ob});
                    %[truth_contained(ob, cloud), contained_failure_times(ob, cloud)] = checkIfInside(metric_cloud, metric_truth, tau);
                    [likelihood_metric_state_space{ob, cloud}(tau+2), ent2_det_cov{ob, cloud}(tau+2), mahalanobis{ob, cloud}(tau+2), RMSE{ob, cloud}(tau+2, :), std_dev{ob, cloud}(tau+2, :), MC_std_dev{ob, cloud}(tau+2), mat_weight_metric{ob, cloud}(tau+2, :), MC_consistency{ob, cloud}(tau+2), num_cluster{ob, cloud}(tau+2), num_particles{ob, cloud}(tau+2)] = getStateSpaceMetrics(Ke, metric_cloud, metric_truth, cluster_by); % Get entropy as if you still are using six clusters
                end
                %end
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
            
            % Resample Clouds
            [p3_idx, p3_K, ~] = cluster(p3_simple, cluster_by, Kmax);
            [~, p3_mu, p3_P, p3_w, ~] = calcGMMStatistics({p3_simple}, {p3_idx}, 1, 1, {p3_K}, Kmax);
            p3_resampled = zeros(Lp, length(Xprop_truth{ob}));
            for i = 1:Lp
                [p3_resampled(i,:), ~] = drawFrom(p3_w{1}, p3_mu, p3_P); 
            end
    
            % Convert clouds back to Topo Frame
            Xp_cloudp{1, num_clouds_per_agent+1} = convertToTopo(p3_resampled, tpr, obs_lat{1}, obs_lon{1});
            Xp_cloudp{1, num_clouds_per_agent+2} = convertToTopo(weight_update_p3_1, tpr, obs_lat{1}, obs_lon{1});
            Xp_cloudp{2, num_clouds_per_agent+1} = convertToTopo(p3_resampled, tpr, obs_lat{2}, obs_lon{2});
            Xp_cloudp{2, num_clouds_per_agent+2} = convertToTopo(weight_update_p3_2, tpr, obs_lat{2}, obs_lon{2});
            cloud_names = [cloud_names, sprintf("%s %3.1f", fusion_types(2), tpr*time2hr), sprintf("%s %3.1f", fusion_types(3), tpr*time2hr)];
            
            % Add space for metrics of new clouds
            for ob = 1:num_agents
                for new_cloud = num_clouds_per_agent+1:num_clouds_per_agent + num_new_clouds_per_agent
                    ent1{ob, new_cloud} = NaN(size(ent1{ob, 1}));
                    ent2_det_cov{ob, new_cloud} = NaN(size(ent2_det_cov{ob, 1}));
                    likelihood_metric_state_space{ob, new_cloud} = NaN(size(likelihood_metric_state_space{ob, 1}));
                    likelihood_metric_msmt_space{ob, new_cloud} = NaN(size(likelihood_metric_msmt_space{ob, 1}));
                    mahalanobis{ob, new_cloud} = NaN(size(mahalanobis{ob, 1}));
                    RMSE{ob, new_cloud} = NaN(size(RMSE{ob, 1}));
                    std_dev{ob, new_cloud} = NaN(size(std_dev{ob, 1}));
                    MC_std_dev{ob, new_cloud} = NaN(size(MC_std_dev{ob, 1}));
                    mat_weight_metric{ob, new_cloud} = NaN(size(mat_weight_metric{ob, 1}));
                    orig_weight_metric{ob, new_cloud} = NaN(size(orig_weight_metric{ob, 1}));
                    MC_consistency{ob, new_cloud} = NaN(size(MC_consistency{ob, 1}));
                    num_cluster{ob, new_cloud} = NaN(size(num_cluster{ob, 1}));
                    num_particles{ob, new_cloud} = NaN(size(num_particles{ob, 1}));
                end
            end
            
            fuse_orig_clouds(fusion_idx) = false;
            num_clouds_per_agent = num_clouds_per_agent + num_new_clouds_per_agent;
            fusion_idx = min(size(fuse_orig_clouds, 2), fusion_idx + 1);
        end
    
    
        %% Reset Number of Particles
    
        %if(abs(tpr - cTimes{1}(1)) < 1e-10)
        %    Lp = 3000;
        %elseif(abs(tpr - cTimes{1}(3)) < 1e-10)
        %    Lp = 2000;
        %end
    end
    
    %% Final Plots
    
    Xp_cloudp = cell(ob, num_clouds_per_agent);
    c_id = cell(ob, num_clouds_per_agent);
    for ob = 1:num_agents
        x = 0:l_filt{ob}-1;
        for cloud = 1:num_clouds_per_agent
            fprintf('Final State Truth:\n')
            disp(Xprop_truth{ob});
            Xp_cloudp_temp = zeros(Lp, length(Xprop_truth{ob}));
            c_id_temp = zeros(Lp,1);
            for i = 1:Lp
                [Xp_cloudp_temp(i,:), c_id_temp(i)] = drawFrom(wp{ob, cloud}, mu_p(ob, cloud, :), P_p(ob, cloud, :)); 
            end
            Xp_cloudp{ob, cloud} = Xp_cloudp_temp;
            c_id{ob, cloud} = c_id_temp;
            
            ent1{ob, cloud}(end,:) = getDiagCov(Xp_cloudp{ob, cloud});
        end
    
        % Plot the results
        plotMetricsPerState(fig_num, x, std_dev(ob, :), dist2km, vel2kms, cloud_names, colors, save_loc, ob, 'Sigma', 'Standard Deviation', 'StdDev.png');
        fig_num = fig_num + 1;
        plotMetricsPerState(fig_num, x, RMSE(ob, :), dist2km, vel2kms, cloud_names, colors, save_loc, ob, 'RMSE', 'RMSE', 'RMSE.png');
        fig_num = fig_num + 1;
    
        plotMetrics(fig_num, x, ent2_det_cov(ob, :), cloud_names, colors, save_loc, ob, 'Det Cov Entropy Metric', 'Det Cov Entropy Ob: %i', 'DetCovEntropy.png');
        fig_num = fig_num + 1;
        
        plotMetrics(fig_num, x, likelihood_metric_state_space(ob, :), cloud_names, colors, save_loc, ob, 'Log-Likelihood Metric', 'Log-Likelihood Ob: %i', 'Likelihood.png');
        fig_num = fig_num + 1;
        plotMetrics(fig_num, x, likelihood_metric_msmt_space(ob, :), cloud_names, colors, save_loc, ob, 'Log-Likelihood Msmt Space', 'Log-Likelihood Ob: %i', 'MsmtLikelihood.png');
        fig_num = fig_num + 1;
    
        plotMetrics(fig_num, x, mahalanobis(ob, :), cloud_names, colors, save_loc, ob, 'NEES', 'NEES Ob: %i', 'NEES.png');
        fig_num = fig_num + 1;
        plotMetrics(fig_num, x, MC_std_dev(ob, :), cloud_names, colors, save_loc, ob, 'Monte Carlo (Single Run) 2 Sigma Example', '2 Sigma Ob: %i', 'MC_2_sigma.png');
        fig_num = fig_num + 1;
        plotMetrics(fig_num, x, MC_consistency(ob, :), cloud_names, colors, save_loc, ob, 'Monte Carlo (Single Run) Consistency Example', 'Consistency Ob: %i', 'MC_consistency.png');
        fig_num = fig_num + 1;
        %plotMetrics(fig_num, x, RMSE(ob, :), cloud_names, colors, save_loc, ob, 'RMSE', 'RMSE Ob: %i', 'RMSE.png');
        %fig_num = fig_num + 1;
    
        plotMetrics(fig_num, x, num_cluster(ob, :), cloud_names, colors, save_loc, ob, 'Number of Clusters', 'Number of Clusters Ob: %i', 'NumClusters.png');
        fig_num = fig_num + 1;
        plotMetrics(fig_num, x, num_particles(ob, :), cloud_names, colors, save_loc, ob, 'Number of Particles', 'Number of Particles Ob: %i', 'NumParticles.png');
        fig_num = fig_num + 1;
        
        f = figure(fig_num);
        fig_num = fig_num + 1;
        f.WindowState = 'maximized';
        hold on;
        for cloud = 1:num_clouds_per_agent
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
        for cloud = 1:num_clouds_per_agent
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
    
        for cloud = 1:num_clouds_per_agent
            % Plot the results
            f = figure;
            fig_num = fig_num + 1;
            subplot(2,1,1)
            hold on;
            for k = 1:K{ob, cloud}
                clusterPoints = Xp_cloudp{ob, cloud}(c_id{ob, cloud} == k, :);
                mu_pExp(k,:) = mu_p{ob, cloud, k};
                scatter3(clusterPoints(:,1), clusterPoints(:,2), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
                hold on;
            end
            plot3(mu_pExp(:,1), mu_pExp(:,2), mu_pExp(:,3), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
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
            plot3(mu_pExp(:,4), mu_pExp(:,5), mu_pExp(:,6), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
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
            plot(mu_pExp(:,1), mu_pExp(:,2), '+', 'MarkerSize', 10, 'LineWidth', 3);
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
            plot(mu_pExp(:,1), mu_pExp(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
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
            plot(mu_pExp(:,2), mu_pExp(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
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
            plot(mu_pExp(:,4), mu_pExp(:,5), '+', 'MarkerSize', 10, 'LineWidth', 3);
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
            plot(mu_pExp(:,4), mu_pExp(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3);
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
            plot(mu_pExp(:,5), mu_pExp(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3);
            hold on;
            plot(Xprop_truth{ob}(5), Xprop_truth{ob}(6), 'kx','MarkerSize', 20, 'LineWidth', 3)
            title('Ydot-Zdot');
            xlabel('Ydot');
            ylabel('Zdot');
            legend(legend_string);
            hold off;
            
            sg = sprintf('Timestamp: %1.5f Ob: %i', [tpr*time2hr, ob]);
            sgtitle(sg)
            saveas(gcf, save_loc + '/Observer' + num2str(ob) + '/finalDistribution_normK_cloud_' + num2str(cloud) + '.png', 'png');
            close(f);
            % savefig(gcf, 'nextObservedTracklet_normK.fig');
            %}
            
            %%save("./Outside2/stdevs.mat", "ent1");
        end
    end
    
    save(save_loc + '/MC_consistency.mat', 'MC_consistency')
    save(save_loc + '/MC_std_dev.mat', 'MC_std_dev')
    
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

%% IOD Functions

function [dX_coeffs] = polyDeriv(X_coeffs)
    
    dX_coeffs = zeros(1, length(X_coeffs)-1);
    for j = length(X_coeffs):-1:2
        dX_coeffs(length(X_coeffs)+1-j) = X_coeffs(length(X_coeffs)+1-j)*(j-1);
    end
end


function [Xfit] = stateEstCloud(pf, order_fit, theta_f, R_f, obTr, tdiff, low_lim, up_lim, load_loc)
    noised_obs = obTr;

    R_t = zeros(3*length(noised_obs(:,1)),1); % We shall diagonalize this later
    mu_t = zeros(3*length(noised_obs(:,1)),1);

    load(load_loc + "/partial_ts.mat"); % Noiseless observation data
    dist2km = 384400; % Kilometers per non-dimensionalized distance

    for i = 1:length(obTr(:,1))
        mu_t(3*(i-1)+1:3*(i-1)+3, 1) = [partial_ts(i,2); partial_ts(i,3); partial_ts(i,4)];
        R_t(3*(i-1)+1:3*(i-1)+3, 1) = [(R_f*partial_ts(i,2))^2; (theta_f*4.84814e-6)^2; (theta_f*4.84814e-6)^2];
    end

    R_t = diag(R_t);
    data_vec = mvnrnd(mu_t, R_t)';

    for i = 1:length(noised_obs(:,1))
        noised_obs(i,2:4) = data_vec(3*(i-1)+1:3*(i-1)+3,1); % AZ-EL Measurements
        %noised_obs(i,2) = unifrnd(low_lim, up_lim)/dist2km;
        % while(noised_obs(i,2) < low_lim/dist2km || noised_obs(i,2) > up_lim/dist2km)
        %     noised_obs(i,2) = mvnrnd(partial_ts(i,2), (R_f*partial_ts(i,2))^2);
        % end
    end

    % Extract the first continuous observation track

    hdo = []; % Matrix for a half day observation
    i = 1;
    while(noised_obs(i+1,1) - noised_obs(i,1) < tdiff) % Add small epsilon due to roundoff error
        hdo(i,:) = noised_obs(i+1,:);
        i = i + 1;
    end
    
    % Convert observation data into [X, Y, Z] data in the topographic frame.

    hdR = zeros(length(hdo(:,1)),4); % Convert quantities of hdo to [X, Y, Z]
    hdR(:,1) = hdo(:,1); % Timestamp stays the same
    hdR(:,2) = hdo(:,2) .* cos(hdo(:,4)) .* cos(hdo(:,3)); % Conversion to X
    hdR(:,3) = hdo(:,2) .* cos(hdo(:,4)) .* sin(hdo(:,3)); % Conversion to Y
    hdR(:,4) = hdo(:,2) .* sin(hdo(:,4)); % Conversion to Z

    in_len = round(pf * length(hdR(:,1))); % Length of interpolation interval
    hdR_p = hdR(1:in_len,:); % Matrix for a partial half-day observation

    % Fit polynomials for X, Y, and Z (Cubic for X, Quadratic for X and Y)
    coeffs_X = polyfit(hdR_p(:,1), hdR_p(:,2), order_fit);
    coeffs_Y = polyfit(hdR_p(:,1), hdR_p(:,3), order_fit);
    coeffs_Z = polyfit(hdR_p(:,1), hdR_p(:,4), order_fit);

    % Predicted values for X, Y, and Z given the polynomial fits
    X_fit = polyval(coeffs_X, hdR_p(:,1));
    Y_fit = polyval(coeffs_Y, hdR_p(:,1));
    Z_fit = polyval(coeffs_Z, hdR_p(:,1));

    % Now that you have analytically calculated the coefficients of the fitted
    % polynomial, use them to obtain values for X_dot, Y_dot, and Z_dot.
    % 1) Plot the X_dot, Y_dot, and Z_dot values for the time points for the
    % slides. 
    % 2) Find a generic way of obtaining and plotting X_dot, Y_dot, and Z_dot
    % values given some set of [X_coeffs, Y_coeffs, Z_coeffs]. 

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


function [X_bt] = backConvertSynodic(X_ot, t_stamp, obs_lat, obs_lon)
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
function [X_ot] = convertToTopo(X_bt, t_stamp, obs_lat, obs_lon)
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
        X_ECI(particle, :) = [rot_ECI', vot_ECI'];
    end
end


%% PGM Filter Functions

function w = weightUpdate(wc, cluster_points, idx, zk, R, h)
    wGains = zeros(length(wc));
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


function Xm_cloud = propagate(Xcloud, t_int, interval, obs_lat, obs_lon)
    Xm_bt = zeros(size(Xcloud));
    Xbt = backConvertSynodic(Xcloud, t_int, obs_lat, obs_lon);
    num_particles = size(Xcloud, 1);
    for particle = 1:num_particles
        % Next, propagate each X_{bt} in your particle cloud by a single time 
        % step and convert back to the topographic frame.
        opts = odeset('Events', @termSat, 'RelTol', 1e-6, 'AbsTol', 1e-8); 
        [~, X] = ode15s(@cr3bp_dyn, [0 interval], Xbt(particle, :), opts);
        Xm_bt(particle, :) = X(end,:);
    end
    Xm_cloud = convertToTopo(Xm_bt, t_int + interval, obs_lat, obs_lon);
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


function clipped_cloud = enforceCislunarBounds(Xm_cloud, t_prior, obs_lat, obs_lon, dist2km, vel2kms, low_lim, up_lim, vel_lim)
    clipped_cloud = [];
    reo = getObserverPos(t_prior, obs_lat, obs_lon);
    reo = reshape(reo, [1 length(reo)]);

    for i = 1:length(Xm_cloud(:,1))
        if(norm(Xm_cloud(i,1:3) + reo)*dist2km > low_lim && ...
                norm(Xm_cloud(i,1:3) + reo)*dist2km <= up_lim && ...
                norm(Xm_cloud(i,4:6))*vel2kms < vel_lim)
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


function plotMetricsPerState(fig_num, x, y_data, dist2km, vel2kms, cloud_names, colors, save_loc, ob, y_label, title_str, filename)
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


function plotStateSpace(cloud, truth, K, cluster_idx, dist2km, vel2kms, colors, plot_title, filename)
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


function plotStateSpaceCombined(plotting_clouds, plotting_truth, num_clouds_per_agent, dist2km, vel2kms, colors, cloud_names, plot_title, filename)
    f = figure('visible','off','Position', get(0,'ScreenSize'));
    f.WindowState = 'maximized';
    
    legend_string = [cloud_names, "Truth"];

    subplot(2,3,1)
    hold on; 
    for cloud = 1:num_clouds_per_agent
        scatter(dist2km*plotting_clouds{cloud}(:,1), dist2km*plotting_clouds{cloud}(:,2), 'filled', ...
        'MarkerFaceColor', colors(cloud), 'DisplayName', legend_string(cloud));
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
    for cloud = 1:num_clouds_per_agent
        scatter(dist2km*plotting_clouds{cloud}(:,1), dist2km*plotting_clouds{cloud}(:,3), 'filled', ...
        'MarkerFaceColor', colors(cloud), 'DisplayName', legend_string(cloud));
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
    for cloud = 1:num_clouds_per_agent
        scatter(dist2km*plotting_clouds{cloud}(:,2), dist2km*plotting_clouds{cloud}(:,3), 'filled', ...
        'MarkerFaceColor', colors(cloud), 'DisplayName', legend_string(cloud));
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
    for cloud = 1:num_clouds_per_agent
        scatter(vel2kms*plotting_clouds{cloud}(:,4), vel2kms*plotting_clouds{cloud}(:,5), 'filled', ...
        'MarkerFaceColor', colors(cloud), 'DisplayName', legend_string(cloud));
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
    for cloud = 1:num_clouds_per_agent
        scatter(vel2kms*plotting_clouds{cloud}(:,4), vel2kms*plotting_clouds{cloud}(:,6), 'filled', ...
        'MarkerFaceColor', colors(cloud), 'DisplayName', legend_string(cloud));
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
    for cloud = 1:num_clouds_per_agent
        scatter(vel2kms*plotting_clouds{cloud}(:,5), vel2kms*plotting_clouds{cloud}(:,6), 'filled', ...
        'MarkerFaceColor', colors(cloud), 'DisplayName', legend_string(cloud));
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


function plotMsmtSpace(cloud, truth, zt, h, K, cluster_idx, colors, plot_title, filename, msmt_exists)
    f = figure('visible','off','Position', get(0,'ScreenSize'));
    f.WindowState = 'maximized';
    
    legend_string = ["Truth"];
    hold on;
    %scatter_handles = gobjects(k,1);
    for k = 1:K
        pts = cloud(cluster_idx == k, :);
        Zmcloud = zeros(length(pts(:,1)), length(zt));
        for i = 1:length(Zmcloud(:,1))
            Zmcloud(i,:) = h(pts(i,:))';
        end
        scatter(180/pi*Zmcloud(:,1), 180/pi*Zmcloud(:,2), 'filled', ...
        'MarkerFaceColor', colors(k));%, 'DisplayName', sprintf('k: %i; w: %.3f', [k, wp{ob, cloud}(k)]));
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

    legend(legend_string);
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
    plot(180/pi*Ztruth(1), 180/pi*Ztruth(2), 'kx', 'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string(3));
    if msmt_exists
        legend_string = [legend_string, "Noisy Truth"];
        plot(180/pi*zt(1), 180/pi*zt(2), 'ko', 'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string(4));
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
%{
function [inside, time] = checkIfInside(Xcloud, Xtruth, time, threshold)
    if nargin < 4
        threshold = 0.05;
    end
    
    [bin_table, edges] = binCloud(Xcloud);

    % ----------------------------
    % 3. Find truth particle bin
    % ----------------------------
    num_dims = size(Xcloud, 2);
    truth_idx = zeros(1, num_dims);
    for i = 1:num_dims
        truth_idx(i) = discretize(Xtruth(i), edges{i});
    end

    if any(isnan(truth_idx))
        inside = false;
        return;
    end

    % ----------------------------
    % 4. Check probability mass in that bin
    % ----------------------------
    match = ismember(bin_table.subs, truth_idx, 'rows');
    if any(match)
        prob_mass = bin_table.probs(match);
    else
        prob_mass = 0;
    end

    inside = prob_mass >= threshold;
    if (inside)
        time = 0;
    end
end
%}


function [likelihood, ent_det_cov, NEES, RMSE, std_dev, MC_std_dev, weights, MC_consistency, Kp, Lp] = getStateSpaceMetrics(Kp, Xcloud, Xtruth, cluster_by)
    Kp = 8;
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
        norm_vc = (vc - mean_vc)./norm(std_vc); % Normalizing the velocity
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
    best_weight = gmm_unnorm.ComponentProportion(best_mode);
    best_samples = mvnrnd(best_mu, best_cov, 1000000);

    likelihood = log(mvnpdf(Xtruth, best_mu, best_cov));

    diff = Xtruth - best_mu;
    NEES = gmm_unnorm.ComponentProportion(best_mode)* (diff * (best_cov \ diff'));  % Mahalanobis distance
    RMSE = sqrt(mean((best_samples - Xtruth).^2, 1));

    std_dev = sqrt(diag(best_cov));
    ent_det_cov = log(gmm_unnorm.ComponentProportion(best_mode)*det(best_cov));

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

%{
function [likelihood, ent_det_cov_orig, ent_det_cov, ent_diff_ent, ent_discr_ent, Dsum, RMSE, Kp, Lp] = getStateSpaceMetrics(Kp, Xcloud, Xtruth, cluster_by)
    Kp = 6;
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
        norm_vc = (vc - mean_vc)./norm(std_vc); % Normalizing the velocity
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

    wsum = 0;
    for k = 1:Kp
        wsum = wsum + w(k)*det(P{k});
    end
    ent_det_cov_orig = log(wsum);
    
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

    RMSE = sqrt(mean(vecnorm(Xcloud - Xtruth, 2, 2).^2));

    likelihood = log(pdf(gmm_unnorm, Xtruth));

    ent_samples = random(gmm_unnorm, 1000000);
    ent_diff_ent = -mean(log(pdf(gmm_unnorm, ent_samples) + 1e-300));

    D = zeros(1,Kp);
    wsum = 0;
    for k = 1:Kp
        mu_k = gmm_unnorm.mu(k, :);             % 1×d mean of component k
        Sigma_k = gmm_unnorm.Sigma(:, :, k);    % d×d covariance of component k
        diff = Xtruth - mu_k;
        D(k) = diff * (Sigma_k \ diff');  % Mahalanobis distance
        wsum = wsum + gmm_unnorm.ComponentProportion(k)*det(Sigma_k);
    end
    Dsum = sum(gmm_unnorm.ComponentProportion .* D);
    ent_det_cov = log(wsum);

    %[binned_cloud, ~] = binCloud(ent_samples);
    %nonzeros_binned_cloud = binned_cloud(binned_cloud > 0);
    %ent_discr_ent = -sum(nonzeros_binned_cloud .* log(nonzeros_binned_cloud));
    ent_discr_ent = 0;
end
%}

function [likelihood] = getMsmtSpaceMetrics(Kp, Xcloud, Xmsmt, h)
    Kp = 8;
    msmt_cloud = zeros(length(Xcloud), 2);
    for j = 1:length(Xcloud)
        msmt_cloud(j,:) = h(Xcloud(j,:));
    end
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
    best_weight = gmm_unnorm.ComponentProportion(best_mode);

    likelihood = best_weight*log(mvnpdf(Xmsmt', best_mu, best_cov));
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
        norm_rc = (rc - mean_rc)./norm(std_rc); % Normalizing the position

        vc = data(:,4:6);
        mean_vc = mean(vc, 1);
        std_vc = std(vc,0,1);
        norm_vc = (vc - mean_vc)./norm(std_vc); % Normalizing the velocity
        Xm_norm = [norm_rc, norm_vc];
    end
    if cluster_by == "Velocity"
        vc = data(:,4:6);
        mean_vc = mean(vc, 1);
        std_vc = std(vc,0,1);
        norm_vc = (vc - mean_vc)./norm(std_vc); % Normalizing the velocity
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


function [cPoints, mu_c, P_c, wm, wp] = calcGMMStatistics(Xcloud, idx, num_agents, num_clouds_per_agent, K, Kmax)
    cPoints = cell(num_agents, num_clouds_per_agent, Kmax);
    mu_c = cell(num_agents, num_clouds_per_agent, Kmax);
    P_c = cell(num_agents, num_clouds_per_agent, Kmax);
    wm = cell(num_agents, num_clouds_per_agent);
    wp = cell(num_agents, num_clouds_per_agent);
    % Calculate covariances and weights for each cluster
    for ob = 1:num_agents
        for cloud = 1:num_clouds_per_agent
            %if (truth_contained(ob, cloud) == 1)
            wm_temp = zeros(K{ob, cloud}, 1);
            wp{ob, cloud} = wm_temp;
            for k = 1:K{ob, cloud}
                cluster_points = Xcloud{ob, cloud}(idx{ob, cloud} == k, :); 
                cPoints{ob, cloud, k} = cluster_points; 
                mu_c{ob, cloud, k} = mean(cluster_points, 1); % Cell of GMM means 
                if (length(cluster_points(:,1)) == 1)
                    P_c{ob, cloud, k} = zeros(length(mu_c{ob, cloud, k}));
                else
                    P_c{ob, cloud, k} = cov(cluster_points); % Cell of GMM covariances
                end
                wm_temp(k) = size(cluster_points, 1) / size(Xcloud{ob, cloud}, 1); % Vector of (prior) weights
            end
            wm{ob, cloud} = wm_temp;
            %end
        end
    end
end


function [optimal_gmm, optimal_K] = matlabGMM(Kp, Xcloud)
    delta = 1;  % how far to search on either side of Kp
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


function displayBins(binned_cloud, edges, dist2km, vel2kms, save_loc, specific_plot_info)
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
        centers{i} = centers{i} * dist2km;
    end
    for i = 4:6
        centers{i} = centers{i} * vel2kms;
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


function [p3, p3_tallest_peaks, weight_update_p3_1, weight_update_p3_2, bin_axes]  = fusionMethods(p1, p2, cluster_by, Kmax, num_agents, num_particles, display_diagnostics, save_loc, dist2km, vel2kms, fusion_idx)
    bin_axes = cell(1, 3);
    p3 = vertcat(p1, p2);
    [p3_simple, bin_axes{1}] = binCloud(p3);
    displayBins(p3_simple, bin_axes{1}, dist2km, vel2kms, save_loc, "simple_" + num2str(fusion_idx))

    [binned_p1, ~] = binCloud(p1, bin_axes{1});
    [binned_p2, ~] = binCloud(p2, bin_axes{1});
    displayBins(binned_p1, bin_axes{1}, dist2km, vel2kms, save_loc, "1_" + num2str(fusion_idx))
    displayBins(binned_p2, bin_axes{1}, dist2km, vel2kms, save_loc, "2_" + num2str(fusion_idx))
    
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
    [weight_update_p3_1, bin_axes{2}] = postWeights({p1, p2}, cluster_by, Kmax, num_agents, num_particles, display_diagnostics, save_loc, dist2km, vel2kms, 1, fusion_idx);
    [weight_update_p3_2, bin_axes{3}] = postWeights({p2, p1}, cluster_by, Kmax, num_agents, num_particles, display_diagnostics, save_loc, dist2km, vel2kms, 2, fusion_idx);
    
    %if(display_diagnostics == true)
    %    disp("Diagnostics: Fusion Sums: " + num2str(sum(p3_simple, [],  "all"))\ ...
    %        + ", " + num2str(sum(p3_simple, [],  "all"))\ ...
    %        + ", " + num2str(sum(weight_update_p3_1, [],  "all"))\ ...
    %        + ", " + num2str(sum(weight_update_p3_2, [],  "all")))
    %end
end


function [new_particles, bin_axes] = postWeights(data, cluster_by, Kmax, num_agents, num_particles, display_diagnostics, save_loc, dist2km, vel2kms, ob, fusion_idx)
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
    displayBins(p3, bin_axes, dist2km, vel2kms, save_loc, "Ishan" + num2str(ob) + "_" + num2str(fusion_idx))
end


function [post_mu, post_P] = KalmanFilter(mu_1, P_1, mu_2, P_2)
    post_mu = mu_1 + P_1 * inv(P_1 + P_2) * (mu_2 - mu_1);
    post_P = P_1 - P_1 * inv(P_1 + P_2) * P_1;
    post_P = (post_P + post_P') / 2;
end
