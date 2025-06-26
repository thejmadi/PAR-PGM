% Start the clock
tic
save_loc = "D:/PythonProjects/EDP/PGM/ParticleFusionTest/6_20_25_meeting/Cislunar2ObsTest/Test1";
load_loc = "D:/PythonProjects/EDP/PGM/ParticleFusionTest/Obs";
rng(5, "twister")
cluster_by = "FullState";
L = 1000;
num_agents = 2;
num_clouds_per_agent = 1;
num_clouds = num_agents * num_clouds_per_agent;
combine = 1;
time_of_fusion = 5;
% College Station
obs_lat{1} = 30.618963;
obs_lon{1} = -96.339214;
%obs_lat{2} = 30.618963;
%obs_lon{2} = -96.339214;
% Buenos Aires
obs_lat{2} = -34.612979;
obs_lon{2} = -58.453656;

% Load noiseless observation data and other important .mat files
partial_ts = cell(1, num_agents);
full_ts = cell(1, num_agents);
full_vts = cell(1, num_agents);
for ob = 1:num_agents
    for cloud = ob:ob+num_clouds_per_agent-1
        partial_ts{ob} = load(load_loc + num2str(ob) + "/partial_ts.mat").partial_ts; % Noiseless observation data
        full_ts{ob} = load(load_loc + num2str(ob) + "/full_ts.mat").full_ts; % Position truth (topocentric frame)
        full_vts{ob} = load(load_loc + num2str(ob) + "/full_vts.mat").full_vts; % Velocity truth (topocentric frame)
    end
end
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
R_f = 0.05; % Range percentage error covariance

dist2km = 384400; % Kilometers per non-dimensionalized distance
time2hr = 4.342*24; % Hours per non-dimensionalized time
vel2kms = dist2km/(time2hr*60*60); % Kms per non-dimensionalized velocity

noised_obs = cell(1, num_agents);
interval = cell(1, num_agents);
cTimes = cell(1, num_agents); % Array of important time points
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
    for cloud = ob:ob+num_clouds_per_agent-1
        noised_obs{ob} = partial_ts{ob};
    
    
        R_t = zeros(3*length(noised_obs{ob}(:,1)),1); % We shall diagonalize this later
        mu_t = zeros(3*length(noised_obs{ob}(:,1)),1);
        
        for i = 1:length(partial_ts{ob}(:,1))
            mu_t(3*(i-1)+1:3*(i-1)+3, 1) = [partial_ts{ob}(i,2); partial_ts{ob}(i,3); partial_ts{ob}(i,4)];
            R_t(3*(i-1)+1:3*(i-1)+3, 1) = [(0.05*partial_ts{ob}(i,2))^2; (theta_f*4.84814e-6)^2; (theta_f*4.84814e-6)^2];
        end
    
        R_t = diag(R_t);
        data_vec = mvnrnd(mu_t, R_t)';
        %%save("./dataVec/data_vec.mat", "data_vec");
        
        for i = 1:length(noised_obs{ob}(:,1))
            noised_obs{ob}(i,2:4) = data_vec(3*(i-1)+1:3*(i-1)+3,1);
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
        
        larger_diff = noised_obs{ob}(end,1) - noised_obs{ob}(end-1,1);
        for j = 2:length(noised_obs{ob}(:,1))
            if (noised_obs{ob}(j,1) - noised_obs{ob}(j-1,1) > larger_diff+1e-11)
                cVal{ob} = noised_obs{ob}(j,1); break;
            else
                cVal{ob} = noised_obs{ob}(end,1);
            end
        end
    
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
        
        pf{ob} = 0.25; % A factor between 0 to 1 describing the length of the day to interpolate [x, y]
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
end

Lp = 1*L;
X0cloud = cell(1, num_clouds);
fig_num = 1;
for ob = 1:num_agents
    for cloud = ob:ob+num_clouds_per_agent-1
        obs_X0cloud = zeros(L,6);
        parfor i = 1:length(obs_X0cloud(:,1))
            obs_X0cloud(i,:) = stateEstCloud(pf{ob}, partial_ts{ob}, (partial_ts{ob}(2,1) - partial_ts{ob}(1,1)) + 1e-15, i, load_loc + num2str(ob));
        end
        X0cloud{cloud} = obs_X0cloud;
        %%save("./X0cloud/X0cloud.mat", "X0cloud");
    
        figure(fig_num);
        fig_num = fig_num + 1;
        set(gcf, 'units','normalized','outerposition',[0 0 1 1])
        subplot(2,3,1)
        plot(dist2km*X0cloud{cloud}(:,1), dist2km*X0cloud{cloud}(:,2), '.')
        hold on;
        plot(dist2km*Xot_truth{ob}(1), dist2km*Xot_truth{ob}(2), 'kx','MarkerSize', 15, 'LineWidth', 3)
        title('X-Y');
        xlabel('X (km.)');
        ylabel('Y (km.)');
        legend('Estimate','Truth');
        hold off;
        
        subplot(2,3,2)
        plot(dist2km*X0cloud{cloud}(:,1), dist2km*X0cloud{cloud}(:,3), '.')
        hold on;
        plot(dist2km*Xot_truth{ob}(1), dist2km*Xot_truth{ob}(3), 'kx','MarkerSize', 15, 'LineWidth', 3)
        title('X-Z');
        xlabel('X (km.)');
        ylabel('Z (km.)');
        legend('Estimate','Truth');
        hold off;
        
        subplot(2,3,3)
        plot(dist2km*X0cloud{cloud}(:,2), dist2km*X0cloud{cloud}(:,3), '.')
        hold on;
        plot(dist2km*Xot_truth{ob}(2), dist2km*Xot_truth{ob}(3), 'kx','MarkerSize', 15, 'LineWidth', 3)
        title('Y-Z');
        xlabel('Y (km.)');
        ylabel('Z (km.)');
        legend('Estimate','Truth');
        hold off;
        
        subplot(2,3,4)
        plot(vel2kms*X0cloud{cloud}(:,4), vel2kms*X0cloud{cloud}(:,5), '.')
        hold on;
        plot(vel2kms*Xot_truth{ob}(4), vel2kms*Xot_truth{ob}(5), 'kx','MarkerSize', 15, 'LineWidth', 3)
        title('Xdot-Ydot');
        xlabel('Xdot (km/s)');
        ylabel('Ydot (km/s)');
        legend('Estimate','Truth');
        hold off;
        
        subplot(2,3,5)
        plot(vel2kms*X0cloud{cloud}(:,4), vel2kms*X0cloud{cloud}(:,6), '.')
        hold on;
        plot(vel2kms*Xot_truth{ob}(4), vel2kms*Xot_truth{ob}(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
        title('Xdot-Zdot');
        xlabel('Xdot (km/s)');
        ylabel('Zdot (km/s)');
        legend('Estimate','Truth');
        hold off;
        
        subplot(2,3,6)
        plot(vel2kms*X0cloud{cloud}(:,5), vel2kms*X0cloud{cloud}(:,6), '.')
        hold on;
        plot(vel2kms*Xot_truth{ob}(5), vel2kms*Xot_truth{ob}(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
        title('Ydot-Zdot');
        xlabel('Ydot (km/s)');
        ylabel('Zdot (km/s)');
        legend('Estimate','Truth');
        hold off;
        
        sg = sprintf('Timestep: %3.4f Hours Obs: %i', [t_truth{ob}*time2hr, ob]);
        sgtitle(sg);
        savefig(gcf, "iodCloud_obs_" + num2str(ob) + "_cloud_" + num2str(cloud) + ".fig");
        saveas(gcf, save_loc + "/Observer" + num2str(ob) + "/iodCloud_cloud_" + num2str(cloud) + ".png", 'png');
        % saveas(gcf, './Simulations/Different Orbit Simulations/iodCloud.png', 'png');
    end
end

t_int = cell(1, num_agents);
%tspan = 0:interval:interval; % Integrate over just a single time step
Xm_cloud = X0cloud;
%Xbt = zeros(L, 6);
%X_all = cell(length(X0cloud(:,1)), 1);
%T_all = cell(length(X0cloud(:,1)), 1);
%Xm_bt = zeros(size(X0cloud));
for ob = 1:num_agents
    t_int{ob} = hdR_p{ob}(end,1); % Time at which we are obtaining a state cloud
    for i = 1:length(X0cloud{ob}(:,1))
        % First, convert from X_{ot} in the topocentric frame to X_{bt} in the
        % synodic frame.
        Xbt = backConvertSynodic(X0cloud{ob}(i,:)', t_int{ob}, obs_lat{ob}, obs_lon{ob});
        % Next, propagate each X_{bt} in your particle cloud by a single time 
        % step and convert back to the topographic frame.
         % Call ode45()
        opts = odeset('Events', @termSat);
        [t,X] = ode45(@cr3bp_dyn, [0 interval{ob}], Xbt, opts); % Assumes termination event (i.e. target enters LEO)
        %X_all{i} = X;
        %T_all{i} = t;
        Xm_bt = X(end,:)';
        Xm_cloud{ob}(i,:) = convertToTopo(Xm_bt, t_int{ob} + interval{ob}, obs_lat{ob}, obs_lon{ob});
        %bCS_idx = bCS_idx + 1;
        % Xm_cloud(i,:) = procNoise(Xm_cloud(i,:)); % Adds process noise
    end
end
%%save("./Outside/Xbt_Outside.mat", "Xbt");
%%save("./Outside/X_Outside.mat", "X_all");
%%save("./Outside/T_Outside.mat", "T_all");
%%save("./Outside/Xm_bt_Outside.mat", "Xm_bt")
%%save("./Outside/Xm_cloud_Outside.mat", "Xm_cloud");
% Initialize variables
Kn = 8; % Number of clusters (original)
K = {Kn, Kn}; % Number of clusters (changeable)
Kmax = 8; % Maximum number of clusters (Kmax = 1 for EnKF)
idx = cell(1, num_clouds);

for ob = 1:num_agents
    for cloud = ob:ob+num_clouds_per_agent-1
        [idx{cloud}, K{cloud}, C] = cluster(Xm_cloud{cloud}, cluster_by, K{cloud});
        colors = ["Red", "Blue", "Green", "Yellow", "Magenta", "Cyan", "Black", "#500000", "#bf5700", "#00274c"];
        contourCols = lines(Kmax);
    end
end

cPoints = cell(Kmax, num_clouds);

mu_c = cell(Kmax, num_clouds);
P_c = cell(Kmax, num_clouds);
wm = cell(1, num_clouds);

for ob = 1:num_agents
    for cloud = ob:ob+num_clouds_per_agent-1
        wm{cloud} = zeros(K{cloud}, 1);
        % Calculate covariances and weights for each cluster
        for k = 1:K{cloud}
            cluster_points = Xm_cloud{cloud}(idx{cloud} == k, :); % Keep clustering very separate from mean, covariance, weight calculations
            cPoints{k, cloud} = cluster_points; cSize = size(cPoints{k, cloud});
            mu_c{k, cloud} = mean(cluster_points, 1); % Cell of GMM means 
        
            if(cSize(1) == 1)
                P_c{k, cloud} = zeros(length(K{cloud}));
            else
                P_c{k, cloud} = cov(cluster_points); % Cell of GMM covariances 
            end
            wm{cloud}(k) = size(cluster_points, 1) / size(Xm_cloud{cloud}, 1); % Vector of weights
        end
    end
end

% Plot the results
warning('off', 'MATLAB:legend:IgnoringExtraEntries');

%{
figure(1)
subplot(2,1,1)
hold on;
for k = 1:K
    clusterPoints = Xm_cloud(idx == k, :);
    scatter3(clusterPoints(:,1), clusterPoints(:,2), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
end
plot3(C_unorm(:,1), C_unorm(:,2), C_unorm(:,3), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
hold on;
plot3(Xprop_truth(1), Xprop_truth(2), Xprop_truth(3), 'x','MarkerSize', 15, 'LineWidth', 3)
title('k-Means Clustered Distribution (Position)');
xlabel('X');
ylabel('Y');
zlabel('Z');

legend_string = {};
for k = 1:K
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
    scatter3(clusterPoints(:,4), clusterPoints(:,5), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
end
plot3(C_unorm(:,4), C_unorm(:,5), C_unorm(:,6), 'k+', 'MarkerSize', 15, 'LineWidth', 3);
hold on;
plot3(Xprop_truth(4), Xprop_truth(5), Xprop_truth(6), 'x','MarkerSize', 15, 'LineWidth', 3)
title('k-Means Clustered Distribution (Velocity)');
xlabel('Vx');
ylabel('Vy');
zlabel('Vz');
legend(legend_string);
grid on;
view(3);
hold off;
%}

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
plot3(dist2km*C_unorm(:,1), dist2km*C_unorm(:,2), dist2km*C_unorm(:,3), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
hold on;
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
plot3(vel2kms*C_unorm(:,4), vel2kms*C_unorm(:,5), vel2kms*C_unorm(:,6), 'k+', 'MarkerSize', 15, 'LineWidth', 3);
hold on;
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
    for cloud = ob:ob+num_clouds_per_agent-1
        % Plot planar projections
        figure(fig_num)
        fig_num = fig_num + 1;
        set(gcf, 'units','normalized','outerposition',[0 0 1 1])
        subplot(2,3,1)
        for k = 1:K{cloud}
            clusterPoints = Xm_cloud{cloud}(idx{cloud} == k, :);
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
        for k = 1:K{cloud}
            clusterPoints = Xm_cloud{cloud}(idx{cloud} == k, :);
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
        for k = 1:K{cloud}
            clusterPoints = Xm_cloud{cloud}(idx{cloud} == k, :);
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
        for k = 1:K{cloud}
            clusterPoints = Xm_cloud{cloud}(idx{cloud} == k, :);
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
        for k = 1:K{cloud}
            clusterPoints = Xm_cloud{cloud}(idx{cloud} == k, :);
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
        for k = 1:K{cloud}
            clusterPoints = Xm_cloud{cloud}(idx{cloud} == k, :);
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
for ob = 1:num_agents
    tpr{ob} = t_int{ob} + interval{ob}; % Time stamp of the prior means, weights, and covariances
    [idx_meas{ob}, ~] = find(abs(noised_obs{ob}(:,1) - tpr{ob}) < 1e-10); % Find row with time
    
    if (idx_meas{ob} ~= 0) % i.e. there exists a measurement
        R_vv = [theta_f*pi/648000, 0; 0, theta_f*pi/648000].^2;
        h = @(x) [atan2(x(2),x(1)); pi/2 - acos(x(3)/sqrt(x(1)^2 + x(2)^2 + x(3)^2))]; % Nonlinear measurement model
        zt = getNoisyMeas(Xprop_truth{ob}, R_vv, h, 1);
    end
    for cloud = ob:ob+num_clouds_per_agent-1
        if (idx_meas{ob} ~= 0) % i.e. there exists a measurement
            for i = 1:K{cloud}
                % [mu_p{i}, P_p{i}] = ukfUpdate(zt, R_vv, mu_c{i}, P_c{i}, h);
                [mu_p{i, cloud}, P_p{i, cloud}] = kalmanUpdate(zt, cPoints{i, cloud}, R_vv, mu_c{i, cloud}, P_c{i, cloud}, h);
            end
        
            % Weight update
            wp{cloud} = weightUpdate(wm{cloud}, Xm_cloud{cloud}, idx{cloud}, zt, R_vv, h, 1);
        
        else
            for i = 1:K{cloud}
                wp{cloud}(i) = wm{cloud}(i);
                mu_p{i, cloud} = mu_c{i, cloud};
                P_p{i, cloud} = P_c{i, cloud};
            end
        end
            
        c_id = zeros(length(Xp_cloud{cloud}(:,1)),1);
        for i = 1:L
            [Xp_cloud{cloud}(i,:), c_id(i)] = drawFrom(wp{cloud}, mu_p(:, cloud), P_p(:, cloud), draw_from_idx+i); 
        end
        draw_from_idx = draw_from_idx+L;
    
        mu_pExp = zeros(K{cloud}, length(mu_p{1}));
        %%save("./Outside2/Xp_cloud_Outside.mat", "Xp_cloud")
        aa = zeros(3);
        % Plot the results
        figure(fig_num)
        fig_num = fig_num + 1;
        subplot(2,1,1)
        hold on;
        for k = 1:K{cloud}
            clusterPoints = Xp_cloud{cloud}(c_id == k, :);
            mu_pExp(k,:) = mu_p{k, cloud};
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
        for k = 1:K{cloud}
            clusterPoints = Xp_cloud{cloud}(c_id == k, :);
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
        for k = 1:K{cloud}
            clusterPoints = Xp_cloud{cloud}(c_id == k, :);
            mu_pExp(k,:) = mu_p{k, cloud};
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
        for k = 1:K{cloud}
            clusterPoints = Xp_cloud{cloud}(c_id == k, :);
            mu_pExp(k,:) = mu_p{k, cloud};
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
        for k = 1:K{cloud}
            clusterPoints = Xp_cloud{cloud}(c_id == k, :);
            mu_pExp(k,:) = mu_p{k, cloud};
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
        for k = 1:K{cloud}
            clusterPoints = Xp_cloud{cloud}(c_id == k, :);
            mu_pExp(k,:) = mu_p{k, cloud};
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
        for k = 1:K{cloud}
            clusterPoints = Xp_cloud{cloud}(c_id == k, :);
            mu_pExp(k,:) = mu_p{k, cloud};
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
        for k = 1:K{cloud}
            clusterPoints = Xp_cloud{cloud}(c_id == k, :);
            mu_pExp(k,:) = mu_p{k, cloud};
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
idx_crit = cell(1, num_agents);
t_end = cell(1, num_agents);
idx_end = cell(1, num_agents);
idx_start = cell(1, num_agents);
l_filt = cell(1, num_agents);
for ob = 1:num_agents
    [idx_meas{ob}, c_meas{ob}] = find(abs(hdR{ob}(:,1) - tpr{ob}) < 1e-10);
    interval{ob} = hdR{ob}(idx_meas{ob},c_meas{ob}) - hdR{ob}(idx_meas{ob}-1,c_meas{ob});

    [idx_crit{ob}, ~] = find(abs(full_ts{ob}(:,1)) >= (28*24)/time2hr, 1, 'first'); % Find the index of the last time step before a certain number of days have passed since orbit propagation
    t_end{ob} = full_ts{ob}(end,1); % First observation of new pass + one more time step
    
    tau = 0;
    [idx_end{ob}, ~] = find(abs(full_ts{ob}(:,1) - t_end{ob}) < 1e-10);
    [idx_start{ob}, ~] = find(abs(full_ts{ob}(:,1) - tpr{ob}) < 1e-10);
    
    l_filt{ob} = length(full_ts{ob}(idx_start{ob}:idx_end{ob},1))+1;
end

ent2 = cell(num_clouds);
ent1 = cell(num_clouds);
mahalanobis = cell(num_clouds);
num_cluster = cell(num_clouds);
num_particles = cell(num_clouds);
for ob = 1:num_agents
    for cloud = ob:ob+num_clouds_per_agent-1
        ent2{cloud} = zeros(l_filt{ob}+1,1);
        ent1{cloud} = zeros(l_filt{ob}+1,length(mu_c{1, cloud})); 
        mahalanobis{cloud} = zeros(l_filt{ob}+1, 1);
        num_cluster{cloud} = zeros(l_filt{ob}+1, 1);
        num_particles{cloud} = zeros(l_filt{ob}+1, 1);
        
        [ent2{cloud}(1), mahalanobis{cloud}(1), num_cluster{cloud}(1), num_particles{cloud}(1)] = getMetrics(1, X0cloud{cloud}, Xot_truth{ob}', cluster_by);%log(det(cov(X0cloud)));
        [ent2{cloud}(2), mahalanobis{cloud}(2), num_cluster{cloud}(2), num_particles{cloud}(2)] = getMetrics(K{cloud}, Xp_cloud{cloud}, Xprop_truth{ob}, cluster_by);%log(det(cov(Xp_cloud)));
        ent1{cloud}(1,:) = getDiagCov(X0cloud{cloud});
    end
end
Xp_cloudp = Xp_cloud;
%check = convertToTopo(backConvertSynodic(Xprop_truth{2}', tpr{2}, obs_lat{2}, obs_lon{2}), tpr{1}, obs_lat{1}, obs_lon{1});
%converted_cloud = zeros(size(Xp_cloudp{2}));
%for i = 1:length(Xp_cloudp{2}(:,1))
%    converted_cloud(i, :) = convertToTopo(backConvertSynodic(Xp_cloudp{2}(i, :)', tpr{2}, obs_lat{2}, obs_lon{2}), tpr{1}, obs_lat{1}, obs_lon{1});
%end
%alpha = calcAlpha({Xp_cloudp{1}, converted_cloud}, [Xp_cloudp{1}; converted_cloud], Kmax, 10, cluster_by, dist2km, vel2kms, save_loc);
%Xp_cloud = csvread("D:/PythonProjects/EDP/PGM_Git/PAR-PGM/Xp_cloud_py.csv");

% for to = tpr:interval:(t_end-1e-11) % Looping over the times of observation for easier propagation
for ts = min(idx_start{:}):(max(idx_end{:})-1)

    % Resampling Step (needlessly repeated)
    % if(idx_meas ~= 0)
    %     Xp_cloud = Xm_cloud;
    %     parfor i = 1:Lp
    %         [Xp_cloud(i,:), ~] = drawFrom(wp, mu_p, P_p); 
    %     end 
    % end
    Xm_cloud = cell(1, num_agents);
    interval = cell(1, num_agents);
    for ob=1:num_agents
        to = full_ts{ob}(ts,1);
        interval{ob} = full_ts{ob}(ts+1,1) - full_ts{ob}(ts,1);
        ent1{ob}(tau+2,:) = getDiagCov(Xp_cloudp{ob});
    
        % Propagation Step
        Xm_cloud{ob} = propagate(Xp_cloudp{ob}, to, interval{ob}, ts, "Cloud", obs_lat{ob}, obs_lon{ob});
        Xprop_truth{ob} = propagate(Xprop_truth{ob}, to, interval{ob}, ts, "Truth", obs_lat{ob}, obs_lon{ob});
    end
    %save("prop_cloud_" + num2str(ts) + ".mat", "Xm_cloud")
    % Verification Step
    tpr = to + interval{1}; % Time stamp of the prior means, weights, and covariances
    idx_meas = cell(1, num_agents);
    [idx_meas{1}, ~] = find(abs(noised_obs{1}(:,1) - tpr) < 1e-10); % Find row with time
    idx_meas{2} = idx_meas{1};
    tau = tau + 1;
    
    if(idx_meas{1} ~= 0)  
        % Split propagated cloud into position and velocity data before
        % normalization.
        % K = Kn;
        idx = cell(1, num_agents);
        idx_meas = cell(1, num_agents);
        for ob = 1:num_agents
            if (tpr >= cVal{ob})
                K{ob} = Kmax;
            else
                K{ob} = Kn;
            end
            [idx{ob}, K{ob}] = cluster(Xm_cloud{ob}, cluster_by, K{ob});
            % Verification Step
            [idx_meas{ob}, ~] = find(abs(noised_obs{ob}(:,1) - tpr) < 1e-10); % Find row with time
        end
        %{
        if(cluster_by == "Msmt")
            msmt_cloud = zeros(length(Xm_cloud{ob}), 2);
            h = @(x) [atan2(x(2),x(1)); pi/2 - acos(x(3)/sqrt(x(1)^2 + x(2)^2 + x(3)^2))]; % Nonlinear measurement model
            parfor j = 1:length(Xm_cloud)
                msmt_cloud(j,:) = h(Xm_cloud(j,:));
            end
            
            mean_msmt = mean(msmt_cloud, 1);
            std_msmt = std(msmt_cloud,0,1);
            
            norm_msmt_az = (msmt_cloud(:, 1) - mean_msmt(1))./std_msmt(1); % Normalizing the msmts
            norm_msmt_el = (msmt_cloud(:, 2) - mean_msmt(2))./std_msmt(2);
            
            Xm_norm = [norm_msmt_az, norm_msmt_el];
        end
        if cluster_by == "FullState"
            rc = Xm_cloud(:,1:3);
            mean_rc = mean(rc, 1);
            std_rc = std(rc,0,1);
            norm_rc = (rc - mean_rc)./norm(std_rc); % Normalizing the position

            vc = Xm_cloud(:,4:6);
            mean_vc = mean(vc, 1);
            std_vc = std(vc,0,1);
            norm_vc = (vc - mean_vc)./norm(std_vc); % Normalizing the velocity
            Xm_norm = [norm_rc, norm_vc];
        end
        if cluster_by == "Velocity"
            vc = Xm_cloud(:,4:6);
            mean_vc = mean(vc, 1);
            std_vc = std(vc,0,1);
            norm_vc = (vc - mean_vc)./norm(std_vc); % Normalizing the velocity
            Xm_norm = [norm_vc];
        end
        %}

        fprintf("Timestamp: %1.5f\n", tpr*time2hr);
        
        % Cluster using K-means clustering algorithm
        %{
        [idx, ~] = kmeans(Xm_norm, K);
        num_times_clustered = 1;
        while any(histcounts(idx) <= 6) % Ensure at least 6 points in each cluster
            if num_times_clustered >= 3
                K = K - 1; 
            end
            [idx, ~] = kmeans(Xm_norm, K);
            num_times_clustered = num_times_clustered + 1;
        end
        %}
        cPoints = cell(Kmax, num_agents);
        mu_c = cell(Kmax, num_agents); mu_p = mu_c;
        P_c = cell(Kmax, num_agents); P_p = P_c;
        wm = cell(1, num_agents);
        wp = cell(1, num_agents);
        mu_mExp = cell(1, num_agents);
        rto = cell(1, num_agents);
        zt = cell(1, num_agents);
        % Calculate covariances and weights for each cluster
        for ob = 1:num_agents
            wm_temp = zeros(K{ob}, 1);
            wp{ob} = wm_temp;
            for k = 1:K{ob}
                cluster_points = Xm_cloud{ob}(idx{ob} == k, :); 
                cPoints{k, ob} = cluster_points; 
                mu_c{k, ob} = mean(cluster_points, 1); % Cell of GMM means 
                if (length(cluster_points(:,1)) == 1)
                    P_c{k, ob} = zeros(length(mu_c{k, ob}));
                else
                    P_c{k, ob} = cov(cluster_points); % Cell of GMM covariances
                end
                wm_temp(k) = size(cluster_points, 1) / size(Xm_cloud{ob}, 1); % Vector of (prior) weights
            end
            wm{ob} = wm_temp;
            % Extract means
            mu_mExp{ob} = zeros(K{ob},length(mu_c{1, ob}));
            for k = 1:K{ob}
                mu_mExp{ob}(k,:) = mu_c{k, ob};
            end

            % [idx_trth, ~] = find(abs(full_ts(:,1) - tpr) < 1e-10);
            % Xprop_truth = [full_ts(idx_trth,2:4), full_vts(idx_trth,2:4)]';

            zc = noised_obs{ob}(idx_meas{ob},2:4)'; % Presumption: An observation occurs at this time step
            xto = zc(1)*cos(zc(2))*cos(zc(3)); 
            yto = zc(1)*sin(zc(2))*cos(zc(3)); 
            zto = zc(1)*sin(zc(3)); 
            rto{ob} = [xto, yto, zto];
        
            legend_string = {};
            parfor k = 1:K{ob}
                R_vv = [R_f*partial_ts{ob}(idx_meas{ob},2), 0, 0; 0 theta_f*pi/648000, 0; 0, 0, theta_f*pi/648000].^2;
                Hxk = linHx(mu_c{k, ob}); % Linearize about prior mean component
                legend_string{k} = sprintf('Distribution %i',k);
                % legend_string{K+k} = sprintf('\\omega =  %1.4f, l = %1.4d', wm(k), gaussProb(zc, h(mu_c{k}), Hxk*P_c{k}*Hxk' + R_vv));
            end
            % legend_string{K+1} = "Centroids";
            legend_string{K{ob}+1} = "Truth";
            if(1) % Use for all time steps
                % legend_string{K+1} = "Centroids";
                legend_string{K{ob}+1} = "Truth";
        
                mu_mat = cell2mat(mu_c(:, ob));
                P_mat = cat(3, P_c{:, ob});
        
                
                f = figure('visible','off','Position', get(0,'ScreenSize'));
                f.WindowState = 'maximized';
        
                subplot(2,3,1)
                plot_dims = [1,2];
                mu_marg = mu_mat(:, plot_dims);
                P_marg = P_mat(plot_dims, plot_dims, :);
                
                [X1, X2] = meshgrid(linspace(min(Xm_cloud{ob}(:,plot_dims(1))), max(Xm_cloud{ob}(:,plot_dims(1))), 100), ...
                                linspace(min(Xm_cloud{ob}(:,plot_dims(2))), max(Xm_cloud{ob}(:,plot_dims(2))), 100));
                X_grid = [X1(:) X2(:)];
        
                Z_cell = cell(K{ob},1); contours_cell = cell(K{ob},1); 
                
                parfor k = 1:K{ob}
                    Z = zeros(size(X1));
                    if det(P_marg(:, :, k)) == 0
                        P_marg(:,:,k) = P_marg(:,:,k) + 1e-12*eye(size(P_marg(:,:,k)));
                    end
                    for i = 1:size(X_grid, 1)
                        Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
                    end
                    Z = reshape(Z, size(X1));
                    Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
                    contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
                end 
        
                hold on;
                for k = 1:K{ob}
                    contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
                end
                % Overlay scatter points
                %scatter(dist2km*Xm_cloud(:, plot_dims(1)), dist2km*Xm_cloud(:, plot_dims(2)), ...
                %    'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    
                % Overlay a special marker for truth
                scatter(dist2km*Xprop_truth{ob}(1), dist2km*Xprop_truth{ob}(2), 200, 'k', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'Marker', 'x');
                % hold on;
                % plot(rto(1), rto(2), 'o', 'MarkerSize', 10, 'LineWidth', 3);
                title('X-Y');
                xlabel('X (km.)');
                ylabel('Y (km.)');
                legend(legend_string);
                hold off;
        
                subplot(2,3,2)
                plot_dims = [1,3];
                mu_marg = mu_mat(:, plot_dims);
                P_marg = P_mat(plot_dims, plot_dims, :);
                
                [X1, X2] = meshgrid(linspace(min(Xm_cloud{ob}(:,plot_dims(1))), max(Xm_cloud{ob}(:,plot_dims(1))), 100), ...
                                linspace(min(Xm_cloud{ob}(:,plot_dims(2))), max(Xm_cloud{ob}(:,plot_dims(2))), 100));
                X_grid = [X1(:) X2(:)];
        
                Z_cell = cell(K{ob},1); contours_cell = cell(K{ob},1); 
                
                parfor k = 1:K{ob}
                    Z = zeros(size(X1));
                    if det(P_marg(:, :, k)) == 0
                        P_marg(:,:,k) = P_marg(:,:,k) + 1e-12*eye(size(P_marg(:,:,k)));
                    end
                    for i = 1:size(X_grid, 1)
                        Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
                    end
                    Z = reshape(Z, size(X1));
                    Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
                    contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
                end 
                
                hold on;
                for k = 1:K{ob}
                    contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
                end    
                % Overlay scatter points
                %scatter(dist2km*Xm_cloud(:, plot_dims(1)), dist2km*Xm_cloud(:, plot_dims(2)), ...
                %    'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    
                scatter(dist2km*Xprop_truth{ob}(1), dist2km*Xprop_truth{ob}(3), 200, 'k', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'Marker', 'x');
                % hold on;
                % plot(rto(1), rto(3), 'o', 'MarkerSize', 10, 'LineWidth', 3);
                title('X-Z');
                xlabel('X (km.)');
                ylabel('Z (km.)');
                legend(legend_string);
                hold off;
        
                subplot(2,3,3)
                plot_dims = [2,3];
                mu_marg = mu_mat(:, plot_dims);
                P_marg = P_mat(plot_dims, plot_dims, :);
                
                [X1, X2] = meshgrid(linspace(min(Xm_cloud{ob}(:,plot_dims(1))), max(Xm_cloud{ob}(:,plot_dims(1))), 100), ...
                                linspace(min(Xm_cloud{ob}(:,plot_dims(2))), max(Xm_cloud{ob}(:,plot_dims(2))), 100));
                X_grid = [X1(:) X2(:)];
        
                Z_cell = cell(K{ob},1); contours_cell = cell(K{ob},1); 
                
                parfor k = 1:K{ob}
                    Z = zeros(size(X1));
                    if det(P_marg(:, :, k)) == 0
                        P_marg(:,:,k) = P_marg(:,:,k) + 1e-12*eye(size(P_marg(:,:,k)));
                    end
                    for i = 1:size(X_grid, 1)
                        Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
                    end
                    Z = reshape(Z, size(X1));
                    Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
                    contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
                end 
                
                hold on;
                for k = 1:K{ob}
                    contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
                end    
                % Overlay scatter points
                %scatter(dist2km*Xm_cloud(:, plot_dims(1)), dist2km*Xm_cloud(:, plot_dims(2)), ...
                %    'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    
                scatter(dist2km*Xprop_truth{ob}(2), dist2km*Xprop_truth{ob}(3), 200, 'k', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'Marker', 'x');
                % hold on;
                % plot(rto(2), rto(3), 'o', 'MarkerSize', 10, 'LineWidth', 3);
                title('X-Z');
                xlabel('X (km.)');
                ylabel('Z (km.)');
                legend(legend_string);
                hold off;
        
                subplot(2,3,4)
                plot_dims = [4,5];
                mu_marg = mu_mat(:, plot_dims);
                P_marg = P_mat(plot_dims, plot_dims, :);
                
                [X1, X2] = meshgrid(linspace(min(Xm_cloud{ob}(:,plot_dims(1))), max(Xm_cloud{ob}(:,plot_dims(1))), 100), ...
                                linspace(min(Xm_cloud{ob}(:,plot_dims(2))), max(Xm_cloud{ob}(:,plot_dims(2))), 100));
                X_grid = [X1(:) X2(:)];
        
                Z_cell = cell(K{ob},1); contours_cell = cell(K{ob},1); 
                
                parfor k = 1:K{ob}
                    Z = zeros(size(X1));
                    if det(P_marg(:, :, k)) == 0
                        P_marg(:,:,k) = P_marg(:,:,k) + 1e-12*eye(size(P_marg(:,:,k)));
                    end
                    for i = 1:size(X_grid, 1)
                        Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
                    end
                    Z = reshape(Z, size(X1));
                    Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
                    contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
                end 
                
                hold on;
                for k = 1:K{ob}
                    contour(vel2kms*X1, vel2kms*X2, vel2kms*Z_cell{k}, vel2kms*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
                end    
                % Overlay scatter points
                %scatter(vel2kms*Xm_cloud(:, plot_dims(1)), vel2kms*Xm_cloud(:, plot_dims(2)), ...
                %    'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    
                scatter(vel2kms*Xprop_truth{ob}(4), vel2kms*Xprop_truth{ob}(5), ...
                    200, 'k', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'Marker', 'x');
                title('Xdot-Ydot');
                xlabel('Xdot (km/s)');
                ylabel('Ydot (km/s)');
                legend(legend_string);
                hold off;
        
                subplot(2,3,5)
                plot_dims = [4,6];
                mu_marg = mu_mat(:, plot_dims);
                P_marg = P_mat(plot_dims, plot_dims, :);
                
                [X1, X2] = meshgrid(linspace(min(Xm_cloud{ob}(:,plot_dims(1))), max(Xm_cloud{ob}(:,plot_dims(1))), 100), ...
                                linspace(min(Xm_cloud{ob}(:,plot_dims(2))), max(Xm_cloud{ob}(:,plot_dims(2))), 100));
                X_grid = [X1(:) X2(:)];
        
                Z_cell = cell(K{ob},1); contours_cell = cell(K{ob},1); 
                
                parfor k = 1:K{ob}
                    Z = zeros(size(X1));
                    if det(P_marg(:, :, k)) == 0
                        P_marg(:,:,k) = P_marg(:,:,k) + 1e-12*eye(size(P_marg(:,:,k)));
                    end
                    for i = 1:size(X_grid, 1)
                        Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
                    end
                    Z = reshape(Z, size(X1));
                    Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
                    contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
                end 
                
                hold on;
                for k = 1:K{ob}
                    contour(vel2kms*X1, vel2kms*X2, vel2kms*Z_cell{k}, vel2kms*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
                end    
                %scatter(vel2kms*Xm_cloud(:, plot_dims(1)), vel2kms*Xm_cloud(:, plot_dims(2)), ...
                %    'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    
                scatter(vel2kms*Xprop_truth{ob}(4), vel2kms*Xprop_truth{ob}(6), ...
                    200, 'k', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'Marker', 'x');
                title('Xdot-Zdot');
                xlabel('Xdot (km/s)');
                ylabel('Zdot (km/s)');
                legend(legend_string);
                hold off;
        
                subplot(2,3,6)
                plot_dims = [5,6];
                mu_marg = mu_mat(:, plot_dims);
                P_marg = P_mat(plot_dims, plot_dims, :);
                
                [X1, X2] = meshgrid(linspace(min(Xm_cloud{ob}(:,plot_dims(1))), max(Xm_cloud{ob}(:,plot_dims(1))), 100), ...
                                linspace(min(Xm_cloud{ob}(:,plot_dims(2))), max(Xm_cloud{ob}(:,plot_dims(2))), 100));
                X_grid = [X1(:) X2(:)];
        
                Z_cell = cell(K{ob},1); contours_cell = cell(K{ob},1); 
                
                parfor k = 1:K{ob}
                    Z = zeros(size(X1));
                    if det(P_marg(:, :, k)) == 0
                        P_marg(:,:,k) = P_marg(:,:,k) + 1e-12*eye(size(P_marg(:,:,k)));
                    end
                    for i = 1:size(X_grid, 1)
                        Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
                    end
                    Z = reshape(Z, size(X1));
                    Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
                    contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
                end 
                
                hold on;
                for k = 1:K{ob}
                    contour(vel2kms*X1, vel2kms*X2, vel2kms*Z_cell{k}, vel2kms*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
                end    
                % Overlay scatter points
                %scatter(vel2kms*Xm_cloud(:, plot_dims(1)), vel2kms*Xm_cloud(:, plot_dims(2)), ...
                %    'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    
                scatter(vel2kms*Xprop_truth{ob}(5), vel2kms*Xprop_truth{ob}(6), ...
                    200, 'k', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'Marker', 'x');
                title('Ydot-Zdot');
                xlabel('Ydot (km/s)');
                ylabel('Zdot (km/s)');
                legend(legend_string);
                hold off;
        
                sgt = sprintf('Timestep: %3.4f Hours (Prior) Obs: %i', [tpr*time2hr, ob]);
                sgtitle(sgt);
        
                sg = sprintf(save_loc + '/Observer' + num2str(ob) + '/Timestep_%i_1A.png', tau);
                % sg = sprintf('./Simulations/Different Orbit Simulations/Timestep_%i_1A.png', tau);
                saveas(f, sg, 'png');
                close(f);
                %}
    
                f = figure('visible','off','Position', get(0,'ScreenSize'));
                f.WindowState = 'maximized';
        
                legend_string = "Truth";
        
                subplot(2,3,1)
                % gscatter(Xm_cloud(:,1), Xm_cloud(:,2), idx);
                % hold on;
                % plot(mu_mExp(:,1), mu_mExp(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
                % hold on;
                mu_mExp_temp = zeros(K{ob},length(mu_c{1, ob}));
                parfor k = 1:K{ob}
                    cPoints{k, ob} = Xm_cloud{ob}(idx{ob} == k, :);
                    mu_mExp_temp(k,:) = mu_c{k, ob};
                end
                mu_mExp{ob} = mu_mExp_temp;
                hold on; 
                for k = 1:K{ob}
                    scatter(dist2km*cPoints{k, ob}(:,1), dist2km*cPoints{k, ob}(:,2), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
                end
        
                plot(dist2km*Xprop_truth{ob}(1), dist2km*Xprop_truth{ob}(2), 'kx', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
                % hold on;
                % plot(rto(1), rto(2), 'o', 'MarkerSize', 10, 'LineWidth', 3);
                title('X-Y');
                xlabel('X (km.)');
                ylabel('Y (km.)');
                legend(legend_string);
                hold off;
                
                subplot(2,3,2)
                mu_mExp_temp = zeros(K{ob},length(mu_c{1, ob}));
                parfor k = 1:K{ob}
                    cPoints{k, ob} = Xm_cloud{ob}(idx{ob} == k, :);
                    mu_mExp_temp(k,:) = mu_c{k, ob};
                end
                mu_mExp{ob} = mu_mExp_temp;
                hold on; 
                for k = 1:K{ob}
                    scatter(dist2km*cPoints{k, ob}(:,1), dist2km*cPoints{k, ob}(:,3), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
                end
                plot(dist2km*Xprop_truth{ob}(1), dist2km*Xprop_truth{ob}(3), 'kx', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
                % hold on;
                % plot(rto(1), rto(3), 'o', 'MarkerSize', 10, 'LineWidth', 3);
                title('X-Z');
                xlabel('X (km.)');
                ylabel('Z (km.)');
                legend(legend_string);
                hold off;
                
                subplot(2,3,3)
                % gscatter(Xm_cloud(:,2), Xm_cloud(:,3), idx);
                % hold on;
                % plot(mu_mExp(:,2), mu_mExp(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
                % hold on;
                mu_mExp_temp = zeros(K{ob},length(mu_c{1, ob}));
                parfor k = 1:K{ob}
                    cPoints{k, ob} = Xm_cloud{ob}(idx{ob} == k, :);
                    mu_mExp_temp(k,:) = mu_c{k, ob};
                end
                mu_mExp{ob} = mu_mExp_temp;
                hold on; 
                for k = 1:K{ob}
                    scatter(dist2km*cPoints{k, ob}(:,2), dist2km*cPoints{k, ob}(:,3), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
                end
                plot(dist2km*Xprop_truth{ob}(2), dist2km*Xprop_truth{ob}(3), 'kx', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
                % hold on;
                % plot(rto(2), rto(3), 'o', 'MarkerSize', 10, 'LineWidth', 3);
                title('Y-Z');
                xlabel('Y (km.)');
                ylabel('Z (km.)');
                legend(legend_string);
                hold off;
                
                subplot(2,3,4)
                mu_mExp_temp = zeros(K{ob},length(mu_c{1, ob}));
                parfor k = 1:K{ob}
                    cPoints{k, ob} = Xm_cloud{ob}(idx{ob} == k, :);
                    mu_mExp_temp(k,:) = mu_c{k, ob};
                end
                mu_mExp{ob} = mu_mExp_temp;
                hold on; 
                for k = 1:K{ob}
                    scatter(vel2kms*cPoints{k, ob}(:,4), vel2kms*cPoints{k, ob}(:,5), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
                end
                plot(vel2kms*Xprop_truth{ob}(4), vel2kms*Xprop_truth{ob}(5), 'kx', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
                title('Xdot-Ydot');
                xlabel('Xdot (km/s)');
                ylabel('Ydot (km/s)');
                legend(legend_string);
                hold off;
                
                subplot(2,3,5)
                mu_mExp_temp = zeros(K{ob},length(mu_c{1, ob}));
                parfor k = 1:K{ob}
                    cPoints{k, ob} = Xm_cloud{ob}(idx{ob} == k, :);
                    mu_mExp_temp(k,:) = mu_c{k, ob};
                end
                mu_mExp{ob} = mu_mExp_temp;
                hold on; 
                for k = 1:K{ob}
                    scatter(vel2kms*cPoints{k, ob}(:,4), vel2kms*cPoints{k, ob}(:,6), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
                end
                plot(vel2kms*Xprop_truth{ob}(4), vel2kms*Xprop_truth{ob}(6), 'kx', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
                title('Xdot-Zdot');
                xlabel('Xdot (km/s)');
                ylabel('Zdot (km/s)');
                legend(legend_string);
                hold off;
                
                subplot(2,3,6)
        
                mu_mExp_temp = zeros(K{ob},length(mu_c{1, ob}));
                parfor k = 1:K{ob}
                    cPoints{k, ob} = Xm_cloud{ob}(idx{ob} == k, :);
                    mu_mExp_temp(k,:) = mu_c{k, ob};
                end
                mu_mExp{ob} = mu_mExp_temp;
                hold on; 
                for k = 1:K{ob}
                    scatter(vel2kms*cPoints{k, ob}(:,5), vel2kms*cPoints{k, ob}(:,6), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
                end
                plot(vel2kms*Xprop_truth{ob}(5), vel2kms*Xprop_truth{ob}(6), 'kx', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
                title('Ydot-Zdot');
                xlabel('Ydot (km/s)');
                ylabel('Zdot (km/s)');
                legend(legend_string);
                hold off;
        
                sgt = sprintf('Timestep: %3.4f Hours (Prior) Ob: %i', [tpr*time2hr, ob]);
                sgtitle(sgt);
        
                sg = sprintf(save_loc + '/Observer' + num2str(ob) + '/Timestep_%i_1B.png', tau);
                % sg = sprintf('./Simulations/Different Orbit Simulations/Timestep_%i_1B.png', tau);
                saveas(f, sg, 'png');
                close(f);

                f = figure('visible','off','Position', get(0,'ScreenSize'));
                f.WindowState = 'maximized';
                %legend_weight{K+1} = "Truth";
                R_weight = [theta_f*pi/648000, 0; 0, theta_f*pi/648000].^2;
                h = @(x) [atan2(x(2),x(1)); pi/2 - acos(x(3)/sqrt(x(1)^2 + x(2)^2 + x(3)^2))]; % Nonlinear measurement model
                hold on;
                scatter_handles = gobjects(K{ob},1);
                zt{ob} = getNoisyMeas(Xprop_truth{ob}, R_weight, h, ts);
                %%save("./clusters/cPoints_" + num2str(ts) + ".mat", "cPoints")
                for k = 1:K{ob}
                    Zmcloud = zeros(length(cPoints{k, ob}(:,1)), length(zt{ob}));
                    for i = 1:length(Zmcloud(:,1))
                        Zmcloud(i,:) = h(cPoints{k, ob}(i,:))';
                    end
                    
                    likeli = mvnpdf(zt{ob}', mean(Zmcloud,1), cov(Zmcloud) + R_weight);
                    %legend_weight{k} = sprintf('k: %i; w: %.3f, l: %.3f', [k, wm(k), likeli]);
      
                    scatter_handles(k) = scatter(180/pi*Zmcloud(:,1), 180/pi*Zmcloud(:,2), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off', 'DisplayName', sprintf('k: %i; w: %.3f, l: %.2e', [k, wm{ob}(k), likeli]));
                end
                Ztruth = h(Xprop_truth{ob})';
                truth_handle = plot(180/pi*Ztruth(1), 180/pi*Ztruth(2), 'kx', ... 
                'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', 'Truth');
                zt_handle = plot(180/pi*zt{ob}(1), 180/pi*zt{ob}(2), 'ko', ... 
                'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', 'Noisy Truth');
                title(sprintf('AZ-EL Ob: %i', ob))
                xlabel('Azimuth Angle (deg)')
                ylabel('Elevation Angle (deg)')
                
                legend([scatter_handles; truth_handle;zt_handle], 'Location', 'northeastoutside'); 
                sg = sprintf(save_loc + '/Observer' + num2str(ob) + '/Timestep_%i_1C.png', tau);
                % sg = sprintf('./Simulations/Different Orbit Simulations/Timestep_%i_1B.png', tau);
                saveas(f, sg, 'png');
                close(f);
            end
        end
  
        if(abs(to - (t_end{1}-interval{1})) < 1e-10) % At final time step possible
            % Save the a priori estimate particle cloud
            save('aPriori.mat', 'Xm_cloud');
            for ob = 1:num_agents
                % Extract means
                for k = 1:K{ob}
                    mu_mExp{ob}(k,:) = mu_c{k, ob};
                end
        
                % Show where observation lies (position only)
                if(idx_meas{ob} ~= 0)
                    zc = noised_obs(idx_meas{ob},2:4)'; % Presumption: An observation occurs at this time step
                    xto = zc(1)*cos(zc(2))*cos(zc(3)); 
                    yto = zc(1)*sin(zc(2))*cos(zc(3)); 
                    zto = zc(1)*sin(zc(3)); 
                    rto{ob} = [xto, yto, zto];
                end
        
                % Plot planar projections
                figure(fig_num)
                fig_num = fig_num + 1;
                subplot(2,3,1)
                gscatter(dist2km*Xm_cloud{ob}(:,1), dist2km*Xm_cloud{ob}(:,2), idx{ob});
                hold on;
                plot(dist2km*mu_mExp{ob}(:,1), dist2km*mu_mExp{ob}(:,2), '+', 'MarkerSize', 10, 'LineWidth', 3);
                hold on;
                plot(dist2km*Xprop_truth{ob}(1), dist2km*Xprop_truth{ob}(2), 'kx','MarkerSize', 15, 'LineWidth', 3);
                title('X-Y');
                xlabel('X (km.)');
                ylabel('Y (km.)');
                legend(legend_string);
                hold off;
                
                subplot(2,3,2)
                gscatter(dist2km*Xm_cloud{ob}(:,1), dist2km*Xm_cloud{ob}(:,3), idx{ob});
                hold on;
                plot(dist2km*mu_mExp{ob}(:,1), dist2km*mu_mExp{ob}(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
                hold on;
                plot(dist2km*Xprop_truth{ob}(1), dist2km*Xprop_truth{ob}(3), 'kx','MarkerSize', 15, 'LineWidth', 3);
                title('X-Z');
                xlabel('X (km.)');
                ylabel('Z (km.)');
                legend(legend_string);
                hold off;
                
                subplot(2,3,3)
                gscatter(dist2km*Xm_cloud{ob}(:,2), dist2km*Xm_cloud{ob}(:,3), idx{ob});
                hold on;
                plot(dist2km*mu_mExp{ob}(:,2), dist2km*mu_mExp{ob}(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
                hold on;
                plot(dist2km*Xprop_truth{ob}(2), dist2km*Xprop_truth{ob}(3), 'kx','MarkerSize', 15, 'LineWidth', 3);
                title('Y-Z');
                xlabel('Y (km.)');
                ylabel('Z (km.)');
                legend(legend_string);
                hold off;
                
                subplot(2,3,4)
                gscatter(vel2kms*Xm_cloud{ob}(:,4), vel2kms*Xm_cloud{ob}(:,5), idx{ob});
                hold on;
                plot(vel2kms*mu_mExp{ob}(:,4), vel2kms*mu_mExp{ob}(:,5), '+', 'MarkerSize', 10, 'LineWidth', 3);
                hold on;
                plot(vel2kms*Xprop_truth{ob}(4), vel2kms*Xprop_truth{ob}(5), 'kx','MarkerSize', 15, 'LineWidth', 3)
                title('Xdot-Ydot');
                xlabel('Xdot (km/s)');
                ylabel('Ydot (km/s)');
                legend(legend_string);
                hold off;
                
                subplot(2,3,5)
                gscatter(vel2kms*Xm_cloud{ob}(:,4), vel2kms*Xm_cloud{ob}(:,6), idx{ob});
                hold on;
                plot(vel2kms*mu_mExp{ob}(:,4), vel2kms*mu_mExp{ob}(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3);
                hold on;
                plot(vel2kms*Xprop_truth{ob}(4), vel2kms*Xprop_truth{ob}(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
                title('Xdot-Zdot');
                xlabel('Xdot (km/s)');
                ylabel('Zdot (km/s)');
                legend(legend_string);
                hold off;
                
                subplot(2,3,6)
                gscatter(vel2kms*Xm_cloud{ob}(:,5), vel2kms*Xm_cloud{ob}(:,6), idx{ob});
                hold on;
                plot(vel2kms*mu_mExp{ob}(:,5), vel2kms*mu_mExp{ob}(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3);
                hold on;
                plot(vel2kms*Xprop_truth{ob}(5), vel2kms*Xprop_truth{ob}(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
                title('Ydot-Zdot');
                xlabel('Ydot (km/s)');
                ylabel('Zdot (km/s)');
                legend(legend_string);
                hold off;
                savefig(gcf, save_loc + '/Observer' + num2str(ob) + '/postClusteringDistribution.fig');
            end
        end
        for ob = 1:num_agents
            % Update Step
            R_vv = [theta_f*pi/648000, 0; 0, theta_f*pi/648000].^2;
            % Hxk = linHx(mu_c{i}); % Linearize about prior mean component
            %h = @(x) [atan2(x(2),x(1)); pi/2 - acos(x(3)/sqrt(x(1)^2 + x(2)^2 + x(3)^2))]; % Nonlinear measurement model
            %zt = getNoisyMeas(Xprop_truth, R_vv, h, ts);
    
            for i = 1:K{ob}
                % [mu_p{i}, P_p{i}] = ukfUpdate(zt, R_vv, mu_c{i}, P_c{i}, h);
                [mu_p{i, ob}, P_p{i, ob}] = kalmanUpdate(zt{ob}, cPoints{i, ob}, R_vv, mu_c{i, ob}, P_c{i, ob}, h);
                P_p{i, ob} = (P_p{i, ob} + P_p{i, ob}')/2;
                %P_p{i, ob} = P_c{i, ob};
                %mu_p{i, ob} = mu_c{i, ob};
            end
    
            % Weight update
            wp{ob} = weightUpdate(wm{ob}, Xm_cloud{ob}, idx{ob}, zt{ob}, R_vv, h, ts);
        end
    else
        fprintf("Timestamp: %1.5f\n", tpr*time2hr);
        mu_p = cell(1, num_agents); 
        P_p = cell(1, num_agents); 
        wm = cell(1, num_agents);
        cPoints = cell(1, num_agents);
        for ob = 1:num_agents
            wm{ob} = zeros(1, 1);
            if(tpr*time2hr>700)
                length(Xm_cloud{ob})
            end
            Xp_cloud{ob} = Xm_cloud{ob}; cPoints{1, ob} = Xp_cloud{ob};
            wp{ob} = [1];
            mu_p{1, ob} = mean(Xp_cloud{ob});
            P_p{1, ob} = cov(Xp_cloud{ob});
        end
        if(combine == 1 || tpr*time2hr >= time_of_fusion)
            num_clouds = 4;
            converted_cloud = zeros(size(Xp_cloud{2}));
            for i = 1:length(Xp_cloud{2}(:,1))
                converted_cloud(i, :) = convertToTopo(backConvertSynodic(Xp_cloud{2}(i, :)', tpr, obs_lat{2}, obs_lon{2}), tpr, obs_lat{1}, obs_lon{1});
            end
            % Xp_cloud output should be Agent 1 Hellinger, Agent 2 Hellinger,
            % Agent 1 Entropy, Agent 2 Entropy
            if(entropy_choice == "TallestPeaks")
                % Calc for all fusion types
                [p3_simple, p3_tallest_peaks, ~, ~, fusion_bin_edges] = fusionMethods(Xp_cloud{1}, converted_cloud);
                Xp_cloud_combined{1} = sampleFromFusedPDF(p3_simple, fusion_bin_edges{1}, Lp);
                Xp_cloud_combined{2} = sampleFromFusedPDF(p3_tallest_peaks, x, y, z, Lp);
                Xp_cloud = cell(1, num_clouds);
                Xp_cloud{1} = Xp_cloud_combined{1};
                Xp_cloud{3} = Xp_cloud_combined{2};
                % % TODO: Check whether Xp_cloud_combined is correct shape
                % % in next line
                for i = 1:length(Xp_cloud_combined{2}(:,1))
                    Xp_cloud{2} = convertToTopo(backConvertSynodic(Xp_cloud_combined{1}(i, :)', tpr, obs_lat{1}, obs_lon{1}), tpr, obs_lat{2}, obs_lon{2});
                    Xp_cloud{4} = convertToTopo(backConvertSynodic(Xp_cloud_combined{2}(i, :)', tpr, obs_lat{1}, obs_lon{1}), tpr, obs_lat{2}, obs_lon{2});
                end
            else
                [p3_simple, ~, weight_update_p3_1, weight_update_p3_2, fusion_bin_edges] = fusionMethods(Xp_cloud{1}, converted_cloud);
                Xp_cloud_combined{1} = sampleFromFusedPDF(p3_simple, fusion_bin_edges{[1, 3, 4]}, Lp);
                Xp_cloud_combined{2} = sampleFromFusedPDF(weight_update_p3_1, x, y, z, Lp);
                Xp_cloud_combined{3} = sampleFromFusedPDF(weight_update_p3_2, x, y, z, Lp);
                Xp_cloud{1} = Xp_cloud_combined{1};
                Xp_cloud{3} = Xp_cloud_combined{2};
                for i = 1:length(Xp_cloud_combined{2}(:,1))
                    Xp_cloud{2} = convertToTopo(backConvertSynodic(Xp_cloud_combined{1}(i, :)', tpr, obs_lat{1}, obs_lon{1}), tpr, obs_lat{2}, obs_lon{2});
                    Xp_cloud{4} = convertToTopo(backConvertSynodic(Xp_cloud_combined{3}(i, :)', tpr, obs_lat{1}, obs_lon{1}), tpr, obs_lat{2}, obs_lon{2});
                end
            end
            
            %[alpha, mu_comb, P_comb, w_comb] = calcAlpha({Xp_cloud{1}, converted_cloud}, [Xp_cloud{1}; converted_cloud], Kmax, 1000, cluster_by, dist2km, vel2kms, save_loc);
            combine = 0;
        end
    end

    % Resampling
    c_id = cell(1, num_agents);
    for ob = 1:num_agents
        if (idx_meas{ob} ~= 0)
            % K = Kn;
            Xp_cloudp_temp = zeros(Lp, length(Xprop_truth{ob}));
            c_id_temp = zeros(Lp,1);
            for i = 1:Lp
                [Xp_cloudp_temp(i,:), c_id_temp(i)] = drawFrom(wp{ob}, mu_p(:, ob), P_p(:, ob), draw_from_idx+i); 
            end
            %Xp_cloudp_temp = Xm_cloud{ob};
            %c_id_temp = idx{ob};
            draw_from_idx = draw_from_idx+Lp;
            Xp_cloudp{ob} = Xp_cloudp_temp;
            c_id{ob} = c_id_temp;
        else
            K{ob} = 1;
            Xp_cloudp{ob} = Xm_cloud{ob}; c_id{ob} = ones(length(Xp_cloudp{ob}(:,1)),1);
        end
    end

    if(1)
        % [idx_trth, ~] = find(abs(full_ts(:,1) - tpr) < 1e-10);
        % Xprop_truth = [full_ts(idx_trth,2:4), full_vts(idx_trth,2:4)];

        % Extract means
        %mu_pExp = cell(1, num_agents);
        for ob = 1:num_agents
            mu_pExp = zeros(K{ob}, length(mu_p{1}));
            for k = 1:K{ob}
                mu_pExp(k,:) = mu_p{k, ob};
            end
        
            legend_string = {};
            parfor k = 1:K{ob}
                legend_string{k} = sprintf('Contour %i', k);
                % legend_string{K+k} = sprintf('\\omega = %1.4f', wp(k));
            end
            % legend_string{K+1} = "Centroids";
            legend_string{K{ob}+1} = "Truth";
        
            mu_mat = mu_pExp;
            P_mat = cat(3, P_p{:, ob});
        
            
            f = figure('visible','off','Position', get(0,'ScreenSize'));
            f.WindowState = 'maximized';
        
            subplot(2,3,1)
            plot_dims = [1,2];
            mu_marg = mu_mat(:, plot_dims);
            P_marg = P_mat(plot_dims, plot_dims, :);
        
            [X1, X2] = meshgrid(linspace(min(Xp_cloudp{ob}(:,plot_dims(1))), max(Xp_cloudp{ob}(:,plot_dims(1))), 100), ...
                                linspace(min(Xp_cloudp{ob}(:,plot_dims(2))), max(Xp_cloudp{ob}(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
        
            Z_cell = cell(K{ob},1); contours_cell = cell(K{ob},1); 
            
            parfor k = 1:K{ob}
                Z = zeros(size(X1));
                if det(P_marg(:, :, k)) == 0
                        P_marg(:,:,k) = P_marg(:,:,k) + 1e-12*eye(size(P_marg(:,:,k)));
                end
                for i = 1:size(X_grid, 1)
                    Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
                end
                Z = reshape(Z, size(X1));
                Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
                contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
            end 
            
            hold on;
            for k = 1:K{ob}
                contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
            end
            % Overlay scatter points
            %scatter(dist2km*Xp_cloudp(:, plot_dims(1)), dist2km*Xp_cloudp(:, plot_dims(2)), ...
            %        'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    
            scatter(dist2km*Xprop_truth{ob}(1), dist2km*Xprop_truth{ob}(2), ...
                    200, 'k', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'Marker', 'x');
            title('X-Y');
            xlabel('X (km.)');
            ylabel('Y (km.)');
            legend(legend_string);
            hold off;
        
            subplot(2,3,2)
            plot_dims = [1,3];
            mu_marg = mu_mat(:, plot_dims);
            P_marg = P_mat(plot_dims, plot_dims, :);
        
            [X1, X2] = meshgrid(linspace(min(Xp_cloudp{ob}(:,plot_dims(1))), max(Xp_cloudp{ob}(:,plot_dims(1))), 100), ...
                                linspace(min(Xp_cloudp{ob}(:,plot_dims(2))), max(Xp_cloudp{ob}(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
        
            Z_cell = cell(K{ob},1); contours_cell = cell(K{ob},1); 
            
            parfor k = 1:K{ob}
                Z = zeros(size(X1));
                if det(P_marg(:, :, k)) == 0
                        P_marg(:,:,k) = P_marg(:,:,k) + 1e-12*eye(size(P_marg(:,:,k)));
                end
                for i = 1:size(X_grid, 1)
                    Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
                end
                Z = reshape(Z, size(X1));
                Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
                contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
            end 
            
            hold on;
            for k = 1:K{ob}
                contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
            end
            % Overlay scatter points
            %scatter(dist2km*Xp_cloudp(:, plot_dims(1)), dist2km*Xp_cloudp(:, plot_dims(2)), ...
            %        'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    
            scatter(dist2km*Xprop_truth{ob}(1), dist2km*Xprop_truth{ob}(3), ...
                    200, 'k', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'Marker', 'x');
            title('X-Z');
            xlabel('X (km.)');
            ylabel('Z (km.)');
            legend(legend_string);
            hold off;
        
            subplot(2,3,3)
            plot_dims = [2,3];
            mu_marg = mu_mat(:, plot_dims);
            P_marg = P_mat(plot_dims, plot_dims, :);
        
            [X1, X2] = meshgrid(linspace(min(Xp_cloudp{ob}(:,plot_dims(1))), max(Xp_cloudp{ob}(:,plot_dims(1))), 100), ...
                                linspace(min(Xp_cloudp{ob}(:,plot_dims(2))), max(Xp_cloudp{ob}(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
        
            Z_cell = cell(K{ob},1); contours_cell = cell(K{ob},1); 
            
            parfor k = 1:K{ob}
                Z = zeros(size(X1));
                if det(P_marg(:, :, k)) == 0
                        P_marg(:,:,k) = P_marg(:,:,k) + 1e-12*eye(size(P_marg(:,:,k)));
                end
                for i = 1:size(X_grid, 1)
                    Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
                end
                Z = reshape(Z, size(X1));
                Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
                contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
            end 
            
            hold on;
            for k = 1:K{ob}
                contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
            end
            % Overlay scatter points
            %scatter(dist2km*Xp_cloudp(:, plot_dims(1)), dist2km*Xp_cloudp(:, plot_dims(2)), ...
            %        'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    
            scatter(dist2km*Xprop_truth{ob}(2), dist2km*Xprop_truth{ob}(3), ...
                    200, 'k', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'Marker', 'x');
            title('Y-Z');
            xlabel('Y (km.)');
            ylabel('Z (km.)');
            legend(legend_string);
            hold off;
        
            subplot(2,3,4)
            plot_dims = [4,5];
            mu_marg = mu_mat(:, plot_dims);
            P_marg = P_mat(plot_dims, plot_dims, :);
        
            [X1, X2] = meshgrid(linspace(min(Xp_cloudp{ob}(:,plot_dims(1))), max(Xp_cloudp{ob}(:,plot_dims(1))), 100), ...
                                linspace(min(Xp_cloudp{ob}(:,plot_dims(2))), max(Xp_cloudp{ob}(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
        
            Z_cell = cell(K{ob},1); contours_cell = cell(K{ob},1); 
            
            parfor k = 1:K{ob}
                Z = zeros(size(X1));
                if det(P_marg(:, :, k)) == 0
                        P_marg(:,:,k) = P_marg(:,:,k) + 1e-12*eye(size(P_marg(:,:,k)));
                end
                for i = 1:size(X_grid, 1)
                    Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
                end
                Z = reshape(Z, size(X1));
                Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
                contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
            end 
            
            hold on;
            for k = 1:K{ob}
                contour(vel2kms*X1, vel2kms*X2, vel2kms*Z_cell{k}, vel2kms*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
            end 
            % Overlay scatter points
            %scatter(vel2kms*Xp_cloudp(:, plot_dims(1)), vel2kms*Xp_cloudp(:, plot_dims(2)), ...
            %        'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    
            scatter(vel2kms*Xprop_truth{ob}(4), vel2kms*Xprop_truth{ob}(5), ...
                    200, 'k', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'Marker', 'x');
            title('Xdot-Ydot');
            xlabel('Xdot (km/s)');
            ylabel('Ydot (km/s)');
            legend(legend_string);
            hold off;
        
            subplot(2,3,5)
            plot_dims = [4,6];
            mu_marg = mu_mat(:, plot_dims);
            P_marg = P_mat(plot_dims, plot_dims, :);
        
            [X1, X2] = meshgrid(linspace(min(Xp_cloudp{ob}(:,plot_dims(1))), max(Xp_cloudp{ob}(:,plot_dims(1))), 100), ...
                                linspace(min(Xp_cloudp{ob}(:,plot_dims(2))), max(Xp_cloudp{ob}(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
        
            Z_cell = cell(K{ob},1); contours_cell = cell(K{ob},1); 
            
            parfor k = 1:K{ob}
                Z = zeros(size(X1));
                if det(P_marg(:, :, k)) == 0
                        P_marg(:,:,k) = P_marg(:,:,k) + 1e-12*eye(size(P_marg(:,:,k)));
                end
                for i = 1:size(X_grid, 1)
                    Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
                end
                Z = reshape(Z, size(X1));
                Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
                contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
            end 
            
            hold on;
            for k = 1:K{ob}
                contour(vel2kms*X1, vel2kms*X2, vel2kms*Z_cell{k}, vel2kms*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
            end
            % Overlay scatter points
            %scatter(vel2kms*Xp_cloudp(:, plot_dims(1)), vel2kms*Xp_cloudp(:, plot_dims(2)), ...
            %        'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    
            scatter(vel2kms*Xprop_truth{ob}(4), vel2kms*Xprop_truth{ob}(6), ...
                    200, 'k', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'Marker', 'x');
            title('Xdot-Zdot');
            xlabel('Xdot (km/s)');
            ylabel('Zdot (km/s)');
            legend(legend_string);
            hold off;
        
            subplot(2,3,6)
            plot_dims = [5,6];
            mu_marg = mu_mat(:, plot_dims);
            P_marg = P_mat(plot_dims, plot_dims, :);
        
            [X1, X2] = meshgrid(linspace(min(Xp_cloudp{ob}(:,plot_dims(1))), max(Xp_cloudp{ob}(:,plot_dims(1))), 100), ...
                                linspace(min(Xp_cloudp{ob}(:,plot_dims(2))), max(Xp_cloudp{ob}(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
        
            Z_cell = cell(K{ob},1); contours_cell = cell(K{ob},1); 
            
            parfor k = 1:K{ob}
                Z = zeros(size(X1));
                if det(P_marg(:, :, k)) == 0
                        P_marg(:,:,k) = P_marg(:,:,k) + 1e-12*eye(size(P_marg(:,:,k)));
                end
                for i = 1:size(X_grid, 1)
                    Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
                end
                Z = reshape(Z, size(X1));
                Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
                contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
            end 
            
            hold on;
            for k = 1:K{ob}
                contour(vel2kms*X1, vel2kms*X2, vel2kms*Z_cell{k}, vel2kms*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
            end 
            % Overlay scatter points
            %scatter(vel2kms*Xp_cloudp(:, plot_dims(1)), vel2kms*Xp_cloudp(:, plot_dims(2)), ...
            %        'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
    
            scatter(vel2kms*Xprop_truth{ob}(5), vel2kms*Xprop_truth{ob}(6), ...
                    200, 'k', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'Marker', 'x');
            title('Ydot-Zdot');
            xlabel('Ydot (km/s)');
            ylabel('Zdot (km/s)');
            legend(legend_string);
            hold off;
        
            sgt = sprintf('Timestep: %3.4f Hours (Posterior) Ob: %i', [tpr*time2hr, ob]);
            sgtitle(sgt);
        
            sg = sprintf(save_loc + '/Observer' + num2str(ob) + '/Timestep_%i_2A.png', tau);
            % sg = sprintf('./Simulations/Different Orbit Simulations/Timestep_%i_2A.png', tau);
            saveas(f, sg, 'png');
            close(f);
            %}
        
            f = figure('visible','off','Position', get(0,'ScreenSize'));
            f.WindowState = 'maximized';
    
            % fprintf("Plotting Particles at Timestep: %d\n", tau);
        
            legend_string = "Truth";
        
            subplot(2,3,1)
            % parfor k = 1:K
            %     cPoints{k} = Xp_cloudp(c_id == k, :);
            %     mu_pExp(k,:) = mu_p{k};
            % end
            hold on;
            for k = 1:K{ob}
                scatter(dist2km*Xp_cloudp{ob}(c_id{ob} == k,1), dist2km*Xp_cloudp{ob}(c_id{ob} == k,2), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
            end
            
        
            % plot(mu_pExp(:,1), mu_pExp(:,2), '+', 'MarkerSize', 10, 'LineWidth', 3);
            % hold on;
            plot(dist2km*Xprop_truth{ob}(1), dist2km*Xprop_truth{ob}(2), 'kx', ...
                'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string)
            title('X-Y');
            xlabel('X (km.)');
            ylabel('Y (km.)');
            legend(legend_string);
            hold off;
        
            subplot(2,3,2)
            % parfor k = 1:K
            %     cPoints{k} = Xp_cloudp(c_id == k, :);
            %     mu_pExp(k,:) = mu_p{k};
            % end
            hold on;
            for k = 1:K{ob}
                scatter(dist2km*Xp_cloudp{ob}(c_id{ob} == k,1), dist2km*Xp_cloudp{ob}(c_id{ob} == k,3), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
            end
            plot(dist2km*Xprop_truth{ob}(1), dist2km*Xprop_truth{ob}(3), 'kx', ...
                'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string)
            title('X-Z');
            xlabel('X (km.)');
            ylabel('Z (km.)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,3)
            % parfor k = 1:K
            %     cPoints{k} = Xp_cloudp(c_id == k, :);
            %     mu_pExp(k,:) = mu_p{k};
            % end
            hold on;
            for k = 1:K{ob}
                scatter(dist2km*Xp_cloudp{ob}(c_id{ob} == k,2), dist2km*Xp_cloudp{ob}(c_id{ob} == k,3), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
            end
            plot(dist2km*Xprop_truth{ob}(2), dist2km*Xprop_truth{ob}(3), 'kx', ...
                'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string)
            title('Y-Z');
            xlabel('Y (km.)');
            ylabel('Z (km.)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,4)
            % parfor k = 1:K
            %     cPoints{k} = Xp_cloudp(c_id == k, :);
            %     mu_pExp(k,:) = mu_p{k};
            % end
            hold on;
            for k = 1:K{ob}
                scatter(vel2kms*Xp_cloudp{ob}(c_id{ob} == k,4), vel2kms*Xp_cloudp{ob}(c_id{ob} == k,5), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
            end
            plot(vel2kms*Xprop_truth{ob}(4), vel2kms*Xprop_truth{ob}(5), 'kx', ...
                'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string)
        
            title('Xdot-Ydot');
            xlabel('Xdot (km/s)');
            ylabel('Ydot (km/s)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,5)
            % parfor k = 1:K
            %     cPoints{k} = Xp_cloudp(c_id == k, :);
            %     mu_pExp(k,:) = mu_p{k};
            % end
            hold on;
            for k = 1:K{ob}
                scatter(vel2kms*Xp_cloudp{ob}(c_id{ob} == k,4), vel2kms*Xp_cloudp{ob}(c_id{ob} == k,6), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
            end
            plot(vel2kms*Xprop_truth{ob}(4), vel2kms*Xprop_truth{ob}(6), 'kx', ...
                'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string)
            title('Xdot-Zdot');
            xlabel('Xdot (km/s)');
            ylabel('Zdot (km/s)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,6)
            % parfor k = 1:K
            %     cPoints{k} = Xp_cloudp(c_id == k, :);
            %     mu_pExp(k,:) = mu_p{k};
            % end
            hold on;
            for k = 1:K{ob}
                scatter(vel2kms*Xp_cloudp{ob}(c_id{ob} == k,5), vel2kms*Xp_cloudp{ob}(c_id{ob} == k,6), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
            end
            plot(vel2kms*Xprop_truth{ob}(5), vel2kms*Xprop_truth{ob}(6), 'kx', ...
                'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string)
            title('Ydot-Zdot');
            xlabel('Ydot (km/s)');
            ylabel('Zdot (km/s)');
            legend(legend_string);
            hold off;
        
            sgt = sprintf('Timestep: %3.4f Hours (Posterior) Ob: %i', [tpr*time2hr, ob]);
            sgtitle(sgt);
        
            sg = sprintf(save_loc + '/Observer' + num2str(ob) + '/Timestep_%i_2B.png', tau);
            % sg = sprintf('./Simulations/Different Orbit Simulations/Timestep_%i_2B.png', tau);
            saveas(f, sg, 'png');
            close(f);
    
            f = figure('visible','off','Position', get(0,'ScreenSize'));
            f.WindowState = 'maximized';
    
            legend_string = "Truth";
            hold on;
            %scatter_handles = gobjects(k,1);
            for k = 1:K{ob}
                pts = Xp_cloudp{ob}(c_id{ob} == k, :);
                Zmcloud = zeros(length(pts(:,1)), length(zt));
                for i = 1:length(Zmcloud(:,1))
                    Zmcloud(i,:) = h(pts(i,:))';
                end
    
                Ztruth = h(Xprop_truth{ob})';
                scatter(180/pi*Zmcloud(:,1), 180/pi*Zmcloud(:,2), 'filled', ...
                'MarkerFaceColor', colors(k), 'HandleVisibility', 'off', 'DisplayName', sprintf('k: %i; w: %.3f', [k, wp{ob}(k)]));
            end
            plot(180/pi*Ztruth(1), 180/pi*Ztruth(2), 'kx', ... 
            'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', 'Truth');
            %zt_handle = plot(180/pi*zt(1), 180/pi*zt(2), 'ko', ... 
            %'MarkerSize', 20', 'LineWidth', 3, 'DisplayName', 'Noisy Truth');
            %legend([scatter_handles; truth_handle; zt_handle], 'Location', 'northeastoutside'); 
            title(sprintf('AZ-EL Ob: %i', ob))
            xlabel('Azimuth Angle (deg)')
            ylabel('Elevation Angle (deg)')
    
            sg = sprintf(save_loc + '/Observer' + num2str(ob) + '/Timestep_%i_2C.png', tau);
            saveas(f, sg, 'png');
            close(f);
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
    for ob = 1:num_agents
        if (idx_meas{ob} ~= 0)
            %wsum = 0;
            %for k = 1:K
            %    wsum = wsum + wp(k)*det(P_p{k});
            %end
            %ent2(tau+2) = log(wsum);
            [ent2{ob}(tau+2), mahalanobis{ob}(tau+2), num_cluster{ob}(tau+2), num_particles{ob}(tau+2)] = getMetrics(K{ob}, Xp_cloudp{ob}, Xprop_truth{ob}, cluster_by);
        else
            if (tpr >= cVal{ob})
                Ke = Kmax; % Clusters used for calculating entropy
            else
                Ke = Kn; % Clusters used for calculating entropy
            end
            [ent2{ob}(tau+2), mahalanobis{ob}(tau+2), num_cluster{ob}(tau+2), num_particles{ob}(tau+2)] = getMetrics(Ke, Xp_cloudp{ob}, Xprop_truth{ob}, cluster_by); % Get entropy as if you still are using six clusters
        end
    end

    %if(abs(tpr - cTimes(2)) < 1e-10)
    %    Lp = 1250;
    %elseif(abs(tpr - cTimes(4)) < 1e-10)
    %    Lp = 1500;
    %elseif(abs(tpr - cTimes(8)) < 1e-10)
    %    Lp = 2500;
        %%save("./elseif/elseif" + num2str(ts) + ".mat", "Xp_cloudp", "tpr", "noised_obs","Xprop_truth");
        
    %end

end

if(combine == 0)
    for i=ts_temp:ts
        to = full_ts{1}(i,1);
        interval = full_ts{1}(i+1,1) - full_ts{1}(i,1);
    
        % Propagation Step
        X_combine = propagate(X_combine, to, interval{ob}, ts, "Cloud", obs_lat{1}, obs_lon{1});
        X_combine_truth = propagate(X_combine_truth, to, interval, ts, "Truth", obs_lat{1}, obs_lon{1});
    end
end

Xp_cloudp = cell(1, num_agents);
c_id = cell(1, num_agents);
for ob = 1:num_agents
    fprintf('Final State Truth:\n')
    disp(Xprop_truth{ob});
    Xp_cloudp_temp = zeros(Lp, length(Xprop_truth{ob}));
    c_id_temp = zeros(Lp,1);
    parfor i = 1:Lp
        [Xp_cloudp_temp(i,:), c_id_temp(i)] = drawFrom(wp{ob}, mu_p(:, ob), P_p(:, ob), draw_from_idx+i); 
    end
    draw_from_idx = draw_from_idx + Lp;
    Xp_cloudp{ob} = Xp_cloudp_temp;
    c_id{ob} = c_id_temp;
    
    ent1{ob}(end,:) = getDiagCov(Xp_cloudp{ob});
    ent2{ob}(end) = [];
    mahalanobis{ob}(end) = [];
    num_cluster{ob}(end) = [];
    num_particles{ob}(end) = [];

    figure(fig_num)
    fig_num = fig_num + 1;
    subplot(2,3,1)
    plot(0:l_filt, dist2km*sqrt(ent1{ob}(:,1)))
    xlabel('Filter Step #')
    ylabel('Log \\sigma_X (km.)')
    title('X Standard Deviation')
    
    subplot(2,3,2)
    plot(0:l_filt, dist2km*sqrt(ent1{ob}(:,2)))
    xlabel('Filter Step #')
    ylabel('Log \\sigma_Y (km.)')
    title('Y Standard Deviation')
    
    subplot(2,3,3)
    plot(0:l_filt, dist2km*sqrt(ent1{ob}(:,3)))
    xlabel('Filter Step #')
    ylabel('Log \\sigma_Z (km.)')
    title('Z Standard Deviation')
    
    subplot(2,3,4)
    plot(0:l_filt, vel2kms*sqrt(ent1{ob}(:,4)))
    xlabel('Filter Step #')
    ylabel('Log \\sigma_Xdot (km/s)')
    title('Xdot Standard Deviation')
    
    subplot(2,3,5)
    plot(0:l_filt, vel2kms*sqrt(ent1{ob}(:,5)))
    xlabel('Filter Step #')
    ylabel('\\sigma_Ydot (km/s)')
    title('Ydot Standard Deviation')
    
    subplot(2,3,6)
    plot(0:l_filt, vel2kms*sqrt(ent1{ob}(:,6)))
    xlabel('Filter Step #')
    ylabel('\\sigma_Zdot (km/s)')
    title('Zdot Standard Deviation')
    
    savefig(gcf, save_loc + '/Observer' + num2str(ob) + '/StDevEvols.fig');
    
    % Xprop_truth = [full_ts(idx_end,2:4), full_vts(idx_end,2:4)];
    % mu_pExp = zeros(K, length(mu_p{1}));

    % Plot the results
    figure(fig_num)
    fig_num = fig_num + 1;
    f.WindowState = 'maximized';
    plot(0:l_filt-1, ent2{ob})
    xlabel('Filter Step #')
    ylabel('Entropy Metric')
    title('Entropy Ob: %i', ob)
    sg = sprintf(save_loc + '/Observer' + num2str(ob) + '/Entropy.png', tau);
    saveas(f, sg, 'png');
    close(f);
    
    figure(fig_num)
    fig_num = fig_num + 1;
    f.WindowState = 'maximized';
    x = 0:l_filt-1;
    plot(x, mahalanobis{ob})
    xlabel('Filter Step #')
    ylabel('Mahalanobis Distance')
    title('Mahalanobis Distance Ob: %i', ob)
    semilogy(x,mahalanobis);
    sg = sprintf(save_loc + '/Observer' + num2str(ob) + '/Mahalanobis.png', tau);
    saveas(f, sg, 'png');
    close(f);
    
    figure(fig_num)
    fig_num = fig_num + 1;
    f.WindowState = 'maximized';
    plot(0:l_filt-1, num_cluster{ob})
    xlabel('Filter Step #')
    ylabel('Number of Clusters')
    title('Number of Clusters Ob: %i', ob)
    sg = sprintf(save_loc + '/Observer' + num2str(ob) + '/NumClusters.png', tau);
    saveas(f, sg, 'png');
    close(f);
    
    figure(fig_num)
    fig_num = fig_num + 1;
    f.WindowState = 'maximized';
    plot(0:l_filt-1, num_particles{ob})
    xlabel('Filter Step #')
    ylabel('Number of Particles')
    title('Number of Particles Ob: %i', ob)
    sg = sprintf(save_loc + '/Observer' + num2str(ob) + '/NumParticles.png', tau);
    saveas(f, sg, 'png');
    close(f);

    % Plot the results
    figure(fig_num)
    fig_num = fig_num + 1;
    subplot(2,1,1)
    hold on;
    for k = 1:K{ob}
        clusterPoints = Xp_cloudp{ob}(c_id{ob} == k, :);
        mu_pExp(k,:) = mu_p{k, ob};
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
    for k = 1:K{ob}
        clusterPoints = Xp_cloudp{ob}(c_id{ob} == k, :);
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

    % Plot planar projections
    figure(fig_num)
    fig_num = fig_num + 1;
    set(gcf, 'units','normalized','outerposition',[0 0 1 1])
    subplot(2,3,1)
    for k = 1:K{ob}
        clusterPoints = Xp_cloudp{ob}(c_id{ob} == k, :);
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
    for k = 1:K{ob}
        clusterPoints = Xp_cloudp{ob}(c_id{ob} == k, :);
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
    for k = 1:K{ob}
        clusterPoints = Xp_cloudp{ob}(c_id{ob} == k, :);
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
    for k = 1:K{ob}
        clusterPoints = Xp_cloudp{ob}(c_id{ob} == k, :);
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
    for k = 1:K{ob}
        clusterPoints = Xp_cloudp{ob}(c_id{ob} == k, :);
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
    for k = 1:K{ob}
        clusterPoints = Xp_cloudp{ob}(c_id{ob} == k, :);
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
    saveas(gcf, save_loc + '/Observer' + num2str(ob) + '/finalDistribution_normK.png', 'png');
    % savefig(gcf, 'nextObservedTracklet_normK.fig');
    %}
    
    %%save("./Outside2/stdevs.mat", "ent1");
end
% Finish timer
toc

%% Functions

function [idx, K, C] = cluster(data, cluster_by, K)
    if(cluster_by == "Range")
        msmt_cloud = zeros(length(data), 1);
        h = @(x) [sqrt(x(1)^2 + x(2)^2 + x(3)^2)]; % Nonlinear measurement model
        parfor j = 1:length(data)
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
        parfor j = 1:length(data)
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
    [idx, C] = kmeans(Xm_norm, K);
    num_times_clustered = 1;
    while any(histcounts(idx) <= 6) % Ensure at least 6 points in each cluster
        if num_times_clustered >= 3
            K = K - 1; 
        end
        [idx, C] = kmeans(Xm_norm, K);
        num_times_clustered = num_times_clustered + 1;
    end
end

function Hx = linHx(mu)
    Hk_AZ = [-mu(2)/(mu(1)^2 + mu(2)^2), mu(1)/(mu(1)^2 + mu(2)^2), 0, 0, 0, 0]; % Azimuth angle linearization
    Hk_EL = [-(mu(1)*mu(3))/((mu(1)^2 + mu(2)^2 + mu(3)^2)*sqrt(mu(1)^2+mu(2)^2)), ...
             -(mu(2)*mu(3))/((mu(1)^2 + mu(2)^2 + mu(3)^2)*sqrt(mu(1)^2+mu(2)^2)), ...
             sqrt(mu(1)^2 + mu(2)^2)/(mu(1)^2 + mu(2)^2 + mu(3)^2), 0, 0, 0];

    Hx = [Hk_AZ; Hk_EL];
end

function w = weightUpdate(wc, cluster_points, idx, zk, R, h, id)
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
    %%save("./weights/wGains_"+num2str(id)+".mat", "wGains");
    w = wc .* wGains / sum(wc .* wGains);
end

function [dX_coeffs] = polyDeriv(X_coeffs)
    
    dX_coeffs = zeros(1, length(X_coeffs)-1);
    for j = length(X_coeffs):-1:2
        dX_coeffs(length(X_coeffs)+1-j) = X_coeffs(length(X_coeffs)+1-j)*(j-1);
    end
end

% Adds process noise to the un-noised state vector
function [Xm] = procNoise(X, id)
    Q = (0.000*diag(abs(X))).^2; % Process noise is 1% of each state vector component
    Xm = mvnrnd(X,Q);
    %%save("./processNoise/procNoise_" + num2str(id) + ".mat", "Xm");
end

function [Xfit] = stateEstCloud(pf, obTr, tdiff, id, load_loc)
    noised_obs = obTr;

    R_t = zeros(3*length(noised_obs(:,1)),1); % We shall diagonalize this later
    mu_t = zeros(3*length(noised_obs(:,1)),1);

    load(load_loc + "/partial_ts.mat"); % Noiseless observation data
    %partial_ts = csvread("D:/PythonProjects/EDP/PGM_Git/PAR-PGM/partial_ts.csv");
    for i = 1:length(obTr(:,1))
        mu_t(3*(i-1)+1:3*(i-1)+3, 1) = [partial_ts(i,2); partial_ts(i,3); partial_ts(i,4)];
        R_t(3*(i-1)+1:3*(i-1)+3, 1) = [0.05*partial_ts(i,2); 7.2722e-6; 7.2722e-6].^2;
    end

    R_t = diag(R_t);
    data_vec = mvnrnd(mu_t, R_t)';
    %%save("./sEC/stateEstCloud_" + num2str(id) + ".mat", "data_vec");

    for i = 1:length(noised_obs(:,1))
        noised_obs(i,2:4) = data_vec(3*(i-1)+1:3*(i-1)+3,1);
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
    coeffs_X = polyfit(hdR_p(:,1), hdR_p(:,2), 4);
    coeffs_Y = polyfit(hdR_p(:,1), hdR_p(:,3), 4);
    coeffs_Z = polyfit(hdR_p(:,1), hdR_p(:,4), 4);

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

% Convert from one topocentric RF to another
function [X_ot_f] = topoToTopo(X_ot_0, t_stamp_0, obs_lat_0, obs_lon_0, t_stamp_f, obs_lat_f, obs_lon_f)
    rot_topo = X_ot_0(1:3); % First three components of the state vector
    vot_topo = X_ot_0(4:6); % Last three components of the state vector

    % First step: Obtain X_{eo}^{ECI} 
    elevation = 103.8;
    mu = 1.2150582e-2;

    UTC_vec_orig = [2024	5	3	2	41 15];%15.1261889999956]; % Initial UTC vector at t_stamp = 0
    t_add_dim = t_stamp_0 * (4.342); % Convert the time to add to a dimensional quantity
    UTC_vec = datevec(datetime(UTC_vec_orig) + t_add_dim); % You will need this for calculating r_{eo} and v_{eo}

    delt_add_dim = t_add_dim - 1/86400;
    delt_updatedUTCtime = datetime(UTC_vec_orig) + delt_add_dim;
    delt_updatedUTCvec = datevec(delt_updatedUTCtime);

    reo_dim = lla2eci([obs_lat_0 obs_lon_0, elevation], UTC_vec);
    delt_reodim = lla2eci([obs_lat_0 obs_lon_0, elevation], delt_updatedUTCvec);
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

    rot_ECI = A^(-1)*rot_topo;
    vot_ECI = A^(-1)*(vot_topo - dA_dt*rot_ECI);
    
    % Insert code for obtaining vector between center of Earth and observer
    rbe = [-mu, 0, 0]'; % Position vector relating center of earth to barycenter

    UTC_vec_orig = [2024	5	3	2	41	15];%.1261889999956];
    t_add_dim = t_stamp_f * (4.342);
    UTC_vec = datevec(datetime(UTC_vec_orig) + t_add_dim);

    delt_add_dim = -1/86400;
    delt_updatedUTCtime = datetime(UTC_vec) + delt_add_dim;
    delt_updatedUTCvec = datevec(delt_updatedUTCtime);

    reo_dim = lla2eci([obs_lat_f obs_lon_f, elevation], UTC_vec);
    delt_reodim = lla2eci([obs_lat_f obs_lon_f, elevation], delt_updatedUTCvec);
    veo_dim = reo_dim - delt_reodim;

    reo_nondim = reo_dim'/(1000*384400); % Conversion to non-dimensional units and ECI frame
    veo_nondim = veo_dim'*(4.342*86400)/(1000*384400);

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

    X_ot_f = [rot_topo'; vot_topo];

end

function [X_bt] = backConvertSynodic(X_ot, t_stamp, obs_lat, obs_lon)

    rot_topo = X_ot(1:3); % First three components of the state vector
    vot_topo = X_ot(4:6); % Last three components of the state vector

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

    X_bt = [r_bt; v_bt];
end

% Used for converting between X_{BT} in the synodic frame and X_{OT} in the
% topocentric frame for a single state
function [X_ot] = convertToTopo(X_bt, t_stamp, obs_lat, obs_lon)
    % Insert code for obtaining vector between center of Earth and observer

    elevation = 103.8;
    
    mu = 1.2150582e-2;
    rbe = [-mu, 0, 0]'; % Position vector relating center of earth to barycenter

    UTC_vec_orig = [2024	5	3	2	41	15];%.1261889999956];
    t_add_dim = t_stamp * (4.342);
    UTC_vec = datevec(datetime(UTC_vec_orig) + t_add_dim);

    delt_add_dim = -1/86400;
    delt_updatedUTCtime = datetime(UTC_vec) + delt_add_dim;
    delt_updatedUTCvec = datevec(delt_updatedUTCtime);

    reo_dim = lla2eci([obs_lat obs_lon, elevation], UTC_vec);
    delt_reodim = lla2eci([obs_lat obs_lon, elevation], delt_updatedUTCvec);
    %save("./cTT/cTT_" + num2str(id) + ".mat", "reo_dim", "delt_reodim");
    veo_dim = reo_dim - delt_reodim;

    R_z = [cos(t_stamp), -sin(t_stamp), 0; sin(t_stamp), cos(t_stamp), 0; 0, 0, 1];
    dRz_dt = [-sin(t_stamp), -cos(t_stamp), 0; cos(t_stamp), -sin(t_stamp), 0; 0, 0, 0];

    reo_nondim = reo_dim'/(1000*384400); % Conversion to non-dimensional units and ECI frame
    veo_nondim = veo_dim'*(4.342*86400)/(1000*384400);

    rot_ECI = -reo_nondim + R_z*(-rbe + X_bt(1:3));
    vot_ECI = -veo_nondim + R_z*(X_bt(4:6)) + dRz_dt*(-rbe + X_bt(1:3));

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

    X_ot = [rot_topo'; vot_topo];
end

function [x_p, pos] = drawFrom(w, mu, P, id)
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
            pos_old = mid;
            return
        elseif (cmf_w(mid) < wtoken)
            left = mid + 1;
        else
            right = mid - 1;
        end
    end
    pos_old = left;

    if(pos_old > length(mu))
        pos_old = length(mu); % Correction for rounding error
    end
    
    pos = discretize(wtoken, [-Inf, cmf_w', Inf]);

    if(pos_old ~= pos)
        pos_old
        pos
    end

    mu_t = mu{pos};
    R_t = (P{pos} + P{pos}')/2;
    x_p = mvnrnd(mu_t, R_t)';
    %%save("./dF/drawFrom_" + num2str(id) + ".mat", "x_p", "pos", "wtoken");
end

function [x_p, pos] = drawFrom3(w, mu, P, id)
    % Use histcounts for efficient sampling
    pos = histcounts(rand, [0; cumsum(w(:))]);
    pos = find(pos, 1);

    if (isempty(pos) || pos > length(mu))
        error('Sampling error: invalid position');
    end
    
    x_p = mvnrnd(mu{pos}, P{pos});
    %%save("./dF3/drawFrom3_" + num2str(id) + ".mat", "x_p");
end

function zk = getNoisyMeas(Xtruth, R, h, id)
    mzkm = h(Xtruth);
    zk = mvnrnd(mzkm, R);
    %%save("./NoisyMeas/zk_" + num2str(id) + ".mat", "zk")
    zk = zk'; % Make into column vector
end

function Xm_cloud = propagate(Xcloud, t_int, interval, ts, truth, obs_lat, obs_lon)
    % Xcloud = zeros(L,length(mu{1}));
    % for i = 1:L
    %     [Xcloud(i,:), ~] = drawFrom(w, mu, P);
    % end
    % 
    %X_all = cell(length(Xcloud(:,1)), 1);
    %T_all = cell(length(Xcloud(:,1)), 1);
    %Xbt = zeros(size(Xcloud));
    Xm_cloud = zeros(size(Xcloud));
    %Xm_bt = zeros(size(Xcloud));
    %load("./Propagate/Xcloud_" + num2str(ts) + truth + ".mat", "Xcloud");
    %%save("./Propagate/Xcloud_" + num2str(ts) + truth + ".mat", "Xcloud")
    parfor i = 1:length(Xcloud(:,1))
        %if i == 47
        %    aaa = 1;
        %end
        Xbt = backConvertSynodic(Xcloud(i,:)', t_int, obs_lat, obs_lon);
        %%save("./Propagate/Xbt_" + num2str(ts) + truth + ".mat", "Xbt")
        % Next, propagate each X_{bt} in your particle cloud by a single time 
        % step and convert back to the topographic frame.
        % Call ode45()

        % opts = odeset('Events', @termSat);
        % [~,X] = ode45(@cr3bp_dyn, [0 interval], Xbt, opts); % Assumes termination event (i.e. target enters LEO)
        opts = odeset('Events', @termSat, 'RelTol', 1e-6, 'AbsTol', 1e-8); 
        %[~, X] = ode15s(@cr3bp_dyn, [0 interval], Xbt, opts); 
        [~, X] = ode15s(@cr3bp_dyn, [0 interval], Xbt, opts);
        %X_all{i} = X;
        %T_all{i} = T;
        Xm_bt = X(end,:)';
        %%save("./Propagate/ivp_" + num2str(ts) + truth + ".mat", "X_all", "T_all", "Xm_bt")
        Xm_cloud(i,:) = convertToTopo(Xm_bt, t_int + interval, obs_lat, obs_lon);
    end
    %%save("./Propagate/Xm_cloud_" + num2str(ts) + truth + ".mat", "Xm_cloud")
        
    % Xm_cloud(i,:) = procNoise(Xm_cloud(i,:)); % Adds process noise   
end

% Kalman update using particles from each cluster
function [mu_p, P_p] = kalmanUpdate(zk, Xcloud, R, mu_m, P_m, h)
    N = size(Xcloud,1);
    Zcloud = zeros(N,length(zk));

    parfor i = 1:N
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
    mu_p = mu_m' + K_k*(zk - h(mu_m));
    P_p = P_m - K_k*Pzz*K_k';
    
    % mu_p = mu_m' + Pxz'*Pzz^(-1)*(zk - h(mu_m));
    % P_p = P_m - Pxz'*Pzz^(-1)*Pxz;
    
    P_p = (P_p + P_p')/2;

    [V, D] = eig(P_p);
    D = max(D,0);
    P_p = V*D*V';
end

% Kalman update for a single time step (with linearized measurement model)
function [mu_p, P_p] = ekfUpdate(zk, H, R, mu_m, P_m, h)
    Pxz = P_m*H';
    Pzz = H*P_m*H' + R;
    K_k = Pxz/Pzz;
    % K_k = P_m*H'*(H*P_m*H' + R)^(-1); % Kalman gain

    %{
    ek = zk - h(mu_m);
    if(ek(2) > pi)
        ek(2) = ek(2) - 2*pi;
    elseif(ek(2) < -pi)
        ek(2) = ek(2) + 2*pi;
    end
    %}

    mu_p = mu_m + K_k*(zk - h(mu_m));
    % mu_p = mu_m + K_k*ek;
    P_p = (eye(length(mu_m)) - K_k*H)*P_m;
    % P_p = P_m - K_k*(H*P_m*H' + R)^(-1)*K_k';
    % P_p = P_m - Pxz*K_k' - K_k*Pxz' + K_k*Pzz*K_k';
    % P_p = (eye(length(mu_m)) - K_k*H)*P_m*(eye(length(mu_m)) - K_k*H)' + K_k*R*K_k'; % Joseph formula

    % Ensure Kalman update is symmetric
    P_p = (P_p + P_p')/2;
end

function [mu_c, P_c] = ukfProp(t_int, interval, mu_p, P_p, obs_lat, obs_lon)
    % Generate 2L+1 sigma vectors
    n = length(mu_p); % Length of state vector
    alpha = 1e-3;
    beta = 2;
    kappa = 0;
    lambda = alpha^2*(n + kappa) - n;

    % Calculate square root factor
    P_p = (P_p + P_p')/2;

    [V, D] = eig(P_p);
    D = max(D,0);
    P_p = V*D*V';

    S = chol(P_p, 'lower'); % Obtain SRF via Cholesky decomposition

    sigs = zeros(2*n+1, n); % Matrix of sigma vectors

    wm = zeros(1,2*n+1); % Vector of mean weights
    wc = zeros(1,2*n+1); % Vector of covariance weights
    
    % Vectors and weights
    sigs(1,:) = mu_p; 
    wm(1) = lambda/(n + lambda);
    wc(1) = lambda/(n + lambda) - (1 - alpha^2 + beta);

    for i = 2:(n+1)
        sigs(i,:) = (mu_p' + sqrt(n + lambda)*S(:,i-1))'; 
        sigs(i+n,:) = (mu_p' - sqrt(n + lambda)*S(:,i-1))';

        wm(i) = 0.5/(n + lambda); wm(i+n) = wm(i);
        wc(i) = 0.5/(n + lambda); wc(i+n) = wc(i);
    end
    
    prop_sigs = zeros(size(sigs));
    % Propagation of sigma points
    for i = 1:length(sigs(i,:))
        prop_sigs(i,:) = propagate(sigs(i,:), t_int, interval, obs_lat, obs_lon);
    end

    % Get a priori mean
    mu_c = zeros(n, 1);

    for i = 1:(2*n+1)
        mu_c = mu_c + wm(i)*prop_sigs(i,:);
    end

    P_c = zeros(size(P_p));
    for i = 1:(2*n+1)
        P_c = P_c + wc(i) * ((prop_sigs(i,:) - mu_c)*(prop_sigs(i,:) - mu_c)');
    end
    
    P_c = (P_c + P_c')/2;

    [V, D] = eig(P_c);
    D = max(D,0);
    P_c = V*D*V';
end

function [mu_p, P_p] = ukfUpdate(zk, R, mu_m, P_m, h)
    % Generate 2L+1 sigma vectors
    n = length(mu_m); % Length of state vector
    alpha = 1e-3;
    beta = 2;
    kappa = 0;
    lambda = alpha^2*(n + kappa) - n;

    % Calculate square root factor
    P_m = (P_m + P_m')/2;

    [V, D] = eig(P_m);
    D = max(D,0);
    P_m = V*D*V';

    S = chol(P_m, 'lower'); % Obtain SRF via Cholesky decomposition

    sigs = zeros(2*n+1, n); % Matrix of sigma vectors
    zetas = zeros(2*n+1, length(zk));

    wm = zeros(1,2*n+1); % Vector of mean weights
    wc = zeros(1,2*n+1); % Vector of covariance weights
    
    % Vectors and weights
    sigs(1,:) = mu_m; 
    zetas(1,:) = h(mu_m);
    wm(1) = lambda/(n + lambda);
    wc(1) = lambda/(n + lambda) - (1 - alpha^2 + beta);

    for i = 2:(n+1)
        sigs(i,:) = (mu_m' + sqrt(n + lambda)*S(:,i-1))'; 
        sigs(i+n,:) = (mu_m' - sqrt(n + lambda)*S(:,i-1))';

        zetas(i,:) = h(sigs(i,:)); zetas(i+n,:) = h(sigs(i+n,:));

        wm(i) = 0.5/(n + lambda); wm(i+n) = wm(i);
        wc(i) = 0.5/(n + lambda); wc(i+n) = wc(i);
    end

    Pxz = zeros(length(mu_m), length(zk));
    Pzz = zeros(length(zk), length(zk));
    mzk = zk*0.0;

    for i = 1:(2*n+1)
        mzk = mzk + wm(i)*zetas(i,:)';
    end

    for i = 1:(2*n+1)
        Pzz = Pzz + wc(i)*(zetas(i,:)' - mzk)*(zetas(i,:)' - mzk)';
        Pxz = Pxz + wc(i)*(sigs(i,:)' - mu_m')*(zetas(i,:)' - mzk)';
    end
    Pzz = Pzz + R;

    % Compute optimal Kalman Gain
    K_k = Pxz/Pzz;

    % Update mean and covariances
    mu_p = mu_m' + K_k*(zk - mzk);
    % P_p = P_m - Pxz*K_k' - K_k*Pxz' + K_k*Pzz*K_k';
    P_p = P_m - K_k*Pzz*K_k';
    P_p = (P_p + P_p')/2;

    [V, D] = eig(P_p);
    D = max(D,0);
    P_p = V*D*V';
end

function [ent, Dsum, Kp, Lp] = getMetrics(Kp, Xcloud, Xtruth, cluster_by)
    if(cluster_by == "Range")
        msmt_cloud = zeros(length(Xcloud), 1);
        h = @(x) [sqrt(x(1)^2 + x(2)^2 + x(3)^2)]; % Nonlinear measurement model
        parfor j = 1:length(Xcloud)
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
        norm_rc = (rc - mean_rc)./norm(std_rc); % Normalizing the position
        vc = Xcloud(:,4:6);
        mean_vc = mean(vc, 1);
        std_vc = std(vc,0,1);
        norm_vc = (vc - mean_vc)./norm(std_vc); % Normalizing the velocity
        Xm_norm = [norm_rc, norm_vc];
    end
    if(cluster_by == "Velocity")
        vc = Xcloud(:,4:6);
        mean_vc = mean(vc, 1);
        std_vc = std(vc,0,1);
        norm_vc = (vc - mean_vc)./norm(std_vc); % Normalizing the velocity
        Xm_norm = [norm_vc];
    end

    [idx, ~] = kmeans(Xm_norm, Kp); % Cluster just on position and velocity; Normalize the whole thing
    num_times_clustered = 1;
    while any(histcounts(idx) <= 6) % Ensure at least 6 points in each cluster
        if num_times_clustered >= 3
            Kp = Kp - 1; 
        end
        [idx, ~] = kmeans(Xm_norm, Kp);
        num_times_clustered = num_times_clustered + 1;
    end
    cPoints = cell(Kp,1); P = cell(Kp,1);
    w = zeros(Kp,1);
    D = zeros(Kp,1);
    
    % Calculate covariances and weights for each cluster
    for k = 1:Kp
        cluster_points = Xcloud(idx == k, :); % Keep clustering very separate from mean, covariance, weight calculations
        cPoints{k} = cluster_points; cSize = size(cPoints{k});
    
        if(cSize(1) == 1)
            P{k} = zeros(length(w));
        else
            P{k} = cov(cluster_points); % Cell of GMM covariances 
        end
        
        w(k) = size(cluster_points, 1) / size(Xm_norm, 1); % Vector of weights
        if(cSize(1) > cSize(2))
            D(k) = mahal(Xtruth, cluster_points);
        else
            D(k) = 0;
        end
    end

    wsum = 0;
    Dsum = 0;
    for k = 1:Kp
        wsum = wsum + w(k)*det(P{k});
        Dsum = Dsum + w(k)*D(k);
    end
    ent = log(wsum);
    Lp = length(Xcloud(:, 1));
end

function ent = getDiagCov(Xcloud)
    P = cov(Xcloud);
    ent = diag(P);
end

function [alpha, mu, P_cov, w] = calcAlpha(X_initial, X_final, K_max, num_alpha, cluster_by, dist2km, vel2kms, save_loc)
    x_eval = mvnrnd(mean(X_final, 1), cov(X_final), num_alpha);
    cloud = {X_initial{:}, X_final};
    p = cell(1, length(cloud));
    
    f = figure('visible','off','Position', get(0,'ScreenSize'));
    f.WindowState = 'maximized';
    color = ["Blue", "Green", "Red"];
    %legend_string = "Truth";

    subplot(2,3,1)
    hold on; 
    scatter(dist2km*X_initial{1}(:,1), dist2km*X_initial{1}(:,2), 'filled', 'MarkerFaceColor', color(1), 'HandleVisibility', 'off');
    scatter(dist2km*X_initial{2}(:,1), dist2km*X_initial{2}(:,2), 'filled', 'MarkerFaceColor', color(2), 'HandleVisibility', 'off');
    scatter(dist2km*x_eval(:,1), dist2km*x_eval(:,2), 'filled', 'MarkerFaceColor', color(2), 'HandleVisibility', 'off');
    title('X-Y');
    xlabel('X (km.)');
    ylabel('Y (km.)');
    %legend(legend_string);
    hold off;
    
    subplot(2,3,2)
    hold on; 
    scatter(dist2km*X_initial{1}(:,1), dist2km*X_initial{1}(:,3), 'filled', 'MarkerFaceColor', color(1), 'HandleVisibility', 'off');
    scatter(dist2km*X_initial{2}(:,1), dist2km*X_initial{2}(:,3), 'filled', 'MarkerFaceColor', color(2), 'HandleVisibility', 'off');
    scatter(dist2km*x_eval(:,1), dist2km*x_eval(:,3), 'filled', 'MarkerFaceColor', color(2), 'HandleVisibility', 'off');
    title('X-Z');
    xlabel('X (km.)');
    ylabel('Z (km.)');
    %legend(legend_string);
    hold off;
    
    subplot(2,3,3)
    hold on;
    scatter(dist2km*X_initial{1}(:,2), dist2km*X_initial{1}(:,3), 'filled', 'MarkerFaceColor', color(1), 'HandleVisibility', 'off');
    scatter(dist2km*X_initial{2}(:,2), dist2km*X_initial{2}(:,3), 'filled', 'MarkerFaceColor', color(2), 'HandleVisibility', 'off');
    scatter(dist2km*x_eval(:,2), dist2km*x_eval(:,3), 'filled', 'MarkerFaceColor', color(2), 'HandleVisibility', 'off');
    title('Y-Z');
    xlabel('Y (km.)');
    ylabel('Z (km.)');
    %legend(legend_string);
    hold off;
    
    subplot(2,3,4)
    hold on;
    scatter(vel2kms*X_initial{1}(:,4), vel2kms*X_initial{1}(:,5), 'filled', 'MarkerFaceColor', color(1), 'HandleVisibility', 'off');
    scatter(vel2kms*X_initial{2}(:,4), vel2kms*X_initial{2}(:,5), 'filled', 'MarkerFaceColor', color(2), 'HandleVisibility', 'off');
    scatter(vel2kms*x_eval(:,4), vel2kms*x_eval(:,5), 'filled', 'MarkerFaceColor', color(2), 'HandleVisibility', 'off');
    title('Xdot-Ydot');
    xlabel('Xdot (km/s)');
    ylabel('Ydot (km/s)');
    %legend(legend_string);
    hold off;
    
    subplot(2,3,5)
    hold on; 
    scatter(vel2kms*X_initial{1}(:,4), vel2kms*X_initial{1}(:,6), 'filled', 'MarkerFaceColor', color(1), 'HandleVisibility', 'off');
    scatter(vel2kms*X_initial{2}(:,4), vel2kms*X_initial{2}(:,6), 'filled', 'MarkerFaceColor', color(2), 'HandleVisibility', 'off');
    scatter(vel2kms*x_eval(:,4), vel2kms*x_eval(:,6), 'filled', 'MarkerFaceColor', color(2), 'HandleVisibility', 'off');
    title('Xdot-Zdot');
    xlabel('Xdot (km/s)');
    ylabel('Zdot (km/s)');
    %legend(legend_string);
    hold off;
    
    subplot(2,3,6)
    hold on;
    scatter(vel2kms*X_initial{1}(:,5), vel2kms*X_initial{1}(:,6), 'filled', 'MarkerFaceColor', color(1), 'HandleVisibility', 'off');
    scatter(vel2kms*X_initial{2}(:,5), vel2kms*X_initial{2}(:,6), 'filled', 'MarkerFaceColor', color(2), 'HandleVisibility', 'off');
    scatter(vel2kms*x_eval(:,5), vel2kms*x_eval(:,6), 'filled', 'MarkerFaceColor', color(2), 'HandleVisibility', 'off');
    title('Ydot-Zdot');
    xlabel('Ydot (km/s)');
    ylabel('Zdot (km/s)');
    %legend(legend_string);
    hold off;

    sgt = sprintf('Alpha Calc');
    sgtitle(sgt);

    sg = sprintf(save_loc + '/alpha.png');
    saveas(f, sg, 'png');
    close(f);
    %cloud{3} = cloud{3}(1001:end, :);
    mu = cell(K_max, 3);
    P_cov = cell(K_max, 3);
    w = zeros(K_max, 3);
    
    for ob = 1:length(cloud)
        [idx, K, ~] = cluster(cloud{ob}, cluster_by, K_max);
        p{ob} = zeros(num_alpha, 1);
        for k = 1:K
            cluster_points = cloud{ob}(idx == k, :); 
            mu{k, ob} = mean(cluster_points, 1); % Cell of GMM means 
            if (length(cluster_points(:,1)) == 1)
                P_cov{k, ob} = zeros(length(mu{k}));
            else
                P_cov{k, ob} = cov(cluster_points); % Cell of GMM covariances
            end
            w(k, ob) = size(cluster_points, 1) / size(cloud{ob}, 1); % Vector of (prior) weights
            p_k = mvnpdf(x_eval, mu{k, ob}, P_cov{k, ob}); % N x 1
            p_k_m = mvnpdf_manual(x_eval, mu{k, ob}, P_cov{k, ob});
            p_k_m2 = exp(mvnpdf_chol_batch(x_eval, mu{k, ob}, P_cov{k, ob}));
            p{ob}(:) = p{ob}(:) + w(k, ob) * p_k(:);
        end
    end
    alpha = (p{3} - p{2})./(p{1} - p{2});
end

function p = mvnpdf_manual(X, mu, Sigma)
    [n, d] = size(X);
    Xc = X - mu;                     % center each row
    invSigma = inv(Sigma);
    quadform = sum((Xc * invSigma) .* Xc, 2);  % efficient batch quadratic form
    denom = sqrt((2 * pi)^d * det(Sigma));
    
    p = exp(-0.5 * quadform) / denom;
end

function logp = mvnpdf_chol_batch(X, mu, Sigma)
    % Inputs:
    % - X: (n x d) matrix of n samples
    % - mu: (d x 1) mean vector
    % - Sigma: (d x d) covariance matrix
    % Output:
    % - logp: (n x 1) log-probabilities of each row in X

    [n, d] = size(X);

    % Center the data
    Xc = X - mu;  % subtract mu row-wise

    % Cholesky decomposition
    [L, p] = chol(Sigma, 'lower');
    if p > 0
        % Sigma not positive definite, regularize and retry
        epsilon = 1e-6;
        [L, p] = chol(Sigma + epsilon * eye(d), 'lower');
        if p > 0
            error('Regularized Sigma still not positive definite.');
        end
    end

    % Solve L * y = (x - mu)' => y = L \ (x - mu)' for all x
    Y = (L \ Xc')';  % each row: (x - mu) transformed

    % Mahalanobis distance (row-wise sum of squared transformed components)
    quadform = sum(Y.^2, 2);

    % Log-determinant from Cholesky factor
    logdet = 2 * sum(log(diag(L)));

    % Log-pdf
    logp = -0.5 * (d * log(2*pi) + logdet + quadform);
end

function [binned_cloud, edges] = binCloud(cloud, display_diagnostics)
    num_dims = 6;
    N = [10, 10, 10, 10, 10, 10];
    num_points = size(cloud, 1);

    edges = cell(1, num_dims);
    idxs = zeros(num_points, num_dims);
    valid = true(num_points, 1);

    % Main loop: compute edges, discretize, build validity mask
    for i = 1:num_dims
        edges{i} = linspace(min(cloud(:, i)), max(cloud(:, i)), N(i)+1);
        idxs(:, i) = discretize(cloud(:, i), edges{i});
        valid = valid & ~isnan(idxs(:, i));
    end

    % Remove invalid samples
    subs = idxs(valid, :);

    % Bin counts
    counts = accumarray(subs, 1, N);

    % Normalize to PDF
    binned_cloud = counts / sum(counts(:));

    % Optional diagnostics
    if display_diagnostics
        fprintf('Diagnostics: Binning Sum = %.6f\n', sum(binned_cloud(:)));
    end
end

%{
function binned_cloud = binCloud(cloud)
    N = [100; 100; 100; 100; 100; 100];
    
    edges = cell(6, 1);
    centers = cell(6, 1);
    idxs = cell(6, 1);
    subs = [];
    for i=1:6
        edges{i} = linspace(min(cloud(:,i)), max(cloud(:,i)), N(i)+1);
        centers{i} = (edges{i}(1:end-1) + edges{i}(2:end)) / 2;
        idxs{i} = discretize(cloud(:,i), edges{i});
        valid = valid & ~isnan(idxs{i});
        subs = [subs, idxs{i}(valid)];
    end
    % Count samples in each 3D bin
    counts = accumarray(subs, 1, N);
    
    % ----- Normalize to get PDF estimate -----
    binned_cloud = counts / sum(counts(:));

    %end
     if(display_diagnostics == true)
        disp("Diagnostics: Binning Sum: " + num2str(sum(binned_cloud, [],  "all")))
    end
end
%}

function [p3_simple, p3_tallest_peaks, weight_update_p3_1, weight_update_p3_2, bin_axes]  = fusionMethods(p1, p2, cluster_by, Kmax, num_agents, num_particles, display_diagnostics)
    bin_axes = cell(1, 3);
    %[binned_p1, bin_axes{1}] = binCloud(p1);
    %[binned_p2, bin_axes{2}] = binCloud(p2);
    p3 = vertcat(p1, p2);
    [p3_simple, bin_axes{1}] = binCloud(p3);

    binned_max_pdf = binned_p1 + binned_p2;
    %original_pdf_dim = size(binned_max_pdf);
    [sorted_max_pdf, sort_idx] = sort(binned_max_pdf(:), 'descend');
    
    cumulative_sum = cumsum(sorted_max_pdf);
    threshold_index = find(cumulative_sum >= 1, 1);
    
    threshold = sorted_max_pdf(threshold_index);
    tallest_peaks = binned_max_pdf(binned_max_pdf >= threshold);
    p3_tallest_peaks = binned_max_pdf;
    p3_tallest_peaks(p3_tallest_peaks < threshold) = 0;

    %[mu_p, P_p] = kalmanUpdate(zk, Xcloud, R, mu_m, P_m, h);
    [weight_update_p3_1, bin_axes{3}] = postWeights({binned_p1, binned_p2}, cluster_by, Kmax, num_agents, num_particles);
    [weight_update_p3_2, bin_axes{4}] = postWeights({binned_p2, binned_p1}, cluster_by, Kmax, num_agents, num_particles);
    
    if(display_diagnostics == True)
        disp("Diagnostics: Fusion Sums: " + num2str(sum(p3_simple, [],  "all"))\ ...
            + ", " + num2str(sum(p3_simple, [],  "all"))\ ...
            + ", " + num2str(sum(weight_update_p3_1, [],  "all"))\ ...
            + ", " + num2str(sum(weight_update_p3_2, [],  "all")))
    end
end

function [p3, bin_axes] = postWeights(data, cluster_by, Kmax, num_agents, num_particles)
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
    for i = 1:K{1}
        for j = 1:K{2}
            [post_mu, post_P] = KalmanFilter(mu_c{i, 1}, P_c{i, 1}, mu_c{j, 2}, P_c{j, 2});
            new_particles = [new_particles; mvnrnd(post_mu, post_P, round(num_particles*post_weights(i, j)))];
        end
    end
    [p3, bin_axes] = bin_cloud(new_particles);
end

function samples = sampleFromNDpdf(pdf_nd, axes, N_particles)
    % pdf_nd: N-D normalized probability density array
    % axes: cell array of bin center vectors for each dimension
    % N_particles: number of samples to generate
    % Returns: samples [N_particles x N_dims]

    N_dims = ndims(pdf_nd);
    assert(length(axes) == N_dims, 'Axes must match the number of dimensions in pdf');

    % Flatten PDF and normalize
    pdf_flat = pdf_nd(:);
    pdf_flat = pdf_flat / sum(pdf_flat);
    cdf_flat = cumsum(pdf_flat);

    % Sample from CDF using inverse transform sampling
    rand_vals = rand(N_particles, 1);
    lin_idx = arrayfun(@(r) find(cdf_flat >= r, 1, 'first'), rand_vals);

    % Convert linear indices to subscripts
    subs = cell(1, N_dims);
    [subs{:}] = ind2sub(size(pdf_nd), lin_idx);

    % Interpolate inside bins
    samples = zeros(N_particles, N_dims);
    for d = 1:N_dims
        centers = axes{d};
        dx = diff(centers); dx = [dx, dx(end)];

        % Calculate bin edges from centers
        edges = centers - 0.5 * dx;
        edges(end+1) = centers(end) + 0.5 * dx(end);

        % Interpolate inside bin using uniform sampling
        idx = subs{d};
        samples(:, d) = edges(idx)' + dx(idx)' .* rand(N_particles, 1);
    end
end


function [x_sampled, y_sampled, z_sampled] = sampleFromFusedPDF(pdf3d, x, y, z, N_particles)
    [Nx, Ny, Nz] = size(pdf3d);

    pdf_flat = pdf3d(:);
    cdf_flat = cumsum(pdf_flat);
    rand_samples = rand(N_particles, 1);
    
    indices = arrayfun(@(r) find(cdf_flat >= r, 1, 'first'), rand_samples);
    [ix, iy, iz] = ind2sub(size(pdf3d), indices);
    
    dx = diff(x); dx = [dx, dx(end)];
    dy = diff(y); dy = [dy, dy(end)];
    dz = diff(z); dz = [dz, dz(end)];

    x_edge = x - 0.5 * dx;
    y_edge = y - 0.5 * dy;
    z_edge = z - 0.5 * dz;

    x_edge(end+1) = x(end) + 0.5 * dx(end);
    y_edge(end+1) = y(end) + 0.5 * dy(end);
    z_edge(end+1) = z(end) + 0.5 * dz(end);

    x_sampled = x_edge(ix) + dx(ix) .* rand(N_particles, 1);
    y_sampled = y_edge(iy) + dy(iy) .* rand(N_particles, 1);
    z_sampled = z_edge(iz) + dz(iz) .* rand(N_particles, 1);
end

