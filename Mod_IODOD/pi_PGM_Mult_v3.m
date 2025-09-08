% Start the clock
tic

% Load noiseless observation data and other important .mat files
load("partial_ts_mult.mat"); % Noiseless observation data
load("full_ts_mult.mat"); % Position truth (topocentric frame)
load("full_vts_mult.mat"); % Velocity truth (topocentric frame)

% Add observation noise to the observation data as follows:
% Range - 5% of the current (i.e. noiseless) range
% Azimuth - 1.5 arc sec (credit: Dr. Utkarsh Ranjan Mishra)
% Elevation - 1.5 arc sec (credit: Dr. Utkarsh Ranjan Mishra)
% Note: All above quantities are drawn in a zero-mean Gaussian fashion.

theta_f = 1.5; % Arc-seconds of error covariance
R_f = 0.01; % Range percentage error covariance

dist2km = 384400; % Kilometers per non-dimensionalized distance
time2hr = 4.342*24; % Hours per non-dimensionalized time
vel2kms = dist2km/(time2hr*60*60); % Kms per non-dimensionalized velocity

Nt = length(Partial_ts); % Number of targets
noised_obs = Partial_ts;

R_t = cell(1,Nt); mu_t = cell(1,Nt); data_vec = cell(1,Nt);
for j = 1:Nt
    R_t{j} = zeros(3*length(noised_obs(:,1)),1); % We shall diagonalize this later
    mu_t{j} = zeros(3*length(noised_obs(:,1)),1);
    for i = 1:length(Partial_ts{j}(:,1))
        mu_t{j}(3*(i-1)+1:3*(i-1)+3, 1) = [Partial_ts{j}(i,2); Partial_ts{j}(i,3); Partial_ts{j}(i,4)];
        R_t{j}(3*(i-1)+1:3*(i-1)+3, 1) = [(0.05*Partial_ts{j}(i,2))^2; (theta_f*4.84814e-6)^2; (theta_f*4.84814e-6)^2];
    end
    R_t{j} = diag(R_t{j});
    data_vec{j} = mvnrnd(mu_t{j}, R_t{j})';

    for i = 1:length(noised_obs{j}(:,1))
        noised_obs{j}(i,2:4) = data_vec{j}(3*(i-1)+1:3*(i-1)+3,1);
    end
end

% Extract the first continuous observation track
hdo = cell(1,Nt); % Matrix for a half day observation

for j = 1:Nt
    hdo{j}(1,:) = noised_obs{j}(1,:);
    i = 1;
    while(noised_obs{j}(i+1,1) - noised_obs{j}(i,1) < Full_ts{j}(2,1) + 1e-15) % Add small epsilon due to roundoff error
        hdo{j}(i,:) = noised_obs{j}(i+1,:);
        i = i + 1;
    end
end

% Convert observation data into [X, Y, Z] data in the topographic frame.
hdR = cell(1,Nt); hdR_p = cell(1,Nt); X0cloud = cell(1,Nt);
pf = 0.50; % A factor between 0 to 1 describing the length of the day to interpolate [x, y]
Xot_fitted = cell(1,Nt); Xot_truth = cell(1,Nt); Xprop_truth = cell(1,Nt);

L = 1000;
Lp = 3*L;

% c = parcluster('local');
% c.NumWorkers = 16;   % Or however many your CPU supports
% saveProfile(c);     % Persist this change
% 
% delete(gcp('nocreate'))
% parpool(16, 'IdleTimeout', Inf);

for j = 1:Nt
    hdR{j} = zeros(length(hdo{j}(:,1)),4); % Convert quantities of hdo to [X, Y, Z]
    hdR{j}(:,1) = hdo{j}(:,1); % Timestamp stays the same
    hdR{j}(:,2) = hdo{j}(:,2) .* cos(hdo{j}(:,4)) .* cos(hdo{j}(:,3)); % Conversion to X
    hdR{j}(:,3) = hdo{j}(:,2) .* cos(hdo{j}(:,4)) .* sin(hdo{j}(:,3)); % Conversion to Y
    hdR{j}(:,4) = hdo{j}(:,2) .* sin(hdo{j}(:,4)); % Conversion to Z

    in_len = round(pf * length(hdR{j}(:,1))); % Length of interpolation interval
    hdR_p{j} = hdR{j}(1:in_len,:); % Matrix for a partial half-day observation

    % Fit polynomials for X, Y, and Z (Cubic for X, Quadratic for X and Y)
    coeffs_X = polyfit(hdR_p{j}(:,1), hdR_p{j}(:,2), 4);
    coeffs_Y = polyfit(hdR_p{j}(:,1), hdR_p{j}(:,3), 4);
    coeffs_Z = polyfit(hdR_p{j}(:,1), hdR_p{j}(:,4), 4);
    
    % Predicted values for X, Y, and Z given the polynomial fits
    X_fit = polyval(coeffs_X, hdR_p{j}(:,1));
    Y_fit = polyval(coeffs_Y, hdR_p{j}(:,1));
    Z_fit = polyval(coeffs_Z, hdR_p{j}(:,1));

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
    Xdot_fit = polyval(coeffs_dX, hdR_p{j}(:,1));
    Ydot_fit = polyval(coeffs_dY, hdR_p{j}(:,1));
    Zdot_fit = polyval(coeffs_dZ, hdR_p{j}(:,1));

    partial_vts = [];
    partial_rts = [];
    l = 1;
    k = 1;
    while (l <= length(hdR_p{j}(:,1)))
        if(abs(hdR{j}(l,1) - Full_vts{j}(k,1)) < 1e-10) % Matching time index
            partial_vts(l,:) = Full_vts{j}(k,:);
            partial_rts(l,:) = Full_ts{j}(k,:);
            l = l + 1;
        end
        k = k + 1;
    end

    Xot_fitted{j} = [X_fit(end,1); Y_fit(end,1); Z_fit(end,1); Xdot_fit(end,1); Ydot_fit(end,1); Zdot_fit(end,1)];
    Xot_truth{j} = [partial_rts(end,2:4), partial_vts(end,2:4)]';

    X0cloud{j} = zeros(Lp,6);

    t_truth = partial_rts(end,1);
    [idx_prop, ~] = find(Full_ts{j}(:,1) == t_truth);
    Xprop_truth{j} = [Full_ts{j}(idx_prop+1,2:4), Full_vts{j}(idx_prop+1,2:4)]';

    for i = 1:length(X0cloud{j}(:,1))
        X0cloud{j}(i,:) = stateEstCloud(pf, theta_f, R_f, Partial_ts{j}, (Partial_ts{j}(2,1) - Partial_ts{j}(1,1)) + 1e-15);
    end
end

% Number of shades per color
numShades = 8;

% colors = ["#0000ff", "#0020ff", "#0040ff", "#0060ff", "#0080ff", "#00a0ff", "#00c0ff", "#00e0ff", ...
%     "#ff0000", "#ff2000", "#ff4000", "#ff6000", "#ff8000", "#ffa000", "#ffc000", "#ffe000"];
colors = ["Red", "Blue", "Green", "Yellow", "Magenta", "Cyan", ...
    "#500000", "#bf5700", "#00274c", "#ba0c2f", "#a7b1b7", "#cfb991", ...
    "#ebd99f", "#c4bfc0", "#ff5f05", "Black"];

figure(1)
set(gcf, 'units','normalized','outerposition',[0 0 1 1])
subplot(2,3,1)
for j = 1:Nt
    plot(dist2km*X0cloud{j}(:,1), dist2km*X0cloud{j}(:,2), '.')
    hold on;
end
for j = 1:Nt
    plot(dist2km*Xot_truth{j}(1), dist2km*Xot_truth{j}(2), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('X-Y');
xlabel('X (km.)');
ylabel('Y (km.)');
legend('Estimate 1', 'Estimate 2', 'Truth 1', 'Truth 2');
hold off;

subplot(2,3,2)
for j = 1:Nt
    plot(dist2km*X0cloud{j}(:,1), dist2km*X0cloud{j}(:,3), '.')
    hold on;
end
for j = 1:Nt
    plot(dist2km*Xot_truth{j}(1), dist2km*Xot_truth{j}(3), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('X-Z');
xlabel('X (km.)');
ylabel('Z (km.)');
legend('Estimate 1', 'Estimate 2', 'Truth 1', 'Truth 2');
hold off;

subplot(2,3,3)
for j = 1:Nt
    plot(dist2km*X0cloud{j}(:,2), dist2km*X0cloud{j}(:,3), '.')
    hold on;
end
for j = 1:Nt
    plot(dist2km*Xot_truth{j}(2), dist2km*Xot_truth{j}(3), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('Y-Z');
xlabel('Y (km.)');
ylabel('Z (km.)');
legend('Estimate 1', 'Estimate 2', 'Truth 1', 'Truth 2');
hold off;

subplot(2,3,4)
for j = 1:Nt
    plot(vel2kms*X0cloud{j}(:,4), vel2kms*X0cloud{j}(:,5), '.')
    hold on;
end
for j = 1:Nt
    plot(vel2kms*Xot_truth{j}(4), vel2kms*Xot_truth{j}(5), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('Xdot-Ydot');
xlabel('Xdot (km/s)');
ylabel('Ydot (km/s)');
legend('Estimate 1', 'Estimate 2', 'Truth 1', 'Truth 2');
hold off;

subplot(2,3,5)
for j = 1:Nt
    plot(vel2kms*X0cloud{j}(:,4), vel2kms*X0cloud{j}(:,6), '.')
    hold on;
end
for j = 1:Nt
    plot(vel2kms*Xot_truth{j}(4), vel2kms*Xot_truth{j}(6), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('Xdot-Zdot');
xlabel('Xdot (km/s)');
ylabel('Zdot (km/s)');
legend('Estimate 1', 'Estimate 2', 'Truth 1', 'Truth 2');
hold off;

subplot(2,3,6)
for j = 1:Nt
    plot(vel2kms*X0cloud{j}(:,5), vel2kms*X0cloud{j}(:,6), '.')
    hold on;
end
for j = 1:Nt
    plot(vel2kms*Xot_truth{j}(5), vel2kms*Xot_truth{j}(6), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('Ydot-Zdot');
xlabel('Ydot (km/s)');
ylabel('Zdot (km/s)');
legend('Estimate 1', 'Estimate 2', 'Truth 1', 'Truth 2');
hold off;

sg = sprintf('Timestep: %3.4f Hours', t_truth*time2hr);
sgtitle(sg);
savefig(gcf, 'iodCloud_mult.fig');
saveas(gcf, './Multi_Sims/iodCloud.png', 'png');
% saveas(gcf, './Simulations/Different Orbit Simulations/iodCloud.png', 'png');

% Extract important time points from the noised_obs variable

CTimes = cell(1,Nt); % Array of important time points
for j = 1:Nt
    i = 2;
    interval = noised_obs{j}(2,1) - noised_obs{j}(1,1);

    cTimes = [];
    while (i <= length(noised_obs{j}(:,1)))
        if (noised_obs{j}(i,1) - noised_obs{j}(i-1,1) > (interval+1e-11))
            cTimes = [cTimes, noised_obs{j}(i-1,1), noised_obs{j}(i,1)];
        end
        i = i + 1;
    end
    CTimes{j} = cTimes;
end

cVal = zeros(1,Nt);
larger_diff = zeros(1,Nt);
for i = 1:Nt
    larger_diff = CTimes{i}(2) - CTimes{i}(1);
    for j = 2:length(noised_obs{i}(:,1))
        if (noised_obs{i}(j,1) - noised_obs{i}(j-1,1) > (larger_diff+1e-11))
            cVal(i) = noised_obs{i}(j,1); break;
        end
    end
    if(cVal(i) == 0) % If, after all that, the break statement is avoided
        cVal(i) = CTimes{i}(2);
    end
end

t_int = zeros(1,Nt); tspan = cell(1,Nt);
parfor j = 1:Nt
    t_int(j) = hdR_p{j}(end,1); % Time at which we are obtaining a state cloud
    interval = noised_obs{j}(2,1) - noised_obs{j}(1,1);
    tspan{j} = 0:interval:interval; % Integrate over just a single time step
end
Xm_cloud = X0cloud;

parfor j = 1:Nt
    for i = 1:length(X0cloud{j}(:,1))
        % First, convert from X_{ot} in the topocentric frame to X_{bt} in the
        % synodic frame.
        Xbt = backConvertSynodic(X0cloud{j}(i,:)', t_int(j));
    
        % Next, propagate each X_{bt} in your particle cloud by a single time 
        % step and convert back to the topographic frame.
         % Call ode45()
        opts = odeset('Events', @termSat);
        interval = noised_obs{j}(2,1) - noised_obs{j}(1,1);
        [t,X] = ode15s(@cr3bp_dyn, [0 interval], Xbt, opts); % Assumes termination event (i.e. target enters LEO)
        Xm_bt = X(end,:)';
        Xm_cloud{j}(i,:) = convertToTopo(Xm_bt, t_int(j) + interval);
        % Xm_cloud(i,:) = procNoise(Xm_cloud(i,:)); % Adds process noise
    end
end

% Initialize variables

Kn = 6; % Number of clusters (original)
K = Kn; % Number of clusters (changeable)
Kmax = 8; % Maximum number of clusters (Kmax = 1 for EnKF)

mu_c = cell(K, Nt);
P_c = cell(K, Nt);
wm = zeros(K, Nt);
cPoints = cell(K,Nt);
idx = zeros(Lp,Nt);

% Split propagated cloud into position and velocity data before
% normalization.

for j = 1:Nt
    rc = Xm_cloud{j}(:,1:3);
    vc = Xm_cloud{j}(:,4:6);
    
    mean_rc = mean(rc, 1);
    mean_vc = mean(vc, 1);
    
    std_rc = std(rc,0,1);
    std_vc = std(vc,0,1);
    
    norm_rc = (rc - mean_rc)./norm(std_rc); % Normalizing the position 
    norm_vc = (vc - mean_vc)./norm(std_vc); % Normalizing the velocity
    
    Xm_norm = [norm_rc, norm_vc];
    
    % Cluster using K-means clustering algorithm
    % [idx, C] = kmeans(Xm_cloud, K); 
    [idx(:,j), C] = kmeans(Xm_norm, K); % Cluster just on position and velocity; Normalize the whole thing
    contourCols = lines(6);
    
    % Convert cluster centers back to non-dimensionalized units
    C_unorm = C;
    C_unorm(:,1:3) = (C(:,1:3).*std_rc) + mean_rc; % Conversion of position
    C_unorm(:,4:6) = (C(:,4:6).*std_vc) + mean_vc; % Conversion of velocity

    % Calculate covariances and weights for each cluster
    for k = 1:K
        cluster_points = Xm_cloud{j}(idx(:,j) == k, :); % Keep clustering very separate from mean, covariance, weight calculations
        cPoints{k,j} = cluster_points; cSize = size(cPoints{k,j});
        mu_c{k,j} = mean(cluster_points, 1); % Cell of GMM means 
    
        if(cSize(1) == 1)
            P_c{k,j} = zeros(length(wm));
        else
            P_c{k,j} = cov(cluster_points); % Cell of GMM covariances 
        end
        wm(k,j) = size(cluster_points, 1) / size(Xm_norm, 1); % Vector of weights
    end
end
    
Xprop_truth = cell(1,Nt);
for j = 1:Nt
    interval = noised_obs{j}(2,1) - noised_obs{j}(1,1);
    [idx_prop, ~] = find(abs(Full_ts{j}(:,1) - (t_int(j) + interval)) < 1e-10);
    Xprop_truth{j} = [Full_ts{j}(idx_prop,2:4), Full_vts{j}(idx_prop,2:4)];
end
tpr = Full_ts{1}(idx_prop,1); % Time stamp of the prior means, weights, and covariances

legend_string = "Truth";
% Plot the results
warning('off', 'MATLAB:legend:IgnoringExtraEntries');

% Plot planar projections
figure(2)
set(gcf, 'units','normalized','outerposition',[0 0 1 1])
subplot(2,3,1)
for k = 1:Nt
    scatter(dist2km*Xm_cloud{k}(:,1), dist2km*Xm_cloud{k}(:,2), 'filled', ...
        'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
    hold on;
end
for j = 1:Nt
    plot(dist2km*Xprop_truth{j}(1), dist2km*Xprop_truth{j}(2), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('X-Y');
xlabel('X (km.)');
ylabel('Y (km.)');
legend(legend_string);
hold off;

subplot(2,3,2)
for k = 1:Nt
    scatter(dist2km*Xm_cloud{k}(:,1), dist2km*Xm_cloud{k}(:,3), 'filled', ...
        'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
    hold on;
end
for j = 1:Nt
    plot(dist2km*Xprop_truth{j}(1), dist2km*Xprop_truth{j}(3), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('X-Z');
xlabel('X (km.)');
ylabel('Z (km.)');
legend(legend_string);
hold off;

subplot(2,3,3)
for k = 1:Nt
    scatter(dist2km*Xm_cloud{k}(:,2), dist2km*Xm_cloud{k}(:,3), 'filled', ...
        'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
    hold on;
end
for j = 1:Nt
    plot(dist2km*Xprop_truth{j}(2), dist2km*Xprop_truth{j}(3), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('Y-Z');
xlabel('Y (km.)');
ylabel('Z (km.)');
legend(legend_string);
hold off;

subplot(2,3,4)
for k = 1:Nt
    scatter(vel2kms*Xm_cloud{k}(:,4), vel2kms*Xm_cloud{k}(:,5), 'filled', ...
        'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
    hold on;
end
for j = 1:Nt
    plot(vel2kms*Xprop_truth{j}(4), vel2kms*Xprop_truth{j}(5), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('Xdot-Ydot');
xlabel('Xdot (km/s)');
ylabel('Ydot (km/s)');
legend(legend_string);
hold off;

subplot(2,3,5)
for k = 1:Nt
    scatter(vel2kms*Xm_cloud{k}(:,4), vel2kms*Xm_cloud{k}(:,6), 'filled', ...
        'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
    hold on;
end
for j = 1:Nt
    plot(vel2kms*Xprop_truth{j}(4), vel2kms*Xprop_truth{j}(6), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('Xdot-Zdot');
xlabel('Xdot (km/s)');
ylabel('Zdot (km/s)');
legend(legend_string);
hold off;

subplot(2,3,6)
for k = 1:Nt
    scatter(vel2kms*Xm_cloud{k}(:,5), vel2kms*Xm_cloud{k}(:,6), 'filled', ...
        'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
    hold on;
end
for j = 1:Nt
    plot(vel2kms*Xprop_truth{j}(5), vel2kms*Xprop_truth{j}(6), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('Ydot-Zdot');
xlabel('Ydot (km/s)');
ylabel('Zdot (km/s)');
legend(legend_string);
hold off;

sg = sprintf('Timestep: %3.4f Hours (Prior)', time2hr*Full_ts{1}(idx_prop,1));
sgtitle(sg);
saveas(gcf, './Multi_Sims/Timestep_0_1B1', 'png');

% Plot planar projections
figure(3)
set(gcf, 'units','normalized','outerposition',[0 0 1 1])
subplot(2,3,1)
for j = 1:Nt
    for k = 1:K
        clusterPoints = Xm_cloud{j}(idx(:,j) == k, :);
        scatter(dist2km*clusterPoints(:,1), dist2km*clusterPoints(:,2), 'filled', ...
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
        hold on;
    end
end
for j = 1:Nt
    plot(dist2km*Xprop_truth{j}(1), dist2km*Xprop_truth{j}(2), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('X-Y');
xlabel('X (km.)');
ylabel('Y (km.)');
legend(legend_string);
hold off;

subplot(2,3,2)
for j = 1:Nt
    for k = 1:K
        clusterPoints = Xm_cloud{j}(idx(:,j) == k, :);
        scatter(dist2km*clusterPoints(:,1), dist2km*clusterPoints(:,3), 'filled', ...
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
        hold on;
    end
end
for j = 1:Nt
    plot(dist2km*Xprop_truth{j}(1), dist2km*Xprop_truth{j}(3), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('X-Z');
xlabel('X (km.)');
ylabel('Z (km.)');
legend(legend_string);
hold off;

subplot(2,3,3)
for j = 1:Nt
    for k = 1:K
        clusterPoints = Xm_cloud{j}(idx(:,j) == k, :);
        scatter(dist2km*clusterPoints(:,2), dist2km*clusterPoints(:,3), 'filled', ...
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
        hold on;
    end
end
for j = 1:Nt
    plot(dist2km*Xprop_truth{j}(2), dist2km*Xprop_truth{j}(3), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('Y-Z');
xlabel('Y (km.)');
ylabel('Z (km.)');
legend(legend_string);
hold off;

subplot(2,3,4)
for j = 1:Nt
    for k = 1:K
        clusterPoints = Xm_cloud{j}(idx(:,j) == k, :);
        scatter(vel2kms*clusterPoints(:,4), vel2kms*clusterPoints(:,5), 'filled', ...
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
        hold on;
    end
end
for j = 1:Nt
    plot(vel2kms*Xprop_truth{j}(4), vel2kms*Xprop_truth{j}(5), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('Xdot-Ydot');
xlabel('Xdot (km/s)');
ylabel('Ydot (km/s)');
legend(legend_string);
hold off;

subplot(2,3,5)
for j = 1:Nt
    for k = 1:K
        clusterPoints = Xm_cloud{j}(idx(:,j) == k, :);
        scatter(vel2kms*clusterPoints(:,4), vel2kms*clusterPoints(:,6), 'filled', ...
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
        hold on;
    end
end
for j = 1:Nt
    plot(vel2kms*Xprop_truth{j}(4), vel2kms*Xprop_truth{j}(6), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('Xdot-Zdot');
xlabel('Xdot (km/s)');
ylabel('Zdot (km/s)');
legend(legend_string);
hold off;

subplot(2,3,6)
for j = 1:Nt
    for k = 1:K
        clusterPoints = Xm_cloud{j}(idx(:,j) == k, :);
        scatter(vel2kms*clusterPoints(:,5), vel2kms*clusterPoints(:,6), 'filled', ...
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
        hold on;
    end
end
for j = 1:Nt
    plot(vel2kms*Xprop_truth{j}(5), vel2kms*Xprop_truth{j}(6), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('Ydot-Zdot');
xlabel('Ydot (km/s)');
ylabel('Zdot (km/s)');
legend(legend_string);
hold off;

sg = sprintf('Clustered Timestep: %3.4f Hours (Prior)', time2hr*Full_ts{1}(idx_prop,1));
sgtitle(sg);
saveas(gcf, './Multi_Sims/Timestep_0_1B2', 'png');

% Now that we have a GMM representing the prior distribution, we have to
% use a Kalman update for each component: weight, mean, and covariance.

% Posterior variables
wp = wm;
mu_p = mu_c;
P_p = P_c;
Xp_cloud = Xm_cloud;
c_id = zeros(length(Xp_cloud{j}(:,1)),Nt);

% Comment this out if you wish to use noise.
% noised_obs = partial_ts;

h = @(x) [atan2(x(2),x(1)); pi/2 - acos(x(3)/sqrt(x(1)^2 + x(2)^2 + x(3)^2))]; % Nonlinear measurement model
R_vv = [theta_f*pi/648000, 0; 0, theta_f*pi/648000].^2;
zt = zeros(Nt, length(h(Xprop_truth{j})));

% Obtain measurements without automatic association
for j = 1:Nt
    [idx_meas, ~] = find(abs(noised_obs{j}(:,1) - tpr) < 1e-10); % Find row with time
    
    if (idx_meas ~= 0) % i.e. there exists a measurement
        zti = getNoisyMeas(Xprop_truth{j}, R_vv, h);
        zt(j,:) = zti;
    else
        zti(j,:) = [NaN, NaN];
    end
end
zt_truth = zt
zt = zt(randperm(size(zt,1)), :)
zt_perm = zt

zt = dataAssoc(cPoints, wm, zt_perm, h, R_vv)

if any(any(abs(zt - zt_truth) > 1e-5 & ~isnan(zt)))
    fprintf("Association not performed correctly with Munkres Algorithm!\n");
    fprintf("Munkres Association: \n");
    disp(zt);
    fprintf("Actual Association: \n");
    disp(zt_truth);
else
    fprintf("Association performed correctly with Munkres Algorithm!\n");
end

for j = 1:Nt
    if(idx_meas ~= 0)  
        for i = 1:K 
            % [mu_p{i}, P_p{i}] = ukfUpdate(zt, R_vv, mu_c{i}, P_c{i}, h);
            [mu_p{i,j}, P_p{i,j}] = kalmanUpdate(zt(j,:)', cPoints{i,j}, R_vv, mu_c{i,j}, P_c{i,j}, h);
        end
    
        % Weight update
        wp(:,j) = weightUpdate(wm(:,j), Xm_cloud{j}, idx(:,j), zt(j,:)', R_vv, h);
    else
        for i = 1:K
            wp(i,j) = wm(i,j);
            mu_p{i,j} = mu_c{i,j};
            P_p{i,j} = P_c{i,j};
        end
    end
    
    for i = 1:Lp
        [Xp_cloud{j}(i,:), c_id(i,j)] = drawFrom(wp(:,j), mu_p(:,j), P_p(:,j)); 
    end
end

fprintf("First Step Prior Weights:\n")
disp(wm);
fprintf("First Step Posterior Weights:\n")
disp(wp);

mu_pExp = zeros(K, length(mu_p{1}));

% Plot the results

% legend_string = {"Truth 1", "Truth 2"};

% Plot planar projections
figure(4)
set(gcf, 'units','normalized','outerposition',[0 0 1 1])
subplot(2,3,1)
for j = 1:Nt
    for k = 1:K
        clusterPoints = Xp_cloud{j}(c_id(:,j) == k, :);
        mu_pExp(k,:) = mu_p{k};
        scatter(dist2km*clusterPoints(:,1), dist2km*clusterPoints(:,2), 'filled', ...
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
        hold on;
    end
end
for j = 1:Nt
    plot(dist2km*Xprop_truth{j}(1), dist2km*Xprop_truth{j}(2), 'x','MarkerSize', 20, 'LineWidth', 3)
    hold on;
end
title('X-Y');
xlabel('X (km.)');
ylabel('Y (km.)');
legend(legend_string);
hold off;

subplot(2,3,2)
for j = 1:Nt
    for k = 1:K
        clusterPoints = Xp_cloud{j}(c_id(:,j) == k, :);
        mu_pExp(k,:) = mu_p{k};
        scatter(dist2km*clusterPoints(:,1), dist2km*clusterPoints(:,3), 'filled', ...
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
        hold on;
    end
end
for j = 1:Nt
    plot(dist2km*Xprop_truth{j}(1), dist2km*Xprop_truth{j}(3), 'x','MarkerSize', 20, 'LineWidth', 3)
    hold on;
end
title('X-Z');
xlabel('X (km.)');
ylabel('Z (km.)');
legend(legend_string);
hold off;

subplot(2,3,3)
for j = 1:Nt
    for k = 1:K
        clusterPoints = Xp_cloud{j}(c_id(:,j) == k, :);
        mu_pExp(k,:) = mu_p{k};
        scatter(dist2km*clusterPoints(:,2), dist2km*clusterPoints(:,3), 'filled', ...
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
        hold on;
    end
end
for j = 1:Nt
    plot(dist2km*Xprop_truth{j}(2), dist2km*Xprop_truth{j}(3), 'x','MarkerSize', 20, 'LineWidth', 3)
    hold on;
end
title('Y-Z');
xlabel('Y (km.)');
ylabel('Z (km.)');
legend(legend_string);
hold off;

subplot(2,3,4)
for j = 1:Nt
    for k = 1:K
        clusterPoints = Xp_cloud{j}(c_id(:,j) == k, :);
        mu_pExp(k,:) = mu_p{k};
        scatter(vel2kms*clusterPoints(:,4), vel2kms*clusterPoints(:,5), 'filled', ... 
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
        hold on;
    end
end
for j = 1:Nt
    plot(vel2kms*Xprop_truth{j}(4), vel2kms*Xprop_truth{j}(5), 'x','MarkerSize', 20, 'LineWidth', 3)
    hold on;
end
title('Xdot-Ydot');
xlabel('Xdot (km/s)');
ylabel('Ydot (km/s)');
legend(legend_string);
hold off;

subplot(2,3,5)
for j = 1:Nt
    for k = 1:K
        clusterPoints = Xp_cloud{j}(c_id(:,j) == k, :);
        mu_pExp(k,:) = mu_p{k};
        scatter(vel2kms*clusterPoints(:,4), vel2kms*clusterPoints(:,6), 'filled', ... 
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
        hold on;
    end
end
for j = 1:Nt
    plot(vel2kms*Xprop_truth{j}(4), vel2kms*Xprop_truth{j}(6), 'x','MarkerSize', 20, 'LineWidth', 3)
    hold on;
end
title('Xdot-Zdot');
xlabel('Xdot (km/s)');
ylabel('Zdot (km/s)');
legend(legend_string);
hold off;

subplot(2,3,6)
for j = 1:Nt
    for k = 1:K
        clusterPoints = Xp_cloud{j}(c_id(:,j) == k, :);
        mu_pExp(k,:) = mu_p{k};
        scatter(vel2kms*clusterPoints(:,5), vel2kms*clusterPoints(:,6), 'filled', ... 
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
        hold on;
    end
end
for j = 1:Nt
    plot(vel2kms*Xprop_truth{j}(5), vel2kms*Xprop_truth{j}(6), 'x','MarkerSize', 20, 'LineWidth', 3)
    hold on;
end
title('Ydot-Zdot');
xlabel('Ydot (km/s)');
ylabel('Zdot (km/s)');
legend(legend_string);
hold off;

sg = sprintf('Timestep: %3.4f Hours (Posterior)', time2hr*Full_ts{1}(idx_prop,1));
sgtitle(sg);
saveas(gcf,'./Multi_Sims/Timestep_0_2B.png', 'png')
% saveas(gcf,'./Simulations/Different Orbit Simulations/Timestep_0_2B.png', 'png')
%}

% At this point, we have shown a PGM-I propagation and update step. The
% next step is to utilize this PGM-I update across all time steps during
% which the target is within our sensor FOV and see how the particle clouds
% (i.e. GM components) evolve over time. If we're lucky, we should see that
% the GMM tracks the truth over the interval.

% Find and set the start and end times to simulation
[idx_meas, ~] = find(abs(Partial_ts{1}(:,1) - tpr) < 1e-10);
% interval = hdR(idx_meas,c_meas) - hdR(idx_meas-1,c_meas);

% [idx_crit, ~] = find(abs(Full_ts{1}(:,1)) >= (5*24)/time2hr, 1, 'first'); % Find the index of the last time step before a certain number of days have passed since orbit propagation
t_end = max(cVal); % End simulation at first point at which we get an observation from both targets
% t_end = Full_ts{1}(end,1); % First observation of new pass + one more time step

tau = 0;
[idx_end, ~] = find(abs(Full_ts{1}(:,1) - t_end) < 1e-10);
[idx_start, ~] = find(abs(Full_ts{1}(:,1) - tpr) < 1e-10);

l_filt = length(Full_ts{1}(idx_start:idx_end,1))+1;

ent2 = zeros(l_filt+1,Nt);
ent1 = zeros(Nt,l_filt+1,length(Xprop_truth{1})); Xp_cloudp = Xp_cloud;

for i = 1:Nt
    ent2(1,i) = log(det(cov(X0cloud{j})));
    ent2(2,i) = log(det(cov(Xp_cloudp{j})));
    ent1(i,1,:) = getDiagCov(X0cloud{j}); 
end

cPoints = cell(K, Nt); c_id = zeros(Lp,Nt);
mu_c = cell(K, Nt); mu_p = mu_c; idx = cell(1,Nt);
P_c = cell(K, Nt); P_p = P_c;
wm = zeros(K, Nt); wp = wm;
zt = zeros(Nt, length(h(Xprop_truth{Nt}))); 

%% Main Loop
% for to = tpr:interval:(t_end-1e-11) % Looping over the times of observation for easier propagation
for ts = idx_start:(idx_end-1) 

    % Step 1: Propagate all target estimates and truth
    for b = 1:Nt
        to = Full_ts{b}(ts,1);
        interval = Full_ts{b}(ts+1,1) - Full_ts{b}(ts,1);
    
        ent1(b,tau+2,:) = getDiagCov(Xp_cloudp{b});
    
        % Propagation Step
        Xm_cloud{b} = propagate(Xp_cloudp{b}, to, interval);
        Xprop_truth{b} = propagate(Xprop_truth{b}, to, interval);
    
        % Verification Step
        tpr = to + interval; % Time stamp of the prior means, weights, and covariances
    end

    % Steps 2 & 3: Check how many measurements have occurred. Cluster if a
    % measurement has occurred

    meas_num = 0; % Keeps track of number of measurements

    for b = 1:Nt
        if (tpr >= cVal(b))
            K = Kmax;
        else
            K = Kn;
        end
    end

    wm = zeros(K,Nt); wp = wm;

    for b = 1:Nt
        [idx_meas, ~] = find(abs(noised_obs{b}(:,1) - tpr) < 1e-10); % Find row with time

        if (idx_meas ~= 0) % i.e. there exists a measurement
            zti = getNoisyMeas(Xprop_truth{b}, R_vv, h);
            zt(b,:) = zti;
            meas_num = meas_num + 1;

            fprintf("Timestamp: %1.5f\n", tpr*time2hr);

            % Split propagated cloud into position and velocity data before
            % normalization.
            
            rc = Xm_cloud{b}(:,1:3);
            vc = Xm_cloud{b}(:,4:6);
        
            mean_rc = mean(rc, 1);
            mean_vc = mean(vc, 1);
        
            std_rc = std(rc,0,1);
            std_vc = std(vc,0,1);
        
            norm_rc = (rc - mean_rc)./norm(std_rc); % Normalizing the position 
            norm_vc = (vc - mean_vc)./norm(std_vc); % Normalizing the velocity
        
            Xm_norm = [norm_rc, norm_vc];
        
            % Verification Step
            [idx_meas, ~] = find(abs(noised_obs{b}(:,1) - tpr) < 1e-10); % Find row with time
            
            % Cluster using K-means clustering algorithm
            [idx{b}, ~] = kmeans(Xm_norm, K);
    
            % Calculate covariances and weights for each cluster
            for k = 1:K
                cluster_points = Xm_cloud{b}(idx{b} == k, :); 
                cPoints{k,b} = cluster_points; 
                mu_c{k,b} = mean(cluster_points, 1); % Cell of GMM means 
                if (length(cluster_points(:,1)) == 1)
                    P_c{k,b} = zeros(length(mu_c{k,b}));
                else
                    P_c{k,b} = cov(cluster_points); % Cell of GMM covariances
                end
                wm(k,b) = size(cluster_points, 1) / size(Xm_cloud{b}, 1); % Vector of (prior) weights
            end    

        else
            zt(b,:) = [NaN, NaN];
        end
    end

    % Step 4: Associate measurements with estimates 

    zt_truth = zt;
    zt = zt(randperm(size(zt,1)), :);
    % zt_perm = zt;

    fprintf("Number of measurements: %d\n", meas_num);

    for q1 = 1:Nt
        for q2 = 1:Nt
            if (q1 ~= q2)
                fprintf("Distance between targets %d and %d: %8.7f km.\n", ...
                    q1, q2, dist2km*norm(Xprop_truth{q1}(1:3) - Xprop_truth{q2}(1:3)))
            end
        end
    end

    if (meas_num == Nt)
        zt = dataAssoc(cPoints, wm, zt, h, R_vv);
    elseif (meas_num > 0)
        zt = zt_truth; % Need to replace this with a single-observation association function
    end

    % Check if association is done correctly
    % if(norm(zt_truth - zt) > 1e-10)
    if any(any(abs(zt - zt_truth) > 1e-5 & ~isnan(zt)))
        fprintf("Association not performed correctly with Munkres Algorithm!\n");
        fprintf("Munkres Association: \n");
        disp(zt);
        fprintf("Actual Association: \n");
        disp(zt_truth);
    else
        fprintf("Association performed correctly with Munkres Algorithm!\n");
    end

    % Step 4: Update both PDFs if an observation has occurred
    for b = 1:Nt
        [idx_meas, ~] = find(abs(noised_obs{b}(:,1) - tpr) < 1e-10); % Find row with time

        if(idx_meas ~= 0)
            
            % Update Step
            noised_obs{b}(idx_meas,end-1:end) = zt(b,:); % Replace noised_obs vector values at beginning with actual observation values as associated
    
            for i = 1:K
                % [mu_p{i}, P_p{i}] = ukfUpdate(zt, R_vv, mu_c{i}, P_c{i}, h);
                [mu_p{i,b}, P_p{i,b}] = kalmanUpdate(zt(b,:)', cPoints{i,b}, R_vv, mu_c{i,b}, P_c{i,b}, h);
                P_p{i,b} = (P_p{i,b} + P_p{i,b}')/2;
            end
    
            % Weight update
            fprintf("Measurement Update Target %i:\n", b);
            fprintf("A Priori Weights:\n");
            disp(wm(:,b));
            wp(:,b) = weightUpdate(wm(:,b), Xm_cloud{b}, idx{b}, zt(b,:), R_vv, h);

            % Resampling
            % K = Kn;
            Xp_cloudp{b} = Xm_cloud{b};
            
            for i = 1:Lp
                [Xp_cloudp{b}(i,:), c_id(i,b)] = drawFrom(wp(:,b), mu_p(:,b), P_p(:,b)); 
            end

            wsum = 0;
            for k = 1:K
                wsum = wsum + wp(k,b)*det(P_p{k,b});
            end
            ent2(tau+2,b) = log(wsum);
        else
            fprintf("Timestamp: %1.5f\n", tpr*time2hr);
            
            K = 1;
    
            Xp_cloud{b} = Xm_cloud{b}; Xp_cloudp{b} = Xp_cloud{b};
            cPoints{b} = Xp_cloud{b}; c_id(:,b) = ones(length(Xp_cloudp{b}(:,1)),1);
            mu_p{K,b} = mean(Xp_cloud{b});
            P_p{K,b} = cov(Xp_cloud{b});

            if (tpr >= cVal(b))
                Ke = Kmax; % Clusters used for calculating entropy
            else
                Ke = Kn; % Clusters used for calculating entropy
            end
            ent2(tau+2,b) = getKnEntropy(Ke, Xp_cloudp{b}); % Get entropy as if you still are using six clusters
        end
    end

    fprintf("Prior Weights:\n")
    disp(wm);
    fprintf("Posterior Weights:\n")
    disp(wp);

    tau = tau + 1;
    % legend_string = {"Truth 1", "Truth 2"};

    isCrit = any(cellfun (@(v) any(abs(v - tpr) < 1e-11), CTimes));
    % if (isCrit)
    if(1) % Use for all time steps
        %{
        % legend_string{K+1} = "Centroids";
        legend_string{K+1} = "Truth";

        mu_mat = cell2mat(mu_c);
        P_mat = cat(3, P_c{:});

        
        f = figure('visible','off','Position', get(0,'ScreenSize'));
        f.WindowState = 'maximized';

        subplot(2,3,1)
        plot_dims = [1,2];
        mu_marg = mu_mat(:, plot_dims);
        P_marg = P_mat(plot_dims, plot_dims, :);
        
        [X1, X2] = meshgrid(linspace(min(Xm_cloud(:,plot_dims(1))), max(Xm_cloud(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud(:,plot_dims(2))), max(Xm_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];

        Z_cell = cell(K,1); contours_cell = cell(K,1); 
        
        parfor k = 1:K
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
            contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
        end 

        hold on;
        for k = 1:K
            contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end    
        plot(dist2km*Xprop_truth(1), dist2km*Xprop_truth(2), 'kx','MarkerSize', 15, 'LineWidth', 3);
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
        
        [X1, X2] = meshgrid(linspace(min(Xm_cloud(:,plot_dims(1))), max(Xm_cloud(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud(:,plot_dims(2))), max(Xm_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];

        Z_cell = cell(K,1); contours_cell = cell(K,1); 
        
        parfor k = 1:K
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
            contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
        end 
        
        hold on;
        for k = 1:K
            contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end    
        plot(dist2km*Xprop_truth(1), dist2km*Xprop_truth(3), 'kx','MarkerSize', 15, 'LineWidth', 3);
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
        
        [X1, X2] = meshgrid(linspace(min(Xm_cloud(:,plot_dims(1))), max(Xm_cloud(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud(:,plot_dims(2))), max(Xm_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];

        Z_cell = cell(K,1); contours_cell = cell(K,1); 
        
        parfor k = 1:K
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
            contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
        end 
        
        hold on;
        for k = 1:K
            contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end    
        plot(dist2km*Xprop_truth(2), dist2km*Xprop_truth(3), 'kx','MarkerSize', 15, 'LineWidth', 3);
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
        
        [X1, X2] = meshgrid(linspace(min(Xm_cloud(:,plot_dims(1))), max(Xm_cloud(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud(:,plot_dims(2))), max(Xm_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];

        Z_cell = cell(K,1); contours_cell = cell(K,1); 
        
        parfor k = 1:K
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
            contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
        end 
        
        hold on;
        for k = 1:K
            contour(vel2kms*X1, vel2kms*X2, vel2kms*Z_cell{k}, vel2kms*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end    
        plot(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(5), 'kx','MarkerSize', 15, 'LineWidth', 3);
        title('Xdot-Ydot');
        xlabel('Xdot (km/s)');
        ylabel('Ydot (km/s)');
        legend(legend_string);
        hold off;

        subplot(2,3,5)
        plot_dims = [4,6];
        mu_marg = mu_mat(:, plot_dims);
        P_marg = P_mat(plot_dims, plot_dims, :);
        
        [X1, X2] = meshgrid(linspace(min(Xm_cloud(:,plot_dims(1))), max(Xm_cloud(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud(:,plot_dims(2))), max(Xm_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];

        Z_cell = cell(K,1); contours_cell = cell(K,1); 
        
        parfor k = 1:K
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
            contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
        end 
        
        hold on;
        for k = 1:K
            contour(vel2kms*X1, vel2kms*X2, vel2kms*Z_cell{k}, vel2kms*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end    
        plot(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(6), 'kx','MarkerSize', 15, 'LineWidth', 3);
        title('Xdot-Zdot');
        xlabel('Xdot (km/s)');
        ylabel('Zdot (km/s)');
        legend(legend_string);
        hold off;

        subplot(2,3,6)
        plot_dims = [5,6];
        mu_marg = mu_mat(:, plot_dims);
        P_marg = P_mat(plot_dims, plot_dims, :);
        
        [X1, X2] = meshgrid(linspace(min(Xm_cloud(:,plot_dims(1))), max(Xm_cloud(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud(:,plot_dims(2))), max(Xm_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];

        Z_cell = cell(K,1); contours_cell = cell(K,1); 
        
        parfor k = 1:K
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
            contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
        end 
        
        hold on;
        for k = 1:K
            contour(vel2kms*X1, vel2kms*X2, vel2kms*Z_cell{k}, vel2kms*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end    
        plot(vel2kms*Xprop_truth(5), vel2kms*Xprop_truth(6), 'kx','MarkerSize', 15, 'LineWidth', 3);
        title('Ydot-Zdot');
        xlabel('Ydot (km/s)');
        ylabel('Zdot (km/s)');
        legend(legend_string);
        hold off;

        sgt = sprintf('Timestep: %3.4f Hours (Prior)', tpr*time2hr);
        sgtitle(sgt);

        sg = sprintf('./Simulations/Timestep_%i_1A.png', tau);
        % sg = sprintf('./Simulations/Different Orbit Simulations/Timestep_%i_1A.png', tau);
        saveas(f, sg, 'png');
        close(f);
        %}
        
        % Count number of measurements for this epoch
        mnum = 0;
        for b = 1:Nt
            [idx_meas, ~] = find(abs(noised_obs{b}(:,1) - tpr) < 1e-10); % Find row with time
            if (idx_meas ~= 0)
                mnum = mnum + 1;
            end
        end

        if (mnum > 0)
            legend_string_x = {};
            parfor j = 1:Nt
                legend_string_x{j} = sprintf('Truth %i', j);
            end

            for b = 1:Nt
                f = figure('visible','off','Position', get(0,'ScreenSize'));
                f.WindowState = 'maximized';

                for k = 1:K
                    cPoints{k,b} = Xm_cloud{b}(idx{b} == k, :);
                end
                
                subplot(2,3,1)
                hold on; 
                for k = 1:K
                    scatter(dist2km*cPoints{k,b}(:,1), dist2km*cPoints{k,b}(:,2), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
                    hold on;
                end

                plot(dist2km*Xprop_truth{b}(1), dist2km*Xprop_truth{b}(2), 'kx', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
                hold on;
                
                title('X-Y');
                xlabel('X (km.)');
                ylabel('Y (km.)');
                legend(legend_string);
                hold off;
                
                subplot(2,3,2)
                hold on; 
                for k = 1:K
                    scatter(dist2km*cPoints{k,b}(:,1), dist2km*cPoints{k,b}(:,3), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
                    hold on;
                end
                plot(dist2km*Xprop_truth{b}(1), dist2km*Xprop_truth{b}(3), 'kx', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
                hold on;

                title('X-Z');
                xlabel('X (km.)');
                ylabel('Z (km.)');
                legend(legend_string);
                hold off;
                
                subplot(2,3,3)
                hold on;

                for k = 1:K
                    scatter(dist2km*cPoints{k,b}(:,2), dist2km*cPoints{k,b}(:,3), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
                    hold on;
                end

                plot(dist2km*Xprop_truth{b}(2), dist2km*Xprop_truth{b}(3), 'kx', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
                hold on;

                title('Y-Z');
                xlabel('Y (km.)');
                ylabel('Z (km.)');
                legend(legend_string);
                hold off;
                
                subplot(2,3,4)
                hold on;
                for k = 1:K
                    scatter(vel2kms*cPoints{k,b}(:,4), vel2kms*cPoints{k,b}(:,5), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
                end
                plot(vel2kms*Xprop_truth{b}(4), vel2kms*Xprop_truth{b}(5), 'kx', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
                hold on;

                title('Xdot-Ydot');
                xlabel('Xdot (km/s)');
                ylabel('Ydot (km/s)');
                legend(legend_string);
                hold off;
                
                subplot(2,3,5)
                hold on; 
                for k = 1:K
                    scatter(vel2kms*cPoints{k,b}(:,4), vel2kms*cPoints{k,b}(:,6), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
                end
                plot(vel2kms*Xprop_truth{b}(4), vel2kms*Xprop_truth{b}(6), 'kx', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
                hold on;

                title('Xdot-Zdot');
                xlabel('Xdot (km/s)');
                ylabel('Zdot (km/s)');
                legend(legend_string);
                hold off;
                
                subplot(2,3,6)
                hold on; 
                for k = 1:K
                    scatter(vel2kms*cPoints{k,b}(:,5), vel2kms*cPoints{k,b}(:,6), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
                end
                plot(vel2kms*Xprop_truth{b}(5), vel2kms*Xprop_truth{b}(6), 'kx', ... 
                        'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
                hold on;

                title('Ydot-Zdot');
                xlabel('Ydot (km/s)');
                ylabel('Zdot (km/s)');
                legend(legend_string);
                hold off;
        
                sgt = sprintf('Timestep: %3.4f Hours (Prior) - Target %d', tpr*time2hr, b);
                sgtitle(sgt);
        
                sg = sprintf('./Multi_Sims/Timestep_%i_1B_T%i.png', tau, b);
                % sg = sprintf('./Simulations/Different Orbit Simulations/Timestep_%i_1B.png', tau);
                saveas(f, sg, 'png');
                close(f);
            end
    
            f = figure('visible','off','Position', get(0,'ScreenSize'));
            f.WindowState = 'maximized';
    
            for b = 1:Nt
                for k = 1:K
                    cPoints{k,b} = Xm_cloud{b}(idx{b} == k, :);
                end
            end
    
            subplot(2,3,1)
            hold on; 
            for b = 1:Nt
                for k = 1:K
                    scatter(dist2km*cPoints{k,b}(:,1), dist2km*cPoints{k,b}(:,2), 'filled', ...
                    'MarkerFaceColor', colors(b), 'HandleVisibility', 'off');
                    hold on;
                end
            end
    
            for b = 1:Nt
                plot(dist2km*Xprop_truth{b}(1), dist2km*Xprop_truth{b}(2), 'x', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string_x{b});
                hold on;
            end
            title('X-Y');
            xlabel('X (km.)');
            ylabel('Y (km.)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,2)
            hold on; 
            for b = 1:Nt
                for k = 1:K
                    scatter(dist2km*cPoints{k,b}(:,1), dist2km*cPoints{k,b}(:,3), 'filled', ...
                    'MarkerFaceColor', colors(b), 'HandleVisibility', 'off');
                    hold on;
                end
            end
            for b = 1:Nt
                plot(dist2km*Xprop_truth{b}(1), dist2km*Xprop_truth{b}(3), 'x', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string_x{b});
                hold on;
            end
            title('X-Z');
            xlabel('X (km.)');
            ylabel('Z (km.)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,3)
            hold on;
            for b = 1:Nt
                for k = 1:K
                    scatter(dist2km*cPoints{k,b}(:,2), dist2km*cPoints{k,b}(:,3), 'filled', ...
                    'MarkerFaceColor', colors(b), 'HandleVisibility', 'off');
                    hold on;
                end
            end
            for b = 1:Nt
                plot(dist2km*Xprop_truth{b}(2), dist2km*Xprop_truth{b}(3), 'x', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string_x{b});
                hold on;
            end
            title('Y-Z');
            xlabel('Y (km.)');
            ylabel('Z (km.)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,4)
            hold on; 
            for b = 1:Nt
                for k = 1:K
                    scatter(vel2kms*cPoints{k,b}(:,4), vel2kms*cPoints{k,b}(:,5), 'filled', ...
                    'MarkerFaceColor', colors(b), 'HandleVisibility', 'off');
                end
            end
            for b = 1:Nt
                plot(vel2kms*Xprop_truth{b}(4), vel2kms*Xprop_truth{b}(5), 'x', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string_x{b});
                hold on;
            end
            title('Xdot-Ydot');
            xlabel('Xdot (km/s)');
            ylabel('Ydot (km/s)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,5)
            hold on; 
            for b = 1:Nt
                for k = 1:K
                    scatter(vel2kms*cPoints{k,b}(:,4), vel2kms*cPoints{k,b}(:,6), 'filled', ...
                    'MarkerFaceColor', colors(b), 'HandleVisibility', 'off');
                end
            end
            for b = 1:Nt
                plot(vel2kms*Xprop_truth{b}(4), vel2kms*Xprop_truth{b}(6), 'x', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string_x{b});
                hold on;
            end
            title('Xdot-Zdot');
            xlabel('Xdot (km/s)');
            ylabel('Zdot (km/s)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,6)
            hold on; 
            for b = 1:Nt
                for k = 1:K
                    scatter(vel2kms*cPoints{k,b}(:,5), vel2kms*cPoints{k,b}(:,6), 'filled', ...
                    'MarkerFaceColor', colors(b), 'HandleVisibility', 'off');
                end
            end
            for b = 1:Nt
                plot(vel2kms*Xprop_truth{b}(5), vel2kms*Xprop_truth{b}(6), 'x', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string_x{b});
                hold on;
            end
            title('Ydot-Zdot');
            xlabel('Ydot (km/s)');
            ylabel('Zdot (km/s)');
            legend(legend_string);
            hold off;
    
            sgt = sprintf('Timestep: %3.4f Hours (Prior)', tpr*time2hr);
            sgtitle(sgt);
    
            sg = sprintf('./Multi_Sims/Timestep_%i_1B.png', tau);
            % sg = sprintf('./Simulations/Different Orbit Simulations/Timestep_%i_1B.png', tau);
            saveas(f, sg, 'png');
            close(f);

            for j = 1:Nt
                f = figure('visible','off','Position', get(0,'ScreenSize'));
                f.WindowState = 'maximized';
        
                % legend_string = {"Truth 1", "Truth 2"};
                hold on;

                for k = 1:K
                    
                    Zmcloud = zeros(length(cPoints{k,j}(:,1)), length(h(cPoints{k,j}(1,:))));
                    for i = 1:length(Zmcloud(:,1))
                        Zmcloud(i,:) = h(cPoints{k,j}(i,:))';
                    end

                    scatter(180/pi*Zmcloud(:,1), 180/pi*Zmcloud(:,2), 'filled', ...
                    'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');    
                end
                Ztruth = h(Xprop_truth{j})';
                plot(180/pi*Ztruth(1), 180/pi*Ztruth(2), 'kx', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
                plot(180/pi*zt(j,1), 180/pi*zt(j,2), 'o', ... 
                        'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', "Assoc Meas");
                plot(180/pi*zt_truth(j,1), 180/pi*zt_truth(j,2), 'o', ... 
                        'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', "Actual Meas");
                
                sgt = sprintf('Timestep: %3.4f Hours (Prior AZ-EL) - Target %i', tpr*time2hr, j);
                sgtitle(sgt);
                xlabel('Azimuth Angle (deg)')
                ylabel('Elevation Angle (deg)')
                legend(legend_string)
        
                sg = sprintf('./Multi_Sims/Timestep_%i_1C_T%i.png', tau, j);
                % sg = sprintf('./Simulations/Different Orbit Simulations/Timestep_%i_1B.png', tau);
                saveas(f, sg, 'png');
                close(f);
            end

            f = figure('visible','off','Position', get(0,'ScreenSize'));
            f.WindowState = 'maximized';
    
            % legend_string = {"Truth 1", "Truth 2"};
            hold on;
    
            for j = 1:Nt
                
                Zm_cloud = zeros(length(Xm_cloud{j}(:,1)), length(h(cPoints{k,j}(1,:))));
                for i = 1:length(Xm_cloud{j}(:,1))
                    Zm_cloud(i,:) = h(Xm_cloud{j}(i,:));
                end
                
                scatter(180/pi*Zm_cloud(:,1), 180/pi*Zm_cloud(:,2), 'filled', ...
                'MarkerFaceColor', colors(j), 'HandleVisibility', 'off');     
                Ztruth = h(Xprop_truth{j})';
                plot(180/pi*Ztruth(1), 180/pi*Ztruth(2), 'kx', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string_x{j});  
                plot(180/pi*zt(j,1), 180/pi*zt(j,2), 'o', 'MarkerSize', 20, 'LineWidth', 3);
            end
    
            sgt = sprintf('Timestep: %3.4f Hours (Prior AZ-EL)', tpr*time2hr);
            sgtitle(sgt);
            xlabel('Azimuth Angle (deg)')
            ylabel('Elevation Angle (deg)')
    
            sg = sprintf('./Multi_Sims/Timestep_%i_1C.png', tau);
            % sg = sprintf('./Simulations/Different Orbit Simulations/Timestep_%i_1B.png', tau);
            saveas(f, sg, 'png');
            close(f);
        end
    end

    %{
    if(abs(to - (t_end-interval)) < 1e-10) % At final time step possible
        % Save the a priori estimate particle cloud
        save('aPriori.mat', 'Xm_cloud');

        % Extract means
        parfor k = 1:K
            mu_mExp(k,:) = mu_c{k};
        end

        % t_truth = to + interval;
        % [idx_final, ~] = find(abs(full_ts(:,1) - (to+interval)) < 1e-10);
        % Xprop_truth = [full_ts(idx_final,2:4), full_vts(idx_final,2:4)]';

        % Show where observation lies (position only)
        if(idx_meas ~= 0)
            zc = noised_obs(idx_meas,2:4)'; % Presumption: An observation occurs at this time step
            xto = zc(1)*cos(zc(2))*cos(zc(3)); 
            yto = zc(1)*sin(zc(2))*cos(zc(3)); 
            zto = zc(1)*sin(zc(3)); 
            rto = [xto, yto, zto];
        end

        % Plot planar projections
        figure(8)
        subplot(2,3,1)
        gscatter(dist2km*Xm_cloud(:,1), dist2km*Xm_cloud(:,2), idx);
        hold on;
        plot(dist2km*mu_mExp(:,1), dist2km*mu_mExp(:,2), '+', 'MarkerSize', 10, 'LineWidth', 3);
        hold on;
        plot(dist2km*Xprop_truth(1), dist2km*Xprop_truth(2), 'kx','MarkerSize', 15, 'LineWidth', 3);
        title('X-Y');
        xlabel('X (km.)');
        ylabel('Y (km.)');
        legend(legend_string);
        hold off;
        
        subplot(2,3,2)
        gscatter(dist2km*Xm_cloud(:,1), dist2km*Xm_cloud(:,3), idx);
        hold on;
        plot(dist2km*mu_mExp(:,1), dist2km*mu_mExp(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
        hold on;
        plot(dist2km*Xprop_truth(1), dist2km*Xprop_truth(3), 'kx','MarkerSize', 15, 'LineWidth', 3);
        title('X-Z');
        xlabel('X (km.)');
        ylabel('Z (km.)');
        legend(legend_string);
        hold off;
        
        subplot(2,3,3)
        gscatter(dist2km*Xm_cloud(:,2), dist2km*Xm_cloud(:,3), idx);
        hold on;
        plot(dist2km*mu_mExp(:,2), dist2km*mu_mExp(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
        hold on;
        plot(dist2km*Xprop_truth(2), dist2km*Xprop_truth(3), 'kx','MarkerSize', 15, 'LineWidth', 3);
        title('Y-Z');
        xlabel('Y (km.)');
        ylabel('Z (km.)');
        legend(legend_string);
        hold off;
        
        subplot(2,3,4)
        gscatter(vel2kms*Xm_cloud(:,4), vel2kms*Xm_cloud(:,5), idx);
        hold on;
        plot(vel2kms*mu_mExp(:,4), vel2kms*mu_mExp(:,5), '+', 'MarkerSize', 10, 'LineWidth', 3);
        hold on;
        plot(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(5), 'kx','MarkerSize', 15, 'LineWidth', 3)
        title('Xdot-Ydot');
        xlabel('Xdot (km/s)');
        ylabel('Ydot (km/s)');
        legend(legend_string);
        hold off;
        
        subplot(2,3,5)
        gscatter(vel2kms*Xm_cloud(:,4), vel2kms*Xm_cloud(:,6), idx);
        hold on;
        plot(vel2kms*mu_mExp(:,4), vel2kms*mu_mExp(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3);
        hold on;
        plot(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
        title('Xdot-Zdot');
        xlabel('Xdot (km/s)');
        ylabel('Zdot (km/s)');
        legend(legend_string);
        hold off;
        
        subplot(2,3,6)
        gscatter(vel2kms*Xm_cloud(:,5), vel2kms*Xm_cloud(:,6), idx);
        hold on;
        plot(vel2kms*mu_mExp(:,5), vel2kms*mu_mExp(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3);
        hold on;
        plot(vel2kms*Xprop_truth(5), vel2kms*Xprop_truth(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
        title('Ydot-Zdot');
        xlabel('Ydot (km/s)');
        ylabel('Zdot (km/s)');
        legend(legend_string);
        hold off;
        savefig(gcf, 'postClusteringDistribution.fig');
    end
    %}
  
    % if (isCrit)
    if(1)
        % [idx_trth, ~] = find(abs(full_ts(:,1) - tpr) < 1e-10);
        % Xprop_truth = [full_ts(idx_trth,2:4), full_vts(idx_trth,2:4)];

        % Extract means

        %{
        mu_pExp = zeros(K, length(mu_p{1}));
        parfor k = 1:K
            mu_pExp(k,:) = mu_p{k};
        end
    
        legend_string = {};
        parfor k = 1:K
            legend_string{k} = sprintf('Contour %i', k);
            % legend_string{K+k} = sprintf('\\omega = %1.4f', wp(k));
        end
        % legend_string{K+1} = "Centroids";
        legend_string{K+1} = "Truth";
    
        mu_mat = mu_pExp;
        P_mat = cat(3, P_p{:});
    
        
        f = figure('visible','off','Position', get(0,'ScreenSize'));
        f.WindowState = 'maximized';
    
        subplot(2,3,1)
        plot_dims = [1,2];
        mu_marg = mu_mat(:, plot_dims);
        P_marg = P_mat(plot_dims, plot_dims, :);
    
        [X1, X2] = meshgrid(linspace(min(Xm_cloud(:,plot_dims(1))), max(Xm_cloud(:,plot_dims(1))), 100), ...
                            linspace(min(Xm_cloud(:,plot_dims(2))), max(Xm_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
    
        Z_cell = cell(K,1); contours_cell = cell(K,1); 
        
        parfor k = 1:K
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
            contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
        end 
        
        hold on;
        for k = 1:K
            contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end
        plot(dist2km*Xprop_truth(1), dist2km*Xprop_truth(2), 'kx','MarkerSize', 20, 'LineWidth', 3)
        title('X-Y');
        xlabel('X (km.)');
        ylabel('Y (km.)');
        legend(legend_string);
        hold off;
    
        subplot(2,3,2)
        plot_dims = [1,3];
        mu_marg = mu_mat(:, plot_dims);
        P_marg = P_mat(plot_dims, plot_dims, :);
    
        [X1, X2] = meshgrid(linspace(min(Xm_cloud(:,plot_dims(1))), max(Xm_cloud(:,plot_dims(1))), 100), ...
                            linspace(min(Xm_cloud(:,plot_dims(2))), max(Xm_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
    
        Z_cell = cell(K,1); contours_cell = cell(K,1); 
        
        parfor k = 1:K
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
            contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
        end 
        
        hold on;
        for k = 1:K
            contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end
        plot(dist2km*Xprop_truth(1), dist2km*Xprop_truth(3), 'kx','MarkerSize', 20, 'LineWidth', 3)
        title('X-Z');
        xlabel('X (km.)');
        ylabel('Z (km.)');
        legend(legend_string);
        hold off;
    
        subplot(2,3,3)
        plot_dims = [2,3];
        mu_marg = mu_mat(:, plot_dims);
        P_marg = P_mat(plot_dims, plot_dims, :);
    
        [X1, X2] = meshgrid(linspace(min(Xm_cloud(:,plot_dims(1))), max(Xm_cloud(:,plot_dims(1))), 100), ...
                            linspace(min(Xm_cloud(:,plot_dims(2))), max(Xm_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
    
        Z_cell = cell(K,1); contours_cell = cell(K,1); 
        
        parfor k = 1:K
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
            contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
        end 
        
        hold on;
        for k = 1:K
            contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end
        plot(dist2km*Xprop_truth(2), dist2km*Xprop_truth(3), 'kx','MarkerSize', 20, 'LineWidth', 3)
        title('Y-Z');
        xlabel('Y (km.)');
        ylabel('Z (km.)');
        legend(legend_string);
        hold off;
    
        subplot(2,3,4)
        plot_dims = [4,5];
        mu_marg = mu_mat(:, plot_dims);
        P_marg = P_mat(plot_dims, plot_dims, :);
    
        [X1, X2] = meshgrid(linspace(min(Xm_cloud(:,plot_dims(1))), max(Xm_cloud(:,plot_dims(1))), 100), ...
                            linspace(min(Xm_cloud(:,plot_dims(2))), max(Xm_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
    
        Z_cell = cell(K,1); contours_cell = cell(K,1); 
        
        parfor k = 1:K
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
            contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
        end 
        
        hold on;
        for k = 1:K
            contour(vel2kms*X1, vel2kms*X2, vel2kms*Z_cell{k}, vel2kms*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end 
        plot(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(5), 'kx','MarkerSize', 20, 'LineWidth', 3)
        title('Xdot-Ydot');
        xlabel('Xdot (km/s)');
        ylabel('Ydot (km/s)');
        legend(legend_string);
        hold off;
    
        subplot(2,3,5)
        plot_dims = [4,6];
        mu_marg = mu_mat(:, plot_dims);
        P_marg = P_mat(plot_dims, plot_dims, :);
    
        [X1, X2] = meshgrid(linspace(min(Xm_cloud(:,plot_dims(1))), max(Xm_cloud(:,plot_dims(1))), 100), ...
                            linspace(min(Xm_cloud(:,plot_dims(2))), max(Xm_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
    
        Z_cell = cell(K,1); contours_cell = cell(K,1); 
        
        parfor k = 1:K
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
            contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
        end 
        
        hold on;
        for k = 1:K
            contour(vel2kms*X1, vel2kms*X2, vel2kms*Z_cell{k}, vel2kms*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end
        plot(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(6), 'kx','MarkerSize', 20, 'LineWidth', 3)
        title('Xdot-Zdot');
        xlabel('Xdot (km/s)');
        ylabel('Zdot (km/s)');
        legend(legend_string);
        hold off;
    
        subplot(2,3,6)
        plot_dims = [5,6];
        mu_marg = mu_mat(:, plot_dims);
        P_marg = P_mat(plot_dims, plot_dims, :);
    
        [X1, X2] = meshgrid(linspace(min(Xm_cloud(:,plot_dims(1))), max(Xm_cloud(:,plot_dims(1))), 100), ...
                            linspace(min(Xm_cloud(:,plot_dims(2))), max(Xm_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
    
        Z_cell = cell(K,1); contours_cell = cell(K,1); 
        
        parfor k = 1:K
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mu_marg(k,:)) * P_marg(:,:,k)^(-1) * (X_grid(i,:) - mu_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(P_marg(:,:,k)))); Z_cell{k} = Z;
            contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals
        end 
        
        hold on;
        for k = 1:K
            contour(vel2kms*X1, vel2kms*X2, vel2kms*Z_cell{k}, vel2kms*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end 
        plot(vel2kms*Xprop_truth(5), vel2kms*Xprop_truth(6), 'kx','MarkerSize', 20, 'LineWidth', 3)
        title('Ydot-Zdot');
        xlabel('Ydot (km/s)');
        ylabel('Zdot (km/s)');
        legend(legend_string);
        hold off;
    
        sgt = sprintf('Timestep: %3.4f Hours (Posterior)', tpr*time2hr);
        sgtitle(sgt);
    
        sg = sprintf('./Simulations/Timestep_%i_2A.png', tau);
        % sg = sprintf('./Simulations/Different Orbit Simulations/Timestep_%i_2A.png', tau);
        saveas(f, sg, 'png');
        close(f);
        %}

        for j = 1:Nt
            f = figure('visible','off','Position', get(0,'ScreenSize'));
            f.WindowState = 'maximized';
    
            % fprintf("Plotting Particles at Timestep: %d\n", tau);
        
            % legend_string = {"Truth 1", "Truth 2"};
        
            subplot(2,3,1)
            for k = 1:K
                scatter(dist2km*Xp_cloudp{j}(c_id(:,j) == k,1), dist2km*Xp_cloudp{j}(c_id(:,j) == k,2), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
                hold on;
            end
            plot(dist2km*Xprop_truth{j}(1), dist2km*Xprop_truth{j}(2), 'kx','MarkerSize', 15, 'LineWidth', 3)
            hold on;

            title('X-Y');
            xlabel('X (km.)');
            ylabel('Y (km.)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,2)
            for k = 1:K
                scatter(dist2km*Xp_cloudp{j}(c_id(:,j) == k,1), dist2km*Xp_cloudp{j}(c_id(:,j) == k,3), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
                hold on;
            end
            plot(dist2km*Xprop_truth{j}(1), dist2km*Xprop_truth{j}(3), 'kx','MarkerSize', 15, 'LineWidth', 3)
            hold on;

            title('X-Z');
            xlabel('X (km.)');
            ylabel('Z (km.)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,3)
            for k = 1:K
                scatter(dist2km*Xp_cloudp{j}(c_id(:,j) == k,2), dist2km*Xp_cloudp{j}(c_id(:,j) == k,3), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
                hold on;
            end
            plot(dist2km*Xprop_truth{j}(2), dist2km*Xprop_truth{j}(3), 'kx','MarkerSize', 15, 'LineWidth', 3)
            hold on;

            title('Y-Z');
            xlabel('Y (km.)');
            ylabel('Z (km.)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,4)
            for k = 1:K
                scatter(vel2kms*Xp_cloudp{j}(c_id(:,j) == k,4), vel2kms*Xp_cloudp{j}(c_id(:,j) == k,5), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
                hold on;
            end
            plot(vel2kms*Xprop_truth{j}(4), vel2kms*Xprop_truth{j}(5), 'kx','MarkerSize', 15, 'LineWidth', 3)
            hold on;

            title('Xdot-Ydot');
            xlabel('Xdot (km/s)');
            ylabel('Ydot (km/s)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,5)
            for k = 1:K
                scatter(vel2kms*Xp_cloudp{j}(c_id(:,j) == k,4), vel2kms*Xp_cloudp{j}(c_id(:,j) == k,6), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
                hold on;
            end
            plot(vel2kms*Xprop_truth{j}(4), vel2kms*Xprop_truth{j}(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
            hold on;

            title('Xdot-Zdot');
            xlabel('Xdot (km/s)');
            ylabel('Zdot (km/s)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,6)
            for k = 1:K
                scatter(vel2kms*Xp_cloudp{j}(c_id(:,j) == k,5), vel2kms*Xp_cloudp{j}(c_id(:,j) == k,6), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
                hold on;
            end
            plot(vel2kms*Xprop_truth{j}(5), vel2kms*Xprop_truth{j}(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
            hold on;

            title('Ydot-Zdot');
            xlabel('Ydot (km/s)');
            ylabel('Zdot (km/s)');
            legend(legend_string);
            hold off;
    
            sgt = sprintf('Timestep: %3.4f Hours (Posterior) - Target %i', tpr*time2hr, j);
            sgtitle(sgt);
        
            sg = sprintf('./Multi_Sims/Timestep_%i_2B_T%i.png', tau, j);
            saveas(f, sg, 'png');
            close(f);
        end
    
        f = figure('visible','off','Position', get(0,'ScreenSize'));
        f.WindowState = 'maximized';

        % fprintf("Plotting Particles at Timestep: %d\n", tau);
    
        subplot(2,3,1)
        for j = 1:Nt
            for k = 1:K
                scatter(dist2km*Xp_cloudp{j}(c_id(:,j) == k,1), dist2km*Xp_cloudp{j}(c_id(:,j) == k,2), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
                hold on;
            end
        end
        for j = 1:Nt
            plot(dist2km*Xprop_truth{j}(1), dist2km*Xprop_truth{j}(2), 'x','MarkerSize', 15, 'LineWidth', 3)
            hold on;
        end
        title('X-Y');
        xlabel('X (km.)');
        ylabel('Y (km.)');
        legend(legend_string_x);
        hold off;
        
        subplot(2,3,2)
        for j = 1:Nt
            for k = 1:K
                scatter(dist2km*Xp_cloudp{j}(c_id(:,j) == k,1), dist2km*Xp_cloudp{j}(c_id(:,j) == k,3), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
                hold on;
            end
        end
        for j = 1:Nt
            plot(dist2km*Xprop_truth{j}(1), dist2km*Xprop_truth{j}(3), 'x','MarkerSize', 15, 'LineWidth', 3)
            hold on;
        end
        title('X-Z');
        xlabel('X (km.)');
        ylabel('Z (km.)');
        legend(legend_string_x);
        hold off;
        
        subplot(2,3,3)
        for j = 1:Nt
            for k = 1:K
                scatter(dist2km*Xp_cloudp{j}(c_id(:,j) == k,2), dist2km*Xp_cloudp{j}(c_id(:,j) == k,3), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
                hold on;
            end
        end
        for j = 1:Nt
            plot(dist2km*Xprop_truth{j}(2), dist2km*Xprop_truth{j}(3), 'x','MarkerSize', 15, 'LineWidth', 3)
            hold on;
        end
        title('Y-Z');
        xlabel('Y (km.)');
        ylabel('Z (km.)');
        legend(legend_string_x);
        hold off;
        
        subplot(2,3,4)
        for j = 1:Nt
            for k = 1:K
                scatter(vel2kms*Xp_cloudp{j}(c_id(:,j) == k,4), vel2kms*Xp_cloudp{j}(c_id(:,j) == k,5), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
                hold on;
            end
        end
        for j = 1:Nt
            plot(vel2kms*Xprop_truth{j}(4), vel2kms*Xprop_truth{j}(5), 'x','MarkerSize', 15, 'LineWidth', 3)
            hold on;
        end
        title('Xdot-Ydot');
        xlabel('Xdot (km/s)');
        ylabel('Ydot (km/s)');
        legend(legend_string_x);
        hold off;
        
        subplot(2,3,5)
        for j = 1:Nt
            for k = 1:K
                scatter(vel2kms*Xp_cloudp{j}(c_id(:,j) == k,4), vel2kms*Xp_cloudp{j}(c_id(:,j) == k,6), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
                hold on;
            end
        end
        for j = 1:Nt
            plot(vel2kms*Xprop_truth{j}(4), vel2kms*Xprop_truth{j}(6), 'x','MarkerSize', 15, 'LineWidth', 3)
            hold on;
        end
        title('Xdot-Zdot');
        xlabel('Xdot (km/s)');
        ylabel('Zdot (km/s)');
        legend(legend_string_x);
        hold off;
        
        subplot(2,3,6)
        for j = 1:Nt
            for k = 1:K
                scatter(vel2kms*Xp_cloudp{j}(c_id(:,j) == k,5), vel2kms*Xp_cloudp{j}(c_id(:,j) == k,6), 'filled', ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(j));
                hold on;
            end
        end
        for j = 1:Nt
            plot(vel2kms*Xprop_truth{j}(5), vel2kms*Xprop_truth{j}(6), 'x','MarkerSize', 15, 'LineWidth', 3)
            hold on;
        end
        title('Ydot-Zdot');
        xlabel('Ydot (km/s)');
        ylabel('Zdot (km/s)');
        legend(legend_string_x);
        hold off;

        sgt = sprintf('Timestep: %3.4f Hours (Posterior)', tpr*time2hr);
        sgtitle(sgt);
    
        sg = sprintf('./Multi_Sims/Timestep_%i_2B.png', tau);
        saveas(f, sg, 'png');
        close(f);

        for j = 1:Nt
            f = figure('visible','off','Position', get(0,'ScreenSize'));
            f.WindowState = 'maximized';
    
            legend_string_z = "Truth";
            hold on;
            for k = 1:K
                pts = Xp_cloudp{j}(c_id(:,j) == k, :);
                if(isempty(pts))
                    continue;
                end

                Zmcloud = zeros(length(pts(:,1)), length(h(pts(1,:))));
                for i = 1:length(Zmcloud(:,1))
                    Zmcloud(i,:) = h(pts(i,:))';
                end  
                scatter(180/pi*Zmcloud(:,1), 180/pi*Zmcloud(:,2), 'filled', ...
                'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
            end
            Ztruth = h(Xprop_truth{j})';
            plot(180/pi*Ztruth(1), 180/pi*Ztruth(2), 'kx', ... 
                'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string_z);
            plot(180/pi*zt(j,1), 180/pi*zt(j,2), 'o', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', "Assoc Meas");
            plot(180/pi*zt_truth(j,1), 180/pi*zt_truth(j,2), 'o', ... 
                    'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', "Actual Meas");

            sgt = sprintf('Timestep: %3.4f Hours (Posterior AZ-EL) - Target %i', tpr*time2hr, j);
            sgtitle(sgt);

            xlabel('Azimuth Angle (deg)')
            ylabel('Elevation Angle (deg)')
            legend(legend_string)
    
            sg = sprintf('./Multi_Sims/Timestep_%i_2C_T%i.png', tau, j);
            saveas(f, sg, 'png');
            close(f);
        end
        
        f = figure('visible','off','Position', get(0,'ScreenSize'));
        f.WindowState = 'maximized';

        legend_string_z = {};
        parfor j = 1:Nt
            legend_string_z{j} = sprintf('Truth %i', j);
        end
        hold on;

        for j = 1:Nt
            Zmcloud = zeros(length(Xm_cloud{j}(:,1)), length(h(Xprop_truth{j})));
            for i = 1:length(Zmcloud(:,1))
                Zmcloud(i,:) = h(Xp_cloudp{j}(i,:))';
            end  
            scatter(180/pi*Zmcloud(:,1), 180/pi*Zmcloud(:,2), 'filled', ...
            'MarkerFaceColor', colors(j), 'HandleVisibility', 'off');

            Ztruth = h(Xprop_truth{j})';
            plot(180/pi*Ztruth(1), 180/pi*Ztruth(2), 'kx', ... 
                'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string_z{j});
        end

        sgt = sprintf('Timestep: %3.4f Hours (Posterior AZ-EL)', tpr*time2hr);
        sgtitle(sgt);

        xlabel('Azimuth Angle (deg)')
        ylabel('Elevation Angle (deg)')

        sg = sprintf('./Multi_Sims/Timestep_%i_2C.png', tau);
        saveas(f, sg, 'png');
        close(f);
        %}
    end

    % if(1)
    %{
    if(idx_meas ~= 0)
        K = Kn;
    else
        K = 1;
    end
    %}

    %{
    if(abs(tpr - max(cellfun(@(u) u(2), CTimes))) < 1e-10)
        Lp = 1500;
    % elseif(abs(tpr - CTimes{1}(4)) < 1e-10)
    %     Lp = 1500;
    % elseif(abs(tpr - cVal) < 1e-10)
    %     Lp = 2500;
    %     save("Xm_cloud.mat", "Xp_cloudp"); save("t_int.mat", "tpr"); save("noised_obs.mat", "noised_obs"); save("Xtruth.mat", "Xprop_truth");
    end
    %}
end

for j = 1:Nt
    Xp_cloudn = zeros(Lp, length(Xprop_truth{j}));
    for i = 1:Lp
        [Xp_cloudn(i,:), ~] = drawFrom(wp(:,j), mu_p(:,j), P_p(:,j)); 
    end
    ent1(j,end,:) = getDiagCov(Xp_cloudn);
end
ent2(end-1:end,:) = [];

%%
% What we have: Observations zt; A priori and posterior means and
% covariances mu_c, P_c, mu_p, P_p; A priori and posterior GMM weights
% wm, wp
% What we need: Likelihood function p_i(z) = p_v(z-h(x), Pvv)
% Goal: Design a table of measurement likelihoods for each observation

%{
ml_table = cell(1,Nt); % One cell for one observation
likes = zeros(K,Nt); zMeans = cell(K,Nt); zCovs = cell(K,Nt);
av_likes = zeros(Nt, Nt);

% zw = zt; zw(:,2) = getNoisyMeas(Xprop_truth{2}, R_vv, h);

for j = 1:Nt
    for i = 1:Nt
        for k = 1:K
            cPts = cPoints{k,i};
            zPts = zeros(length(cPts(:,1)), length(zt(:,j)));
            for l = 1:length(cPts(:,1))
                zPts(l,:) = h(cPts(l,:));
            end
            zPredMean = mean(zPts,1);
            zPredCov = cov(zPts) + R_vv;
            likes(k,i) = mvnpdf(zt(:,j)' - zPredMean, zeros(size(zt(:,j)')), zPredCov);
            zMeans{k,i} = zPredMean; zCovs{k,i} = zPredCov;
        end
        av_likes(j,i) = dot(wm(:,i),likes(:,i)); % Weighted likelihoods
    end
    ml_table{j} = likes;
end

save ./Multi_Sims/ml_table.mat ml_table; save ./Multi_Sims/av_likes.mat av_likes;
save ./Multi_Sims/wm.mat wm;
%}

figure(7)

subplot(2,3,1)
for j = 1:Nt
    plot(0:l_filt, dist2km*sqrt(ent1(j,:,1)))
    hold on;
end
xlabel('Filter Step #')
ylabel('Log \\sigma_X (km.)')
title('X Standard Deviation')
legend('Target 1', 'Target 2')

subplot(2,3,2)
for j = 1:Nt
    plot(0:l_filt, dist2km*sqrt(ent1(j,:,2)))
    hold on;
end
xlabel('Filter Step #')
ylabel('Log \\sigma_Y (km.)')
title('Y Standard Deviation')
legend('Target 1', 'Target 2')

subplot(2,3,3)
for j = 1:Nt
    plot(0:l_filt, dist2km*sqrt(ent1(j,:,3)))
    hold on;
end
xlabel('Filter Step #')
ylabel('Log \\sigma_Z (km.)')
title('Z Standard Deviation')
legend('Target 1', 'Target 2')

subplot(2,3,4)
for j = 1:Nt
    plot(0:l_filt, vel2kms*sqrt(ent1(j,:,4)))
    hold on;
end
xlabel('Filter Step #')
ylabel('Log \\sigma_Xdot (km/s)')
title('Xdot Standard Deviation')
legend('Target 1', 'Target 2')

subplot(2,3,5)
for j = 1:Nt
    plot(0:l_filt, vel2kms*sqrt(ent1(j,:,5)))
    hold on;
end
xlabel('Filter Step #')
ylabel('\\sigma_Ydot (km/s)')
title('Ydot Standard Deviation')
legend('Target 1', 'Target 2')

subplot(2,3,6)
for j = 1:Nt
    plot(0:l_filt, vel2kms*sqrt(ent1(j,:,6)))
    hold on;
end
xlabel('Filter Step #')
ylabel('\\sigma_Zdot (km/s)')
title('Zdot Standard Deviation')
legend('Target 1', 'Target 2')

savefig(gcf, './Multi_Sims/StDevEvols.fig');

% Plot the results
figure(8)
for j = 1:Nt
    plot(0:l_filt-2, ent2(:,j))
    hold on;
end
xlabel('Filter Step #')
ylabel('Entropy Metric')
title('Entropy')
legend('Target 1', 'Target 2')
savefig(gcf,'./Multi_Sims/Entropy.fig');

% Plot the results
figure(9)
% legend_string = {"Truth 1", "Truth 2"}; 

% for j = 1:Nt
%     for i = 1:Lp
%         [Xp_cloudp{j}(i,:), ~] = drawFrom(wp(:,j), mu_p(:,j), P_p(:,j)); 
%     end
% end
    
subplot(2,3,1)
for j = 1:Nt
    for k = 1:K
        scatter(dist2km*cPoints{k,j}(:,1), dist2km*cPoints{k,j}(:,2), 'filled', ...
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
        hold on;
    end
end
for j = 1:Nt
    plot(dist2km*Xprop_truth{j}(1), dist2km*Xprop_truth{j}(2), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('X-Y');
xlabel('X (km.)');
ylabel('Y (km.)');
legend(legend_string);
hold off;

subplot(2,3,2)
for j = 1:Nt
    for k = 1:K
        scatter(dist2km*cPoints{k,j}(:,1), dist2km*cPoints{k,j}(:,3), 'filled', ...
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
        hold on;
    end
end
for j = 1:Nt
    plot(dist2km*Xprop_truth{j}(1), dist2km*Xprop_truth{j}(3), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('X-Z');
xlabel('X (km.)');
ylabel('Z (km.)');
legend(legend_string);
hold off;

subplot(2,3,3)
for j = 1:Nt
    for k = 1:K
        scatter(dist2km*cPoints{k,j}(:,2), dist2km*cPoints{k,j}(:,3), 'filled', ...
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
        hold on;
    end
end
for j = 1:Nt
    plot(dist2km*Xprop_truth{j}(2), dist2km*Xprop_truth{j}(3), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('Y-Z');
xlabel('Y (km.)');
ylabel('Z (km.)');
legend(legend_string);
hold off;

subplot(2,3,4)
for j = 1:Nt
    for k = 1:K
        scatter(vel2kms*cPoints{k,j}(:,4), vel2kms*cPoints{k,j}(:,5), 'filled', ...
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
        hold on;
    end
end
for j = 1:Nt
    plot(vel2kms*Xprop_truth{j}(4), vel2kms*Xprop_truth{j}(5), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('Xdot-Ydot');
xlabel('Xdot (km/s)');
ylabel('Ydot (km/s)');
legend(legend_string);
hold off;

subplot(2,3,5)
for j = 1:Nt
    for k = 1:K
        scatter(vel2kms*cPoints{k,j}(:,4), vel2kms*cPoints{k,j}(:,6), 'filled', ...
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
        hold on;
    end
end
for j = 1:Nt
    plot(vel2kms*Xprop_truth{j}(4), vel2kms*Xprop_truth{j}(6), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('Xdot-Zdot');
xlabel('Xdot (km/s)');
ylabel('Zdot (km/s)');
legend(legend_string);
hold off;

subplot(2,3,6)
for j = 1:Nt
    for k = 1:K
        scatter(vel2kms*cPoints{k,j}(:,5), vel2kms*cPoints{k,j}(:,6), 'filled', ...
            'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
        hold on;
    end
end
for j = 1:Nt
    plot(vel2kms*Xprop_truth{j}(5), vel2kms*Xprop_truth{j}(6), 'x','MarkerSize', 15, 'LineWidth', 3)
    hold on;
end
title('Ydot-Zdot');
xlabel('Ydot (km/s)');
ylabel('Zdot (km/s)');
legend(legend_string);
hold off;

sgt = sprintf('Timestep: %3.4f Hours (Prior)', tpr*time2hr);
sgtitle(sgt);

figure(10)

legend_string = "Truth";
hold on;

for j = 1:Nt
    for k = 1:K
        pts = Xm_cloud{j}(c_id(:,j) == k, :);
        Zmcloud = zeros(length(pts(:,1)), length(zt(:,j)));
        for i = 1:length(Zmcloud(:,1))
            Zmcloud(i,:) = h(pts(i,:))';
        end  
        scatter(180/pi*Zmcloud(:,1), 180/pi*Zmcloud(:,2), 'filled', ...
        'MarkerFaceColor', colors(j), 'HandleVisibility', 'off');
    end
    Ztruth = h(Xprop_truth{j})';
    plot(180/pi*Ztruth(1), 180/pi*Ztruth(2), 'x', ... 
        'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string_z{j});
end

title('Post-Shutoff A Priori Measurement Signatures')
xlabel('Azimuth Angle (deg)')
ylabel('Elevation Angle (deg)')

savefig(gcf, './Multi_Sims/Signatures.fig');

figure(11)

hold on;
for j = 1:Nt
    [idx_lastObs, ~] = find(abs(noised_obs{j}(:,1) - t_end) < 1e-11);
    [idx_firstObs, ~] = find(abs(noised_obs{j}(:,1) - Full_ts{j}(idx_start,1)) < 1e-11);
    streakObs = noised_obs{j}(idx_firstObs:idx_lastObs,3:4);

    plot(180/pi*streakObs(:,1), 180/pi*streakObs(:,2), 'o-', 'Color', colors(j))
    hold on;
    arrow_scale = 0.2;
    dAZ = diff(180/pi*streakObs(:,1)); dEL = diff(180/pi*streakObs(:,2));
    quiver(180/pi*streakObs(1:end-1,1), 180/pi*streakObs(1:end-1,2), ...
        arrow_scale*dAZ, arrow_scale*dEL, 0, 'Color', colors(j), 'MaxHeadSize', 1.5, 'LineWidth', 1.2);
end
xlabel('Azimuth Angle (deg)')
ylabel('Elevation Angle (deg)')
title('Measurement Streaks')
legend('Target 1', '', 'Target 2')

savefig(gcf, './Multi_Sims/MeasurementStreaks.fig');
save("stdevs.mat", "ent1");

% Finish timer
toc

%% Functions

function Hx = linHx(mu)
    Hk_AZ = [-mu(2)/(mu(1)^2 + mu(2)^2), mu(1)/(mu(1)^2 + mu(2)^2), 0, 0, 0, 0]; % Azimuth angle linearization
    Hk_EL = [-(mu(1)*mu(3))/((mu(1)^2 + mu(2)^2 + mu(3)^2)*sqrt(mu(1)^2+mu(2)^2)), ...
             -(mu(2)*mu(3))/((mu(1)^2 + mu(2)^2 + mu(3)^2)*sqrt(mu(1)^2+mu(2)^2)), ...
             sqrt(mu(1)^2 + mu(2)^2)/(mu(1)^2 + mu(2)^2 + mu(3)^2), 0, 0, 0];

    Hx = [Hk_AZ; Hk_EL];
end

function w = weightUpdate(wc, cluster_points, idx, zk, R, h)
    zk = reshape(zk, [length(zk) 1]);
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

function [dX_coeffs] = polyDeriv(X_coeffs)
    
    dX_coeffs = zeros(1, length(X_coeffs)-1);
    for j = length(X_coeffs):-1:2
        dX_coeffs(length(X_coeffs)+1-j) = X_coeffs(length(X_coeffs)+1-j)*(j-1);
    end
end

% Adds process noise to the un-noised state vector
function [Xm] = procNoise(X)
    Q = 0.000^2*diag(abs(X)); % Process noise is 1% of each state vector component
    Xm = mvnrnd(X,Q);
end

function [Xfit] = stateEstCloud(pf, theta_f, R_f, obTr, tdiff)
    noised_obs = obTr;

    R_t = zeros(3*length(noised_obs(:,1)),1); % We shall diagonalize this later
    mu_t = zeros(3*length(noised_obs(:,1)),1);

    % load("partial_ts.mat"); % Noiseless observation data

    for i = 1:length(obTr(:,1))
        mu_t(3*(i-1)+1:3*(i-1)+3, 1) = [obTr(i,2); obTr(i,3); obTr(i,4)];
        R_t(3*(i-1)+1:3*(i-1)+3, 1) = [R_f*obTr(i,2); (theta_f*4.84814e-6); (theta_f*4.84814e-6)].^2;
    end

    R_t = diag(R_t);
    data_vec = mvnrnd(mu_t, R_t)';

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

function [X_bt] = backConvertSynodic(X_ot, t_stamp)

    rot_topo = X_ot(1:3); % First three components of the state vector
    vot_topo = X_ot(4:6); % Last three components of the state vector

    % First step: Obtain X_{eo}^{ECI} 
    obs_lat = 30.618963;
    obs_lon = -96.339214;
    elevation = 103.8;
    mu = 1.2150582e-2;

    UTC_vec_orig = [2024	5	3	2	41	15.1261889999956]; % Initial UTC vector at t_stamp = 0
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
function [X_ot] = convertToTopo(X_bt, t_stamp)
    % Insert code for obtaining vector between center of Earth and observer

    obs_lat = 30.618963;
    obs_lon = -96.339214;
    elevation = 103.8;
    
    mu = 1.2150582e-2;
    rbe = [-mu, 0, 0]'; % Position vector relating center of earth to barycenter

    UTC_vec_orig = [2024	5	3	2	41	15.1261889999956];
    t_add_dim = t_stamp * (4.342);
    UTC_vec = datevec(datetime(UTC_vec_orig) + t_add_dim);

    delt_add_dim = -1/86400;
    delt_updatedUTCtime = datetime(UTC_vec) + delt_add_dim;
    delt_updatedUTCvec = datevec(delt_updatedUTCtime);

    reo_dim = lla2eci([obs_lat, obs_lon, elevation], UTC_vec);
    delt_reodim = lla2eci([obs_lat, obs_lon, elevation], delt_updatedUTCvec);
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

function [x_p, pos] = drawFrom3(w, mu, P)
    % Use histcounts for efficient sampling
    pos = histcounts(rand, [0; cumsum(w(:))]);
    pos = find(pos, 1);

    if (isempty(pos) || pos > length(mu))
        error('Sampling error: invalid position');
    end
    
    x_p = mvnrnd(mu{pos}, P{pos});
end

function zk = getNoisyMeas(Xtruth, R, h)
    mzkm = h(Xtruth);
    zk = mvnrnd(mzkm, R);
    zk = zk'; % Make into column vector
end

function Xm_cloud = propagate(Xcloud, t_int, interval)
    % Xcloud = zeros(L,length(mu{1}));
    % for i = 1:L
    %     [Xcloud(i,:), ~] = drawFrom(w, mu, P);
    % end
    % 
    Xm_cloud = zeros(size(Xcloud));
    parfor i = 1:length(Xcloud(:,1))
        % First, convert from X_{ot} in the topocentric frame to X_{bt} in the
        % synodic frame.
        Xbt = backConvertSynodic(Xcloud(i,:)', t_int);

        % Next, propagate each X_{bt} in your particle cloud by a single time 
        % step and convert back to the topographic frame.
        % Call ode45()

        % opts = odeset('Events', @termSat);
        % [~,X] = ode45(@cr3bp_dyn, [0 interval], Xbt, opts); % Assumes termination event (i.e. target enters LEO)
        
        opts = odeset('Events', @termSat, 'RelTol', 1e-6, 'AbsTol', 1e-8); 
        [~, X] = ode15s(@cr3bp_dyn, [0 interval], Xbt, opts); 

        Xm_bt = X(end,:)';
        Xm_cloud(i,:) = convertToTopo(Xm_bt, t_int + interval);
        % Xm_cloud(i,:) = procNoise(Xm_cloud(i,:)); % Adds process noise
    end    
end

% Inputs - cPts (KxNt size cell of particles), zt_perm (randomized
% permutation of observations)
% Output - zc (Properly associated
function zc = dataAssoc(cPts, wc, zt_perm, h, R_vv)
    
    gamma = 1.2;
    zc = zt_perm;
    [K, Nt] = size(cPts);
    
    ml_table = cell(1,Nt); % One cell for one observation
    av_likes = zeros(Nt, Nt);
    
    for j = 1:Nt % Observation #
        likes = zeros(K,Nt); zMeans = cell(K,Nt); zCovs = cell(K,Nt);
        for i = 1:Nt % Estimate #
            for k = 1:K % Cluster #
                clusterPts = cPts{k,i};
                zPts = zeros(length(clusterPts(:,1)), length(zt_perm(j,:)));
                for l = 1:length(clusterPts(:,1))
                    zPts(l,:) = h(clusterPts(l,:));
                end
                zPredMean = mean(zPts,1);
                zPredCov = cov(zPts) + R_vv;
                likes(k,i) = mvnpdf(zt_perm(j,:), zPredMean, zPredCov);
                zMeans{k,i} = zPredMean; zCovs{k,i} = zPredCov;
            end
            av_likes(j,i) = -dot(wc(:,i),likes(:,i)); % Weighted likelihoods
        end
        av_likes(j,:) = (av_likes(j,:) - min(av_likes(j,:))); % Sharpens cost matrix
        ml_table{j} = likes;
    end
    fprintf("Association Matrix: \n");
    disp(av_likes);
    [assoc, ~, ~] = assignmunkres(av_likes, 1e10);
    % disp(assoc);
    
    for j = 1:length(assoc(:,1))
        zc(assoc(j,1),:) = zt_perm(assoc(j,2),:);
    end
    zc
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
    % mu_p = mu_m' + K_k*(zk - h(mu_m));
    mu_p = mu_m' + K_k*(zk - mzk_m);
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

function [mu_c, P_c] = ukfProp(t_int, interval, mu_p, P_p)
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
        sigs(i,:) = (mu_m' + sqrt(n + lambda)*S(:,i-1))'; 
        sigs(i+n,:) = (mu_m' - sqrt(n + lambda)*S(:,i-1))';

        wm(i) = 0.5/(n + lambda); wm(i+n) = wm(i);
        wc(i) = 0.5/(n + lambda); wc(i+n) = wc(i);
    end
    
    prop_sigs = zeros(size(sigs));
    % Propagation of sigma points
    for i = 1:length(sigs(i,:))
        prop_sigs(i,:) = propagate(sigs(i,:), t_int, interval);
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
    K_k = Pxz*Pzz^(-1);

    % Update mean and covariances
    mu_p = mu_m' + K_k*(zk - mzk);
    % P_p = P_m - Pxz*K_k' - K_k*Pxz' + K_k*Pzz*K_k';
    P_p = P_m - K_k*Pzz*K_k';
    P_p = (P_p + P_p')/2;

    [V, D] = eig(P_p);
    D = max(D,0);
    P_p = V*D*V';
end

function ent = getKnEntropy(Kp, Xcloud)
    rc = Xcloud(:,1:3);
    vc = Xcloud(:,4:6);
    
    mean_rc = mean(rc, 1);
    mean_vc = mean(vc, 1);
    
    std_rc = std(rc,0,1);
    std_vc = std(vc,0,1);
    
    norm_rc = (rc - mean_rc)./norm(std_rc); % Normalizing the position 
    norm_vc = (vc - mean_vc)./norm(std_vc); % Normalizing the velocity
    
    Xm_norm = [norm_rc, norm_vc];

    [idx, ~] = kmeans(Xm_norm, Kp); % Cluster just on position and velocity; Normalize the whole thing
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
        
        w(k) = size(cluster_points, 1) / size(Xm_norm, 1); % Vector of weights
    end

    wsum = 0;
    for k = 1:Kp
        wsum = wsum + w(k)*det(P{k});
    end
    ent = log(wsum);
end

function ent = getDiagCov(Xcloud)
    P = cov(Xcloud);
    ent = diag(P);
end