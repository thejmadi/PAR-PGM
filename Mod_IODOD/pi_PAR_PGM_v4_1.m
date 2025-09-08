% Start the clock
tic

% Load noiseless observation data and other important .mat files
load("partial_ts.mat"); % Noiseless observation data
load("full_ts.mat"); % Position truth (topocentric frame)
load("full_vts.mat"); % Velocity truth (topocentric frame)

% Add observation noise to the observation data as follows:
% Range - 5% of the current (i.e. noiseless) range
% Azimuth - 1.5 arc sec (credit: Dr. Utkarsh Ranjan Mishra)
% Elevation - 1.5 arc sec (credit: Dr. Utkarsh Ranjan Mishra)
% Note: All above quantities are drawn in a zero-mean Gaussian fashion.

noised_obs = partial_ts;
R_t = zeros(3*length(noised_obs(:,1)),1); % We shall diagonalize this later
mu_t = zeros(3*length(noised_obs(:,1)),1);

theta_f = 1.5; % Arc-seconds of error covariance
R_f = 0.05; % Range percentage error covariance

dist2km = 384400; % Kilometers per non-dimensionalized distance
time2hr = 4.342*24; % Hours per non-dimensionalized time
vel2kms = dist2km/(time2hr*60*60); % Kms per non-dimensionalized velocity

for i = 1:length(partial_ts(:,1))
    mu_t(3*(i-1)+1:3*(i-1)+3, 1) = [partial_ts(i,2); partial_ts(i,3); partial_ts(i,4)];
    R_t(3*(i-1)+1:3*(i-1)+3, 1) = [(0.05*partial_ts(i,2))^2; (theta_f*4.84814e-6)^2; (theta_f*4.84814e-6)^2];
end

R_t = diag(R_t);
data_vec = mvnrnd(mu_t, R_t)';

for i = 1:length(noised_obs(:,1))
    noised_obs(i,2:4) = data_vec(3*(i-1)+1:3*(i-1)+3,1);
end

% Extract important time points from the noised_obs variable
i = 2;
interval = noised_obs(2,1) - partial_ts(1,1);
cTimes = []; % Array of important time points

while (i <= length(noised_obs(:,1)))
    if (noised_obs(i,1) - noised_obs(i-1,1) > (interval+1e-11))
        cTimes = [cTimes, noised_obs(i-1,1), noised_obs(i,1)];
    end
    i = i + 1;
end

larger_diff = noised_obs(end,1) - noised_obs(end-1,1);
for j = 2:length(noised_obs(:,1))
    if (noised_obs(j,1) - noised_obs(j-1,1) > larger_diff+1e-11)
        cVal = noised_obs(j,1); break;
    else
        cVal = noised_obs(end,1);
    end
end

% Extract the first continuous observation track
hdo = []; % Matrix for a half day observation
hdo(1,:) = noised_obs(1,:);
i = 1;
while(noised_obs(i+1,1) - noised_obs(i,1) < full_ts(2,1) + 1e-15) % Add small epsilon due to roundoff error
    hdo(i,:) = noised_obs(i+1,:);
    i = i + 1;
end

% Convert observation data into [X, Y, Z] data in the topographic frame.

hdR = zeros(length(hdo(:,1)),4); % Convert quantities of hdo to [X, Y, Z]
hdR(:,1) = hdo(:,1); % Timestamp stays the same
hdR(:,2) = hdo(:,2) .* cos(hdo(:,4)) .* cos(hdo(:,3)); % Conversion to X
hdR(:,3) = hdo(:,2) .* cos(hdo(:,4)) .* sin(hdo(:,3)); % Conversion to Y
hdR(:,4) = hdo(:,2) .* sin(hdo(:,4)); % Conversion to Z

pf = 0.25; % A factor between 0 to 1 describing the length of the day to interpolate [x, y]
nfit = 4; % Order of polynomial fitting (typically around 3-4)
in_len = round(pf * length(hdR(:,1))); % Length of interpolation interval

% Modify interpolation interval length such that you are pieceing through
% enough points.
if (in_len < nfit + 1)
    in_len = nfit + 1;
    pf = in_len/length(hdR(:,1)); % Modify pf such that it meets minimum condition
end

hdR_p = hdR(1:in_len,:); % Matrix for a partial half-day observation

% Fit polynomials for X, Y, and Z (Cubic for X, Quadratic for X and Y)
coeffs_X = polyfit(hdR_p(:,1), hdR_p(:,2), nfit);
coeffs_Y = polyfit(hdR_p(:,1), hdR_p(:,3), nfit);
coeffs_Z = polyfit(hdR_p(:,1), hdR_p(:,4), nfit);

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

partial_vts = [];
partial_rts = [];
j = 1;
i = 1;
while (j <= length(hdR_p(:,1)))
    if(hdR(j,1) == full_vts(i,1)) % Matching time index
        partial_vts(j,:) = full_vts(i,:);
        partial_rts(j,:) = full_ts(i,:);
        j = j + 1;
    end
    i = i + 1;
end

Xot_fitted = [X_fit(end,1); Y_fit(end,1); Z_fit(end,1); Xdot_fit(end,1); Ydot_fit(end,1); Zdot_fit(end,1)];
Xot_truth = [partial_rts(end,2:4), partial_vts(end,2:4)]';

t_truth = partial_rts(end,1);
[idx_prop, c_prop] = find(full_ts == t_truth);
Xprop_truth = [full_ts(idx_prop+1,2:4), full_vts(idx_prop+1,2:4)]';

L = 500;
Lp = 1*L;
X0cloud = zeros(L,6);

% delete(gcp('nocreate'))
% parpool(4, 'IdleTimeout', Inf);

parfor i = 1:length(X0cloud(:,1))
    X0cloud(i,:) = stateEstCloud(pf, partial_ts, (partial_ts(2,1) - partial_ts(1,1)) + 1e-15);
end

figure(1)
set(gcf, 'units','normalized','outerposition',[0 0 1 1])
subplot(2,3,1)
plot(dist2km*X0cloud(:,1), dist2km*X0cloud(:,2), '.')
hold on;
plot(dist2km*Xot_truth(1), dist2km*Xot_truth(2), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('X-Y');
xlabel('X (km.)');
ylabel('Y (km.)');
legend('Estimate','Truth');
hold off;

subplot(2,3,2)
plot(dist2km*X0cloud(:,1), dist2km*X0cloud(:,3), '.')
hold on;
plot(dist2km*Xot_truth(1), dist2km*Xot_truth(3), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('X-Z');
xlabel('X (km.)');
ylabel('Z (km.)');
legend('Estimate','Truth');
hold off;

subplot(2,3,3)
plot(dist2km*X0cloud(:,2), dist2km*X0cloud(:,3), '.')
hold on;
plot(dist2km*Xot_truth(2), dist2km*Xot_truth(3), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('Y-Z');
xlabel('Y (km.)');
ylabel('Z (km.)');
legend('Estimate','Truth');
hold off;

subplot(2,3,4)
plot(vel2kms*X0cloud(:,4), vel2kms*X0cloud(:,5), '.')
hold on;
plot(vel2kms*Xot_truth(4), vel2kms*Xot_truth(5), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('Xdot-Ydot');
xlabel('Xdot (km/s)');
ylabel('Ydot (km/s)');
legend('Estimate','Truth');
hold off;

subplot(2,3,5)
plot(vel2kms*X0cloud(:,4), vel2kms*X0cloud(:,6), '.')
hold on;
plot(vel2kms*Xot_truth(4), vel2kms*Xot_truth(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('Xdot-Zdot');
xlabel('Xdot (km/s)');
ylabel('Zdot (km/s)');
legend('Estimate','Truth');
hold off;

subplot(2,3,6)
plot(vel2kms*X0cloud(:,5), vel2kms*X0cloud(:,6), '.')
hold on;
plot(vel2kms*Xot_truth(5), vel2kms*Xot_truth(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('Ydot-Zdot');
xlabel('Ydot (km/s)');
ylabel('Zdot (km/s)');
legend('Estimate','Truth');
hold off;

sg = sprintf('Timestep: %3.4f Hours', t_truth*time2hr);
sgtitle(sg);
savefig(gcf, 'iodCloud.fig');
saveas(gcf, './Simulations/iodCloud.png', 'png');
% saveas(gcf, './Simulations/Different Orbit Simulations/iodCloud.png', 'png');

t_int = hdR_p(end,1); % Time at which we are obtaining a state cloud
tspan = 0:interval:interval; % Integrate over just a single time step
Xm_cloud = X0cloud;

parfor i = 1:length(X0cloud(:,1))
    % First, convert from X_{ot} in the topocentric frame to X_{bt} in the
    % synodic frame.
    Xbt = backConvertSynodic(X0cloud(i,:)', t_int);

    % Next, propagate each X_{bt} in your particle cloud by a single time 
    % step and convert back to the topographic frame.
     % Call ode45()
    opts = odeset('Events', @termSat);
    [t,X] = ode45(@cr3bp_dyn, [0 interval], Xbt, opts); % Assumes termination event (i.e. target enters LEO)
    Xm_bt = X(end,:)';
    Xm_cloud(i,:) = convertToTopo(Xm_bt, t_int + interval);
    % Xm_cloud(i,:) = procNoise(Xm_cloud(i,:)); % Adds process noise
end

% Initialize variables
Kn = 1; % Number of clusters (original)
K = Kn; % Number of clusters (changeable)
Kmax = 1; % Maximum number of clusters (Kmax = 1 for EnKF)

mu_c = cell(K, 1);
P_c = cell(K, 1);
wm = zeros(K, 1);

% Split propagated cloud into position and velocity data before
% normalization.
rc = Xm_cloud(:,1:3);
vc = Xm_cloud(:,4:6);

mean_rc = mean(rc, 1);
mean_vc = mean(vc, 1);

std_rc = std(rc,0,1);
std_vc = std(vc,0,1);

norm_rc = (rc - mean_rc)./norm(std_rc); % Normalizing the position 
norm_vc = (vc - mean_vc)./norm(std_vc); % Normalizing the velocity

Xm_norm = [norm_rc, norm_vc];

% Cluster using K-means clustering algorithm
% [idx, C] = kmeans(Xm_cloud, K); 
[idx, C] = kmeans(Xm_norm, K); % Cluster just on position and velocity; Normalize the whole thing
colors = ["Red", "Blue", "Green", "Yellow", "Magenta", "Cyan", "Black", "#500000", "#bf5700", "#00274c"];
contourCols = lines(Kmax);

% Convert cluster centers back to non-dimensionalized units
C_unorm = C;
C_unorm(:,1:3) = (C(:,1:3).*std_rc) + mean_rc; % Conversion of position
C_unorm(:,4:6) = (C(:,4:6).*std_vc) + mean_vc; % Conversion of velocity

cPoints = cell(K,1);

% Calculate covariances and weights for each cluster
for k = 1:K
    cluster_points = Xm_cloud(idx == k, :); % Keep clustering very separate from mean, covariance, weight calculations
    cPoints{k} = cluster_points; cSize = size(cPoints{k});
    mu_c{k} = mean(cluster_points, 1); % Cell of GMM means 

    if(cSize(1) == 1)
        P_c{k} = zeros(length(wm));
    else
        P_c{k} = cov(cluster_points); % Cell of GMM covariances 
    end
    wm(k) = size(cluster_points, 1) / size(Xm_norm, 1); % Vector of weights
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

legend_string = "Truth";

% Plot planar projections
figure(2)
set(gcf, 'units','normalized','outerposition',[0 0 1 1])
subplot(2,3,1)
for k = 1:K
    clusterPoints = Xm_cloud(idx == k, :);
    scatter(dist2km*clusterPoints(:,1), dist2km*clusterPoints(:,2), 'filled', ...
        'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
    hold on;
end
% plot(dist2km*C_unorm(:,1), dist2km*C_unorm(:,2), '+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(dist2km*Xprop_truth(1), dist2km*Xprop_truth(2), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('X-Y');
xlabel('X (km.)');
ylabel('Y (km.)');
legend(legend_string);
hold off;

subplot(2,3,2)
for k = 1:K
    clusterPoints = Xm_cloud(idx == k, :);
    scatter(dist2km*clusterPoints(:,1), dist2km*clusterPoints(:,3), 'filled', ...
        'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
    hold on;
end
% plot(dist2km*C_unorm(:,1), dist2km*C_unorm(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(dist2km*Xprop_truth(1), dist2km*Xprop_truth(3), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('X-Z');
xlabel('X (km.)');
ylabel('Z (km.)');
legend(legend_string);
hold off;

subplot(2,3,3)
for k = 1:K
    clusterPoints = Xm_cloud(idx == k, :);
    scatter(dist2km*clusterPoints(:,2), dist2km*clusterPoints(:,3), 'filled', ...
        'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
    hold on;
end
% plot(dist2km*C_unorm(:,2), dist2km*C_unorm(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(dist2km*Xprop_truth(2), dist2km*Xprop_truth(3), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('Y-Z');
xlabel('Y (km.)');
ylabel('Z (km.)');
legend(legend_string);
hold off;

subplot(2,3,4)
for k = 1:K
    clusterPoints = Xm_cloud(idx == k, :);
    scatter(vel2kms*clusterPoints(:,4), vel2kms*clusterPoints(:,5), 'filled', ...
        'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
    hold on;
end
% plot(vel2kms*C_unorm(:,4), vel2kms*C_unorm(:,5), '+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(5), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('Xdot-Ydot');
xlabel('Xdot (km/s)');
ylabel('Ydot (km/s)');
legend(legend_string);
hold off;

subplot(2,3,5)
for k = 1:K
    clusterPoints = Xm_cloud(idx == k, :);
    scatter(vel2kms*clusterPoints(:,4), vel2kms*clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
end
% plot(vel2kms*C_unorm(:,4), vel2kms*C_unorm(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('Xdot-Zdot');
xlabel('Xdot (km/s)');
ylabel('Zdot (km/s)');
legend(legend_string);
hold off;

subplot(2,3,6)
for k = 1:K
    clusterPoints = Xm_cloud(idx == k, :);
    scatter(vel2kms*clusterPoints(:,5), vel2kms*clusterPoints(:,6), 'filled', ...
        'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
    hold on;
end
% plot(vel2kms*C_unorm(:,5), vel2kms*C_unorm(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3, 'HandleVisibility', 'off');
% hold on;
plot(vel2kms*Xprop_truth(5), vel2kms*Xprop_truth(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('Ydot-Zdot');
xlabel('Ydot (km/s)');
ylabel('Zdot (km/s)');
legend(legend_string);
hold off;

sg = sprintf('Timestep: %3.4f Hours (Prior)', time2hr*full_ts(idx_prop+1,1));
sgtitle(sg);
saveas(gcf, './Simulations/Timestep_0_1B', 'png');
% saveas(gcf, './Simulations/Different Orbit Simulations/Timestep_0_1B', 'png');

Xprop_truth = [full_ts(idx_prop+1,2:4), full_vts(idx_prop+1,2:4)];
fprintf('Truth State: \n');
disp(Xprop_truth);

% Now that we have a GMM representing the prior distribution, we have to
% use a Kalman update for each component: weight, mean, and covariance.

% Posterior variables
wp = wm;
mu_p = mu_c;
P_p = P_c;

% Comment this out if you wish to use noise.
% noised_obs = partial_ts;

tpr = t_int + interval; % Time stamp of the prior means, weights, and covariances
[idx_meas, ~] = find(abs(noised_obs(:,1) - tpr) < 1e-10); % Find row with time

if (idx_meas ~= 0) % i.e. there exists a measurement
    R_vv = [theta_f*pi/648000, 0; 0, theta_f*pi/648000].^2;
    h = @(x) [atan2(x(2),x(1)); pi/2 - acos(x(3)/sqrt(x(1)^2 + x(2)^2 + x(3)^2))]; % Nonlinear measurement model
    zt = getNoisyMeas(Xprop_truth, R_vv, h);

    for i = 1:K 
        % [mu_p{i}, P_p{i}] = ukfUpdate(zt, R_vv, mu_c{i}, P_c{i}, h);
        [mu_p{i}, P_p{i}] = kalmanUpdate(zt, cPoints{i}, R_vv, mu_c{i}, P_c{i}, h);
    end

    % Weight update
    wp = weightUpdate(wm, Xm_cloud, idx, zt, R_vv, h);

else
    for i = 1:K
        wp(i) = wm(i);
        mu_p{i} = mu_c{i};
        P_p{i} = P_c{i};
    end
end
    
Xp_cloud = Xm_cloud;
c_id = zeros(length(Xp_cloud(:,1)),1);
parfor i = 1:L
    [Xp_cloud(i,:), c_id(i)] = drawFrom(wp, mu_p, P_p); 
end

mu_pExp = zeros(K, length(mu_p{1}));

% Plot the results
figure(3)
subplot(2,1,1)
hold on;
for k = 1:K
    clusterPoints = Xp_cloud(c_id == k, :);
    mu_pExp(k,:) = mu_p{k};
    scatter3(dist2km*clusterPoints(:,1), dist2km*clusterPoints(:,2), dist2km*clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
end
plot3(dist2km*mu_pExp(:,1), dist2km*mu_pExp(:,2), dist2km*mu_pExp(:,3), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
hold on;
plot3(dist2km*Xprop_truth(1), dist2km*Xprop_truth(2), dist2km*Xprop_truth(3), 'x','MarkerSize', 20, 'LineWidth', 3)
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
for k = 1:K
    clusterPoints = Xp_cloud(c_id == k, :);
    scatter3(vel2kms*clusterPoints(:,4), vel2kms*clusterPoints(:,5), vel2kms*clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
end
plot3(vel2kms*mu_pExp(:,4), vel2kms*mu_pExp(:,5), vel2kms*mu_pExp(:,6), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
hold on;
plot3(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(5), vel2kms*Xprop_truth(6), 'x','MarkerSize', 20, 'LineWidth', 3)
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
for k = 1:K
    clusterPoints = Xp_cloud(c_id == k, :);
    mu_pExp(k,:) = mu_p{k};
    scatter(dist2km*clusterPoints(:,1), dist2km*clusterPoints(:,2), 'filled', ...
        'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
    hold on;
end
plot(dist2km*Xprop_truth(1), dist2km*Xprop_truth(2), 'kx','MarkerSize', 20, 'LineWidth', 3)
title('X-Y');
xlabel('X (km.)');
ylabel('Y (km.)');
legend(legend_string);
hold off;

subplot(2,3,2)
for k = 1:K
    clusterPoints = Xp_cloud(c_id == k, :);
    mu_pExp(k,:) = mu_p{k};
    scatter(dist2km*clusterPoints(:,1), dist2km*clusterPoints(:,3), 'filled', ...
        'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
    hold on;
end
plot(dist2km*Xprop_truth(1), dist2km*Xprop_truth(3), 'kx','MarkerSize', 20, 'LineWidth', 3)
title('X-Z');
xlabel('X (km.)');
ylabel('Z (km.)');
legend(legend_string);
hold off;

subplot(2,3,3)
for k = 1:K
    clusterPoints = Xp_cloud(c_id == k, :);
    mu_pExp(k,:) = mu_p{k};
    scatter(dist2km*clusterPoints(:,2), dist2km*clusterPoints(:,3), 'filled', ...
        'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
    hold on;
end
plot(dist2km*Xprop_truth(2), dist2km*Xprop_truth(3), 'kx','MarkerSize', 20, 'LineWidth', 3)
title('Y-Z');
xlabel('Y (km.)');
ylabel('Z (km.)');
legend(legend_string);
hold off;

subplot(2,3,4)
for k = 1:K
    clusterPoints = Xp_cloud(c_id == k, :);
    mu_pExp(k,:) = mu_p{k};
    scatter(vel2kms*clusterPoints(:,4), vel2kms*clusterPoints(:,5), 'filled', ... 
        'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
    hold on;
end
% plot(vel2kms*mu_pExp(:,4), vel2kms*mu_pExp(:,5), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(5), 'kx','MarkerSize', 20, 'LineWidth', 3)
title('Xdot-Ydot');
xlabel('Xdot (km/s)');
ylabel('Ydot (km/s)');
legend(legend_string);
hold off;

subplot(2,3,5)
for k = 1:K
    clusterPoints = Xp_cloud(c_id == k, :);
    mu_pExp(k,:) = mu_p{k};
    scatter(vel2kms*clusterPoints(:,4), vel2kms*clusterPoints(:,6), 'filled', ...
        'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
    hold on;
end
% plot(vel2kms*mu_pExp(:,4), vel2kms*mu_pExp(:,6), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(6), 'kx','MarkerSize', 20, 'LineWidth', 3)
title('Xdot-Zdot');
xlabel('Xdot (km/s)');
ylabel('Zdot (km/s)');
legend(legend_string);
hold off;

subplot(2,3,6)
for k = 1:K
    clusterPoints = Xp_cloud(c_id == k, :);
    mu_pExp(k,:) = mu_p{k};
    scatter(vel2kms*clusterPoints(:,5), vel2kms*clusterPoints(:,6), 'filled', ...
        'HandleVisibility', 'off', 'MarkerFaceColor', colors(k));
    hold on;
end
% plot(vel2kms*mu_pExp(:,5), vel2kms*mu_pExp(:,6), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(vel2kms*Xprop_truth(5), vel2kms*Xprop_truth(6), 'kx','MarkerSize', 20, 'LineWidth', 3)
title('Ydot-Zdot');
xlabel('Ydot (km/s)');
ylabel('Zdot (km/s)');
legend(legend_string);
hold off;

sg = sprintf('Timestep: %3.4f Hours (Posterior)', time2hr*noised_obs(idx_meas,1));
sgtitle(sg);
saveas(gcf,'./Simulations/Timestep_0_2B.png', 'png')
% saveas(gcf,'./Simulations/Different Orbit Simulations/Timestep_0_2B.png', 'png')

% At this point, we have shown a PGM-I propagation and update step. The
% next step is to utilize this PGM-I update across all time steps during
% which the target is within our sensor FOV and see how the particle clouds
% (i.e. GM components) evolve over time. If we're lucky, we should see that
% the GMM tracks the truth over the interval.

% Find and set the start and end times to simulation
[idx_meas, c_meas] = find(abs(hdR(:,1) - tpr) < 1e-10);
interval = hdR(idx_meas,c_meas) - hdR(idx_meas-1,c_meas);

[idx_crit, ~] = find(abs(full_ts(:,1)) >= (28*24)/time2hr, 1, 'first'); % Find the index of the last time step before a certain number of days have passed since orbit propagation
t_end = full_ts(end,1); % First observation of new pass + one more time step

tau = 0;
[idx_end, ~] = find(abs(full_ts(:,1) - t_end) < 1e-10);
[idx_start, ~] = find(abs(full_ts(:,1) - tpr) < 1e-10);

l_filt = length(full_ts(idx_start:idx_end,1))+1;

ent2 = zeros(l_filt+1,1);
ent1 = zeros(l_filt+1,length(mu_c{1})); 

ent2(1) = log(det(cov(X0cloud)));
ent2(2) = log(det(cov(Xp_cloud)));
ent1(1,:) = getDiagCov(X0cloud); Xp_cloudp = Xp_cloud;

% for to = tpr:interval:(t_end-1e-11) % Looping over the times of observation for easier propagation
for ts = idx_start:(idx_end-1)

    to = full_ts(ts,1);
    interval = full_ts(ts+1,1) - full_ts(ts,1);

    % Resampling Step (needlessly repeated)
    % if(idx_meas ~= 0)
    %     Xp_cloud = Xm_cloud;
    %     parfor i = 1:Lp
    %         [Xp_cloud(i,:), ~] = drawFrom(wp, mu_p, P_p); 
    %     end 
    % end

    ent1(tau+2,:) = getDiagCov(Xp_cloudp);

    % Propagation Step
    Xm_cloud = propagate(Xp_cloudp, to, interval);
    Xprop_truth = propagate(Xprop_truth, to, interval);

    % Verification Step
    tpr = to + interval; % Time stamp of the prior means, weights, and covariances
    [idx_meas, ~] = find(abs(noised_obs(:,1) - tpr) < 1e-10); % Find row with time
    tau = tau + 1;

    if(idx_meas ~= 0)  
        % Split propagated cloud into position and velocity data before
        % normalization.
        % parfor j = 1:length(Xm_cloud(:,1))
        %     Xm_cloud(j,:) = procNoise(Xm_cloud(j,:)); % Adds process noise only when measurement is to be made
        % end

        if (tpr >= cVal)
            K = Kmax;
        else
            K = Kn;
        end

        rc = Xm_cloud(:,1:3);
        vc = Xm_cloud(:,4:6);
    
        mean_rc = mean(rc, 1);
        mean_vc = mean(vc, 1);
    
        std_rc = std(rc,0,1);
        std_vc = std(vc,0,1);
    
        norm_rc = (rc - mean_rc)./norm(std_rc); % Normalizing the position 
        norm_vc = (vc - mean_vc)./norm(std_vc); % Normalizing the velocity
    
        Xm_norm = [norm_rc, norm_vc];
    
        % Verification Step
        [idx_meas, ~] = find(abs(noised_obs(:,1) - tpr) < 1e-10); % Find row with time

        fprintf("Timestamp: %1.5f\n", tpr*time2hr);
        
        % Cluster using K-means clustering algorithm
        [idx, ~] = kmeans(Xm_norm, K);

        cPoints = cell(K, 1);
        mu_c = cell(K, 1); mu_p = mu_c;
        P_c = cell(K, 1); P_p = P_c;
        wm = zeros(K, 1); wp = wm;

        % Calculate covariances and weights for each cluster
        parfor k = 1:K
            cluster_points = Xm_cloud(idx == k, :); 
            cPoints{k} = cluster_points; 
            mu_c{k} = mean(cluster_points, 1); % Cell of GMM means 
            if (length(cluster_points(:,1)) == 1)
                P_c{k} = zeros(length(mu_c{k}));
            else
                P_c{k} = cov(cluster_points); % Cell of GMM covariances
            end
            wm(k) = size(cluster_points, 1) / size(Xm_cloud, 1); % Vector of (prior) weights
        end

        % Extract means
        mu_mExp = zeros(K,length(mu_c{1}));
        parfor k = 1:K
            mu_mExp(k,:) = mu_c{k};
        end

        % [idx_trth, ~] = find(abs(full_ts(:,1) - tpr) < 1e-10);
        % Xprop_truth = [full_ts(idx_trth,2:4), full_vts(idx_trth,2:4)]';

        zc = noised_obs(idx_meas,2:4)'; % Presumption: An observation occurs at this time step
        xto = zc(1)*cos(zc(2))*cos(zc(3)); 
        yto = zc(1)*sin(zc(2))*cos(zc(3)); 
        zto = zc(1)*sin(zc(3)); 
        rto = [xto, yto, zto];

        legend_string = {};
        parfor k = 1:K
            R_vv = [R_f*partial_ts(idx_meas,2), 0, 0; 0 theta_f*pi/648000, 0; 0, 0, theta_f*pi/648000].^2;
            Hxk = linHx(mu_c{k}); % Linearize about prior mean component
            legend_string{k} = sprintf('Distribution %i',k);
            % legend_string{K+k} = sprintf('\\omega =  %1.4f, l = %1.4d', wm(k), gaussProb(zc, h(mu_c{k}), Hxk*P_c{k}*Hxk' + R_vv));
        end
        % legend_string{K+1} = "Centroids";
        legend_string{K+1} = "Truth";

        if(1) % Use for all time steps
            % legend_string{K+1} = "Centroids";
            legend_string{K+1} = "Truth";
    
            mu_mat = cell2mat(mu_c);
            P_mat = cat(3, P_c{:});
    
            %{
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
                if isempty(Z_cell{k}) || isempty(contours_cell{k})
                    continue;
                end

                contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
            end    

            % Overlay scatter points
            scatter(dist2km*Xm_cloud(:, plot_dims(1)), dist2km*Xm_cloud(:, plot_dims(2)), ...
                'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');

            % Overlay a special marker for truth
            scatter(dist2km*Xprop_truth(1), dist2km*Xprop_truth(2), ...
                200, 'k', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'Marker', 'x');
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

                contours_cell{k} = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals 
            end 
    
            hold on;
            for k = 1:K
                if isempty(Z_cell{k}) || isempty(contours_cell{k})
                    continue;
                end

                contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
            end 

            % Overlay scatter points
            scatter(dist2km*Xm_cloud(:, plot_dims(1)), dist2km*Xm_cloud(:, plot_dims(2)), ...
                'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');

            scatter(dist2km*Xprop_truth(1), dist2km*Xprop_truth(3), ...
                200, 'k', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'Marker', 'x');
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
                if isempty(Z_cell{k}) || isempty(contours_cell{k})
                    continue;
                end

                contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
            end 

            % Overlay scatter points
            scatter(dist2km*Xm_cloud(:, plot_dims(1)), dist2km*Xm_cloud(:, plot_dims(2)), ...
                'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');

            scatter(dist2km*Xprop_truth(2), dist2km*Xprop_truth(3), ...
                200, 'k', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'Marker', 'x');

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
                if isempty(Z_cell{k}) || isempty(contours_cell{k})
                    continue;
                end

                contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
            end 

            % Overlay scatter points
            scatter(vel2kms*Xm_cloud(:, plot_dims(1)), vel2kms*Xm_cloud(:, plot_dims(2)), ...
                'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');

            scatter(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(5), ...
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
                if isempty(Z_cell{k}) || isempty(contours_cell{k})
                    continue;
                end

                contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
            end     
            scatter(vel2kms*Xm_cloud(:, plot_dims(1)), vel2kms*Xm_cloud(:, plot_dims(2)), ...
                'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');

            scatter(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(6), ...
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
                if isempty(Z_cell{k}) || isempty(contours_cell{k})
                    continue;
                end

                contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
            end     
            % Overlay scatter points
            scatter(vel2kms*Xm_cloud(:, plot_dims(1)), vel2kms*Xm_cloud(:, plot_dims(2)), ...
                'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');

            scatter(vel2kms*Xprop_truth(5), vel2kms*Xprop_truth(6), ...
                200, 'k', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'Marker', 'x');
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

            f = figure('visible','off','Position', get(0,'ScreenSize'));
            f.WindowState = 'maximized';
    
            legend_string = "Truth";
    
            subplot(2,3,1)
            % gscatter(Xm_cloud(:,1), Xm_cloud(:,2), idx);
            % hold on;
            % plot(mu_mExp(:,1), mu_mExp(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
            % hold on;
    
            parfor k = 1:K
                cPoints{k} = Xm_cloud(idx == k, :);
                mu_mExp(k,:) = mu_c{k};
            end
            hold on; 
            for k = 1:K
                scatter(dist2km*cPoints{k}(:,1), dist2km*cPoints{k}(:,2), 'filled', ...
                'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
            end
    
            plot(dist2km*Xprop_truth(1), dist2km*Xprop_truth(2), 'kx', ... 
                'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
            % hold on;
            % plot(rto(1), rto(2), 'o', 'MarkerSize', 10, 'LineWidth', 3);
            title('X-Y');
            xlabel('X (km.)');
            ylabel('Y (km.)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,2)
            
            parfor k = 1:K
                cPoints{k} = Xm_cloud(idx == k, :);
                mu_mExp(k,:) = mu_c{k};
            end
            hold on; 
            for k = 1:K
                scatter(dist2km*cPoints{k}(:,1), dist2km*cPoints{k}(:,3), 'filled', ...
                'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
            end
            plot(dist2km*Xprop_truth(1), dist2km*Xprop_truth(3), 'kx', ... 
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
            
            parfor k = 1:K
                cPoints{k} = Xm_cloud(idx == k, :);
                mu_mExp(k,:) = mu_c{k};
            end
            hold on; 
            for k = 1:K
                scatter(dist2km*cPoints{k}(:,2), dist2km*cPoints{k}(:,3), 'filled', ...
                'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
            end
            plot(dist2km*Xprop_truth(2), dist2km*Xprop_truth(3), 'kx', ... 
                'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
            % hold on;
            % plot(rto(2), rto(3), 'o', 'MarkerSize', 10, 'LineWidth', 3);
            title('Y-Z');
            xlabel('Y (km.)');
            ylabel('Z (km.)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,4)
            
            parfor k = 1:K
                cPoints{k} = Xm_cloud(idx == k, :);
                mu_mExp(k,:) = mu_c{k};
            end
            hold on; 
            for k = 1:K
                scatter(vel2kms*cPoints{k}(:,4), vel2kms*cPoints{k}(:,5), 'filled', ...
                'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
            end
            plot(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(5), 'kx', ... 
                'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
            title('Xdot-Ydot');
            xlabel('Xdot (km/s)');
            ylabel('Ydot (km/s)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,5)
            
            parfor k = 1:K
                cPoints{k} = Xm_cloud(idx == k, :);
                mu_mExp(k,:) = mu_c{k};
            end
            hold on; 
            for k = 1:K
                scatter(vel2kms*cPoints{k}(:,4), vel2kms*cPoints{k}(:,6), 'filled', ...
                'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
            end
            plot(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(6), 'kx', ... 
                'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
            title('Xdot-Zdot');
            xlabel('Xdot (km/s)');
            ylabel('Zdot (km/s)');
            legend(legend_string);
            hold off;
            
            subplot(2,3,6)
    
            parfor k = 1:K
                cPoints{k} = Xm_cloud(idx == k, :);
                mu_mExp(k,:) = mu_c{k};
            end
            hold on; 
            for k = 1:K
                scatter(vel2kms*cPoints{k}(:,5), vel2kms*cPoints{k}(:,6), 'filled', ...
                'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
            end
            plot(vel2kms*Xprop_truth(5), vel2kms*Xprop_truth(6), 'kx', ... 
                'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
            title('Ydot-Zdot');
            xlabel('Ydot (km/s)');
            ylabel('Zdot (km/s)');
            legend(legend_string);
            hold off;
    
            sgt = sprintf('Timestep: %3.4f Hours (Prior)', tpr*time2hr);
            sgtitle(sgt);
    
            sg = sprintf('./Simulations/Timestep_%i_1B.png', tau);
            % sg = sprintf('./Simulations/Different Orbit Simulations/Timestep_%i_1B.png', tau);
            saveas(f, sg, 'png');
            close(f);
            %}

            f = figure('visible','off','Position', get(0,'ScreenSize'));
            f.WindowState = 'maximized';
    
            legend_string = "Truth";
            hold on;

            for k = 1:K
                Zmcloud = zeros(length(cPoints{k}(:,1)), length(zt));
                for i = 1:length(Zmcloud(:,1))
                    Zmcloud(i,:) = h(cPoints{k}(i,:))';
                end
    
                Ztruth = h(Xprop_truth)';
                scatter(180/pi*Zmcloud(:,1), 180/pi*Zmcloud(:,2), 'filled', ...
                'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');

                plot(180/pi*Ztruth(1), 180/pi*Ztruth(2), 'kx', ... 
                'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
                
                title('AZ-EL')
                xlabel('Azimuth Angle (deg)')
                ylabel('Elevation Angle (deg)')
            end

            sg = sprintf('./Simulations/Timestep_%i_1C.png', tau);
            % sg = sprintf('./Simulations/Different Orbit Simulations/Timestep_%i_1B.png', tau);
            saveas(f, sg, 'png');
            close(f);
        end
  
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

        % Update Step
        R_vv = [theta_f*pi/648000, 0; 0, theta_f*pi/648000].^2;
        % Hxk = linHx(mu_c{i}); % Linearize about prior mean component
        h = @(x) [atan2(x(2),x(1)); pi/2 - acos(x(3)/sqrt(x(1)^2 + x(2)^2 + x(3)^2))]; % Nonlinear measurement model
        zt = getNoisyMeas(Xprop_truth, R_vv, h);

        for i = 1:K
            % [mu_p{i}, P_p{i}] = ukfUpdate(zt, R_vv, mu_c{i}, P_c{i}, h);
            [mu_p{i}, P_p{i}] = kalmanUpdate(zt, cPoints{i}, R_vv, mu_c{i}, P_c{i}, h);
            P_p{i} = (P_p{i} + P_p{i}')/2;
        end

        % Weight update
        wp = weightUpdate(wm, Xm_cloud, idx, zt, R_vv, h);

    else
        fprintf("Timestamp: %1.5f\n", tpr*time2hr);

        mu_p = cell(1, 1); 
        P_p = cell(1, 1); 
        wm = zeros(1, 1);
        cPoints = cell(1, 1); 

        Xp_cloud = Xm_cloud; cPoints{1} = Xp_cloud;
        wp = [1];
        mu_p{1} = mean(Xp_cloud);
        P_p{1} = cov(Xp_cloud);
    end

    % Resampling
    if (idx_meas ~= 0)
        % K = Kn;
        Xp_cloudp = zeros(Lp, length(Xprop_truth));
        c_id = zeros(Lp,1);
        parfor i = 1:Lp
            [Xp_cloudp(i,:), c_id(i)] = drawFrom(wp, mu_p, P_p); 
        end
    else
        K = 1;
        Xp_cloudp = Xm_cloud; c_id = ones(length(Xp_cloudp(:,1)),1);
    end

    if(1)
        % [idx_trth, ~] = find(abs(full_ts(:,1) - tpr) < 1e-10);
        % Xprop_truth = [full_ts(idx_trth,2:4), full_vts(idx_trth,2:4)];

        % Extract means
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
        
        %{
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
            if isempty(Z_cell{k}) || isempty(contours_cell{k})
                continue;
            end

            contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end 
        % Overlay scatter points
        scatter(dist2km*Xp_cloudp(:, plot_dims(1)), dist2km*Xp_cloudp(:, plot_dims(2)), ...
                'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');

        scatter(dist2km*Xprop_truth(1), dist2km*Xprop_truth(2), ...
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
    
        [X1, X2] = meshgrid(linspace(min(Xp_cloudp(:,plot_dims(1))), max(Xp_cloudp(:,plot_dims(1))), 100), ...
                            linspace(min(Xp_cloudp(:,plot_dims(2))), max(Xp_cloudp(:,plot_dims(2))), 100));
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
            if isempty(Z_cell{k}) || isempty(contours_cell{k})
                continue;
            end

            contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end 
        % Overlay scatter points
        scatter(dist2km*Xp_cloudp(:, plot_dims(1)), dist2km*Xp_cloudp(:, plot_dims(2)), ...
                'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');

        scatter(dist2km*Xprop_truth(1), dist2km*Xprop_truth(3), ...
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
    
        [X1, X2] = meshgrid(linspace(min(Xp_cloudp(:,plot_dims(1))), max(Xp_cloudp(:,plot_dims(1))), 100), ...
                            linspace(min(Xp_cloudp(:,plot_dims(2))), max(Xp_cloudp(:,plot_dims(2))), 100));
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
            if isempty(Z_cell{k}) || isempty(contours_cell{k})
                continue;
            end

            contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end 
        % Overlay scatter points
        scatter(dist2km*Xp_cloudp(:, plot_dims(1)), dist2km*Xp_cloudp(:, plot_dims(2)), ...
                'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');

        scatter(dist2km*Xprop_truth(2), dist2km*Xprop_truth(3), ...
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
    
        [X1, X2] = meshgrid(linspace(min(Xp_cloudp(:,plot_dims(1))), max(Xp_cloudp(:,plot_dims(1))), 100), ...
                            linspace(min(Xp_cloudp(:,plot_dims(2))), max(Xp_cloudp(:,plot_dims(2))), 100));
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
            if isempty(Z_cell{k}) || isempty(contours_cell{k})
                continue;
            end

            contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end 
        % Overlay scatter points
        scatter(vel2kms*Xp_cloudp(:, plot_dims(1)), vel2kms*Xp_cloudp(:, plot_dims(2)), ...
                'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');

        scatter(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(5), ...
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
            if isempty(Z_cell{k}) || isempty(contours_cell{k})
                continue;
            end

            contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end 
        % Overlay scatter points
        scatter(vel2kms*Xp_cloudp(:, plot_dims(1)), vel2kms*Xp_cloudp(:, plot_dims(2)), ...
                'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');

        scatter(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(6), ...
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
    
        [X1, X2] = meshgrid(linspace(min(Xp_cloudp(:,plot_dims(1))), max(Xp_cloudp(:,plot_dims(1))), 100), ...
                            linspace(min(Xp_cloudp(:,plot_dims(2))), max(Xp_cloudp(:,plot_dims(2))), 100));
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
            if isempty(Z_cell{k}) || isempty(contours_cell{k})
                continue;
            end

            contour(dist2km*X1, dist2km*X2, dist2km*Z_cell{k}, dist2km*contours_cell{k}, 'LineWidth', 2, 'LineColor', contourCols(k,:));
        end 
        % Overlay scatter points
        scatter(vel2kms*Xp_cloudp(:, plot_dims(1)), vel2kms*Xp_cloudp(:, plot_dims(2)), ...
                'filled', 'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');

        scatter(vel2kms*Xprop_truth(5), vel2kms*Xprop_truth(6), ...
                200, 'k', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'Marker', 'x');
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
        for k = 1:K
            scatter(dist2km*Xp_cloudp(c_id == k,1), dist2km*Xp_cloudp(c_id == k,2), 'filled', ...
                'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
        end
        
    
        % plot(mu_pExp(:,1), mu_pExp(:,2), '+', 'MarkerSize', 10, 'LineWidth', 3);
        % hold on;
        plot(dist2km*Xprop_truth(1), dist2km*Xprop_truth(2), 'kx', ...
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
        for k = 1:K
            scatter(dist2km*Xp_cloudp(c_id == k,1), dist2km*Xp_cloudp(c_id == k,3), 'filled', ...
                'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
        end
        plot(dist2km*Xprop_truth(1), dist2km*Xprop_truth(3), 'kx', ...
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
        for k = 1:K
            scatter(dist2km*Xp_cloudp(c_id == k,2), dist2km*Xp_cloudp(c_id == k,3), 'filled', ...
                'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
        end
        plot(dist2km*Xprop_truth(2), dist2km*Xprop_truth(3), 'kx', ...
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
        for k = 1:K
            scatter(vel2kms*Xp_cloudp(c_id == k,4), vel2kms*Xp_cloudp(c_id == k,5), 'filled', ...
                'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
        end
        plot(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(5), 'kx', ...
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
        for k = 1:K
            scatter(vel2kms*Xp_cloudp(c_id == k,4), vel2kms*Xp_cloudp(c_id == k,6), 'filled', ...
                'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
        end
        plot(vel2kms*Xprop_truth(4), vel2kms*Xprop_truth(6), 'kx', ...
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
        for k = 1:K
            scatter(vel2kms*Xp_cloudp(c_id == k,5), vel2kms*Xp_cloudp(c_id == k,6), 'filled', ...
                'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');
        end
        plot(vel2kms*Xprop_truth(5), vel2kms*Xprop_truth(6), 'kx', ...
            'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string)
        title('Ydot-Zdot');
        xlabel('Ydot (km/s)');
        ylabel('Zdot (km/s)');
        legend(legend_string);
        hold off;
    
        sgt = sprintf('Timestep: %3.4f Hours (Posterior)', tpr*time2hr);
        sgtitle(sgt);
    
        sg = sprintf('./Simulations/Timestep_%i_2B.png', tau);
        % sg = sprintf('./Simulations/Different Orbit Simulations/Timestep_%i_2B.png', tau);
        saveas(f, sg, 'png');
        close(f);
        %}

        f = figure('visible','off','Position', get(0,'ScreenSize'));
        f.WindowState = 'maximized';

        legend_string = "Truth";
        hold on;

        for k = 1:K
            pts = Xp_cloudp(c_id == k, :);
            Zmcloud = zeros(length(pts(:,1)), length(zt));
            for i = 1:length(Zmcloud(:,1))
                Zmcloud(i,:) = h(pts(i,:))';
            end

            Ztruth = h(Xprop_truth)';
            scatter(180/pi*Zmcloud(:,1), 180/pi*Zmcloud(:,2), 'filled', ...
            'MarkerFaceColor', colors(k), 'HandleVisibility', 'off');

            plot(180/pi*Ztruth(1), 180/pi*Ztruth(2), 'kx', ... 
            'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', legend_string);
            
            title('AZ-EL')
            xlabel('Azimuth Angle (deg)')
            ylabel('Elevation Angle (deg)')
        end

        sg = sprintf('./Simulations/Timestep_%i_2C.png', tau);
        saveas(f, sg, 'png');
        close(f);
    end

    if (idx_meas ~= 0)
        wsum = 0;
        for k = 1:K
            wsum = wsum + wp(k)*det(P_p{k});
        end
        ent2(tau+2) = log(wsum);
    else
        if (tpr >= cVal)
            Ke = Kmax; % Clusters used for calculating entropy
        else
            Ke = Kn; % Clusters used for calculating entropy
        end
        ent2(tau+2) = getKnEntropy(Ke, Xp_cloudp); % Get entropy as if you still are using six clusters
    end

    if(abs(tpr - cTimes(2)) < 1e-10)
        Lp = 1500;
    elseif(abs(tpr - cTimes(4)) < 1e-10)
        Lp = 2000;
    elseif(abs(tpr - cTimes(6)) < 1e-10)
        Lp = 2500;
        save("Xm_cloud.mat", "Xp_cloudp"); save("t_int.mat", "tpr"); save("noised_obs.mat", "noised_obs"); save("Xtruth.mat", "Xprop_truth");
    end
    %}

end

Xp_cloudp = zeros(Lp, length(Xprop_truth));
c_id = zeros(Lp,1);
parfor i = 1:Lp
    [Xp_cloudp(i,:), c_id(i)] = drawFrom(wp, mu_p, P_p); 
end
ent1(end,:) = getDiagCov(Xp_cloudp);
ent2(end) = [];

figure(7)

subplot(2,3,1)
plot(0:l_filt, dist2km*sqrt(ent1(:,1)))
xlabel('Filter Step #')
ylabel('Log \\sigma_X (km.)')
title('X Standard Deviation')

subplot(2,3,2)
plot(0:l_filt, dist2km*sqrt(ent1(:,2)))
xlabel('Filter Step #')
ylabel('Log \\sigma_Y (km.)')
title('Y Standard Deviation')

subplot(2,3,3)
plot(0:l_filt, dist2km*sqrt(ent1(:,3)))
xlabel('Filter Step #')
ylabel('Log \\sigma_Z (km.)')
title('Z Standard Deviation')

subplot(2,3,4)
plot(0:l_filt, vel2kms*sqrt(ent1(:,4)))
xlabel('Filter Step #')
ylabel('Log \\sigma_Xdot (km/s)')
title('Xdot Standard Deviation')

subplot(2,3,5)
plot(0:l_filt, vel2kms*sqrt(ent1(:,5)))
xlabel('Filter Step #')
ylabel('\\sigma_Ydot (km/s)')
title('Ydot Standard Deviation')

subplot(2,3,6)
plot(0:l_filt, vel2kms*sqrt(ent1(:,6)))
xlabel('Filter Step #')
ylabel('\\sigma_Zdot (km/s)')
title('Zdot Standard Deviation')

savefig(gcf, './Simulations/StDevEvols.fig');

% Xprop_truth = [full_ts(idx_end,2:4), full_vts(idx_end,2:4)];
% mu_pExp = zeros(K, length(mu_p{1}));

fprintf('Final State Truth:\n')
disp(Xprop_truth);

%%
% Plot the results
figure(6)
plot(0:l_filt-1, ent2)
xlabel('Filter Step #')
ylabel('Entropy Metric')
title('Entropy')
savefig(gcf,'./Simulations/Entropy.fig');

%%
% Plot the results
figure(9)
subplot(2,1,1)
hold on;
for k = 1:K
    clusterPoints = Xp_cloudp(c_id == k, :);
    mu_pExp(k,:) = mu_p{k};
    scatter3(clusterPoints(:,1), clusterPoints(:,2), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
end
plot3(mu_pExp(:,1), mu_pExp(:,2), mu_pExp(:,3), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
hold on;
plot3(Xprop_truth(1), Xprop_truth(2), Xprop_truth(3), 'x','MarkerSize', 20, 'LineWidth', 3)
title('Posterior Distribution (Position)');
xlabel('X');
ylabel('Y');
zlabel('Z');
legend(legend_string);
grid on;
view(3);
hold off;

subplot(2,1,2)
hold on;
for k = 1:K
    clusterPoints = Xp_cloudp(c_id == k, :);
    scatter3(clusterPoints(:,4), clusterPoints(:,5), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
end
plot3(mu_pExp(:,4), mu_pExp(:,5), mu_pExp(:,6), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
hold on;
plot3(Xprop_truth(4), Xprop_truth(5), Xprop_truth(6), 'x','MarkerSize', 20, 'LineWidth', 3)
title('Posterior Distribution (Velocity)');
xlabel('Vx');
ylabel('Vy');
zlabel('Vz');
legend(legend_string);
grid on;
view(3);
hold off;

% Plot planar projections
figure(10)
subplot(2,3,1)
for k = 1:K
    clusterPoints = Xp_cloudp(c_id == k, :);
    scatter(clusterPoints(:,1), clusterPoints(:,2), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
end
plot(mu_pExp(:,1), mu_pExp(:,2), '+', 'MarkerSize', 10, 'LineWidth', 3);
hold on;
plot(Xprop_truth(1), Xprop_truth(2), 'kx','MarkerSize', 20, 'LineWidth', 3)
title('X-Y');
xlabel('X');
ylabel('Y');
legend(legend_string);
hold off;

subplot(2,3,2)
for k = 1:K
    clusterPoints = Xp_cloudp(c_id == k, :);
    scatter(clusterPoints(:,1), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
end
plot(mu_pExp(:,1), mu_pExp(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
hold on;
plot(Xprop_truth(1), Xprop_truth(3), 'kx','MarkerSize', 20, 'LineWidth', 3)
title('X-Z');
xlabel('X');
ylabel('Z');
legend(legend_string);
hold off;

subplot(2,3,3)
for k = 1:K
    clusterPoints = Xp_cloudp(c_id == k, :);
    scatter(clusterPoints(:,2), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
end
plot(mu_pExp(:,2), mu_pExp(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
hold on;
plot(Xprop_truth(2), Xprop_truth(3), 'kx','MarkerSize', 20, 'LineWidth', 3)
title('Y-Z');
xlabel('Y');
ylabel('Z');
legend(legend_string);
hold off;

subplot(2,3,4)
for k = 1:K
    clusterPoints = Xp_cloudp(c_id == k, :);
    scatter(clusterPoints(:,4), clusterPoints(:,5), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
end
plot(mu_pExp(:,4), mu_pExp(:,5), '+', 'MarkerSize', 10, 'LineWidth', 3);
hold on;
plot(Xprop_truth(4), Xprop_truth(5), 'kx','MarkerSize', 20, 'LineWidth', 3)
title('Xdot-Ydot');
xlabel('Xdot');
ylabel('Ydot');
legend(legend_string);
hold off;

subplot(2,3,5)
for k = 1:K
    clusterPoints = Xp_cloudp(c_id == k, :);
    scatter(clusterPoints(:,4), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
end
plot(mu_pExp(:,4), mu_pExp(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3);
hold on;
plot(Xprop_truth(4), Xprop_truth(6), 'kx','MarkerSize', 20, 'LineWidth', 3)
title('Xdot-Zdot');
xlabel('Xdot');
ylabel('Zdot');
legend(legend_string);
hold off;

subplot(2,3,6)
for k = 1:K
    clusterPoints = Xp_cloudp(c_id == k, :);
    scatter(clusterPoints(:,5), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
end
plot(mu_pExp(:,5), mu_pExp(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3);
hold on;
plot(Xprop_truth(5), Xprop_truth(6), 'kx','MarkerSize', 20, 'LineWidth', 3)
title('Ydot-Zdot');
xlabel('Ydot');
ylabel('Zdot');
legend(legend_string);
hold off;

sg = sprintf('Timestamp: %1.5f', tpr);
sgtitle(sg)

% savefig(gcf, 'nextObservedTracklet_normK.fig');
savefig(gcf, 'finalDistribution_normK.fig');
%}

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
    Q = (0.000*diag(abs(X))).^2; % Process noise is 1% of each state vector component
    Xm = mvnrnd(X,Q);
end

function [Xfit] = stateEstCloud(pf, obTr, tdiff)
    noised_obs = obTr;

    R_t = zeros(3*length(noised_obs(:,1)),1); % We shall diagonalize this later
    mu_t = zeros(3*length(noised_obs(:,1)),1);

    load("partial_ts.mat"); % Noiseless observation data

    for i = 1:length(obTr(:,1))
        mu_t(3*(i-1)+1:3*(i-1)+3, 1) = [partial_ts(i,2); partial_ts(i,3); partial_ts(i,4)];
        R_t(3*(i-1)+1:3*(i-1)+3, 1) = [0.05*partial_ts(i,2); 7.2722e-6; 7.2722e-6].^2;
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

    reo_dim = lla2eci([obs_lat obs_lon, elevation], UTC_vec);
    delt_reodim = lla2eci([obs_lat obs_lon, elevation], delt_updatedUTCvec);
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
    for i = 1:length(Xcloud(:,1))
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