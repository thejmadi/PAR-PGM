% Start the clock
tic

% Load noiseless observation data and other important .mat files
load("partial_ts.mat"); % Noiseless observation data
load("full_ts.mat"); % Position truth (topocentric frame)
load("full_vts.mat"); % Velocity truth (topocentric frame)

load('iodCloud_highTol.mat','Xcloud'); % IOD cloud
load('noisedObs.mat','noised_obs'); % Observations injected with sensor noise
load('t_int.mat'); % Starting timestep of IOD cloud

[idx_prop, ~] = find(abs(full_ts_highTol - t_int) < 1e-11);
Xot_truth = [full_ts_highTol(idx_prop,2:4), full_vts_highTol(idx_prop,2:4)]';


samplePt = Xot_truth;
samplePtlt = Xot_truth_lowTol;

figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,3,1)
plot(Xcloud(:,1), Xcloud(:,2), '.')
hold on;
plot(Xot_truth(1), Xot_truth(2), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('X-Y');
xlabel('X');
ylabel('Y');
legend('Estimate', 'Truth');
hold off;

subplot(2,3,2)
plot(Xcloud(:,1), Xcloud(:,3), '.')
hold on;
plot(Xot_truth(1), Xot_truth(3), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('X-Z');
xlabel('X');
ylabel('Z');
legend('Estimate', 'Truth');
hold off;

subplot(2,3,3)
plot(Xcloud(:,2), Xcloud(:,3), '.')
hold on;
plot(Xot_truth(2), Xot_truth(3), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('Y-Z');
xlabel('Y');
ylabel('Z');
legend('Estimate', 'Truth');
hold off;

subplot(2,3,4)
plot(Xcloud(:,4), Xcloud(:,5), '.')
hold on;
plot(Xot_truth(4), Xot_truth(5), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('Xdot-Ydot');
xlabel('Xdot');
ylabel('Ydot');
legend('Estimate', 'Truth');
hold off;

subplot(2,3,5)
plot(Xcloud(:,4), Xcloud(:,6), '.')
hold on;
plot(Xot_truth(4), Xot_truth(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('Xdot-Zdot');
xlabel('Xdot');
ylabel('Zdot');
legend('Estimate', 'Truth');
hold off;

subplot(2,3,6)
plot(Xcloud(:,5), Xcloud(:,6), '.')
hold on;
plot(Xot_truth(5), Xot_truth(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
title('Ydot-Zdot');
xlabel('Ydot');
ylabel('Zdot');
legend('Estimate', 'Truth');
hold off;

sg = sprintf("Timestep: %1.5f", full_ts_highTol(idx_prop,1));
sgtitle(sg)
savefig('startingEstimateCloud.fig')
saveas(gcf, './Small_Sims/StartingEstimate.png','png');

% Extract important time points from the noised_obs variable
i = 2;
interval = noised_obs(2,1) - noised_obs(1,1);
cTimes = []; % Array of important time points

while (i <= length(noised_obs(:,1)))
    if (noised_obs(i,1) - noised_obs(i-1,1) > (interval+1e-11))
        cTimes = [cTimes, noised_obs(i-1,1), noised_obs(i,1)];
    end
    i = i + 1;
end

% Other important variables
L = length(Xcloud(:,1));
Lp = L;

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

tspan = 0:interval:interval; % Integrate over just a single time step
Xm_cloud_pgm = Xcloud; % Cloud to be updated via PGM filter with 6 clusters
Xm_cloud_enkf = Xcloud; % Cloud to be updated via ENKF

for i = 1:Lp
    % First, convert from X_{ot} in the topocentric frame to X_{bt} in the
    % synodic frame.
    Xbt_pgm = backConvertSynodic(Xm_cloud_pgm(i,:)', t_int);
    Xbt_enkf = backConvertSynodic(Xm_cloud_enkf(i,:)', t_int);

    % Next, propagate each X_{bt} in your particle cloud by a single time 
    % step and convert back to the topographic frame.
    
    % Call ode45()
    opts = odeset('Events', @termSat);
    [~,X] = ode45(@cr3bp_dyn, [0 interval], Xbt_pgm, opts); % Assumes termination event (i.e. target enters LEO)
    Xmlt_pgm = X(end,:)';
    [~,X] = ode45(@cr3bp_dyn, [0 interval], Xbt_enkf, opts); % Assumes termination event (i.e. target enters LEO)
    Xmht_enkf = X(end,:)';
    Xm_cloud_pgm(i,:) = convertToTopo(Xmlt_pgm, t_int + interval);
    Xm_cloud_enkf(i,:) = convertToTopo(Xmht_enkf, t_int + interval);
end

% Initialize variables
Kn = 6; % Number of clusters (original)

tpr = t_int + interval; % Time stamp of the prior means, weights, and covariances
[idx_meas, ~] = find(abs(noised_obs(:,1) - tpr) < 1e-10); % Find row with time

if (idx_meas ~= 0)
    K = Kn; % Number of clusters (changeable)
else
    K = 1;
end
% L = 300*Kn; % Make L larger for larger numbers of clusters

muP_c = cell(K, 1); muE_c = cell(1, 1);
PP_c = cell(K, 1); PE_c = cell(1, 1);
wmP = zeros(K, 1); wmE = zeros(1, 1);

% Split propagated cloud into position and velocity data before
% normalization.
rc = Xm_cloud_pgm(:,1:3);
vc = Xm_cloud_pgm(:,4:6);

mean_rc = mean(rc, 1);
mean_vc = mean(vc, 1);

std_rc = std(rc,0,1);
std_vc = std(vc,0,1);

norm_rc = (rc - mean_rc)./norm(std_rc); % Normalizing the position 
norm_vc = (vc - mean_vc)./norm(std_vc); % Normalizing the velocity

Xm_norm_pgm = [norm_rc, norm_vc];

% Cluster using K-means clustering algorithm
% [idx, C] = kmeans(Xm_cloud, K); 
[idx_lt, C_lt] = kmeans(Xm_norm_pgm, K); % Cluster just on position and velocity; Normalize the whole thing
colors = ["Red", "Blue", "Green", "Yellow", "Magenta", "Cyan", "#8601b3", "#500000", "#bf5700", "#00274c", "#465047", "#b7d393", "#c3fc49"];
contourCols = lines(6);

cPoints_pgm = cell(K,1);

% Calculate covariances and weights for each cluster
for k = 1:K
    cluster_points = Xm_cloud_lowTol(idx_lt == k, :); % Keep clustering very separate from mean, covariance, weight calculations
    cPoints_pgm{k} = cluster_points; cSize = size(cPoints_pgm{k});
    muP_c{k} = mean(cluster_points); % Cell of GMM means 

    if(cSize(1) == 1)
        PP_c{k} = zeros(length(wmP));
    else
        PP_c{k} = cov(cluster_points); % Cell of GMM covariances 
    end
    wmP(k) = size(cluster_points, 1) / size(Xm_norm_pgm, 1); % Vector of weights

    cluster_points = Xm_cloud_highTol(idx_ht == k, :); % Keep clustering very separate from mean, covariance, weight calculations
end

[idx_prop, ~] = find(abs(full_ts_highTol(:,1) - (t_int+interval)) < 1e-11);
Xprop_truth = [full_ts_highTol(idx_prop,2:4), full_vts_highTol(idx_prop,2:4)]';

[idx_prop, ~] = find(abs(full_ts_lowTol(:,1) - (t_int+interval)) < 1e-11);
Xprop_truth_lowTol = [full_ts(idx_prop,2:4), full_vts_lowTol(idx_prop,2:4)]';

% Plot the results
warning('off', 'MATLAB:legend:IgnoringExtraEntries');

legend_string = {};
for k = 1:K
    legend_string{k} = sprintf('\\omega = %1.4f', wmP(k));
    legend_string{K+k} = sprintf('\\omega = %1.4f', wmE(k));
end
% legend_string{1} = "Centroids";
legend_string{2*K+1} = "Truth (High Tol)";
legend_string{2*K+2} = "Truth (Low Tol)";

% Plot planar projections
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,3,1)
for k = 1:K
    clusterPoints = Xm_cloud_lowTol(idx_lt == k, :);
    scatter(clusterPoints(:,1), clusterPoints(:,2), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
    clusterPoints = Xm_cloud_highTol(idx_ht == k, :);
    scatter(clusterPoints(:,1), clusterPoints(:,2), 'filled', 'MarkerFaceColor', colors(K+k));
    hold on;
end
% plot(C_unorm(:,1), C_unorm(:,2), '+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(Xprop_truth(1), Xprop_truth(2), 'kx','MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(Xprop_truth_lowTol(1), Xprop_truth_lowTol(2), 'rx','MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(samplePt(1), samplePt(2), 'o', 'MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(samplePtlt(1), samplePtlt(2), 'o', 'MarkerSize', 15, 'LineWidth', 3)
hold on;
title('X-Y');
xlabel('X');
ylabel('Y');
% legend(legend_string);
hold off;

subplot(2,3,2)
for k = 1:K
    clusterPoints = Xm_cloud_lowTol(idx_lt == k, :);
    scatter(clusterPoints(:,1), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
    clusterPoints = Xm_cloud_highTol(idx_ht == k, :);
    scatter(clusterPoints(:,1), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(K+k));
    hold on;
end
% plot(C_unorm(:,1), C_unorm(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(Xprop_truth(1), Xprop_truth(3), 'kx','MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(Xprop_truth_lowTol(1), Xprop_truth_lowTol(3), 'rx','MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(samplePt(1), samplePt(3), 'o', 'MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(samplePtlt(1), samplePtlt(3), 'o', 'MarkerSize', 15, 'LineWidth', 3)
hold on;
title('X-Z');
xlabel('X');
ylabel('Z');
% legend(legend_string);
hold off;

subplot(2,3,3)
for k = 1:K
    clusterPoints = Xm_cloud_lowTol(idx_lt == k, :);
    scatter(clusterPoints(:,2), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
    clusterPoints = Xm_cloud_highTol(idx_ht == k, :);
    scatter(clusterPoints(:,2), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(K+k));
    hold on;
end
% plot(C_unorm(:,2), C_unorm(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(Xprop_truth(2), Xprop_truth(3), 'kx','MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(Xprop_truth_lowTol(2), Xprop_truth_lowTol(3), 'rx','MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(samplePt(2), samplePt(3), 'o', 'MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(samplePtlt(2), samplePtlt(3), 'o', 'MarkerSize', 15, 'LineWidth', 3)
hold on;
title('Y-Z');
xlabel('Y');
ylabel('Z');
% legend(legend_string);
hold off;

subplot(2,3,4)
for k = 1:K
    clusterPoints = Xm_cloud_lowTol(idx_lt == k, :);
    scatter(clusterPoints(:,4), clusterPoints(:,5), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
    clusterPoints = Xm_cloud_highTol(idx_ht == k, :);
    scatter(clusterPoints(:,4), clusterPoints(:,5), 'filled', 'MarkerFaceColor', colors(K+k));
    hold on;
end
% plot(C_unorm(:,4), C_unorm(:,5), '+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(Xprop_truth(4), Xprop_truth(5), 'kx','MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(Xprop_truth_lowTol(4), Xprop_truth_lowTol(5), 'rx','MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(samplePt(4), samplePt(5), 'o', 'MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(samplePtlt(4), samplePtlt(5), 'o', 'MarkerSize', 15, 'LineWidth', 3)
hold on;
title('Xdot-Ydot');
xlabel('Xdot');
ylabel('Ydot');
% legend(legend_string);
hold off;

subplot(2,3,5)
for k = 1:K
    clusterPoints = Xm_cloud_lowTol(idx_lt == k, :);
    scatter(clusterPoints(:,4), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
    clusterPoints = Xm_cloud_highTol(idx_ht == k, :);
    scatter(clusterPoints(:,4), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(K+k));
    hold on;
end
% plot(C_unorm(:,4), C_unorm(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(Xprop_truth(4), Xprop_truth(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(Xprop_truth_lowTol(4), Xprop_truth_lowTol(6), 'rx','MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(samplePt(4), samplePt(6), 'o', 'MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(samplePtlt(4), samplePtlt(6), 'o', 'MarkerSize', 15, 'LineWidth', 3)
hold on;
title('Xdot-Zdot');
xlabel('Xdot');
ylabel('Zdot');
% legend(legend_string);
hold off;

subplot(2,3,6)
for k = 1:K
    clusterPoints = Xm_cloud_lowTol(idx_lt == k, :);
    scatter(clusterPoints(:,5), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
    clusterPoints = Xm_cloud_highTol(idx_ht == k, :);
    scatter(clusterPoints(:,5), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(K+k));
    hold on;
end
% plot(C_unorm(:,5), C_unorm(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(Xprop_truth(5), Xprop_truth(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(Xprop_truth_lowTol(5), Xprop_truth_lowTol(6), 'rx','MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(samplePt(5), samplePt(6), 'o', 'MarkerSize', 15, 'LineWidth', 3)
hold on;
plot(samplePtlt(5), samplePtlt(6), 'o', 'MarkerSize', 15, 'LineWidth', 3)
hold on;
title('Ydot-Zdot');
xlabel('Ydot');
ylabel('Zdot');
% legend(legend_string);
hold off;

sgt = sprintf('Timestep: %1.5f (Posterior)', full_ts_lowTol(idx_prop,1));
sgtitle(sgt);
saveas(gcf, './Small_Sims/Timestep_0_1B.png', 'png');

% Xprop_truth = [full_ts(idx_prop,2:4), full_vts(idx_prop,2:4)];
fprintf('Truth State (High Tol): \n');
disp(Xprop_truth);
fprintf('Truth State (Low Tol): \n');
disp(Xprop_truth_lowTol);

% Now that we have a GMM representing the prior distribution, we have to
% use a Kalman update for each component: weight, mean, and covariance.

% Posterior variables
wplt = wmP; wpht = wmE;
mult_p = muP_c; muht_p = muht_c;
Plt_p = PP_c; Pht_p = PE_c;

% Comment this out if you wish to use noise.
% noised_obs = partial_ts;

for i = 1:K
    if (idx_meas ~= 0) % i.e. there exists a measurement
        R_vv = [0.01*partial_ts_lowTol(idx_meas,2), 0, 0; 0 7.2722e-6, 0; 0, 0, 7.2722e-6].^2;
        Hxk = linHx(muP_c{i}); % Linearize about prior mean component
        h = @(x) [sqrt(x(1)^2 + x(2)^2 + x(3)^2); atan2(x(2),x(1)); pi/2 - acos(x(3)/sqrt(x(1)^2 + x(2)^2 + x(3)^2))]; % Nonlinear measurement model
        zt = noised_obs(idx_meas,2:4)';
      
        % [mu_p{i}, P_p{i}] = kalmanUpdate(zt, Hxk, R_vv, mu_c{i}', P_c{i}, h);
        [mult_p{i}, Plt_p{i}] = kalmanUpdate(zt, cPoints_pgm{i}, R_vv, muP_c{i}, PP_c{i}, h);
        
        % Weight update
        num = wmP(i)*gaussProb(zt, h(muP_c{i}), Hxk*PP_c{i}*Hxk' + R_vv);
        den = 0;
        for j = 1:K
            Hxk = linHx(muP_c{j});
            den = den + wmP(j)*gaussProb(zt, h(muP_c{j}), Hxk*PP_c{j}*Hxk' + R_vv);
        end
        wplt(i) = num/den;

        R_vv = [0.01*partial_ts_highTol(idx_meas,2), 0, 0; 0 7.2722e-6, 0; 0, 0, 7.2722e-6].^2;
        Hxk = linHx(muht_c{i}); % Linearize about prior mean component
        zt = noised_obs(idx_meas,2:4)';
      
        % [mu_p{i}, P_p{i}] = kalmanUpdate(zt, Hxk, R_vv, mu_c{i}', P_c{i}, h);
        [muht_p{i}, Pht_p{i}] = kalmanUpdate(zt, cPoints_ht{i}, R_vv, muht_c{i}, PE_c{i}, h);
        
        % Weight update
        num = wmE(i)*gaussProb(zt, h(muht_c{i}), Hxk*PE_c{i}*Hxk' + R_vv);
        den = 0;
        for j = 1:K
            Hxk = linHx(muht_c{j});
            den = den + wmE(j)*gaussProb(zt, h(muht_c{j}), Hxk*PE_c{j}*Hxk' + R_vv);
        end
        wpht(i) = num/den;
    else
        wplt(i) = wmP(i); wpht(i) = wmE(i);
        mult_p{i} = muP_c{i}; muht_p{i} = muht_c{i};
        Plt_p{i} = PP_c{i}; Pht_p{i} = PE_c{i};
    end
end

Xplt_cloud = Xm_cloud_lowTol; Xpht_cloud = Xm_cloud_highTol;
clt_id = zeros(length(Xplt_cloud(:,1)),1); cht_id = clt_id;

if (idx_meas ~= 0)
    for i = 1:L
        [Xplt_cloud(i,:), clt_id(i)] = drawFrom(wplt, mult_p, Plt_p); 
        [Xpht_cloud(i,:), cht_id(i)] = drawFrom(wpht, muht_p, Pht_p); 
    end
end

% Plot the results

% Plot planar projections
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,3,1)
for k = 1:K
    clusterPoints = Xplt_cloud(clt_id == k, :);
    scatter(clusterPoints(:,1), clusterPoints(:,2), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
    clusterPoints = Xpht_cloud(cht_id == k, :);
    scatter(clusterPoints(:,1), clusterPoints(:,2), 'filled', 'MarkerFaceColor', colors(K+k));
    hold on;
end
% hold on;
% plot(mu_pExp(:,1), mu_pExp(:,2), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(Xprop_truth(1), Xprop_truth(2), 'kx','MarkerSize', 20, 'LineWidth', 3)
hold on;
plot(Xprop_truth_lowTol(1), Xprop_truth_lowTol(2), 'rx','MarkerSize', 20, 'LineWidth', 3)
hold on;
title('X-Y');
xlabel('X');
ylabel('Y');
% legend(legend_string);
hold off;

subplot(2,3,2)
for k = 1:K
    clusterPoints = Xplt_cloud(clt_id == k, :);
    scatter(clusterPoints(:,1), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
    clusterPoints = Xpht_cloud(cht_id == k, :);
    scatter(clusterPoints(:,1), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(K+k));
    hold on;
end
% plot(mu_pExp(:,1), mu_pExp(:,3), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(Xprop_truth(1), Xprop_truth(3), 'kx','MarkerSize', 20, 'LineWidth', 3)
hold on;
plot(Xprop_truth_lowTol(1), Xprop_truth_lowTol(3), 'rx','MarkerSize', 20, 'LineWidth', 3)
hold on;
title('X-Z');
xlabel('X');
ylabel('Z');
% legend(legend_string);
hold off;

subplot(2,3,3)
for k = 1:K
    clusterPoints = Xplt_cloud(clt_id == k, :);
    scatter(clusterPoints(:,2), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
    clusterPoints = Xpht_cloud(cht_id == k, :);
    scatter(clusterPoints(:,2), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(K+k));
    hold on;
end
% plot(mu_pExp(:,2), mu_pExp(:,3), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(Xprop_truth(2), Xprop_truth(3), 'kx','MarkerSize', 20, 'LineWidth', 3)
hold on;
plot(Xprop_truth_lowTol(2), Xprop_truth_lowTol(3), 'rx','MarkerSize', 20, 'LineWidth', 3)
hold on;
title('Y-Z');
xlabel('Y');
ylabel('Z');
% legend(legend_string);
hold off;

subplot(2,3,4)
for k = 1:K
    clusterPoints = Xplt_cloud(clt_id == k, :);
    scatter(clusterPoints(:,4), clusterPoints(:,5), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
    clusterPoints = Xpht_cloud(cht_id == k, :);
    scatter(clusterPoints(:,4), clusterPoints(:,5), 'filled', 'MarkerFaceColor', colors(K+k));
    hold on;
end
% plot(mu_pExp(:,4), mu_pExp(:,5), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(Xprop_truth(4), Xprop_truth(5), 'kx','MarkerSize', 20, 'LineWidth', 3)
hold on;
plot(Xprop_truth_lowTol(4), Xprop_truth_lowTol(5), 'rx','MarkerSize', 20, 'LineWidth', 3)
hold on;
title('Xdot-Ydot');
xlabel('Xdot');
ylabel('Ydot');
% legend(legend_string);
hold off;

subplot(2,3,5)
for k = 1:K
    clusterPoints = Xplt_cloud(clt_id == k, :);
    scatter(clusterPoints(:,4), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
    clusterPoints = Xpht_cloud(cht_id == k, :);
    scatter(clusterPoints(:,4), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(K+k));
    hold on;
end
% plot(mu_pExp(:,4), mu_pExp(:,6), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(Xprop_truth(4), Xprop_truth(6), 'kx','MarkerSize', 20, 'LineWidth', 3)
hold on;
plot(Xprop_truth_lowTol(4), Xprop_truth_lowTol(6), 'rx','MarkerSize', 20, 'LineWidth', 3)
hold on;
title('Xdot-Zdot');
xlabel('Xdot');
ylabel('Zdot');
% legend(legend_string);
hold off;

subplot(2,3,6)
for k = 1:K
    clusterPoints = Xplt_cloud(clt_id == k, :);
    scatter(clusterPoints(:,5), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
    hold on;
    clusterPoints = Xpht_cloud(cht_id == k, :);
    scatter(clusterPoints(:,5), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(K+k));
    hold on;
end
% plot(mu_pExp(:,5), mu_pExp(:,6), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
% hold on;
plot(Xprop_truth(5), Xprop_truth(6), 'kx','MarkerSize', 20, 'LineWidth', 3)
hold on;
plot(Xprop_truth_lowTol(5), Xprop_truth_lowTol(6), 'rx','MarkerSize', 20, 'LineWidth', 3)
title('Ydot-Zdot');
xlabel('Ydot');
ylabel('Zdot');
% legend(legend_string);
hold off;

sgt = sprintf('Timestep: %1.5f (Posterior)', tpr);
sgtitle(sgt);
saveas(gcf, './Small_Sims/Timestep_0_2B.png', 'png');

% At this point, we have shown a PGM-I propagation and update step. The
% next step is to utilize this PGM-I update across all time steps during
% which the target is within our sensor FOV and see how the particle clouds
% (i.e. GM components) evolve over time. If we're lucky, we should see that
% the GMM tracks the truth over the interval.

% Find and set the start and end times to simulation
% [idx_meas, c_meas] = find(abs(hdR(:,1) - tpr) < 1e-10);
% interval = hdR(idx_meas,c_meas) - hdR(idx_meas-1,c_meas);

% l_filt = 2; % Number of total time steps that the filter is run (i.e. filter length)
% t_end = tpr + l_filt*interval;
% t_end = hdR(idx_meas + (l_filt - 1), c_meas); % Add small epsilon to avoid roundoff

[idx_crit, ~] = find(abs(noised_obs(:,1) - cTimes(2)) < 1e-10); % Find the index of the last observation before the half-day gap
t_end = noised_obs(idx_crit+5,1); % First observation of new pass + one more time step
% t_end = cTimes(2);
l_filt = int32((t_end - tpr)/interval + 1); % Filter time length

% Find time step at which we can observe the target again
% [idx_lastObs, ~] = find(abs(noised_obs(:,1) - hdR(end,1)) < 1e-10); % Find the index of the last observation before the half-day gap
% t_end = noised_obs(idx_lastObs+1, 1) - 1*interval; % Final time step before we can re-incorporate observations
% l_filt = int32((noised_obs(idx_lastObs+1,1) - 1*interval - tpr)/interval + 1); % Filter time length

tau = 0;
for to = tpr:interval:(t_end-1e-11) % Looping over the times of observation for easier propagation

    % Propagation Step
    if (idx_meas ~= 0)
        Xplt_cloud = zeros(Lp,6); Xpht_cloud = Xplt_cloud;
        for i = 1:Lp
            [Xplt_cloud(i,:), ~] = drawFrom(wplt, mult_p, Plt_p); 
            [Xpht_cloud(i,:), ~] = drawFrom(wpht, muht_p, Pht_p);
        end 
    end

    %{
    Xplt_cloud = zeros(Lp,6); Xpht_cloud = Xplt_cloud;
    for i = 1:Lp
        [Xplt_cloud(i,:), ~] = drawFrom(wplt, mult_p, Plt_p); 
        [Xpht_cloud(i,:), ~] = drawFrom(wpht, muht_p, Pht_p);
    end
    %}

    Xm_cloud_lowTol = propagate(Xplt_cloud, to, interval, 1);
    Xm_cloud_highTol = propagate(Xpht_cloud, to, interval, 1);
    samplePt = propagate(samplePt, to, interval, 1);
    samplePtlt = propagate(samplePtlt, to, interval, 2);

    % Verification Step
    tpr = to + interval; % Time stamp of the prior means, weights, and covariances
    [idx_meas, ~] = find(abs(noised_obs(:,1) - tpr) < 1e-10); % Find row with time
    tau = tau + 1;

    if (idx_meas ~= 0)  
        % Split propagated cloud into position and velocity data before
        % normalization.
        if(abs(to - cTimes(2)) < 1e-10 || to < cTimes(2))
            K = Kn;
        else
            K = 1;
        end
        % K = Kn;

        rc = Xm_cloud_lowTol(:,1:3);
        vc = Xm_cloud_lowTol(:,4:6);
    
        mean_rc = mean(rc, 1);
        mean_vc = mean(vc, 1);
    
        std_rc = std(rc,0,1);
        std_vc = std(vc,0,1);
    
        norm_rc = (rc - mean_rc)./norm(std_rc); % Normalizing the position 
        norm_vc = (vc - mean_vc)./norm(std_vc); % Normalizing the velocity
    
        Xmlt_norm = [norm_rc, norm_vc];
        
        rc = Xm_cloud_highTol(:,1:3);
        vc = Xm_cloud_highTol(:,4:6);
    
        mean_rc = mean(rc, 1);
        mean_vc = mean(vc, 1);
    
        std_rc = std(rc,0,1);
        std_vc = std(vc,0,1);
    
        norm_rc = (rc - mean_rc)./norm(std_rc); % Normalizing the position 
        norm_vc = (vc - mean_vc)./norm(std_vc); % Normalizing the velocity
    
        Xmht_norm = [norm_rc, norm_vc];
    
        % Verification Step
        [idx_meas, ~] = find(abs(noised_obs(:,1) - tpr) < 1e-10); % Find row with time

        fprintf("Timestamp: %1.5f\n", tpr);
        
        % Cluster using K-means clustering algorithm
        [idx_lt, ~] = kmeans(Xmlt_norm, K);
        [idx_ht, ~] = kmeans(Xmht_norm, K);

        cPoints_pgm = cell(K, 1); cPoints_ht = cPoints_pgm;
        muP_c = cell(K, 1); mult_p = muP_c;
        PP_c = cell(K, 1); Plt_p = PP_c;
        wmP = zeros(K, 1); wplt = wmP;

        muht_c = cell(K, 1); muht_p = muht_c;
        PE_c = cell(K, 1); Pht_p = PE_c;
        wmE = zeros(K, 1); wpht = wmE;

        % Calculate covariances and weights for each cluster
        for k = 1:K
            cluster_points = Xm_cloud_lowTol(idx_lt == k, :); 
            cPoints_pgm{k} = cluster_points; 
            muP_c{k} = mean(cluster_points, 1); % Cell of GMM means 
            PP_c{k} = cov(cluster_points); % Cell of GMM covariances
            wmP(k) = size(cluster_points, 1) / size(Xm_cloud_lowTol, 1); % Vector of (prior) weights

            cluster_points = Xm_cloud_highTol(idx_ht == k, :); 
            cPoints_ht{k} = cluster_points; 
            muht_c{k} = mean(cluster_points, 1); % Cell of GMM means 
            PE_c{k} = cov(cluster_points); % Cell of GMM covariances
            wmE(k) = size(cluster_points, 1) / size(Xm_cloud_highTol, 1); % Vector of (prior) weights
        end

        [idx_trth, ~] = find(abs(full_ts_highTol(:,1) - tpr) < 1e-10);
        Xprop_truth = [full_ts_highTol(idx_trth,2:4), full_vts_highTol(idx_trth,2:4)]';

        [idx_trth, ~] = find(abs(full_ts_lowTol(:,1) - tpr) < 1e-10);
        Xprop_truth_lowTol = [full_ts_lowTol(idx_trth,2:4), full_vts_lowTol(idx_trth,2:4)]';

        zc = noised_obs(idx_meas,2:4)'; % Presumption: An observation occurs at this time step
        xto = zc(1)*cos(zc(2))*cos(zc(3)); 
        yto = zc(1)*sin(zc(2))*cos(zc(3)); 
        zto = zc(1)*sin(zc(3)); 
        rto = [xto, yto, zto];

        legend_string = {};
        for k = 1:K
            R_vv = [0.05*partial_ts_lowTol(idx_meas,2), 0, 0; 0 7.2722e-6, 0; 0, 0, 7.2722e-6].^2;
            Hxk = linHx(muP_c{k}); % Linearize about prior mean component
            legend_string{k} = sprintf('Distribution %i',k);
            legend_string{K+k} = sprintf('Distribution %i',k);
        end
        % legend_string{K+1} = "Centroids";
        legend_string{2*K+1} = "Truth (Low Tol)";
        legend_string{2*K+2} = "Truth (High Tol)";
  
        mult_mat = cell2mat(muP_c); muht_mat = cell2mat(muP_c);
        Plt_mat = cat(3, PP_c{:}); Pht_mat = cat(3, PE_c{:});

        f = figure('visible','off','Position', get(0,'ScreenSize'));
        f.WindowState = 'maximized';

        subplot(2,3,1)
        plot_dims = [1,2];
        mult_marg = mult_mat(:, plot_dims); muht_marg = muht_mat(:, plot_dims);
        Plt_marg = Plt_mat(plot_dims, plot_dims, :); Pht_marg = Pht_mat(plot_dims, plot_dims, :);
        
        for k = 1:K
            [X1, X2] = meshgrid(linspace(min(Xm_cloud_lowTol(:,plot_dims(1))), max(Xm_cloud_lowTol(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud_lowTol(:,plot_dims(2))), max(Xm_cloud_lowTol(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
            
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mult_marg(k,:)) * Plt_marg(:,:,k)^(-1) * (X_grid(i,:) - mult_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(Plt_marg(:,:,k))));
            contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

            contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(1,:));
            hold on;

            [X1, X2] = meshgrid(linspace(min(Xm_cloud_highTol(:,plot_dims(1))), max(Xm_cloud_highTol(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud_highTol(:,plot_dims(2))), max(Xm_cloud_highTol(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
            
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - muht_marg(k,:)) * Pht_marg(:,:,k)^(-1) * (X_grid(i,:) - muht_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(Pht_marg(:,:,k))));
            contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

            contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(2,:));
            hold on;
        end 
        plot(Xprop_truth(1), Xprop_truth(2), 'x','MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(Xprop_truth_lowTol(1), Xprop_truth_lowTol(2), 'x','MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(samplePt(1), samplePt(2), 'o', 'MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(samplePtlt(1), samplePtlt(2), 'o', 'MarkerSize', 15, 'LineWidth', 3);
        title('X-Y');
        xlabel('X');
        ylabel('Y');
        % legend(legend_string);
        hold off;

        subplot(2,3,2)
        plot_dims = [1,3];
        mult_marg = mult_mat(:, plot_dims); muht_marg = muht_mat(:, plot_dims);
        Plt_marg = Plt_mat(plot_dims, plot_dims, :); Pht_marg = Pht_mat(plot_dims, plot_dims, :);
        
        for k = 1:K
            [X1, X2] = meshgrid(linspace(min(Xm_cloud_lowTol(:,plot_dims(1))), max(Xm_cloud_lowTol(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud_lowTol(:,plot_dims(2))), max(Xm_cloud_lowTol(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
            
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mult_marg(k,:)) * Plt_marg(:,:,k)^(-1) * (X_grid(i,:) - mult_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(Plt_marg(:,:,k))));
            contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

            contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(1,:));
            hold on;

            [X1, X2] = meshgrid(linspace(min(Xm_cloud_highTol(:,plot_dims(1))), max(Xm_cloud_highTol(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud_highTol(:,plot_dims(2))), max(Xm_cloud_highTol(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
            
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - muht_marg(k,:)) * Pht_marg(:,:,k)^(-1) * (X_grid(i,:) - muht_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(Pht_marg(:,:,k))));
            contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

            contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(2,:));
            hold on;
        end 
        plot(Xprop_truth(1), Xprop_truth(3), 'kx','MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(Xprop_truth_lowTol(1), Xprop_truth_lowTol(3), 'x','MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(samplePt(1), samplePt(3), 'o', 'MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(samplePtlt(1), samplePtlt(3), 'o', 'MarkerSize', 15, 'LineWidth', 3);
        title('X-Z');
        xlabel('X');
        ylabel('Z');
        % legend(legend_string);
        hold off;

        subplot(2,3,3)
        plot_dims = [2,3];
        mult_marg = mult_mat(:, plot_dims); muht_marg = muht_mat(:, plot_dims);
        Plt_marg = Plt_mat(plot_dims, plot_dims, :); Pht_marg = Pht_mat(plot_dims, plot_dims, :);
        
        for k = 1:K
            [X1, X2] = meshgrid(linspace(min(Xm_cloud_lowTol(:,plot_dims(1))), max(Xm_cloud_lowTol(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud_lowTol(:,plot_dims(2))), max(Xm_cloud_lowTol(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
            
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mult_marg(k,:)) * Plt_marg(:,:,k)^(-1) * (X_grid(i,:) - mult_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(Plt_marg(:,:,k))));
            contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

            contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(1,:));
            hold on;

            [X1, X2] = meshgrid(linspace(min(Xm_cloud_highTol(:,plot_dims(1))), max(Xm_cloud_highTol(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud_highTol(:,plot_dims(2))), max(Xm_cloud_highTol(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
            
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - muht_marg(k,:)) * Pht_marg(:,:,k)^(-1) * (X_grid(i,:) - muht_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(Pht_marg(:,:,k))));
            contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

            contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(2,:));
            hold on;
        end 
        plot(Xprop_truth(2), Xprop_truth(3), 'x','MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(Xprop_truth_lowTol(2), Xprop_truth_lowTol(3), 'x','MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(samplePt(2), samplePt(3), 'o', 'MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(samplePtlt(2), samplePtlt(3), 'o', 'MarkerSize', 15, 'LineWidth', 3);
        title('X-Z');
        xlabel('X');
        ylabel('Z');
        % legend(legend_string);
        hold off;

        subplot(2,3,4)
        plot_dims = [4,5];
        mult_marg = mult_mat(:, plot_dims); muht_marg = muht_mat(:, plot_dims);
        Plt_marg = Plt_mat(plot_dims, plot_dims, :); Pht_marg = Pht_mat(plot_dims, plot_dims, :);
        
        for k = 1:K
            [X1, X2] = meshgrid(linspace(min(Xm_cloud_lowTol(:,plot_dims(1))), max(Xm_cloud_lowTol(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud_lowTol(:,plot_dims(2))), max(Xm_cloud_lowTol(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
            
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mult_marg(k,:)) * Plt_marg(:,:,k)^(-1) * (X_grid(i,:) - mult_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(Plt_marg(:,:,k))));
            contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

            contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(1,:));
            hold on;

            [X1, X2] = meshgrid(linspace(min(Xm_cloud_highTol(:,plot_dims(1))), max(Xm_cloud_highTol(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud_highTol(:,plot_dims(2))), max(Xm_cloud_highTol(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
            
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - muht_marg(k,:)) * Pht_marg(:,:,k)^(-1) * (X_grid(i,:) - muht_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(Pht_marg(:,:,k))));
            contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

            contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(2,:));
            hold on;
        end 
        plot(Xprop_truth(4), Xprop_truth(5), 'x','MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(Xprop_truth_lowTol(4), Xprop_truth_lowTol(5), 'x','MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(samplePt(4), samplePt(5), 'o', 'MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(samplePtlt(4), samplePtlt(5), 'o', 'MarkerSize', 15, 'LineWidth', 3);
        title('Xdot-Ydot');
        xlabel('Xdot');
        ylabel('Ydot');
        % legend(legend_string);
        hold off;

        subplot(2,3,5)
        plot_dims = [4,6];
        mult_marg = mult_mat(:, plot_dims); muht_marg = muht_mat(:, plot_dims);
        Plt_marg = Plt_mat(plot_dims, plot_dims, :); Pht_marg = Pht_mat(plot_dims, plot_dims, :);
        
        for k = 1:K
            [X1, X2] = meshgrid(linspace(min(Xm_cloud_lowTol(:,plot_dims(1))), max(Xm_cloud_lowTol(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud_lowTol(:,plot_dims(2))), max(Xm_cloud_lowTol(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
            
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mult_marg(k,:)) * Plt_marg(:,:,k)^(-1) * (X_grid(i,:) - mult_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(Plt_marg(:,:,k))));
            contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

            contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(1,:));
            hold on;

            [X1, X2] = meshgrid(linspace(min(Xm_cloud_highTol(:,plot_dims(1))), max(Xm_cloud_highTol(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud_highTol(:,plot_dims(2))), max(Xm_cloud_highTol(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
            
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - muht_marg(k,:)) * Pht_marg(:,:,k)^(-1) * (X_grid(i,:) - muht_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(Pht_marg(:,:,k))));
            contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

            contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(2,:));
            hold on;
        end 
        plot(Xprop_truth(4), Xprop_truth(6), 'x','MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(Xprop_truth_lowTol(4), Xprop_truth_lowTol(6), 'x','MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(samplePt(4), samplePt(6), 'o', 'MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(samplePtlt(4), samplePtlt(6), 'o', 'MarkerSize', 15, 'LineWidth', 3);
        title('Xdot-Zdot');
        xlabel('Xdot');
        ylabel('Zdot');
        % legend(legend_string);
        hold off;

        subplot(2,3,6)
        plot_dims = [5,6];
        mult_marg = mult_mat(:, plot_dims); muht_marg = muht_mat(:, plot_dims);
        Plt_marg = Plt_mat(plot_dims, plot_dims, :); Pht_marg = Pht_mat(plot_dims, plot_dims, :);
        
        for k = 1:K
            [X1, X2] = meshgrid(linspace(min(Xm_cloud_lowTol(:,plot_dims(1))), max(Xm_cloud_lowTol(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud_lowTol(:,plot_dims(2))), max(Xm_cloud_lowTol(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
            
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - mult_marg(k,:)) * Plt_marg(:,:,k)^(-1) * (X_grid(i,:) - mult_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(Plt_marg(:,:,k))));
            contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

            contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(1,:));
            hold on;

            [X1, X2] = meshgrid(linspace(min(Xm_cloud_highTol(:,plot_dims(1))), max(Xm_cloud_highTol(:,plot_dims(1))), 100), ...
                        linspace(min(Xm_cloud_highTol(:,plot_dims(2))), max(Xm_cloud_highTol(:,plot_dims(2))), 100));
            X_grid = [X1(:) X2(:)];
            
            Z = zeros(size(X1));
            for i = 1:size(X_grid, 1)
                Z(i) = exp(-0.5 * (X_grid(i,:) - muht_marg(k,:)) * Pht_marg(:,:,k)^(-1) * (X_grid(i,:) - muht_marg(k,:))');
            end
            Z = reshape(Z, size(X1));
            Z = Z/(2*pi*sqrt(det(Pht_marg(:,:,k))));
            contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

            contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(2,:));
            hold on;
        end 
        plot(Xprop_truth(5), Xprop_truth(6), 'x','MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(Xprop_truth_lowTol(5), Xprop_truth_lowTol(6), 'x','MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(samplePt(5), samplePt(6), 'o', 'MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(samplePtlt(5), samplePtlt(6), 'o', 'MarkerSize', 15, 'LineWidth', 3);
        title('Ydot-Zdot');
        xlabel('Ydot');
        ylabel('Zdot');
        % legend(legend_string);
        hold off;

        sgt = sprintf('Timestep: %1.5f (Prior)', tpr);
        sgtitle(sgt);

        sg = sprintf('./Small_Sims/Timestep_%i_1A.png', tau);
        saveas(f, sg, 'png');
        close(f);
        

        f = figure('visible','off','Position', get(0,'ScreenSize'));
        f.WindowState = 'maximized';

        legend_string = {};
        for k = 1:K
            % legend_string{k} = sprintf('Contour %i', k);
            legend_string{k} = sprintf('\\omega = %1.4f', wmP(k));
            legend_string{K+k} = sprintf('\\omega = %1.4f', wmE(k));
        end
        % legend_string{K+1} = "Centroids";
        legend_string{2*K+1} = "Truth (High Tol)";
        legend_string{2*K+2} = "Truth (Low Tol)";

        subplot(2,3,1)
        % gscatter(Xm_cloud(:,1), Xm_cloud(:,2), idx);
        % hold on;
        % plot(mu_mExp(:,1), mu_mExp(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
        % hold on;

        for k = 1:K
            clusterPoints = Xm_cloud_lowTol(idx_lt == k, :);
            scatter(clusterPoints(:,1), clusterPoints(:,2), 'filled', 'MarkerFaceColor', colors(k));
            hold on;
            clusterPoints = Xm_cloud_highTol(idx_ht == k, :);
            scatter(clusterPoints(:,1), clusterPoints(:,2), 'filled', 'MarkerFaceColor', colors(K+k));
            hold on;
        end
        plot(Xprop_truth(1), Xprop_truth(2), 'x','MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(Xprop_truth_lowTol(1), Xprop_truth_lowTol(2), 'x','MarkerSize', 15, 'LineWidth', 3);
        % hold on;
        % plot(rto(1), rto(2), 'o', 'MarkerSize', 10, 'LineWidth', 3);
        title('X-Y');
        xlabel('X');
        ylabel('Y');
        % legend(legend_string);
        hold off;
        
        subplot(2,3,2)
        % gscatter(Xm_cloud(:,1), Xm_cloud(:,3), idx);
        % hold on;
        % plot(mu_mExp(:,1), mu_mExp(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
        % hold on;
        
        for k = 1:K
            clusterPoints = Xm_cloud_lowTol(idx_lt == k, :);
            scatter(clusterPoints(:,1), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(1));
            hold on;
            clusterPoints = Xm_cloud_highTol(idx_ht == k, :);
            scatter(clusterPoints(:,1), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(2));
            hold on;
        end
        plot(Xprop_truth(1), Xprop_truth(3), 'x','MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(Xprop_truth_lowTol(1), Xprop_truth_lowTol(3), 'x','MarkerSize', 15, 'LineWidth', 3);
        % hold on;
        % plot(rto(1), rto(3), 'o', 'MarkerSize', 10, 'LineWidth', 3);
        title('X-Z');
        xlabel('X');
        ylabel('Z');
        % legend(legend_string);
        hold off;
        
        subplot(2,3,3)
        % gscatter(Xm_cloud(:,2), Xm_cloud(:,3), idx);
        % hold on;
        % plot(mu_mExp(:,2), mu_mExp(:,3), '+', 'MarkerSize', 10, 'LineWidth', 3);
        % hold on;
        
        for k = 1:K
            clusterPoints = Xm_cloud_lowTol(idx_lt == k, :);
            scatter(clusterPoints(:,2), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
            hold on;
            clusterPoints = Xm_cloud_highTol(idx_ht == k, :);
            scatter(clusterPoints(:,2), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(K+k));
            hold on;
        end
        plot(Xprop_truth(2), Xprop_truth(3), 'x','MarkerSize', 15, 'LineWidth', 3);
        hold on;
        plot(Xprop_truth_lowTol(2), Xprop_truth_lowTol(3), 'x','MarkerSize', 15, 'LineWidth', 3);
        % hold on;
        % plot(rto(2), rto(3), 'o', 'MarkerSize', 10, 'LineWidth', 3);
        title('Y-Z');
        xlabel('Y');
        ylabel('Z');
        % legend(legend_string);
        hold off;
        
        subplot(2,3,4)
        % gscatter(Xm_cloud(:,4), Xm_cloud(:,5), idx);
        % hold on;
        % plot(mu_mExp(:,4), mu_mExp(:,5), '+', 'MarkerSize', 10, 'LineWidth', 3);
        % hold on;
        
        for k = 1:K
            clusterPoints = Xm_cloud_lowTol(idx_lt == k, :);
            scatter(clusterPoints(:,4), clusterPoints(:,5), 'filled', 'MarkerFaceColor', colors(k));
            hold on;
            clusterPoints = Xm_cloud_highTol(idx_ht == k, :);
            scatter(clusterPoints(:,4), clusterPoints(:,5), 'filled', 'MarkerFaceColor', colors(K+k));
            hold on;
        end
        plot(Xprop_truth(4), Xprop_truth(5), 'x','MarkerSize', 15, 'LineWidth', 3)
        hold on;
        plot(Xprop_truth_lowTol(4), Xprop_truth_lowTol(5), 'x','MarkerSize', 15, 'LineWidth', 3);
        title('Xdot-Ydot');
        xlabel('Xdot');
        ylabel('Ydot');
        % legend(legend_string);
        hold off;
        
        subplot(2,3,5)
        % gscatter(Xm_cloud(:,4), Xm_cloud(:,6), idx);
        % hold on;
        % plot(mu_mExp(:,4), mu_mExp(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3);
        % hold on;
        
        for k = 1:K
            clusterPoints = Xm_cloud_lowTol(idx_lt == k, :);
            scatter(clusterPoints(:,4), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
            hold on;
            clusterPoints = Xm_cloud_highTol(idx_ht == k, :);
            scatter(clusterPoints(:,4), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(K+k));
            hold on;
        end
        plot(Xprop_truth(4), Xprop_truth(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
        hold on;
        plot(Xprop_truth_lowTol(4), Xprop_truth_lowTol(6), 'x','MarkerSize', 15, 'LineWidth', 3);
        title('Xdot-Zdot');
        xlabel('Xdot');
        ylabel('Zdot');
        % legend(legend_string);
        hold off;
        
        subplot(2,3,6)
        % gscatter(Xm_cloud(:,5), Xm_cloud(:,6), idx);
        % hold on;
        % plot(mu_mExp(:,5), mu_mExp(:,6), '+', 'MarkerSize', 10, 'LineWidth', 3);
        % hold on;
       
        for k = 1:K
            clusterPoints = Xm_cloud_lowTol(idx_lt == k, :);
            scatter(clusterPoints(:,5), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
            hold on;
            clusterPoints = Xm_cloud_highTol(idx_ht == k, :);
            scatter(clusterPoints(:,5), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(K+k));
            hold on;
        end
        plot(Xprop_truth(5), Xprop_truth(6), 'kx','MarkerSize', 15, 'LineWidth', 3)
        hold on;
        plot(Xprop_truth_lowTol(5), Xprop_truth_lowTol(6), 'x','MarkerSize', 15, 'LineWidth', 3);
        title('Ydot-Zdot');
        xlabel('Ydot');
        ylabel('Zdot');
        % legend(legend_string);
        hold off;

        sgt = sprintf('Timestep: %1.5f (Prior)', tpr);
        sgtitle(sgt);

        sg = sprintf('./Small_Sims/Timestep_%i_1B.png', tau);
        saveas(f, sg, 'png');
        close(f);

        % Update Step
        for i = 1:K
            R_vv = [0.05*partial_ts_lowTol(idx_meas,2), 0, 0; 0 7.2722e-6, 0; 0, 0, 7.2722e-6].^2;
            Hxk = linHx(muP_c{i}); % Linearize about prior mean component
            h = @(x) [sqrt(x(1)^2 + x(2)^2 + x(3)^2); atan2(x(2),x(1)); pi/2 - acos(x(3)/sqrt(x(1)^2 + x(2)^2 + x(3)^2))]; % Nonlinear measurement model
            zt = noised_obs(idx_meas,2:4)';
    
            % [mu_p{i}, P_p{i}] = kalmanUpdate(zt, Hxk, R_vv, mu_c{i}', P_c{i}, h);
            [mult_p{i}, Plt_p{i}] = kalmanUpdate(zt, cPoints_pgm{i}, R_vv, muP_c{i}, PP_c{i}, h);
            mult_p{i} = mult_p{i}';
            Plt_p{i} = (Plt_p{i} + Plt_p{i}')/2;
            % P_p{i} = P_p{i} + 1e-10*P_p{i}*eye(length(mu_p{i})); % Add small quantity to avoid the lack of a positive definite matrix 
        
            % Weight update
            num = wmP(i)*gaussProb(zt, h(muP_c{i}), Hxk*PP_c{i}*Hxk' + R_vv);
            den = 0;
            for j = 1:K
                Hxk = linHx(muP_c{j});
                den = den + wmP(j)*gaussProb(zt, h(muP_c{j}), Hxk*PP_c{j}*Hxk' + R_vv);
            end
            wplt(i) = num/den;

            R_vv = [0.05*partial_ts_highTol(idx_meas,2), 0, 0; 0 7.2722e-6, 0; 0, 0, 7.2722e-6].^2;
            Hxk = linHx(muht_c{i}); % Linearize about prior mean component
            zt = noised_obs(idx_meas,2:4)';

            % [mu_p{i}, P_p{i}] = kalmanUpdate(zt, Hxk, R_vv, mu_c{i}', P_c{i}, h);
            [muht_p{i}, Pht_p{i}] = kalmanUpdate(zt, cPoints_ht{i}, R_vv, muht_c{i}, PE_c{i}, h);
            muht_p{i} = muht_p{i}';
            Pht_p{i} = (Pht_p{i} + Pht_p{i}')/2;
            % P_p{i} = P_p{i} + 1e-10*P_p{i}*eye(length(mu_p{i})); % Add small quantity to avoid the lack of a positive definite matrix 
        
            % Weight update
            num = wmE(i)*gaussProb(zt, h(muht_c{i}), Hxk*PE_c{i}*Hxk' + R_vv);
            den = 0;
            for j = 1:K
                Hxk = linHx(muht_c{j});
                den = den + wmE(j)*gaussProb(zt, h(muht_c{j}), Hxk*PE_c{j}*Hxk' + R_vv);
            end
            wpht(i) = num/den;
        end

    else
        fprintf("Timestamp: %1.5f\n", tpr);
        K = 1;

        mult_p = cell(1, 1); muht_p = cell(1, 1);
        Plt_p = cell(1, 1); Pht_p = cell(1, 1);
        wmP = zeros(1, 1); wmE = zeros(1, 1);

        Xplt_cloud = Xm_cloud_lowTol; Xpht_cloud = Xm_cloud_highTol;
        wplt = [1]; wpht = [1];
        mult_p{1} = mean(Xplt_cloud); muht_p{1} = mean(Xpht_cloud);
        Plt_p{1} = cov(Xplt_cloud); Pht_p{1} = cov(Xpht_cloud);
    end
    
    if (idx_meas ~= 0)
        Xplt_cloud = zeros(L, length(Xprop_truth)); Xpht_cloud = Xplt_cloud;
        clt_id = zeros(L,1); cht_id = zeros(L,1);
        for i = 1:L
            [Xplt_cloud(i,:), clt_id(i)] = drawFrom(wplt, mult_p, Plt_p); 
            [Xpht_cloud(i,:), cht_id(i)] = drawFrom(wpht, muht_p, Pht_p);
        end
    else
        Xplt_cloud = Xm_cloud_lowTol; Xpht_cloud = Xm_cloud_highTol;
        clt_id = ones(L,1); cht_id = ones(L,1);
    end

    [idx_trth, ~] = find(abs(full_ts_highTol(:,1) - tpr) < 1e-10);
    Xprop_truth = [full_ts_highTol(idx_trth,2:4), full_vts_highTol(idx_trth,2:4)];
    [idx_trth, ~] = find(abs(full_ts_lowTol(:,1) - tpr) < 1e-10);
    Xprop_truth_lowTol = [full_ts_lowTol(idx_trth,2:4), full_vts_lowTol(idx_trth,2:4)];
    
    %{
    if(idx_meas ~= 0)
        K = Kn;
    else
        K = 1;
    end
    %}

    legend_string = {};
    for k = 1:K
        legend_string{k} = sprintf('Contour %i', k);
        legend_string{K+k} = sprintf('Contour %i', k);
    end
    % legend_string{K+1} = "Centroids";
    legend_string{2*K+1} = "Truth (High Tol)";
    legend_string{2*K+2} = "Truth (Low Tol)";

    mult_mat = cell2mat(mult_p); muht_mat = cell2mat(muht_p);
    Plt_mat = cat(3, Plt_p{:}); Pht_mat = cat(3, Pht_p{:});

    f = figure('visible','off','Position', get(0,'ScreenSize'));
    f.WindowState = 'maximized';

    subplot(2,3,1)
    plot_dims = [1,2];
    mult_marg = mult_mat(:, plot_dims); muht_marg = muht_mat(:, plot_dims);
    Plt_marg = Plt_mat(plot_dims, plot_dims, :); Pht_marg = Pht_mat(plot_dims, plot_dims, :);

    for k = 1:K
        [X1, X2] = meshgrid(linspace(min(Xplt_cloud(:,plot_dims(1))), max(Xplt_cloud(:,plot_dims(1))), 100), ...
                    linspace(min(Xplt_cloud(:,plot_dims(2))), max(Xplt_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
        
        Z = zeros(size(X1));
        for i = 1:size(X_grid, 1)
            Z(i) = exp(-0.5 * (X_grid(i,:) - mult_marg(k,:)) * Plt_marg(:,:,k)^(-1) * (X_grid(i,:) - mult_marg(k,:))');
        end
        Z = reshape(Z, size(X1));
        Z = Z/(2*pi*sqrt(det(Plt_marg(:,:,k))));
        contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

        contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(1,:));
        hold on;

        [X1, X2] = meshgrid(linspace(min(Xpht_cloud(:,plot_dims(1))), max(Xpht_cloud(:,plot_dims(1))), 100), ...
                    linspace(min(Xpht_cloud(:,plot_dims(2))), max(Xpht_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
        
        Z = zeros(size(X1));
        for i = 1:size(X_grid, 1)
            Z(i) = exp(-0.5 * (X_grid(i,:) - muht_marg(k,:)) * Pht_marg(:,:,k)^(-1) * (X_grid(i,:) - muht_marg(k,:))');
        end
        Z = reshape(Z, size(X1));
        Z = Z/(2*pi*sqrt(det(Pht_marg(:,:,k))));
        contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

        contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(2,:));
        hold on;
    end
    plot(Xprop_truth(1), Xprop_truth(2), 'kx','MarkerSize', 20, 'LineWidth', 3)
    hold on;
    plot(Xprop_truth_lowTol(1), Xprop_truth_lowTol(2), 'rx','MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePt(1), samplePt(2), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePtlt(1), samplePtlt(2), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    title('X-Y');
    xlabel('X');
    ylabel('Y');
    % legend(legend_string);
    hold off;

    subplot(2,3,2)
    plot_dims = [1,3];
    mult_marg = mult_mat(:, plot_dims); muht_marg = muht_mat(:, plot_dims);
    Plt_marg = Plt_mat(plot_dims, plot_dims, :); Pht_marg = Pht_mat(plot_dims, plot_dims, :);

    for k = 1:K
        [X1, X2] = meshgrid(linspace(min(Xplt_cloud(:,plot_dims(1))), max(Xplt_cloud(:,plot_dims(1))), 100), ...
                    linspace(min(Xplt_cloud(:,plot_dims(2))), max(Xplt_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
        
        Z = zeros(size(X1));
        for i = 1:size(X_grid, 1)
            Z(i) = exp(-0.5 * (X_grid(i,:) - mult_marg(k,:)) * Plt_marg(:,:,k)^(-1) * (X_grid(i,:) - mult_marg(k,:))');
        end
        Z = reshape(Z, size(X1));
        Z = Z/(2*pi*sqrt(det(Plt_marg(:,:,k))));
        contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

        contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(1,:));
        hold on;

        [X1, X2] = meshgrid(linspace(min(Xpht_cloud(:,plot_dims(1))), max(Xpht_cloud(:,plot_dims(1))), 100), ...
                    linspace(min(Xpht_cloud(:,plot_dims(2))), max(Xpht_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
        
        Z = zeros(size(X1));
        for i = 1:size(X_grid, 1)
            Z(i) = exp(-0.5 * (X_grid(i,:) - muht_marg(k,:)) * Pht_marg(:,:,k)^(-1) * (X_grid(i,:) - muht_marg(k,:))');
        end
        Z = reshape(Z, size(X1));
        Z = Z/(2*pi*sqrt(det(Pht_marg(:,:,k))));
        contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

        contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(2,:));
        hold on;
    end
    plot(Xprop_truth(1), Xprop_truth(3), 'kx','MarkerSize', 20, 'LineWidth', 3)
    hold on;
    plot(Xprop_truth_lowTol(1), Xprop_truth_lowTol(3), 'rx','MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePt(1), samplePt(3), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePtlt(1), samplePtlt(3), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    title('X-Z');
    xlabel('X');
    ylabel('Z');
    % legend(legend_string);
    hold off;

    subplot(2,3,3)
    plot_dims = [2,3];
    mult_marg = mult_mat(:, plot_dims); muht_marg = muht_mat(:, plot_dims);
    Plt_marg = Plt_mat(plot_dims, plot_dims, :); Pht_marg = Pht_mat(plot_dims, plot_dims, :);

    for k = 1:K
        [X1, X2] = meshgrid(linspace(min(Xplt_cloud(:,plot_dims(1))), max(Xplt_cloud(:,plot_dims(1))), 100), ...
                    linspace(min(Xplt_cloud(:,plot_dims(2))), max(Xplt_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
        
        Z = zeros(size(X1));
        for i = 1:size(X_grid, 1)
            Z(i) = exp(-0.5 * (X_grid(i,:) - mult_marg(k,:)) * Plt_marg(:,:,k)^(-1) * (X_grid(i,:) - mult_marg(k,:))');
        end
        Z = reshape(Z, size(X1));
        Z = Z/(2*pi*sqrt(det(Plt_marg(:,:,k))));
        contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

        contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(1,:));
        hold on;

        [X1, X2] = meshgrid(linspace(min(Xpht_cloud(:,plot_dims(1))), max(Xpht_cloud(:,plot_dims(1))), 100), ...
                    linspace(min(Xpht_cloud(:,plot_dims(2))), max(Xpht_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
        
        Z = zeros(size(X1));
        for i = 1:size(X_grid, 1)
            Z(i) = exp(-0.5 * (X_grid(i,:) - muht_marg(k,:)) * Pht_marg(:,:,k)^(-1) * (X_grid(i,:) - muht_marg(k,:))');
        end
        Z = reshape(Z, size(X1));
        Z = Z/(2*pi*sqrt(det(Pht_marg(:,:,k))));
        contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

        contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(2,:));
        hold on;
    end
    plot(Xprop_truth(2), Xprop_truth(3), 'kx','MarkerSize', 20, 'LineWidth', 3)
    hold on;
    plot(Xprop_truth_lowTol(2), Xprop_truth_lowTol(3), 'rx','MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePt(2), samplePt(3), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePtlt(2), samplePtlt(3), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    title('Y-Z');
    xlabel('Y');
    ylabel('Z');
    % legend(legend_string);
    hold off;

    subplot(2,3,4)
    plot_dims = [4,5];
    mult_marg = mult_mat(:, plot_dims); muht_marg = muht_mat(:, plot_dims);
    Plt_marg = Plt_mat(plot_dims, plot_dims, :); Pht_marg = Pht_mat(plot_dims, plot_dims, :);

    for k = 1:K
        [X1, X2] = meshgrid(linspace(min(Xplt_cloud(:,plot_dims(1))), max(Xplt_cloud(:,plot_dims(1))), 100), ...
                    linspace(min(Xplt_cloud(:,plot_dims(2))), max(Xplt_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
        
        Z = zeros(size(X1));
        for i = 1:size(X_grid, 1)
            Z(i) = exp(-0.5 * (X_grid(i,:) - mult_marg(k,:)) * Plt_marg(:,:,k)^(-1) * (X_grid(i,:) - mult_marg(k,:))');
        end
        Z = reshape(Z, size(X1));
        Z = Z/(2*pi*sqrt(det(Plt_marg(:,:,k))));
        contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

        contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(1,:));
        hold on;

        [X1, X2] = meshgrid(linspace(min(Xpht_cloud(:,plot_dims(1))), max(Xpht_cloud(:,plot_dims(1))), 100), ...
                    linspace(min(Xpht_cloud(:,plot_dims(2))), max(Xpht_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
        
        Z = zeros(size(X1));
        for i = 1:size(X_grid, 1)
            Z(i) = exp(-0.5 * (X_grid(i,:) - muht_marg(k,:)) * Pht_marg(:,:,k)^(-1) * (X_grid(i,:) - muht_marg(k,:))');
        end
        Z = reshape(Z, size(X1));
        Z = Z/(2*pi*sqrt(det(Pht_marg(:,:,k))));
        contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

        contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(2,:));
        hold on;
    end
    plot(Xprop_truth(4), Xprop_truth(5), 'kx','MarkerSize', 20, 'LineWidth', 3)
    hold on;
    plot(Xprop_truth_lowTol(4), Xprop_truth_lowTol(5), 'rx','MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePt(4), samplePt(5), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePtlt(4), samplePtlt(5), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    title('Xdot-Ydot');
    xlabel('Xdot');
    ylabel('Ydot');
    % legend(legend_string);
    hold off;

    subplot(2,3,5)
    plot_dims = [4,6];
    mult_marg = mult_mat(:, plot_dims); muht_marg = muht_mat(:, plot_dims);
    Plt_marg = Plt_mat(plot_dims, plot_dims, :); Pht_marg = Pht_mat(plot_dims, plot_dims, :);

    for k = 1:K
        [X1, X2] = meshgrid(linspace(min(Xplt_cloud(:,plot_dims(1))), max(Xplt_cloud(:,plot_dims(1))), 100), ...
                    linspace(min(Xplt_cloud(:,plot_dims(2))), max(Xplt_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
        
        Z = zeros(size(X1));
        for i = 1:size(X_grid, 1)
            Z(i) = exp(-0.5 * (X_grid(i,:) - mult_marg(k,:)) * Plt_marg(:,:,k)^(-1) * (X_grid(i,:) - mult_marg(k,:))');
        end
        Z = reshape(Z, size(X1));
        Z = Z/(2*pi*sqrt(det(Plt_marg(:,:,k))));
        contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

        contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(1,:));
        hold on;

        [X1, X2] = meshgrid(linspace(min(Xpht_cloud(:,plot_dims(1))), max(Xpht_cloud(:,plot_dims(1))), 100), ...
                    linspace(min(Xpht_cloud(:,plot_dims(2))), max(Xpht_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
        
        Z = zeros(size(X1));
        for i = 1:size(X_grid, 1)
            Z(i) = exp(-0.5 * (X_grid(i,:) - muht_marg(k,:)) * Pht_marg(:,:,k)^(-1) * (X_grid(i,:) - muht_marg(k,:))');
        end
        Z = reshape(Z, size(X1));
        Z = Z/(2*pi*sqrt(det(Pht_marg(:,:,k))));
        contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

        contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(2,:));
        hold on;
    end
    plot(Xprop_truth(4), Xprop_truth(6), 'kx','MarkerSize', 20, 'LineWidth', 3)
    hold on;
    plot(Xprop_truth_lowTol(4), Xprop_truth_lowTol(6), 'rx','MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePt(4), samplePt(6), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePtlt(4), samplePtlt(6), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    title('Xdot-Zdot');
    xlabel('Xdot');
    ylabel('Zdot');
    % legend(legend_string);
    hold off;

    subplot(2,3,6)
    plot_dims = [5,6];
    mult_marg = mult_mat(:, plot_dims); muht_marg = muht_mat(:, plot_dims);
    Plt_marg = Plt_mat(plot_dims, plot_dims, :); Pht_marg = Pht_mat(plot_dims, plot_dims, :);

    for k = 1:K
        [X1, X2] = meshgrid(linspace(min(Xplt_cloud(:,plot_dims(1))), max(Xplt_cloud(:,plot_dims(1))), 100), ...
                    linspace(min(Xplt_cloud(:,plot_dims(2))), max(Xplt_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
        
        Z = zeros(size(X1));
        for i = 1:size(X_grid, 1)
            Z(i) = exp(-0.5 * (X_grid(i,:) - mult_marg(k,:)) * Plt_marg(:,:,k)^(-1) * (X_grid(i,:) - mult_marg(k,:))');
        end
        Z = reshape(Z, size(X1));
        Z = Z/(2*pi*sqrt(det(Plt_marg(:,:,k))));
        contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

        contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(1,:));
        hold on;

        [X1, X2] = meshgrid(linspace(min(Xpht_cloud(:,plot_dims(1))), max(Xpht_cloud(:,plot_dims(1))), 100), ...
                    linspace(min(Xpht_cloud(:,plot_dims(2))), max(Xpht_cloud(:,plot_dims(2))), 100));
        X_grid = [X1(:) X2(:)];
        
        Z = zeros(size(X1));
        for i = 1:size(X_grid, 1)
            Z(i) = exp(-0.5 * (X_grid(i,:) - muht_marg(k,:)) * Pht_marg(:,:,k)^(-1) * (X_grid(i,:) - muht_marg(k,:))');
        end
        Z = reshape(Z, size(X1));
        Z = Z/(2*pi*sqrt(det(Pht_marg(:,:,k))));
        contour_levels = max(Z(:)) * exp(-0.5 * [1, 2.3, 3.44].^2);  % Corresponding to sigma intervals

        contour(X1, X2, Z, contour_levels, 'LineWidth', 2, 'LineColor', contourCols(2,:));
        hold on;
    end
    plot(Xprop_truth(5), Xprop_truth(6), 'kx','MarkerSize', 20, 'LineWidth', 3)
    hold on;
    plot(Xprop_truth_lowTol(5), Xprop_truth_lowTol(6), 'rx','MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePt(5), samplePt(6), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePtlt(5), samplePtlt(6), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    title('Ydot-Zdot');
    xlabel('Ydot');
    ylabel('Zdot');
    % legend(legend_string);
    hold off;

    sgt = sprintf('Timestep: %1.5f (Posterior)', tpr);
    sgtitle(sgt);

    sg = sprintf('./Small_Sims/Timestep_%i_2A.png', tau);
    saveas(f, sg, 'png');
    close(f);

    f = figure('visible','off','Position', get(0,'ScreenSize'));
    f.WindowState = 'maximized';

    legend_string = {};
    for k = 1:K
        % legend_string{k} = sprintf('Contour %i', k);
        legend_string{k} = sprintf('\\omega = %1.4f', wplt(k));
        legend_string{K+k} = sprintf('\\omega = %1.4f', wpht(k));
    end
    % legend_string{K+1} = "Centroids";
    legend_string{2*K+1} = "Truth (High Tol)";
    legend_string{2*K+2} = "Truth (Low Tol)";

    subplot(2,3,1)
    for k = 1:K
        clusterPoints = Xplt_cloud(clt_id == k, :);
        scatter(clusterPoints(:,1), clusterPoints(:,2), 'filled', 'MarkerFaceColor', colors(k));
        hold on;
        clusterPoints = Xpht_cloud(cht_id == k, :);
        scatter(clusterPoints(:,1), clusterPoints(:,2), 'filled', 'MarkerFaceColor', colors(K+k));
    end
   
    plot(samplePt(1), samplePt(2), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePtlt(1), samplePtlt(2), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(Xprop_truth(1), Xprop_truth(2), 'kx','MarkerSize', 20, 'LineWidth', 3)
    hold on;
    plot(Xprop_truth_lowTol(1), Xprop_truth_lowTol(2), 'rx','MarkerSize', 15, 'LineWidth', 3);
    title('X-Y');
    xlabel('X');
    ylabel('Y');
    % legend(legend_string);
    hold off;

    subplot(2,3,2)
    for k = 1:K
        clusterPoints = Xplt_cloud(clt_id == k, :);
        scatter(clusterPoints(:,1), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
        hold on;
        clusterPoints = Xpht_cloud(cht_id == k, :);
        scatter(clusterPoints(:,1), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(K+k));
    end
    plot(samplePt(1), samplePt(3), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePtlt(1), samplePtlt(3), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(Xprop_truth(1), Xprop_truth(3), 'kx','MarkerSize', 20, 'LineWidth', 3)
    hold on;
    plot(Xprop_truth_lowTol(1), Xprop_truth_lowTol(3), 'rx','MarkerSize', 15, 'LineWidth', 3);
    title('X-Z');
    xlabel('X');
    ylabel('Z');
    % legend(legend_string);
    hold off;
    
    subplot(2,3,3)
    for k = 1:K
        clusterPoints = Xplt_cloud(clt_id == k, :);
        scatter(clusterPoints(:,2), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(k));
        hold on;
        clusterPoints = Xpht_cloud(cht_id == k, :);
        scatter(clusterPoints(:,2), clusterPoints(:,3), 'filled', 'MarkerFaceColor', colors(K+k));
    end
    plot(samplePt(2), samplePt(3), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePtlt(2), samplePtlt(3), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(Xprop_truth(2), Xprop_truth(3), 'kx','MarkerSize', 20, 'LineWidth', 3)
    hold on;
    plot(Xprop_truth_lowTol(2), Xprop_truth_lowTol(3), 'rx','MarkerSize', 15, 'LineWidth', 3);
    title('Y-Z');
    xlabel('Y');
    ylabel('Z');
    % legend(legend_string);
    hold off;
    
    subplot(2,3,4)
    for k = 1:K
        clusterPoints = Xplt_cloud(clt_id == k, :);
        scatter(clusterPoints(:,4), clusterPoints(:,5), 'filled', 'MarkerFaceColor', colors(k));
        hold on;
        clusterPoints = Xpht_cloud(cht_id == k, :);
        scatter(clusterPoints(:,4), clusterPoints(:,5), 'filled', 'MarkerFaceColor', colors(k+K));
    end
    plot(samplePt(4), samplePt(5), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePtlt(4), samplePtlt(5), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(Xprop_truth(4), Xprop_truth(5), 'kx','MarkerSize', 20, 'LineWidth', 3)
    hold on;
    plot(Xprop_truth_lowTol(4), Xprop_truth_lowTol(5), 'rx','MarkerSize', 15, 'LineWidth', 3);
    title('Xdot-Ydot');
    xlabel('Xdot');
    ylabel('Ydot');
    % legend(legend_string);
    hold off;
    
    subplot(2,3,5)
    for k = 1:K
        clusterPoints = Xplt_cloud(clt_id == k, :);
        scatter(clusterPoints(:,4), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
        hold on;
        clusterPoints = Xpht_cloud(cht_id == k, :);
        scatter(clusterPoints(:,4), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k+K));
    end
    plot(samplePt(4), samplePt(6), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePtlt(4), samplePtlt(6), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(Xprop_truth(4), Xprop_truth(6), 'kx','MarkerSize', 20, 'LineWidth', 3)
    hold on;
    plot(Xprop_truth_lowTol(4), Xprop_truth_lowTol(6), 'rx','MarkerSize', 15, 'LineWidth', 3);
    title('Xdot-Zdot');
    xlabel('Xdot');
    ylabel('Zdot');
    % legend(legend_string);
    hold off;
    
    subplot(2,3,6)
    for k = 1:K
        clusterPoints = Xplt_cloud(clt_id == k, :);
        scatter(clusterPoints(:,5), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k));
        hold on;
        clusterPoints = Xpht_cloud(cht_id == k, :);
        scatter(clusterPoints(:,5), clusterPoints(:,6), 'filled', 'MarkerFaceColor', colors(k+K));
    end
    plot(samplePt(5), samplePt(6), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(samplePtlt(5), samplePtlt(6), 'o', 'MarkerSize', 15, 'LineWidth', 3);
    hold on;
    plot(Xprop_truth(5), Xprop_truth(6), 'kx','MarkerSize', 20, 'LineWidth', 3)
    hold on;
    plot(Xprop_truth_lowTol(5), Xprop_truth_lowTol(6), 'rx','MarkerSize', 15, 'LineWidth', 3);
    title('Ydot-Zdot');
    xlabel('Ydot');
    ylabel('Zdot');
    % legend(legend_string);
    hold off;

    sgt = sprintf('Timestep: %1.5f (Posterior)', tpr);
    sgtitle(sgt);

    sg = sprintf('./Small_Sims/Timestep_%i_2B.png', tau);
    saveas(f, sg, 'png');
    close(f);

    %{
    if (abs(tpr-hdR(end,1)) < 1e-10)
        Xp_cloudp = zeros(L, length(Xprop_truth));
        c_id = zeros(length(Xp_cloudp(:,1)),1);
        for i = 1:L
            [Xp_cloudp(i,:), c_id(i)] = drawFrom(wp, mu_p, P_p); 
        end

        [idx_trackEnd, ~] = find(abs(full_ts(:,1) - hdR(end,1)) < 1e-10);
        Xprop_truth = [full_ts(idx_trackEnd,2:4), full_vts(idx_trackEnd,2:4)];
        mu_pExp = zeros(K, length(mu_p{1}));

        % Extract means
        for k = 1:K
            mu_pExp(k,:) = mu_p{k};
        end

        fprintf('Last Observable Regime Timestep Truth:\n')
        disp(Xprop_truth);

        % Plot planar projections
        figure(5)
        subplot(2,3,1)
        gscatter(Xp_cloudp(:,1), Xp_cloudp(:,2), c_id);
        hold on;
        plot(mu_pExp(:,1), mu_pExp(:,2), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
        hold on;
        plot(Xprop_truth(1), Xprop_truth(2), 'x','MarkerSize', 20, 'LineWidth', 3)
        title('X-Y');
        xlabel('X');
        ylabel('Y');
        legend(legend_string);
        hold off;

        subplot(2,3,2)
        gscatter(Xp_cloudp(:,1), Xp_cloudp(:,3), c_id);
        hold on;
        plot(mu_pExp(:,1), mu_pExp(:,3), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
        hold on;
        plot(Xprop_truth(1), Xprop_truth(3), 'x','MarkerSize', 20, 'LineWidth', 3)
        title('X-Z');
        xlabel('X');
        ylabel('Z');
        legend(legend_string);
        hold off;
        
        subplot(2,3,3)
        gscatter(Xp_cloudp(:,2), Xp_cloudp(:,3), c_id);
        hold on;
        plot(mu_pExp(:,2), mu_pExp(:,3), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
        hold on;
        plot(Xprop_truth(2), Xprop_truth(3), 'x','MarkerSize', 20, 'LineWidth', 3)
        title('Y-Z');
        xlabel('Y');
        ylabel('Z');
        legend(legend_string);
        hold off;
        
        subplot(2,3,4)
        gscatter(Xp_cloudp(:,4), Xp_cloudp(:,5), c_id);
        hold on;
        plot(mu_pExp(:,4), mu_pExp(:,5), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
        hold on;
        plot(Xprop_truth(4), Xprop_truth(5), 'x','MarkerSize', 20, 'LineWidth', 3)
        title('Xdot-Ydot');
        xlabel('Xdot');
        ylabel('Ydot');
        legend(legend_string);
        hold off;
        
        subplot(2,3,5)
        gscatter(Xp_cloudp(:,4), Xp_cloudp(:,6), c_id);
        hold on;
        plot(mu_pExp(:,4), mu_pExp(:,6), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
        hold on;
        plot(Xprop_truth(4), Xprop_truth(6), 'x','MarkerSize', 20, 'LineWidth', 3)
        title('Xdot-Zdot');
        xlabel('Xdot');
        ylabel('Zdot');
        legend(legend_string);
        hold off;
        
        subplot(2,3,6)
        gscatter(Xp_cloudp(:,5), Xp_cloudp(:,6), c_id);
        hold on;
        plot(mu_pExp(:,5), mu_pExp(:,6), 'k+', 'MarkerSize', 10, 'LineWidth', 3);
        hold on;
        plot(Xprop_truth(5), Xprop_truth(6), 'x','MarkerSize', 20, 'LineWidth', 3)
        title('Ydot-Zdot');
        xlabel('Ydot');
        ylabel('Zdot');
        legend(legend_string);
        hold off;
        
        % sg = sprintf('Time Step: %i', int32((hdR(end,1) - interval - tpr)/interval + 1));
        sgtitle('End of First Pass')
        savefig(gcf, 'endOfTracklet_normK.fig');

    elseif(abs(tpr - (t_end-interval)) < 1e-10) % Second to last possible time step
        Xp_cloudp = zeros(L, length(Xprop_truth));
        c_id = zeros(L,1);
        for i = 1:L
            [Xp_cloudp(i,:), c_id(i)] = drawFrom(wp, mu_p, P_p); 
        end

        [idx_stl, ~] = find(abs(full_ts(:,1) - tpr) < 1e-10);
        Xprop_truth = [full_ts(idx_stl,2:4), full_vts(idx_stl,2:4)];

        % Plot planar projections
        figure(7)
        subplot(2,3,1)

        mu_pExp = zeros(K, length(mu_p{1}));

        if(idx_meas ~= 0)
            K = Kn;
        else
            K = 1;
        end

        for k = 1:K
            clusterPoints = Xp_cloudp(c_id == k, :);
            mu_pExp(k,:) = mu_p{k};
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
        savefig(gcf, 'secondToLastEstimate.fig');

        K = Kn;
    end
    %}

end

%{
Xp_cloudp = zeros(L, length(Xprop_truth));
c_id = zeros(L,1);
for i = 1:L
    [Xp_cloudp(i,:), c_id(i)] = drawFrom(wp, mu_p, P_p); 
end

[idx_end, ~] = find(abs(full_ts(:,1) - t_end) < 1e-10);
Xprop_truth = [full_ts(idx_end,2:4), full_vts(idx_end,2:4)];
mu_pExp = zeros(K, length(mu_p{1}));

fprintf('Final State Truth:\n')
disp(Xprop_truth);

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
%}

% Stop the clock
toc

%% Functions

function pg = gaussProb(x_i, mu, P)
    n = length(mu);
    pg = 1/((2*pi)^(n/2)*sqrt(det(P))) * exp(-0.5*(x_i - mu)'*P^(-1)*(x_i - mu));
end

function Hx = linHx(mu)
    Hk_R = [mu(1)/sqrt(mu(1)^2 + mu(2)^2 + mu(3)^2), ...
            mu(2)/sqrt(mu(1)^2 + mu(2)^2 + mu(3)^2), ...
            mu(3)/sqrt(mu(1)^2 + mu(2)^2 + mu(3)^2), 0, 0, 0]; % Range linearization
    Hk_AZ = [-mu(2)/(mu(1)^2 + mu(2)^2), mu(1)/(mu(1)^2 + mu(2)^2), 0, 0, 0, 0]; % Azimuth angle linearization
    % Hk_EL = [-(mu(1)*mu(3))/((mu(1)^2 + mu(2)^2 + mu(3)^2)^(1.5)*sqrt(1 - mu(3)^2/(mu(1)^2 + mu(2)^2 +mu(3)^2))), ...
    %         -(mu(2)*mu(3))/((mu(1)^2 + mu(2)^2 + mu(3)^2)^(1.5)*sqrt(1 - mu(3)^2/(mu(1)^2 + mu(2)^2 +mu(3)^2))), ...
    %         sqrt(mu(1)^2 + mu(2)^2)/(mu(1)^2 + mu(2)^2 + mu(3)^2), 0, 0, 0];
    Hk_EL = [-(mu(1)*mu(3))/((mu(1)^2 + mu(2)^2 + mu(3)^2)*sqrt(mu(1)^2+mu(2)^2)), ...
             -(mu(2)*mu(3))/((mu(1)^2 + mu(2)^2 + mu(3)^2)*sqrt(mu(1)^2+mu(2)^2)), ...
             sqrt(mu(1)^2 + mu(2)^2)/(mu(1)^2 + mu(2)^2 + mu(3)^2), 0, 0, 0];

    Hx = [Hk_R; Hk_AZ; Hk_EL];
end

% Adds process noise to the un-noised state vector
function [Xm] = procNoise(X)
    Q = 0.000^2*diag(abs(X)); % Process noise is 1% of each state vector component
    Xm = mvnrnd(X,Q);
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
    % Use histcounts for efficient sampling
    pos = histcounts(rand, [0; cumsum(w(:))]);
    pos = find(pos, 1);

    if (isempty(pos) || pos > length(mu))
        error('Sampling error: invalid position');
    end
    
    x_p = mvnrnd(mu{pos}, P{pos});
end

function Xm_cloud = propagate(Xcloud, t_int, interval, propno)
    % Xcloud = zeros(L,length(mu{1}));
    % for i = 1:L
    %     [Xcloud(i,:), ~] = drawFrom(w, mu, P);
    % end
    % 
    % Xm_cloud = Xcloud;
    if (nargin < 4)
        propno = 1;
    end

    if (length(Xcloud(:,1)) == 6)
        Xcloud = Xcloud';
    end

    for i = 1:length(Xcloud(:,1))
        % First, convert from X_{ot} in the topocentric frame to X_{bt} in the
        % synodic frame.
        Xbt = backConvertSynodic(Xcloud(i,:)', t_int);

        % Next, propagate each X_{bt} in your particle cloud by a single time 
        % step and convert back to the topographic frame.
        % Call ode45()
        if(propno == 1)
            opts = odeset('Events', @termSat);
        else
            opts = odeset('AbsTol',1e-6,'RelTol',1e-6,'Events', @termSat);
        end
        [~,X] = ode45(@cr3bp_dyn, [0 interval], Xbt, opts); % Assumes termination event (i.e. target enters LEO)
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

    % K_k = Pxz/Pzz;
    % mu_p = mu_m' + K_k*(zk - h(mu_m));
    % P_p = P_m - K_k*Pzz*K_k';
    
    mu_p = mu_m' + Pxz'*Pzz^(-1)*(zk - h(mu_m));
    P_p = P_m - Pxz'*Pzz^(-1)*Pxz;
    
    P_p = (P_p + P_p')/2;

    [V, D] = eig(P_p);
    D = max(D,0);
    P_p = V*D*V';
end