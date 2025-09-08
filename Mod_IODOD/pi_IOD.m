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

% Coordinate system conversions
dist2km = 384400; % Kilometers per non-dimensionalized distance
time2hr = 4.342*24; % Hours per non-dimensionalized time
vel2kms = dist2km/(time2hr*60*60); % Kms per non-dimensionalized velocity

R_t = zeros(3*length(noised_obs(:,1)),1); % We shall diagonalize this later
mu_t = zeros(3*length(noised_obs(:,1)),1);

theta_f = 1.5; % Arc-seconds of error covariance
R_f = 0.75; % Range percentage error covariance

for i = 1:length(partial_ts(:,1))
    mu_t(3*(i-1)+1:3*(i-1)+3, 1) = [partial_ts(i,2); partial_ts(i,3); partial_ts(i,4)];
    R_t(3*(i-1)+1:3*(i-1)+3, 1) = [(R_f*partial_ts(i,2))^2; (theta_f*4.84814e-6)^2; (theta_f*4.84814e-6)^2];
end

R_t = diag(R_t);
data_vec = mvnrnd(mu_t, R_t)';

for i = 1:length(noised_obs(:,1))
    noised_obs(i,2:4) = data_vec(3*(i-1)+1:3*(i-1)+3,1);
end

% Plot the spherical coordinates of the observer parametrically w.r.t. time
figure(1)
subplot(3,1,1)
plot(noised_obs(:,1), noised_obs(:,2), 'o', partial_ts(:,1), partial_ts(:,2), 'o')
xlabel('Time')
ylabel('Range (non-dim)')
title('Observer Range Measurements (Ideal)')

subplot(3,1,2)
plot(noised_obs(:,1), noised_obs(:,3), 'o', partial_ts(:,1), partial_ts(:,3), 'o')
xlabel('Time')
ylabel('Azimuth Angle (rad)')
title('Observer Azimuth Angle Measurements (Ideal)')

subplot(3,1,3)
plot(noised_obs(:,1), noised_obs(:,4), 'o', partial_ts(:,1), partial_ts(:,4), 'o')
xlabel('Time')
ylabel('Elevation Angle (rad)')
title('Observer Elevation Angle Measurements (Ideal)')
saveas(gcf, 'noisyObservations_ECI.png')

obs_diffs = abs(partial_ts(:,2:4) - noised_obs(:,2:4));

figure(2)
subplot(3,1,1)
plot(noised_obs(:,1), obs_diffs(:,1), 'o')
xlabel('Time')
ylabel('Range (non-dim)')
title('Observer Range Measurements Differences')

subplot(3,1,2)
plot(noised_obs(:,1), obs_diffs(:,2), 'o')
xlabel('Time')
ylabel('Azimuth Angle (rad)')
title('Observer Azimuth Angle Measurements Differences')

subplot(3,1,3)
plot(noised_obs(:,1), obs_diffs(:,3), 'o')
xlabel('Time')
ylabel('Elevation Angle (rad)')
title('Observer Elevation Angle Measurements Differences')

% Extract the first continuous observation track

hdo = []; % Matrix for a half day observation
hdo(1,:) = noised_obs(1,:);
i = 1;
while(noised_obs(i+1,1) - noised_obs(i,1) < full_ts(2,1) + 1e-15) % Add small epsilon due to roundoff error
    hdo(i,:) = noised_obs(i+1,:);
    i = i + 1;
end

% Plot the spherical coordinates of the observer parametrically w.r.t. time
figure(3)
subplot(3,1,1)
plot(hdo(:,1), hdo(:,2), 'ro')
xlabel('Time')
ylabel('Range (non-dim)')
title('Observer Range Measurements')

subplot(3,1,2)
plot(hdo(:,1), hdo(:,3), 'go')
xlabel('Time')
ylabel('Azimuth Angle (rad)')
title('Observer Azimuth Angle Measurements')

subplot(3,1,3)
plot(hdo(:,1), hdo(:,4), 'bo')
xlabel('Time')
ylabel('Elevation Angle (rad)')
title('Observer Elevation Angle Measurements')

% Convert observation data into [X, Y, Z] data in the topographic frame.

hdR = zeros(length(hdo(:,1)),4); % Convert quantities of hdo to [X, Y, Z]
hdR(:,1) = hdo(:,1); % Timestamp stays the same
hdR(:,2) = hdo(:,2) .* cos(hdo(:,4)) .* cos(hdo(:,3)); % Conversion to X
hdR(:,3) = hdo(:,2) .* cos(hdo(:,4)) .* sin(hdo(:,3)); % Conversion to Y
hdR(:,4) = hdo(:,2) .* sin(hdo(:,4)); % Conversion to Z

% Plot the spherical coordinates of the observer parametrically w.r.t. time
figure(4)
subplot(3,1,1)
plot(hdR(:,1), hdR(:,2), 'ro')
xlabel('Time')
ylabel('X (non-dim)')
title('Observer X Magnitudes')

subplot(3,1,2)
plot(hdR(:,1), hdR(:,3), 'go')
xlabel('Time')
ylabel('Y (non-dim)')
title('Observer Y Magnitudes')

subplot(3,1,3)
plot(hdR(:,1), hdR(:,4), 'bo')
xlabel('Time')
ylabel('Z (non-dim)')
title('Observer Z Magnitudes')

saveas(gcf, 'noisedXYZ_singleTrack.png')

pf = 0.5; % A factor between 0 to 1 describing the length of the day to interpolate [x, y]
nfit = 4;
in_len = round(pf * length(hdR(:,1))); % Length of interpolation interval

if(in_len < nfit + 1)
    in_len = nfit + 1;
    pf = in_len/length(hdR(:,1));
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

idx_interpStart = find(abs(full_vts(:,1) - hdR_p(1,1)) < 1e-11);
idx_interpEnd = find(abs(full_vts(:,1) - hdR_p(end,1)) < 1e-11);

% Plotting converted X, Y, Z data with the fitted polynomials
figure(4)
subplot(2,3,1)
scatter(hdR_p(:,1)*time2hr, hdR_p(:,2)*dist2km, 'blue', 'DisplayName', 'Topocentric X (Noisy)');
hold on;
scatter(hdR_p(end,1)*time2hr, X_fit(end)*dist2km, '*', 'red', 'DisplayName', 'Topocentric Z (IOD Point)');
hold on;
scatter(hdR_p(:,1)*time2hr, partial_rts(:,2)*dist2km, 'x', 'black', 'DisplayName', 'Topocentric X (Truth)');
hold on;
plot(hdR_p(:,1)*time2hr, X_fit*dist2km, 'red', 'DisplayName', 'Polynomial Fit')
xlabel('Time (hr.)')
ylabel('X (km.)')
legend('show')

subplot(2,3,2)
scatter(hdR_p(:,1)*time2hr, hdR_p(:,3)*dist2km, 'blue', 'DisplayName', 'Topocentric Y (Noisy)');
hold on;
scatter(hdR_p(end,1)*time2hr, Y_fit(end)*dist2km, '*', 'green', 'DisplayName', 'Topocentric Y (IOD Point)');
hold on;
scatter(hdR_p(:,1)*time2hr, partial_rts(:,3)*dist2km, 'x', 'black', 'DisplayName', 'Topocentric Y (Truth)');
hold on;
plot(hdR_p(:,1)*time2hr, Y_fit*dist2km, 'green', 'DisplayName', 'Polynomial Fit')
xlabel('Time (hr.)')
ylabel('Y (km.)')
legend('show')

subplot(2,3,3)
scatter(hdR_p(:,1)*time2hr, hdR_p(:,4)*dist2km, 'blue', 'DisplayName', 'Topocentric Z (Noisy)');
hold on;
scatter(hdR_p(end,1)*time2hr, Z_fit(end)*dist2km, '*', 'cyan', 'DisplayName', 'Topocentric Z (IOD Point)');
hold on;
scatter(hdR_p(:,1)*time2hr, partial_rts(:,4)*dist2km, 'x', 'black', 'DisplayName', 'Topocentric Z (Truth)');
hold on;
plot(hdR_p(:,1)*time2hr, Z_fit*dist2km, 'cyan', 'DisplayName', 'Polynomial Fit')
xlabel('Time (hr.)')
ylabel('Z (km.)')
legend('show')

subplot(2,3,4)
scatter(partial_vts(:,1)*time2hr, partial_vts(:,2)*vel2kms, 'x', 'black', 'DisplayName', 'Topocentric Xdot (Truth)');
hold on;
scatter(hdR_p(end,1)*time2hr, Xdot_fit(end)*vel2kms, '*', 'red', 'DisplayName', 'Topocentric Xdot (IOD Point)');
hold on;
plot(partial_vts(:,1)*time2hr, Xdot_fit*vel2kms, 'red', 'DisplayName', 'Polynomial Fit')
xlabel('Time (hr.)')
ylabel('Xdot (km/s)')
legend('show')

subplot(2,3,5)
scatter(partial_vts(:,1)*time2hr, partial_vts(:,3)*vel2kms, 'x', 'black', 'DisplayName', 'Topocentric Ydot (Truth)');
hold on;
scatter(hdR_p(end,1)*time2hr, Ydot_fit(end)*vel2kms, '*', 'green', 'DisplayName', 'Topocentric Ydot (IOD Point)');
hold on;
plot(partial_vts(:,1)*time2hr, Ydot_fit*vel2kms, 'green', 'DisplayName', 'Polynomial Fit')
xlabel('Time (hr.)')
ylabel('Ydot (km/s)')
legend('show')

subplot(2,3,6)
scatter(partial_vts(:,1)*time2hr, partial_vts(:,4)*vel2kms, 'x', 'black', 'DisplayName', 'Topocentric Zdot (Truth)');
hold on;
scatter(hdR_p(end,1)*time2hr, Zdot_fit(end)*vel2kms, '*', 'cyan', 'DisplayName', 'Topocentric Zdot (IOD Point)');
hold on;
plot(partial_vts(:,1)*time2hr, Zdot_fit*vel2kms, 'cyan', 'DisplayName', 'Polynomial Fit')
xlabel('Time (hr.)')
ylabel('Zdot (km/s)')
legend('show')

savefig(gcf, 'fittedStates.fig')
saveas(gcf, 'fittedStates.png')

%{
% Finding Point of best polynomial fit within interval
fitDiff_r = abs(hdR_p(:,2:4) - [X_fit, Y_fit, Z_fit]);
fitDiff_r = sum(fitDiff_r, 2);
[cmin, imin] = min(fitDiff_r);
%}

Xot_fitted = [X_fit(end,1); Y_fit(end,1); Z_fit(end,1); Xdot_fit(end,1); Ydot_fit(end,1); Zdot_fit(end,1)];
Xot_truth = [partial_rts(end,2:4), partial_vts(end,2:4)]';

t_truth = partial_rts(end,1);
[idx_prop, c_prop] = find(full_ts == t_truth);
Xprop_truth = [full_ts(idx_prop+1,2:4), full_vts(idx_prop+1,2:4)]';

L = 4000;
X0cloud_all = zeros(L,6);
% nfit = 2;

low_lim = (2*42164)/dist2km; % 2x the GEO distance
up_lim = 550000/dist2km; % Upper limit of cislunar space, as defined by the Aerospace Corporation

parfor i = 1:length(X0cloud_all(:,1))
    X0cloud_all(i,:) = stateEstCloud(pf, nfit, theta_f, R_f, partial_ts, (partial_ts(2,1) - partial_ts(1,1)) + 1e-15, low_lim, up_lim);
end

%%
% Show 2D plane plots of the above figure
figure(6)

subplot(2,3,1)
plot(X0cloud_all(:,1)*dist2km, X0cloud_all(:,2)*dist2km, '.');
hold on;
plot(Xot_truth(1)*dist2km, Xot_truth(2)*dist2km, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Position (km.)')
ylabel('Y Position (km.)')
title('X-Y')

subplot(2,3,2)
plot(X0cloud_all(:,1)*dist2km, X0cloud_all(:,3)*dist2km, '.');
hold on;
plot(Xot_truth(1)*dist2km, Xot_truth(3)*dist2km, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Position (km.)')
ylabel('Z Position (km.)')
title('X-Z')

subplot(2,3,3)
plot(X0cloud_all(:,2)*dist2km, X0cloud_all(:,3)*dist2km, '.');
hold on;
plot(Xot_truth(2)*dist2km, Xot_truth(3)*dist2km, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('Y Position (km.)')
ylabel('Z Position (km.)')
title('Y-Z')

subplot(2,3,4)
plot(X0cloud_all(:,4)*vel2kms, X0cloud_all(:,5)*vel2kms, '.');
hold on;
plot(Xot_truth(4)*vel2kms, Xot_truth(5)*vel2kms, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Velocity (km/s)')
ylabel('Y Velocity (km/s)')
title('Xdot-Ydot')

subplot(2,3,5)
plot(X0cloud_all(:,4)*vel2kms, X0cloud_all(:,6)*vel2kms, '.');
hold on;
plot(Xot_truth(4)*vel2kms, Xot_truth(6)*vel2kms, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Velocity (km/s)')
ylabel('Z Velocity (km/s)')
title('Xdot-Zdot')

subplot(2,3,6)
plot(X0cloud_all(:,5)*vel2kms, X0cloud_all(:,6)*vel2kms, '.');
hold on;
plot(Xot_truth(5)*vel2kms, Xot_truth(6)*vel2kms, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('Y Velocity (km/s)')
ylabel('Z Velocity (km/s)')
title('Ydot-Zdot')

Xcloud_posFilt = zeros(size(X0cloud_all)); j = 0;
reo = getObserverPos(t_truth); reo = reshape(reo, [1 length(reo)]);

for i = 1:length(X0cloud_all(:,1))
    if(norm(X0cloud_all(i,1:3) + reo) > low_lim && ...
            norm(X0cloud_all(i,1:3)) <= up_lim) % Upper bound according to The Aerospace Corporation
        j = j + 1;
        Xcloud_posFilt(j,:) = X0cloud_all(i,:);
    end
end
Xcloud_posFilt = Xcloud_posFilt(1:j,:);

Xcloud_velFilt = zeros(size(Xcloud_posFilt)); j = 0;

for i = 1:length(Xcloud_posFilt(:,1))
    if(norm(Xcloud_posFilt(i,4:6)) < 42.2/vel2kms) % Upper bound according to The Aerospace Corporation
        j = j + 1;
        Xcloud_velFilt(j,:) = Xcloud_posFilt(i,:);
    end
end
Xcloud = Xcloud_velFilt(1:j,:);

% Show 2D plane plots of the above figure
figure(7)

subplot(2,3,1)
plot(Xcloud_posFilt(:,1)*dist2km, Xcloud_posFilt(:,2)*dist2km, '.');
hold on;
plot(Xot_truth(1)*dist2km, Xot_truth(2)*dist2km, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Position (km.)')
ylabel('Y Position (km.)')
title('X-Y')

subplot(2,3,2)
plot(Xcloud_posFilt(:,1)*dist2km, Xcloud_posFilt(:,3)*dist2km, '.');
hold on;
plot(Xot_truth(1)*dist2km, Xot_truth(3)*dist2km, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Position (km.)')
ylabel('Z Position (km.)')
title('X-Z')

subplot(2,3,3)
plot(Xcloud_posFilt(:,2)*dist2km, Xcloud_posFilt(:,3)*dist2km, '.');
hold on;
plot(Xot_truth(2)*dist2km, Xot_truth(3)*dist2km, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('Y Position (km.)')
ylabel('Z Position (km.)')
title('Y-Z')

subplot(2,3,4)
plot(Xcloud_posFilt(:,4)*vel2kms, Xcloud_posFilt(:,5)*vel2kms, '.');
hold on;
plot(Xot_truth(4)*vel2kms, Xot_truth(5)*vel2kms, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Velocity (km/s)')
ylabel('Y Velocity (km/s)')
title('Xdot-Ydot')

subplot(2,3,5)
plot(Xcloud_posFilt(:,4)*vel2kms, Xcloud_posFilt(:,6)*vel2kms, '.');
hold on;
plot(Xot_truth(4)*vel2kms, Xot_truth(6)*vel2kms, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Velocity (km/s)')
ylabel('Z Velocity (km/s)')
title('Xdot-Zdot')

subplot(2,3,6)
plot(Xcloud_posFilt(:,5)*vel2kms, Xcloud_posFilt(:,6)*vel2kms, '.');
hold on;
plot(Xot_truth(5)*vel2kms, Xot_truth(6)*vel2kms, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('Y Velocity (km/s)')
ylabel('Z Velocity (km/s)')
title('Ydot-Zdot')

% Plot the cloud and the truth
figure(5)

subplot(1,2,1)
plot3(Xcloud(:,1), Xcloud(:,2), Xcloud(:,3), '.', 'DisplayName', 'Interpolated State');
hold on;
plot3(Xot_truth(1), Xot_truth(2), Xot_truth(3), '+', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Target Truth')
xlabel('X Position')
ylabel('Y Position')
zlabel('Z Position')
title('Position Space')

subplot(1,2,2)
plot3(Xcloud(:,4), Xcloud(:,5), Xcloud(:,6), '.', 'DisplayName', 'Interpolated State');
hold on;
plot3(Xot_truth(4), Xot_truth(5), Xot_truth(6), '+', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'Target Truth')
xlabel('X Velocity')
ylabel('Y Velocity')
zlabel('Z Velocity')
title('Velocity Space')

% Show 2D plane plots of the above figure
figure(8)

subplot(2,3,1)
plot(Xcloud(:,1)*dist2km, Xcloud(:,2)*dist2km, '.');
hold on;
plot(Xot_truth(1)*dist2km, Xot_truth(2)*dist2km, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Position (km.)')
ylabel('Y Position (km.)')
title('X-Y')

subplot(2,3,2)
plot(Xcloud(:,1)*dist2km, Xcloud(:,3)*dist2km, '.');
hold on;
plot(Xot_truth(1)*dist2km, Xot_truth(3)*dist2km, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Position (km.)')
ylabel('Z Position (km.)')
title('X-Z')

subplot(2,3,3)
plot(Xcloud(:,2)*dist2km, Xcloud(:,3)*dist2km, '.');
hold on;
plot(Xot_truth(2)*dist2km, Xot_truth(3)*dist2km, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('Y Position (km.)')
ylabel('Z Position (km.)')
title('Y-Z')

subplot(2,3,4)
plot(Xcloud(:,4)*vel2kms, Xcloud(:,5)*vel2kms, '.');
hold on;
plot(Xot_truth(4)*vel2kms, Xot_truth(5)*vel2kms, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Velocity (km/s)')
ylabel('Y Velocity (km/s)')
title('Xdot-Ydot')

subplot(2,3,5)
plot(Xcloud(:,4)*vel2kms, Xcloud(:,6)*vel2kms, '.');
hold on;
plot(Xot_truth(4)*vel2kms, Xot_truth(6)*vel2kms, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Velocity (km/s)')
ylabel('Z Velocity (km/s)')
title('Xdot-Zdot')

subplot(2,3,6)
plot(Xcloud(:,5)*vel2kms, Xcloud(:,6)*vel2kms, '.');
hold on;
plot(Xot_truth(5)*vel2kms, Xot_truth(6)*vel2kms, '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('Y Velocity (km/s)')
ylabel('Z Velocity (km/s)')
title('Ydot-Zdot')

% Here, we see how the distribution of range looks.
pos = Xcloud(:,1:3); % All particle clouds in the position space
ranges = dist2km*vecnorm(pos'); % Get the norm of each particle

figure(9)
h = histogram(ranges);
xlabel('Range (km.)')
ylabel('Frequency')

%{
subplot(2,1,1)
plot3(Xcloud(:,1), Xcloud(:,2), Xcloud(:,3), '.');
hold on;
plot3(Xot_truth(1), Xot_truth(2), Xot_truth(3), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Position')
ylabel('Y Position')
zlabel('Z Position')
title('Position Cloud')

subplot(2,1,2)
plot3(Xcloud(:,4), Xcloud(:,5), Xcloud(:,6), '.');
hold on;
plot3(Xot_truth(4), Xot_truth(5), Xot_truth(6), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Velocity')
ylabel('Y Velocity')
zlabel('Z Velocity')
title('Velocity Cloud')

savefig(gcf, "interpolationCloud.fig");
saveas(gcf, "interpolationCloud.png")

% Now that you have an initial cloud of particles, let's approximate the
% multi-variate mean and covariance of this data so that we may be able to
% approximate it as a single-mode Gaussian

% First, we calculate the mean
mu_c = zeros(1,length(Xcloud(1,:)));
% P_c = zeros(length(Xcloud(1,:)));
Xc = Xcloud; % Variable for re-centering Xcloud
for i = 1:length(Xcloud(1,:))
    mu_c(1,i) = mean(Xcloud(:,i));
    Xc(:,i) = Xcloud(:,i) - mu_c(1,i)*ones(length(Xcloud(:,1)),1);
end

P_c = (1/(length(Xcloud(:,1))-1)) * (Xc)' * Xc;

fprintf('Mean: \n');
disp(mu_c);
fprintf('Covariance:\n');
disp(P_c);

t_int = hdR_p(end,1); % Time at which we are obtaining a state cloud
interval = partial_ts(2,1) - partial_ts(1,1);
tspan = 0:interval:interval; % Integrate over just a single time step
Xm_cloud = Xcloud;

for i = 1:length(Xcloud(:,1))
    % First, convert from X_{ot} in the topocentric frame to X_{bt} in the
    % synodic frame.
    Xbt = backConvertSynodic(Xcloud(i,:)', t_int);

    % Next, propagate each X_{bt} in your particle cloud by a single time 
    % step and convert back to the topographic frame.
     % Call ode45()
    opts = odeset('Events', @termSat);
    [t,X] = ode45(@cr3bp_dyn, [0 interval], Xbt, opts); % Assumes termination event (i.e. target enters LEO)
    Xm_bt = X(end,:)';
    Xm_cloud(i,:) = convertToTopo(Xm_bt, t_int + interval);
end

% Plot the cloud and the truth
figure(7)
subplot(2,1,1)
plot3(Xm_cloud(:,1), Xm_cloud(:,2), Xm_cloud(:,3), '.');
hold on;
plot3(Xprop_truth(1), Xprop_truth(2), Xprop_truth(3), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Position')
ylabel('Y Position')
zlabel('Z Position')
title('Position Cloud')

subplot(2,1,2)
plot3(Xm_cloud(:,4), Xm_cloud(:,5), Xm_cloud(:,6), '.');
hold on;
plot3(Xprop_truth(4), Xprop_truth(5), Xprop_truth(6), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Velocity')
ylabel('Y Velocity')
zlabel('Z Velocity')
title('Velocity Cloud')

savefig(gcf, "interpolationCloud.fig");
saveas(gcf, "interpolationCloud.png");

% Show 2D plane plots of the above figure
figure(8)

subplot(2,3,1)
plot(Xm_cloud(:,1), Xm_cloud(:,2), '.');
hold on;
plot(Xprop_truth(1), Xprop_truth(2), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Position')
ylabel('Y Position')
title('X-Y')

subplot(2,3,2)
plot(Xm_cloud(:,1), Xm_cloud(:,3), '.');
hold on;
plot(Xprop_truth(1), Xprop_truth(3), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Position')
ylabel('Z Position')
title('X-Z')

subplot(2,3,3)
plot(Xm_cloud(:,2), Xm_cloud(:,3), '.');
hold on;
plot(Xprop_truth(2), Xprop_truth(3), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('Y Position')
ylabel('Z Position')
title('Y-Z')

subplot(2,3,4)
plot(Xm_cloud(:,4), Xm_cloud(:,5), '.');
hold on;
plot(Xprop_truth(4), Xprop_truth(5), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Velocity')
ylabel('Y Velocity')
title('Xdot-Ydot')

subplot(2,3,5)
plot(Xm_cloud(:,4), Xm_cloud(:,6), '.');
hold on;
plot(Xprop_truth(4), Xprop_truth(6), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Velocity')
ylabel('Z Velocity')
title('Xdot-Zdot')

subplot(2,3,6)
plot(Xm_cloud(:,5), Xm_cloud(:,6), '.');
hold on;
plot(Xprop_truth(5), Xprop_truth(6), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('Y Velocity')
ylabel('Z Velocity')
title('Ydot-Zdot')

mu_c = mean(Xm_cloud);
P_c = (Xm_cloud - mu_c)' * (Xm_cloud - mu_c) / (L - 1);

%{
% First, we calculate the mean
mu_c = zeros(1,length(Xm_cloud(1,:)));
% P_c = zeros(length(Xcloud(1,:)));
Xc = Xm_cloud; % Variable for re-centering Xcloud
for i = 1:length(Xm_cloud(1,:))
    mu_c(1,i) = mean(Xm_cloud(:,i));
    Xc(:,1) = Xm_cloud(:,1) - mu_c(1,i)*ones(length(Xm_cloud(:,1)),1);
end

P_c = (1/(length(Xm_cloud(:,1))-1)) * (Xc)' * Xc;
%}

fprintf('Mean (Prior Estimate): \n');
disp(mu_c);
fprintf('Covariance (Prior Estimate):\n');
disp(P_c);

% Update the mean and covariance with a Kalman update
t_stamp = t_int + interval;
[idx_meas, ~] = find(partial_ts == t_stamp);
if (idx_meas ~= 0) % i.e. there exists a measurement
    R_vv = R_t(3*idx_meas-2:3*idx_meas,3*idx_meas-2:3*idx_meas);
    Hk_R = [mu_c(1)/sqrt(mu_c(1)^2 + mu_c(2)^2 + mu_c(3)^2), ...
        mu_c(2)/sqrt(mu_c(1)^2 + mu_c(2)^2 + mu_c(3)^2), ...
        mu_c(3)/sqrt(mu_c(1)^2 + mu_c(2)^2 + mu_c(3)^2), 0, 0, 0]; % Range linearization
    Hk_AZ = [-mu_c(2)/(mu_c(1)^2 + mu_c(2)^2), mu_c(1)/(mu_c(1)^2 + mu_c(2)^2), 0, 0, 0, 0]; % Azimuth angle linearization
    Hk_EL = [-(mu_c(1)*mu_c(3))/((mu_c(1)^2 + mu_c(2)^2 + mu_c(3)^2)^(1.5)*sqrt(1 - mu_c(3)^2/(mu_c(1)^2 + mu_c(2)^2 +mu_c(3)^2))), ...
       -(mu_c(2)*mu_c(3))/((mu_c(1)^2 + mu_c(2)^2 + mu_c(3)^2)^(1.5)*sqrt(1 - mu_c(3)^2/(mu_c(1)^2 + mu_c(2)^2 +mu_c(3)^2))), ...
       sqrt(mu_c(1)^2 + mu_c(2)^2)/(mu_c(1)^2 + mu_c(2)^2 + mu_c(3)^2), 0, 0, 0];

    Hxk = [Hk_R; Hk_AZ; Hk_EL];
    h = @(x) [sqrt(x(1)^2 + x(2)^2 + x(3)^2); atan2(x(2),x(1)); pi/2 - acos(x(3)/sqrt(x(1)^2 + x(2)^2 + x(3)^2))]; % Nonlinear measurement model
    zt = noised_obs(idx_meas,2:4)';

    [mu_p, P_p] = kalmanUpdate(zt, Hxk, R_vv, mu_c', P_c, h);
else
    mu_p = mu_c';
    P_p = P_c';
end

fprintf('Mean (Posterior Estimate): \n');
disp(mu_p);
fprintf('Covariance (Posterior Estimate):\n');
disp(P_p);

fprintf('Truth State: \n');
disp([full_ts(idx_prop+1,2:4), full_vts(idx_prop+1,2:4)]);

Xp_cloud = Xm_cloud;
for i = 1:L
    Xp_cloud(i,:) = mvnrnd(mu_p, P_p);
end

% Xprop_truth = [full_ts(idx_prop+1,2:4), full_vts(idx_prop+1,2:4)];

% Plot the cloud and the truth
figure(9)
subplot(2,1,1)
plot3(Xp_cloud(:,1), Xp_cloud(:,2), Xp_cloud(:,3), '.');
hold on;
plot3(Xprop_truth(1), Xprop_truth(2), Xprop_truth(3), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Position')
ylabel('Y Position')
zlabel('Z Position')
title('Position Cloud')

subplot(2,1,2)
plot3(Xp_cloud(:,4), Xp_cloud(:,5), Xp_cloud(:,6), '.');
hold on;
plot3(Xprop_truth(4), Xprop_truth(5), Xprop_truth(6), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Velocity')
ylabel('Y Velocity')
zlabel('Z Velocity')
title('Velocity Cloud')

savefig(gcf, "updatedCloud.fig");
saveas(gcf, "updatedCloud.png");

% Show 2D plane plots of the above figure
figure(10)

subplot(2,3,1)
plot(Xp_cloud(:,1), Xp_cloud(:,2), '.');
hold on;
plot(Xprop_truth(1), Xprop_truth(2), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Position')
ylabel('Y Position')
title('X-Y')

subplot(2,3,2)
plot(Xp_cloud(:,1), Xp_cloud(:,3), '.');
hold on;
plot(Xprop_truth(1), Xprop_truth(3), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Position')
ylabel('Z Position')
title('X-Z')

subplot(2,3,3)
plot(Xp_cloud(:,2), Xp_cloud(:,3), '.');
hold on;
plot(Xprop_truth(2), Xprop_truth(3), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('Y Position')
ylabel('Z Position')
title('Y-Z')

subplot(2,3,4)
plot(Xp_cloud(:,4), Xp_cloud(:,5), '.');
hold on;
plot(Xprop_truth(4), Xprop_truth(5), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Velocity')
ylabel('Y Velocity')
title('Xdot-Ydot')

subplot(2,3,5)
plot(Xp_cloud(:,4), Xp_cloud(:,6), '.');
hold on;
plot(Xprop_truth(4), Xprop_truth(6), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Velocity')
ylabel('Z Velocity')
title('Xdot-Zdot')

subplot(2,3,6)
plot(Xp_cloud(:,5), Xp_cloud(:,6), '.');
hold on;
plot(Xprop_truth(5), Xprop_truth(6), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('Y Velocity')
ylabel('Z Velocity')
title('Ydot-Zdot')

% Set up a loop to propagate and update the EnKF mixture for the remaining
% epochs of the tracklet or half-day observation streak.

% Finding current time step index of tracklet
[idx_meas, c_meas] = find(abs(hdR(:,1) - t_stamp) < 1e-10);
interval = hdR(idx_meas,c_meas) - hdR(idx_meas-1,c_meas);
t_end = hdR(idx_meas + 1, c_meas);
% t_end = hdR(end,1) - interval; % Last observation of tracklet

for to = t_stamp:interval:t_end % Looping over the times of observation for easier propagation
    [mu_c, P_c] = propagate(mu_p, P_p, to, interval, L);

    % Update the mean and covariance with a Kalman update
    tstamp = to + interval;
    fprintf("Timestamp: %1.4f\n", tstamp);
    % [idx_meas, c_meas] = find(partial_ts(:,1) == tstamp);
    [idx_meas, c_meas] = find(abs(partial_ts(:,1) - tstamp) < 1e-10);
    if (idx_meas ~= 0) % i.e. there exists a measurement
        % R_vv = R_t(3*idx_meas-2:3*idx_meas,3*idx_meas-2:3*idx_meas);
        R_vv = [0.05*partial_ts(idx_meas,2), 0, 0; 0 7.2722e-6, 0; 0, 0, 7.2722e-6].^2;
        Hk_R = [mu_c(1)/sqrt(mu_c(1)^2 + mu_c(2)^2 + mu_c(3)^2), ...
            mu_c(2)/sqrt(mu_c(1)^2 + mu_c(2)^2 + mu_c(3)^2), ...
            mu_c(3)/sqrt(mu_c(1)^2 + mu_c(2)^2 + mu_c(3)^2), 0, 0, 0]; % Range linearization
        Hk_AZ = [-mu_c(2)/(mu_c(1)^2 + mu_c(2)^2), mu_c(1)/(mu_c(1)^2 + mu_c(2)^2), 0, 0, 0, 0]; % Azimuth angle linearization
        Hk_EL = [-(mu_c(1)*mu_c(3))/((mu_c(1)^2 + mu_c(2)^2 + mu_c(3)^2)^(1.5)*sqrt(1 - mu_c(3)^2/(mu_c(1)^2 + mu_c(2)^2 +mu_c(3)^2))), ...
            -(mu_c(2)*mu_c(3))/((mu_c(1)^2 + mu_c(2)^2 + mu_c(3)^2)^(1.5)*sqrt(1 - mu_c(3)^2/(mu_c(1)^2 + mu_c(2)^2 +mu_c(3)^2))), ...
            sqrt(mu_c(1)^2 + mu_c(2)^2)/(mu_c(1)^2 + mu_c(2)^2 + mu_c(3)^2), 0, 0, 0];

        Hxk = [Hk_R; Hk_AZ; Hk_EL];
        h = @(x) [sqrt(x(1)^2 + x(2)^2 + x(3)^2); atan2(x(2),x(1)); pi/2 - acos(x(3)/sqrt(x(1)^2 + x(2)^2 + x(3)^2))]; % Nonlinear measurement model
        zt = noised_obs(idx_meas,2:4)';

        [mu_p, P_p] = kalmanUpdate(zt, Hxk, R_vv, mu_c', P_c, h);
    else
        fprintf("Measurement not found at t = %1.4f\n", to);
        mu_p = mu_c';
        P_p = P_c;
    end
end

[idx_truth, c_truth] = find(abs(full_ts - t_end) < 1e-10);
Xprop_truth = [full_ts(idx_truth+1,2:4), full_vts(idx_truth+1,2:4)]'; % Location of Truth

Xp_cloud = Xm_cloud;
for i = 1:L
    Xp_cloud(i,:) = mvnrnd(mu_p, P_p);
end

% Plot the cloud and the truth
figure(5)

subplot(2,1,1)
plot3(Xp_cloud(:,1), Xp_cloud(:,2), Xp_cloud(:,3), '.');
hold on;
plot3(Xprop_truth(1), Xprop_truth(2), Xprop_truth(3), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Position')
ylabel('Y Position')
zlabel('Z Position')
title('Position Cloud')

subplot(2,1,2)
plot3(Xp_cloud(:,4), Xp_cloud(:,5), Xp_cloud(:,6), '.');
hold on;
plot3(Xprop_truth(4), Xprop_truth(5), Xprop_truth(6), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Velocity')
ylabel('Y Velocity')
zlabel('Z Velocity')
title('Velocity Cloud')

savefig(gcf, "updatedCloudFinal.fig");
saveas(gcf, "updatedCloudFinal.png");

% Show 2D plane plots of the above figure
figure(9)

subplot(2,3,1)
plot(Xp_cloud(:,1), Xp_cloud(:,2), '.');
hold on;
plot(Xprop_truth(1), Xprop_truth(2), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Position')
ylabel('Y Position')
title('X-Y')

subplot(2,3,2)
plot(Xp_cloud(:,1), Xp_cloud(:,3), '.');
hold on;
plot(Xprop_truth(1), Xprop_truth(3), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Position')
ylabel('Z Position')
title('X-Z')

subplot(2,3,3)
plot(Xp_cloud(:,2), Xp_cloud(:,3), '.');
hold on;
plot(Xprop_truth(2), Xprop_truth(3), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('Y Position')
ylabel('Z Position')
title('Y-Z')

subplot(2,3,4)
plot(Xp_cloud(:,4), Xp_cloud(:,5), '.');
hold on;
plot(Xprop_truth(4), Xprop_truth(5), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Velocity')
ylabel('Y Velocity')
title('Xdot-Ydot')

subplot(2,3,5)
plot(Xp_cloud(:,4), Xp_cloud(:,6), '.');
hold on;
plot(Xprop_truth(4), Xprop_truth(6), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('X Velocity')
ylabel('Z Velocity')
title('Xdot-Zdot')

subplot(2,3,6)
plot(Xp_cloud(:,5), Xp_cloud(:,6), '.');
hold on;
plot(Xprop_truth(5), Xprop_truth(6), '+', 'MarkerSize', 12, 'LineWidth', 2)
xlabel('Y Velocity')
ylabel('Z Velocity')
title('Ydot-Zdot')

fprintf('Final Mean: \n');
disp(mu_p);
fprintf('\nFinal Covariance: \n');
disp(P_p);
fprintf('\nTruth: \n')
disp(Xprop_truth);
%}

%% Functions
function [dX_coeffs] = polyDeriv(X_coeffs)
    
    dX_coeffs = zeros(1, length(X_coeffs)-1);
    for j = length(X_coeffs):-1:2
        dX_coeffs(length(X_coeffs)+1-j) = X_coeffs(length(X_coeffs)+1-j)*(j-1);
    end
end

function [Xfit] = stateEstCloud(pf, nfit, theta_f, R_f, obTr, tdiff, low_lim, up_lim)
    noised_obs = obTr;

    R_t = zeros(3*length(noised_obs(:,1)),1); % We shall diagonalize this later
    mu_t = zeros(3*length(noised_obs(:,1)),1);

    load("partial_ts.mat"); % Noiseless observation data
    % dist2km = 384400; % Kilometers per non-dimensionalized distance

    for i = 1:length(obTr(:,1))
        mu_t(3*(i-1)+1:3*(i-1)+3, 1) = [partial_ts(i,2); partial_ts(i,3); partial_ts(i,4)];
        R_t(3*(i-1)+1:3*(i-1)+3, 1) = [(R_f*partial_ts(i,2))^2; (theta_f*4.84814e-6)^2; (theta_f*4.84814e-6)^2];
    end

    R_t = diag(R_t);
    data_vec = mvnrnd(mu_t, R_t)';

    for i = 1:length(noised_obs(:,1))
        noised_obs(i,3:4) = data_vec(3*(i-1)+2:3*(i-1)+3,1);
        noised_obs(i,2) = unifrnd(low_lim, up_lim);
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

    Xfit = [X_fit(end,1); Y_fit(end,1); Z_fit(end,1); Xdot_fit(end,1); Ydot_fit(end,1); Zdot_fit(end,1)];
end

function [reo_topo] = getObserverPos(t_stamp)
    % First step: Obtain X_{eo}^{ECI} 
    obs_lat = 30.618963;
    obs_lon = -96.339214;
    elevation = 103.8;

    UTC_vec_orig = [2024	5	3	2	41	15.1261889999956]; % Initial UTC vector at t_stamp = 0
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

% Propagation step for EnKF
function [mu_m, P_m] = propagate(mu, P, t_int, interval, L)
    P = (P + P')/2;
    Xcloud = zeros(L,length(mu)); % Draw from a particle cloud
    for i = 1:L
        Xcloud(i,:) = mvnrnd(mu,P);
    end
    
    Xm_cloud = Xcloud;
    for i = 1:length(Xcloud(:,1))
        % First, convert from X_{ot} in the topocentric frame to X_{bt} in the
        % synodic frame.
        Xbt = backConvertSynodic(Xcloud(i,:)', t_int);

        % Next, propagate each X_{bt} in your particle cloud by a single time 
        % step and convert back to the topographic frame.
        % Call ode45()
        opts = odeset('Events', @termSat);
        [~,X] = ode45(@cr3bp_dyn, [0 interval], Xbt, opts); % Assumes termination event (i.e. target enters LEO)
        Xm_bt = X(end,:)';
        Xm_cloud(i,:) = convertToTopo(Xm_bt, t_int + interval);
    end

    mu_m = mean(Xm_cloud);
    P_m = cov(Xm_cloud);
    % P_m = (Xm_cloud - mu_m)' * (Xm_cloud - mu_m) / (L - 1);
    
    %{
    mu_m = zeros(1,length(mu));
    Xc = Xm_cloud; % Variable for re-centering Xcloud
    % P_m = zeros(length(mu), length(mu));
    for i = 1:length(Xm_cloud(1,:))
        mu_m(1,i) = mean(Xm_cloud(:,i));
        Xc(:,i) = Xm_cloud(:,i) - mu_m(1,i)*ones(length(Xm_cloud(:,1)),1);
    end

    P_m = (1/(L-1)) * (Xc)' * Xc;
    % P_m = cov(Xm_cloud);
    %}
end

% Kalman update for a single time step
function [mu_p, P_p] = kalmanUpdate(zk, H, R, mu_m, P_m, h)
    % fprintf('zk = \n');
    % disp(zk);
    % fprintf('H = \n');
    % disp(H);
    % fprintf('mu_m = \n');
    % disp(mu_m);
    K_k = P_m*H'*(H*P_m*H' + R)^(-1); % Kalman gain
    % inn = zk - H*mu_m; % Innovation
    % mu_p = mu_m + K_k*inn;

    % h = @(x) [sqrt(x(1)^2 + x(2)^2 + x(3)^2); atan2(x(2),x(1)); pi/2 - acos(x(3)/sqrt(x(1)^2 + x(2)^2 + x(3)^2))]; % Nonlinear measurement model
    mu_p = mu_m + K_k*(zk - h(mu_m));
    P_p = (eye(length(mu_m)) - K_k*H)*P_m;

    % Ensure Kalman update is symmetric
    P_p = (P_p + P_p')/2;
end

% Converts X_ot in the topographic frame to X_bt in the CR3BP synodic frame
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