% Load necessary .mat files
load('filter_residuals.mat', 'resids');
load('mean_corrections.mat', 'z_corrs');

dist2km = 384400; % Kilometers per non-dimensionalized distance
time2hr = 4.342*24; % Hours per non-dimensionalized time
vel2kms = dist2km/(time2hr*60*60); % Kms per non-dimensionalized velocity

figInnov = open("./Simulations/Simulation 10 (Pure EnKF)/Innovations.fig");
figInnov = gcf;

axObjs = figInnov.Children;
dataObjs = axObjs.Children;

XData = cell(3,1);
YData = cell(3,1);

for i = 1:3
    % Get the line object from each axes (the plot data)
    lineObj = axObjs(i).Children;
    
    % Extract X and Y data from the line object
    XData{i} = lineObj.XData;
    YData{i} = lineObj.YData;
end

ResR = YData{1}; ResAZ = YData{2}; ResEL = YData{3};
noiseMat = [ResR; ResAZ; ResEL];

% Extract index of beginning of second pass
idx_2b = 0;

i = 1;
while(1)
    if(ResAZ(i+1) ~= 0 && ResAZ(i) == 0)
        idx_2b = i + 1;
        break;
    else
        i = i + 1;
    end
end

% Extract index of end of second pass
idx_2e = idx_2b;

i = idx_2b;
while(1)
    if(ResAZ(i+1) == 0 && ResAZ(i) ~= 0)
        idx_2e = i;
        break;
    else
        i = i + 1;
    end
end

testSlice = noiseMat(1, idx_2b:idx_2e);

% Define window size and offset

k = 1; % Offset size
noiseStat = zeros(2,21);
noiseStat(1,:) = 1:length(noiseStat(1,:));

for T = 1:length(noiseStat(1,:))
    nSum = 0;
    for i = 1:T
        v1 = testSlice(:,i);
        v2 = testSlice(:,i+k);
        nSum = nSum + v1'*v2;
    end
    
    nSum = nSum/T;
    noiseStat(2,T) = nSum;
end

figure(2)
plot(noiseStat(1,:), noiseStat(2,:))
xlabel('Size of T')
ylabel('Noise Sum')
title('Whiteness Test (k = 1)')

noiseMatNonzero = noiseMat(:, any(noiseMat ~= 0, 1));

% Define window size and offset

k = 2; % Offset size
noiseStat = zeros(2,140);
noiseStat(1,:) = 1:length(noiseStat(1,:));

for T = 1:length(noiseStat(1,:))
    nSum = 0;
    for i = 1:T
        v1 = noiseMatNonzero(:,i);
        v2 = noiseMatNonzero(:,i+k);
        nSum = nSum + v1'*v2;
    end
    
    nSum = nSum/T;
    noiseStat(2,T) = nSum;
end

figure(3)
plot(noiseStat(1,:), noiseStat(2,:))
xlabel('Size of T')
ylabel('Noise Sum')
title('Whiteness Test (k = 2)')

load("innov_seq.mat", "innov_comps");

%{
% Load truth propagation .mat files
load('full_ts_highTol.mat','full_ts'); 
full_ts_highTol = full_ts;
load('full_ts_lowTol.mat', 'full_ts');
full_ts_lowTol = full_ts;
load('full_ts_recursive.mat', 'full_ts');
full_ts_rec = full_ts;

load('full_vts_highTol.mat','full_vts'); 
full_vts_highTol = full_vts;
load('full_vts_lowTol.mat', 'full_vts');
full_vts_lowTol = full_vts;
load('full_vts_recursive.mat', 'full_vts');
full_vts_rec = full_vts;


R_resids = squeeze(resids(1,:,:)); AZ_resids = squeeze(resids(2,:,:)); EL_resids = squeeze(resids(3,:,:));
x_corr = squeeze(mu_corrs(1,:,:)); y_corr = squeeze(mu_corrs(2,:,:)); z_corr = squeeze(mu_corrs(3,:,:));
xdot_corr = squeeze(mu_corrs(4,:,:)); ydot_corr = squeeze(mu_corrs(5,:,:)); zdot_corr = squeeze(mu_corrs(6,:,:));
ct_corr = squeeze(mu_corrs(:,end,:));

figure(1)

subplot(2,3,1)
plot(full_ts_lowTol(:,1), full_ts_highTol(:,2))
hold on;
plot(full_ts_lowTol(:,1), full_ts_lowTol(:,2))
hold on;
plot(full_ts_lowTol(:,1), full_ts_rec(:,2))
xlabel('Time (non-dim)')
ylabel('X Position')
title('X')
legend('High Tol','Low Tol','Recursive')
% xlim([0.35 0.4])

subplot(2,3,2)
plot(full_ts_lowTol(:,1), full_ts_highTol(:,3))
hold on;
plot(full_ts_lowTol(:,1), full_ts_lowTol(:,3))
hold on;
plot(full_ts_lowTol(:,1), full_ts_rec(:,3))
xlabel('Time (non-dim)')
ylabel('Y Position')
title('Y')
legend('High Tol','Low Tol','Recursive')
% xlim([0.35 0.4])

subplot(2,3,3)
plot(full_ts_lowTol(:,1), full_ts_highTol(:,4))
hold on;
plot(full_ts_lowTol(:,1), full_ts_lowTol(:,4))
hold on;
plot(full_ts_lowTol(:,1), full_ts_rec(:,4))
xlabel('Time (non-dim)')
ylabel('Z Position')
title('Z')
legend('High Tol','Low Tol','Recursive')
% xlim([0.35 0.4])

subplot(2,3,4)
plot(full_vts_lowTol(:,1), full_vts_highTol(:,2))
hold on;
plot(full_vts_lowTol(:,1), full_vts_lowTol(:,2))
hold on;
plot(full_vts_lowTol(:,1), full_vts_rec(:,2))
xlabel('Time (non-dim)')
ylabel('Xdot Velocity')
title('V_x')
legend('High Tol','Low Tol','Recursive')
% xlim([0.35 0.4])

subplot(2,3,5)
plot(full_vts_lowTol(:,1), full_vts_highTol(:,3))
hold on;
plot(full_vts_lowTol(:,1), full_vts_lowTol(:,3))
hold on;
plot(full_vts_lowTol(:,1), full_vts_rec(:,3))
xlabel('Time (non-dim)')
ylabel('Ydot Velocity')
title('V_y')
legend('High Tol','Low Tol','Recursive')
% xlim([0.35 0.4])

subplot(2,3,6)
plot(full_vts_lowTol(:,1), full_vts_highTol(:,4))
hold on;
plot(full_vts_lowTol(:,1), full_vts_lowTol(:,4))
hold on;
plot(full_vts_lowTol(:,1), full_vts_rec(:,4))
xlabel('Time (non-dim)')
ylabel('Zdot Velocity')
title('V_z')
legend('High Tol','Low Tol','Recursive')
% xlim([0.35 0.4])

savefig(gcf, 'prop2.fig')

figure(2)

subplot(2,3,1)
plot(full_ts_lowTol(:,1), abs(full_ts_highTol(:,2) - full_ts_lowTol(:,2)))
hold on;
plot(full_ts_lowTol(:,1), abs(full_ts_rec(:,2) - full_ts_lowTol(:,2)))
xlabel('Time (non-dim)')
ylabel('X Position')
title('X Difference')
legend('Low - High','Low - Recursive')
% xlim([0.35 0.4])

subplot(2,3,2)
plot(full_ts_lowTol(:,1), abs(full_ts_highTol(:,3) - full_ts_lowTol(:,3)))
hold on;
plot(full_ts_lowTol(:,1), abs(full_ts_rec(:,3) - full_ts_lowTol(:,3)))
xlabel('Time (non-dim)')
ylabel('Y Position')
title('Y Difference')
legend('Low - High','Low - Recursive')
% xlim([0.35 0.4])

subplot(2,3,3)
plot(full_ts_lowTol(:,1), abs(full_ts_highTol(:,4) - full_ts_lowTol(:,4)))
hold on;
plot(full_ts_lowTol(:,1), abs(full_ts_rec(:,4) - full_ts_lowTol(:,4)))
xlabel('Time (non-dim)')
ylabel('Z Position')
title('Z Difference')
legend('Low - High','Low - Recursive')
% xlim([0.35 0.4])

subplot(2,3,4)
plot(full_vts_lowTol(:,1), abs(full_vts_highTol(:,2) - full_vts_lowTol(:,2)))
hold on;
plot(full_vts_lowTol(:,1), abs(full_vts_rec(:,2) - full_vts_lowTol(:,2)))
xlabel('Time (non-dim)')
ylabel('Xdot Velocity')
title('V_x Difference')
legend('Low - High','Low - Recursive')
% xlim([0.35 0.4])

subplot(2,3,5)
plot(full_vts_lowTol(:,1), abs(full_vts_highTol(:,3) - full_vts_lowTol(:,3)))
hold on;
plot(full_vts_lowTol(:,1), abs(full_vts_rec(:,3) - full_vts_lowTol(:,3)))
xlabel('Time (non-dim)')
ylabel('Ydot Velocity')
title('V_y Difference')
legend('Low - High','Low - Recursive')
% xlim([0.35 0.4])

subplot(2,3,6)
plot(full_vts_lowTol(:,1), abs(full_vts_highTol(:,4) - full_vts_lowTol(:,4)))
hold on;
plot(full_vts_lowTol(:,1), abs(full_vts_rec(:,4) - full_vts_lowTol(:,4)))
xlabel('Time (non-dim)')
ylabel('Zdot Velocity')
title('V_z Difference')
legend('Low - High','Low - Recursive')
% xlim([0.35 0.4])

savefig(gcf, 'propDiff.fig')

% Here, we run ode45() multiple times with two different tolerances to
% obtain "uncertainty point clouds"

% Define initial conditions
mu = 1.2150582e-2;
x0 = [0.5-mu, 0.0455, 0, -0.5, 0.5, 0.0]'; % Sample Starting Point

tstamp = 0; % For long term trajectories 
% tstamp = 0.3570;
end_t = 59*6.25e-3 + 1e-11; % Add small epsilon to ensure that we have an endpoint
tspan = 0:6.25e-3:end_t; % For our modified trajectory 

% Call ode45() with higher tolerance

% opts = odeset('AbsTol',1e-6,'RelTol',1e-6,'Events', @termSat);

rbht = zeros(500, 3); vbht = zeros(500, 3);
rblt = zeros(500, 3); vblt = zeros(500, 3);

for i = 1:500
    opts = odeset('Events', @termSat);
    [~,dx_dt] = ode45(@cr3bp_dyn, tspan, x0, opts); % Assumes termination event (i.e. target enters LEO)
    
    rbht(i,:) = dx_dt(end,1:3); % Position evolutions from barycenter
    vbht(i,:) = dx_dt(end,4:6); % Velocity evolutions from barycenter

    opts = odeset('AbsTol',1e-6,'RelTol',1e-6,'Events', @termSat);
    [~,dx_dt] = ode45(@cr3bp_dyn, tspan, x0, opts); % Assumes termination event (i.e. target enters LEO)

    rblt(i,:) = dx_dt(end,1:3); % Position evolutions from barycenter
    vblt(i,:) = dx_dt(end,4:6); % Velocity evolutions from barycenter
end

figure(3)
subplot(2,3,1)
plot(full_ts_lowTol(:,1), abs(full_ts_rec(:,2) - full_ts_lowTol(:,2)))
xlabel('Time (non-dim)')
ylabel('X Position')
title('X Difference')
legend('Low - Recursive')
% xlim([0.35 0.4])

subplot(2,3,2)
plot(full_ts_lowTol(:,1), abs(full_ts_rec(:,3) - full_ts_lowTol(:,3)))
xlabel('Time (non-dim)')
ylabel('Y Position')
title('Y Difference')
legend('Low - Recursive')
% xlim([0.35 0.4])

subplot(2,3,3)
plot(full_ts_lowTol(:,1), abs(full_ts_rec(:,4) - full_ts_lowTol(:,4)))
xlabel('Time (non-dim)')
ylabel('Z Position')
title('Z Difference')
legend('Low - Recursive')
% xlim([0.35 0.4])

subplot(2,3,4)
plot(full_vts_lowTol(:,1), abs(full_vts_rec(:,2) - full_vts_lowTol(:,2)))
xlabel('Time (non-dim)')
ylabel('Xdot Velocity')
title('V_x Difference')
legend('Low - Recursive')
% xlim([0.35 0.4])

subplot(2,3,5)
plot(full_vts_lowTol(:,1), abs(full_vts_rec(:,3) - full_vts_lowTol(:,3)))
xlabel('Time (non-dim)')
ylabel('Ydot Velocity')
title('V_y Difference')
legend('Low - Recursive')
% xlim([0.35 0.4])

subplot(2,3,6)
plot(full_vts_lowTol(:,1), abs(full_vts_rec(:,4) - full_vts_lowTol(:,4)))
xlabel('Time (non-dim)')
ylabel('Zdot Velocity')
title('V_z Difference')
legend('Low - Recursive')
% xlim([0.35 0.4])

figure(4)
subplot(2,3,1)
plot(full_ts_lowTol(:,1), abs(full_ts_rec(:,2) - full_ts_highTol(:,2)))
xlabel('Time (non-dim)')
ylabel('X Position')
title('X Difference')
legend('High - Recursive')
% xlim([0.35 0.4])

subplot(2,3,2)
plot(full_ts_lowTol(:,1), abs(full_ts_rec(:,3) - full_ts_highTol(:,3)))
xlabel('Time (non-dim)')
ylabel('Y Position')
title('Y Difference')
legend('High - Recursive')
% xlim([0.35 0.4])

subplot(2,3,3)
plot(full_ts_lowTol(:,1), abs(full_ts_rec(:,4) - full_ts_highTol(:,4)))
xlabel('Time (non-dim)')
ylabel('Z Position')
title('Z Difference')
legend('High - Recursive')
% xlim([0.35 0.4])

subplot(2,3,4)
plot(full_vts_lowTol(:,1), abs(full_vts_rec(:,2) - full_vts_highTol(:,2)))
xlabel('Time (non-dim)')
ylabel('Xdot Velocity')
title('V_x Difference')
legend('High - Recursive')
% xlim([0.35 0.4])

subplot(2,3,5)
plot(full_vts_lowTol(:,1), abs(full_vts_rec(:,3) - full_vts_highTol(:,3)))
xlabel('Time (non-dim)')
ylabel('Ydot Velocity')
title('V_y Difference')
legend('High - Recursive')
% xlim([0.35 0.4])

subplot(2,3,6)
plot(full_vts_lowTol(:,1), abs(full_vts_rec(:,4) - full_vts_highTol(:,4)))
xlabel('Time (non-dim)')
ylabel('Zdot Velocity')
title('V_z Difference')
legend('High - Recursive')
% xlim([0.35 0.4])


figure(5)
subplot(2,3,1)
hold on;
plot(dist2km*rbht(:,1), dist2km*rbht(:,2), '.', 'MarkerSize', 10, 'LineWidth', 3);
hold on;
plot(dist2km*rblt(:,1), dist2km*rblt(:,2), '.', 'MarkerSize', 10, 'LineWidth', 3);
title('X-Y');
xlabel('X (km.)');
ylabel('Y (km.)');
legend('Possible Position Truths');

subplot(2,3,2)
hold on;
plot(dist2km*rbht(:,1), dist2km*rbht(:,3), '.', 'MarkerSize', 10, 'LineWidth', 3);
hold on;
plot(dist2km*rblt(:,1), dist2km*rblt(:,3), '.', 'MarkerSize', 10, 'LineWidth', 3);
title('X-Z');
xlabel('X (km.)');
ylabel('Z (km.)');
legend('Possible Position Truths');

subplot(2,3,3)
hold on;
plot(dist2km*rbht(:,2), dist2km*rbht(:,3), '.', 'MarkerSize', 10, 'LineWidth', 3);
hold on;
plot(dist2km*rblt(:,2), dist2km*rblt(:,3), '.', 'MarkerSize', 10, 'LineWidth', 3);
title('Y-Z');
xlabel('Y (km.)');
ylabel('Z (km.)');
legend('Possible Position Truths');

subplot(2,3,4)
hold on;
plot(vel2kms*vbht(:,1), vel2kms*vbht(:,2), '.', 'MarkerSize', 15, 'LineWidth', 3);
hold on;
plot(vel2kms*vblt(:,1), vel2kms*vblt(:,2), '.', 'MarkerSize', 15, 'LineWidth', 3);
title('Xdot - Ydot');
xlabel('Vx (km/s)');
ylabel('Vy (km/s)');
legend('Possible Velocity Truths');

subplot(2,3,5)
hold on;
plot(vel2kms*vbht(:,1), vel2kms*vbht(:,3), '.', 'MarkerSize', 15, 'LineWidth', 3);
hold on;
plot(vel2kms*vblt(:,1), vel2kms*vblt(:,3), '.', 'MarkerSize', 15, 'LineWidth', 3);
title('Xdot - Zdot');
xlabel('Vx (km/s)');
ylabel('Vz (km/s)');
legend('Possible Velocity Truths');

subplot(2,3,6)
hold on;
plot(vel2kms*vbht(:,2), vel2kms*vbht(:,3), '.', 'MarkerSize', 15, 'LineWidth', 3);
hold on;
plot(vel2kms*vblt(:,2), vel2kms*vblt(:,3), '.', 'MarkerSize', 15, 'LineWidth', 3);
title('Ydot - Zdot');
xlabel('Vy (km/s)');
ylabel('Vz (km/s)');
legend('Possible Velocity Truths');
%}