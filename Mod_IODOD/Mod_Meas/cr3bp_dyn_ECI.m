% Define initial conditions
mu = 1.2150582e-2;
x0 = [0.5-mu, 0.0455, 0, -0.5, 0.5, 0.0]'; % Sample Starting Point
% x0 = [-1.00506 0 0 -0.5 0.5 0.0]'; % L3 Lagrange Point
% x0 = [0.5-mu, sqrt(3/4), 0, 0, 0, 0]'; % L4 Lagrange Point

% Lagrange points that we may not use
% x0 = [0.836915 0 0 0 0 0]'; % L1 Lagrange Point
% x0 = [1.15568 0 0 0 0 0]'; % L2 Lagrange Point
% x0 = [0.5-mu, -sqrt(3/4), 0, 0, 0, 0]'; % L5 Lagrange Point

% x0 = [0.836915 0 0 -0.02 0.02 0]'; 

% Coordinate system conversions
dist2km = 384400; % Kilometers per non-dimensionalized distance
time2hr = 4.342*24; % Hours per non-dimensionalized time
vel2kms = dist2km/(time2hr*60*60); % Kms per non-dimensionalized velocity

% Define time span
tstamp = 0; % For long term trajectories 
% tstamp = 0.3570;
end_t = 2 - tstamp;
tspan = 0:6.25e-3:end_t; % For our modified trajectory 

% Call ode45()

opts = odeset('Events', @termSat);
% [t,dx_dt] = ode45(@cr3bp_dyn, tspan, x0, opts); % Assumes termination event (i.e. target enters LEO)

dx_dt = zeros(length(tspan), length(x0)); dx_dt(1,:) = x0;
t = zeros(length(tspan),1);

for i = 1:(length(tspan)-1)
   % [t_tmp,dx_dt_tmp] = ode45(@cr3bp_dyn, [tspan(i) tspan(i+1)], x0, opts); % Assumes termination event (i.e. target enters LEO)
   [t_tmp,dx_dt_tmp] = ode45(@cr3bp_dyn, [0, tspan(i+1) - tspan(i)], x0, opts); 
   x0 = dx_dt_tmp(end,:); dx_dt(i+1,:) = x0; t(i+1) = tspan(i+1);
end

% [t,dx_dt] = ode45(@cr3bp_dyn, tspan, x0); % Assumes no termination event

rb = dx_dt(:,1:3); % Position evolutions from barycenter
vb = dx_dt(:,4:6); % Velocity evolutions from barycenter
rbe = [-mu, 0, 0]'; % Position vector relating center of earth to barycenter

% Insert code for obtaining vector between center of Earth and observer

obs_lat = 30.618963;
obs_lon = -96.339214;
elevation = 103.8;

% dtLCL = datetime('now', 'TimeZone','local'); % Current Local Time
% dtUTC = datetime(dtLCL, 'TimeZone','Z');     % Current UTC Time
% UTC_vec = datevec(dtUTC); % Convert to vector

UTC_vec = [2024	5	3	2	41	15.1261889999956];
t_add_dim = tstamp * (4.342);
UTC_vec = datevec(datetime(UTC_vec) + t_add_dim);

% reo_dim = lla2eci([obs_lat obs_lon elevation], UTC_vec); % Position vector between observer and center of Earth in meters
% reo_nondim = reo_dim/(1000*384400);

rem = [1, 0, 0]'; % Earth center - moon center 

reo_nondim = zeros(length(t),3);
veo_nondim = zeros(length(t),3); % Array for non-dimensionalized EO velocity vectors

rot = []; % Observer - Target 
rom = []; % Observer - Moon Center
vot = []; % Observer - Target Velocity

for i = 1:length(rb(:,1))
    t_add_nondim = t(i) - t(1); % Time since first point of orbit
    t_add_dim = t_add_nondim * (4.342); % Conversion to dimensionalized time
    delt_add_dim = t_add_dim - 1/86400; 

    updated_UTCtime = datetime(UTC_vec) + t_add_dim;
    updated_UTCvec = datevec(updated_UTCtime);

    delt_updatedUTCtime = datetime(UTC_vec) + delt_add_dim;
    delt_updatedUTCvec = datevec(delt_updatedUTCtime);

    % reo_dim = lla2ecef([obs_lat obs_lon elevation]); % Position vector between observer and center of Earth in meters
    reo_dim = lla2eci([obs_lat obs_lon, elevation], updated_UTCvec);
    delt_reodim = lla2eci([obs_lat obs_lon, elevation], delt_updatedUTCvec);
    veo_dim = reo_dim - delt_reodim; 
    
    R_z = [cos(t_add_nondim + tstamp), -sin(t_add_nondim + tstamp), 0; sin(t_add_nondim + tstamp), cos(t_add_nondim + tstamp), 0; 0, 0, 1];
    dRz_dt = [-sin(t_add_nondim + tstamp), -cos(t_add_nondim + tstamp), 0; cos(t_add_nondim + tstamp), -sin(t_add_nondim + tstamp), 0; 0, 0, 0];

    reo_nondim(i,:) = reo_dim'/(1000*384400); % Conversion to non-dimensional units and ECI frame
    veo_nondim(i,:) = veo_dim'*(4.342*86400)/(1000*384400);

    rot(i,:) = -reo_nondim(i,:)' + R_z*(-rbe + rb(i,:)');
    rom(i,:) = -reo_nondim(i,:)' + R_z*rem;
    vot(i,:) = -veo_nondim(i,:)' + R_z*(vb(i,:)') + dRz_dt*(-rbe + rb(i,:)');
end

% Plot the trajectory
figure(1)
plot3(dx_dt(:,1), dx_dt(:,2), dx_dt(:,3));
xlabel('x');
ylabel('y');
zlabel('z');
title('CR3BP Trajectory');
grid on;
hold on;

% Plot masses
plot3(-mu, 0, 0, 'ko')
labels = {'Earth'};
text(-mu, 0, 0, labels,'VerticalAlignment','bottom','HorizontalAlignment','right')

plot3(1-mu, 0, 0, 'go')
labels = {'Moon'};
text(1-mu, 0, 0, labels,'VerticalAlignment','bottom','HorizontalAlignment','right')
% xlim([0.95 1.05])

% plot3(reo_nondim(:,1), reo_nondim(:,2), reo_nondim(:,3), 'r+')
% labels = {'Observer'};
% text(reo_nondim(1), reo_nondim(2), reo_nondim(3), labels,'VerticalAlignment','bottom','HorizontalAlignment','right')

[ReX, ReY, ReZ] = sphere;

% Here, we nondimensionalize Earth's radius by the distance between the
% Earth and Moon centers

ReX = 6371/384400 * ReX;
ReY = 6371/384400 * ReY;
ReZ = 6371/384400 * ReZ;
surf(ReX, ReY, ReZ)

% xlim([-0.03, 0.03])
% ylim([-0.03, 0.03])
% zlim([-0.03, 0.03])

savefig(gcf, 'trajectory_ECI.fig')
saveas(gcf, 'trajectory_ECI.png')

% Plot the position parametrically w.r.t. time
figure(2)
subplot(3,1,1)
plot(t, dx_dt(:,1), 'r-')
xlabel('Time')
ylabel('x-Position')
title('CB3RP x-Evolution')

subplot(3,1,2)
plot(t, dx_dt(:,2), 'g-')
xlabel('Time')
ylabel('y-Position')
title('CB3RP y-Evolution')

subplot(3,1,3)
plot(t, dx_dt(:,3), 'b-')
xlabel('Time')
ylabel('z-Position')
title('CB3RP z-Evolution')
saveas(gcf, 'posEvolution.png')

% Plot position evolutions between observer and target
figure(3)
plot3(rot(:,1), rot(:,2), rot(:,3), 'g-');
xlabel('x');
ylabel('y');
zlabel('z');
title('Observer - Target Trajectory');
grid on;
hold on;

% Plot observer
plot3(0, 0, 0, 'ro')
labels = {'Observer'};
text(0, 0, 0, labels,'VerticalAlignment','bottom','HorizontalAlignment','right')

savefig(gcf, 'rot_trajectory_ECI.fig')
saveas(gcf, 'rot_trajectory_ECI.png')

% Before we obtain AZ and EL quantities, we must convert our
% observer-target vector into a topocentric frame.

rot_topo = zeros(length(t),length(rot(1,:)));
rom_topo = zeros(length(t),length(rot(1,:)));
vot_topo = zeros(length(t),length(rot(1,:)));

for i = 1:length(t)
    % Step 1: Find the unit vectors governing this topocentric frame
    z_hat_topo = reo_nondim(i,:)/norm(reo_nondim(i,:));

    x_hat_topo_unorm = cross(z_hat_topo, [0, 0, 1]'); % We choose a reference vector 
    x_hat_topo = x_hat_topo_unorm/norm(x_hat_topo_unorm); % Remember to normalize
    % such as the North Pole, we have several choices regarding this

    y_hat_topo_unorm = cross(x_hat_topo, z_hat_topo);
    y_hat_topo = y_hat_topo_unorm/norm(y_hat_topo_unorm); % Remember to normalize

    % Step 2: Convert all of the components of 'rot' from our aligned reference
    % frames to this new topocentric frame.
    
    rot_topo(i,:) = [dot(rot(i,:), x_hat_topo), dot(rot(i,:), y_hat_topo), dot(rot(i,:), z_hat_topo)];
    rom_topo(i,:) = [dot(rom(i,:), x_hat_topo), dot(rom(i,:), y_hat_topo), dot(rom(i,:), z_hat_topo)];

    % Step 3: Handle the time derivatives of vot_topo = d/dt (rot_topo)
    R_topo = [x_hat_topo; y_hat_topo; z_hat_topo]; % DCM relating ECI to topocentric coordinate frame
    dmag_dt = dot(reo_nondim(i,:), veo_nondim(i,:))/norm(reo_nondim(i,:)); % How the magnitude of r_eo changes w.r.t. time
    
    zhat_dot_topo = (veo_nondim(i,:)*norm(reo_nondim(i,:)) - reo_nondim(i,:)*dmag_dt)/(norm(reo_nondim(i,:)))^2;
    xhat_dot_topo = cross(zhat_dot_topo, [0, 0, 1]')/norm(cross(z_hat_topo, [0,0,1]')) - dot(x_hat_topo, cross(zhat_dot_topo, [0, 0, 1]'))*x_hat_topo;
    yhat_dot_topo = (cross(xhat_dot_topo, z_hat_topo) + cross(x_hat_topo, zhat_dot_topo))/norm(cross(x_hat_topo, z_hat_topo)) - dot(y_hat_topo, cross(xhat_dot_topo, z_hat_topo) + cross(x_hat_topo, zhat_dot_topo))*y_hat_topo;

    dA_dt = [xhat_dot_topo; yhat_dot_topo; zhat_dot_topo];

    % vot_topo(i,:) = [dot(vot(i,:), x_hat_topo), dot(vot(i,:), y_hat_topo), dot(vot(i,:), z_hat_topo)]' + dA_dt*rot_topo(i,:)';
    vot_topo(i,:) = R_topo*vot(i,:)' + dA_dt*rot(i,:)';
end

% Due to not being able to see targets behindthe moon, design function such 
% that if the rot_topo vector passes through the Moon, then data for that 
% time step is considered invalid.

rot_valid = [];
vot_valid = [];
t_valid = [];

j = 0;
Rm = 1740/384400; % Nondimensionalized radius of the moon

for i = 1:length(t)
    if (norm(cross(rot_topo(i,:), rom_topo(i,:)))/norm(rot_topo(i,:)) > Rm)
        j = j + 1;
        t_valid(j,1) = t(i);
        rot_valid(j,:) = rot_topo(i,:);
        vot_valid(j,:) = vot_topo(i,:);
    end
end

% Convert observer - target position vectors into range, azimuth, and
% elevation quantities

Rho = zeros(length(rot_valid(:,1)),1);
AZ = zeros(length(rot_valid(:,1)),1);
EL = zeros(length(rot_valid(:,1)),1);

for i = 1:length(rot_valid(:,1))
    Rho(i,1) = sqrt(rot_valid(i,1)^2 + rot_valid(i,2)^2 + rot_valid(i,3)^2);
    AZ(i,1) = atan2(rot_valid(i,2), rot_valid(i,1));
    EL(i,1) = pi/2 - acos(rot_valid(i,3)/Rho(i,1));
end

% Last Step: Due to elevation angle constraints, all t, Rho, AZ, EL data
% for which EL < 0 is considered invalid and should be discarded

full_ts = [t_valid, Rho, AZ, EL]; % Full augmented time-series vector
partial_ts_ECI = full_ts; % You start with a copy but work your way down

for i = 1:length(t_valid(:,1))
    if (full_ts(i, 4) < 0)
        partial_ts_ECI = partial_ts_ECI(~any(partial_ts_ECI == [t_valid(i), Rho(i), AZ(i), EL(i)], 2), :);
    end
end

% Plot the spherical coordinates of the observer parametrically w.r.t. time
figure(4)
subplot(3,1,1)
plot(partial_ts_ECI(:,1), partial_ts_ECI(:,2), 'ro')
xlabel('Time')
ylabel('Range (non-dim)')
xlim([-tstamp t(end)])
title('Observer Range Measurements (Ideal)')

subplot(3,1,2)
plot(partial_ts_ECI(:,1), partial_ts_ECI(:,3), 'go')
xlabel('Time')
ylabel('Azimuth Angle (rad)')
xlim([-tstamp t(end)])
title('Observer Azimuth Angle Measurements (Ideal)')

subplot(3,1,3)
plot(partial_ts_ECI(:,1), partial_ts_ECI(:,4), 'bo')
xlabel('Time')
ylabel('Elevation Angle (rad)')
xlim([-tstamp t(end)])
title('Observer Elevation Angle Measurements (Ideal)')
saveas(gcf, 'observations_ECI.png')

partial_ts = partial_ts_ECI;
save('partial_ts.mat', 'partial_ts');

full_ts = [t, rot_topo];
save('full_ts.mat', "full_ts");

% Construct and save a similar data file for velocity data
full_vts = [t, vot_topo];
save('full_vts.mat', 'full_vts')