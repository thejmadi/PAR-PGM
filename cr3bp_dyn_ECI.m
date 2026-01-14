clear all; close all;
save_loc = "D:/PythonProjects/EDP/PGM/ParticleFusionTest/12_15_25_meeting/Matlab2Python/Test9/OrbitData/Agent3a";
%save_loc = "D:/PythonProjects/EDP/PGM_Git/PAR-PGM";
dynamics = "CR3BP"; % 2Body or CR3BP
% Observer locations
% College Station
obs_lat = 30.618963;
obs_lon = -96.339214;
obs_lata = 30.618963;
obs_lona = -96.339214;
% Buenos Aires
obs_latb = -34.612979;
obs_lonb = -58.453656;
elevation = 103.8;

% 2-Body Initial Conditions
if (dynamics == "2 Body")
    mu = 398600.4418; % km^3/s^2
    R_earth = 6378; % km
    r0 = [6778, 0, 0]; % km
    v0 = [0, 7.668, 6.02]; % km/s
    
    % 2-Body Non-dimensionalizations
    %normalization_quantities.dist2km = norm(r0);
    %normalization_quantities.vel2kms = norm(v0);
    %normalization_quantities.time2hr = (norm(r0)/norm(v0))/3600;
    %normalization_quantities.mu = mu / (norm(v0)^2 * norm(r0));
    normalization_quantities = Dynamics.normalize_2Body(r0, v0, mu);
    x0 = [r0/normalization_quantities.dist2km, v0/normalization_quantities.vel2kms];
    dynamics_model = @(t, x) Dynamics.two_body_dyn(t, x, normalization_quantities.mu);
end

% CR3BP Initial Conditions
if (dynamics == "CR3BP")
    normalization_quantities.mu = 1.2150582e-2;
    %x0 = [-0.144158380406153	-0.000697738382717277	0	0.0100115754530300	-3.45931892135987	0]; % Planar Mirror Orbit "Loop-Dee-Loop" Sub-Trajectory
    x0 = [1.0221, 0, -0.1821, 0, -0.1033, 0];
    %x0 = [1.16429257222878, -0.0144369085836121, 0, -0.0389308426824481, 0.0153488211249537, 0]; % L2 Lagrange Point Approach
    %x0 = [1.16961297958960	-0.0154599483859532	0	-0.0506271631673632	0.00166461443708329	0]; % L2 Lagrange Point Approach (after 60 hours)
    
    % CR3BP Non-dimensionalizations
    normalization_quantities.dist2km = 384400; % Kilometers per non-dimensionalized distance
    normalization_quantities.time2hr = 4.342*24; % Hours per non-dimensionalized time
    normalization_quantities.vel2kms = normalization_quantities.dist2km/(normalization_quantities.time2hr*60*60); % Kms per non-dimensionalized velocity
    dynamics_model = @(t, x) Dynamics.cr3bp_dyn(t, x, normalization_quantities.mu);
end
%orbElements(r0, v0, mu);
%normalization_quantities = Dynamics.normalize_2Body(r0, v0, mu);
% Define time span
%}
tstamp1 = 0; % For long term trajectories 
end_t = 80;
tspan = tstamp1/normalization_quantities.time2hr:(2/normalization_quantities.time2hr):end_t/normalization_quantities.time2hr; % For our modified trajectory 

%opts = odeset('Events', @termSat);
% [t,dx_dt] = ode45(@cr3bp_dyn, tspan, x0, opts); % Assumes termination event (i.e. target enters LEO)

dx_dt = zeros(length(tspan), length(x0)); dx_dt(1,:) = x0;
t = zeros(length(tspan),1);
t(1) = tspan(1);
t_end = tspan(end);
JC = zeros(length(tspan),1);
%{
syn2eci = CoordFunctions.Synodic2ECI(x0, t_end, obs_lata, obs_lona, normalization_quantities);
eci2topo = CoordFunctions.ECI2Topo(syn2eci, t_end, obs_lata, obs_lona, normalization_quantities);
syn2topo = CoordFunctions.Synodic2Topo(x0, t_end, obs_lata, obs_lona, normalization_quantities);

topo2eci = CoordFunctions.Topo2ECI(syn2topo, t_end, obs_lata, obs_lona, normalization_quantities);
eci2syn = CoordFunctions.ECI2Synodic(topo2eci, t_end, obs_lata, obs_lona, normalization_quantities);
topo2syn = CoordFunctions.Topo2Synodic(syn2topo, t_end, obs_lata, obs_lona, normalization_quantities);


syn2topoa = CoordFunctions.Synodic2Topo(x0, t_end, obs_lata, obs_lona, normalization_quantities);
syn2topob = CoordFunctions.Synodic2Topo(x0, t_end, obs_latb, obs_lonb, normalization_quantities);

topo2ecia = CoordFunctions.Topo2ECI(syn2topoa, t_end, obs_lata, obs_lona, normalization_quantities);
topo2ecib = CoordFunctions.Topo2ECI(syn2topob, t_end, obs_latb, obs_lonb, normalization_quantities);

eci2topoa = CoordFunctions.Topo2ECI(topo2ecia, t_end, obs_lata, obs_lona, normalization_quantities);
eci2topob = CoordFunctions.Topo2ECI(topo2ecib, t_end, obs_lata, obs_lona, normalization_quantities);
%}
for i = 1:(length(tspan)-1)
   % [t_tmp,dx_dt_tmp] = ode45(@cr3bp_dyn, [tspan(i) tspan(i+1)], x0, opts); % Assumes termination event (i.e. target enters LEO)
   [t_tmp,dx_dt_tmp] = ode45(dynamics_model, [0, tspan(i+1) - tspan(i)], x0); 
   x0 = dx_dt_tmp(end,:); dx_dt(i+1,:) = x0; t(i+1) = tspan(i+1);
end

% Longer-term scheduling
tstamp = t(end); % Begin new trajectory where we left off
end_t = (250)/normalization_quantities.time2hr;
tspan = tstamp:(8/normalization_quantities.time2hr):end_t; % Schedule to take measurements once every 8 hours
x0_tmp = dx_dt(end,:); t(end) = []; dx_dt(end,:) = []; 

dx_dts = zeros(length(tspan), length(x0)); dx_dts(1,:) = x0_tmp; % Start at end of pass
ts = zeros(length(tspan),1); ts(1) = tstamp;
for i = 1:(length(tspan)-1)
    [t_tmp,dx_dt_tmp] = ode45(dynamics_model, [tspan(i) tspan(i+1)], x0_tmp); % Assumes termination event (i.e. target enters LEO)
    x0_tmp = dx_dt_tmp(end,:); dx_dts(i+1,:) = x0_tmp; ts(i+1) = t_tmp(end);
end

t = [t; ts];
dx_dt = [dx_dt; dx_dts];
%for i = 1:length(dx_dt)
%    JC(i) = Dynamics.jacobi_constant(dx_dt(i, :));
%end
%{
rb = dx_dt(:,1:3); % Position evolutions from barycenter
vb = dx_dt(:,4:6); % Velocity evolutions from barycenter
rbe = [-normalization_quantities.mu, 0, 0]'; % Position vector relating center of earth to barycenter

UTC_vec = [2024	5	3	2	41	15];
t_add_dim = tstamp1 * (4.342);
UTC_vec = datevec(datetime(UTC_vec) + t_add_dim);

rem = [1, 0, 0]'; % Earth center - moon center 

reo_nondim = zeros(length(t),3);
veo_nondim = zeros(length(t),3); % Array for non-dimensionalized EO velocity vectors

rot = zeros(size(rb)); % Observer - Target 
rom = zeros(size(rb)); % Observer - Moon Center
vot = zeros(size(rb)); % Observer - Target Velocity
%}
obj_eci_pos = zeros(size(t, 1), 3); % Observer - Target 
obj_eci_vel = zeros(size(t, 1), 3); % Observer - Target Velocity
moon_eci_pos = zeros(size(t, 1), 3); % Observer - Moon Center
observer_eci_pos = zeros(size(t, 1), 3);
if (dynamics == "2 Body")
    obj_eci_pos = dx_dt(:, 1:3);
    obj_eci_vel = dx_dt(:, 4:6);
end

if (dynamics == "CR3BP")
    rem = [1, 0, 0]'; % Earth center - moon center
    rbe = [-normalization_quantities.mu, 0, 0]'; % Position vector relating center of earth to barycenter
    for i = 1:size(t, 1)
        %{
        t_add_nondim = t(i) - tstamp1; % Time since first point of orbit
        t_add_dim1 = t_add_nondim * (4.342); % Conversion to dimensionalized time
        t_add_dim = t_add_nondim * normalization_quantities.time2hr/24; % Conversion to dimensionalized time
        delt_add_dim = t_add_dim - 1/86400; 
    
        updated_UTCtime = datetime(UTC_vec) + t_add_dim;
        updated_UTCvec = datevec(updated_UTCtime);
    
        delt_updatedUTCtime = datetime(UTC_vec) + delt_add_dim;
        delt_updatedUTCvec = datevec(delt_updatedUTCtime);
    
        reo_dim = lla2eci([obs_lat obs_lon, elevation], updated_UTCvec);
        delt_reodim = lla2eci([obs_lat obs_lon, elevation], delt_updatedUTCvec);
        veo_dim = reo_dim - delt_reodim; 
        
        R_z = [cos(t_add_nondim + tstamp1), -sin(t_add_nondim + tstamp1), 0; sin(t_add_nondim + tstamp1), cos(t_add_nondim + tstamp1), 0; 0, 0, 1];
        dRz_dt = [-sin(t_add_nondim + tstamp1), -cos(t_add_nondim + tstamp1), 0; cos(t_add_nondim + tstamp1), -sin(t_add_nondim + tstamp1), 0; 0, 0, 0];
    
        reo_nondim(i,:) = reo_dim'/(1000*normalization_quantities.dist2km); % Conversion to non-dimensional units and ECI frame
        veo_nondim(i,:) = veo_dim'/(1000*normalization_quantities.vel2kms);
    
        rot(i,:) = -reo_nondim(i,:)' + R_z*(-rbe + rb(i,:)');
        rom(i,:) = -reo_nondim(i,:)' + R_z*rem;
        vot(i,:) = -veo_nondim(i,:)' + R_z*(vb(i,:)') + dRz_dt*(-rbe + rb(i,:)');
        %}
        moon_eci_temp = CoordFunctions.Synodic2ECI([rem'+rbe', 0, 0, 0], t(i) - tstamp1, obs_lat, obs_lon, normalization_quantities);
        moon_eci_pos(i, :) = moon_eci_temp(1, 1:3);
        obj_eci_temp = CoordFunctions.Synodic2ECI(dx_dt(i, :), t(i) - tstamp1, obs_lat, obs_lon, normalization_quantities);
        obj_eci_pos(i, :) = obj_eci_temp(1, 1:3);
        obj_eci_vel(i, :) = obj_eci_temp(1, 4:6);
    end
end

for i = 1:size(t, 1)
    observer_eci_pos_temp = CoordFunctions.LLA2ECI(t(i) - tstamp1, obs_lat, obs_lon, normalization_quantities);
    observer_eci_pos(i, :) = observer_eci_pos_temp./normalization_quantities.dist2km/1000;
end

% Plot the position parametrically w.r.t. time
figure(2)
subplot(3,1,1)
plot(t*normalization_quantities.time2hr, dx_dt(:,1), 'r-')
xlabel('Time')
ylabel('x-Position')
title('CB3RP x-Evolution')

subplot(3,1,2)
plot(t*normalization_quantities.time2hr, dx_dt(:,2), 'g-')
xlabel('Time')
ylabel('y-Position')
title('CB3RP y-Evolution')

subplot(3,1,3)
plot(t*normalization_quantities.time2hr, dx_dt(:,3), 'b-')
xlabel('Time')
ylabel('z-Position')
title('CB3RP z-Evolution')
saveas(gcf, save_loc + "/posEvolution.png", 'png');

% Plot position evolutions between observer and target
figure(3)
plot3(obj_eci_pos(:,1), obj_eci_pos(:,2), obj_eci_pos(:,3), 'g-');
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
saveas(gcf, save_loc + "/rot_trajectory_ECI.png", 'png');

% Before we obtain AZ and EL quantities, we must convert our
% observer-target vector into a topocentric frame.
%{
rot_topo = zeros(length(t),length(rot(1,:)));
rom_topo = zeros(length(t),length(rot(1,:)));
vot_topo = zeros(length(t),length(rot(1,:)));
%}
obj_topo_pos = zeros(size(t, 1), 3); % Observer - Target 
moon_topo_pos = zeros(size(t, 1), 3); % Observer - Moon Center
obj_topo_vel = zeros(size(t, 1), 3); % Observer - Target Velocity

for i = 1:size(t, 1)
    %{
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

    vot_topo(i,:) = R_topo*vot(i,:)' + dA_dt*rot(i,:)';
    %}
    moon_topo_temp = CoordFunctions.ECI2Topo([moon_eci_pos(i, :), 0, 0, 0], t(i) - tstamp1, obs_lat, obs_lon, normalization_quantities);
    moon_topo_pos(i, :) = moon_topo_temp(1, 1:3);
    obj_topo_temp = CoordFunctions.ECI2Topo([obj_eci_pos(i, :), obj_eci_vel(i, :)], t(i) - tstamp1, obs_lat, obs_lon, normalization_quantities);
    obj_topo_pos(i, :) = obj_topo_temp(1, 1:3);
    obj_topo_vel(i, :) = obj_topo_temp(1, 4:6);
end

% Due to not being able to see targets behindthe moon, design function such 
% that if the rot_topo vector passes through the Moon, then data for that 
% time step is considered invalid.

Rm = 1740/384400; % Nondimensionalized radius of the moon

Rho = zeros(length(obj_topo_pos(:,1)),1);
AZ = zeros(length(obj_topo_pos(:,1)),1);
EL = zeros(length(obj_topo_pos(:,1)),1);

for i = 1:length(obj_topo_pos(:,1))
    Rho(i,1) = sqrt(obj_topo_pos(i,1)^2 + obj_topo_pos(i,2)^2 + obj_topo_pos(i,3)^2);
    AZ(i,1) = atan2(obj_topo_pos(i,2), obj_topo_pos(i,1));
    EL(i,1) = pi/2 - acos(obj_topo_pos(i,3)/Rho(i,1));
end
    %if (norm(cross(rot_topo(i,:), rom_topo(i,:)))/norm(rot_topo(i,:)) > Rm && ...
time_mask = (t*normalization_quantities.time2hr >= 0 & t*normalization_quantities.time2hr <= 80) | (t*normalization_quantities.time2hr > 200);
full_ts = [t(time_mask), Rho(time_mask), AZ(time_mask), EL(time_mask)]; % Full augmented time-series vector
elev_mask = EL >= -100000;
mask = time_mask & elev_mask;
t_valid = t(mask);
dx_dt_valid = dx_dt(mask, :);
observer_eci_pos_valid = observer_eci_pos(mask, :);
obj_topo_pos_valid = obj_topo_pos(mask,:);
obj_topo_vel_valid = obj_topo_vel(mask,:);
Rho_valid = Rho(mask);
AZ_valid = AZ(mask);
EL_valid = EL(mask);
partial_ts_ECI = [t_valid, Rho_valid, AZ_valid, EL_valid];
% Convert observer - target position vectors into range, azimuth, and
% elevation quantities

% Last Step: Due to elevation angle constraints, all t, Rho, AZ, EL data
% for which EL < 0 is considered invalid and should be discarded

% < 0 to remove any below elevation of 0


%for i = 1:length(t_valid(:,1))
%    if (full_ts(i, 4) < 0)
%        partial_ts_ECI = partial_ts_ECI(~any(partial_ts_ECI == [t_valid(i), Rho(i), AZ(i), EL(i)], 2), :);
%    end
%end

% Plot the trajectory
figure(1)
plot3(dx_dt(:,1), dx_dt(:,2), dx_dt(:,3));
xlabel('x');
ylabel('y');
zlabel('z');
title('CR3BP Trajectory');
grid on;
hold on;

% Plot valid measurements as open circles
plot3(dx_dt_valid(:,1), dx_dt_valid(:,2), dx_dt_valid(:,3), ...
      'ko', ...      % black open circle markers
      'LineWidth', 1.2, ...
      'MarkerSize', 6);
% Plot masses
plot3(observer_eci_pos_valid(:,1), observer_eci_pos_valid(:,2), observer_eci_pos_valid(:,3), 'r.-', ...
      'LineWidth',1.3,'MarkerSize',10);
%X = [observer_eci_pos_valid(:, 1), dx_dt_valid(:, 1)]
for k = find(mask)
    plot3([observer_eci_pos(k,1), dx_dt(k,1)]', ...
          [observer_eci_pos(k,2), dx_dt(k,2)]', ...
          [observer_eci_pos(k,3), dx_dt(k,3)]', ...
          'b-', 'LineWidth', 1);
end
if (dynamics == "CR3BP")
    plot3(-normalization_quantities.mu, 0, 0, 'ko')
    labels = {'Earth'};
    text(-normalization_quantities.mu, 0, 0, labels,'VerticalAlignment','bottom','HorizontalAlignment','right')
    plot3(1-normalization_quantities.mu, 0, 0, 'go')
    labels = {'Moon'};
    text(1-normalization_quantities.mu, 0, 0, labels,'VerticalAlignment','bottom','HorizontalAlignment','right')
end
% xlim([0.95 1.05])

[ReX, ReY, ReZ] = sphere;

% Here, we nondimensionalize Earth's radius by the distance between the
% Earth and Moon centers

ReX = 6371/normalization_quantities.dist2km * ReX;
ReY = 6371/normalization_quantities.dist2km * ReY;
ReZ = 6371/normalization_quantities.dist2km * ReZ;
surf(ReX, ReY, ReZ)

savefig(gcf, 'trajectory_ECI.fig')
saveas(gcf, save_loc + "/trajectory_ECI.png", 'png');


% Plot the spherical coordinates of the observer parametrically w.r.t. time
figure(4)
subplot(3,1,1)
plot(partial_ts_ECI(:,1)*normalization_quantities.time2hr, partial_ts_ECI(:,2), 'ro')
xlabel('Time')
ylabel('Range (non-dim)')
% xlim([-tstamp1 t(end)])
title('Observer Range Measurements (Ideal)')

subplot(3,1,2)
plot(partial_ts_ECI(:,1)*normalization_quantities.time2hr, partial_ts_ECI(:,3), 'go')
xlabel('Time')
ylabel('Azimuth Angle (rad)')
% xlim([-tstamp1 t(end)])
title('Observer Azimuth Angle Measurements (Ideal)')

subplot(3,1,3)
plot(partial_ts_ECI(:,1)*normalization_quantities.time2hr, partial_ts_ECI(:,4), 'bo')
xlabel('Time')
ylabel('Elevation Angle (rad)')
% xlim([-tstamp1 t(end)])
title('Observer Elevation Angle Measurements (Ideal)')
saveas(gcf, save_loc + "/observations_ECI.png", 'png');
partial_ts = partial_ts_ECI;
save(save_loc + '/partial_ts.mat', 'partial_ts');

full_ts = [t, obj_topo_pos];
save(save_loc + '/full_ts.mat', "full_ts");

% Construct and save a similar data file for velocity data
full_vts = [t, obj_topo_vel];
save(save_loc + '/full_vts.mat', 'full_vts')

save(save_loc + '/normalization_quantities.mat', 'normalization_quantities')


function orbElements(r0, v0, mu)
    % --- Magnitudes ----------------------------------------------
    r = norm(r0);
    v = norm(v0);
    
    % --- Specific angular momentum -------------------------------
    h = cross(r0, v0);
    h_mag = norm(h);
    
    % --- Inclination ---------------------------------------------
    i = acos(h(3)/h_mag);
    
    % --- Node vector ---------------------------------------------
    k = [0 0 1]';
    n = cross(k, h);
    n_mag = norm(n);
    
    % --- Eccentricity vector -------------------------------------
    e_vec = (1/mu)*((v^2 - mu/r)*r0 - dot(r0, v0)*v0);
    e = norm(e_vec);
    
    % --- RAAN (Right Ascension of Ascending Node) ----------------
    if n_mag ~= 0
        RAAN = acos(n(1)/n_mag);
        if n(2) < 0
            RAAN = 2*pi - RAAN;
        end
    else
        RAAN = 0;  % undefined for equatorial orbits
    end
    
    % --- Argument of Perigee -------------------------------------
    if n_mag ~= 0 && e > 1e-10
        omega = acos(dot(n, e_vec)/(n_mag*e));
        if e_vec(3) < 0
            omega = 2*pi - omega;
        end
    else
        omega = 0;  % undefined for circular or equatorial orbits
    end
    
    % --- True Anomaly --------------------------------------------
    if e > 1e-10
        nu = acos(dot(e_vec, r0)/(e*r));
        if dot(r0, v0) < 0
            nu = 2*pi - nu;
        end
    else
        % Circular orbit → true anomaly measured from ascending node
        cp = cross(n, r0);
        nu = acos(dot(n, r0)/(n_mag*r));
        if cp(3) < 0
            nu = 2*pi - nu;
        end
    end
    
    % --- Semi-major axis -----------------------------------------
    a = 1/(2/r - v^2/mu);
    
    
    % Perigee and apogee radii
    rp = a * (1 - e);      % km
    ra = a * (1 + e);      % km
    
    % Altitudes above Earth
    hp = rp - 6378;          % km
    ha = ra - 6378;          % km
    T = 2*pi * sqrt(a^3 / mu)/3600;

    % --- Print results -------------------------------------------
    fprintf('Semi-major axis (a)     = %.3f km\n', a);
    fprintf('Eccentricity (e)        = %.6f\n', e);
    fprintf('Inclination (i)         = %.3f deg\n', rad2deg(i));
    fprintf('RAAN (Ω)                = %.3f deg\n', rad2deg(RAAN));
    fprintf('Arg. of perigee (ω)     = %.3f deg\n', rad2deg(omega));
    fprintf('True anomaly (ν)        = %.3f deg\n', rad2deg(nu));
    fprintf('Perigee radius rp     = %.3f km\n', rp);
    fprintf('Apogee radius ra      = %.3f km\n', ra);
    fprintf('Perigee altitude hp   = %.3f km\n', hp);
    fprintf('Apogee altitude ha    = %.3f km\n\n', ha);
    fprintf('Orbital period T = %.3f hours\n', T);
end