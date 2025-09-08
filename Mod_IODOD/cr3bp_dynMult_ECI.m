% Define initial conditions
mu = 1.2150582e-2;
% x0 = [0.5-mu, 0.0455, 0, -0.5, 0.5, 0.0]'; % Sample Starting Point
% x0 = [-1.00506 0 0 -0.5 0.5 0.0]'; % L3 Lagrange Point
% x0 = [0.5-mu, sqrt(3/4), 0, 0, 0, 0]'; % L4 Lagrange Point

% Lagrange points that we may not use
% x0 = [0.836915 0 0 0 0 0]'; % L1 Lagrange Point
% x0 = [1.15568 0 0 0 0 0]'; % L2 Lagrange Point
% x0 = [0.5-mu, -sqrt(3/4), 0, 0, 0, 0]'; % L5 Lagrange Point

% x0 = [-0.144158380406153	-0.000697738382717277	0	0.0100115754530300	-3.45931892135987	0]; % Planar Mirror Orbit "Loop-Dee-Loop" Sub-Trajectory
% x0 = [1.15568 0 0 0 0.04 0]';
% x0 = [1.16429257222878	-0.0144369085836121	0	-0.0389308426824481	0.0153488211249537	0]; % L2 Lagrange Point Approach
x0 = [1.0221 0 -0.1821 0 -0.1033 0]; % 9:2 Resonant Orbit NRHO (from Thangavelu MS Thesis 2019)

% Coordinate system conversions
dist2km = 384400; % Kilometers per non-dimensionalized distance
time2hr = 4.342*24; % Hours per non-dimensionalized time
vel2kms = dist2km/(time2hr*60*60); % Kms per non-dimensionalized velocity

Nt = 2; % Number of targets
sd = 4000/dist2km; % Separation distance from first target, in nd units
vd = 0.01/vel2kms;
Q0 = diag([sd, 0, 0, 0, 0, 0].^2);
% Q0_dim = diag([500, 0, 0, 0, 0, 0].^2); % Variances in km and km/s
% Q0 = Q0_dim ./ ([dist2km, 1, 1, 1, 1, 1]' * [dist2km, 1, 1, 1, 1, 1]);

X0 = zeros(Nt, length(x0));
X0(1,:) = mvnrnd(x0, Q0);

for j = 2:Nt
    n_p = length(x0(1:2));
    v = randn(n_p,1);
    v = v/norm(v);

    X0(j,:) = X0(1,:); % Keep velocities and CR3BP z-direction same
    X0(j,1:n_p) = X0(j-1,1:n_p)' + sd*v;
end

% X0 = mvnrnd(x0, Q0, Nt);
X0_0 = squeeze(X0);

% Define time span
tstamp1 = 0; tstamp = 50;
end_t = (24/time2hr) - tstamp1;
tspan = 0:(0.625/time2hr):end_t; % For our modified trajectory 

% Call ode45()

opts = odeset('Events', @termSat);
% [t,dx_dt] = ode45(@cr3bp_dyn, tspan, x0, opts); % Assumes termination event (i.e. target enters LEO)

Dx_Dt = zeros(Nt, length(tspan), length(x0)); Dx_Dt(:,1,:) = X0;
t = zeros(length(tspan),1);

for j = 1:Nt
    for i = 1:(length(tspan)-1)
       [t_tmp,dx_dt_tmp] = ode45(@cr3bp_dyn, [0, tspan(i+1) - tspan(i)], X0(j,:), opts); 
       X0(j,:) = dx_dt_tmp(end,:); Dx_Dt(j,i+1,:) = X0(j,:); t(i+1) = tspan(i+1);
    end
end
% tstamp = 40*24/time2hr;

% Longer-term scheduling
tstamp = t(end); % Begin new trajectory where we left off
end_t = (30*24/time2hr) - tstamp1;
tspan = tstamp:(16/time2hr):end_t; % Schedule to take measurements once every 8 hours
X0_tmp = Dx_Dt(:,end,:); t(end) = []; Dx_Dt(:,end,:) = []; 

Dx_Dts = zeros(Nt, length(tspan), length(x0)); Dx_Dts(:,1,:) = X0_tmp; % Start at end of pass
ts = zeros(length(tspan),1); ts(1) = tstamp;
opts = odeset('Events', @termSat);
for j = 1:Nt
    for i = 1:(length(tspan)-1)
        [t_tmp,dx_dt_tmp] = ode45(@cr3bp_dyn, [tspan(i) tspan(i+1)], X0_tmp(j,:), opts); % Assumes termination event (i.e. target enters LEO)
        X0_tmp(j,:) = dx_dt_tmp(end,:); Dx_Dts(j,i+1,:) = X0_tmp(j,:); ts(i+1) = t_tmp(end);
    end
end

t = [t; ts];
Dx_Dt = [Dx_Dt, Dx_Dts];
%}

Rb = Dx_Dt(:,:,1:3); % Position evolutions from barycenter
Vb = Dx_Dt(:,:,4:6); % Velocity evolutions from barycenter
rbe = [-mu, 0, 0]'; % Position vector relating center of earth to barycenter

% Insert code for obtaining vector between center of Earth and observer

obs_lat = 30.618963;
obs_lon = -96.339214;
elevation = 103.8;

% dtLCL = datetime('now', 'TimeZone','local'); % Current Local Time
% dtUTC = datetime(dtLCL, 'TimeZone','Z');     % Current UTC Time
% UTC_vec = datevec(dtUTC); % Convert to vector

UTC_vec = [2024	5	3	2	41	15.1261889999956];
t_add_dim = tstamp1 * (4.342);
UTC_vec = datevec(datetime(UTC_vec) + t_add_dim);

% reo_dim = lla2eci([obs_lat obs_lon elevation], UTC_vec); % Position vector between observer and center of Earth in meters
% reo_nondim = reo_dim/(1000*384400);

rem = [1, 0, 0]'; % Earth center - moon center 
Reo_nondim = zeros(Nt, length(t),length(x0)/2);
Veo_nondim = zeros(Nt, length(t),length(x0)/2); % Array for non-dimensionalized EO velocity vectors

Rot = zeros(size(Rb)); % Observer - Target Position
Rom = zeros(size(Rb)); % Observer - Moon Center
Vot = zeros(size(Rb)); % Observer - Target Velocity

for j = 1:Nt
    for i = 1:length(Rb(j,:,1))
        t_add_nondim = t(i) - tstamp1; % Time since first point of orbit
        t_add_dim = t_add_nondim * (4.342); % Conversion to dimensionalized time
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
    
        reo_nondim = reo_dim'/(1000*384400); % Conversion to non-dimensional units and ECI frame
        veo_nondim = veo_dim'*(4.342*86400)/(1000*384400);
        rb = squeeze(Rb(j,i,:));
        vb = squeeze(Vb(j,i,:));
    
        Rot(j,i,:) = -reo_nondim' + (R_z*(-rbe + rb))';
        Rom(j,i,:) = -reo_nondim' + (R_z*rem)';
        Vot(j,i,:) = -veo_nondim' + (R_z*vb)' + (dRz_dt*(-rbe + rb))';

        Reo_nondim(j,i,:) = reo_nondim; Veo_nondim(j,i,:) = veo_nondim;
    end
end

% Plot the trajectory
colors = ["Red", "Blue", "Green", "Yellow", "Magenta", "Cyan", "Black", "#500000", "#bf5700", "#00274c"];

figure(1)
for i = 1:Nt
    plot3(Dx_Dt(i,:,1), Dx_Dt(i,:,2), Dx_Dt(i,:,3));
    hold on;
end
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

% Plot the position paxrametrically w.r.t. time

figure(2)
subplot(3,1,1)
for i = 1:Nt
    plot(t, Dx_Dt(i,:,1), 'Color', colors(i))
    hold on;
end
xlabel('Time')
ylabel('x-Position')
title('CB3RP x-Evolution')

subplot(3,1,2)
for i = 1:Nt
    plot(t, Dx_Dt(i,:,2), 'Color', colors(i))
    hold on;
end
xlabel('Time')
ylabel('y-Position')
title('CB3RP y-Evolution')

subplot(3,1,3)
for i = 1:Nt
    plot(t, Dx_Dt(i,:,3), 'Color', colors(i))
    hold on;
end
xlabel('Time')
ylabel('z-Position')
title('CB3RP z-Evolution')
saveas(gcf, 'posEvolution.png')

% Plot position evolutions between observer and target
figure(3)
for i = 1:Nt
    plot3(Rot(i,:,1), Rot(i,:,2), Rot(i,:,3), 'Color', colors(i));
    hold on;
end
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

Rot_topo = zeros(Nt, length(t),length(Rot(1,1,:)));
Rom_topo = zeros(Nt, length(t),length(Rot(1,1,:)));
Vot_topo = zeros(Nt, length(t),length(Rot(1,1,:)));

for j = 1:Nt
    reo_nondim = squeeze(Reo_nondim(j,:,:));
    veo_nondim = squeeze(Veo_nondim(j,:,:));
    rom = squeeze(Rom(j,:,:));
    rot = squeeze(Rot(j,:,:));
    vot = squeeze(Vot(j,:,:));
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
        
        rot_topo = [dot(rot(i,:), x_hat_topo), dot(rot(i,:), y_hat_topo), dot(rot(i,:), z_hat_topo)];
        rom_topo = [dot(rom(i,:), x_hat_topo), dot(rom(i,:), y_hat_topo), dot(rom(i,:), z_hat_topo)];
    
        % Step 3: Handle the time derivatives of vot_topo = d/dt (rot_topo)
        R_topo = [x_hat_topo; y_hat_topo; z_hat_topo]; % DCM relating ECI to topocentric coordinate frame
        dmag_dt = dot(reo_nondim(i,:), veo_nondim(i,:))/norm(reo_nondim(i,:)); % How the magnitude of r_eo changes w.r.t. time
        
        zhat_dot_topo = (veo_nondim(i,:)*norm(reo_nondim(i,:)) - reo_nondim(i,:)*dmag_dt)/(norm(reo_nondim(i,:)))^2;
        xhat_dot_topo = cross(zhat_dot_topo, [0, 0, 1]')/norm(cross(z_hat_topo, [0,0,1]')) - dot(x_hat_topo, cross(zhat_dot_topo, [0, 0, 1]'))*x_hat_topo;
        yhat_dot_topo = (cross(xhat_dot_topo, z_hat_topo) + cross(x_hat_topo, zhat_dot_topo))/norm(cross(x_hat_topo, z_hat_topo)) - dot(y_hat_topo, cross(xhat_dot_topo, z_hat_topo) + cross(x_hat_topo, zhat_dot_topo))*y_hat_topo;
    
        dA_dt = [xhat_dot_topo; yhat_dot_topo; zhat_dot_topo];
        vot_topo = R_topo*vot(i,:)' + dA_dt*rot(i,:)';

        Rot_topo(j,i,:) = rot_topo;
        Rom_topo(j,i,:) = rom_topo;
        Vot_topo(j,i,:) = vot_topo;
    end
end

% Due to not being able to see targets behind the moon, design function such 
% that if the rot_topo vector passes through the Moon, then data for that 
% time step is considered invalid.

Rot_valid = cell(Nt,1);
Vot_valid = cell(Nt,1);
T_valid = cell(Nt,1);

Rm = 1740/384400; % Nondimensionalized radius of the moon

for k = 1:Nt
    rot_topo = squeeze(Rot_topo(k,:,:));
    vot_topo = squeeze(Vot_topo(k,:,:));
    rom_topo = squeeze(Rom_topo(k,:,:));
    j = 0;
    for i = 1:length(t)
        if (norm(cross(rot_topo(i,:), rom_topo(i,:)))/norm(rot_topo(i,:)) > Rm ...
                && (t(i) <= tstamp || t(i) > (24*24)/time2hr))
            j = j + 1;
            T_valid{k}(j) = t(i);
            Rot_valid{k}(j,:) = rot_topo(i,:);
            Vot_valid{k}(j,:) = vot_topo(i,:);
        end
    end
end

% Convert observer - target position vectors into range, azimuth, and
% elevation quantities

Rho = cell(1,Nt);
AZ = cell(1,Nt);
EL = cell(1,Nt);

for j = 1:Nt
    Rho{j} = zeros(length(Rot_valid{j}(:,1)), 1);
    AZ{j} = zeros(length(Rot_valid{j}(:,1)), 1);
    EL{j} = zeros(length(Rot_valid{j}(:,1)), 1);

    for i = 1:length(Rot_valid{j}(:,1))
        Rho{j}(i) = sqrt(Rot_valid{j}(i,1)^2 + Rot_valid{j}(i,2)^2 + Rot_valid{j}(i,3)^2);
        AZ{j}(i) = atan2(Rot_valid{j}(i,2), Rot_valid{j}(i,1));
        EL{j}(i) = pi/2 - acos(Rot_valid{j}(i,3)/Rho{j}(i));
    end
end

% Last Step: Due to elevation angle constraints, all t, Rho, AZ, EL data
% for which EL < 0 is considered invalid and should be discarded

% partial_tvalid = T_valid;
% partial_Rho = Rho;
% partial_AZ = AZ;
% partial_EL = EL;

Partial_ts = cell(1,Nt);
for i = 1:Nt
    Partial_ts{i} = [T_valid{i}', Rho{i}, AZ{i}, EL{i}];
end


Full_ts = cell(1,2); Full_vts = cell(1,2);
for i = 1:Nt
    Full_ts{i} = [t, squeeze(Rot_topo(i,:,:))];
    Full_vts{i} = [t, squeeze(Vot_topo(i,:,:))];
end

for j = 1:Nt
    partial_ts_ECI = Partial_ts{j};
    for i = 1:length(T_valid{j})
        if (Partial_ts{j}(i, 4) < 0)
            partial_ts_ECI = partial_ts_ECI(~any(partial_ts_ECI == [T_valid{j}(i), Rho{j}(i), AZ{j}(i), EL{j}(i)], 2), :);
        end
    end
    Partial_ts{j} = partial_ts_ECI;
end

% Plot the spherical coordinates of the observer parametrically w.r.t. time
figure(4)
subplot(3,1,1)
for i = 1:Nt
    plot(Partial_ts{i}(:,1), Partial_ts{i}(:,2), 'o')
    hold on;
end
xlabel('Time')
ylabel('Range (non-dim)')
% xlim([-tstamp1 t(end)])
title('Observer Range Measurements (Ideal)')

subplot(3,1,2)
for i = 1:Nt
    plot(Partial_ts{i}(:,1), Partial_ts{i}(:,3), 'o')
    hold on;
end
xlabel('Time')
ylabel('Azimuth Angle (rad)')
% xlim([-tstamp1 t(end)])
title('Observer Azimuth Angle Measurements (Ideal)')

subplot(3,1,3)
for i = 1:Nt
    plot(Partial_ts{i}(:,1), Partial_ts{i}(:,4), 'o')
    hold on;
end
xlabel('Time')
ylabel('Elevation Angle (rad)')
% xlim([-tstamp1 t(end)])
title('Observer Elevation Angle Measurements (Ideal)')
saveas(gcf, 'observations_ECI.png')

save('partial_ts_mult.mat', 'Partial_ts');
save('full_ts_mult.mat', "Full_ts");
save('full_vts_mult.mat', 'Full_vts')