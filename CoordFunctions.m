classdef CoordFunctions
    methods(Static)
        function [reo_topo] = getObserverPos(t_stamp, obs_lat, obs_lon, normalization_quantities)
            % First step: Obtain X_{eo}^{ECI} 
            elevation = 103.8;
        
            UTC_vec_orig = [2024	5	3	2	41	15]; % Initial UTC vector at t_stamp = 0
            t_add_dim = t_stamp * normalization_quantities.time2hr/24; % Convert the time to add to a dimensional quantity
            UTC_vec = datevec(datetime(UTC_vec_orig) + t_add_dim); % You will need this for calculating r_{eo} and v_{eo}
        
            reo_dim = lla2eci([obs_lat obs_lon, elevation], UTC_vec); % ECI frame only
            reo_nondim = reo_dim'/(1000*normalization_quantities.dist2km); % Conversion to non-dimensional units in the ECI frame
        
            z_hat_topo = reo_nondim/norm(reo_nondim); % Convert to topocentric reference frame
            
            x_hat_topo_unorm = cross(z_hat_topo, [0, 0, 1]'); 
            x_hat_topo = x_hat_topo_unorm/norm(x_hat_topo_unorm); % Remember to normalize
        
            y_hat_topo_unorm = cross(x_hat_topo, z_hat_topo);
            y_hat_topo = y_hat_topo_unorm/norm(y_hat_topo_unorm); % Remember to normalize
        
            reo_topo = [dot(reo_nondim, x_hat_topo), dot(reo_nondim, y_hat_topo), dot(reo_nondim, z_hat_topo)];
        end
        
        
        function [X_bt] = Topo2Synodic(X_ot, t_stamp, obs_lat, obs_lon, normalization_quantities)
            % First step: Obtain X_{eo}^{ECI} 
        
            elevation = 103.8;
            mu = 1.2150582e-2;
        
            UTC_vec_orig = [2024	5	3	2	41	15]; % Initial UTC vector at t_stamp = 0
            t_add_dim = t_stamp * normalization_quantities.time2hr/24; % Convert the time to add to a dimensional quantity
            UTC_vec = datevec(datetime(UTC_vec_orig) + t_add_dim); % You will need this for calculating r_{eo} and v_{eo}
        
            delt_add_dim = t_add_dim - 1/86400;
            delt_updatedUTCtime = datetime(UTC_vec_orig) + delt_add_dim;
            delt_updatedUTCvec = datevec(delt_updatedUTCtime);
        
            reo_dim = lla2eci([obs_lat obs_lon, elevation], UTC_vec);
            delt_reodim = lla2eci([obs_lat obs_lon, elevation], delt_updatedUTCvec);
            veo_dim = reo_dim - delt_reodim; % Finite difference
        
            reo_nondim = reo_dim'/(1000*normalization_quantities.dist2km); % Conversion to non-dimensional units in the ECI frame
            veo_nondim = veo_dim'/(1000*normalization_quantities.vel2kms); % Conversion to non-dimensional units in the ECI frame
        
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
                %disp("Topo2Synodic")
                %ret_ECI
                %vet_ECI
            
                R3 = [cos(t_stamp), -sin(t_stamp), 0; sin(t_stamp), cos(t_stamp), 0; 0, 0, 1];
                dR3_dt = [-sin(t_stamp), -cos(t_stamp), 0; cos(t_stamp), -sin(t_stamp), 0; 0, 0, 0];
            
                ret_S = R3^(-1)*ret_ECI;
                vet_S = R3^(-1)*(vet_ECI - dR3_dt*ret_S);
            
                r_be = [-normalization_quantities.mu, 0, 0]';
                v_be = [0, 0, 0]';
            
                r_bt = r_be + ret_S; % In synodic reference frame
                v_bt = v_be + vet_S; % In synodic reference frame
            
                X_bt(particle, :) = [r_bt', v_bt'];
            end
        end
        
        
        % Used for converting between X_{BT} in the synodic frame and X_{OT} in the
        % topocentric frame for a single state
        function [X_ot] = Synodic2Topo(X_bt, t_stamp, obs_lat, obs_lon, normalization_quantities)
            % Insert code for obtaining vector between center of Earth and observer
            elevation = 103.8;
            
            mu = 1.2150582e-2;
            rbe = [-normalization_quantities.mu, 0, 0]'; % Position vector relating center of earth to barycenter
        
            UTC_vec_orig = [2024	5	3	2	41	15];
            t_add_dim = t_stamp * normalization_quantities.time2hr/24;
            UTC_vec = datevec(datetime(UTC_vec_orig) + t_add_dim);
        
            delt_add_dim = -1/86400;
            delt_updatedUTCtime = datetime(UTC_vec) + delt_add_dim;
            delt_updatedUTCvec = datevec(delt_updatedUTCtime);
        
            reo_dim = lla2eci([obs_lat obs_lon, elevation], UTC_vec);
            delt_reodim = lla2eci([obs_lat obs_lon, elevation], delt_updatedUTCvec);
            veo_dim = reo_dim - delt_reodim;
        
            R_z = [cos(t_stamp), -sin(t_stamp), 0; sin(t_stamp), cos(t_stamp), 0; 0, 0, 1];
            dRz_dt = [-sin(t_stamp), -cos(t_stamp), 0; cos(t_stamp), -sin(t_stamp), 0; 0, 0, 0];
        
            reo_nondim = reo_dim'/(1000*normalization_quantities.dist2km); % Conversion to non-dimensional units and ECI frame
            veo_nondim = veo_dim'/(1000*normalization_quantities.vel2kms);
        
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
        
        
        function [X_ECI] = Topo2ECI(X_ot, t_stamp, obs_lat, obs_lon, normalization_quantities)
        
            % First step: Obtain X_{eo}^{ECI} 
            elevation = 103.8;
            %mu = 1.2150582e-2;
        
            UTC_vec_orig = [2024	5	3	2	41 15];%15.1261889999956]; % Initial UTC vector at t_stamp = 0
            t_add_dim = t_stamp * normalization_quantities.time2hr/24; % Convert the time to add to a dimensional quantity
            UTC_vec = datevec(datetime(UTC_vec_orig) + t_add_dim); % You will need this for calculating r_{eo} and v_{eo}
        
            delt_add_dim = t_add_dim - 1/86400;
            delt_updatedUTCtime = datetime(UTC_vec_orig) + delt_add_dim;
            delt_updatedUTCvec = datevec(delt_updatedUTCtime);
        
            reo_dim = lla2eci([obs_lat obs_lon, elevation], UTC_vec);
            delt_reodim = lla2eci([obs_lat obs_lon, elevation], delt_updatedUTCvec);
            %save("./bCS/bCS_" + num2str(id) + ".mat", "reo_dim", "delt_reodim");
            veo_dim = reo_dim - delt_reodim; % Finite difference
        
            reo_nondim = reo_dim'/(1000*normalization_quantities.dist2km); % Conversion to non-dimensional units in the ECI frame
            veo_nondim = veo_dim'/(1000*normalization_quantities.vel2kms); % Conversion to non-dimensional units in the ECI frame
        
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
        
                % Include next 2 lines???
                ret_ECI = reo_nondim + rot_ECI;
                vet_ECI = veo_nondim + vot_ECI;
                %disp("Topo2ECI")
                %ret_ECI
                %vet_ECI
                X_ECI(particle, :) = [ret_ECI', vet_ECI'];
            end
        end

        function [X_topo] = ECI2Topo(X_ECI, t_stamp, obs_lat, obs_lon, normalization_quantities)
            elevation = 103.8;

            UTC_vec_orig = [2024	5	3	2	41	15];
            t_add_dim = t_stamp * normalization_quantities.time2hr/24;
            UTC_vec = datevec(datetime(UTC_vec_orig) + t_add_dim);
        
            delt_add_dim = -1/86400;
            delt_updatedUTCtime = datetime(UTC_vec) + delt_add_dim;
            delt_updatedUTCvec = datevec(delt_updatedUTCtime);
        
            reo_dim = lla2eci([obs_lat, obs_lon, elevation], UTC_vec);
            delt_reodim = lla2eci([obs_lat, obs_lon, elevation], delt_updatedUTCvec);
            veo_dim = reo_dim - delt_reodim;
        
            reo_nondim = reo_dim'/(1000*normalization_quantities.dist2km); % Conversion to non-dimensional units and ECI frame
            veo_nondim = veo_dim'/(1000*normalization_quantities.vel2kms);
        
            num_particles = size(X_ECI, 1);
            X_topo = zeros(size(X_ECI));
            for particle = 1:num_particles
                % Finally, we convert from the ECI frame to the topographic frame
                %rot_ECI = X_ECI(particle, 1:3)';
                %vot_ECI = X_ECI(particle, 4:6)';
                rot_ECI = -reo_nondim + X_ECI(particle, 1:3)';
                vot_ECI = -veo_nondim + X_ECI(particle, 4:6)';
            
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
            
                X_topo(particle, :) = [rot_topo, vot_topo'];
            end
        end

        function [X_ECI] = Synodic2ECI(X_syn, t_stamp, obs_lat, obs_lon, normalization_quantities)
            % Insert code for obtaining vector between center of Earth and observer
            elevation = 103.8;
            
            mu = 1.2150582e-2;
            rbe = [-normalization_quantities.mu, 0, 0]'; % Position vector relating center of earth to barycenter
        
            UTC_vec_orig = [2024	5	3	2	41	15];
            t_add_dim = t_stamp * normalization_quantities.time2hr/24;
            UTC_vec = datevec(datetime(UTC_vec_orig) + t_add_dim);
        
            delt_add_dim = -1/86400;
            delt_updatedUTCtime = datetime(UTC_vec) + delt_add_dim;
            delt_updatedUTCvec = datevec(delt_updatedUTCtime);
        
            reo_dim = lla2eci([obs_lat, obs_lon, elevation], UTC_vec);
            delt_reodim = lla2eci([obs_lat, obs_lon, elevation], delt_updatedUTCvec);
            veo_dim = reo_dim - delt_reodim;
        
            R_z = [cos(t_stamp), -sin(t_stamp), 0; sin(t_stamp), cos(t_stamp), 0; 0, 0, 1];
            dRz_dt = [-sin(t_stamp), -cos(t_stamp), 0; cos(t_stamp), -sin(t_stamp), 0; 0, 0, 0];
        
            reo_nondim = reo_dim'/(1000*normalization_quantities.dist2km); % Conversion to non-dimensional units and ECI frame
            veo_nondim = veo_dim'/(1000*normalization_quantities.vel2kms);
        
            num_particles = size(X_syn, 1);
            X_ECI = zeros(size(X_syn));
            for particle = 1:num_particles
                %rot_ECI = -reo_nondim + R_z*(-rbe + X_syn(particle, 1:3)');
                %vot_ECI = -veo_nondim + R_z*(X_syn(particle, 4:6)') + dRz_dt*(-rbe + X_syn(particle, 1:3)');
                rot_ECI = R_z*(-rbe + X_syn(particle, 1:3)');
                vot_ECI = R_z*(X_syn(particle, 4:6)') + dRz_dt*(-rbe + X_syn(particle, 1:3)');
                X_ECI(particle, :) = [rot_ECI', vot_ECI'];
            end
        end

        function [X_syn] = ECI2Synodic(X_ECI, t_stamp, obs_lat, obs_lon, normalization_quantities)
            % First step: Obtain X_{eo}^{ECI} 
        
            elevation = 103.8;
            mu = 1.2150582e-2;
        
            UTC_vec_orig = [2024	5	3	2	41	15]; % Initial UTC vector at t_stamp = 0
            t_add_dim = t_stamp * normalization_quantities.time2hr/24; % Convert the time to add to a dimensional quantity
            UTC_vec = datevec(datetime(UTC_vec_orig) + t_add_dim); % You will need this for calculating r_{eo} and v_{eo}
        
            delt_add_dim = t_add_dim - 1/86400;
            delt_updatedUTCtime = datetime(UTC_vec_orig) + delt_add_dim;
            delt_updatedUTCvec = datevec(delt_updatedUTCtime);
        
            reo_dim = lla2eci([obs_lat obs_lon, elevation], UTC_vec);
            delt_reodim = lla2eci([obs_lat obs_lon, elevation], delt_updatedUTCvec);
            veo_dim = reo_dim - delt_reodim; % Finite difference
        
            reo_nondim = reo_dim'/(1000*normalization_quantities.dist2km); % Conversion to non-dimensional units in the ECI frame
            veo_nondim = veo_dim'/(1000*normalization_quantities.vel2kms); % Conversion to non-dimensional units in the ECI frame
        
            z_hat_topo = reo_nondim/norm(reo_nondim);
            x_hat_topo = cross(z_hat_topo, [0, 0, 1]')/norm(cross(z_hat_topo, [0, 0, 1]'));
            y_hat_topo = cross(x_hat_topo, z_hat_topo)/norm(cross(x_hat_topo, z_hat_topo));
            
            A = [x_hat_topo'; y_hat_topo'; z_hat_topo']; % Computing A as DCM for transforming between ECI and topographic reference frame
        
            dmag_dt = dot(reo_nondim, veo_nondim)/norm(reo_nondim);
            
            zhat_dot_topo = (veo_nondim * norm(reo_nondim) - reo_nondim * dmag_dt)/(norm(reo_nondim))^2;
            xhat_dot_topo = cross(zhat_dot_topo, [0, 0, 1]')/norm(cross(z_hat_topo, [0,0,1]')) - dot(x_hat_topo, cross(zhat_dot_topo, [0, 0, 1]'))*x_hat_topo;
            yhat_dot_topo = (cross(xhat_dot_topo, z_hat_topo) + cross(x_hat_topo, zhat_dot_topo))/norm(cross(x_hat_topo, z_hat_topo)) - dot(y_hat_topo, cross(xhat_dot_topo, z_hat_topo) + cross(x_hat_topo, zhat_dot_topo))*y_hat_topo;
        
            dA_dt = [xhat_dot_topo'; yhat_dot_topo'; zhat_dot_topo'];
        
            num_particles = size(X_ECI, 1);
            X_syn = zeros(size(X_ECI));
            for particle = 1:num_particles
                %rot_topo = X_ot(particle, 1:3)'; % First three components of the state vector
                %vot_topo = X_ot(particle, 4:6)'; % Last three components of the state vector
        
                %rot_ECI = A^(-1)*rot_topo;
                %vot_ECI = A^(-1)*(vot_topo - dA_dt*rot_ECI);
            
                % Calculating X_{ET} in the synodic frame with our above quantities
                
                %ret_ECI = reo_nondim + X_ECI(particle, 1:3)';%reo_nondim + rot_ECI;
                %vet_ECI = veo_nondim + X_ECI(particle, 4:6)';%veo_nondim + vot_ECI;
                ret_ECI = X_ECI(particle, 1:3)';
                vet_ECI = X_ECI(particle, 4:6)';
            
                R3 = [cos(t_stamp), -sin(t_stamp), 0; sin(t_stamp), cos(t_stamp), 0; 0, 0, 1];
                dR3_dt = [-sin(t_stamp), -cos(t_stamp), 0; cos(t_stamp), -sin(t_stamp), 0; 0, 0, 0];
            
                ret_S = R3^(-1)*ret_ECI;
                vet_S = R3^(-1)*(vet_ECI - dR3_dt*ret_S);
            
                r_be = [-normalization_quantities.mu, 0, 0]';
                v_be = [0, 0, 0]';
            
                r_bt = r_be + ret_S; % In synodic reference frame
                v_bt = v_be + vet_S; % In synodic reference frame
                
                X_syn(particle, :) = [r_bt', v_bt'];
            end
        end
        
        function [X_ECI] = LLA2ECI(t_stamp, obs_lat, obs_lon, normalization_quantities)
            elevation = 103.8;
        
            UTC_vec_orig = [2024	5	3	2	41 15];%15.1261889999956]; % Initial UTC vector at t_stamp = 0
            t_add_dim = t_stamp * normalization_quantities.time2hr/24; % Convert the time to add to a dimensional quantity
            UTC_vec = datevec(datetime(UTC_vec_orig) + t_add_dim); % You will need this for calculating r_{eo} and v_{eo}
        
            X_ECI = lla2eci([obs_lat obs_lon, elevation], UTC_vec);
        end
    end
end