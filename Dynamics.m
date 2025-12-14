classdef Dynamics
    methods(Static)
        function [dx_dt] = cr3bp_dyn(t, x, ~)
        
        % Target dynamics
        mu = 1.2150582e-2; % Dimensionless mass of the moon
        r1 = sqrt((x(1) + mu)^2 + x(2)^2 + x(3)^2);
        r2 = sqrt((x(1) - 1 + mu)^2 + x(2)^2 + x(3)^2);
        
        cx = 1 - (1-mu)/r1^3 - mu/r2^3;
        cy = 1 - (1 - mu)/r1^3 - mu/r2^3;
        cz = -((1 - mu)/r1^3 + mu/r2^3);
        
        bx = (mu - mu^2)/r1^3 + (-mu + mu^2)/r2^3;
        
        dx_dt = [x(4), x(5), x(6), cx*x(1)+2*x(5)-bx, cy*x(2)-2*x(4), cz*x(3)]';
        % dx_dt = [x(4), x(5), x(6), 2*x(5) + x(1) - (1-mu)*(x(1)+mu)/r1^3 - mu*(x(1)-1+mu)/r2^3, cy*x(2)-2*x(4), cz*x(3)]';
        
        end

        function [dx_dt] = two_body_dyn(t, x, mu)
            %mu = 398600.4418; % km^3/s^2
        
            % Extract position and velocity
            r = x(1:3);
            v = x(4:6);
            
            % Norm of position
            r_norm = norm(r);
            
            % Acceleration due to gravity
            a = -mu * r / r_norm^3;
            
            % Assemble derivative
            dx_dt = [v; a];
        end

        function normalization_quantities = normalize_2Body(r0, v0, mu)
            r = norm(r0);
            v = norm(v0);
            
            energy = v^2/2 - mu/r;
            a = -mu/(2*energy);
            
            normalization_quantities.dist2km = a;
            normalization_quantities.vel2kms = sqrt(mu/a);
            normalization_quantities.time2hr = (normalization_quantities.dist2km/normalization_quantities.vel2kms)/3600;
            normalization_quantities.mu = mu / (normalization_quantities.vel2kms^2 * normalization_quantities.dist2km);
        end
    end
end