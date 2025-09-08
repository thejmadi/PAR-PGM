function [value, isterminal, direction] = termSat(T, Y)

mu         = 1.2150582e-2; % Dimensionless mass of the moon (and position of Earth w.r.t. barycenter)
Rm         = 1740/384400; % Nondimensionalized radius of the moon
value      = (sqrt((Y(1) + mu)^2 + Y(2)^2 + Y(3)^2) < 6371/384400) || (sqrt((Y(1) - (1-mu))^2 + Y(2)^2 + Y(3)^2) < Rm); % Stop when the target hits the Earth's or the Moon's surface
isterminal = 1;   % Stop the integration
direction  = 0;

end