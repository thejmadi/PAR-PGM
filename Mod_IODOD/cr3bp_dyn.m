function [dx_dt] = cr3bp_dyn(t, x)

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