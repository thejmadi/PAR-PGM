load("mu_c.mat"); load("P_c.mat");
load("wm.mat"); load("zt.mat"); load("R_vv.mat");

load("mu_p.mat"); mu_ptmp = mu_p; 
load("P_p.mat"); P_ptmp = P_p;
load("wp.mat"); wptmp = wp;

h = @(x) [sqrt(x(1)^2 + x(2)^2 + x(3)^2); atan2(x(2),x(1)); pi/2 - acos(x(3)/sqrt(x(1)^2 + x(2)^2 + x(3)^2))]; % Nonlinear measurement model

for i = 1:length(wm)
    Hxk = linHx(mu_c{i});
    num = wm(i)*gaussProb(zt, h(mu_c{i}), Hxk*P_c{i}*Hxk' + R_vv);
    den = 0;
    for j = 1:length(wm)
        Hxk = linHx(mu_c{j});
        den = den + wm(j)*gaussProb(zt, h(mu_c{j}), Hxk*P_c{j}*Hxk' + R_vv);
    end
    wp(i) = num/den;
end

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