figUKF = open("./Simulations/9-2 Resonant NRHO (Full Orbit UKF)/Entropy.fig");
% figENKF = gcf;

XData = cell(3,1);
YData = cell(3,1);

axObjs = figUKF.Children;
dataObjs = axObjs.Children;

lineObj = axObjs(1).Children;

% Extract X and Y data from the line object
XData{1} = lineObj.XData;
YData{1} = lineObj.YData;

figPGM = open("./Simulations/9-2 Resonant NRHO (Full-Orbit PGM)/Entropy.fig");
% figPGM = gcf;

axObjs = figPGM.Children;
dataObjs = axObjs.Children;

lineObj = axObjs(1).Children;

% Extract X and Y data from the line object
XData{2} = lineObj.XData;
YData{2} = lineObj.YData;

figEnKF = open("./Simulations/9-2 Resonant NRHO (Full-Orbit EnKF)/Entropy.fig");
% figENKF = gcf;

axObjs = figEnKF.Children;
dataObjs = axObjs.Children;

lineObj = axObjs(1).Children;

% Extract X and Y data from the line object
XData{3} = lineObj.XData;
YData{3} = lineObj.YData;

figure(4)
plot(XData{1}, YData{1}, XData{2}, YData{2}, XData{3}, YData{3})
xlabel('Filter Time Step #')
ylabel('Filter Entropy')
title('Entropy Comparison (UKF vs. PGM vs. EnKF)')
legend('UKF', 'PGM', 'EnKF')
savefig(gcf, "entropyComp_9-2NRHO.fig");

%{
dist2km = 384400; % Kilometers per non-dimensionalized distance
time2hr = 4.342*24; % Hours per non-dimensionalized time
vel2kms = dist2km/(time2hr*60*60); % Kms per non-dimensionalized velocity

load("stdevs_EnKF.mat", "ent1");
ent_EnKF = ent1;
load("stDevs_PGM.mat", "ent1");
ent_PGM = ent1;

l_filt = length(ent1(:,1))-1;

figure(1)

subplot(2,3,1)
semilogy(0:l_filt, dist2km*sqrt(ent_PGM(:,1)), 0:l_filt, dist2km*sqrt(ent_EnKF(:,1)))
xlabel('Filter Step #')
ylabel('StDev in X (km.)')
legend('PGM','EnKF')
title('X Standard Deviation Evolution')

subplot(2,3,2)
semilogy(0:l_filt, dist2km*sqrt(ent_PGM(:,2)), 0:l_filt, dist2km*sqrt(ent_EnKF(:,2)))
xlabel('Filter Step #')
ylabel('StDev in Y (km.)')
legend('PGM','EnKF')
title('Y Standard Deviation Evolution')

subplot(2,3,3)
semilogy(0:l_filt, dist2km*sqrt(ent_PGM(:,3)), 0:l_filt, dist2km*sqrt(ent_EnKF(:,3)))
xlabel('Filter Step #')
ylabel('StDev in Z (km.)')
legend('PGM','EnKF')
title('Z Standard Deviation Evolution')

subplot(2,3,4)
semilogy(0:l_filt, vel2kms*sqrt(ent_PGM(:,4)), 0:l_filt, vel2kms*sqrt(ent_EnKF(:,4)))
xlabel('Filter Step #')
ylabel('StDev in Xdot (km/s)')
legend('PGM','EnKF')
title('Xdot Standard Deviation Evolution')

subplot(2,3,5)
semilogy(0:l_filt, vel2kms*sqrt(ent_PGM(:,5)), 0:l_filt, vel2kms*sqrt(ent_EnKF(:,5)))
xlabel('Filter Step #')
ylabel('StDev in Ydot (km/s)')
legend('PGM','EnKF')
title('Ydot Standard Deviation Evolution')

subplot(2,3,6)
semilogy(0:l_filt, vel2kms*sqrt(ent_PGM(:,6)), 0:l_filt, vel2kms*sqrt(ent_EnKF(:,6)))
xlabel('Filter Step #')
ylabel('StDev in Zdot (km/s)')
legend('PGM','EnKF')
title('Zdot Standard Deviation Evolution')

rel_errors = zeros(l_filt+1, length(ent_EnKF(1,:)));

for i = 0:l_filt
    for j = 1:length(ent_EnKF(1,:))
        % rel_errors(i+1,j) = 100*abs((ent_PGM(i+1,j) - ent_EnKF(i+1,j))/ent_EnKF(i+1,j));
        rel_errors(i+1,j) = 100*abs(ent_PGM(i+1,j)/ent_EnKF(i+1,j));
    end
end

figure(2)

subplot(2,3,1)
plot(0:l_filt, rel_errors(:,1))
xlabel('Filter Step #')
ylabel('Relative StDev in X (%)')
title('X Standard Deviation (PGM vs. EnKF) Evolution')

subplot(2,3,2)
plot(0:l_filt, rel_errors(:,2))
xlabel('Filter Step #')
ylabel('Relative StDev in Y (%)')
title('Y Standard Deviation (PGM vs. EnKF) Evolution')

subplot(2,3,3)
plot(0:l_filt, rel_errors(:,3))
xlabel('Filter Step #')
ylabel('Relative StDev in Z (%)')
title('Z Standard Deviation (PGM vs. EnKF) Evolution')

subplot(2,3,4)
plot(0:l_filt, rel_errors(:,4))
xlabel('Filter Step #')
ylabel('Relative StDev in Xdot (%)')
title('Xdot Standard Deviation (PGM vs. EnKF) Evolution')

subplot(2,3,5)
plot(0:l_filt, rel_errors(:,5))
xlabel('Filter Step #')
ylabel('Relative StDev in Ydot (%)')
title('Ydot Standard Deviation (PGM vs. EnKF) Evolution')

subplot(2,3,6)
plot(0:l_filt, rel_errors(:,6))
xlabel('Filter Step #')
ylabel('Relative StDev in Zdot (%)')
title('Zdot Standard Deviation (PGM vs. EnKF) Evolution')
%}