% Parameters
dt = 0.1;               % Time step
Q = 0.01;               % Process noise covariance
steps = 50;             % Number of time steps
x0 = 1;                 % Initial mean
P0 = 0.1;               % Initial covariance

% UKF parameters
alpha = 0.5;           % UKF scaling parameter
beta = 2;               % UKF scaling parameter
kappa = 0;              % UKF scaling parameter
n = 1;                  % State dimension

% Compute weights for sigma points
lambda = alpha^2 * (n + kappa) - n;
Wm = [lambda/(n+lambda), 0.5/(n+lambda) + zeros(1, 2*n)]; % Mean weights
Wc = Wm;
Wc(1) = Wc(1) + (1 - alpha^2 + beta); % Covariance weights

% Initialize
x_correct = x0;         % Mean (correct approach)
P_correct = P0;         % Covariance (correct approach)
x_incorrect = x0;       % Mean (incorrect approach)
P_incorrect = P0;       % Covariance (incorrect approach)

% Storage for results
x_correct_history = zeros(1, steps);
x_incorrect_history = zeros(1, steps);
P_correct_history = zeros(1, steps);
P_incorrect_history = zeros(1, steps);

% Main loop
for k = 1:steps
    % Correct approach: Recompute sigma points at each step
    sigmaPoints_correct = computeSigmaPoints(x_correct, P_correct, lambda);
    propagatedSigmaPoints_correct = propagateDynamics(sigmaPoints_correct, dt);
    [x_correct, P_correct] = computeMeanAndCovariance(propagatedSigmaPoints_correct, Wm, Wc, Q);
    
    % Incorrect approach: Propagate initial sigma points without recomputation
    if k == 1
        sigmaPoints_incorrect = computeSigmaPoints(x_incorrect, P_incorrect, lambda);
    end
    propagatedSigmaPoints_incorrect = propagateDynamics(sigmaPoints_incorrect, dt);
    [x_incorrect, P_incorrect] = computeMeanAndCovariance(propagatedSigmaPoints_incorrect, Wm, Wc, Q);
    sigmaPoints_incorrect = propagatedSigmaPoints_incorrect; % Propagate same sigma points
    
    % Store results
    x_correct_history(k) = x_correct;
    x_incorrect_history(k) = x_incorrect;
    P_correct_history(k) = P_correct;
    P_incorrect_history(k) = P_incorrect;
end

% Plot results
figure;
subplot(2, 1, 1);
plot(1:steps, x_correct_history, 'b', 'LineWidth', 2); hold on;
plot(1:steps, x_incorrect_history, 'r--', 'LineWidth', 2);
legend('Correct Approach', 'Incorrect Approach');
xlabel('Time Step');
ylabel('Mean');
title('Mean Propagation');

subplot(2, 1, 2);
plot(1:steps, P_correct_history, 'b', 'LineWidth', 2); hold on;
plot(1:steps, P_incorrect_history, 'r--', 'LineWidth', 2);
legend('Correct Approach', 'Incorrect Approach');
xlabel('Time Step');
ylabel('Covariance');
title('Covariance Propagation');

% Helper functions
function sigmaPoints = computeSigmaPoints(x, P, lambda)
    n = length(x);
    sigmaPoints = zeros(n, 2*n+1);
    sigmaPoints(:, 1) = x;
    sqrtP = sqrtm((n + lambda) * P);
    for i = 1:n
        sigmaPoints(:, i+1) = x + sqrtP(:, i);
        sigmaPoints(:, i+n+1) = x - sqrtP(:, i);
    end
end

function propagatedSigmaPoints = propagateDynamics(sigmaPoints, dt)
    propagatedSigmaPoints = sigmaPoints + sin(sigmaPoints) * dt;
end

function [x, P] = computeMeanAndCovariance(sigmaPoints, Wm, Wc, Q)
    x = sum(Wm .* sigmaPoints, 2); % Weighted mean
    diff = sigmaPoints - x;
    P = diff * diag(Wc) * diff' + Q; % Weighted covariance
end