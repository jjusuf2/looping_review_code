%% PlotThresholdingMetrics.m
% This script plots the probability of successful proximity detection and 
% mean inferred lifetime for an ideal contact. These numbers provide
% best-case performance metrics for thresholding-based proximity detection.

clear;
close all;

dim = 3;

r = linspace(0, 3, 100);

%add point r=1 as a reference;
r = union(r, 1);
refIdx = find(r==1);

% calculate detection probability
pEvent = ncx2cdf(r.^2, dim, 0);

% calculate tau (mean inferred lifetime)
tau_mri_L2 = 1./(1 - pEvent);

subplot(1,2,1);
plot(r, pEvent, 'LineWidth', 1.5); hold on;
plot(r(refIdx), pEvent(refIdx), '.', 'MarkerSize', 12);
lString = sprintf('$p_1(1)\\simeq %.2f$', pEvent(refIdx));
text(r(refIdx)+0.1, pEvent(refIdx), lString, 'Interpreter','latex');
xlabel('Relative threshold $r= \frac{\alpha}{\sigma}$', 'Interpreter','latex');
ylabel('Detection probability $p_1(r)$', 'Interpreter','latex');

subplot(1,2,2);
plot(r, tau_mri_L2, 'LineWidth', 1.5); hold on;
plot(r(refIdx), tau_mri_L2(refIdx), '.', 'MarkerSize', 12);
lString = sprintf('$\\hat{\\tau}(1)\\simeq %.2f$', tau_mri_L2(refIdx));
text(r(refIdx)-0.5, tau_mri_L2(refIdx)+2, lString, 'Interpreter','latex')
xlabel('Relative threshold $r = \frac{\alpha}{\sigma}$', 'Interpreter','latex');
ylabel('Mean inferred duration $\hat{\tau}(r)$ [frames]', 'Interpreter','latex');