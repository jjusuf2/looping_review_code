%% RunBayesianThresholding.m
% This script runs Bayesian thresholing on simulated 3D trajectories and
% compares it to simple thresholding. The trajectories are simulated from a
% Rouse model, but conditionally on a defined proximity interval. That
% means, the true distance is larger than the proximity threshold alpha
% everywhere except inside the considered proximity window, where it is
% smaller than alpha. We then use Bayesian and simple thresholding to
% analyze how the inferred proximity durations compare to the true one. The
% script uses parallelization to simulate 10k trajectories, which should
% finish within a few minutes on a laptop / PC.

clear;
close all;

rng(100);

addpath('mvrandn');

cols = GetDefaultColors;

dim = 3;

% Parameters for the Rouse dynamics.
deltaS = 16;
D = 289;
gamma = 1;
kappa = 1/17;
T = 150;
dt = 2;
tGrid = 0:dt:T;
K = length(tGrid);

% Localization error.
sigma_loc = sqrt(2)*20;

% Creates the Rouse model and associated statistics.
rm = CreateRouseModel(tGrid, deltaS, D, gamma, kappa, sigma_loc);

% Defines the vector of proximity intervals in frames.
winSizeVec = [2:25];

% Select which proximity intervals should be plotted as examples.
selWin = [15, 25];

% Define proximity threshold
alpha = 50;

for u=1:length(winSizeVec)

    winSize = winSizeVec(u);
    contactInterval = zeros(size(tGrid));

    kStart = floor(K/2-winSize/2);
    kEnd = kStart + winSize-1;

    contactInterval(kStart:kEnd)=1;

    % Simulates K trajectories with defined proximity duration and
    % calculate posterior distances using Bayesian inference (muPost). X is
    % the true distance and Y is the distance including localization error.
    [X, Y, muPost] = SimulateProximityCond(tGrid, rm, contactInterval, alpha, 10000, dim);

    % Ground truth (must be the same as the prescribed contact interval by
    % construction).
    dX = max(abs(X), [], 3);

    % Thresholded distance Y.
    dY = max(abs(Y), [], 3);

    % Thresholded posterior distance muPost.
    dmu = max(abs(muPost), [], 3);

    % Extracts inferred proximity intervals.
    [timeI_True, EventTimesTotal_True, ~] = DetectEventTimes(dX<alpha);
    [timeI_Bayes, ~, EventTimesTotal_Bayes] = DetectEventTimes(dmu<alpha);
    [timeI_Threshold, ~, EventTimesTotal_Threshold] = DetectEventTimes(dY<alpha);

    % Calculate summary statistics
    dProbBayes(u) =  mean(sum(and(dX<alpha, dmu<alpha), 2))/winSize;
    meanTau_Bayes(u) = nanmean(EventTimesTotal_Bayes);
    quantiles_Bayes([1,2], u) = 2.5*nanstd(EventTimesTotal_Bayes)/sqrt(length(EventTimesTotal_Bayes));
    dProbThreshold(u) =  mean(sum(and(dX<alpha, dY<alpha), 2))/winSize;
    meanTau_Threshold(u) = nanmean(EventTimesTotal_Threshold);
    quantiles_Threshold([1, 2], u) = 2.5*nanstd(EventTimesTotal_Threshold)/sqrt(length(EventTimesTotal_Threshold));

    winSizeIdx = find(winSize == selWin);

    % Plotting
    if (~isempty(winSizeIdx))
        figure(1)
        subplot(3,length(selWin),winSizeIdx);
        dt = tGrid(2) - tGrid(1);
        plot(tGrid, dX(1, :), '-', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1.5); hold on;
        plot(tGrid, dY(1, :), '.', 'Color', cols(2, :));
        plot(tGrid, dmu(1, :), '--', 'Color', cols(1, :), 'LineWidth', 1.5);
        plot(tGrid, alpha*ones(size(tGrid)), '--', 'Color',[0.2, 0.2, 0.2], 'LineWidth', 1.5);
        ylim([0, 400]);
        ylabel('3D Distance');
        subplot(3,length(selWin),1*length(selWin)+winSizeIdx);
        bar(tGrid+dt/2, dY(1, :)<alpha, 'BarWidth', 1, 'FaceColor', cols(2, :), 'LineStyle','none'); hold on;
        stairs(tGrid, dX(1, :)<alpha, 'LineWidth', 1.5, 'Color', [0.5, 0.5, 0.5]);
        legend('Inferred Interval', 'True Interval');
        ylabel('Inferred proximity (Threshold)');
        ylim([0, 1.5]);
        subplot(3,length(selWin),2*length(selWin)+winSizeIdx);
        bar(tGrid+dt/2, dmu(1, :)<alpha, 'BarWidth', 1, 'FaceColor', cols(1, :), 'LineStyle','none'); hold on;
        stairs(tGrid, dX(1, :)<alpha, 'LineWidth', 1.5, 'Color', [0.5, 0.5, 0.5]);
        ylabel('Inferred proximity (Bayes)');
        ylim([0, 1.5]);
        xlabel('Time (min)');
        legend('Inferred Interval', 'True Interval');

    end

end


dim = 3;

r = alpha / sigma_loc;
% Calculate the theoretical detection probability for an ideal contact.
pEvent = ncx2cdf(r.^2, dim, 0);

% Calculate the mean inferred lifetime for an ideal contact.
tau_mri_L2 = 1./(1 - pEvent);
relDurationLimit = 1 - abs(tau_mri_L2 - winSizeVec)./winSizeVec;
dProbLimit = repmat(pEvent, 1, length(winSizeVec));

% Plot summary statistics / results
figure;
subplot(1,2,1);
plot(winSizeVec*dt, meanTau_Bayes*dt, '.-', 'Color', cols(1, :), 'LineWidth', 1.5, 'MarkerSize', 12); hold on;
plot(winSizeVec*dt, meanTau_Threshold*dt, '.-', 'Color', cols(2, :), 'LineWidth', 1.5, 'MarkerSize', 12); hold on;
plot([0, winSize*dt], [0, winSize*dt], '--', 'LineWidth', 1.5, 'Color', [0.5, 0.5, 0.5]); hold on;
plot([0, 25*dt], [1, 1]*tau_mri_L2*dt, 'k--', 'LineWidth',1.5);
xlim([0, 25*dt]);
ylim([0, 25*dt]);
xlabel('True contact lifetime [min]');
ylabel('Inferred contact lifetime [min]');

subplot(1,2,2);
relDurationBayes = 1-abs(meanTau_Bayes-winSizeVec)./winSizeVec;
relDurationThreshold = 1-abs(meanTau_Threshold-winSizeVec)./winSizeVec;
plot(dProbBayes, relDurationBayes, '.-', 'Color', cols(1, :), 'LineWidth', 1.5); hold on;
plot(dProbThreshold, relDurationThreshold, '-', 'Color', cols(2, :), 'LineWidth', 1.5); hold on;

for k=1:length(dProbBayes)
    plot(dProbBayes(k), relDurationBayes(k), '.', 'MarkerSize', 8 + 12*(winSizeVec(k))/30, 'Color', cols(1, :));
    plot(dProbThreshold(k), relDurationThreshold(k), '.', 'MarkerSize', 8 + 12*(winSizeVec(k))/30, 'Color', cols(2, :));
end

xlabel('True positive rate');
ylabel('Interval agreement');
