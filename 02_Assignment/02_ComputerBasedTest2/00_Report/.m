%% This program trains a Bayesian classifier
% using the different methods and compares the difference in classification

% We train it on the helthy and diseased datasets
% provided in cbt2data.mat with and without

% Whilst all 4 methods (ML and MAP Naive, ML and MAP wo Naive)

% Based on the following files
% Reference 1: bayesclass.m, A First Course in Machine Learning, Chapter 5.
% Reference 2: bayestraintest.m, Machine Learning (Extended)

% NOTE: do we want to compare the labels between ML, MAP, Naive, WONaive
% build separate program for this without plots?
%% ******************* ANALYSIS *******************
%  ************************************************
%% We load the data
clear all; close all;
load cbt2data.mat
X_train = [diseased'; healthy']; % We put the training data into a single matrix
t_train = [ones(length(diseased'),1); ones(length(healthy'),1).*2]; % We give diseased a label of 1, healthy a label of 2
X_new = newpts'; % load new points
savePlots = 0; % 0 = don't save plots, 1 = save plots

%% We compute the prior
class1Total = sum(t_train==1); % We count the number of diseased patients
class2Total = sum(t_train==2); % We count the number of healthy patients
probClass1 = class1Total/(class1Total + class2Total); % We compute the dieased prior based on frequency
probClass2 = class2Total/(class1Total + class2Total); % We compute the healthy prior based on frequency
probPrior = [probClass1; probClass2]; % We store priors in vector, make matrix and put in other prior values?

% Uncomment to experiment with different priors:
probPrior = [.99; 0.01]; % We give diseased class a strong prior
% probPrior = [0.5; 0.5]; % We compute prior based on 1/C
% probPrior = [1,0];

%% Find the mean and variance 
% Using the Naive (independence) assumption
cl = unique(t_train); % get total number of classes from test set

for c = 1:length(cl) % We loop over the number of classes (1,diseased and 2,healthy)
    pos = find(t_train==cl(c)); % We store position of class label in vector
    Class_mean(c,:) = mean(X_train(pos,:)); % We compute the mean for both attributes and for each class
    Class_var(c,:) = var(X_train(pos,:),1); % We compute the variance for both attributes and for each class
    
    pos = find(t_train==cl(c)); % We store position of class label in vector
    WONclass_mean(c,:) = mean(X_train(pos,:)); % class-wise & attribute-wise mean
    WONclass_var(:,:,c) = cov(X_train(pos,:),1); % class-wise & attribute-wise variance
end

%% Maximum likelihood (ML) NAIVE classification for new data
MLclass_probs_new = [];
MAPclass_probs_new = [];

for c = 1:length(cl)
    sigma_naive = diag(Class_var(c,:)); % we convert row to diagonal elements
    diff_train = [X_new(:,1)-Class_mean(c,1) X_new(:,2)-Class_mean(c,2)]; % We compute difference between data point and respective mean
    const = 1/sqrt((2*pi()^size(X_new,2)*det(sigma_naive))); % we compute the constanst for the gaussian function
    MLclass_probs_new = [MLclass_probs_new const*exp(-1/2*diag(diff_train * inv(sigma_naive) * diff_train'))]; % We use the gaussian function to compute ML
    MAPclass_probs_new = [MAPclass_probs_new const*exp(-1/2*diag(diff_train * inv(sigma_naive) * diff_train'))* probPrior(c)]; % We use the gaussian function to compute MAP
end

%% Maximum likelihood (ML) WITHOUT naive classification for new data
% Without the naive assumption, thus we don't take the diag of variance
% (see sigma)
MLWONclass_probs_new = [];
MAPWONclass_probs_new = [];

for c = 1:length(cl)
    sigma = WONclass_var(:,:,c); % we convert row to diagonal elements
    diff_train = [X_new(:,1)-WONclass_mean(c,1) X_new(:,2)-WONclass_mean(c,2)]; % We compute difference between data point and respective mean
    const = 1/sqrt((2*pi()^size(X_new,2)*det(sigma))); % we compute the constanst for the gaussian function
    MLWONclass_probs_new = [MLWONclass_probs_new const * exp(-1/2*diag(diff_train * inv(sigma) * diff_train'))]; % We use the gaussian function to compute MLend
    MAPWONclass_probs_new = [MAPWONclass_probs_new const * exp(-1/2*diag(diff_train * inv(sigma) * diff_train'))* probPrior(c)]; % We use the gaussian function to compute MLend
end

%% Classify each example
% We classify each example by getting the column index which has the higher
% maximum likelihood (ML). Column 1 and column 2 give the ML of belonging to
% each class respectively. For example, a column 1 has a ML of 0.7 and
% column 2 is 0.3, therefore it returns the index (or column) 1.
difference = [];
countDisease = [];
countHealthy = [];

[~,MLClass_new] = max(MLclass_probs_new, [], 2); 
[~,MLWONClass_new] = max(MLWONclass_probs_new, [], 2); 
[~,MAPClass_new] = max(MAPclass_probs_new, [], 2); 
[~,MAPWONClass_new] = max(MAPWONclass_probs_new, [], 2);

countDisease = [sum(MLClass_new==1); sum(MLWONClass_new==1); sum(MAPClass_new==1); sum(MAPWONClass_new==1)];
countHealthy = [sum(MLClass_new==2); sum(MLWONClass_new==2); sum(MAPClass_new==2); sum(MAPWONClass_new==2)];
difference = [length(MLClass_new)-sum(MLClass_new==MLWONClass_new), length(MAPClass_new)-sum(MAPClass_new==MAPWONClass_new), length(MLClass_new)-sum(MLClass_new==MAPClass_new)];