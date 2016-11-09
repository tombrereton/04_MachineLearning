%% bayestraintest.m
% Based on bayesclass.m
% in A First Course in Machine Learning, Chapter 5.
% Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
% Bayesian classifier

clear all;close all;

%% Load the data
load bc_data;
% split data (X and t) in training and testing subsets
X_train = X([1:24 31:54 61:84],:); % 24 samples from each class in training
X_test = X([25:30 55:60 85:90],:);
t_train = t([1:24 31:54 61:84],:); % 6 samples from each class in testing
t_test = t([25:30 55:60 85:90],:);

%% Plot the data
cl = unique(t_train); % find the number of unique classes from labels
col_train = {'go','bo','ko'};
col_test = {'gx','bx','kx'};
figure(1);
hold on
for c = 1:length(cl)
    pos_train = find(t_train==cl(c));
    pos_test = find(t_test==cl(c));
    plot(X_train(pos_train,1),X_train(pos_train,2),col_train{c},...
        'markersize',10,'linewidth',2);
    plot(X_test(pos_test,1),X_test(pos_test,2),col_test{c},...
        'markersize',10,'linewidth',2);
end
xlim([-3 7]), ylim([-6 6]) % set the x-axis/y-axis display range

%% Fit class-conditional Gaussians for each class, from training samples
% Using the Naive (independence) assumption
for c = 1:length(cl)
    pos = find(t_train==cl(c));
    % Find the means
    class_mean(c,:) = mean(X_train(pos,:)); % class-wise & attribute-wise mean
    class_var(c,:) = var(X_train(pos,:),1); % class-wise & attribute-wise variance
end

%% Compute the predictive probabilities (with Naive assumption)
% for training samples and testing samples
probab_train = [];
probab_test = [];
% testSigma = diag(diag(class_var));
for c = 1:length(cl)
    sigmac = diag(class_var(c,:));
    
    diff_train = [X_train(:,1)-class_mean(c,1) X_train(:,2)-class_mean(c,2)];
    const_train = 1/sqrt((2*pi)^size(X_train,2) * det(sigmac));
    probab_train(:,c) = const_train*exp(-0.5*diag(diff_train*inv(sigmac)*diff_train'));
    
    diff_test = [X_test(:,1)-class_mean(c,1) X_test(:,2)-class_mean(c,2)];
    const_test = 1/sqrt((2*pi)^size(X_test,2) * det(sigmac));
    probab_test(:,c) = const_train*exp(-0.5*diag(diff_test*inv(sigmac)*diff_test')); % shoud this be const_test not const_train?
    % this is using maximum likelihood, given the uniform size of classes
end

% get proper probability estimates
% probab_train = probab_train./repmat(sum(probab_train,2),[1,3]); % doesn't use prior?
% probab_test = probab_test./repmat(sum(probab_test,2),[1,3]);

%% find class label predictions from probabilities (with Naive assumption)
[~,p_train_with] = max(probab_train,[],2); % assign labels as per highest probability
compare_train_with=[t_train p_train_with t_train-p_train_with]; % for comparison
error_train_with=sum(t_train~=p_train_with); % error - # of mis-classifications
[~,p_test_with] = max(probab_test,[],2); % assign labels as per highest probability
compare_test_with=[t_test p_test_with t_test-p_test_with];  % for comparison
error_test_with=sum(t_test~=p_test_with); % error - # of mis-classifications

%% Plot the data and predictions (with Naive assumption)
cl = unique(t_train); % find the number of unique classes from labels
col_train = {'go','co','ko'};
col_test = {'rx','bx','kx'};
figure(1);
hold on
for c = 1:length(cl)
    pos_train = find(t_train==cl(c));
    pos_test = find(p_test_with==cl(c));
    plot(X_train(pos_train,1),X_train(pos_train,2),col_train{c},...
        'markersize',10,'linewidth',2);
    plot(X_test(pos_test,1),X_test(pos_test,2),col_test{c},...
        'markersize',10,'linewidth',2);
end

%% Fit class-conditional Gaussians for each class, from training samples
% without using Naive assumption
class_mean = [];
class_var = [];
for c = 1:length(cl)
    pos = find(t_train==cl(c));
    % Find the means
    class_mean(c,:) = mean(X_train(pos,:)); % class-wise & attribute-wise mean
    class_var(:,:,c) = cov(X_train(pos,:),1); % class-wise & attribute-wise co-variance
end

%% Compute the predictive probabilities (without Naive assumption)
% for training samples and testing samples
probab_train = [];
probab_test = [];
for c = 1:length(cl)
    sigmac = class_var(:,:,c); % this is the main difference, in with/without using Naive assumption
    
    diff_train = [X_train(:,1)-class_mean(c,1) X_train(:,2)-class_mean(c,2)];
    const_train = 1/sqrt((2*pi)^size(X_train,2) * det(sigmac));
    probab_train(:,c) = const_test*exp(-0.5*diag(diff_train*inv(sigmac)*diff_train'));
    
    diff_test = [X_test(:,1)-class_mean(c,1) X_test(:,2)-class_mean(c,2)];
    const_test = 1/sqrt((2*pi)^size(X_test,2) * det(sigmac));
    probab_test(:,c) = const_test*exp(-0.5*diag(diff_test*inv(sigmac)*diff_test'));
    % this is using maximum likelihood, given the uniform size of classes
end

% get proper probability estimates
probab_train = probab_train./repmat(sum(probab_train,2),[1,3]);
probab_test = probab_test./repmat(sum(probab_test,2),[1,3]);

%% find class label predictions from probabilities (without Naive assumption)
[~,p_train_without] = max(probab_train,[],2); % assign labels as per highest probability
compare_train_without=[t_train p_train_without t_train-p_train_without]; % for comparison
error_train_without=sum(t_train~=p_train_without); % error - # of mis-classifications
[~,p_test_without] = max(probab_test,[],2); % assign labels as per highest probability
compare_test_without=[t_test p_test_without t_test-p_test_without]; % for comparison
error_test_without=sum(t_test~=p_test_without); % error - # of mis-classifications

%% Plot the data and predictions (without Naive assumption)
cl = unique(t_train); % find the number of unique classes from labels
col_train = {'go','bo','ko'};
col_test = {'gx','bx','kx'};
figure(1);
hold on
for c = 1:length(cl)
    pos_train = find(t_train==cl(c));
    pos_test = find(p_test_without==cl(c));
    plot(X_train(pos_train,1),X_train(pos_train,2),col_train{c},...
        'markersize',10,'linewidth',2);
    plot(X_test(pos_test,1),X_test(pos_test,2),col_test{c},...
        'markersize',10,'linewidth',2);
end