%% cv_demo.m
% From A First Course in Machine Learning, Chapter 1.
% Simon Rogers, 31/10/11 [simon.rogers@glasgow.ac.uk]
% Demonstration of cross-validation for model selection
clear all;close all;
%% Generate some data
% Generate x between -5 and 5
load olympics.mat
%% rescale data
% why do we have to rescale data? worked for first 3 orders!!!!
x = male400(:,1);
x = [x - x(1)];
x = [x]./x(2);
t= male400(:,2); %target labels
N = length(x(:,1));

%% Run a cross-validation over model orders
maxorder = 4;
X = [];
testX = [];
K = 10 %K-fold CV
sizes = repmat(floor(N/K),1,K);
sizes(end) = sizes(end) + N - sum(sizes);
csizes = [0 cumsum(sizes)];

% Note that it is often sensible to permute the data objects before
% performing CV.  It is not necessary here as x was created randomly.  If
% it were necessary, the following code would work:
% order = randperm(N);
% x = x(order); Or: X = X(order,:) if it is multi-dimensional.
% t = t(order);

for k = 0:maxorder
    X = [X x.^k];
%     testX = [testX testx.^k];
    for fold = 1:K
        % Partition the data
        % foldX contains the data for just one fold
        % trainX contains all other data
        
        foldX = X(csizes(fold)+1:csizes(fold+1),:);
        trainX = X;
        trainX(csizes(fold)+1:csizes(fold+1),:) = [];
        foldt = t(csizes(fold)+1:csizes(fold+1));
        traint = t;
        traint(csizes(fold)+1:csizes(fold+1)) = [];
        
        w = inv(trainX'*trainX)*trainX'*traint;
        fold_pred = foldX*w;
        cv_loss(fold,k+1) = mean((fold_pred-foldt).^2);
%         ind_pred = testX*w;
%         ind_loss(fold,k+1) = mean((ind_pred - testt).^2);
        train_pred = trainX*w;
        train_loss(fold,k+1) = mean((train_pred - traint).^2);
    end
end

%% Plot the results
figure(1);
subplot(121)
plot(0:maxorder,mean(cv_loss,1),'linewidth',2)
xlabel('Model Order');
ylabel('Loss');
title('CV Loss');
subplot(122)
plot(0:maxorder,mean(train_loss,1),'linewidth',2)
xlabel('Model Order');
ylabel('Loss');
title('Train Loss');
% subplot(133)
% plot(0:maxorder,mean(ind_loss,1),'linewidth',2)
% xlabel('Model Order');
% ylabel('Loss');
title('Independent Test Loss')