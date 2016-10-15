%% CTB1_400metree
% Fit of 1st, 2nd, 3rd, and 4th order polynomial to
% the 400 metre sprint data
% We will use 10-fold cross-validation to choose the 'best'
% polynomial order which has lowers error
clear all;close all;
%%
load olympics.mat
%% rescale data
% why do we have to rescale data? worked for first 3 orders
x = male400(:,1);
x = [x - x(1)];
x = [x]./x(2);
%%
t= male400(:,2); %target labels
% ttest= male400(21:end,2); %target labels
plot(x,t,'bx')
hold on;
% plot(xtest,ttest,'ro')
hold off;
%% folding
totalFolds = 10;
N = length(x(:,1));
partition = floor(N/totalFolds);

%%
% Polynomial Order of n
% Changing value of 'order' to choose polynomial order
order = 3;
X = [];
Xtest = [];

% for k = 0:order
    X = [X x.^0];
%     Xtest1 = [Xtest x.^k];
    
    for fold = 0:totalFolds-1
        Xtest = X((fold*partition)+1 : (fold+1)*partition,:);
        Xtrain1 = X(1 : fold*partition,:);
        Xtrain2 = X(fold*partition + (partition + 1) : end,:);
        Xtrain = [Xtrain1; Xtrain2];

        ttrain1 = t(1 : fold*partition,1);
        ttrain2 = t(fold*partition + (partition + 1) : end,1);
        ttrain = [ttrain1; ttrain2];

        % compute model parameters
        w = inv(Xtrain'*Xtrain)*Xtrain'*ttrain; 
        % model predictions
        mpred_test = w'*Xtest';
        % model for observed data
        mpred_observed = w'*Xtrain';
        % compare values
        compare = [Xtest mpred_test' Xtest-mpred_test'];
        % loss

        foldLoss(fold+1,1)  = mean((Xtest-mpred_test').^2);
        trainLoss(fold+1,1) = mean((Xtrain-mpred_observed').^2);
    end
% end

% %% plot of loss
% figure(1);
% subplot(1)
% plot(0:maxorder,mean(foldLoss,1),'linewidth',2)
% xlabel('Model Order');
% ylabel('Loss');
% title('CV Loss');
%% use ^0 to set first data points to 1
%fc = ones(size(xtrain)); 
%fc_test = ones(size(xtest));
%X = [fc];
%Xtest = [fc_test];
%%
% Polynomial Order of n
% Changing value of 'order' to choose polynomial order
% order = 3;
% X = [];
% Xtest = [];

% for k = 0:order
%     X = [X xtrain.^k];
%     Xtest = [Xtest xtest.^k];
% end
% 
% % compute model parameters
% w = inv(X'*X)*X'*ttrain; 
% % model predictions
% mpred_test = w'*Xtest';
% 
% compare = [ttest mpred_test' ttest-mpred_test'];
% %
% plot(xtrain,ttrain,'b.','markersize', 25); %plotting training data as scatterplot (blue)
% hold on;
% plot(xtrain, w'*X','b'); % plotting line of best fit (n order) to training data (blue)
% plot(xtest,ttest,'ro'); % plotting test data as scatterplot (red circles)
% plot(xtest,mpred_test,'r'); % plotting line of best fit (n order) to test data (red)
% set(gca,'Color',[0.5 0.5 0.5]);