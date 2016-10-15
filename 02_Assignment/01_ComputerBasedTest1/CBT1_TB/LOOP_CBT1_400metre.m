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
order = 8;
X = [];
Xtest = [];

 for k = 0:order
     X = [X x.^k];
    
    for fold = 0:totalFolds-1
        Xfold = X((fold*partition)+1 : (fold+1)*partition,:);
        Xtrain1 = X(1 : fold*partition,:);
        Xtrain2 = X(fold*partition + (partition + 1) : end,:);
        Xtrain = [Xtrain1; Xtrain2];

        ttrain1 = t(1 : fold*partition,1);
        ttrain2 = t(fold*partition + (partition + 1) : end,1);
        ttrain = [ttrain1; ttrain2];
        tfold = t((fold*partition)+1 : (fold+1)*partition,:);

        % compute model parameters
        w = inv(Xtrain'*Xtrain)*Xtrain'*ttrain; 
        % model predictions
        mpred_fold = w'*Xfold';
        % model for observed data
        mpred_observed = w'*Xtrain';
        % compare values
        compare = [tfold mpred_fold' tfold-mpred_fold'];
        % loss

        foldLoss(fold+1,k+1)  = mean((mpred_fold' - tfold).^2);
        trainLoss(fold+1,k+1) = mean((mpred_observed' - ttrain).^2);
    end
 end

%% Plot the results
figure(1);
subplot(121)
plot(0:order,mean(foldLoss,1),'linewidth',2) % order vs mean of foldloss(vector)(column = order number?)
xlabel('Model Order');
ylabel('Loss');
title('CV Loss');
subplot(122)
plot(0:order,mean(trainLoss,1),'linewidth',2)
xlabel('Model Order');
ylabel('Loss');
title('Train Loss');