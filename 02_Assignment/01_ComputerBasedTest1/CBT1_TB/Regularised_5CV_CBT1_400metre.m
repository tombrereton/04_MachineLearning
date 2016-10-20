%% CTB1_400metree
% Fit of 1st, 2nd, 3rd, and 4th order polynomial to
% the 400 metre sprint data
% We will use 10-fold cross-validation to choose the 'best'
% polynomial order which has lowers error
clear all;close all;
%%
load olympics.mat
%% rescale data
% why do we have to rescale data? worked for first 3 orders!!!!
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


%% folding details
totalFolds = 10; % total number of folds
Npartition = length(x(:,1)); % we find the length of the first column in matrix x
partition = floor(Npartition/totalFolds); % we determine the size of the partition (2.7 in this case) and take the floor
%% Polynomial Order of n
% Changing value of 'order' to choose polynomial order
order = 4; % order of polynomial
X = [];
Xtest = [];

%% regularisation parameters
lambda = 10;
N = length(x(:,1));
 for k = 0:order % we loop to compute CV loss and Train loss for each order of the polynomial
     X = [X x.^k]; % we add a column to matrix for every order n where the new column is x^k
    
    for fold = 0:totalFolds-1 % we loop again to change the partition being tested and trained
        Xfold = X((fold*partition)+1 : (fold+1)*partition,:); % we slice X to get the rows for the CV (test) partition (1)
        Xtrain1 = X(1 : fold*partition,:); % we slice X to get the training partition before the CV partition (2)
        Xtrain2 = X(fold*partition + (partition + 1) : end,:); % we slice X to get the training partition afer the CV partition (3)
        Xtrain = [Xtrain1; Xtrain2]; % we join the training paritions into one matrix (4)

        tfold = t((fold*partition)+1 : (fold+1)*partition,:); % we partition the labels the same as (1) to validate the results
        ttrain1 = t(1 : fold*partition,1); % we partition the labels the same as (2) to train the model
        ttrain2 = t(fold*partition + (partition + 1) : end,1); % we partition the labels the same as (3) train the model
        ttrain = [ttrain1; ttrain2]; % we join the label training data into one matrix similar to (4)
        

        
        E = eye(size(k+1));
        % we learn the model parameters from the training data and label(5)
        w = inv(Xtrain'*Xtrain + N*lambda*E)*Xtrain'*ttrain; 
        % we find the predicted results using the params in (5) and (1)
        mpred_fold = w'*Xfold';
        % we see how well the model reflects the observed (training) data
        mpred_observed = w'*Xtrain';
        % we compare the CV labels to the predicted results
        compare = [tfold mpred_fold' tfold-mpred_fold'];

        % we store the mean of the squared difference of CV labels and
        % predicted results to be plotted later
        foldLoss(fold+1,k+1)  = mean((mpred_fold' - tfold).^2);
        trainLoss(fold+1,k+1) = mean((mpred_observed' - ttrain).^2);

    end

%     % plots and saves to file
%     plot(x,t,'bx');
%     plotTitle = strcat('Fit of Polynomial Model x.^',int2str(k+1),' to The Data'); 
%     title(plotTitle);
%     xlabel('Rescaled Year Data');
%     ylabel('Time (seconds)');
%     hold on;
%     plot(x,w'*X','r');
%     filename = strcat('model',int2str(k+1),'.png'); 
% %     saveas(gcf,filename);
%     clf;
    
 end

%% Plot the results
% reference: A first course in machine learning chapter one code

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

filename = strcat('CVLossANDTrainLoss',int2str(order),'.png'); 
saveas(gcf,filename);