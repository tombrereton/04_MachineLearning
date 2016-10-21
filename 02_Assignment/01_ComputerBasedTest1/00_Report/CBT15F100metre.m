%% CTB1_100metre
% Fit of 1st and 4th order polynomial to
% the 100 metre sprint data
% We vary the regularisation factor, lambda,
% to determine the best value for the
% polynomial models.
clear all;close all;
load olympics.mat
%% Preamble and rescaling data
x = male100(:,1);
x = [x - x(1)];
x = [x]./x(2);
t= male100(:,2); %target labels
%% Folding details
totalFolds = 5; % total number of folds
Npartition = length(x(:,1)); % we find the length of the first column in matrix x
partition = repmat(floor(Npartition/totalFolds),1,totalFolds); % we determine the size of the partition (2.7 in this case) and take the floor
partition(end) = Npartition - (totalFolds-1)*partition(1,1);
pcum = [0 cumsum(partition)];
%% Polynomial Order of n
% Change value of 'order' to choose polynomial order n
order = 1; % order of polynomial
X = [];
orderFoldMat = [];
orderTrainMat = [];
matCVLoss = [];
matLambda = [];

for k = 0:order % we loop to build the attribute set for order n
    X = [X x.^k]; % we add a column to matrix for every order n where the new column is x^k
end

for lambda = 0:0.1:1 % we loop through lambda values to vary the regularisation
    for fold = 1:totalFolds % we loop again to change the partition being tested and trained
        Xfold = X(pcum(fold)+1 : pcum(fold+1),:); % we slice X to get the rows for the CV (test) partition (1)
        Xtrain1 = X(1 : pcum(fold),:); % we slice X to get the training partition before the CV partition (2)
        Xtrain2 = X(pcum(fold+1)+1 : end,:); % we slice X to get the training partition afer the CV partition (3)
        Xtrain = [Xtrain1; Xtrain2]; % we join the training paritions into one matrix (4)

        tfold = t(pcum(fold)+1 : pcum(fold+1),:); % we partition the labels the same as (1) to validate the results
        ttrain1 = t(1 : pcum(fold),:); % we partition the labels the same as (2) to train the model
        ttrain2 = t(pcum(fold+1)+1 : end,:); % we partition the labels the same as (3) train the model
        ttrain = [ttrain1; ttrain2]; % we join the label training data into one matrix similar to (4)


        E = eye(size(Xtrain,2));
        % we learn the model parameters from the training data and labels(5)
        w = inv(Xtrain'*Xtrain + Npartition*lambda*E)*Xtrain'*ttrain; 
        % we find the predicted results using the params in (5) and (1)
        mpred_fold = Xfold*w;
        % we see how well the model reflects the observed (training) data
        mpred_observed = Xtrain*w;
        % we compare the CV labels to the predicted results
        compare = [tfold mpred_fold tfold-mpred_fold];

        % we store the mean squared difference of CV labels and
        % predicted results, to plot later
        foldLoss(fold,1)  = mean((mpred_fold - tfold).^2);
    end
    
    matCVLoss = [matCVLoss mean(foldLoss)];
    matLambda = [matLambda lambda];
    
    % plots and saves to file
    plot(x,t,'bx');
    plotTitle = strcat('Fit of Polynomial Model x.^',int2str(k),' with lambda = ',num2str(lambda),' to Mens 100m Data'); 
    title(plotTitle);
    xlabel('Rescaled Year Data','fontsize',16);
    ylabel('Time (seconds)','fontsize',16);
    hold on;
    plot(x,w'*X','r');
    filename = strcat('LambdaModel',num2str(lambda),'_',int2str(order),'.png'); 
    saveas(gcf,filename);
    clf;
end
%% Plot the results
% Reference: A First Course in Machine Learning, Chapter One, cv_demo.m
figure(1);
plot(matLambda,matCVLoss,'linewidth',2);
xlabel('Lambda','fontsize',16);
ylabel('Loss','fontsize',16);
title('CV Loss vs Lambda for Olympic Mens 100m Data','fontsize',16);
filename = strcat('CVLossvsLambda',int2str(order),'.png'); 
saveas(gcf,filename);