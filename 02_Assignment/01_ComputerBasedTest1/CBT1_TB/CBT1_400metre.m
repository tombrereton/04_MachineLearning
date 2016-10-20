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
xtrain = x(1:20,1); %attributes
ttrain= male400(1:20,2); %target labels
xtest = x(21:end,1); %attributes
ttest= male400(21:end,2); %target labels
t = male400(:,2);
plot(xtrain,ttrain,'bo')
hold on;
plot(xtest,ttest,'ro')
hold off;%%
%%
fc = ones(size(xtrain));
fc_test = ones(size(xtest));
X = [fc xtrain];
Xtest = [fc_test xtest];
%% compute model parameters
w = inv(X'*X)*X'*ttrain;
%% model predictions
mpred_test = w'*Xtest';
%%
compare = [ttest mpred_test' ttest-mpred_test'];
%%
plot(xtrain,ttrain,'b.','markersize', 25);
hold on;
plot(xtrain, w'*X','b');
plot(xtest,ttest,'ro');
plot(xtest,mpred_test,'red');
hold off;
%% Polynomial Order of 2
% Changing data to order of 2
X2 = [fc xtrain xtrain.^2];
Xtest2 = [fc_test xtest xtest.^2];
%% compute model parameters ^2
w2 = inv(X2'*X2)*X2'*ttrain; 
%% model predictions
mpred_test2 = w2'*Xtest2';
%%
compare2 = [ttest mpred_test2' ttest-mpred_test2'];
%%
plot(xtrain,ttrain,'b.','markersize', 25); %plotting training data as scatterplot (blue)
hold on;
plot(xtrain, w2'*X2','b'); % plotting line of best fit (2nd order) to training data (blue)
plot(xtest,ttest,'ro'); % plotting test data as scatterplot (red circles)
plot(xtest,mpred_test2,'red'); % plotting line of best fit (2nd order) to test data (red)
%% Polynomial Order of 3
% Changing data to order of 3
X3 = [X2 xtrain.^3];
Xtest3 = [Xtest2 xtest.^3];
%% compute model parameters ^3
w3 = inv(X3'*X3)*X3'*ttrain; 
%% model predictions
mpred_test3 = w3'*Xtest3';
%%
compare3 = [ttest mpred_test3' ttest-mpred_test3'];
%%
plot(xtrain,ttrain,'b.','markersize', 25); %plotting training data as scatterplot (blue)
hold on;
plot(xtrain, w3'*X3','b'); % plotting line of best fit (3rd order) to training data (blue)
plot(xtest,ttest,'mo'); % plotting test data as scatterplot (red circles)
plot(xtest,mpred_test3,'m'); % plotting line of best fit (3rd order) to test data (red)
%% Polynomial Order of 4
% Changing data to order of 4
X4 = [fc xtrain xtrain.^2 xtrain.^3 xtrain.^4];
Xtest4 = [fc_test xtest xtest.^2 xtest.^3 xtest.^4];
X4_plot = [x.^0 x x.^2 x.^3 x.^4];

%% regularisation parameters
lambda = .01;
order = 5;
N = order;
E = eye(size(order));
% compute model parameters ^4
w4 = inv(X4'*X4 +N*lambda*E)*X4'*ttrain; 
% model predictions
mpred_test4 = w4'*Xtest4';
%
compare4 = [ttest mpred_test4' ttest-mpred_test4'];
%
clf;
plot(t,w4'*X4_plot','rx');
plot(xtrain,ttrain,'b.','markersize', 25); %plotting training data as scatterplot (blue)
hold on;
plot(xtrain, w4'*X4','b'); % plotting line of best fit (4th order) to training data (blue)
plot(xtest,ttest,'ro'); % plotting test data as scatterplot (red circles)
plot(xtest,mpred_test4,'r'); % plotting line of best fit (4th order) to test data (red)
%% Polynomial Order of 5
% Changing data to order of 5
X5 = [fc xtrain xtrain.^2 xtrain.^3 xtrain.^4 xtrain.^5];
Xtest5 = [fc_test xtest xtest.^2 xtest.^3 xtest.^4 xtest.^5];
% %% compute model parameters ^5
% w5 = inv(X5'*X5)*X5'*ttrain; 
% %% model predictions
% mpred_test5 = w5'*Xtest5';
% %%
% compare5 = [ttest mpred_test5' ttest-mpred_test5'];
% %%
% plot(xtrain,ttrain,'b.','markersize', 25); %plotting training data as scatterplot (blue)
% hold on;
% plot(xtrain, w5'*X5','b'); % plotting line of best fit (4th order) to training data (blue)
% plot(xtest,ttest,'go'); % plotting test data as scatterplot (red circles)
% plot(xtest,mpred_test5,'g'); % plotting line of best fit (4th order) to test data (red)
set(gca,'Color',[0.5 0.5 0.5]);