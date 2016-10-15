%%
xtrain = male100(1:20,1); %attributes
ttrain= male100(1:20,2); %target labels
xtest = male100(21:end,1); %attributes
ttest= male100(21:end,2); %target labels
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
plot(x,t,'b.','markersize', 25);
hold on;
plot(x,mpred2,'red');

