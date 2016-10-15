%%
x = male100(:,1); %attributes
t= male100(:,2); %target labels
%%
fc = ones(size(x));
X = [fc x];
%% compute model parameters
w = inv(X'*X)*X'*t;
%% model predictions
mpred2 = w'*X';
%%
plot(x,t,'b.','markersize', 25);
hold on;
plot(x,mpred2,'red');