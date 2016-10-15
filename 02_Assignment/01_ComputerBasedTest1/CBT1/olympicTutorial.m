%%
x = male100(:,1); %attributes
t= male100(:,2); %target labels
%% compute averages
xm = mean(x);
tm = mean(t);
xsm = mean(x.^2);
xms = xm^2;
xtm = mean(x.*t);
%% compute w0 and w1 - model parameters
w1 = (xtm - xm*tm)/(xsm - xms);
w0 = tm - w1*xm;
%% compute model predictions
mpred = w0 + w1*x;
%%
plot(x,t,'b.','markersize',20)
%%
hold on;
plot(x,mpred,'r');
%%
x2012 = 2012;
x2016 = 2016;
x2020 = 2020;
t2012 = w0 + w1*x2012;
t2016 = w0 + w1*x2016;
t2020 = w0 + w1*x2020;