%% Clear
clear all;close all;
%% Variables
x = [2.2, 3.3, 5, 4.8, 6.7, 8.9]';
y = [1.1, 2.1, 2.3, 3.5, 4.9, 5.9]';
% xnew = [1.1,2.1,3.1,4.0,4.9,5.9]';
%% Model Order;
order = 1;
X = [];
Xnew = [];
for i = 0 : order;
    X = [X x.^i];
%     Xnew = [Xnew xnew.^i];
end
%% Learn model
w = inv(X'*X)*X'*y;
%% Compute Predictions
% pred = w' * Xnew'; 
pred = X * w;
%% Loss
difference = [x, pred, pred - x];
loss = mean((pred - y).^2);
%% Learn model numerically
meanx = mean(x);
meany = mean(y);
meanxy = mean(x.*y);
sqmeanx = mean(x)^2;
meanxsq = mean(x.^2);
w1 = (meanxy - meanx * meany) / (meanxsq - sqmeanx);
w0 = meany - w1 * meanx;
%% Plot model
plot(x,y,'bx');
hold on;
plot(x, X * w, 'r');
% plot(x, pred, 'b');