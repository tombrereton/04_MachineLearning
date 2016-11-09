%% bayesclass.m
% From A First Course in Machine Learning, Chapter 5.
% Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
% Bayesian classifier
clear all;close all;

%% Load the data
load bc_data

% Plot the data

cl = unique(t); % get total number of classes from test set
col = {'ko','kd','ks'}; 
fcol = {[1 0 0],[0 1 0],[0 0 1]}; % colour of markers
figure(1);
hold off
for c = 1:length(cl)
    pos = find(t==cl(c)); % get position, or row, of each data point in class
    plot(X(pos,1),X(pos,2),col{c},... % plot points of column 1 on x and column 2 on y
        'markersize',10,'linewidth',2,...
        'markerfacecolor',fcol{c});
    hold on
end
xlim([-3 7]) % set x scale on graph
ylim([-6 6]) % set y scale on graph



%% Fit class-conditional Gaussians for each class
% Using the Naive (independence) assumption
for c = 1:length(cl)
    pos = find(t==cl(c)); % we get the position, or row, of each point in a class
    % Find the means
    class_mean(c,:) = mean(X(pos,:)); % we find the means for each class
    class_var(c,:) = var(X(pos,:),1); % we find the variance for each class
end


%% Compute the predictive probabilities
[Xv,Yv] = meshgrid(-3:0.1:7,-6:0.1:6); % what is this mesh grid for??, 
Probs = [];
for c = 1:length(cl)
    temp = [Xv(:)-class_mean(c,1) Yv(:)-class_mean(c,2)]; % xnew - mean, tnew - mean
    tempc = diag(class_var(c,:)); % only diagonal, assuming naive? , (co)variance for class
    const = -log(2*pi) - log(det(tempc)); % is it actually easier with log?, why not -1/2 out front?
    Probs(:,:,c) = reshape(exp(const - 0.5*diag(temp*inv(tempc)*temp')),size(Xv)); % reshape exp(of function) into size of Xv
end

% Probs = Probs./repmat(sum(Probs,3),[1,1,3]); % what is this? full bayes rule? how?, is value of contour line?
% above, divide by sum of probs i.e. divide by marginal??? how does
% function do this???
%% Plot the predictive contours
figure(1);hold off
for i = 1:3
    subplot(1,3,i);
    hold off
    for c = 1:length(cl) % plot the data points for each class
        pos = find(t==cl(c));
        plot(X(pos,1),X(pos,2),col{c},...
            'markersize',10,'linewidth',2,...
            'markerfacecolor',fcol{c});
        hold on
    end
    xlim([-3 7]) % set scales
    ylim([-6 6])
    
    contour(Xv,Yv,Probs(:,:,i)); % plot contour line
    ti = sprintf('Probability contours for class %g',i);
    title(ti);
end


%% Repeat without Naive assumption
class_var = [];
for c = 1:length(cl)
    pos = find(t==cl(c));
    % Find the means
    class_mean(c,:) = mean(X(pos,:));
    class_var(:,:,c) = cov(X(pos,:),1);
end


%% Compute the predictive probabilities
[Xv,Yv] = meshgrid(-3:0.1:7,-6:0.1:6);
Probs = [];
for c = 1:length(cl)
    temp = [Xv(:)-class_mean(c,1) Yv(:)-class_mean(c,2)];
    tempc = class_var(:,:,c);
    const = -log(2*pi) - log(det(tempc));
    Probs(:,:,c) = reshape(exp(const - 0.5*diag(temp*inv(tempc)*temp')),size(Xv));
end

Probs = Probs./repmat(sum(Probs,3),[1,1,3]);

%% Plot the predictive contours
figure(1);hold off
for i = 1:3
    subplot(1,3,i);
    hold off
    for c = 1:length(cl)
        pos = find(t==cl(c));
        plot(X(pos,1),X(pos,2),col{c},...
            'markersize',10,'linewidth',2,...
            'markerfacecolor',fcol{c});
        hold on
    end
    xlim([-3 7])
    ylim([-6 6])
    
    contour(Xv,Yv,Probs(:,:,i));
    ti = sprintf('Probability contours for class %g',i);
    title(ti);
end