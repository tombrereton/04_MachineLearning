%% This program trains a Bayesian classifier
% using the Maximum likelihood estimate.
% We train it on the helthy and diseased datasets
% provided in cbt2data.mat with and without
% the naive assumption
%% We load the data
clear all; close all;
load cbt2data.mat
% train_diseased = diseased';
% train_healthy = healthy';
% X_train = [train_diseased(:,1); train_healthy(:,1)];
% t_train = [train_diseased(:,2); train_healthy(:,2)];
X_train = [diseased'; healthy']; % We put the training data into a single matrix
t_train = [ones(length(diseased'),1); ones(length(healthy'),1).*2]; % We create a label vector based on the diseased and healthy samples
X_test = newpts';
%% We compute the prior
class1Total = sum(t_train==1); % We count the number of diseased patients
class2Total = sum(t_train==2); % We count the number of healthy patients
probClass1 = class1Total/(class1Total + class2Total); % We compute the dieased prior based on frequency
probClass2 = class2Total/(class1Total + class2Total); % We compute the healthy prior based on frequency
probPrior = [probClass1; probClass2]; % We store priors in vector
% probPrior = [.99; 0.01]; % We give diseased class a strong prior
% probPriot = [0.5; 0.5]; % We compute prior based on 1/C
%% We plot the data
train_colours = {[1,0,1],[0,1,1]};
cl = unique(t_train); % get total number of classes from test set
tag = {'ko','kd'};
figure(1);
hold off
for c = 1:length(cl)
    pos = find(t_train==cl(c)); % get position, or row, of each data point in class
    plot(X_train(pos,1),X_train(pos,2),tag{c},... % plot points of column 1 on x and column 2 on y
        'markersize',7,'linewidth',.5,...
        'markerfacecolor',train_colours{c});
    hold on
end
xlim([0 11]), ylim([4 15]) % max the x and y scales square

%% Find the mean and variance 
% Using the Naive (independence) assumption

for c = 1:length(cl)
    pos = find(t_train==cl(c));
    % Find the means
    class_mean(c,:) = mean(X_train(pos,:)); % class-wise & attribute-wise mean
    class_var(c,:) = var(X_train(pos,:),1); % class-wise & attribute-wise variance
end
%% Compute the predictive probabilities
[Xv,Yv] = meshgrid(0:0.1:11,4:0.1:15); % what is this mesh grid for??, 
Probs = [];
for c = 1:length(cl)
    temp = [Xv(:)-class_mean(c,1) Yv(:)-class_mean(c,2)]; % xnew - mean, tnew - mean
    tempc = diag(class_var(c,:)); % only diagonal, assuming naive? , (co)variance for class
    const = -log(2*pi) - log(det(tempc)); % is it actually easier with log?, why not -1/2 out front?
    Probs(:,:,c) = reshape(exp(const - 0.5*diag(temp*inv(tempc)*temp')),size(Xv)) * probPrior(c); % reshape exp(of function) into size of Xv
end

Probs = Probs./repmat(sum(Probs,3),[1,1,2]); % what is this? full bayes rule? how?, is value of contour line?
% above, divide by sum of probs i.e. divide by marginal??? how does
% function do this???
%% Plot the predictive contours
figure(3);hold off
for i = 1:2
    subplot(1,2,i);
    hold off
    for c = 1:length(cl) % plot the data points for each class
        pos = find(t_train==cl(c));
        plot(X_train(pos,1),X_train(pos,2),tag{c},...
            'markersize',2,'linewidth',.5,...
            'markerfacecolor',train_colours{c});
        hold on
    end
    xlim([0 11])
    ylim([4 15])
    
    contour(Xv,Yv,Probs(:,:,i)); % plot contour line
    ti = sprintf('Probability contours for class %g',i);
    title(ti);
end
%% Maximum likelihood classification for training data
% test = class_var(1,:);
class_probs = [];
for c = 1:length(cl)
    sigma_naive = diag(class_var(c,:)); % we convert row to diagonal elements
    diff_train = [X_train(:,1)-class_mean(c,1) X_train(:,2)-class_mean(c,2)];
    const = 1/sqrt((2*pi()^size(X_train,2)*det(sigma_naive))); % we compute the constanst for the gaussian function
    class_probs = [class_probs const * exp(-1/2*diag(diff_train * inv(sigma_naive) * diff_train'))* probPrior(c)];
end
%% MAP classification for new data
% test = class_var(1,:);
class_probs_new = [];
for c = 1:length(cl)
    sigma_naive = diag(class_var(c,:)); % we convert row to diagonal elements
    diff_train = [X_test(:,1)-class_mean(c,1) X_test(:,2)-class_mean(c,2)];
    const = 1/sqrt((2*pi()^size(X_test,2)*det(sigma_naive))); % we compute the constanst for the gaussian function
    class_probs_new = [class_probs_new const * exp(-1/2*diag(diff_train * inv(sigma_naive) * diff_train'))* probPrior(c)];
end
%% Classify each attribute
[M,Class_train] = max(class_probs,[],2);
[M,Class_new] = max(class_probs_new, [], 2);
%% Plot the training samples
%% Plot the data
train_colours = {[1,0,1],[0,1,1]};
new_colours = {'b','r'};
cl = unique(t_train); % get total number of classes from test set
tag = {'ko','kd'};
tag_new = {'kx','kd'};
figure(2);
hold off
for c = 1:length(cl)
%     pos = find(t_train==cl(c)); % get position, or row, of each data point in class
%     plot(X_train(pos,1),X_train(pos,2),tag{c},... % plot points of column 1 on x and column 2 on y
%         'markersize',7,'linewidth',.5,...
%         'markerfacecolor',train_colours{c});
    pos = find(Class_new==cl(c)); % get position, or row, of each data point in class
    plot(X_test(pos,1),X_test(pos,2),tag_new{c},... % plot points of column 1 on x and column 2 on y
        'markersize',7,'linewidth',.5,...
        'markerfacecolor',new_colours{c});
    hold on
end
xlim([0 11]), ylim([4 20]) % max the x and y scales square
