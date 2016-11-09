%% This program trains a Bayesian classifier
% using the Maximum likelihood estimate.
% We train it on the helthy and diseased datasets
% provided in cbt2data.mat with and without
% the naive assumption
% Whilst all 4 methods (ML and MAP Naive, ML and MAP wo Naive)
% could be done in one program, it was found that separate
% files were easier for experimenting and gaining insight of data
%% We load the data
clear all; close all;
load cbt2data.mat
X_train = [diseased'; healthy']; % We put the training data into a single matrix
t_train = [ones(length(diseased'),1); ones(length(healthy'),1).*2]; % We give diseased a label of 1, healthy a label of 2
X_test = newpts';

%% We plot the data
train_colours = {'r','blue'}; 
cl = unique(t_train); % get total number of classes from test set
tag = {'o','o'};
figure(1);
hold off
for c = 1:length(cl)
    pos = find(t_train==cl(c)); % get position, or row, of each data point in class
    plot(X_train(pos,1),X_train(pos,2),tag{c},... % plot points of column 1 on x and column 2 on y
        'markersize',4,'linewidth',1,...
        'color',train_colours{c});
    hold on
end
set(gca,'Color',[0.7 0.7 0.7]);
xlim([-2 10]), ylim([4 16]) % max the x and y scales square
xlabel('Concentration of Chemical 1','fontsize',16);
ylabel('Concentration of Chemical 2','fontsize',16);
title({'Increase in concentration of both chemicals';...
    'suggests patient more likely to be diseased'},'fontsize',18); 
% filename = strcat('MLRawData',int2str(order),'.png'); 
% saveas(gcf,filename);

%% Find the mean and variance 
% Using the Naive (independence) assumption
for c = 1:length(cl) % We loop over the number of classes (1,diseased and 2,healthy)
    pos = find(t_train==cl(c)); % We store position of class label in vector
    class_mean(c,:) = mean(X_train(pos,:)); % We compute the mean for both attributes and for each class
    class_var(c,:) = var(X_train(pos,:),1); % We compute the variance for both attributes and for each class
end

%% Compute the predictive probabilities
[Xv,Yv] = meshgrid(-2:0.1:10,4:0.1:16); % We create mesh grid to compute contour plots
Probs = [];
for c = 1:length(cl)
    temp = [Xv(:)-class_mean(c,1) Yv(:)-class_mean(c,2)]; % x1i - mean, x2i - mean
    tempc = diag(class_var(c,:)); %
    const = -log(2*pi) - log(det(tempc)); % We compute constant using log laws
    Probs(:,:,c) = reshape(exp(const - 0.5*diag(temp*inv(tempc)*temp')),size(Xv)); % We compute probability for each point
end

%% Plot the density contours for class-conditional distributions
% figure(2);hold off
for i = 1:2
    figure(i+1);hold off
    hold off
    for c = 1:length(cl) % plot the data points for each class
        pos = find(t_train==cl(c));
        plot(X_train(pos,1),X_train(pos,2),tag{c},...
            'markersize',2,'linewidth',1,...
            'color',train_colours{c});
        hold on
    end
    
    contour(Xv,Yv,Probs(:,:,i)); % plot contour line
    ti = sprintf('Probability contours for class %g',i);
    title(ti);
    set(gca,'Color',[0.7 0.7 0.7]);
    xlim([-2 10]), ylim([4 16])
end


% xlabel('Concentration of Chemical 1','fontsize',16);
% ylabel('Concentration of Chemical 2','fontsize',16);
% title({'Increase in concentration of both chemicals';...
%     'suggests patient more likely to be diseased'},'fontsize',18); 
% filename = strcat('MLRawData',int2str(order),'.png'); 
% saveas(gcf,filename);

%% Plot the contours of the classification probabilities
Probs = Probs./repmat(sum(Probs,3),[1,1,2]); % We normalise to get probability

for i = 1:2
    figure(i+3);hold off
    hold off
    for c = 1:length(cl) % plot the data points for each class
        pos = find(t_train==cl(c));
        plot(X_train(pos,1),X_train(pos,2),tag{c},...
            'markersize',2,'linewidth',1,...
            'color',train_colours{c});
        hold on
    end
    
    contour(Xv,Yv,Probs(:,:,i)); % plot contour line
    ti = sprintf('Probability contours for class %g',i);
    title(ti);
    set(gca,'Color',[0.7 0.7 0.7]);
    xlim([-2 10]), ylim([4 16]);
end


% xlabel('Concentration of Chemical 1','fontsize',16);
% ylabel('Concentration of Chemical 2','fontsize',16);
% title({'Increase in concentration of both chemicals';...
%     'suggests patient more likely to be diseased'},'fontsize',18); 
% filename = strcat('MLRawData',int2str(order),'.png'); 
% saveas(gcf,filename);

%% Maximum likelihood (ML) classification for new data
% test = class_var(1,:);
class_probs_new = [];
for c = 1:length(cl)
    sigma_naive = diag(class_var(c,:)); % we convert row to diagonal elements
    diff_train = [X_test(:,1)-class_mean(c,1) X_test(:,2)-class_mean(c,2)]; % We compute difference between data point and respective mean
    const = 1/sqrt((2*pi()^size(X_test,2)*det(sigma_naive))); % we compute the constanst for the gaussian function
    class_probs_new = [class_probs_new const * exp(-1/2*diag(diff_train * inv(sigma_naive) * diff_train'))]; % We use the gaussian function to compute ML
end

%% Classify each example
% [M,Class_train] = max(class_probs,[],2);¡
[M,Class_new] = max(class_probs_new, [], 2);

%% Plot the training samples
new_colours = {'r','b'};
cl = unique(t_train); % get total number of classes from test set
tag_new = {'x','x'};
figure(6);
hold off
for c = 1:length(cl)
    pos = find(Class_new==cl(c)); % get position, or row, of each data point in class
    plot(X_test(pos,1),X_test(pos,2),tag_new{c},... % plot points of column 1 on x and column 2 on y
        'markersize',3,'linewidth',.1,...
        'markerfacecolor',new_colours{c});
    hold on
end
set(gca,'Color',[0.7 0.7 0.7]);
xlim([-2 12]), ylim([4 18]);

% xlabel('Concentration of Chemical 1','fontsize',16);
% ylabel('Concentration of Chemical 2','fontsize',16);
% title({'Increase in concentration of both chemicals';...
%     'suggests patient more likely to be diseased'},'fontsize',18); 
% filename = strcat('MLRawData',int2str(order),'.png'); 
% saveas(gcf,filename);
