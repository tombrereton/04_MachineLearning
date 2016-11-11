%% This program trains a Bayesian classifier
% using the Maximum likelihood estimate.
% We train it on the helthy and diseased datasets
% provided in cbt2data.mat with and without
% the naive assumption
% Whilst all 4 methods (ML and MAP Naive, ML and MAP wo Naive)
% could be done in one program, it was found separate
% files were easier for experimenting and gaining insight of data

% Based on the following files
% Refe0ence 1: bayesclass.m, A First Course in Machine Learning, Chapter 5.
% Reference 2: bayestraintest.m, Machine Learning (Extended)

% NOTE: do we want to compare the labels between ML, MAP, Naive, WONaive
% build separate program for this without plots?
%% ******************* ANALYSIS *******************
%  ************************************************
%% We load the data
clear all; close all;
load cbt2data.mat
X_train = [diseased'; healthy']; % We put the training data into a single matrix
t_train = [ones(length(diseased'),1); ones(length(healthy'),1).*2]; % We give diseased a label of 1, healthy a label of 2
X_new = newpts'; % load new points
savePlots = 0; % 0 = don't save plots, 1 = save plots

%% Find the mean and variance 
% Using the Naive (independence) assumption
cl = unique(t_train); % get total number of classes from test set

for c = 1:length(cl) % We loop over the number of classes (1,diseased and 2,healthy)
    pos = find(t_train==cl(c)); % We store position of class label in vector
    class_mean(c,:) = mean(X_train(pos,:)); % We compute the mean for both attributes and for each class
    class_var(c,:) = var(X_train(pos,:),1); % We compute the variance for both attributes and for each class
end

%% Maximum likelihood (ML) classification for new data
class_probs_new = [];

for c = 1:length(cl)
    sigma_naive = diag(class_var(c,:)); % we convert row to diagonal elements
    diff_train = [X_new(:,1)-class_mean(c,1) X_new(:,2)-class_mean(c,2)]; % We compute difference between data point and respective mean
    const = 1/sqrt((2*pi()^size(X_new,2)*det(sigma_naive))); % we compute the constanst for the gaussian function
    class_probs_new = [class_probs_new const * exp(-1/2*diag(diff_train * inv(sigma_naive) * diff_train'))]; % We use the gaussian function to compute ML
end

%% Classify each example
% We classify each example by getting the column index which has the higher
% maximum likelihood (ML). Column 1 and column 2 give the ML of belonging to
% each class respectively. For example, a column 1 has a ML of 0.7 and
% column 2 is 0.3, therefore it returns the index (or column) 1.
[M,Class_new] = max(class_probs_new, [], 2); 

%% Compute the predictive probabilities
% We calculate the proabilities for random data (mesh grid)
% so we can plot contours of probabilities and ML
[Xv,Yv] = meshgrid(-2:0.1:12,4:0.1:18); % We create mesh grid to compute contour plots
Probs = [];

for c = 1:length(cl) % loop over unique classes
    temp = [Xv(:)-class_mean(c,1) Yv(:)-class_mean(c,2)]; % x1i - mean, x2i - mean
    tempc = diag(class_var(c,:)); %
    const = -log(2*pi) - log(det(tempc)); % We compute constant using log laws
    Probs(:,:,c) = reshape(exp(const - 0.5*diag(temp*inv(tempc)*temp')),size(Xv)); % We compute probability for each point
end

%% ******************* PLOTS *******************
%  *********************************************
% Plots kept seperate so we can easily 
% adjust titles and labels

%% We plot the training data
train_colours = {'r','b'}; 
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

xlabel('Concentration of Chemical 1','fontsize',16);
ylabel('Concentration of Chemical 2','fontsize',16);
title({'Increase in concentration of both chemicals';...
    'suggests patient more likely to be diseased'},'fontsize',18); 
xlim([-2 12]), ylim([4 18]); % set x and y scales square
colorbar('eastoutside');
legend('Diseased', 'Healthy');
set(gca,'Color',[0.7 0.7 0.7]);

if (savePlots == 1)
    filename = strcat('MLtrainingData.png'); 
    saveas(gcf,filename);
end

%% We plot the new data
new_colours = {'r','b'};
tag_new = {'x','x'};
figure(2);
hold off

for c = 1:length(cl)
    pos = find(Class_new==cl(c)); % get position, or row, of each data point in class
    plot(X_new(pos,1),X_new(pos,2),tag_new{c},... % plot points of column 1 on x and column 2 on y
        'markersize',5,'linewidth',.1,...
        'markerfacecolor',new_colours{c});
    hold on
end

xlabel('Concentration of Chemical 1','fontsize',16);
ylabel('Concentration of Chemical 2','fontsize',16);
title({'Classification of new data with ML'},'fontsize',18); 
xlim([-2 12]), ylim([4 18]);
colorbar('eastoutside');
legend('Diseased', 'Healthy');
set(gca,'Color',[0.7 0.7 0.7]);

if (savePlots == 1)
    filename = strcat('MLnewData.png'); 
    saveas(gcf,filename);
end

%% Plot the density contours for class-conditional distributions
classLabel = {'Diseased';'Healthy'};

for i = 1:2
    figure(i+2);hold off
    hold off
    for c = 1:length(cl) % plot the data points for each class
        pos = find(t_train==cl(c));
        plot(X_train(pos,1),X_train(pos,2),tag{c},...
            'markersize',4,'linewidth',1,...
            'color',train_colours{c});
        hold on
    end
    
    contour(Xv,Yv,Probs(:,:,i)); % plot contour line
    ti = sprintf('Density contours for class conditional %g, %s',i, classLabel{i});
    xlabel('Concentration of Chemical 1','fontsize',16);
    ylabel('Concentration of Chemical 2','fontsize',16);
    title(ti, 'fontsize',18);
    xlim([-2 12]), ylim([4 18]);
    colorbar('eastoutside');
    legend('Diseased', 'Healthy');
    set(gca,'Color',[0.7 0.7 0.7]);
    
    if (savePlots == 1)
        filename = sprintf('MLclassCondContours%s.png', classLabel{i}); 
        saveas(gcf,filename);
    end
end

%% Plot the contours of the classification probabilities
Probs = Probs./repmat(sum(Probs,3),[1,1,2]); % We normalise to get probability

for i = 1:2
    figure(i+4);hold off
    hold off
    for c = 1:length(cl) % plot the data points for each class
        pos = find(t_train==cl(c));
        plot(X_train(pos,1),X_train(pos,2),tag{c},...
            'markersize',4,'linewidth',1,...
            'color',train_colours{c});
        hold on
    end
    
    contour(Xv,Yv,Probs(:,:,i)); % plot contour line
    ti = sprintf('Probability contours for class %g, %s',i, classLabel{i});
    xlabel('Concentration of Chemical 1','fontsize',16);
    ylabel('Concentration of Chemical 2','fontsize',16);
    title(ti, 'fontsize', 18);
    xlim([-2 12]), ylim([4 18]);
    colorbar('eastoutside');
    legend('Diseased', 'Healthy');
    set(gca,'Color',[0.7 0.7 0.7]);
    
    if (savePlots == 1)
        filename = sprintf('MLprobContours%s.png', classLabel{i}); 
        saveas(gcf,filename);
    end
end
