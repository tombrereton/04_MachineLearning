%% This program trains a Bayesian classifier
% using the Maximum likelihood estimate WITHOUT naive assumption.
% We train it on the helthy and diseased datasets
% provided in cbt2data.mat

% Whilst all 4 methods (ML and MAP Naive, ML and MAP wo Naive)
% could be done in one program, it was found separate
% files were easier for experimenting and gaining insight of data

% Based on the following files
% Reference 1: bayesclass.m, A First Course in Machine Learning, Chapter 5.
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
savePlots = 0; % 0 = don't save, 1 = save

%% We compute the prior
class1Total = sum(t_train==1); % We count the number of diseased patients
class2Total = sum(t_train==2); % We count the number of healthy patients
probClass1 = class1Total/(class1Total + class2Total); % We compute the dieased prior based on frequency
probClass2 = class2Total/(class1Total + class2Total); % We compute the healthy prior based on frequency
probPrior = [probClass1; probClass2]; % We store priors in vector, make matrix and put in other prior values?

% Uncomment to experiment with different priors:
% probPrior = [.99; 0.01]; % We give diseased class a strong prior
% probPrior = [0.5; 0.5]; % We compute prior based on 1/C
% probPrior = [0;1]; % We give diseased class a strong prior

%% Find the mean and variance 
% Using the Naive (independence) assumption
cl = unique(t_train); % get total number of classes from test set

for c = 1:length(cl) % We loop over the number of classes (1,diseased and 2,healthy)
    pos = find(t_train==cl(c)); % We store position of class label in vector
    class_mean(c,:) = mean(X_train(pos,:)); % class-wise & attribute-wise mean
    class_var(:,:,c) = cov(X_train(pos,:),1); % class-wise & attribute-wise variance
end

%% Maximum A Posteriori (MAP) classification for new data
% Without the naive assumption, thus we don't take the diag of variance
% (see sigma)
class_probs_new = [];

for c = 1:length(cl)
    sigma = class_var(:,:,c); % we convert row to diagonal elements
    diff_train = [X_new(:,1)-class_mean(c,1) X_new(:,2)-class_mean(c,2)]; % We compute difference between data point and respective mean
    const = 1/sqrt((2*pi()^size(X_new,2)*det(sigma))); % we compute the constanst for the gaussian function
    class_probs_new = [class_probs_new const * exp(-1/2*diag(diff_train * inv(sigma) * diff_train'))* probPrior(c)]; % We use the gaussian function to compute MLend
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
    temp = [Xv(:)-class_mean(c,1) Yv(:)-class_mean(c,2)]; % xnew - mean, tnew - mean
    tempc = (class_var(:,:,c)); % get all rows and columns for each class
    const = -log(2*pi) - log(det(tempc)); % % We compute constant using log laws, is it actually easier with log?, why not -1/2 out front?
    Probs(:,:,c) = reshape(exp(const - 0.5*diag(temp*inv(tempc)*temp')),size(Xv)) * probPrior(c); % reshape exp(of function) into size of Xv
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
    filename = strcat('MAPWONtrainingData.png'); 
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
title({'Increase in concentration of chemical 1';...
    'indicates patient more likely to be diseased,';...
    'chemical 2 concentration of ~16 is worst'},'fontsize',18); 
xlim([-2 12]), ylim([4 18]);
colorbar('eastoutside');
legend('Diseased', 'Healthy');
set(gca,'Color',[0.7 0.7 0.7]);

if (savePlots == 1)
    filename = strcat('MAPWONnewData.png'); 
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
        filename = sprintf('MAPWONclassCondContours%s.png', classLabel{i}); 
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
        filename = sprintf('MAPWONprobContours%s.png', classLabel{i}); 
        saveas(gcf,filename);
    end
end
