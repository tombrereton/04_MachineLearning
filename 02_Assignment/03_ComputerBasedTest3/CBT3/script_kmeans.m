
%%
clear all; close all; clc;

%% Load the data
load('kmeansdata.mat');

%% Plot the data
figure(1);hold off
plot(X(:,1),X(:,2),'ko');

%% perform clustering
for K = 2:7; % The number of clusters
    [cluster_means,ClusterIndex] = kmeans_cluster2(X,K);
    % Plot the assigned data
    cols = {'r','g','b','k','m','c','y'};
    figure(1); hold off
    for k = 1:K
        plot(X(ClusterIndex==k,1),X(ClusterIndex==k,2),...
            'ko','markerfacecolor',cols{k});
        hold on
    end
    % Plot the means
    figure(1)
    for k = 1:K
        plot(cluster_means(k,1),cluster_means(k,2),'ks','markersize',15,...
            'markerfacecolor',cols{k});
    end
    title(sprintf('K = %d',K));
    pause
end