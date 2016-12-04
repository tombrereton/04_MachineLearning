%%
clear all; close all; clc;

%% Load the data
% load('kmeansdata.mat');
load('kmeansnonlindata.mat');

%% perform clustering
for K = 2:7; % The number of clusters
    [cluster_means,ClusterIndex] = kmeans_cluster3(X,K);
end