
%%
clear all; close all; clc;

%% Load the data
load('cbt3data.mat');

X = diseased(:,:,1)';
Y = healthy(:,:,1)';

%% Cluster data
K = 30 % number of clusters

[Means_diseased,ClusterIndex_diseased] = kmeans_cluster2(X,K);
[Means_healthy,ClusterIndex_healthy] = kmeans_cluster2(Y,K);
%% Initialise Km and meanKm
Km_diseased = zeros(size(ClusterIndex_diseased, 1));
Km_healthy = zeros(size(ClusterIndex_healthy, 1));

for i = 1:90;
    Km_diseased(:,i) = double(repmat(ClusterIndex_diseased(i,1),90,1) == ClusterIndex_diseased);
    Km_healthy(:,i) = double(repmat(ClusterIndex_healthy(i,1),90,1) == ClusterIndex_healthy);
end

meanKm_diseased = Km_diseased;
meanKm_healthy = Km_healthy;

%% Get average community matrix for all (20) patients
for trials = 1:500;
    for person = 1:size(diseased,3);

        X = diseased(:,:,person)';
        Y = healthy(:,:,person)';

        [~,ClusterIndex_diseased] = kmeans_cluster2(X,K);
        [~,ClusterIndex_healthy] = kmeans_cluster2(Y,K);

        for i = 1:90;
            Km_diseased(:,i) = double(repmat(ClusterIndex_diseased(i,1),90,1) == ClusterIndex_diseased);
            Km_healthy(:,i) = double(repmat(ClusterIndex_healthy(i,1),90,1) == ClusterIndex_healthy);
        end

        Km_3d_diseased = cat(3,meanKm_diseased, Km_diseased);
        meanKm_diseased = mean(Km_3d_diseased,3);

        Km_3d_healthy = cat(3,meanKm_healthy, Km_healthy);
        meanKm_healthy = mean(Km_3d_healthy,3);
    end
end
%% View matrix as image
figure(1);
imagesc(meanKm_diseased);
title({'Diseased Patients Community Matrix'},'fontsize',18); 

figure(2);
imagesc(meanKm_healthy);
title({'Healthy controls Community Matrix'},'fontsize',18); 

%% Get difference of healthy and diseased and view as community matrix
Km_diff = meanKm_healthy - meanKm_diseased;

figure(3);
imagesc(Km_diff);
title({'Difference Community Matrix'},'fontsize',18); 

