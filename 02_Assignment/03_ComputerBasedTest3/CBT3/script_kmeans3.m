%%
close all; clear all; clc;
%% read image to cluster
im = imread('rgbeye.jpg');
% im = imread('MRI.jpg');
% im = imread('horse.jpg');

nR = 128; nC = 128; % specify number of rows and columns
im = imresize(im, [nR, nC]);colormap(gray) % reduce size for quick computations
nD = size(im,3);
imagesc(im), colormap gray; impixelinfo, axis equal, axis off
% put it into right shape: one pixel per column
X = double(reshape(im,[nR*nC nD]));
%% cluster in K clusters
K=2;
[cluster_means,ClusterIndex] = kmeans_cluster2(X,K);
%% display clustering results
clusterOutput=reshape(ClusterIndex,nR,nC);
figure; 
subplot(121), imagesc(im)
subplot(122), imagesc(clusterOutput); colormap gray; impixelinfo, axis off