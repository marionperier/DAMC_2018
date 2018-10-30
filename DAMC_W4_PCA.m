clear variables
close all
clc

%% Load data

load('trainSet.mat');
load('trainLabels.mat');
load('testSet.mat');

%% PCA

[coeff, score, variance] = pca(trainData);
%Covariance matrix of the original data 
Cov = cov(trainData);
%covariance matrix of the data projected on the PCs
Cov_projected = diag(variance);

%imshow(Cov, 'Reduce'); => the image is too large to apply this function

