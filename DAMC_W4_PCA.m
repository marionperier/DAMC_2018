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
Cov_PCA = cov(score);

variance = variance/(sum(variance)) * 100;

figure
subplot(1,2,1),
imshow(Cov * 100); %=> the image is too large to apply this function
subplot(1,2,2),
imshow(Cov_PCA * 100);

modified_cov = Cov - diag(10*ones(2048,1)); 
modified_cov_PCA = Cov_PCA - diag(10*ones(596,1)); 

value = max(max(modified_cov));
value_pca = max(max(modified_cov_PCA));

variance_cum = cumsum(variance);

figure
plot(variance_cum)

%% 

outer_folds = 10;
inner_folds = 5;

outer_cvpartition = cvpartition(trainLabels,'kfold',outer_folds);

for i = 1:outer_folds
    outer_indices = test(outer_cvpartition,i); % 0 is for training, 1 is of testing
    
    waitbar(i/outer_folds)

    outer_train_labels = trainLabels(outer_indices == 0);
    outer_train_data = trainData(outer_indices == 0,:);
    outer_test_data = trainData(outer_indices == 1,:);
    outer_test_labels = trainLabels(outer_indices == 1);
    
    inner_cvpartition = cvpartition(outer_train_labels,'kfold',inner_folds);
    
    for j= 1:inner_folds
        inner_indices = test(inner_cvpartition,j);
        
        inner_train_labels = outer_train_labels(inner_indices == 0);
        inner_train_data = outer_train_data(inner_indices == 0,:);
        inner_test_data = outer_train_data(inner_indices == 1,:);
        inner_test_labels = outer_train_labels(inner_indices == 1);
        
        inner_train_data = inner_train_data - mean(inner_train_data, 1);
        inner_test_data = inner_test_data - mean(inner_train_data, 1);
        
        coeff = pca(inner_train_data);
        PCA_data = inner_train_data * coeff;
        PCA_data_te = inner_test_data * coeff;
        
        for N_sel = 1:length(PCA_data(1,:))
            
            train_data_sel = PCA_data(:,1:N_sel);
            val_data_sel = PCA_data_te(:,1:N_sel);
            classifier = fitcdiscr(train_data_sel, inner_train_labels, 'discrimtype', 'diaglinear');

            label_prediction = predict(classifier, train_data_sel);
            label_prediction_val = predict(classifier, val_data_sel);

            classification_error = classification_errors(inner_train_labels, label_prediction);
            classification_error_val = classification_errors(inner_test_labels, label_prediction_val);
            
            error(j,N_sel) = classification_error;
            error_val(j,N_sel) = classification_error_val;
        end
    end
    
    mean_error = mean(error_val, 1);
    [min_error(i), best_N(i)] = min(mean_error);
    
end


%%

[std_data, mu, sigma] = zscore(trainData, 0, 1);
std_data_te = (testData - mu)./sigma;

coeff = pca(std_data);
PCA_data = std_data * coeff;
PCA_data_te = std_data_te * coeff;

classifier = fitcdiscr(PCA_data, trainLabels, 'discrimtype', 'diaglinear');
pred = predict(classifier, PCA_data_te);

labelToCSV(pred, 'PCA_diaglinear.csv', 'csv')