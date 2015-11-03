%% ###Missing values and scaling###

clear all;
close all;

% random data
data = randn([100,3])*diag([1 2 3])+repmat([30 2 3], 100, 1); 
data = [data; randn([100,3])*diag([2 1 3])+repmat([1 10 1], 100, 1)];

% add some missing values
tmp = randperm(200*3);
missinindex = tmp(1:150);
data(missinindex) = NaN;

% scatterplot 3D: only samples with non-missing values
shape = repmat(30,200,1);
color = [repmat(1,100,1);  repmat(3,100,1)];
scatter3(data(:,1), data(:,2), data(:,3), shape, color)
title('Input data - samples with missing values are missing')

%% Example 1:  
% replace missing values by appropriate mean value and visualize the results

meandata = data;

anyNanRow = (isnan(data)*[1;1;1]>0);

color_mean = color;
color_mean(anyNanRow) = 2;

data_mean = nanmean(meandata);
index = 1:200;
for i = index(anyNanRow)
    selsample = meandata(i, :);  
    mask = isnan(selsample);
    if ~all(mask) 
        meandata(i, mask) = data_mean(mask);
    end
end

figure
scatter3(meandata(:,1), meandata(:,2), meandata(:,3), shape, color_mean)
title('Missing values replaced by mean')

%% Assignment 1:  
% replace missing values by 1NN and visualize results
knndata = data;
color_knn = color;
color_knn(anyNanRow) = 0.7;

nanVals = isnan(knndata);
noNans = sum(nanVals,2) == 0;
dataNoNans = data(noNans,:);

index = 1:200;
for i = index(anyNanRow)
    selsample = knndata(i, :);  
    mask = isnan(selsample);
    if ~all(mask)
        selectedOrig = knndata(i,:);
        selected = knndata(i, ~mask);
        noNansMasked = dataNoNans(:, ~mask);
        idx = knnsearch(noNansMasked, selected);
        
        knndata(i, mask) = dataNoNans(idx, mask);
    end
end

figure
scatter3(knndata(:,1), knndata(:,2), knndata(:,3), shape, color_knn)
title('Missing values replaced by 1-NN')

%% Example 2:  2D scatter plot of 3D data by PCA - without standardization

knndata_pca = knndata(~isnan(knndata(:,1)),:); % some rows can contain all NaN values, remove them out
shape_pca = shape(~isnan(knndata(:,1)),:);
color_knn_pca = color_knn(~isnan(knndata(:,1)),:);

coeff = princomp(knndata_pca);
pca=knndata_pca*coeff;

figure
scatter(pca(:,1), pca(:,2), shape_pca, color_knn_pca);
title('Non-standardized 2D visualization');

%% Assignment 2:  2D scatter plot of 3D data by PCA - with standardization

standardized_knndata_pca = knndata_pca;
[h, w] = size(standardized_knndata_pca);
mean_d = mean(knndata_pca);
standard_d = std(knndata_pca);

for i = 1:h
    tmp = standardized_knndata_pca(i,:);
    for j = 1:3
        standardized_knndata_pca(i,j) = (tmp(j) - mean_d(j))/standard_d(j);
    end
    
    %standardized_knndata_pca(i,1) = (tmp(1) - mean_d(1))/standard_d(1);
    %standardized_knndata_pca(i,2) = (tmp(2) - mean_d(2))/standard_d(2);
    %standardized_knndata_pca(i,3) = (tmp(3) - mean_d(3))/standard_d(3);
end

coeff = princomp(standardized_knndata_pca);
pca=knndata_pca*coeff;

figure
scatter(pca(:,1), pca(:,2), shape_pca, color_knn_pca);
title('Standardized 2D visualization');

%%  ###Outlier values###
clear all
load('outlier_data.mat')


figure
subplot(1,2,1)
scatter(data(:,1), data(:,2))
title('Data with outliers');
subplot(1,2,2)
boxplot(data)

%% Example 3: Remove outliers (univariate approach)
% Sample (vector) is removed if any its element is far from its mean more than n*std. 

n = 3; 
c1 = data(:,1);
mu1 = mean(c1); % Data mean
sigma1 = std(c1); % Data standard deviation
index1 = abs(c1 - mu1) > n*sigma1; % outlier candidates

c2 = data(:,2);
mu2 = mean(c2); % Data mean
sigma2 = std(c2); % Data standard deviation
index2 = abs(c2 - mu2) > n*sigma2; % outlier candidates

index = index1 | index2;
outliers = data(index,:)

figure
subplot(1,2,1)
scatter(data(~index,1), data(~index,2))
title('Data without outliers (univariate approach)');
subplot(1,2,2)
boxplot(data(~index,:))

%% Assignment 3:  Remove outliers using k-means

X = data;
m = 10;
th = 0.01;
R = 100;
nReplicates = 40;

% #### Outlier removal algorithm ####
% IN: 
%       X     : data samples  
%       m     : number of clusters
%       th    : threshold for distrortion
%       R     : maximal number of algorithm steps
%       nReplicates : number of runs of k-means (matlab returns the best partitioning)
% OUT:
%       X     : data samples without outliers

% init
i = 0;

while(1)    
    i = i+1;
    % number of samples
    len = size(X,1);
    % find centroids( newC) and partitioning (newP)
    [newP, newC] = kmeans(X,m,'replicates',nReplicates);     
        
    % %%%%%%%%% Zde doplnte kod dle ukolu 3 %%%%%%%%%%%%%    
    newX = X;        
    % ### BEGIN of outlier removal code ###
    
    removed = 0;
    for i = 1:m
        index = i - removed;
        c = newC(index,:);
        
        clusterSize = 0;
        for j = 1:size(newP,1)
            if i == newP(j,1)
                clusterSize = clusterSize + 1;
            end
        end
        
        if clusterSize > 1
            smax = -9999;
            smax_idx = -1;
            smin = 9999;
            
            lenP = size(newP, 1);
            for k = 1:lenP
                if i == newP(k,1)
                    tmp = newX(k,:);
                    d = pdist([c, tmp], 'euclidean');
                    %smax = max(smax, d);
                    if smax < d
                        smax = d;
                        smax_idx = k;
                    end
                    
                    smin = min(smin, d);
                end
            end
            
            distortion = smin/smax;
            if distortion < th
                if smax_idx ~= -1
                    %[newP, newC, newX] = removeSample(smax, xnew)
                    newP(smax_idx,:) = [];
                    newX(smax_idx,:) = [];
                end
            end
        else
            %[newP, newC, newX] = removeCluster(c, xnew)
            
            removedClust = 0;
            for m = 1:size(newP,1)
                idx = m - removedClust;
                if i == newP(idx,1)
                    newP(idx,:) = [];
                    newX(idx,:) = [];
                    removedClust = removedClust + 1;
                end
            end
            
            newC(index,:) = [];
            m = m - 1;
            removed = removed + 1;
        end
    end
    
    % END of outlier removal code
    
    % new number of samples
    new_len = size(newX,1);    
    % compute new centers and partitioning    
    X = newX; 
    
    % stop conditions
    if i>R | new_len==len    
        break 
    end        
end

% compute optimal partitioning on the new data
[P, C, sumd] = kmeans(X, m,'replicates',nReplicates);

figure
subplot(1,2,1)
scatter(X(:,1), X(:,2), ones(size(X,1),1)*10, P)
hold on
scatter(C(:,1), C(:,2), ones(size(C,1),1)*40)
title('Data without outliers')
subplot(1,2,2)
[P, C] = kmeans(data, m);
scatter(data(:,1), data(:,2), ones(size(data,1),1)*10, P)
title('Data with outliers')

figure
subplot(1,2,1)
scatter(X(:,1), X(:,2))
title('Data without outliers (multivariate approach)');
subplot(1,2,2)
boxplot(X)


