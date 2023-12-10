% Generate a 2D uncorrelated dataset
% Implement KMeans from first principles 
%
clc;
close all;
clear all;
Mean1 = [-4; -1];
Std1 = 0.75;
Mean2 = [3; 4];
Std2 = 2.0; 
samples = 1000; 
data1 = Std1 * randn(2, samples) + repmat(Mean1, 1, samples);
data2 = Std2 * randn(2, samples) + repmat(Mean2, 1, samples);
figure;
plot(data1(1, :), data1(2,:),'b.', 'MarkerSize',12);
hold on;
plot(data2(1,:), data2(2,:), 'r.', 'MarkerSize',12);
legend('Dataset 1', 'Dataset 2', 'location', 'NW'); 
title('Original Dataset');
hold off;
xlabel('x-value');
ylabel('y-value');
xlim([-10 10]);
ylim([-8 12]);
grid on;

trainData = [data1 data2];
figure;
plot(trainData(1, :), trainData(2,:),'b.', 'MarkerSize',12);
legend('Dataset','location', 'NW'); %'best');
title('Merged Dataset');
xlabel('x-value');
ylabel('y-value');
xlim([-10 10]);
ylim([-8 12]);
grid on;

X = trainData';
K = 2;
max_iterations = 10; %5;
% randomly assign the K data points as the initial "centroids" of the K clusters
centroids = initCentroids(X,K);
sumD = zeros(K,1);
for i = 1:max_iterations
    indices = getClosestCentroids(X, centroids); % reassign the indices to the relevant clusters
    [centroids, distances] = computCentroidsandDistances(X, indices, K); % recalculate centroids, and distances among all clusters
    sumD(i) = distances; 
end
%idx = kmeans (X, K); %this is to use kmeans function to calculate the
%indices for each cluster
   figure;
   plot(sumD, 'bo-', 'linewidth',2);
   title('Overall distances vs. number of iterations');
    xlabel('Iterations');
    ylabel('Overall distances');
    grid on;

message = sprintf('KMeans Clustering (MaxIterations = %d)', max_iterations);
figure;
plot(trainData(1, indices == 1),trainData(2, indices == 1),'r.', 'MarkerSize',12);% for 1st cluster
hold on;
plot(trainData(1, indices == 2), trainData(2, indices == 2), 'b.', 'MarkerSize',12); %for 2nd cluster
hold on;
plot(centroids(1,1),centroids(1,2),'kx', 'MarkerSize',15,'LineWidth',3); %for the 1st centroid
hold on;
plot(centroids(2,1),centroids(2,2),'kx', 'MarkerSize', 15, 'LineWidth', 3); %for the 2nd centroid
legend('Cluster 1', 'Cluster 2', 'Centroids', 'location', 'NW'); %'best');
title(message);
hold off;
xlabel('x-value');
ylabel('y-value');
xlim([-10 10]);
ylim([-8 12]);
grid on;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





samples = 500; 
data1 = Std1 * randn(2, samples) + repmat(Mean1, 1, samples);
data2 = Std2 * randn(2, samples) + repmat(Mean2, 1, samples);
figure;
plot(data1(1, :), data1(2,:),'b.', 'MarkerSize',12);
hold on;
plot(data2(1,:), data2(2,:), 'r.', 'MarkerSize',12);
legend('Dataset 1', 'Dataset 2', 'location', 'NW'); 
title('Original Dataset');
hold off;
xlabel('x-value');
ylabel('y-value');
xlim([-10 10]);
ylim([-8 12]);
grid on;

testData = [data1 data2];
figure;
plot(testData(1, :), testData(2,:),'b.', 'MarkerSize',12);
legend('Dataset','location', 'NW'); %'best');
title('Merged Dataset');
xlabel('x-value');
ylabel('y-value');
xlim([-10 10]);
ylim([-8 12]);
grid on;

X = testData';
K = 2;
max_iterations = 10; %5;
% randomly assign the K data points as the initial "centroids" of the K clusters
centroids = initCentroids(X,K);
sumD = zeros(K,1);
for i = 1:max_iterations
    indices = getClosestCentroids(X, centroids); % reassign the indices to the relevant clusters
    [centroids, distances] = computCentroidsandDistances(X, indices, K); % recalculate centroids, and distances among all clusters
    sumD(i) = distances; 
end
%idx = kmeans (X, K); %this is to use kmeans function to calculate the
%indices for each cluster
   figure;
   plot(sumD, 'bo-', 'linewidth',2);
   title('Overall distances vs. number of iterations');
    xlabel('Iterations');
    ylabel('Overall distances');
    grid on;

message = sprintf('KMeans Clustering (MaxIterations = %d)', max_iterations);
figure;
plot(testData(1, indices == 1),testData(2, indices == 1),'r.', 'MarkerSize',12);% for 1st cluster
hold on;
plot(testData(1, indices == 2), testData(2, indices == 2), 'b.', 'MarkerSize',12); %for 2nd cluster
hold on;
plot(centroids(1,1),centroids(1,2),'kx', 'MarkerSize',15,'LineWidth',3); %for the 1st centroid
hold on;
plot(centroids(2,1),centroids(2,2),'kx', 'MarkerSize', 15, 'LineWidth', 3); %for the 2nd centroid
legend('Cluster 1', 'Cluster 2', 'Centroids', 'location', 'NW'); %'best');
title(message);
hold off;
xlabel('x-value');
ylabel('y-value');
xlim([-10 10]);
ylim([-8 12]);
grid on;
