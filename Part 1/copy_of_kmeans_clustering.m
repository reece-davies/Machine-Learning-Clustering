clc;
close all;
clear all;

% A.) Generate a 2D uncorrelated Gaussian distributed training dataset
samples = 200;
Mean1 = [-3; 4];
Std1 = 1.25;
Mean2 = [2; 7];
Std2 = 1.25;
Mean3 = [1; 2];
Std3 = 1.5;
data1 = Std1 * randn(2, samples) + repmat(Mean1, 1, samples);
data2 = Std2 * randn(2, samples) + repmat(Mean2, 1, samples);
data3 = Std3 * randn(2, samples) + repmat(Mean3, 1, samples);
figure;
plot(data1(1, :), data1(2,:),'ro', 'MarkerSize', 10);
hold on;
plot(data2(1,:), data2(2,:), 'bx', 'MarkerSize', 10);
hold on;
plot(data3(1,:), data3(2,:), 'g+', 'MarkerSize', 10);
legend('Dataset 1', 'Dataset 2', 'Dataset 3', 'location', 'NW'); 
title('Original Dataset');
hold off;
xlabel('x-value');
ylabel('y-value');
xlim([-6 6]);
ylim([-2 12]);
grid on;

% B.) Concatenate the training datasets into a single dataset.
trainData = [data1 data2 data3];
figure;
plot(trainData(1, :), trainData(2,:), 'b.', 'MarkerSize', 10);
legend('Train Dataset','location', 'NW'); %'best');
title('Merged Dataset');
xlabel('x-value');
ylabel('y-value');
xlim([-6 6]);
ylim([-2 12]);
grid on;

% C.) Implement Kmeans clustering from first principles
X = trainData';
K = 3;
max_iterations = 10; %5;
% randomly assign the K data points as the initial "centroids" of the K clusters
%centroids = initCentroids(X,K);

% 1111111111111111111111111 initCentroids 1111111111111111111111111 %
centroids = zeros(K, size(X,2));
randidx = randperm(size(X,1)); %random permutation of the integers of 1:N (here N is the sample size of X).
centroids = X(randidx(1:K), :); % centroids are datapoints corresponding to those random K positions.
% 1111111111111111111111111 1111111111111 1111111111111111111111111 %

sumD = zeros(K,1);

for i = 1:max_iterations
    %indices = getClosestCentroids(X, centroids); % reassign the indices to the relevant clusters
    
    % 2222222222222222222222222 getClosestCentroids 2222222222222222222222222 %
    %for each datapoint x, calculate which cluster it belongs to, and put
    %result in the indices matrix (Nx1), e.g. for 1st datapoint, indices(1) can be 1 or 2 (for 2 clusters).  
    K = size(centroids, 1);
    indices = zeros(size(X,1), 1);
    m = size(X,1);

    for p = 1:m %for loop to go through all data points
        k = 1;
       % min_dist = sum((X(i,:) - centroids(1,:)) .^ 2); 
       % there are two ways to calculate min_dist
       min_dist = (X(p,:) - centroids(1,:)) * (X(p,:) - centroids(1,:))';
        for j = 2:K
            dist = sum((X(p,:) - centroids(j,:)) .^ 2);
            if (dist < min_dist)
                min_dist = dist;
                k = j; % k is the cluster the datapoint belongs to
            end
        end
        indices(p) = k;
    end
    % 2222222222222222222222222 2222222222222222222 2222222222222222222222222 %
        
    %[centroids, distances] = computCentroidsandDistances(X, indices, K); % recalculate centroids, and distances among all clusters
    
    % 3333333333333333333333333 computCentroidsandDistances 3333333333333333333333333 %
    %calculate centroids of each cluster, and sum of distances within each cluster
    [m n] =size(X);
    centroids = zeros(K,n);
    distances = 0;

    for q = 1:K
        xi = X(indices == q, :);
        ck = size(xi, 1);
        %centroids(i, :) = (1/ck)*[sum(xi(:,1)) sum(xi(:,2))];
        centroids(q,:) = mean(xi);
        %d = 0; % distance =0
        %for j = 1:ck
            %d = d + sqrt(sum((xi(j,:) - centroids(i,:)).^2)); %this is the
            %distance within a cluster
        %end
        %distances1 = d + distances1; %this is sum of distances among all clusters.
        t = (xi - mean(xi)).^2;
       distances = distances + sum(sqrt(t(:,1) + t(:,2))); %this is to calculate the sum of distances among all clusters
    end
    % 3333333333333333333333333 333333333333333333333333333 3333333333333333333333333 %
    
    sumD(i) = distances; 
end

%idx = kmeans (X, K); %this is to use kmeans function to calculate the indices for each cluster
figure;
plot(sumD, 'bo-', 'linewidth',2);
title('Overall distances vs. number of iterations');
xlabel('Iterations');
ylabel('Overall distances');
grid on;

figure;
message = sprintf('KMeans Clustering (MaxIterations = %d)', max_iterations);
plot(trainData(1, indices == 1),trainData(2, indices == 1),'r.', 'MarkerSize', 10);% for 1st cluster
hold on;
plot(trainData(1, indices == 2), trainData(2, indices == 2), 'b.', 'MarkerSize', 10); %for 2nd cluster
hold on;
plot(trainData(1, indices == 3), trainData(2, indices == 3), 'g.', 'MarkerSize', 10); %for 3rd cluster
hold on;
plot(centroids(1,1), centroids(1,2),'kx', 'MarkerSize',15,'LineWidth', 3); %for the 1st centroid
hold on;
plot(centroids(2,1), centroids(2,2),'kx', 'MarkerSize', 15, 'LineWidth', 3); %for the 2nd centroid
hold on;
plot(centroids(3,1), centroids(3,2),'kx', 'MarkerSize', 15, 'LineWidth', 3); %for the 3rd centroid
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroids', 'location', 'NW'); %'best');
title(message);
hold off;
xlabel('x-value');
ylabel('y-value');
xlim([-6 6]);
ylim([-2 12]);
grid on;



