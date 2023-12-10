clc;
close all;
clear;

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

% Plot original training dataset
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

% B.) Concatenate the training datasets into a single dataset
trainData = [data1 data2 data3];

% Plot merged dataset
figure;
plot(trainData(1, :), trainData(2,:), 'b.', 'MarkerSize', 10);
legend('Training dataset','location', 'NW');
title('Merged Dataset');
xlabel('x-value');
ylabel('y-value');
xlim([-6 6]);
ylim([-2 12]);
grid on;

% C.) Implement Kmeans clustering from first principles
X = trainData';
K = 3;
max_iterations = 10;

% Create centroids, based on the value of K
centroids = zeros(K, size(X,2));
randidx = randperm(size(X,1));
centroids = X(randidx(1:K), :);

total_dist = zeros(K,1);

for i = 1:max_iterations
    
    K = size(centroids, 1);
    indices = zeros(size(X,1), 1);
    m = size(X,1);

    % Calculate which cluster each datapoint must be assigned to
    for p = 1:m
        
        chosen_cluster = 1;
        min_dist = (X(p,:) - centroids(1,:)) * (X(p,:) - centroids(1,:))';
        
        for j = 2:K
            dist = sum((X(p,:) - centroids(j,:)) .^ 2);
            if (dist < min_dist)
                min_dist = dist;
                chosen_cluster = j;
            end
        end
        
        indices(p) = chosen_cluster;
    end
     
    [m, n] = size(X);
    centroids = zeros(K,n);
    distances = 0;
    
    % Calculate the centroids for each cluster
    for q = 1:K
        xi = X(indices == q, :);
        ck = size(xi, 1);
        centroids(q,:) = mean(xi);
        t = (xi - mean(xi)).^2;
        
        % Calculate the total distance within each cluster
        distances = distances + sum(sqrt(t(:,1) + t(:,2)));
    end
    
    total_dist(i) = distances; 
end

% Plot sum of distance for each iteration
% figure;
% plot(total_dist, 'bo-', 'linewidth',2);
% title('Overall distances vs. number of iterations');
% xlabel('Iterations');
% ylabel('Overall distances');
% grid on;

% Plot Kmeans clustering implementation
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
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroids', 'location', 'NW');
title(message);
hold off;
xlabel('x-value');
ylabel('y-value');
xlim([-6 6]);
ylim([-2 12]);
grid on;


% D.) Generate a testing dataset, with the same Gaussian distribution
samples = 50;
Mean1 = [-3; 4];
Std1 = 1.25;
Mean2 = [2; 7];
Std2 = 1.25;
Mean3 = [1; 2];
Std3 = 1.5;

data1 = Std1 * randn(2, samples) + repmat(Mean1, 1, samples);
data2 = Std2 * randn(2, samples) + repmat(Mean2, 1, samples);
data3 = Std3 * randn(2, samples) + repmat(Mean3, 1, samples);

% Plot original testing dataset
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

% E.) Concatenate the testing datasets into a single dataset
testData = [data1 data2 data3];

% F.) Assign new data to the existing clusters
X = testData';
K = size(centroids, 1);
indices = zeros(size(X,1), 1);
m = size(X,1);

% Calculate closest centroid for each datapoint (which cluster each
% datapoint must be assigned to)
for p = 1:m
        
    chosen_cluster = 1;
    min_dist = (X(p,:) - centroids(1,:)) * (X(p,:) - centroids(1,:))';

    for j = 2:K
        dist = sum((X(p,:) - centroids(j,:)) .^ 2);
        if (dist < min_dist)
            min_dist = dist;   
            chosen_cluster = j;
        end
    end

    indices(p) = chosen_cluster;
end


% Plot Kmeans clustering implementation with training dataset
figure;
message = sprintf('KMeans Clustering');
hold on;

plot(X(1:50, 1),X(1:50, 2),'ro', 'MarkerSize', 10); % for 1st cluster
plot(X(51:100, 1),X(51:100, 2),'bo', 'MarkerSize', 10); % for 2nd cluster
plot(X(101:150, 1),X(101:150, 2),'go', 'MarkerSize', 10); % for 1st cluster

plot(testData(1, indices == 1),testData(2, indices == 1),'r.', 'MarkerSize', 10); % for 1st cluster
plot(testData(1, indices == 2), testData(2, indices == 2), 'b.', 'MarkerSize', 10); %for 2nd cluster
plot(testData(1, indices == 3), testData(2, indices == 3), 'g.', 'MarkerSize', 10); %for 3rd cluster

plot(centroids(1,1), centroids(1,2),'kx', 'MarkerSize',15,'LineWidth', 3); %for the 1st centroid
plot(centroids(2,1), centroids(2,2),'kx', 'MarkerSize', 15, 'LineWidth', 3); %for the 2nd centroid
plot(centroids(3,1), centroids(3,2),'kx', 'MarkerSize', 15, 'LineWidth', 3); %for the 3rd centroid

legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Classified as Cluster 1', 'Classified as Cluster 2', ...
    'Classified as Cluster 3', 'Centroids', 'location', 'NW');
title(message);
hold off;
xlabel('x-value');
ylabel('y-value');
xlim([-6 6]);
ylim([-2 12]);
grid on;

