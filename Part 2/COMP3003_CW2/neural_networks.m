clc;
close all;
clear;

% A.). Load the training and testing dataset

% Load training dataset which contains CT, SBR, BLER, MBL, MOS
load('subj_training.txt');
trainDataset = subj_training;
trainDataset = trainDataset';

% Load testing dataset which contains CT, SBR, BLER, MBL, MOS
load('subj_testing.txt');
testDataset = subj_testing;
testDataset = testDataset';

% B.) Pre-process the data
inputTrainData = trainDataset(1:4, :);
targetTrainData = trainDataset(5, :);

% C.) Select feedforwardnet neural network and use its default neural
%     network function
% +
% D.) Define the number of neurons of the neural network, define the neural
%     network parameters

% trainFcn = 'trainrp'; % For testing the network with a different training function

hiddenLayerSize = 6;
net = feedforwardnet(hiddenLayerSize); % Select feedforward network with 6 hidden neurons

% Default 'trainLm' function paramters (using <help trainlm>)
% epochs            1000  Maximum number of epochs to train
% goal                 0  Performance goal
% max_fail             6  Maximum validation failures
% min_grad          1e-7  Minimum performance gradient
% mu               0.001  Initial Mu
% mu_dec             0.1  Mu decrease factor
% mu_inc              10  Mu increase factor
% mu_max            1e10  Maximum Mu
% show                25  Epochs between displays
% showCommandLine  false  Generate command-line output
% showWindow        true  Show training GUI
% time               inf  Maximum time to train in seconds

net.trainParam.epochs = 10;
net.trainParam.max_fail = 5;
net.trainParam.min_grad = 1e-4;
% net.trainParam.showWindow = false;

% E.) Train the neural network based on the training dataset provided
[net, tr] = train(net, inputTrainData, targetTrainData); % Train the neural network using the dataset
figure;
plotperform(tr); % Plot performance graph
output = net(inputTrainData); % Predicted training output from the trained net.

% Calculate correlation coefficients (R) for the training dataset
R = corrcoef(targetTrainData, output);
R = R(1,2) * 100; % Turn R into a percentage
R = round(R, 2); % Round R off to 2nd decimal point for better percentage value

% Draw a scatter plot between the target and predicted output
figure('Name','Predicted vs measured MOS for training dataset');
scatter(targetTrainData, output);
title(['For training dataset (R=',num2str(R),'%)'])
hold off;
xlabel('Measured MOS');
ylabel('Predicted MOS');
grid on;

view(net);

% F.) Predict MOS scores for the testing dataset using the trained neural network model
 
inputTestData = testDataset(1:4, :); % Testing data inputs (testX from practical 7)
targetTestData = testDataset(5, :); % Testing data target output (testT from practical 7)

% Predict values for inputData
output = net(inputTestData); % Predicted testing output from the trained net (testY from practical 7)
perf = mse(net, targetTestData, output) % Calcualte MSE between testing Target data and testing Predicted data (output)

% Testing the performance
R = corrcoef(targetTestData, output); % Calculate correlation coefficients for the testing dataset
R = R(1,2) * 100; % Turn R into a percentage
R = round(R, 2); % Round R off to 2nd decimal point for better percentage value

% Draw a scatter plot between the target and predicted output)
figure('Name','Predicted vs measured MOS for testing dataset');
scatter(targetTestData, output);
title(['For testing dataset (R=',num2str(R),'%)'])
hold off;
xlabel('Measured MOS');
ylabel('Predicted MOS');
grid on;

% Plot regression (with R value displayed in the title of the graph for the testing dataset)
figure;
plotregression(targetTestData, output);

net % Display the network's specifications in the command window








%%%%%%%%%%%%%%%%% MIGHT NOT BE NEEDED %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Y = net(inputTrainData);
% figure;
% plotregression(targetTrainData, Y);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% MIGHT NOT BE NEEDED %%%%%%%%%%%%%%%%%

