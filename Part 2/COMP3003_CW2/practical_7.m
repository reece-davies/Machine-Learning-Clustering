% Practical 7 requires the Neural Network Toolbox (nnet) to load the
% dataset. It is available on MATLAB Online, but not on v.2018.

clear, clc, close all;

% (1) Load dataset
load bodyfat_dataset.mat
inputData = bodyfatInputs;
targetData = bodyfatTargets;

% (2) Train the neural network using the dataset
net = feedforwardnet(10); %select feedforward network with 12 hidden neurons
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
net.trainParam.showWindow = false;  % Disables the nntraintool display for each run
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[net, tr] = train(net, inputData, targetData); %train the neural network using the dataset
plotperform(tr); %plot performance graph

% (3) Testing the neural network for the testing dataset
testX = inputData(:, tr.testInd); %testing data inputs
testT = targetData(:, tr.testInd); %testing data Target output

testY = net(testX); % predicted testing output from the trained net.
perf = mse(net, testT, testY); %calcualte MSE between testing Target data (testT) and testing Predicted data (testY).

% (4) Testing the performance
corrcoef(testT, testY); %calculate correlation coefficients for the testing dataset
scatter(testT, testY); % draw a scatter plot between the target and predicted output)

plotregression(testT, testY); %this will use plotregression function to draw a regression plot with R value displayed in the title of the graph for the testing dataset.
Y = net(inputData);
plotregression(targetData, Y); % this will draw a regression plot for all samples (similar as in Figure 4)

