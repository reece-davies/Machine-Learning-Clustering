% examples.m
% coursework for COMP3003 - 2020
%
clear all, close all, clc

load('subj_training.txt'); %load training dataset which contains CT, SBR, BLER, MBL, MOS
trainDataset = subj_training;
trainDataset = trainDataset';

load('subj_testing.txt'); % load testing dataset which contains CT, SBR, BLER, MBL, MOS
testDataset = subj_testing;
testDataset = testDataset';
