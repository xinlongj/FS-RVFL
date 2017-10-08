% clean up
close all
clear all
clc

% load the data
bleDataset  = load('Data/JinyuanMall_ble.txt');
wifiDataset = load('Data/JinyuanMall_wifi.txt');

label = wifiDataset(:,1:2);
bleFeature = bleDataset(:,3:7);
wifiFeature = wifiDataset(:,3:9);

DATA = [label bleFeature wifiFeature];
rowrank = randperm(size(DATA, 1));
DATA = DATA(rowrank, :);

TRAIN_DATA = DATA(1:4800,:);
TEST_DATA  = DATA(4801:end,:);

% ------------------ start of FSRVFL --------------------------  
Lamda1 = 0.6;  %0.3
Lamda2 = 0.6;  %0.3
nOuput = 2;
nFirstFeature = 5;
NN1 = 20;
NN2 = 20;
nHiddenNeurons = 1000;
ActivationFunction = 'sig';

fprintf('FSRVFLMO')

TrainingData = TRAIN_DATA;
TestingData  = TEST_DATA;
nOutput = 2;
nFeatureA = 5;
nLabeledData = 1000;
LamdaA = 0.6;
LamdaB = 0.6;
LaplacianOptionsA.NN = 20;
LaplacianOptionsA.GraphDistanceFunction='euclidean';
LaplacianOptionsA.GraphWeights='binary';
LaplacianOptionsA.GraphNormalize=1;
LaplacianOptionsA.GraphWeightParam=1;

LaplacianOptionsB.NN = 20;
LaplacianOptionsB.GraphDistanceFunction='euclidean';
LaplacianOptionsB.GraphWeights='binary';
LaplacianOptionsB.GraphNormalize=1;
LaplacianOptionsB.GraphWeightParam=1;

nHiddenNeurons = 1000;
ActivationFunction = 'sig';
[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy, EstimatedOutputs] = FSRVFL(TrainingData, TestingData, nOutput, nFeatureA, nLabeledData, LamdaA, LamdaB, LaplacianOptionsA, LaplacianOptionsB, nHiddenNeurons, ActivationFunction)
%plot(EstimatedOutputs(:,1), EstimatedOutputs(:,2))
[ErrorDis, Acc] = Accuracy(TestingData(:,1:2), EstimatedOutputs, 38.5)