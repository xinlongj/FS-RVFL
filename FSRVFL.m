function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy, EstimatedOutputs] = FSRVFL(TrainingData, TestingData, nOutput, nFeatureA, nLabeledData, LamdaA, LamdaB, LaplacianOptionsA, LaplacianOptionsB, nHiddenNeurons, ActivationFunction)

% Input:
% TrainingData          - Training data set
% TestingData           - Testing data set
% nOutput               - Number of output dimension e.g. (x y) is 2 outputs
% nFeatureA             - Number of features A (A and B are two kinds of features)
% nLabeledData          - Number of labeled samples, means the real position is given
% LamdaA                - The weight value of A's manifold constraint
% LamdaB                - The weight value of B's manifold constraint
% LaplacianOptionsA     - The parameters of Laplacian A
% LaplacianOptionsB     - The parameters of Laplacian B
%  .NN                     - The number of nearest neighbors
%  .GraphDistanceFunction  - The draph distance function
%                          	 * 'euclidean'
%  .GraphWeights           - Weight type;
%                            * 'distance' for distance weight
%                            * 'binary' for binary weight
%                            * 'heat' for heat kernel sigma
%  .GraphNormalize         - 0 for normalized laplacian, 1 for not normalized laplacian
%  .GraphWeightParam       - The standard deviation when use 'heat' as GraphWeight
% nHiddenNeurons        - Number of hidden neurons assigned to the FSRVFL
% ActivationFunction    - Type of activation function:
%                           'rbf' for radial basis function, G(a,b,x) = exp(-b||x-a||^2)
%                           'sig' for sigmoidal function, G(a,b,x) = 1/(1+exp(-(ax+b)))
%                           'sin' for sine function, G(a,b,x) = sin(ax+b)
%                           'hardlim' for hardlim function, G(a,b,x) = hardlim(ax+b)
% 
% Output£º
% TrainingTime          - Time (seconds) spent on training model
% TestingTime           - Time (seconds) spent on predicting all testing data
% TrainingAccuracy      - Training accuracy:
%                           RMSE for EstimatedPositions of training data
% TestingAccuracy       - Testing accuracy:
%                           RMSE for EstimatedPositions of testing data
% EstimatedPosition     - The estimated postion by FSRVFL
%
%
% ------------------------------------------------------------------------
% Samples;
%
% LaplacianOptionsA.NN = 20;
% LaplacianOptionsA.GraphDistanceFunction='euclidean';
% LaplacianOptionsA.GraphWeights='binary';
% LaplacianOptionsA.GraphNormalize=1;
% LaplacianOptionsA.GraphWeightParam=1;
%
% LaplacianOptionsB.NN = 20;
% LaplacianOptionsB.GraphDistanceFunction='euclidean';
% LaplacianOptionsB.GraphWeights='binary';
% LaplacianOptionsB.GraphNormalize=1;
% LaplacianOptionsB.GraphWeightParam=1;
%
% FSRVFL(TrainingData, TestingData, 2, 5, 1000, 0.6, 0.4, LaplacianOptionsA, LaplacianOptionsB, 1000, 'sig')
% ------------------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Authors:   Xinlon Jiang    
%    Affiliate: Institute of Computing Technology, CAS
%    EMAIL:     jiangxinlong@ict.ac.cn
%    Paper:     Not published yet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Load dataset
train_data = TrainingData;
test_data  = TestingData;

T    = train_data(:,1:nOutput);
P    = train_data(:,nOutput + 1:end);
P_A  = train_data(:,nOutput + 1:nOutput + nFeatureA);
P_B  = train_data(:,nOutput + nFeatureA + 1:end);
TV.T = test_data(:,1:nOutput);
TV.P = test_data(:,nOutput + 1:end);
clear train_data test_data;

nTrainingData = size(P,1); 
nTestingData  = size(TV.P,1);
nInputNeurons = size(P,2);

%%%%%%%%%%% Calculate graph laplacian
LaplacianA = laplacian(P_A,'nn',LaplacianOptionsA);
LaplacianB = laplacian(P_B,'nn',LaplacianOptionsB);

J = zeros(nTrainingData,nTrainingData);
for i = 1:nLabeledData
	J(i,i) = 1;
end

%%%%%%%%%%% Step 1 Initialization Phase
P0 = P(1:nLabeledData,:); 
T0 = T(1:nLabeledData,:);

%=====================================================================
%   TRAINING
%=====================================================================
start_time = cputime;
IW = rand(nHiddenNeurons, nInputNeurons) * 2 - 1;
switch lower(ActivationFunction)
    case{'rbf'}
        Bias = rand(1, nHiddenNeurons);
        H = RBFun(P, IW, Bias);
        HTrain = RBFun(P0, IW, Bias);
    case{'sig'}
       Bias = rand(1, nHiddenNeurons) * 2 - 1;
        H = SigActFun(P, IW, Bias);
        HTrain = SigActFun(P0, IW, Bias);
    case{'sin'}
        Bias = rand(1, nHiddenNeurons) * 2 - 1;
        H = SinActFun(P, IW, Bias);
        HTrain = SinActFun(P0, IW, Bias);
    case{'hardlim'}
        Bias = rand(1, nHiddenNeurons) * 2 - 1;
        H = HardlimActFun(P, IW, Bias);
        H = double(H);
        HTrain = HardlimActFun(P0, IW, Bias);
end

H0 = (J + LamdaA * LaplacianA + LamdaB * LaplacianB) * [P H];
beta = pinv(H0) * (J * T);
end_time = cputime;
TrainingTime = end_time - start_time;
Y = [P0 HTrain] * beta;

%=====================================================================
%   TESTING
%=====================================================================
start_time = cputime;
switch lower(ActivationFunction)
    case{'rbf'}
        Bias = rand(1,nHiddenNeurons);
        HTest = RBFun(TV.P, IW, Bias);
    case{'sig'}
        Bias = rand(1,nHiddenNeurons) * 2 - 1;
        HTest = SigActFun(TV.P, IW, Bias);
    case{'sin'}
        Bias = rand(1,nHiddenNeurons) * 2 - 1;
        HTest = SinActFun(TV.P, IW, Bias);
    case{'hardlim'}
        Bias = rand(1,nHiddenNeurons) * 2 - 1;
        HTest = HardlimActFun(TV.P, IW, Bias);
end
TY = [TV.P HTest] * beta;
EstimatedOutputs = TY;
end_time = cputime;
TestingTime = end_time - start_time;

%%%%%%%%%%%%%% Calculate RMSE in the case of REGRESSION
[m n] = size(T0);
dis = zeros(m,1);
for i=1:1:m
    dis(i,1) = norm(T0(i,:) - Y(i,:));
end
TrainingAccuracy = sqrt(mse(dis))

[m n] = size(TV.T);
dis = zeros(m,1);
for i=1:1:m
    dis(i,1) = norm(TV.T(i,:) - TY(i,:));
end
TestingAccuracy = sqrt(mse(dis))