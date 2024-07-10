close all; clear all; clc
addpath('/Users/hvimalajeewa2/Documents/Tamu_Documents/TAMU/DemosNew/')
addpath('./MatlabFunctions/')

rng(5); % random seed

%Gait_Ca = readtable('./Ex_Features/Gait_Ca.csv', 'VariableNamingRule', 'preserve');
%Gait_Co = readtable('./Ex_Features/Gait_Ca.csv', 'VariableNamingRule', 'preserve');

dirName = sprintf('/Users/hvimalajeewa2/Documents/UNL_Documents/Projects/Gait_Data/GSTRIDE_database/');             %# folder path
files = dir( fullfile(dirName,'*.csv') );

fname = fullfile(dirName,files.name);     %# full path to file
data_registry = readtable(fname, 'VariableNamingRule','preserve', 'DecimalSeparator','.');

load Subject.mat
Case_ID_new = Subject.Case_ID_new;
Control_ID_new = Subject.Control_ID_new;

Gait_Ca = data_registry(Case_ID_new,:);

Gait_Co = data_registry( Control_ID_new,:);

% SPPA Battery tests 

Gait_Fe_Co = zeros( size(Control_ID_new,2), 13);
Gait_Fe_Ca = zeros( size(Case_ID_new,2), 13);


%

Tests = ["StrideTime CV", "Load CV", "FootFlat CV","Push CV", "Swing CV", "Toe-Off Angle CV",...
    "Heal Strike CV", "Cadence CV", "Step Speed CV", "Stride Length CV"] ;


p = 35;  H = [];
for i = 1:13
    AVG_ca = table2array(Gait_Ca(:,p)); 
    STD_ca = table2array(Gait_Ca(:,p+1));
    
    AVG_co = table2array(Gait_Co(:,p)); 
    STD_co = table2array(Gait_Co(:,p+1));

    A = STD_ca./AVG_ca;
    B = STD_co./AVG_co;
    A = A(~isnan(A));B = B(~isnan(B));

    Gait_Fe_Ca(:,i) = A(:,1); Gait_Fe_Co(:,i) = B(:,1);

    p = p + 2;
    [p1, h, stats] = ranksum(A, B);
    H = [H, h];
end  

X = cat(1, Gait_Fe_Ca, Gait_Fe_Co); X(isnan(X))=1;

X = X(:, find(H));

Y_Ca = ones(size(Gait_Fe_Ca,1),1); Y_Co = zeros(size(Gait_Fe_Co,1),1);
y = cat(1,Y_Ca, Y_Co);

%Convert species to a categorical variable
y = categorical(y);

%Number of features
numFeatures = size(X, 2);


K = 2;

%%  Hurst Exponent features from Standard and New Method

load GaitFeaturesMatrix.mat;

if K == 0
    disp"Performance with the Gait Features "
    X = X;

elseif K == 1
    disp"Performance with the Standard H method and Existig Gait Features "

    X_control_Slp = GaitFeaturesMatrix.Slope_Control_old; 
    X_case_Slp = GaitFeaturesMatrix.Slope_Case_old; 

    H = [];
    for i = 1:size(X_case_Slp,2)
        A = X_control_Slp(:,i); B = X_case_Slp(:,i);
    
        [p1, h, stats] = ranksum(A, B);
        H = [H h];
    end 
    
    X_control_Slp = X_control_Slp(:,find(H));
    X_case_Slp = X_case_Slp(:,find(H));
    
    X_slp = cat(1, X_case_Slp, X_control_Slp);

    X = [X X_slp]; X(isnan(X))=1;

elseif K == 2
    disp"Performance with the New H method and Existig Gait Features "

    X_control_Slp = GaitFeaturesMatrix.Slope_Control_New; 
    X_case_Slp = GaitFeaturesMatrix.Slope_Case_New; 

   H = [];
    for i = 1:size(X_case_Slp,2)
        A = X_control_Slp(:,i); B = X_case_Slp(:,i);
    
        [p1, h, stats] = ranksum(A, B);
        H = [H h];
    end 
    
    X_control_Slp = X_control_Slp(:,find(H));
    X_case_Slp = X_case_Slp(:,find(H));
    
    X_slp = cat(1, X_case_Slp, X_control_Slp);
    
    X = [X X_slp]; X_slp(isnan(X_slp))=1;
end 

%%
% Convert species to a categorical variable
y = cat(1,Y_Ca, Y_Co);
y = categorical(y);

%Split the data into training and testing sets
cv = cvpartition(y, 'HoldOut', 0.33);
trainX = X(training(cv), :); trainX = zscore(trainX);
trainY = y(training(cv));
testX = X(test(cv), :); testX = zscore(testX);
testY = y(test(cv));

%% Define the objective function for Bayesian optimization
treeOptFcn = @(x) crossval('mcr', trainX, trainY, ...
    'Predfun', @(trainX, trainY, testX) predict(fitctree(trainX, trainY, ...
    'MinLeafSize', x.MinLeafSize, 'MaxNumSplits', x.MaxSplits, 'NumBins', x.NumBins), testX), 'kfold', 5);

% Define the optimization variables
treeOptVars = [optimizableVariable('MinLeafSize', [2, 20], 'Type', 'integer');
               optimizableVariable('MaxSplits', [2, 20], 'Type', 'integer');
               optimizableVariable('NumBins', [2, 20], 'Type', 'integer')];

% Perform Bayesian optimization
treeResults = bayesopt(treeOptFcn, treeOptVars, 'MaxObjectiveEvaluations', 50);

% Get the best hyperparameters
bestMinLeafSize = treeResults.XAtMinObjective.MinLeafSize;
bestMaxSplits = treeResults.XAtMinObjective.MaxSplits;
bestNumBins = treeResults.XAtMinObjective.NumBins;

% Train the final Decision Tree model with the best hyperparameters
treeModel = fitctree(trainX, trainY, 'MinLeafSize', ...
    bestMinLeafSize, 'MaxNumSplits', ...
    bestMaxSplits, 'NumBins', ...
    bestNumBins,  'Prune','on', 'SplitCriterion','deviance');

%% Hyperparameter optimization for k-NN

% Define the objective function for Bayesian optimization
knnOptFcn = @(x) crossval('mcr', trainX, trainY, ...
    'Predfun', @(trainX, trainY, testX) predict(fitcknn(trainX, trainY, ...
    'NumNeighbors', x.NumNeighbors, ...
    'NSMethod', char(x.NSMethod), ...
    'Distance', char(x.Distance)), testX), 'kfold', 5);

% Define the optimization variables
knnOptVars = [
    optimizableVariable('NumNeighbors', [1, 10], 'Type', 'integer');
    optimizableVariable('NSMethod', {'exhaustive', 'kdtree'}, 'Type', 'categorical');
    optimizableVariable('Distance', {'euclidean', 'cityblock', 'chebychev', 'minkowski'}, 'Type', 'categorical');
];

% Perform Bayesian optimization
knnResults = bayesopt(knnOptFcn, knnOptVars, 'MaxObjectiveEvaluations', 50);

% Get the best hyperparameters
bestKnnNumNeighbors = knnResults.XAtMinObjective.NumNeighbors;
bestKnnNSMethod = char(knnResults.XAtMinObjective.NSMethod);
bestKnnDistance = char(knnResults.XAtMinObjective.Distance);
%bestKnnStandardize = logical(knnResults.XAtMinObjective.Standardize);

% Train the final k-NN model with the best hyperparameters
knnModel = fitcknn(trainX, trainY, ...
    'NumNeighbors', bestKnnNumNeighbors, ...
    'NSMethod', bestKnnNSMethod, ...
    'Distance', bestKnnDistance);


%% Hyperparameter optimization for Naive Bayes

nbOptFcn = @(x) crossval('mcr', trainX, trainY, 'Predfun', @(trainX, trainY, testX) ...
    predict(fitcnb(trainX, trainY, 'DistributionNames', char(x.DistributionNames)), testX), 'kfold', 5);
nbOptVars = optimizableVariable('DistributionNames', {'normal', 'kernel'}, 'Type', 'categorical');
nbResults = bayesopt(nbOptFcn, nbOptVars, 'MaxObjectiveEvaluations', 50);
bestNbDistributionNames = char(nbResults.XAtMinObjective.DistributionNames);
nbModel = fitcnb(trainX, trainY, 'DistributionNames', bestNbDistributionNames);

%% Hyperparameter optimization for Logistic Regression
logRegOptFcn = @(x) crossval('mcr', trainX, trainY, ...
    'Predfun', @(trainX, trainY, testX) predict(fitclinear(trainX, trainY, ...
    'Learner', 'logistic', 'Solver', char(x.Solver), 'Regularization', char(x.Regularization)), testX), 'kfold', 5);

% Define the optimization variables
logRegOptVars = [optimizableVariable('Solver', {'sgd', 'asgd'}, 'Type', 'categorical');
                 optimizableVariable('Regularization', {'ridge', 'lasso'}, 'Type', 'categorical')];

% Perform Bayesian optimization
logRegResults = bayesopt(logRegOptFcn, logRegOptVars, 'MaxObjectiveEvaluations', 50);

% Get the best solver and regularization
bestLogRegSolver = char(logRegResults.XAtMinObjective.Solver);
bestLogRegRegularization = char(logRegResults.XAtMinObjective.Regularization);

% Train the final logistic regression model with the best solver and regularization
logRegModel = fitclinear(trainX, trainY, 'Learner', 'logistic', 'Solver', bestLogRegSolver, 'Regularization', bestLogRegRegularization);

%% Hyperparameter optimization for Random Forest

rfOptFcn = @(x) crossval('mcr', trainX, trainY, 'Predfun', @(trainX, trainY, testX) ...
    predict(TreeBagger(x.NumTrees, trainX, trainY, 'MinLeafSize', x.MinLeafSize), testX), 'kfold', 5);
rfOptVars = [optimizableVariable('NumTrees', [5, 20], 'Type', 'integer'); 
    optimizableVariable('MinLeafSize', [1,10], 'Type', 'integer')];
rfResults = bayesopt(rfOptFcn, rfOptVars, 'MaxObjectiveEvaluations', 50);
bestRfNumTrees = rfResults.XAtMinObjective.NumTrees;
bestRfMinLeafSize = rfResults.XAtMinObjective.MinLeafSize;
rfModel = TreeBagger(bestRfNumTrees, trainX, trainY, 'MinLeafSize', bestRfMinLeafSize);

%% Hyperparameter optimization for SVM

svmOptFcn = @(x) crossval('mcr', trainX, trainY, 'Predfun', @(trainX, trainY, testX) ...
    predict(fitcsvm(trainX, trainY, 'KernelFunction', char(x.KernelFunction)), testX), 'kfold', 5);
svmOptVars = optimizableVariable('KernelFunction', {'linear', 'rbf'}, 'Type', 'categorical');
svmResults = bayesopt(svmOptFcn, svmOptVars, 'MaxObjectiveEvaluations', 50);
bestSvmKernelFunction = char(svmResults.XAtMinObjective.KernelFunction);
svmModel = fitcsvm(trainX, trainY, 'KernelFunction', bestSvmKernelFunction);

%% Model Performance Evaluations

%Make predictions on the test set
predTree = predict(treeModel, trainX);
predKNN = predict(knnModel, trainX);
predNB = predict(nbModel, trainX);
 predLogReg = predict(logRegModel, trainX);
predRF = predict(rfModel, trainX);
predSVM = predict(svmModel, trainX);

% Combine predictions using majority voting
predEnsemble = mode([predTree, predKNN, predNB, predLogReg, predRF, predSVM], 2);

% Compute the confusion matrix
confMat = confusionmat(trainY, predEnsemble);

% Calculate performance metrics
accuracy = sum(predEnsemble == trainY) / numel(trainY);
fprintf('Ensemble Model Accuracy: %.2f%%\n', accuracy * 100);

% Display confusion matrix
disp('Confusion Matrix:');
disp(confMat);

%Make predictions on the test set
predTree = predict(treeModel, testX);
predKNN = predict(knnModel, testX);
predNB = predict(nbModel, testX);
 predLogReg = predict(logRegModel, testX);
predRF = predict(rfModel, testX);
predSVM = predict(svmModel, testX);

% Combine predictions using majority voting
predEnsemble = mode([predTree, predKNN, predNB, predLogReg, predRF, predSVM], 2);

% Compute the confusion matrix
confMat = confusionmat(testY, predEnsemble);

% Calculate performance metrics
accuracy = sum(predEnsemble == testY) / numel(testY);
fprintf('Ensemble Model Accuracy: %.2f%%\n', accuracy * 100);

% Display confusion matrix
disp('Confusion Matrix:');
disp(confMat);
