
function [Train, Test, bestLambda] = LR_Fit(X_ca, X_co, l )
X_case = X_ca; X_control = X_co;

X = cat(1, X_case, X_control); X(isnan(X))=1;

Y_Ca = ones(size(X_case,1),1); Y_Co = zeros(size(X_control,1),1);
Y = cat(1,Y_Ca, Y_Co);
y = Y;

% Split the data into training and testing sets
cv = cvpartition(y, 'HoldOut', 0.3);

trainX = X(training(cv), :); trainX  = zscore(trainX );
trainY = y(training(cv));
testX = X(test(cv), :); testX  = zscore(testX );
testY = y(test(cv));

% Define a range of Lambda values for grid search
lambdaValues = logspace(-l, l, 20);

% Initialize variables to store results
bestLambda = 0;
bestAccuracy = 0;
results = [];

% Perform grid search on the training set
for lambda = lambdaValues
    accuracies = zeros(5, 1);  % Initialize accuracy for each fold
    cvTrain = cvpartition(trainY, 'KFold', 5);
    for i = 1:cvTrain.NumTestSets
        % Extract training and validation sets for the current fold
        trainIdx = training(cvTrain, i);
        valIdx = test(cvTrain, i);
        foldTrainX = trainX(trainIdx, :);
        foldTrainY = trainY(trainIdx);
        foldValX = trainX(valIdx, :);
        foldValY = trainY(valIdx);
        
        % Train logistic regression model with the current Lambda value
        [B, FitInfo] = lassoglm(foldTrainX, foldTrainY, 'binomial', 'Lambda', lambda);

        % Predict and calculate accuracy on the validation set
        predProbs = glmval([FitInfo.Intercept; B], foldValX, 'logit');
        predLabels = round(predProbs);
        accuracies(i) = mean(predLabels == foldValY);
    end
    
    % Calculate mean accuracy for the current Lambda
    meanAccuracy = mean(accuracies);
    results = [results; lambda, meanAccuracy]; %#ok<AGROW>
    
    % Update best parameters if current model is better
    if meanAccuracy > bestAccuracy
        bestLambda = lambda;
        bestAccuracy = meanAccuracy;
    end
end

% Display the best parameters and accuracy
%disp(['Best Lambda: ', num2str(bestLambda)]);
%disp(['Best Accuracy: ', num2str(bestAccuracy)]);

% Fit the final model using the best Lambda on the entire training set
[B, FitInfo] = lassoglm(trainX, trainY, 'binomial', 'Lambda', bestLambda);

% Make Predition on the training set 

% Make predictions on the test set
predProbs = glmval([FitInfo.Intercept; B], trainX, 'logit');
predLabels = round(predProbs);

% Ensure both testY and predLabels are logical arrays
trainY = logical(trainY);
predLabels = logical(predLabels);

% Compute the confusion matrix
confMat = confusionmat(trainY, predLabels);

% Calculate performance metrics
TP = confMat(2, 2);
TN = confMat(1, 1);
FP = confMat(1, 2);
FN = confMat(2, 1);

accuracy = (TP + TN) / sum(confMat(:));
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1score = 2 * (precision * recall) / (precision + recall);

% Display the results
% fprintf('Confusion Matrix:\n');
% disp(confMat);
% fprintf('Accuracy: %.2f%%\n', accuracy * 100);
% fprintf('Precision: %.2f\n', precision);
% fprintf('Recall: %.2f\n', recall);
% fprintf('F1 Score: %.2f\n', f1score);

Train.confMat = confMat; 
Train.accuracy = accuracy*100;
Train.precision = precision;
Train.recall = recall;
Train.f1score = f1score;

% Make predictions on the test set
predProbs = glmval([FitInfo.Intercept; B], testX, 'logit');
predLabels = round(predProbs);

% Ensure both testY and predLabels are logical arrays
testY = logical(testY);
predLabels = logical(predLabels);

% Compute the confusion matrix
confMat = confusionmat(testY, predLabels);

% Calculate performance metrics
TP = confMat(2, 2);
TN = confMat(1, 1);
FP = confMat(1, 2);
FN = confMat(2, 1);

accuracy = (TP + TN) / sum(confMat(:));
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1score = 2 * (precision * recall) / (precision + recall);

% Display the results
% fprintf('Confusion Matrix:\n');
% disp(confMat);
% fprintf('Accuracy: %.2f%%\n', accuracy * 100);
% fprintf('Precision: %.2f\n', precision);
% fprintf('Recall: %.2f\n', recall);
% fprintf('F1 Score: %.2f\n', f1score);
% fprintf('################################## \n' );

Test.confMat = confMat; 
Test.accuracy = accuracy*100;
Test.precision = precision;
Test.recall = recall;
Test.f1score = f1score;

end 