function [bestBoxConstraint, bestKernelScale, Train, Test] = SVM_Fit(X_ca, X_co, K )

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

% Define a range of BoxConstraint and KernelScale values for grid search
boxConstraintValues = logspace(-K, K, 5);
kernelScaleValues = logspace(-K, K, 5);

% Initialize variables to store results
bestBoxConstraint = 0;
bestKernelScale = 0;
bestAccuracy = 0;
results = [];

% Perform grid search on the training set
for boxConstraint = boxConstraintValues
    for kernelScale = kernelScaleValues
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
            
            % Train SVM model with the current BoxConstraint and KernelScale
            mdl = fitcsvm(foldTrainX, foldTrainY, ...
                'KernelFunction', 'rbf', ...
                'BoxConstraint', boxConstraint, ...
                'KernelScale', kernelScale);

            % Predict and calculate accuracy on the validation set
            predLabels = predict(mdl, foldValX);
            accuracies(i) = mean(predLabels == foldValY);
        end
        
        % Calculate mean accuracy for the current parameters
        meanAccuracy = mean(accuracies);
        results = [results; boxConstraint, kernelScale, meanAccuracy]; %#ok<AGROW>
        
        % Update best parameters if current model is better
        if meanAccuracy > bestAccuracy
            bestBoxConstraint = boxConstraint;
            bestKernelScale = kernelScale;
            bestAccuracy = meanAccuracy;
        end
    end
end

% Display the best parameters and accuracy
% disp(['Best BoxConstraint: ', num2str(bestBoxConstraint)]);
% disp(['Best KernelScale: ', num2str(bestKernelScale)]);
% disp(['Best Accuracy: ', num2str(bestAccuracy)]);

% Fit the final model using the best parameters on the entire training set
mdl = fitcsvm(trainX, trainY, ...
    'KernelFunction', 'rbf', ...
    'BoxConstraint', bestBoxConstraint, ...
    'KernelScale', bestKernelScale);

% Make predictions on the training set
predLabels = predict(mdl, trainX);

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
predLabels = predict(mdl, testX);

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