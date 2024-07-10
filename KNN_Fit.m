
function [bestK, Train, Test] = KNN_Fit(X_ca, X_co, K )
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

% Define a range of k values for grid search
kValues = 1:K;

% Initialize variables to store results
bestK = 0;
bestAccuracy = 0;
results = [];

% Perform grid search on the training set
for k = kValues
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
        
        % Train KNN model with the current k value
        mdl = fitcknn(foldTrainX, foldTrainY, 'NumNeighbors', k);

        % Predict and calculate accuracy on the validation set
        predLabels = predict(mdl, foldValX);
        accuracies(i) = mean(predLabels == foldValY);
    end
    
    % Calculate mean accuracy for the current k
    meanAccuracy = mean(accuracies);
    results = [results; k, meanAccuracy]; %#ok<AGROW>
    
    % Update best parameters if current model is better
    if meanAccuracy > bestAccuracy
        bestK = k;
        bestAccuracy = meanAccuracy;
    end
end

% Display the best parameters and accuracy
%disp(['Best k: ', num2str(bestK)]);
%disp(['Best Accuracy: ', num2str(bestAccuracy)]);

% Fit the final model using the best k on the entire training set
mdl = fitcknn(trainX, trainY, 'NumNeighbors', bestK);

% Make predictions on the test set
predLabels_tr = predict(mdl, trainX);
predLabels_ts = predict(mdl, testX);

% Compute the confusion matrix
confMat = confusionmat(trainY, predLabels_tr);

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

% Compute the confusion matrix
confMat = confusionmat(testY, predLabels_ts);

% Calculate performance metrics
TP = confMat(2, 2);
TN = confMat(1, 1);
FP = confMat(1, 2);
FN = confMat(2, 1);

accuracy = (TP + TN) / sum(confMat(:));
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1score = 2 * (precision * recall) / (precision + recall);

% % Display the results
% fprintf('Confusion Matrix:\n');
% disp(confMat);
% fprintf('Accuracy: %.2f%%\n', accuracy * 100);
% fprintf('Precision: %.2f\n', precision);
% fprintf('Recall: %.2f\n', recall);
% fprintf('F1 Score: %.2f\n', f1score);

Test.confMat = confMat; 
Test.accuracy = accuracy*100;
Test.precision = precision;
Test.recall = recall;
Test.f1score = f1score;

end 