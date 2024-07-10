function [ Train, Test bestK] = RandomForest_Fit(X_ca, X_co,K )
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
        cvTrain = cvpartition(trainY, 'KFold', 10);
        for i = 1:cvTrain.NumTestSets
            % Extract training and validation sets for the current fold
            trainIdx = training(cvTrain, i);
            valIdx = test(cvTrain, i);
            foldTrainX = trainX(trainIdx, :);
            foldTrainY = trainY(trainIdx);
            foldValX = trainX(valIdx, :);
            foldValY = trainY(valIdx);
            
            % Train KNN model with the current k value
            %mdl = fitcknn(foldTrainX, foldTrainY, 'NumNeighbors', k);
            % Train Random Forest classifier with the selected features
            rfModel = TreeBagger(k, trainX, trainY, 'OOBPredictorImportance', 'on', ...
        'AlgorithmForCategorical','Exact','MaxNumSplits', 20);
    
            % Predict and calculate accuracy on the validation set
            %predLabels = predict(rfModel, foldValX);

           predLabels = predict(rfModel, trainX);
           predLabels = categorical(predLabels);
           accuracies(i) = sum(predLabels == categorical(trainY)) / numel(trainY);
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
    %mdl = fitcknn(trainX, trainY, 'NumNeighbors', bestK);
    mdl = TreeBagger(bestK, trainX, trainY, 'OOBPredictorImportance', 'on', ...
        'AlgorithmForCategorical','Exact','MaxNumSplits', 50);
    
    % Make predictions on the test set
    predLabels_tr = categorical( predict(mdl, trainX) );
    predLabels_ts = categorical( predict(mdl, testX) );
    
    % Compute the confusion matrix
    confMat = confusionmat(categorical(trainY), predLabels_tr);
    
    % Calculate performance metrics
    TP = confMat(2, 2);
    TN = confMat(1, 1);
    FP = confMat(1, 2);
    FN = confMat(2, 1);
    
    accuracy = (TP + TN) / sum(confMat(:));
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1score = 2 * (precision * recall) / (precision + recall);
    
    
    Train.confMat = confMat; 
    Train.accuracy = accuracy*100;
    Train.precision = precision;
    Train.recall = recall;
    Train.f1score = f1score;
    
    % Compute the confusion matrix
    confMat = confusionmat(categorical(testY), predLabels_ts);
    
    % Calculate performance metrics
    TP = confMat(2, 2);
    TN = confMat(1, 1);
    FP = confMat(1, 2);
    FN = confMat(2, 1);
    
    accuracy = (TP + TN) / sum(confMat(:));
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1score = 2 * (precision * recall) / (precision + recall);
    
    
    Test.confMat = confMat; 
    Test.accuracy = accuracy*100;
    Test.precision = precision;
    Test.recall = recall;
    Test.f1score = f1score;

end 