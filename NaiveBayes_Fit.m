function [ Train Test] = NaiveBayes_Fit(X_ca, X_co, K )
        X_case = X_ca; X_control = X_co;
        
        X = cat(1, X_case, X_control); X(isnan(X))=1;

        Y_Ca = ones(size(X_case,1),1); Y_Co = zeros(size(X_control,1),1);
        Y = cat(1,Y_Ca, Y_Co);
        y = Y;

        % Split the data into training and testing sets
        cv = cvpartition(y, 'HoldOut', 0.3, 'Stratify',true);
        trainX = X(training(cv), :); trainX = zscore(trainX);
        trainY = y(training(cv));
        testX = X(test(cv), :); testX = zscore(testX); 
        testY = y(test(cv));


        % Train ensemble model with hyperparameter optimization
        ensembleModel = fitcnb(trainX, trainY,"DistributionNames","kernel","Kernel", K);
        
        cvModel = crossval(ensembleModel, 'KFold', 5);

        % Compute the cross-validated loss (classification error)
        cvLoss = kfoldLoss(cvModel);

        % Make predictions on the train set
        predLabels = predict(ensembleModel, trainX);

        % Compute the confusion matrix
        confMat = confusionmat(trainY, predLabels);

        % Calculate performance metrics
        accuracyT = sum(predLabels == trainY) / numel(trainY);


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

        % Make predictions on the test set
        predLabels = predict(ensembleModel, testX);

        % Compute the confusion matrix
        confMat = confusionmat(testY, predLabels);

        % Calculate performance metrics
        accuracyTs = sum(predLabels == testY) / numel(testY);
        %fprintf('Accuracy: %.2f%%\n', accuracy * 100);

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
