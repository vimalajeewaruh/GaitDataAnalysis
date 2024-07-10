close all; clear all; clc
addpath('/Users/hvimalajeewa2/Documents/Tamu_Documents/TAMU/DemosNew/')
addpath('./MatlabFunctions/')

%rng(5); % random seed
seed =10;
rng(seed,'twister')

% Load the demographic databased to select gait features
dirName = sprintf('/Users/hvimalajeewa2/Documents/UNL_Documents/Projects/Gait_Data/GSTRIDE_database/');             %# folder path
files = dir( fullfile(dirName,'*.csv') );

fname = fullfile(dirName,files.name);     %# full path to file
data_registry = readtable(fname, 'VariableNamingRule','preserve', 'DecimalSeparator','.');

% load the subject whoes gait features were measures by using the same
% sensor and who have more than 10000 measurments 
load Subject.mat
Case_ID_new = Subject.Case_ID_new;
Control_ID_new = Subject.Control_ID_new;

% Seclect gait feaures of case and controls
Gait_Ca = data_registry(Case_ID_new,:);
Gait_Co = data_registry( Control_ID_new,:);

%%  Anatimocal, functional and congnitive variables
%GaitFeaturs = [17, 18 26:39 35:size(Gait_Co,2)];

Gait_Fe_Co = zeros( size(Control_ID_new,2), 13);
Gait_Fe_Ca = zeros( size(Case_ID_new,2), 13);


%% Gait Features from the IMU Recodings 
% Select meand and std of each gait features intexed from 35  and calculate
% the coefficient of variation 
Tests = ["StrideTime CV", "Load CV", "FootFlat CV","Push CV", "Swing CV", "Toe-Off Angle CV",...
    "Heal Strike CV", "Cadence CV", "Step Speed CV", "Stride Length CV"] ;
GaitF_Names = ["G_1", "G_2", "G_3","G_4", "G_5", "G_6",...
    "G-7", "G_8", "G_9", "G_{10}", "G_{11}, G_{12}", "G_{13}"]; 

p = 35;  H = [];
for i = 1:13
    AVG_ca = table2array(Gait_Ca(:,p)); % Average of pth feaure of cases 
    STD_ca = table2array(Gait_Ca(:,p+1)); % Std of pth feature of cases 
    
    AVG_co = table2array(Gait_Co(:,p));  % Average of pth feaure of controls 
    STD_co = table2array(Gait_Co(:,p+1)); % Std of pth feaure of controls
   
    A = STD_ca./AVG_ca; % coefficient of variation of pth feaure of cases 
    B = STD_co./AVG_co;  % coefficient of variation of pth feaure of controls
    A = A(~isnan(A));B = B(~isnan(B));
    
    % Store the CV of the gait feature in seperate case and control metrices
    Gait_Fe_Ca(:,i) = A(:,1); Gait_Fe_Co(:,i) = B(:,1);
    p = p + 2;
    
    % Perform Wilcoxon rank sum test to check the difference of pth feature between 
    % cases and controls is statistically significant
    
    [p1, h, stats] = ranksum(A, B);
    H = [H, h];

end 

% Create a fearure matrix, combining fearue matrics of cases and controls
X = cat(1, Gait_Fe_Ca, Gait_Fe_Co); X(isnan(X))=0; XX = zscore(X);

% Feature matrix with significant feaures only
GaitF_Names = GaitF_Names(find(H));

X = X(:,find(H));  XX = zscore(X);

% Create the response matirx Y assigning 1 for cases and 0 for controls
Y_Ca = ones(size(Gait_Fe_Ca,1),1); Y_Co = zeros(size(Gait_Fe_Co,1),1);
Y = cat(1,Y_Ca, Y_Co);

%% Gait feaure selection using eature using forward selection

% M is the number used to run different classifiers 
% 1 - LR, 2 - KNN, 3 - SVM, 4 - RF, 5 - NB, 6 - EM
M = 1

if M == 1 % Logistic Regression - LR
    % maxdev = chi2inv(.9,2);     
    % opt = statset('display','iter','TolFun',maxdev,'TolTypeFun','abs');
    % [fs,history] = sequentialfs(@critfun,XX,Y, 'cv', 'none','nullmodel',true, 'options',opt,'direction','forward') 
    
    % Define the criterion function for feature selection using fitglm
    criterion = @(Xtr,Ytr,Xte,Yte) ...
    sum(Yte ~= round(predict(fitglm(Xtr, Ytr, 'Distribution', 'binomial'), Xte)));

    % Set options for sequentialfs including cross-validation
    opts = statset('display', 'iter', 'UseParallel', true);
    
    % Perform Sequential Forward Selection with 10-fold cross-validation
    cvpart = cvpartition(Y, 'KFold', 5);
    [fs, history] = sequentialfs(criterion, XX, Y, 'options', opts, 'cv', cvpart, 'nullmodel',true, 'direction','forward');
    GaitF_Names(find(fs))

elseif M == 2 % K-nearest neighbour KNN
    k = 8;
    c = cvpartition(Y,'k',5); 
    opts = statset('Display','iter', 'TolTypeFun','abs','DerivStep',.001);
    
    fun = @(XT,yT,Xt,yt)loss(fitcknn(XT,yT, 'NumNeighbors',k,'Standardize', 0),Xt,yt);
    [fs,history] = sequentialfs(fun,XX,Y,'cv',c,'options',opts,'direction','forward');
    GaitF_Names(find(fs))

elseif M == 3  % SVM
    c = cvpartition(Y,'k',5);
    opts = statset('Display','iter', 'TolTypeFun','abs','DerivStep',.1,'FunValCheck','on','UseParallel',true);   
    fun = @(XT,yT,Xt,yt)loss(fitcsvm( XT, yT, 'KernelFunction','rbf', 'Standardize',0 ), Xt, yt);
    
    [fs,history] = sequentialfs(fun,XX,Y,'cv',c,'options',opts,'direction','forward','nullmodel',false);
    GaitF_Names(find(fs))

elseif M == 4 % Random Forest - RF
    E = 20;
    % Define the criterion function for feature selection using TreeBagger
    criterion = @(Xtr,Ytr,Xte,Yte) ...
        sum(~strcmp(Yte, predict(TreeBagger(E, Xtr, Ytr, 'Method', 'classification'), Xte)));
    
    % Set options for sequentialfs including cross-validation
    opts = statset('display', 'iter', 'UseParallel', true);
    
    % Perform Sequential Forward Selection with 5-fold cross-validation
    cvpart = cvpartition(Y, 'KFold', 5);
    [fs, history] = sequentialfs(criterion, XX, Y, 'options', opts, 'cv', cvpart);
    GaitF_Names(find(fs))

elseif M ==  5 % Naive Bayes - NB
    opts = statset('display', 'iter', 'Jacobian','off', 'MaxFunEvals', 20, 'MaxIter', 50, 'Tune',1);
    fun = @(trainData, trainLabels, testData, testLabels) ...
    sum(predict(fitcnb(trainData, trainLabels,"DistributionNames",'normal'), testData) ~= testLabels);

    % Perform forward feature selection
    [fs, history] = sequentialfs(fun, XX, Y, 'cv', ...
        cvpartition(Y, 'KFold', 5,'Stratify',true), 'options', opts, 'direction', 'forward');
    GaitF_Names(find(fs))

elseif M == 6 % Ensemble - EM-1
    opts = statset('display', 'iter');
    fun = @(trainData, trainLabels, testData, testLabels) ...
    sum(predict(fitcensemble(trainData, trainLabels, 'Method', 'AdaBoostM1', 'LearnRate', .01), testData) ~= testLabels);

    % Perform sequential feature selection
    [fs, history] = sequentialfs(fun, XX, Y, 'cv', cvpartition(Y, 'KFold', 5), 'options', opts, 'direction', 'forward' );
    GaitF_Names(find(fs))

end 

fprintf('################################## \n' );

%% Feature matrices of cases and controls with sigificant gait feature

X_case = Gait_Fe_Ca(:, find(fs)); X_control = Gait_Fe_Co(:, find(fs));

%%  Classifier Peformance with Gait Features only
disp"Performance with the Existig Gait Features "

if M == 1
    l = 10; % Searching space for lambda  [-l l]
    [Train, Test, g] = LR_Fit(X_case, X_control, l )

elseif M == 2
    K = 10; % Searching space for number of nearest-neighbors 1 - K
    [bestK, Train, Test] = KNN_Fit(X_case, X_control, K )

elseif M == 3
    F = .005; % Searching space for number of kernel scale and box contratint 0  -F
    [bestBoxConstraint, bestKernelScale, Train, Test] = SVM_Fit(X_case, X_control, F )

elseif M == 4
    E = 10; % Searching space for number of slpits 1 - E
    [ Train, Test, besta] = RandomForest_Fit(X_case, X_control, E )
elseif M == 5
    G = "normal"; % Searching space for kernels
    [ Train, Test] = NaiveBayes_Fit(X_case, X_control, G )
elseif M == 6
    S = 10;
    [ Train, Test, LearnN, LearnR] = Ensemnle1_Fit(X_case, X_control, S )

end 

fprintf('################################## \n' );
%% Performance with the Old H method and Existig Gait Features
load GaitFeaturesMatrix.mat;

disp"Performance with the Old H method and Existig Gait Features "

X_control_Slp = GaitFeaturesMatrix.Slope_Control_old; 
X_case_slp = GaitFeaturesMatrix.Slope_Case_old; 


P1 = [];
for i = 1:size(X_case_slp,2)

    p = nchoosek(1:size(X_case_slp,2),i);
    for j = 1 : size(p,1)
        X_case1 = [X_case X_case_slp(:,p(j,:)) ];  
        X_control1 = [X_control X_control_Slp(:,p(j,:))];
        
        if M == 1
            [Train, Test, bestLambda] = LR_Fit(X_case1, X_control1, l );
            P1 = [ P1; [i j Train.accuracy Test.accuracy bestLambda]];
        elseif M == 2
            [bestK, Train, Test] = KNN_Fit(X_case1, X_control1, K );
            P1 = [ P1; [i j Train.accuracy Test.accuracy bestK]];
        elseif M == 3
            [bestBoxConstraint, bestKernelScale, Train, Test] = SVM_Fit(X_case1, X_control1, F );

            P1 = [ P1; [i j Train.accuracy Test.accuracy bestBoxConstraint, bestKernelScale]];
       elseif M == 4
            [ Train, Test besta] = RandomForest_Fit(X_case1, X_control1, E );
            P1 = [ P1; [i j Train.accuracy Test.accuracy besta]];

      elseif M == 5
          
            [ Train, Test] = NaiveBayes_Fit(X_case1, X_control1, G );
            P1 = [ P1; [i j Train.accuracy Test.accuracy]];
      elseif M == 6
            [ Train, Test, LearnN, LearnR] = Ensemnle1_Fit(X_case1, X_control1, S );
            P1 = [ P1; [i j Train.accuracy Test.accuracy LearnN LearnR]];
      end         
    end 
end 

% Select the best slope combination that gives highest performance
P1 = P1( find(P1(:,3)>= P1(:,4)),:);

id_mx = find(P1(:,4) == max(P1(:,4)));

P1(id_mx, :)

N1 = P1(id_mx, :)

ll = nchoosek(1:size(X_case_slp,2),N1(1,1));
ll(N1(1,2),:)

fprintf('################################## \n' );
%% Performance with the New H method and Existig Gait Features
load GaitFeaturesMatrix.mat;

disp"Performance with the New H method and Existig Gait Features "

X_control_Slp = GaitFeaturesMatrix.Slope_Control_New; 
X_case_slp = GaitFeaturesMatrix.Slope_Case_New; 

P = [];
for i = 1:size(X_case_slp,2)

    p = nchoosek(1:size(X_case_slp,2),i);
    for j = 1 : size(p,1)
        X_case1 = [X_case X_case_slp(:,p(j,:)) ];  
        X_control1 = [X_control X_control_Slp(:,p(j,:))];

         if M == 1
            [Train, Test, bestLambda] = LR_Fit(X_case1, X_control1, l );
            P = [ P; [i j Train.accuracy Test.accuracy bestLambda]];
        
         elseif M == 2
            [bestK, Train, Test] = KNN_Fit(X_case1, X_control1, K );
            P = [ P; [i j Train.accuracy Test.accuracy bestK]];
        
         elseif M == 3
            [bestBoxConstraint, bestKernelScale, Train, Test] = SVM_Fit(X_case1, X_control1, F);

            P = [ P; [i j Train.accuracy Test.accuracy bestBoxConstraint, bestKernelScale]];
        elseif M == 4
            [ Train, Test besta] = RandomForest_Fit(X_case1, X_control1, E );
            P = [ P; [i j Train.accuracy Test.accuracy besta]];
        
         elseif M == 5
            [ Train, Test] = NaiveBayes_Fit(X_case1, X_control1, G );
            P = [ P; [i j Train.accuracy Test.accuracy]];
         elseif M == 6
            [ Train, Test, LearnN, LearnR] = Ensemnle1_Fit(X_case1, X_control1, S );
            P = [ P; [i j Train.accuracy Test.accuracy LearnN LearnR]];
        end        
    end 
end 

% Select the best slope combination that gives highest performance

P = P( find(P(:,3)>= P(:,4)),:);

id_mx = find(P(:,4) == max(P(:,4)));

N = P(id_mx, :)

l = nchoosek(1:size(X_case_slp,2),N(1,1));
l(N(1,2),:)

