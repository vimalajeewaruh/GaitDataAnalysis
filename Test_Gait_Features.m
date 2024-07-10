close all; clear all; clc

%addpath('/Users/hvimalajeewa2/Documents/Tamu_Documents/TAMU/DemosNew/')

dirName = sprintf('/Users/hvimalajeewa2/Documents/UNL_Documents/Projects/Gait_Data/GSTRIDE_database/');             %# folder path
files = dir( fullfile(dirName,'*.csv') );

fname = fullfile(dirName,files.name);     %# full path to file
data_registry = readtable(fname, 'VariableNamingRule','preserve', 'DecimalSeparator','.');

load Subject.mat
Case_ID_new = Subject.Case_ID_new;
Control_ID_new = Subject.Control_ID_new;

Gait_Ca = data_registry(Case_ID_new,:);
Gait_Co = data_registry( Control_ID_new,:);

%% SPPA Battery tests 
Tests = ["BalanceTextIndex", "GaitSpeedTestIndex", "ChairStandIndex", "SPPBAssessmentIndex"] ;

Gait_Fe_Co = zeros( size(Control_ID_new,2), 13);
Gait_Fe_Ca = zeros( size(Case_ID_new,2), 13);


h= figure('Renderer', 'painters', 'Position', [2 8 1200 300]);
p = 26;
for i = 1:4
    AVG_ca = table2array(Gait_Ca(:,p));
    STD_ca = table2array(Gait_Ca(:,p+1));
    
    AVG_co = table2array(Gait_Co(:,p));
    STD_co = table2array(Gait_Co(:,p+1));
 
    A = STD_ca./AVG_ca;
    B = STD_co./AVG_co;
    
    Gait_Fe_Ca(:,i) = A(:,1); Gait_Fe_Co(:,i) = B(:,1); 

    subplot(1,4,i)
    grp = [zeros(1,size(A,1)),ones(1,size(B,1))];
    boxplot([B' A'],grp, 'Labels',{'Non-fallers','Fallers'})
    title(Tests(i))
    grid on 
    p = p + 1;
end 

%saveas(h,'./Figures/Test_Gait_Features_SPPA_TestBattery.png')

%%

Tests = ["StrideTime AVG", "StrideTime STD", "Load Avg", "Load STD", "FootFlat AVG","FootFlat STD",...
    "Push AVG", "Push STD","Swing AVG", "Swing STD", "Toe-Off Angle AVG","Toe-Off Angle STD",...
    "Heal Strike AVG", "Heal Strike STD", "Cadence AVG", "Cadence STD","Step Speed AVG",  "Step Speed STD",...
    "Stride Length AVG", "Stride Length STD"] ;

h= figure('Renderer', 'painters', 'Position', [6 8 2400 1200]);
p = 35;
for i = 1:19
    AVG_ca = table2array(Gait_Ca(:,p)); 
    STD_ca = table2array(Gait_Ca(:,p+1));
    
    AVG_co = table2array(Gait_Co(:,p)); 
    STD_co = table2array(Gait_Co(:,p+1));
    A = STD_ca./AVG_ca;
    B = STD_co./AVG_co;

    subplot(5,4,i)
    grp = [zeros(1,size(A,1)),ones(1,size(B,1))];
    boxplot([B' A'],grp, 'Labels',{'Non-fallers','Fallers'})
    title(Tests(i))
    grid on 
    p = p + 1;
end 

%saveas(h,'./Figures/Test_Gait_Features_AVG_STD.png')

%%

Tests = ["Stride Time", "Load", "Foot Flat","Push", "Swing ", "Toe-Off Angle ",...
    "Heal Strike", "Cadence", "Step Speed", "Stride Length", "3D Path", "2D Path", "Clearance"] ;

h= figure('Renderer', 'painters', 'Position', [6 6 1200 1200]);
p = 35;
for i = 1:13
    AVG_ca = table2array(Gait_Ca(:,p));
    STD_ca = table2array(Gait_Ca(:,p+1));
    
    AVG_co = table2array(Gait_Co(:,p));
    STD_co = table2array(Gait_Co(:,p+1));

    A = STD_ca./AVG_ca;
    B = STD_co./AVG_co;

    Gait_Fe_Ca(:,i+4) = A(:,1); Gait_Fe_Co(:,i+4) = B(:,1);

    subplot(3,5,i)
    grp = [zeros(1,size(A,1)),ones(1,size(B,1))];
    boxplot([B' A'],grp, 'Labels',{'Non-fallers','Fallers'})
    title(Tests(i))
    grid on 
    p = p + 2;
end 

%saveas(h,'./Figures/Test_Gait_Features_CV.png')

%% Test the differnece between H of LA and AV from cases and controls is significant
disp("Test the differnece between H of LA and AV from cases and controls from the new method")

% Controls
X_control_Slp = Gait_Fe_Co; 
X_case_slp = Gait_Fe_Ca; 

H = [];
for i = 1:size(X_case_slp,2)
    A = X_control_Slp(:,i); B = X_case_slp(:,i);

    [p1, h, stats] = ranksum(A, B);

    if h == 0
        disp('The test fails to reject the null hypothesis. There is no significant difference between the two samples.');
    else
        disp('The test rejects the null hypothesis. There is a significant difference between the two samples.');
    end
end 

%Slope_Control_New = Slope_Control_New(:,find(H));
%Slope_Case_New = Slope_Case_New(:,find(H));

%% 
% close all
% 
% % %% Construct Feature matrices 
% % FeautreNames = ["BalanceTextIndex", "GaitSpeedTestIndex", "ChairStandIndex", "SPPBAssessmentIndex",...
% %     "StrideTime CV", "Load CV", "FootFlat CV","Push CV", "Swing CV", "Toe-Off Angle CV",...
% %     "Heal Strike CV", "Cadence CV", "Step Speed CV", "Stride Length CV"];
% % 
% Y_Ca = ones(size(Gait_Fe_Ca,1),1); Y_Co = zeros(size(Gait_Fe_Co,1),1);
% % 
% % GaitFeaturesMatrix.Gait_Fe_Co = Gait_Fe_Co; 
% % GaitFeaturesMatrix.Gait_Fe_Ca = Gait_Fe_Ca; 
% % 
% % d = './PythonData/';
% % 
% % filename = sprintf('Gait_Fe_Co.csv');
% %  writematrix(Gait_Fe_Co,fullfile(d,filename))
% % 
% % filename = sprintf('Gait_Fe_Ca.csv');
% %  writematrix( Gait_Fe_Ca,fullfile(d,filename))
% % 
% % 
% % 
% % GaitFeaturesMatrix.FeautreNames = FeautreNames;
% 
% load './Slope/Slope_Case_New.csv'
% load './Slope/Slope_Control_New.csv'
% 
% GaitFeaturesMatrix.Slope_Case_New = Slope_Case_New;
% GaitFeaturesMatrix.Slope_Control_New = Slope_Control_New;
% 
% load './Slope/Slope_Case_old.csv'
% load './Slope/Slope_Control_old.csv'
% 
% GaitFeaturesMatrix.Slope_Case_old = Slope_Case_old;
% GaitFeaturesMatrix.Slope_Control_old = Slope_Control_old;
% 
% 
% GaitFeaturesMatrix.Y_Co = Y_Co;
% GaitFeaturesMatrix.Y_Ca = Y_Ca;
% 
%save('GaitFeaturesMatrix.mat')